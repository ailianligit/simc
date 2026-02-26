import os
import gc
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import linalg
from collections import Counter
import warnings

# [新增] 引入用于宏观密度配额重采样的 KMeans
from sklearn.cluster import MiniBatchKMeans

import torch
from torch.nn import CrossEntropyLoss
from datasets import load_from_disk, load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

# 忽略聚类时的常见无害警告
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 0. 全局配置与路径
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"
BASE_MODEL_PATH = "/home/ubuntu/data/model/gpt2_model" 
EMBEDDING_MODEL_PATH = "/home/ubuntu/data/model/all-mpnet-base-v2"
GEN0_ORACLE_PATH = "/home/ubuntu/data/simc/gpt2_wikitext2/model_real_trained"

# 实验规模
NUM_GENERATIONS = 10
EPOCHS = 5
TARGET_SAMPLES = 10000       # 最终用于训练的精选数据量
OVERSAMPLE_RATIO = 2.0      # 候选池放大倍数 (生成 20000 条供筛选)

GEN_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
GEN_TEMPERATURE = 1.3       # 较高温度，注入必要方差
PROMPT_LEN_BASE = 64

# ==========================================
# 1. 核心评估与采样器 (终极版)
# ==========================================
class AdvancedRejectionSampler:
    def __init__(self, oracle_path, embed_path, device='cuda'):
        self.oracle_path = oracle_path
        self.embed_path = embed_path
        self.device = device
        self.oracle_model = None
        self.oracle_tokenizer = None
        self.embed_model = None
        self.nll_loss_fct = CrossEntropyLoss(reduction='none')
        
        # [核心优化] 使用 KMeans 聚类锚点替代不稳定的 FBD 协方差计算
        self.kmeans = None
        self.real_cluster_proportions = None
        self.NUM_CLUSTERS = 100 # 将语义空间划分为 100 个特征簇

    # --- 模型显存管理 ---
    def _load_oracle(self):
        if self.oracle_model is None:
            print("    | [Mem] Loading Oracle Model...")
            self.oracle_tokenizer = GPT2Tokenizer.from_pretrained(self.oracle_path)
            if self.oracle_tokenizer.pad_token is None:
                self.oracle_tokenizer.pad_token = self.oracle_tokenizer.eos_token
            self.oracle_model = GPT2LMHeadModel.from_pretrained(self.oracle_path).to(self.device)
            self.oracle_model.eval()

    def _unload_oracle(self):
        if self.oracle_model is not None:
            print("    | [Mem] Unloading Oracle Model...")
            del self.oracle_model
            del self.oracle_tokenizer
            self.oracle_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def _load_embedder(self):
        if self.embed_model is None:
            print("    | [Mem] Loading Embedding Model...")
            self.embed_model = SentenceTransformer(self.embed_path, device=self.device)

    def _unload_embedder(self):
        if self.embed_model is not None:
            print("    | [Mem] Unloading Embedding Model...")
            del self.embed_model
            self.embed_model = None
            gc.collect()
            torch.cuda.empty_cache()

    # --- 特征提取计算 ---
    def get_batch_oracle_nll(self, texts, batch_size=32):
        self._load_oracle()
        nlls = []
        for i in tqdm(range(0, len(texts), batch_size), desc="    | Scoring Micro NLL", leave=False):
            batch = texts[i : i + batch_size]
            inputs = self.oracle_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.oracle_model(inputs.input_ids, attention_mask=inputs.attention_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                shift_mask = inputs.attention_mask[..., 1:].contiguous()
                loss = self.nll_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())
                seq_nll = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
                nlls.extend(seq_nll.cpu().tolist())
        self._unload_oracle()
        return nlls

    def get_embeddings(self, texts, batch_size=128):
        self._load_embedder()
        embs = self.embed_model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        self._unload_embedder()
        return embs

    def setup_real_distribution(self, real_texts):
        """[核心重构] 预先将真实数据划分为 K 个特征簇，建立真实分布的定距锚点"""
        print(f">>> Profiling Real Data Macro Distribution (K-Means K={self.NUM_CLUSTERS})...")
        embs = self.get_embeddings(real_texts)
        
        # 训练轻量级的 K-Means
        self.kmeans = MiniBatchKMeans(n_clusters=self.NUM_CLUSTERS, random_state=42, batch_size=256)
        self.kmeans.fit(embs)
        
        # 统计真实数据在每个簇的占比
        labels = self.kmeans.labels_
        counts = Counter(labels)
        total = len(labels)
        self.real_cluster_proportions = {cluster: count / total for cluster, count in counts.items()}

    # --- 核心流水线 ---
    def apply_sampling(self, candidate_texts, target_size, strategy):
        print(f"\n>>> Applying Strategy: [{strategy.upper()}]")
        
        if strategy == "baseline":
            selected = random.sample(candidate_texts, target_size)
            return selected, {}

        df = pd.DataFrame({"text": candidate_texts})

        # ==========================================
        # 阶段 A: 微观动态截断 (Fluency First 硬安检)
        # ==========================================
        if strategy in ["micro_only", "combined"]:
            print("    | Running Dynamic Micro-Level Profiling...")
            df['nll'] = self.get_batch_oracle_nll(candidate_texts)
            
            # 动态相对阈值：剔除当前代最差的幻觉样本 (NLL 过高)
            nll_mean = df['nll'].mean()
            nll_std = df['nll'].std()
            dynamic_thresh = nll_mean + 1.2 * nll_std # 稍微收紧一点，保证进入宏观池的数据质量
            
            valid_df = df[df['nll'] <= dynamic_thresh].copy()
            print(f"    | Dynamic NLL Cutoff: {dynamic_thresh:.2f} (Kept {len(valid_df)}/{len(df)})")
            
            if len(valid_df) < target_size:
                print("    | [Warning] Valid pool too small. Padding with best remaining NLL.")
                missing = target_size - len(valid_df)
                remaining = df.drop(valid_df.index).copy()
                pad_df = remaining.nsmallest(missing, 'nll')
                candidate_df = pd.concat([valid_df, pad_df])
            else:
                candidate_df = valid_df
        else:
            candidate_df = df # macro_only 不做微观过滤

        if strategy == "micro_only":
            final_df = candidate_df.sample(target_size, random_state=42)
            return final_df['text'].tolist(), {"nll_mean": final_df['nll'].mean()}

        # ==========================================
        # 阶段 B: [核心重构] 基于真实锚点的“配额削减”
        # 彻底取代低效的随机 FBD 搜索，消除维度诅咒
        # ==========================================
        if strategy in ["macro_only", "combined"]:
            print("    | Running Macro-Level Density Rebalancing (Quota Undersampling)...")
            pool_texts = candidate_df['text'].tolist()
            pool_embs = self.get_embeddings(pool_texts)
            
            # 将生成的样本映射到真实特征簇中
            cand_labels = self.kmeans.predict(pool_embs)
            candidate_df['cluster'] = cand_labels
            
            selected_indices = []
            
            # 对每一个簇，按照真实数据的比例计算“名额” (Quota)
            for cluster_id in range(self.NUM_CLUSTERS):
                target_quota = int(target_size * self.real_cluster_proportions.get(cluster_id, 0))
                cluster_candidates = candidate_df[candidate_df['cluster'] == cluster_id]
                
                if len(cluster_candidates) == 0:
                    continue # 模式彻底丢失 (Type II error)，无法挽回
                    
                if len(cluster_candidates) > target_quota:
                    # 发生 Mode Collapse (过度拥挤)：强制下采样，随机丢弃多余的！
                    sampled = cluster_candidates.sample(n=target_quota, random_state=42)
                else:
                    # 稀疏区：未达配额，全部保留 (甚至可以考虑过采样)
                    sampled = cluster_candidates
                    
                selected_indices.extend(sampled.index.tolist())
                
            selected_df = candidate_df.loc[selected_indices]
            
            # 最后数量对齐补齐
            if len(selected_df) < target_size:
                print(f"    | [Info] Strict quota resulted in {len(selected_df)} samples. Padding with remaining valid samples.")
                missing = target_size - len(selected_df)
                remaining_df = candidate_df.drop(selected_df.index)
                
                # 如果是 Combined 策略，优先用微观 NLL 好的样本补齐不足的配额
                if strategy == "combined" and len(remaining_df) >= missing:
                    pad_df = remaining_df.nsmallest(missing, 'nll')
                else:
                    pad_df = remaining_df.sample(min(missing, len(remaining_df)), random_state=42)
                    
                selected_df = pd.concat([selected_df, pad_df])
            elif len(selected_df) > target_size:
                selected_df = selected_df.sample(target_size, random_state=42)

            stats = {"retained_clusters": selected_df['cluster'].nunique()}
            if strategy == "combined":
                stats["nll_mean"] = selected_df['nll'].mean()
                
            return selected_df['text'].tolist(), stats

# ==========================================
# 2. 训练与生成工具
# ==========================================
def train_model(texts, tokenizer, output_dir):
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id

    ds = Dataset.from_dict({"text": texts})
    tok_ds = ds.map(lambda x: tokenizer([t + tokenizer.eos_token for t in x["text"]]), batched=True, remove_columns=["text"])
    
    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        tl = len(concat[list(examples.keys())[0]])
        tl = (tl // 1024) * 1024
        res = {k: [t[i:i+1024] for i in range(0, tl, 1024)] for k, t in concat.items()}
        res["labels"] = res["input_ids"].copy()
        return res
        
    packed = tok_ds.map(group_texts, batched=True)
    args = TrainingArguments(
        output_dir=output_dir, 
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=5e-5, 
        optim="adamw_torch_fused", 
        save_strategy="no", 
        report_to="none"
    )
    trainer = Trainer(model=model, args=args, train_dataset=packed, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(output_dir)
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir

def evaluate_ppl(model_dir, tokenizer, valid_texts):
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(DEVICE).eval()
    encodings = tokenizer("\n\n".join(valid_texts), return_tensors="pt")
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + 1024, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ppl

# ==========================================
# 3. 主干流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=["baseline", "micro_only", "macro_only", "combined"], required=True)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    EXP_DIR = f"/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v6/exp_{args.strategy}"
    os.makedirs(EXP_DIR, exist_ok=True)
    metrics_log = os.path.join(EXP_DIR, "metrics.json")
    
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    try: ds = load_from_disk(DATA_PATH)
    except: ds = load_dataset(DATA_PATH)
    real_train = [t for t in ds['train']['text'] if len(t.strip()) > 0][:TARGET_SAMPLES]
    real_valid = [t for t in ds['validation']['text'] if len(t.strip()) > 0][:]

    # [Generation 0] 
    if not os.path.exists(GEN0_ORACLE_PATH):
        print("\n>>> Training Generation 0 (Oracle) ...")
        train_model(real_train, tokenizer, GEN0_ORACLE_PATH)
    
    gen0_ppl = evaluate_ppl(GEN0_ORACLE_PATH, tokenizer, real_valid)
    history = [{"generation": 0, "strategy": "real_data", "ppl": gen0_ppl}]
    print(f"\n>>> Gen 0 PPL (Anchor): {gen0_ppl:.2f}")

    sampler = AdvancedRejectionSampler(GEN0_ORACLE_PATH, EMBEDDING_MODEL_PATH, DEVICE)
    sampler.setup_real_distribution(real_train)

    current_generator_dir = GEN0_ORACLE_PATH

    # [迭代循环]
    OVERSAMPLE_SIZE = int(TARGET_SAMPLES * OVERSAMPLE_RATIO)
    prompt_pool = (real_train * (OVERSAMPLE_SIZE // len(real_train) + 1))[:OVERSAMPLE_SIZE]

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n" + "="*40 + f"\n Generation {gen} | Strategy: {args.strategy}\n" + "="*40)
        
        print(f" -> Generating {OVERSAMPLE_SIZE} candidate samples...")
        generator = GPT2LMHeadModel.from_pretrained(current_generator_dir).to(DEVICE).eval()
        generator.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        
        candidate_texts = []
        for i in tqdm(range(0, OVERSAMPLE_SIZE, GEN_BATCH_SIZE), desc="    | Synthesizing"):
            batch = prompt_pool[i : i + GEN_BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outputs = generator.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=GEN_TEMPERATURE, top_p=0.95)
            candidate_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
        tokenizer.padding_side = "right"
        del generator
        gc.collect()
        torch.cuda.empty_cache()

        # [核心] 运用新的确定性聚类配额采样器
        selected_texts, stats = sampler.apply_sampling(candidate_texts, TARGET_SAMPLES, args.strategy)

        gen_dir = os.path.join(EXP_DIR, f"gen_{gen}")
        print(f" -> Training Generation {gen} Student...")
        train_model(selected_texts, tokenizer, gen_dir)
        
        ppl = evaluate_ppl(gen_dir, tokenizer, real_valid)
        print(f" => Generation {gen} Final PPL: {ppl:.2f}")
        
        history.append({"generation": gen, "strategy": args.strategy, "ppl": ppl, **stats})
        with open(metrics_log, "w") as f: json.dump(history, f, indent=4)
        
        current_generator_dir = gen_dir

    print("\n>>> All Generations Completed! Run this script with other --strategy flags to compare.")

if __name__ == "__main__":
    main()