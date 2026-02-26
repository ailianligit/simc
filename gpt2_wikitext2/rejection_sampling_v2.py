import os
import gc
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings

# 引入用于宏观密度配额重采样的 KMeans
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
# [新增] 强制限制底层 C++ 库的线程数，防止与 PyTorch 抢占资源导致死锁或崩溃
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"
BASE_MODEL_PATH = "/home/ubuntu/data/model/gpt2_model" 
EMBEDDING_MODEL_PATH = "/home/ubuntu/data/model/all-mpnet-base-v2"
GEN0_ORACLE_PATH = "/home/ubuntu/data/simc/gpt2_wikitext2/model_real_trained"

# 实验规模
NUM_GENERATIONS = 10
EPOCHS = 5
TARGET_SAMPLES = 10000       # 最终用于训练的精选数据量
OVERSAMPLE_RATIO = 3.0       # 候选池放大倍数 (生成 20000 条供筛选)

GEN_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
GEN_TEMPERATURE = 1.3        # 较高温度，注入必要方差
PROMPT_LEN_BASE = 64

# ==========================================
# 1. 核心评估与采样器 (Cluster-First 架构)
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
        """预先将真实数据划分为 K 个特征簇，建立真实分布的定距锚点"""
        print(f">>> Profiling Real Data Macro Distribution (K-Means K={self.NUM_CLUSTERS})...")
        embs = self.get_embeddings(real_texts)
        
        self.kmeans = MiniBatchKMeans(n_clusters=self.NUM_CLUSTERS, random_state=42, batch_size=256)
        self.kmeans.fit(embs)
        
        labels = self.kmeans.labels_
        counts = Counter(labels)
        total = len(labels)
        self.real_cluster_proportions = {cluster: count / total for cluster, count in counts.items()}

    # --- 核心流水线 (重构的 Cluster-First 逻辑) ---
    def apply_sampling(self, candidate_texts, target_size, strategy):
        print(f"\n>>> Applying Strategy: [{strategy.upper()}]")
        
        if strategy == "baseline":
            selected = random.sample(candidate_texts, target_size)
            return selected, {}

        df = pd.DataFrame({"text": candidate_texts})

        # === 阶段 1: 全局特征提取 ===
        # 不再做全局硬截断，而是准备好武器 (NLL和Cluster)，留到簇内去打仗
        print("    | Extracting Global Features (NLL & Embeddings)...")
        if strategy in ["micro_only", "combined"]:
            df['nll'] = self.get_batch_oracle_nll(candidate_texts)
            
        if strategy in ["macro_only", "combined"]:
            pool_embs = self.get_embeddings(candidate_texts)
            df['cluster'] = self.kmeans.predict(pool_embs)

        # === 阶段 2: 路由分发 ===
        if strategy == "micro_only":
            # 纯微观：老规矩，动态剔除全局最差的，然后随机抽
            nll_mean, nll_std = df['nll'].mean(), df['nll'].std()
            dynamic_thresh = nll_mean + 1.2 * nll_std
            valid_df = df[df['nll'] <= dynamic_thresh]
            
            if len(valid_df) < target_size:
                final_df = df.nsmallest(target_size, 'nll')
            else:
                final_df = valid_df.sample(target_size, random_state=42)
                
            return final_df['text'].tolist(), {"nll_mean": final_df['nll'].mean()}

        elif strategy == "macro_only":
            # 纯宏观：按照配额盲抽，不管通顺度
            selected_indices = []
            for cluster_id in range(self.NUM_CLUSTERS):
                target_quota = int(target_size * self.real_cluster_proportions.get(cluster_id, 0))
                cluster_candidates = df[df['cluster'] == cluster_id]
                
                if len(cluster_candidates) > target_quota:
                    sampled = cluster_candidates.sample(n=target_quota, random_state=42)
                elif len(cluster_candidates) > 0:
                    # 数量不足，直接复制填满 (Oversampling)
                    sampled = cluster_candidates.sample(n=target_quota, replace=True, random_state=42)
                else:
                    continue
                selected_indices.extend(sampled.index.tolist())
                
            final_df = df.loc[selected_indices]
            if len(final_df) > target_size: final_df = final_df.sample(target_size, random_state=42)
            elif len(final_df) < target_size:
                missing = target_size - len(final_df)
                final_df = pd.concat([final_df, df.sample(missing, random_state=42)])
                
            return final_df['text'].tolist(), {"retained_clusters": final_df['cluster'].nunique()}

        elif strategy == "combined":
            # 终极形态：簇内局部安检 (Intra-Cluster NLL Sorting)
            print("    | Running Intra-Cluster Density Rebalancing & NLL Sorting...")
            selected_df_list = []
            
            for cluster_id in range(self.NUM_CLUSTERS):
                target_quota = int(target_size * self.real_cluster_proportions.get(cluster_id, 0))
                if target_quota == 0: continue
                
                cluster_df = df[df['cluster'] == cluster_id]
                
                if len(cluster_df) == 0:
                    continue # 模式彻底丢失
                
                if len(cluster_df) >= target_quota:
                    # 【核心】优中选优：在拥挤簇里，挑选最像人话的 (NLL最低的) 留下，丢弃乱码
                    sampled = cluster_df.nsmallest(target_quota, 'nll')
                else:
                    # 【核心】长尾保护：在稀疏簇里，直接复制质量最好的几个样本续命，直到填满配额
                    # 先对剩下的少数样本排序，保证复制的是相对较好的
                    sorted_minority = cluster_df.sort_values(by='nll')
                    # 允许有放回抽样，相当于根据 NLL 隐式地给予了复制权重
                    sampled = sorted_minority.sample(n=target_quota, replace=True, random_state=42)
                    
                selected_df_list.append(sampled)
                
            final_df = pd.concat(selected_df_list)
            
            # 精确对齐到 target_size
            if len(final_df) > target_size:
                final_df = final_df.sample(target_size, random_state=42)
            elif len(final_df) < target_size:
                # 差几条配额，用全局最通顺的补齐
                missing = target_size - len(final_df)
                remaining_df = df.drop(final_df.index)
                pad_df = remaining_df.nsmallest(missing, 'nll')
                final_df = pd.concat([final_df, pad_df])
                
            stats = {
                "retained_clusters": final_df['cluster'].nunique(),
                "nll_mean": final_df['nll'].mean()
            }
            return final_df['text'].tolist(), stats

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

    # 保存路径调整
    EXP_DIR = f"/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v8/exp_{args.strategy}"
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

        # [核心] 应用最终版的簇内安检采样器
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