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

import torch
from torch.nn import CrossEntropyLoss
from datasets import load_from_disk, load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

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
# 1. 核心评估与采样器
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
        
        # 缓存真实数据的宏观分布特征
        self.real_mu = None
        self.real_cov = None

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
    @staticmethod
    def compute_bigram_entropy(text):
        tokens = text.strip().split()
        if len(tokens) < 2: return 0.0
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        counts = Counter(bigrams)
        total = len(bigrams)
        return -sum((c/total) * math.log(c/total) for c in counts.values())

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

    @staticmethod
    def compute_fbd(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean): covmean = covmean.real
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

    def setup_real_distribution(self, real_texts):
        """预先计算真实数据的均值和协方差作为参考锚点"""
        print(">>> Profiling Real Data Macro Distribution...")
        embs = self.get_embeddings(real_texts)
        self.real_mu = np.mean(embs, axis=0)
        self.real_cov = np.cov(embs, rowvar=False)

    # --- 核心流水线 ---
    def apply_sampling(self, candidate_texts, target_size, strategy):
        print(f"\n>>> Applying Strategy: [{strategy.upper()}]")
        
        if strategy == "baseline":
            selected = random.sample(candidate_texts, target_size)
            return selected, {}

        # 1. 基础 DataFrame 初始化
        df = pd.DataFrame({"text": candidate_texts})

        # ==========================================
        # 阶段 A: 微观特征过滤 (Micro-Level Filtering)
        # ==========================================
        if strategy in ["micro_only", "combined"]:
            print("    | Running Micro-Level Profiling...")
            df['nll'] = self.get_batch_oracle_nll(candidate_texts)
            df['bi_ent'] = [self.compute_bigram_entropy(t) for t in candidate_texts]
            
            # 宽泛截断 (保留中间 70% 的数据，剔除极端异常值)
            ent_thresh = df['bi_ent'].quantile(0.15) # 防模式崩溃
            nll_thresh = df['nll'].quantile(0.85)    # 防幻觉乱码
            
            valid_df = df[(df['bi_ent'] >= ent_thresh) & (df['nll'] <= nll_thresh)].copy()
            
            # 兜底机制：如果过滤太狠，按综合质量分补齐
            if len(valid_df) < target_size:
                print(f"    | [Warning] Valid pool too small ({len(valid_df)}). Padding...")
                missing = target_size - len(valid_df)
                remaining = df.drop(valid_df.index).copy()
                remaining['score'] = remaining['bi_ent'] - remaining['nll']
                pad_df = remaining.nlargest(missing, 'score')
                candidate_df = pd.concat([valid_df, pad_df])
            else:
                # 即使充裕，我们也需要保留足够的候选供宏观搜索，至少保留 target_size * 1.2
                search_pool_size = max(target_size, int(target_size * 1.2))
                candidate_df = valid_df.sample(min(len(valid_df), search_pool_size), random_state=42)
        else:
            candidate_df = df # macro_only 不做微观过滤

        if strategy == "micro_only":
            final_df = candidate_df.sample(target_size, random_state=42)
            return final_df['text'].tolist(), {"nll_mean": final_df['nll'].mean(), "bi_ent_mean": final_df['bi_ent'].mean()}

        # ==========================================
        # 阶段 B: 宏观分布搜索 (Macro-Level Coreset Selection)
        # ==========================================
        if strategy in ["macro_only", "combined"]:
            print("    | Running Macro-Level FBD Subset Search...")
            pool_texts = candidate_df['text'].tolist()
            pool_embs = self.get_embeddings(pool_texts)
            
            best_idx = None
            best_fbd = float('inf')
            num_trials = 15 # 蒙特卡洛搜索次数
            
            for _ in tqdm(range(num_trials), desc="    | Searching best FBD subset", leave=False):
                subset_idx = np.random.choice(len(pool_embs), target_size, replace=False)
                sub_embs = pool_embs[subset_idx]
                sub_mu = np.mean(sub_embs, axis=0)
                sub_cov = np.cov(sub_embs, rowvar=False)
                
                fbd = self.compute_fbd(self.real_mu, self.real_cov, sub_mu, sub_cov)
                if fbd < best_fbd:
                    best_fbd = fbd
                    best_idx = subset_idx
                    
            selected_texts = [pool_texts[i] for i in best_idx]
            stats = {"fbd_score": best_fbd}
            if strategy == "combined":
                stats["nll_mean"] = candidate_df.iloc[best_idx]['nll'].mean()
            return selected_texts, stats

# ==========================================
# 2. 训练与生成工具
# ==========================================
def train_model(texts, tokenizer, output_dir):
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data Packing
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
        output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=5e-5, optim="adamw_torch_fused", save_strategy="no", report_to="none"
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
    for begin_loc in range(0, min(seq_len, 50000), stride):
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

    # 固定随机种，确保 Baseline 和策略组的初始 Prompt 是相同的
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    EXP_DIR = f"./rejection_sampling_results_v2/exp_{args.strategy}"
    os.makedirs(EXP_DIR, exist_ok=True)
    metrics_log = os.path.join(EXP_DIR, "metrics.json")
    
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    try: ds = load_from_disk(DATA_PATH)
    except: ds = load_dataset(DATA_PATH)
    real_train = [t for t in ds['train']['text'] if len(t.strip()) > 0][:TARGET_SAMPLES]
    real_valid = [t for t in ds['validation']['text'] if len(t.strip()) > 0][:]

    # [Generation 0] 裁判模型与基线训练
    os.makedirs("./rejection_results", exist_ok=True)
    if not os.path.exists(GEN0_ORACLE_PATH):
        print("\n>>> Training Generation 0 (Oracle) ...")
        train_model(real_train, tokenizer, GEN0_ORACLE_PATH)
    
    gen0_ppl = evaluate_ppl(GEN0_ORACLE_PATH, tokenizer, real_valid)
    history = [{"generation": 0, "strategy": "real_data", "ppl": gen0_ppl}]
    print(f"\n>>> Gen 0 PPL (Anchor): {gen0_ppl:.2f}")

    # 初始化采样器 (提取真实分布锚点)
    sampler = AdvancedRejectionSampler(GEN0_ORACLE_PATH, EMBEDDING_MODEL_PATH, DEVICE)
    sampler.setup_real_distribution(real_train)

    current_generator_dir = GEN0_ORACLE_PATH

    # [迭代循环]
    OVERSAMPLE_SIZE = int(TARGET_SAMPLES * OVERSAMPLE_RATIO)
    prompt_pool = (real_train * (OVERSAMPLE_SIZE // len(real_train) + 1))[:OVERSAMPLE_SIZE]

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n" + "="*40 + f"\n Generation {gen} | Strategy: {args.strategy}\n" + "="*40)
        
        # 1. 过量生成 (Oversampling)
        print(f" -> Generating {OVERSAMPLE_SIZE} candidate samples...")
        generator = GPT2LMHeadModel.from_pretrained(current_generator_dir).to(DEVICE).eval()
        generator.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        
        candidate_texts = []
        for i in tqdm(range(0, OVERSAMPLE_SIZE, GEN_BATCH_SIZE), desc="    | Synthesizing"):
            batch = prompt_pool[i : i + GEN_BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outputs = generator.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=GEN_TEMPERATURE, top_p=0.95)
            candidate_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
        tokenizer.padding_side = "right"
        del generator
        gc.collect()
        torch.cuda.empty_cache()

        # 2. 拒绝采样路由 (Rejection Pipeline)
        selected_texts, stats = sampler.apply_sampling(candidate_texts, TARGET_SAMPLES, args.strategy)

        # 3. 后续训练与评估
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