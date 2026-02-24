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
OVERSAMPLE_RATIO = 2.0       # 候选池放大倍数 (生成 20000 条供筛选)

GEN_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
GEN_TEMPERATURE = 1.3        # 较高温度，注入必要方差
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

        df = pd.DataFrame({"text": candidate_texts})

        # ==========================================
        # 阶段 A: 微观动态截断 (Fluency First 硬安检)
        # ==========================================
        if strategy in ["micro_only", "combined"]:
            print("    | Running Dynamic Micro-Level Profiling...")
            df['nll'] = self.get_batch_oracle_nll(candidate_texts)
            
            # [改进点 1&2]: 移除 Entropy 下限，改用 NLL 的动态相对阈值
            nll_mean = df['nll'].mean()
            nll_std = df['nll'].std()
            # 只砍掉当前代最差的那些幻觉样本 (NLL 过高)
            dynamic_thresh = nll_mean + 1.5 * nll_std
            
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
        # 阶段 B: 正则化 FBD 宏观搜索 (Regularized FBD)
        # ==========================================
        if strategy in ["macro_only", "combined"]:
            print("    | Running Macro-Level Regularized FBD Search...")
            pool_texts = candidate_df['text'].tolist()
            pool_embs = self.get_embeddings(pool_texts)
            
            # 如果是 combined，我们需要知道每个样本的 NLL 来计算惩罚项
            # 如果是 macro_only，我们只关心 FBD，不惩罚 NLL
            if strategy == "combined":
                pool_nlls = candidate_df['nll'].values
            
            best_idx = None
            best_score = float('inf')
            best_raw_fbd = 0
            num_trials = 20 # 蒙特卡洛搜索次数
            
            # [改进点 3]: NLL 惩罚系数 (Lambda)
            # FBD 通常在 0.05 ~ 0.3 之间，NLL 通常在 3.0 ~ 5.0 之间
            # 我们希望 NLL 每增加 0.1，就相当于 FBD 恶化了 0.05，因此 lambda 设为 0.5 较为合理
            LAMBDA_NLL = 0.5 
            
            for _ in tqdm(range(num_trials), desc="    | Searching Subsets", leave=False):
                subset_idx = np.random.choice(len(pool_embs), target_size, replace=False)
                sub_embs = pool_embs[subset_idx]
                sub_mu = np.mean(sub_embs, axis=0)
                sub_cov = np.cov(sub_embs, rowvar=False)
                
                raw_fbd = self.compute_fbd(self.real_mu, self.real_cov, sub_mu, sub_cov)
                
                if strategy == "combined":
                    subset_mean_nll = np.mean(pool_nlls[subset_idx])
                    # 正则化打分公式：FBD 越低越好，均值 NLL 越低越好
                    current_score = raw_fbd + LAMBDA_NLL * subset_mean_nll
                else:
                    current_score = raw_fbd # macro_only 退化为只看 FBD
                
                if current_score < best_score:
                    best_score = current_score
                    best_raw_fbd = raw_fbd
                    best_idx = subset_idx
                    
            selected_texts = [pool_texts[i] for i in best_idx]
            
            stats = {"fbd_score": best_raw_fbd, "combined_score": best_score}
            if strategy == "combined":
                stats["nll_mean"] = candidate_df.iloc[best_idx]['nll'].mean()
                
            return selected_texts, stats

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

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    EXP_DIR = f"/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v4/exp_{args.strategy}"
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

        # [核心] 运用新的正则化采样器
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