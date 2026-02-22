import os
import argparse
import random
import json
import gc
import math
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from sentence_transformers import SentenceTransformer
from torch.nn import CrossEntropyLoss

# ==========================================
# 0. 全局配置与路径
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径配置
DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"
BASE_MODEL_PATH = "/home/ubuntu/data/model/gpt2_model"
EMBEDDING_MODEL_PATH = "/home/ubuntu/data/model/all-mpnet-base-v2"
GEN0_ORACLE_PATH = "/home/ubuntu/data/simc/gpt2_wikitext2/model_real_trained"

# 实验规模设置
TARGET_SAMPLES = 10000       # 每代实际用于训练的最终样本数
OVERSAMPLE_FACTOR = 2       # 基础放大倍数 (初始生成 20000 条)
NUM_GENERATIONS = 10         # 观测代数
EPOCHS = 5                  # 每代微调轮数
GEN_TEMP = 1.3              # 高温注入方差
TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1

MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64
GEN_BATCH_SIZE = 256

# ==========================================
# 1. 高阶评估器与采样器 (内存安全版)
# ==========================================
class AdvancedRejectionSampler:
    def __init__(self, oracle_path, device='cuda'):
        self.device = device
        self.oracle_path = oracle_path
        self.embed_model = None
        self.oracle_model = None
        self.oracle_tokenizer = None
        self.nll_loss_fct = CrossEntropyLoss(reduction='none')
        
        # 真实数据的真值锚点 (将在 profile_real_data 中被初始化)
        self.real_nll_mean = None
        self.real_nll_std = None
        self.real_mu = None
        self.real_cov = None

    # --- 显存管理机制 ---
    def _load_embed(self):
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self.device)
            self.embed_model.eval()

    def _unload_embed(self):
        if self.embed_model is not None:
            del self.embed_model
            self.embed_model = None
            gc.collect(); torch.cuda.empty_cache()

    def _load_oracle(self):
        if self.oracle_model is None:
            self.oracle_tokenizer = GPT2Tokenizer.from_pretrained(self.oracle_path)
            if self.oracle_tokenizer.pad_token is None:
                self.oracle_tokenizer.pad_token = self.oracle_tokenizer.eos_token
            self.oracle_model = GPT2LMHeadModel.from_pretrained(self.oracle_path).to(self.device)
            self.oracle_model.eval()

    def _unload_oracle(self):
        if self.oracle_model is not None:
            del self.oracle_model; del self.oracle_tokenizer
            self.oracle_model = None; self.oracle_tokenizer = None
            gc.collect(); torch.cuda.empty_cache()

    # --- 核心特征计算 ---
    def get_embeddings(self, texts, batch_size=128):
        self._load_embed()
        embs = self.embed_model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        self._unload_embed()
        return embs

    def get_oracle_nlls(self, texts, batch_size=32):
        """计算 NLL 向量"""
        self._load_oracle()
        entropies = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring NLL", leave=False):
            batch = texts[i : i + batch_size]
            inputs = self.oracle_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.oracle_model(inputs.input_ids, attention_mask=inputs.attention_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                shift_mask = inputs.attention_mask[..., 1:].contiguous()
                loss = self.nll_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
                seq_nll = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
                entropies.extend(seq_nll.cpu().tolist())
        self._unload_oracle()
        return np.array(entropies)

    @staticmethod
    def compute_fbd(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean): covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

    # --- 基线分布描绘 ---
    def profile_real_data(self, real_texts):
        """建立绝对参照系：Gen 0 数据的 NLL 均值/方差，及 MPNet 均值/方差"""
        print("    | Profiling Reference Distributions...")
        nlls = self.get_oracle_nlls(real_texts)
        self.real_nll_mean = np.mean(nlls)
        self.real_nll_std = np.std(nlls)
        
        embs = self.get_embeddings(real_texts)
        self.real_mu = np.mean(embs, axis=0)
        self.real_cov = np.cov(embs, rowvar=False)
        print(f"    | Real NLL Anchor: mu={self.real_nll_mean:.2f}, std={self.real_nll_std:.2f}")

    # --- 最终策略调度 ---
    def execute_strategy(self, candidate_texts, strategy, target_size):
        """基于四种不同策略执行拒绝采样"""
        print(f"    | Executing Strategy: [{strategy.upper()}] on {len(candidate_texts)} candidates")
        
        # 1. RANDOM (Baseline)
        if strategy == "random":
            selected = random.sample(candidate_texts, target_size)
            return selected, {}
            
        # NLL 绝对阈值计算
        lower_bound = self.real_nll_mean - 1.5 * self.real_nll_std
        upper_bound = self.real_nll_mean + 2.0 * self.real_nll_std
        
        # 2. NLL_ONLY (质量过滤)
        if strategy == "nll_only":
            nlls = self.get_oracle_nlls(candidate_texts)
            valid_idx = np.where((nlls >= lower_bound) & (nlls <= upper_bound))[0]
            if len(valid_idx) < target_size:
                print(f"      [Warning] NLL Strict bounds left {len(valid_idx)} < {target_size}. Falling back to best.")
                valid_idx = np.argsort(np.abs(nlls - self.real_nll_mean))[:target_size]
            else:
                valid_idx = np.random.choice(valid_idx, target_size, replace=False)
            return [candidate_texts[i] for i in valid_idx], {"filtered_nll_mean": float(np.mean(nlls[valid_idx]))}
            
        # 3. FBD_ONLY (全局宏观匹配)
        if strategy == "fbd_only":
            embs_pool = self.get_embeddings(candidate_texts)
            best_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(candidate_texts)), target_size)
            return [candidate_texts[i] for i in best_idx], {"best_fbd": best_fbd}

        # 4. COMBINED (双重过滤防御)
        if strategy == "combined":
            nlls = self.get_oracle_nlls(candidate_texts)
            valid_idx = np.where((nlls >= lower_bound) & (nlls <= upper_bound))[0]
            
            # 动态反馈机制：如果合格数量极度不足，发出警告
            if len(valid_idx) < target_size * 1.2:
                print(f"      [Warning] NLL passed={len(valid_idx)}, minimal required={target_size}. FBD search space severely constrained!")
                if len(valid_idx) < target_size:
                    valid_idx = np.argsort(np.abs(nlls - self.real_nll_mean))[:int(target_size*1.2)]
            
            # 在合法的 NLL 空间内，进行 FBD 匹配
            valid_texts = [candidate_texts[i] for i in valid_idx]
            embs_pool = self.get_embeddings(valid_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(valid_texts)), target_size)
            
            final_selected = [valid_texts[i] for i in best_local_idx]
            return final_selected, {"best_fbd": best_fbd, "nll_pass_rate": len(valid_idx)/len(candidate_texts)}

    def _subset_search(self, embs_pool, search_indices, target_size, trials=15):
        """Monte Carlo Subset Search for minimizing FBD"""
        print(f"    | Running FBD Subset Search (Trials={trials})...")
        best_idx = None
        best_fbd = float('inf')
        for _ in range(trials):
            subset_idx = np.random.choice(search_indices, target_size, replace=False)
            sub_emb = embs_pool[subset_idx]
            sub_mu, sub_cov = np.mean(sub_emb, axis=0), np.cov(sub_emb, rowvar=False)
            
            fbd = self.compute_fbd(self.real_mu, self.real_cov, sub_mu, sub_cov)
            if fbd < best_fbd:
                best_fbd = fbd
                best_idx = subset_idx
        return best_idx, best_fbd

# ==========================================
# 2. 训练、生成与评估脚手架
# ==========================================
def generate_pool_dynamic(model, tokenizer, prompt_pool, initial_size, strategy, sampler):
    """动态生成机制：如果不满足策略阈值的余量，则继续生成"""
    model.eval()
    tokenizer.padding_side = "left"
    synthetic_texts = []
    
    # 基础生成
    batch_size = GEN_BATCH_SIZE
    prompts = (prompt_pool * (initial_size // len(prompt_pool) + 1))[:initial_size]
    
    print(f"    | Generating Initial Pool ({initial_size} samples)...")
    for i in tqdm(range(0, initial_size, batch_size), desc="Generating", leave=False):
        batch = prompts[i : i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=GEN_TEMP, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
            synthetic_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        except RuntimeError:
            torch.cuda.empty_cache(); continue
            
    tokenizer.padding_side = "right"
    return synthetic_texts

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated[list(examples.keys())[0]]) // MAX_LENGTH) * MAX_LENGTH
    result = {k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)] for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result

def train_student(texts, tokenizer, output_dir):
    print("    | Training FRESH base model...")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    ds = Dataset.from_dict({"text": texts})
    tokenized = ds.map(lambda x: tokenizer([t + tokenizer.eos_token for t in x["text"]]), batched=True, remove_columns=["text"])
    packed = tokenized.map(group_texts, batched=True)
    
    args = TrainingArguments(
        output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION, bf16=torch.cuda.is_bf16_supported(),
        learning_rate=5e-5, optim="adamw_torch_fused", save_strategy="no", report_to="none"
    )
    trainer = Trainer(model=model, args=args, train_dataset=packed, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(output_dir)
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

def compute_validation_ppl(model_dir, tokenizer, valid_texts):
    print("    | Evaluating PPL on Real Validation Set...")
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    encodings = tokenizer("\n\n".join(valid_texts), return_tensors="pt")
    stride = 512; seq_len = encodings.input_ids.size(1); nlls = []; prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Eval PPL", leave=False):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone(); target_ids[:, :-trg_len] = -100 
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
    del model; gc.collect(); torch.cuda.empty_cache()
    return ppl

# ==========================================
# 3. 实验主循环
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=['random', 'nll_only', 'fbd_only', 'combined'], required=True)
    args = parser.parse_args()
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    EXP_ROOT = f"./rejection_sampling_results_v3/exp_{args.strategy}"
    os.makedirs(EXP_ROOT, exist_ok=True)
    metrics_log_path = os.path.join(EXP_ROOT, "metrics.json")
    
    print(f"\n{'='*50}\n Starting Experiment Strategy: [{args.strategy.upper()}]\n{'='*50}\n")

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    try: dataset = load_from_disk(DATA_PATH)
    except: dataset = load_dataset(DATA_PATH)
    
    real_texts = [t for t in dataset['train']['text'] if len(t.strip()) > 0][:TARGET_SAMPLES * 2]
    valid_texts = [t for t in dataset['validation']['text'] if len(t.strip()) > 0][:]

    # [Gen 0: Oracle & Reference]
    gen0_dir = GEN0_ORACLE_PATH
    # gen0_dir = os.path.join(EXP_ROOT, "gen_0_real")
    if not os.path.exists(gen0_dir):
        print(">>> Training Gen 0 Baseline (Oracle)...")
        train_student(random.sample(real_texts, TARGET_SAMPLES), tokenizer, gen0_dir)
        
    sampler = AdvancedRejectionSampler(oracle_path=gen0_dir, device=DEVICE)
    sampler.profile_real_data(random.sample(real_texts, TARGET_SAMPLES))
    
    gen0_ppl = compute_validation_ppl(gen0_dir, tokenizer, valid_texts)
    print(f">>> Gen 0 Validation PPL: {gen0_ppl:.2f}\n")
    
    history = [{"generation": 0, "strategy": "real_data", "ppl": gen0_ppl}]
    current_generator_dir = gen0_dir

    # [Iterative Generations]
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{'='*20} Generation {gen} {'='*20}")
        gen_dir = os.path.join(EXP_ROOT, f"gen_{gen}")
        
        # 1. 生成大池子
        generator = GPT2LMHeadModel.from_pretrained(current_generator_dir).to(DEVICE)
        generator.config.pad_token_id = tokenizer.pad_token_id
        pool_texts = generate_pool_dynamic(
            generator, tokenizer, real_texts[:TARGET_SAMPLES], TARGET_SAMPLES * OVERSAMPLE_FACTOR, args.strategy, sampler
        )
        del generator; gc.collect(); torch.cuda.empty_cache()
        
        # 2. 拒绝采样执行
        selected_texts, stats = sampler.execute_strategy(pool_texts, args.strategy, TARGET_SAMPLES)
        
        # 3. 训练最新后代
        train_student(selected_texts, tokenizer, gen_dir)
        
        # 4. 评估灾难程度
        ppl = compute_validation_ppl(gen_dir, tokenizer, valid_texts)
        print(f">>> Gen {gen} Validation PPL: {ppl:.2f}")
        
        history.append({"generation": gen, "strategy": args.strategy, "ppl": ppl, **stats})
        with open(metrics_log_path, "w") as f: json.dump(history, f, indent=4)
        
        current_generator_dir = gen_dir

    print("\n>>> Protocol Completed Successfully.")

if __name__ == "__main__":
    main()