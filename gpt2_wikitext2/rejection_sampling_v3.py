import os
import argparse
import random
import json
import gc
import math
from collections import Counter
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
TARGET_SAMPLES = 5000        # 每代实际用于训练的最终样本数 (调小一点加速实验)
OVERSAMPLE_FACTOR = 4        # 放大 4 倍 (初始生成 20000 条) 提供充足的方差池
NUM_GENERATIONS = 10         
EPOCHS = 4                  

TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64
GEN_BATCH_SIZE = 128

# ==========================================
# 1. 核心评估器与采样器
# ==========================================
class AdvancedRejectionSampler:
    def __init__(self, oracle_path, tokenizer, device='cuda'):
        self.device = device
        self.oracle_path = oracle_path
        self.tokenizer = tokenizer
        self.embed_model = None
        self.oracle_model = None
        self.nll_loss_fct = CrossEntropyLoss(reduction='none')
        
        # 真实数据的真值锚点
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
            self.oracle_model = GPT2LMHeadModel.from_pretrained(self.oracle_path).to(self.device)
            self.oracle_model.eval()

    def _unload_oracle(self):
        if self.oracle_model is not None:
            del self.oracle_model
            self.oracle_model = None
            gc.collect(); torch.cuda.empty_cache()

    # --- 核心特征计算 ---
    def get_embeddings(self, texts, batch_size=128):
        self._load_embed()
        embs = self.embed_model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        self._unload_embed()
        return embs

    def get_oracle_nlls(self, texts, batch_size=32):
        self._load_oracle()
        entropies = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring NLL", leave=False):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
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
    def compute_single_bigram_entropy(text):
        tokens = text.strip().split()
        if len(tokens) < 2: return 0.0
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        counts = Counter(bigrams)
        total = len(bigrams)
        return -sum((c/total) * math.log(c/total) for c in counts.values())

    def get_batch_bigram_entropy(self, texts):
        return np.array([self.compute_single_bigram_entropy(t) for t in texts])

    @staticmethod
    def compute_fbd(mu1, sigma1, mu2, sigma2, eps=1e-6):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # 忽略 scipy 的弃用警告
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if not np.isfinite(covmean).all():
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if np.iscomplexobj(covmean): covmean = covmean.real
            return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

    # --- 基线分布描绘 ---
    def profile_real_data(self, real_texts):
        print("    | Profiling Reference Distributions for FBD...")
        embs = self.get_embeddings(real_texts)
        self.real_mu = np.mean(embs, axis=0)
        self.real_cov = np.cov(embs, rowvar=False)

    # --- 最终策略调度 ---
    def execute_strategy(self, candidate_texts, strategy, target_size):
        """核心拒绝采样：纯信息论与相对分布对齐"""
        print(f"    | Executing Strategy: [{strategy.upper()}] on {len(candidate_texts)} candidates")
        
        # 0. 基线组
        if strategy == "random":
            return random.sample(candidate_texts, target_size), {}
            
        nlls = self.get_oracle_nlls(candidate_texts)
        bi_ents = self.get_batch_bigram_entropy(candidate_texts)

        # 1. 消融组 A：仅控制通顺度 (淘汰最差的 15% NLL)
        if strategy == "nll_only":
            nll_thresh = np.quantile(nlls, 0.85) # 保留 NLL 较小的 85%
            valid_idx = np.where(nlls <= nll_thresh)[0]
            selected_idx = np.random.choice(valid_idx, target_size, replace=False)
            return [candidate_texts[i] for i in selected_idx], {"filtered_nll_mean": float(np.mean(nlls[selected_idx]))}

        # 2. 消融组 B：仅控制多样性 (保留 Entropy 最大的 85%)
        if strategy == "entropy_only":
            ent_thresh = np.quantile(bi_ents, 0.15) # 淘汰 Entropy 最小的 15%
            valid_idx = np.where(bi_ents >= ent_thresh)[0]
            selected_idx = np.random.choice(valid_idx, target_size, replace=False)
            return [candidate_texts[i] for i in selected_idx], {"filtered_ent_mean": float(np.mean(bi_ents[selected_idx]))}

        # 3. 消融组 C：仅宏观对齐 (不做任何微观截断，直接去拼凑高斯分布)
        if strategy == "fbd_only":
            embs_pool = self.get_embeddings(candidate_texts)
            best_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(candidate_texts)), target_size)
            return [candidate_texts[i] for i in best_idx], {"best_fbd": best_fbd}

        # 4. 终极防御组 (COMBINED: 相对 NLL 截断 + 相对 Entropy 截断 + FBD 搜索)
        if strategy == "combined":
            # 步骤 1：去毒 (Denoising) - 淘汰绝对乱码
            nll_thresh = np.quantile(nlls, 0.85) 
            nll_passed = np.where(nlls <= nll_thresh)[0]
            
            # 步骤 2：防同质化 (Anti-Mode-Collapse) - 淘汰车轱辘话
            # 注意：是在 nll_passed 的幸存者里，计算熵的相对分位数
            survivor_ents = bi_ents[nll_passed]
            ent_thresh = np.quantile(survivor_ents, 0.20) # 淘汰剩下的里面最单调的 20%
            
            final_pool_idx = nll_passed[np.where(survivor_ents >= ent_thresh)[0]]
            final_pool_texts = [candidate_texts[i] for i in final_pool_idx]
            
            # 确保余量足够供 FBD 搜索
            if len(final_pool_texts) < target_size:
                print("      [Warning] Squeeze effect detected. Relaxing bounds.")
                final_pool_texts = [candidate_texts[i] for i in nll_passed][:int(target_size*1.5)]
                
            print(f"      -> Filter Pool Size for FBD: {len(final_pool_texts)} / {len(candidate_texts)}")

            # 步骤 3：宏观对齐 (Macro Alignment)
            embs_pool = self.get_embeddings(final_pool_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(final_pool_texts)), target_size)
            
            final_selected = [final_pool_texts[i] for i in best_local_idx]
            
            # 用于记录的数据
            sel_nlls = self.get_oracle_nlls(final_selected, batch_size=64)
            sel_ents = self.get_batch_bigram_entropy(final_selected)
            
            return final_selected, {
                "best_fbd": best_fbd, 
                "filtered_nll_mean": float(np.mean(sel_nlls)),
                "filtered_ent_mean": float(np.mean(sel_ents))
            }

    def _subset_search(self, embs_pool, search_indices, target_size, trials=25):
        # 稍微增加搜索次数以应对扩大了的 OVERSAMPLE_FACTOR
        print(f"    | Running FBD Subset Search (Trials={trials})...")
        best_idx = None
        best_fbd = float('inf')
        for _ in range(trials):
            subset_idx = np.random.choice(search_indices, min(target_size, len(search_indices)), replace=False)
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
def generate_pool_dynamic(model, tokenizer, prompt_pool, initial_size, current_gen):
    """代际温度退火注入方差"""
    model.eval()
    tokenizer.padding_side = "left"
    synthetic_texts = []
    
    # [核心优化] 强制注入物理方差
    # 随着代数加深，强迫模型使用更高的温度，并略微增加 repetition_penalty
    forced_temp = min(2.0, 1.0 + (0.12 * current_gen)) 
    rep_penalty = min(1.2, 1.0 + (0.02 * current_gen))

    batch_size = GEN_BATCH_SIZE
    prompts = (prompt_pool * (initial_size // len(prompt_pool) + 1))[:initial_size]
    
    print(f"    | Generating Initial Pool ({initial_size} samples) at T={forced_temp:.2f}, RepPen={rep_penalty:.2f}...")
    for i in tqdm(range(0, initial_size, batch_size), desc="Generating", leave=False):
        batch = prompts[i : i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True, 
                    temperature=forced_temp, 
                    repetition_penalty=rep_penalty,
                    top_p=0.95, 
                    pad_token_id=tokenizer.eos_token_id
                )
            synthetic_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        except RuntimeError:
            torch.cuda.empty_cache(); continue
            
    tokenizer.padding_side = "right"
    return synthetic_texts, forced_temp

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
    # [优化] 加入了 entropy_only 实验组
    parser.add_argument("--strategy", type=str, choices=['random', 'nll_only', 'entropy_only', 'fbd_only', 'combined'], required=True)
    args = parser.parse_args()
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    EXP_ROOT = f"/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v7/exp_{args.strategy}"
    os.makedirs(EXP_ROOT, exist_ok=True)
    metrics_log_path = os.path.join(EXP_ROOT, "metrics.json")
    
    print(f"\n{'='*50}\n Starting Experiment Strategy: [{args.strategy.upper()}]\n{'='*50}\n")

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    try: dataset = load_from_disk(DATA_PATH)
    except: dataset = load_dataset(DATA_PATH)
    
    real_texts = [t for t in dataset['train']['text'] if len(t.strip()) > 0][:TARGET_SAMPLES * 2]
    valid_texts = [t for t in dataset['validation']['text'] if len(t.strip()) > 0][:] # 略微缩减验证集提速

    # [Gen 0: Oracle & Reference]
    gen0_dir = GEN0_ORACLE_PATH
    if not os.path.exists(gen0_dir):
        print(">>> Training Gen 0 Baseline (Oracle)...")
        train_student(random.sample(real_texts, TARGET_SAMPLES), tokenizer, gen0_dir)
        
    sampler = AdvancedRejectionSampler(oracle_path=gen0_dir, tokenizer=tokenizer, device=DEVICE)
    sampler.profile_real_data(random.sample(real_texts, TARGET_SAMPLES))
    
    gen0_ppl = compute_validation_ppl(gen0_dir, tokenizer, valid_texts)
    print(f">>> Gen 0 Validation PPL: {gen0_ppl:.2f}\n")
    
    history = [{"generation": 0, "strategy": "real_data", "ppl": gen0_ppl}]
    current_generator_dir = gen0_dir

    # [Iterative Generations]
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{'='*20} Generation {gen} {'='*20}")
        gen_dir = os.path.join(EXP_ROOT, f"gen_{gen}")
        
        # 1. 生成大池子 (强行退火加温)
        generator = GPT2LMHeadModel.from_pretrained(current_generator_dir).to(DEVICE)
        generator.config.pad_token_id = tokenizer.pad_token_id
        pool_texts, used_temp = generate_pool_dynamic(
            generator, tokenizer, real_texts[:TARGET_SAMPLES], TARGET_SAMPLES * OVERSAMPLE_FACTOR, current_gen=gen
        )
        del generator; gc.collect(); torch.cuda.empty_cache()
        
        # 2. 拒绝采样执行 (相对截断设计)
        selected_texts, stats = sampler.execute_strategy(pool_texts, args.strategy, TARGET_SAMPLES)
        
        # 3. 训练最新后代
        train_student(selected_texts, tokenizer, gen_dir)
        
        # 4. 评估灾难程度
        ppl = compute_validation_ppl(gen_dir, tokenizer, valid_texts)
        print(f">>> Gen {gen} Validation PPL: {ppl:.2f}")
        
        # 记录
        history.append({
            "generation": gen, "strategy": args.strategy, "ppl": ppl, 
            "applied_temp": used_temp, **stats
        })
        with open(metrics_log_path, "w") as f: json.dump(history, f, indent=4)
        
        current_generator_dir = gen_dir

    print("\n>>> Protocol Completed Successfully.")

if __name__ == "__main__":
    main()