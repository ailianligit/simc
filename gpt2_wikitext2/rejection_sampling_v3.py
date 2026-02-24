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

DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"
BASE_MODEL_PATH = "/home/ubuntu/data/model/gpt2_model"
EMBEDDING_MODEL_PATH = "/home/ubuntu/data/model/all-mpnet-base-v2"
GEN0_ORACLE_PATH = "/home/ubuntu/data/simc/gpt2_wikitext2/model_real_trained"

TARGET_SAMPLES = 10000        
OVERSAMPLE_FACTOR = 3
NUM_GENERATIONS = 10        
EPOCHS = 5        

TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
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
        self.real_bi_ent_mean = None

    def _load_embed(self):
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self.device)
            self.embed_model.eval()

    def _unload_embed(self):
        if self.embed_model is not None:
            del self.embed_model; gc.collect(); torch.cuda.empty_cache()

    def _load_oracle(self):
        if self.oracle_model is None:
            self.oracle_model = GPT2LMHeadModel.from_pretrained(self.oracle_path).to(self.device)
            self.oracle_model.eval()

    def _unload_oracle(self):
        if self.oracle_model is not None:
            del self.oracle_model; gc.collect(); torch.cuda.empty_cache()

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
            warnings.simplefilter("ignore")
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if not np.isfinite(covmean).all():
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if np.iscomplexobj(covmean): covmean = covmean.real
            return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

    def profile_real_data(self, real_texts):
        print("    | Profiling Reference Distributions...")
        embs = self.get_embeddings(real_texts)
        self.real_mu = np.mean(embs, axis=0)
        self.real_cov = np.cov(embs, rowvar=False)
        
        bi_ents = self.get_batch_bigram_entropy(real_texts)
        self.real_bi_ent_mean = float(np.mean(bi_ents))

    def _subset_search(self, embs_pool, search_indices, target_size, trials=25):
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

    # --- 8 个策略的精准路由与执行 ---
    def execute_strategy(self, candidate_texts, strategy, target_size):
        base_strategy = strategy.replace("_fixed", "").replace("_dynamic", "")
        print(f"    | Executing Route: [{base_strategy.upper()}] on {len(candidate_texts)} candidates")
        
        # --- 组 1: Baseline ---
        if base_strategy == "random":
            return random.sample(candidate_texts, target_size), {}
            
        nlls = self.get_oracle_nlls(candidate_texts)
        bi_ents = self.get_batch_bigram_entropy(candidate_texts)

        # 定义 A 模块：NLL 微观去毒 (淘汰最差 15%)
        nll_thresh = np.quantile(nlls, 0.85) 
        idx_pass_A = np.where(nlls <= nll_thresh)[0]

        # 定义 B 模块：Entropy 微观保方差 (淘汰最差 15%)
        ent_thresh = np.quantile(bi_ents, 0.15) 
        idx_pass_B = np.where(bi_ents >= ent_thresh)[0]

        # --- 组 2: 仅看通顺度 (A) ---
        if base_strategy == "nll_only":
            selected_idx = np.random.choice(idx_pass_A, target_size, replace=False)
            return [candidate_texts[i] for i in selected_idx], {"filtered_nll_mean": float(np.mean(nlls[selected_idx]))}

        # --- 组 3: 仅看多样性 (B) ---
        if base_strategy == "entropy_only":
            selected_idx = np.random.choice(idx_pass_B, target_size, replace=False)
            return [candidate_texts[i] for i in selected_idx], {"filtered_ent_mean": float(np.mean(bi_ents[selected_idx]))}

        # --- 组 4: 既要通顺，又要宏观对齐 (A + C) ---
        # 证明如果不管微观重复，FBD 依然会搜索出废话
        if base_strategy == "nll_fbd":
            pool_texts = [candidate_texts[i] for i in idx_pass_A]
            embs_pool = self.get_embeddings(pool_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(pool_texts)), target_size)
            final_selected = [pool_texts[i] for i in best_local_idx]
            return final_selected, {"best_fbd": best_fbd, "filtered_nll_mean": float(np.mean(self.get_oracle_nlls(final_selected)))}

        # --- 组 5: 既要通顺，又要单句词汇丰富 (A + B) ---
        # 证明如果不看宏观对齐，局部完美的句子组合起来依然会跑题
        if base_strategy == "nll_entropy":
            survivor_ents = bi_ents[idx_pass_A]
            ent_thresh_survivors = np.quantile(survivor_ents, 0.20) # 在 A 的基础上砍掉 20% B
            idx_pass_AB = idx_pass_A[np.where(survivor_ents >= ent_thresh_survivors)[0]]
            
            selected_idx = np.random.choice(idx_pass_AB, target_size, replace=False)
            final_selected = [candidate_texts[i] for i in selected_idx]
            return final_selected, {
                "filtered_nll_mean": float(np.mean(nlls[selected_idx])),
                "filtered_ent_mean": float(np.mean(bi_ents[selected_idx]))
            }

        # --- 组 6 & 8: 终极三重防线 (A + B + C) ---
        # 也就是 combined 组：通顺 + 丰富 + 宏观对齐
        if base_strategy == "combined":
            survivor_ents = bi_ents[idx_pass_A]
            ent_thresh_survivors = np.quantile(survivor_ents, 0.20)
            idx_pass_AB = idx_pass_A[np.where(survivor_ents >= ent_thresh_survivors)[0]]
            
            final_pool_texts = [candidate_texts[i] for i in idx_pass_AB]
            
            # Fallback 防止漏斗过窄
            if len(final_pool_texts) < target_size:
                final_pool_texts = [candidate_texts[i] for i in idx_pass_A][:int(target_size*1.5)]
                
            embs_pool = self.get_embeddings(final_pool_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(final_pool_texts)), target_size)
            
            final_selected = [final_pool_texts[i] for i in best_local_idx]
            
            return final_selected, {
                "best_fbd": best_fbd, 
                "filtered_nll_mean": float(np.mean(self.get_oracle_nlls(final_selected, batch_size=64))),
                "filtered_ent_mean": float(np.mean(self.get_batch_bigram_entropy(final_selected)))
            }

# ==========================================
# 2. 闭环控制与动态生成
# ==========================================
def determine_temperature(model, tokenizer, prompt_pool, strategy, real_bi_ent_mean):
    """根据后缀路由生成阶段的物理温度策略"""
    if "fixed" in strategy:
        return 1.3
        
    if "dynamic" in strategy:
        model.eval()
        tokenizer.padding_side = "left"
        test_prompts = random.sample(prompt_pool, min(128, len(prompt_pool)))
        
        inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
            
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        probe_ent = np.mean(AdvancedRejectionSampler.get_batch_bigram_entropy(None, texts)) 
        
        k_p = 1.2 
        temp_gain = k_p * max(0, real_bi_ent_mean - probe_ent)
        target_temp = min(2.0, max(1.0, 1.0 + temp_gain))
        
        print(f"    | [PID Control] Probe Entropy: {probe_ent:.2f} (Target: {real_bi_ent_mean:.2f}) -> Dynamic Temp: {target_temp:.2f}")
        return target_temp

def generate_pool(model, tokenizer, prompt_pool, initial_size, target_temp):
    """纯净生成，只使用测定好的目标温度"""
    model.eval()
    tokenizer.padding_side = "left"
    synthetic_texts = []
    
    batch_size = GEN_BATCH_SIZE
    prompts = (prompt_pool * (initial_size // len(prompt_pool) + 1))[:initial_size]
    
    print(f"    | Generating Pool ({initial_size} samples) at Target T={target_temp:.2f}...")
    for i in tqdm(range(0, initial_size, batch_size), desc="Generating", leave=False):
        batch = prompts[i : i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True, 
                    temperature=target_temp, top_p=0.95, pad_token_id=tokenizer.eos_token_id
                )
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
    # 严格支持这 8 种完整的消融组合
    CHOICES = [
        'random_fixed', 
        'nll_only_fixed', 
        'entropy_only_fixed', 
        'nll_fbd_fixed', 
        'nll_entropy_fixed', 
        'combined_fixed', 
        'random_dynamic', 
        'combined_dynamic'
    ]
    parser.add_argument("--strategy", type=str, choices=CHOICES, required=True)
    args = parser.parse_args()
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    EXP_ROOT = f"/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v9/exp_{args.strategy}"
    os.makedirs(EXP_ROOT, exist_ok=True)
    metrics_log_path = os.path.join(EXP_ROOT, "metrics.json")
    
    print(f"\n{'='*50}\n Starting Ablation Group: [{args.strategy.upper()}]\n{'='*50}\n")

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    try: dataset = load_from_disk(DATA_PATH)
    except: dataset = load_dataset(DATA_PATH)
    
    real_texts = [t for t in dataset['train']['text'] if len(t.strip()) > 0][:TARGET_SAMPLES * 2]
    valid_texts = [t for t in dataset['validation']['text'] if len(t.strip()) > 0][:]

    gen0_dir = GEN0_ORACLE_PATH
    if not os.path.exists(gen0_dir):
        train_student(random.sample(real_texts, TARGET_SAMPLES), tokenizer, gen0_dir)
        
    sampler = AdvancedRejectionSampler(oracle_path=gen0_dir, tokenizer=tokenizer, device=DEVICE)
    sampler.profile_real_data(random.sample(real_texts, TARGET_SAMPLES))
    
    gen0_ppl = compute_validation_ppl(gen0_dir, tokenizer, valid_texts)
    print(f">>> Gen 0 Validation PPL: {gen0_ppl:.2f}\n")
    
    history = [{"generation": 0, "strategy": "real_data", "ppl": gen0_ppl}]
    current_generator_dir = gen0_dir

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{'='*20} Generation {gen} {'='*20}")
        gen_dir = os.path.join(EXP_ROOT, f"gen_{gen}")
        
        # 1. 独立解析温度控制策略
        generator = GPT2LMHeadModel.from_pretrained(current_generator_dir).to(DEVICE)
        generator.config.pad_token_id = tokenizer.pad_token_id
        
        target_temp = determine_temperature(generator, tokenizer, real_texts[:TARGET_SAMPLES], args.strategy, sampler.real_bi_ent_mean)
        
        pool_texts = generate_pool(
            generator, tokenizer, real_texts[:TARGET_SAMPLES], TARGET_SAMPLES * OVERSAMPLE_FACTOR, target_temp
        )
        del generator; gc.collect(); torch.cuda.empty_cache()
        
        # 2. 独立解析过滤截断策略
        selected_texts, stats = sampler.execute_strategy(pool_texts, args.strategy, TARGET_SAMPLES)
        
        # 3. 训练 & 评估
        train_student(selected_texts, tokenizer, gen_dir)
        ppl = compute_validation_ppl(gen_dir, tokenizer, valid_texts)
        print(f">>> Gen {gen} Validation PPL: {ppl:.2f}")
        
        history.append({
            "generation": gen, "strategy": args.strategy, "ppl": ppl, 
            "applied_temp": float(target_temp), **stats
        })
        with open(metrics_log_path, "w") as f: json.dump(history, f, indent=4)
        
        current_generator_dir = gen_dir

    print("\n>>> Experiment Group Completed Successfully.")

if __name__ == "__main__":
    main()