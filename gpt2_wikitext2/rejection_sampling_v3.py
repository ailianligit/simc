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

TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64
GEN_BATCH_SIZE = 256

# ==========================================
# 1. 核心评估器与采样器 (解耦 Filter 逻辑)
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
            # self.embed_model.max_seq_length = 512
            self.embed_model.eval()

    def _unload_embed(self):
        if getattr(self, 'embed_model', None) is not None:
            self.embed_model = None 
            gc.collect()
            torch.cuda.empty_cache()

    def _load_oracle(self):
        if self.oracle_model is None:
            self.oracle_model = GPT2LMHeadModel.from_pretrained(self.oracle_path).to(self.device)
            self.oracle_model.eval()

    def _unload_oracle(self):
        if getattr(self, 'oracle_model', None) is not None:
            self.oracle_model = None 
            gc.collect()
            torch.cuda.empty_cache()

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
        effective_target = min(target_size, len(search_indices))
        for _ in range(trials):
            subset_idx = np.random.choice(search_indices, effective_target, replace=False)
            sub_emb = embs_pool[subset_idx]
            sub_mu, sub_cov = np.mean(sub_emb, axis=0), np.cov(sub_emb, rowvar=False)
            
            fbd = self.compute_fbd(self.real_mu, self.real_cov, sub_mu, sub_cov)
            if fbd < best_fbd:
                best_fbd = fbd
                best_idx = subset_idx
        return best_idx, best_fbd

    # [解耦点 1] 完全纯净的过滤策略执行器，不带任何温度逻辑的影子
    def execute_filter_strategy(self, candidate_texts, filter_strategy, target_size):
        print(f"    | Executing Filter Strategy: [{filter_strategy.upper()}] on {len(candidate_texts)} candidates")
        
        if filter_strategy == "random":
            return random.sample(candidate_texts, target_size), {}
            
        nlls = self.get_oracle_nlls(candidate_texts)
        bi_ents = self.get_batch_bigram_entropy(candidate_texts)

        # 预计算漏斗 A 和 B
        nll_thresh = np.quantile(nlls, 0.85) 
        idx_pass_A = np.where(nlls <= nll_thresh)[0]

        ent_thresh = np.quantile(bi_ents, 0.15) 
        idx_pass_B = np.where(bi_ents >= ent_thresh)[0]

        if filter_strategy == "nll_only":
            actual_size = min(target_size, len(idx_pass_A))
            selected_idx = np.random.choice(idx_pass_A, actual_size, replace=False)
            return [candidate_texts[i] for i in selected_idx], {"filtered_nll_mean": float(np.mean(nlls[selected_idx]))}

        if filter_strategy == "entropy_only":
            actual_size = min(target_size, len(idx_pass_B))
            selected_idx = np.random.choice(idx_pass_B, actual_size, replace=False)
            return [candidate_texts[i] for i in selected_idx], {"filtered_ent_mean": float(np.mean(bi_ents[selected_idx]))}

        # [新增] 仅做 FBD，不剔除任何微观数据
        if filter_strategy == "fbd_only":
            embs_pool = self.get_embeddings(candidate_texts)
            best_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(candidate_texts)), target_size)
            final_selected = [candidate_texts[i] for i in best_idx]
            return final_selected, {"best_fbd": best_fbd}

        # [新增] B+C 组合
        if filter_strategy == "entropy_fbd":
            pool_texts = [candidate_texts[i] for i in idx_pass_B]
            if len(pool_texts) <= target_size:
                 return pool_texts, {"best_fbd": 0.0, "filtered_ent_mean": float(np.mean(self.get_batch_bigram_entropy(pool_texts)))}
                 
            embs_pool = self.get_embeddings(pool_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(pool_texts)), target_size)
            final_selected = [pool_texts[i] for i in best_local_idx]
            return final_selected, {"best_fbd": best_fbd, "filtered_ent_mean": float(np.mean(self.get_batch_bigram_entropy(final_selected)))}

        if filter_strategy == "nll_fbd":
            pool_texts = [candidate_texts[i] for i in idx_pass_A]
            if len(pool_texts) <= target_size:
                 return pool_texts, {"best_fbd": 0.0, "filtered_nll_mean": float(np.mean(self.get_oracle_nlls(pool_texts)))}
                 
            embs_pool = self.get_embeddings(pool_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(pool_texts)), target_size)
            final_selected = [pool_texts[i] for i in best_local_idx]
            return final_selected, {"best_fbd": best_fbd, "filtered_nll_mean": float(np.mean(self.get_oracle_nlls(final_selected)))}

        if filter_strategy == "nll_entropy":
            survivor_ents = bi_ents[idx_pass_A]
            ent_thresh_survivors = np.quantile(survivor_ents, 0.20) 
            idx_pass_AB = idx_pass_A[np.where(survivor_ents >= ent_thresh_survivors)[0]]
            
            actual_size = min(target_size, len(idx_pass_AB))
            selected_idx = np.random.choice(idx_pass_AB, actual_size, replace=False)
            final_selected = [candidate_texts[i] for i in selected_idx]
            return final_selected, {
                "filtered_nll_mean": float(np.mean(nlls[selected_idx])),
                "filtered_ent_mean": float(np.mean(bi_ents[selected_idx]))
            }

        if filter_strategy == "combined":
            survivor_ents = bi_ents[idx_pass_A]
            ent_thresh_survivors = np.quantile(survivor_ents, 0.20)
            idx_pass_AB = idx_pass_A[np.where(survivor_ents >= ent_thresh_survivors)[0]]
            
            final_pool_texts = [candidate_texts[i] for i in idx_pass_AB]
            
            if len(final_pool_texts) <= target_size:
                print("      [Warning] Squeeze effect detected. Relaxing bounds to NLL-only.")
                final_pool_texts = [candidate_texts[i] for i in idx_pass_A]
                
            embs_pool = self.get_embeddings(final_pool_texts)
            best_local_idx, best_fbd = self._subset_search(embs_pool, np.arange(len(final_pool_texts)), target_size)
            
            final_selected = [final_pool_texts[i] for i in best_local_idx]
            
            return final_selected, {
                "best_fbd": best_fbd, 
                "filtered_nll_mean": float(np.mean(self.get_oracle_nlls(final_selected, batch_size=64))),
                "filtered_ent_mean": float(np.mean(self.get_batch_bigram_entropy(final_selected)))
            }

# ==========================================
# 2. 闭环控制与动态生成 (解耦 Temp 逻辑)
# ==========================================

def determine_temperature(model, tokenizer, prompt_pool, temp_strategy, real_bi_ent_mean):
    if temp_strategy == "fixed":
        return 1.3
        
    if temp_strategy == "dynamic":
        model.eval()
        tokenizer.padding_side = "left"
        test_prompts = random.sample(prompt_pool, min(128, len(prompt_pool)))
        
        inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
            
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        probe_ent = np.mean([AdvancedRejectionSampler.compute_single_bigram_entropy(t) for t in texts])
        
        k_p = 1.2 
        temp_gain = k_p * max(0, real_bi_ent_mean - probe_ent)
        target_temp = min(2.0, max(1.0, 1.0 + temp_gain))
        
        print(f"    | [PID Control] Probe Entropy: {probe_ent:.2f} (Target: {real_bi_ent_mean:.2f}) -> Dynamic Temp: {target_temp:.2f}")
        return target_temp

def generate_pool(model, tokenizer, prompt_pool, initial_size, target_temp):
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
# 3. 实验主循环 (双参数解耦版)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--temp_strategy", type=str, choices=['fixed', 'dynamic'], required=True, 
                        help="Temperature generation strategy.")
    # [更新] 在 choices 里加入了全部的 8 种过滤策略
    parser.add_argument("--filter_strategy", type=str, 
                        choices=['random', 'nll_only', 'entropy_only', 'fbd_only', 'nll_fbd', 'entropy_fbd', 'nll_entropy', 'combined'], 
                        required=True, help="Rejection sampling pipeline strategy.")
    
    args = parser.parse_args()
    
    experiment_group_name = f"{args.temp_strategy}_{args.filter_strategy}"
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    # [注意] 如果你希望数据存到新目录，可以修改这里的路径名，这里我先保持为 v10 方便你连贯管理
    EXP_ROOT = f"/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v9/exp_{experiment_group_name}"
    os.makedirs(EXP_ROOT, exist_ok=True)
    metrics_log_path = os.path.join(EXP_ROOT, "metrics.json")
    
    print(f"\n{'='*60}\n Starting Experiment Group: [{experiment_group_name.upper()}]\n Temp: {args.temp_strategy} | Filter: {args.filter_strategy}\n{'='*60}\n")

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
    
    history = [{"generation": 0, "experiment_group": experiment_group_name, "ppl": gen0_ppl}]
    current_generator_dir = gen0_dir

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{'='*20} Generation {gen} {'='*20}")
        gen_dir = os.path.join(EXP_ROOT, f"gen_{gen}")
        
        generator = GPT2LMHeadModel.from_pretrained(current_generator_dir).to(DEVICE)
        generator.config.pad_token_id = tokenizer.pad_token_id
        
        target_temp = determine_temperature(generator, tokenizer, real_texts[:TARGET_SAMPLES], args.temp_strategy, sampler.real_bi_ent_mean)
        pool_texts = generate_pool(
            generator, tokenizer, real_texts[:TARGET_SAMPLES], TARGET_SAMPLES * OVERSAMPLE_FACTOR, target_temp
        )
        del generator; gc.collect(); torch.cuda.empty_cache()
        
        selected_texts, stats = sampler.execute_filter_strategy(pool_texts, args.filter_strategy, TARGET_SAMPLES)
        
        train_student(selected_texts, tokenizer, gen_dir)
        ppl = compute_validation_ppl(gen_dir, tokenizer, valid_texts)
        print(f">>> Gen {gen} Validation PPL: {ppl:.2f}")
        
        history.append({
            "generation": gen, "experiment_group": experiment_group_name, "ppl": ppl, 
            "applied_temp": float(target_temp), **stats
        })
        with open(metrics_log_path, "w") as f: json.dump(history, f, indent=4)
        
        current_generator_dir = gen_dir

    print("\n>>> Experiment Group Completed Successfully.")

if __name__ == "__main__":
    main()