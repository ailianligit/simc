import os
import torch
import random
import json
import numpy as np
import gc
import ot  # POT library
import mauve
from tqdm import tqdm
from scipy import linalg
from datasets import load_from_disk, Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from torch.nn import CrossEntropyLoss

# ==========================================
# 0. 全局路径与配置
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 基础路径
DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"
BASE_MODEL_PATH = "/home/ubuntu/data/model/gpt2_model" 
EMBEDDING_MODEL_PATH = "/home/ubuntu/data/model/all-mpnet-base-v2"

# [关键修改] 指定由第一份代码训练出的 Gen 0 模型路径 (真实数据训练的模型)
# 如果找不到这个路径，脚本会自动回退到使用 gpt2-large (但强烈建议使用 Gen 0)
REAL_DATA_MODEL_PATH = "/home/ubuntu/data/simc/gpt2_wikitext2/model_real_trained"
FALLBACK_ENTROPY_MODEL = "gpt2-large"

# 指定要分析的生成器路径 (Gen i Checkpoint)
ROUND = 0
GENERATOR_PATH = f"/home/ubuntu/data/simc/gpt2_wikitext2/model_collapse_results_v3/gen_{ROUND}_model"

RESULT_ROOT = f"sensitivity_analysis_results_v2/round_{ROUND+1}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 实验设置 ---
EPOCHS = 5                   # 敏感度分析通常不需要训练太久，3 Epoch 足够看趋势
TRAIN_BATCH_SIZE = 8
GEN_BATCH_SIZE = 64
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 1024            # [关键] 全局对齐长度
PROMPT_LEN_BASE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.03
USE_BF16 = True if torch.cuda.is_bf16_supported() else False
USE_FP16 = False

# 评估采样量
OT_SAMPLE_SIZE = 5000        # OT计算慢，限制数量
FBD_SAMPLE_SIZE = 10000       # FBD/MAUVE/Entropy 采样量

# ==========================================
# 1. 实验变量组 (Sensitivity Grid)
# ==========================================
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SENSITIVITY_CONFIGS = [
    {"name": f"Temp_{t}", "temp": t, "top_k": 50, "top_p": 0.95} 
    for t in TEMPERATURES
]

# ==========================================
# 2. 核心指标计算器 (修复版: 动态显存管理)
# ==========================================
class MetricsEvaluator:
    def __init__(self, device='cuda', embedding_path=EMBEDDING_MODEL_PATH):
        self.device = device
        self.embedding_path = embedding_path
        
        # 自动决定使用哪个模型作为 Oracle
        if os.path.exists(REAL_DATA_MODEL_PATH):
            print(f">>> [Metrics] Using Real Data Model (Gen 0) as Oracle: {REAL_DATA_MODEL_PATH}")
            self.entropy_path = REAL_DATA_MODEL_PATH
        else:
            print(f">>> [Metrics] Gen 0 not found. Fallback to generic Oracle: {FALLBACK_ENTROPY_MODEL}")
            self.entropy_path = FALLBACK_ENTROPY_MODEL
        
        # [关键设计] 初始化时不加载模型，防止占用显存
        self.embed_model = None
        self.entropy_model = None
        self.entropy_tokenizer = None
        self.nll_loss_fct = CrossEntropyLoss(reduction='none')
        self.real_stats = None 

    # --- 显存管理方法 ---
    def _load_embed_model(self):
        if self.embed_model is None:
            # print(f"    | Loading Embedding Model...")
            self.embed_model = SentenceTransformer(self.embedding_path, device=self.device)
            self.embed_model.eval()

    def _unload_embed_model(self):
        if self.embed_model is not None:
            del self.embed_model
            self.embed_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def _load_entropy_model(self):
        if self.entropy_model is None:
            # print(f"    | Loading Entropy Model...")
            self.entropy_tokenizer = GPT2Tokenizer.from_pretrained(self.entropy_path)
            if self.entropy_tokenizer.pad_token is None:
                self.entropy_tokenizer.pad_token = self.entropy_tokenizer.eos_token
            self.entropy_model = GPT2LMHeadModel.from_pretrained(self.entropy_path).to(self.device)
            self.entropy_model.eval()

    def _unload_entropy_model(self):
        if self.entropy_model is not None:
            del self.entropy_model
            del self.entropy_tokenizer
            self.entropy_model = None
            self.entropy_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    # --- 指标计算方法 ---
    def get_embeddings(self, texts, batch_size=64):
        self._load_embed_model() 
        embeddings = self.embed_model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, 
            convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings

    def get_oracle_entropy(self, texts, batch_size=16):
        """[修复] 使用 CrossEntropyLoss 计算标准的 NLL"""
        self._load_entropy_model()
        valid_texts = [t for t in texts if len(t.strip()) > 0]
        if not valid_texts: return 0.0
        
        nlls = []
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="Calc Entropy", leave=False):
            batch = valid_texts[i : i + batch_size]
            inputs = self.entropy_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.entropy_model(inputs.input_ids, attention_mask=inputs.attention_mask)
                logits = outputs.logits
                
                # Shift for Causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                shift_mask = inputs.attention_mask[..., 1:].contiguous()

                loss = self.nll_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())
                
                # NLL per sequence
                seq_nll = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
                nlls.extend(seq_nll.cpu().tolist())
                
        return np.mean(nlls) if nlls else 0.0

    def compute_empirical_entropy(self, texts, n=2):
        from collections import Counter
        import math
        total_counts = Counter()
        total_ngrams = 0
        for text in texts:
            tokens = text.strip().split()
            if len(tokens) < n: continue
            ngrams = list(zip(*[tokens[i:] for i in range(n)]))
            total_counts.update(ngrams)
            total_ngrams += len(ngrams)
        
        if total_ngrams == 0: return 0.0
        entropy_val = 0.0
        for count in total_counts.values():
            p = count / total_ngrams
            entropy_val -= p * math.log(p)
        return entropy_val

    def calculate_exact_ot_distance(self, source_emb, target_emb, metric='euclidean'):
        source_emb = np.array(source_emb, dtype=np.float64)
        target_emb = np.array(target_emb, dtype=np.float64)
        n, m = source_emb.shape[0], target_emb.shape[0]
        a, b = np.ones((n,)) / n, np.ones((m,)) / m
        M = ot.dist(source_emb, target_emb, metric=metric)
        ot_cost = ot.emd2(a, b, M, numItermax=1000000)
        return ot_cost

    def compute_fbd(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1).astype(np.float64)
        mu2 = np.atleast_1d(mu2).astype(np.float64)
        sigma1 = np.atleast_2d(sigma1).astype(np.float64)
        sigma2 = np.atleast_2d(sigma2).astype(np.float64)

        diff = mu1 - mu2
        mean_term = diff.dot(diff)
        cov_dot = sigma1.dot(sigma2)
        covmean, _ = linalg.sqrtm(cov_dot, disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return float(mean_term + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def compute_mauve(self, real_texts, syn_texts, max_samples=10000):
        """MAUVE 计算 (显存敏感)"""
        # 强制卸载其他模型
        self._unload_embed_model()
        self._unload_entropy_model()
        
        # print(f"    | Calculating MAUVE ({max_samples} samples)...")
        p_text = random.sample(real_texts, min(len(real_texts), max_samples))
        q_text = random.sample(syn_texts, min(len(syn_texts), max_samples))

        try:
            out = mauve.compute_mauve(
                p_text=p_text, q_text=q_text,
                device_id=0 if self.device == 'cuda' else -1,
                max_text_length=MAX_LENGTH, verbose=False,
                featurize_model_name="gpt2-large"
            )
            score = out.mauve
        except Exception as e:
            print(f"    | [Warning] MAUVE failed: {e}")
            score = -1.0
        
        torch.cuda.empty_cache()
        return score

    def run_evaluation(self, real_texts, syn_texts):
        # 1. 采样
        N_EVAL = min(len(real_texts), len(syn_texts), FBD_SAMPLE_SIZE)
        N_OT = min(len(real_texts), len(syn_texts), OT_SAMPLE_SIZE)
        
        real_sample = random.sample(real_texts, N_EVAL)
        syn_sample = random.sample(syn_texts, N_EVAL)

        # 2. 计算 Real Stats (Cache)
        if self.real_stats is None or self.real_stats["n_samples"] < N_EVAL:
            print("    | [Metrics] Computing Real Stats...")
            real_emb = self.get_embeddings(real_sample)
            real_ent = self.get_oracle_entropy(real_sample) # 趁模型在显存，算一下
            
            self.real_stats = {
                "emb": real_emb,
                "mu": np.mean(real_emb, axis=0),
                "cov": np.cov(real_emb, rowvar=False),
                "ent": real_ent,
                "n_samples": len(real_sample)
            }
            self._unload_entropy_model() # 算完卸载

        # 3. 计算 Syn Stats
        print("    | [Metrics] Computing Syn Stats...")
        syn_emb = self.get_embeddings(syn_sample)
        syn_mu = np.mean(syn_emb, axis=0)
        syn_cov = np.cov(syn_emb, rowvar=False)
        self._unload_embed_model() # 算完卸载 MPNet
        
        syn_ent = self.get_oracle_entropy(syn_sample)
        self._unload_entropy_model() # 算完卸载 Entropy Model

        syn_uni_ent = self.compute_empirical_entropy(syn_texts, 1)
        syn_bi_ent = self.compute_empirical_entropy(syn_texts, 2)

        # 4. 距离计算 (CPU)
        fbd = self.compute_fbd(self.real_stats['mu'], self.real_stats['cov'], syn_mu, syn_cov)
        
        print(f"    | [Metrics] Calculating Exact OT (Subset {N_OT})...")
        ot_dist = self.calculate_exact_ot_distance(
            self.real_stats['emb'][:N_OT], syn_emb[:N_OT]
        )
        
        # 5. MAUVE (独占显存)
        mauve_score = self.compute_mauve(real_texts, syn_texts)

        return {
            "oracle_entropy": float(syn_ent),
            "empirical_unigram_entropy": float(syn_uni_ent),
            "empirical_bigram_entropy": float(syn_bi_ent),
            "fbd_score": float(fbd),
            "exact_ot_distance": float(ot_dist),
            "mauve_score": float(mauve_score),
            "ref_oracle_entropy": float(self.real_stats['ent'])
        }

# ==========================================
# 3. 辅助功能函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_ppl(model, tokenizer, dataset, device=DEVICE):
    model.eval()
    print(f"    -> Evaluating PPL on {len(dataset)} validation samples...")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    # 仅评估前 30% 以加速敏感度分析
    limit = seq_len
    pbar = tqdm(range(0, limit, stride), desc="Eval PPL", leave=False)
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
        
    return torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length >= MAX_LENGTH:
        total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
    result = {
        k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train_one_round(config_name, train_dataset, tokenizer):
    print(f"    | [Train] Initializing FRESH Base Model from {BASE_MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize(examples):
        return tokenizer([t + tokenizer.eos_token for t in examples["text"]])

    tokenized = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    packed = tokenized.map(group_texts, batched=True)
    
    args = TrainingArguments(
        output_dir=os.path.join(RESULT_ROOT, config_name, "model"),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        bf16=USE_BF16,
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=packed, 
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    return model

# ==========================================
# 4. 生成函数 (修复长度问题)
# ==========================================
def generate_synthetic_data(model, tokenizer, prompt_texts_list, num_samples, config):
    model.eval()
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    # 清洗 Prompt
    clean_prompts = [t for t in prompt_texts_list if len(t.strip()) > 0]
    while len(clean_prompts) < num_samples:
        clean_prompts.extend(clean_prompts)
    target_prompts = clean_prompts[:num_samples]
    
    synthetic_texts = []
    
    for i in tqdm(range(0, len(target_prompts), GEN_BATCH_SIZE), desc=f"Syn ({config['name']})"):
        batch_prompts = target_prompts[i : i + GEN_BATCH_SIZE]
        
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE
        ).to(DEVICE)
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float16):
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        # [修复] 允许生成更长的文本，覆盖 WikiText 分布
                        max_length=min(MAX_LENGTH, inputs.input_ids.shape[1] + 512), 
                        do_sample=True,
                        temperature=config["temp"],
                        top_k=config["top_k"],
                        top_p=config["top_p"],
                        pad_token_id=tokenizer.eos_token_id
                    )
            
            gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            synthetic_texts.extend(gen_texts)
            del inputs, outputs

        except RuntimeError:
            torch.cuda.empty_cache()
            continue

    tokenizer.padding_side = "right"
    # 截断到目标数量
    final_data = synthetic_texts[:num_samples]
    return Dataset.from_dict({"text": final_data})

# ==========================================
# 5. 主流程
# ==========================================
def main():
    set_seed(42)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    
    print(">>> Loading Data & Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    try: dataset = load_from_disk(DATA_PATH)
    except: dataset = load_dataset(DATA_PATH)
        
    real_texts_list = dataset["train"]["text"]
    valid_dataset = dataset["validation"]
    
    # 过滤空样本
    real_texts_list = [t for t in real_texts_list if len(t.strip()) > 0]
    SAMPLES_TO_GENERATE = len(real_texts_list)
    print(f">>> [Config] Target Valid Samples: {SAMPLES_TO_GENERATE}")
    
    print(f">>> Loading Generator from: {GENERATOR_PATH}")
    generator = GPT2LMHeadModel.from_pretrained(GENERATOR_PATH).to(DEVICE)
    generator.config.pad_token_id = tokenizer.pad_token_id
    
    # 初始化评估器 (此时不占显存)
    evaluator = MetricsEvaluator(DEVICE)
    
    results_path = os.path.join(RESULT_ROOT, "final_results.json")
    results = []
    finished_experiments = set()

    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
            finished_experiments = {item["name"] for item in results}
            print(f">>> [Resume] Found {len(results)} completed experiments.")
        except: pass

    # 实验循环
    for exp_cfg in SENSITIVITY_CONFIGS:
        if exp_cfg["name"] in finished_experiments:
            print(f"\n=== Skipping: {exp_cfg['name']} ===")
            continue

        print(f"\n=== Running: {exp_cfg['name']} ===")
        
        # 1. 生成
        print("   -> Generating...")
        # [修复 Bug 1] 补全参数
        syn_dataset = generate_synthetic_data(
            model=generator,
            tokenizer=tokenizer,
            prompt_texts_list=real_texts_list,
            num_samples=SAMPLES_TO_GENERATE,
            config=exp_cfg
        )

        # 移动 Generator 到 CPU
        print("   -> Moving Generator to CPU...")
        generator.to('cpu') 
        torch.cuda.empty_cache()
        gc.collect()

        # 2. 评估
        print("   -> Measuring Metrics...")
        # 注意：run_evaluation 内部已经有了 Bug 2 的修复逻辑（参考上文）
        data_metrics = evaluator.run_evaluation(real_texts_list, syn_dataset["text"])
        print(f"   -> Metrics: FBD={data_metrics['fbd_score']:.2f}, OT={data_metrics['exact_ot_distance']:.2f}")
        
        # 再次清理显存，确保 Metrics 模型全部卸载
        evaluator._unload_embed_model()
        evaluator._unload_entropy_model()
        torch.cuda.empty_cache()
        
        # 3. 训练
        print("   -> Training Student Model...")
        student_model = train_one_round(exp_cfg["name"], syn_dataset, tokenizer)
        
        # 4. 评估 PPL
        print("   -> Evaluating Performance...")
        ppl = compute_ppl(student_model, tokenizer, valid_dataset)
        print(f"   -> Validation PPL: {ppl:.2f}")
        
        # 5. 保存
        record = {**exp_cfg, **data_metrics, "validation_ppl": ppl}
        results.append(record)
        
        with open(results_path + ".tmp", "w") as f: json.dump(results, f, indent=4)
        os.replace(results_path + ".tmp", results_path)
            
        # 6. 清理
        del student_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # 将 Generator 移回 GPU 准备下一轮
        generator.to(DEVICE)

if __name__ == "__main__":
    main()