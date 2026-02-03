import os
import torch
import random
import json
import numpy as np
import gc
import ot  # 引入 Python Optimal Transport 库
from tqdm import tqdm
from scipy import linalg
from datasets import load_from_disk, Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

# ==========================================
# 0. 全局路径与配置
# ==========================================
# 原始数据路径
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"

# 指定生成器路径 (Gen i Checkpoint)
ROUND = 0
GENERATOR_PATH = f"/home/ubuntu/data/simc/gpt2_wikitext2/model_collapse_results_v1/gen_{ROUND}_model"

# 模型选择
# 用于训练的学生模型底座 (Fresh Base)
BASE_MODEL_PATH = "/home/ubuntu/data/model/gpt2_model" 
# 用于 Embedding 计算 (OT/FBD) - 推荐使用 MPNet
EMBEDDING_MODEL_PATH = "/home/ubuntu/data/model/all-mpnet-base-v2"
# 用于 Entropy 计算 (Oracle) - 必须是自回归模型
ENTROPY_MODEL_PATH = "/home/ubuntu/data/model/gpt2_large" # 或 gpt2-xl

RESULT_ROOT = f"sensitivity_analysis/round_{ROUND+1}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 实验设置 ---
EPOCHS = 5    
TRAIN_BATCH_SIZE = 32       
GEN_BATCH_SIZE = 128     
GRADIENT_ACCUMULATION = 1
MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.03
USE_BF16 = True if torch.cuda.is_bf16_supported() else False
USE_FP16 = False

# 为了保证 OT 计算速度，评估时最多采样多少样本计算距离
# 注意：精确 OT (emd2) 是 O(N^3) 复杂度，N > 5000 会极慢。建议保持在 2000-5000。
METRIC_SAMPLE_SIZE = 5000 

# ==========================================
# 1. 实验变量组 (Sensitivity Grid)
# ==========================================
TEMPERATURES = [0.2, 0.6, 1.0, 1.4, 1.8]
SENSITIVITY_CONFIGS = [
    {"name": f"Temp_{t}", "temp": t, "top_k": 50, "top_p": 0.95} 
    for t in TEMPERATURES
]

# ==========================================
# 2. 核心指标计算器 (Metrics Evaluator)
# ==========================================
class MetricsEvaluator:
    def __init__(self, device=DEVICE):
        self.device = device
        
        # A. 加载 Embedding 模型 (Sentence Transformer)
        print(f">>> [Metrics] Loading Embedding Model: {EMBEDDING_MODEL_PATH}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)
        self.embed_model.eval()

        # B. 加载 Entropy 模型 (GPT-2 Large)
        print(f">>> [Metrics] Loading Entropy Model: {ENTROPY_MODEL_PATH}...")
        self.entropy_tokenizer = GPT2Tokenizer.from_pretrained(ENTROPY_MODEL_PATH)
        if self.entropy_tokenizer.pad_token is None:
            self.entropy_tokenizer.pad_token = self.entropy_tokenizer.eos_token
        self.entropy_model = GPT2LMHeadModel.from_pretrained(ENTROPY_MODEL_PATH).to(device)
        self.entropy_model.eval()
        
        self.real_stats = None 

    def get_embeddings(self, texts):
        """使用 SentenceTransformer 获取高质量语义向量"""
        # SentenceTransformer 会自动处理 batching 和 padding
        embeddings = self.embed_model.encode(
            texts, 
            batch_size=128, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True # 归一化通常对 OT 和 Cosine 更好
        )
        return embeddings

    def get_oracle_entropy(self, texts, batch_size=32):
        """使用 GPT-2 Large 计算 Oracle Entropy (NLL)"""
        entropies = []
        # 过滤空文本
        valid_texts = [t for t in texts if len(t.strip()) > 0]
        
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="Calc Entropy", leave=False):
            batch = valid_texts[i : i + batch_size]
            inputs = self.entropy_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.entropy_model(inputs.input_ids)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Token-level entropy
                token_ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                # Sequence mean entropy
                seq_ent = (token_ent * inputs.attention_mask).sum(1) / torch.clamp(inputs.attention_mask.sum(1), min=1)
                entropies.extend(seq_ent.cpu().tolist())
                
        return np.mean(entropies)

    def compute_empirical_entropy(self, texts, n=2):
        """计算 N-gram 经验熵"""
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
        """
        [修改点 4] 基于 POT 库计算精确的 Optimal Transport Distance
        """
        # 确保数据量不要太大，否则 OOM。建议 max 5000。
        source_emb = np.array(source_emb, dtype=np.float64)
        target_emb = np.array(target_emb, dtype=np.float64)
        
        n = source_emb.shape[0]
        m = target_emb.shape[0]

        # 均匀分布权重
        a = np.ones((n,)) / n
        b = np.ones((m,)) / m

        # 计算代价矩阵 (Cost Matrix)
        # 'euclidean' -> W1, 'sqeuclidean' -> W2^2
        M = ot.dist(source_emb, target_emb, metric=metric)
        # M /= M.max() # 归一化矩阵以保证数值稳定性 (可选，视具体数值范围而定)

        # 计算精确 OT
        # numItermax 设大一点防止不收敛
        ot_cost = ot.emd2(a, b, M, numItermax=1000000)
        return ot_cost

    def compute_fbd(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        计算 Fréchet BERT Distance (FBD) - 数值稳定优化版
        Args:
            mu1, mu2: 均值向量
            sigma1, sigma2: 协方差矩阵
            eps: 防止 sqrtm 失败的微小扰动项
        """
        # 1. 强制使用 float64 以保证精度
        mu1 = np.atleast_1d(mu1).astype(np.float64)
        mu2 = np.atleast_1d(mu2).astype(np.float64)
        sigma1 = np.atleast_2d(sigma1).astype(np.float64)
        sigma2 = np.atleast_2d(sigma2).astype(np.float64)

        # 2. 计算均值部分的距离
        diff = mu1 - mu2
        mean_term = diff.dot(diff)

        # 3. 计算协方差部分的距离: Tr(∑1 + ∑2 - 2(∑1∑2)^(1/2))
        # product = Sigma1 * Sigma2
        cov_dot = sigma1.dot(sigma2)
        
        # 计算矩阵平方根
        covmean, _ = linalg.sqrtm(cov_dot, disp=False)

        # 4. [关键优化] 处理奇异矩阵导致的 sqrtm 失败
        # 如果 cov_dot 是奇异的，sqrtm 可能会返回无穷大或 NaN
        if not np.isfinite(covmean).all():
            print(f"    | [Warning] FBD: Singular product encountered, adding epsilon {eps} to diagonal.")
            offset = np.eye(sigma1.shape[0]) * eps
            # 重新计算 (∑1 + epsI)(∑2 + epsI)
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 5. 处理数值误差产生的虚部
        if np.iscomplexobj(covmean):
            # 检查虚部是否真的很小
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                max_imag = np.max(np.abs(covmean.imag))
                print(f"    | [Warning] FBD: Large imaginary component {max_imag}. Metric might be unreliable.")
            covmean = covmean.real

        # trace term
        tr_covmean = np.trace(covmean)
        
        return float(mean_term + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def run_evaluation(self, real_texts, syn_texts):
        # 采样用于 metric 计算的子集 (防止 OT 计算爆炸)
        if len(real_texts) > METRIC_SAMPLE_SIZE:
            real_sample = random.sample(real_texts, METRIC_SAMPLE_SIZE)
        else:
            real_sample = real_texts
            
        if len(syn_texts) > METRIC_SAMPLE_SIZE:
            syn_sample = random.sample(syn_texts, METRIC_SAMPLE_SIZE)
        else:
            syn_sample = syn_texts

        # 1. Real Stats (Cache)
        if self.real_stats is None:
            print("    | [Metrics] Computing Reference (Real) Stats...")
            emb = self.get_embeddings(real_sample)
            ent = self.get_oracle_entropy(real_sample)
            uni_ent = self.compute_empirical_entropy(real_sample, n=1)
            bi_ent = self.compute_empirical_entropy(real_sample, n=2)
            
            self.real_stats = {
                "emb": emb, 
                "mu": np.mean(emb, axis=0), 
                "cov": np.cov(emb, rowvar=False), 
                "ent": ent,
                "uni": uni_ent, 
                "bi": bi_ent
            }
        
        # 2. Syn Stats
        print("    | [Metrics] Computing Synthetic Stats...")
        syn_emb = self.get_embeddings(syn_sample)
        syn_ent = self.get_oracle_entropy(syn_sample)
        syn_uni = self.compute_empirical_entropy(syn_texts, n=1) # 经验熵可以用全量算，很快
        syn_bi = self.compute_empirical_entropy(syn_texts, n=2)
        
        syn_mu = np.mean(syn_emb, axis=0)
        syn_cov = np.cov(syn_emb, rowvar=False)
        
        # 3. Distances
        print("    | [Metrics] Calculating Distances (FBD & Exact OT)...")
        fbd = self.compute_fbd(self.real_stats['mu'], self.real_stats['cov'], syn_mu, syn_cov)
        ot_dist = self.calculate_exact_ot_distance(self.real_stats['emb'], syn_emb, metric='euclidean')
        
        return {
            "oracle_entropy": float(syn_ent),
            "empirical_unigram_entropy": float(syn_uni),
            "empirical_bigram_entropy": float(syn_bi),
            "fbd_score": fbd,
            "exact_ot_distance": ot_dist,
            # Refs
            "ref_bigram_entropy": self.real_stats['bi']
        }

# ==========================================
# 3. 功能函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_ppl(model, tokenizer, dataset, device=DEVICE):
    """
    使用滑动窗口计算高精度 PPL。
    """
    model.eval()
    print(f"    -> Evaluating PPL on {len(dataset)} validation samples...")
    
    # 拼接文本
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    # 进度条
    pbar = tqdm(range(0, seq_len, stride), desc="Evaluating PPL", leave=False)
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Mask 掉 Context 部分的 Loss

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # 还原 sum loss
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()

def group_texts(examples):
    """Data Packing: 将短文本拼接成长文本以提高训练效率"""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= MAX_LENGTH:
        total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
    result = {
        k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train_one_round(config_name, train_dataset, tokenizer):
    """
    训练函数：只保存训练结束后的最终模型 (Last Checkpoint)。
    """
    output_dir = os.path.join(RESULT_ROOT, config_name, "model")

    print(f"    | [Train] Initializing FRESH Base Model from {BASE_MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id

    column_names = train_dataset.column_names
    
    def tokenize_function(examples):
        return tokenizer([t + tokenizer.eos_token for t in examples["text"]])
    
    tokenized = train_dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=column_names)
    packed_dataset = tokenized.map(group_texts, batched=True, num_proc=8)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        bf16=USE_BF16,
        fp16=USE_FP16,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        save_strategy="no", 
        eval_strategy="no", 
        report_to="none", 
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=packed_dataset,
    )
    
    trainer.train()

    return model

def generate_synthetic_data(model, tokenizer, prompt_texts_list, num_samples, config):
    """
    镜像生成函数：生成下一代训练数据。
    """
    model.eval()
    
    # 设置 Left Padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # [优化] 同步 Model Config，防止警告
    model.config.pad_token_id = tokenizer.pad_token_id

    # 1. 获取所有原始文本
    # raw_texts = prompt_dataset["text"]
    
    # 2. 清洗 Prompt 源
    clean_prompts = [t for t in prompt_texts_list if len(t.strip()) > 0]
    
    # 3. 填充 Prompt 池
    while len(clean_prompts) < num_samples:
        clean_prompts.extend(clean_prompts)
        
    # 4. 截取
    target_prompts = clean_prompts[:num_samples]
    
    print(f"    -> [Gen] Generating {len(target_prompts)} samples...")

    synthetic_texts = []
    
    # 5. Batch 生成
    for i in tqdm(range(0, len(target_prompts), GEN_BATCH_SIZE), desc="Synthesizing"):
        batch_prompts = target_prompts[i : i + GEN_BATCH_SIZE]
        
        # 估算长度
        batch_lens = [len(t) for t in tokenizer(batch_prompts, add_special_tokens=False)["input_ids"]]
        # 动态长度：保持与原文本长度分布一致
        current_max_target = min(max(batch_lens) + 128, MAX_LENGTH)

        # 截断 Prompt
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=PROMPT_LEN_BASE
        ).to(DEVICE)
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float16):
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=current_max_target, 
                        do_sample=True,
                        temperature=config["temp"],
                        top_k=config["top_k"],
                        top_p=config["top_p"],
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True 
                    )
            
            gen_texts_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            synthetic_texts.extend(gen_texts_batch)
            del inputs, outputs

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    | WARNING: OOM in batch {i}. Skipping & Cleaning cache.")
                torch.cuda.empty_cache()
            continue

    # 6. 后处理
    tokenizer.padding_side = "right"
    
    final_data = [t for t in synthetic_texts if len(t.strip()) > 0]
    
    loss_count = num_samples - len(final_data)
    if loss_count > 0:
        print(f"    -> [Info] Lost {loss_count} samples due to empty generation ({(loss_count/num_samples):.2%}).")
    
    return Dataset.from_dict({"text": final_data})

# ==========================================
# 4. 主流程
# ==========================================
def main():
    set_seed(42)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    
    # 1. 准备分词器与数据
    print(">>> Loading Data & Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        dataset = load_from_disk(DATA_PATH)
    except:
        dataset = load_dataset(DATA_PATH)
        
    real_texts_list = dataset["train"]["text"]
    valid_dataset = dataset["validation"]

    real_clean_samples = [t for t in real_texts_list if len(t.strip()) > 0]
    SAMPLES_TO_GENERATE = len(real_clean_samples)
    print(f">>> [Config] Target Valid Samples: {SAMPLES_TO_GENERATE}")
    
    # 2. [修改点 1] 加载指定的生成器 (Teacher Model)
    print(f">>> Loading Generator from: {GENERATOR_PATH}")
    generator = GPT2LMHeadModel.from_pretrained(GENERATOR_PATH).to(DEVICE)
    generator.config.pad_token_id = tokenizer.pad_token_id
    
    # 3. 初始化评估器 (包含 MPNet 和 GPT2-Large)
    evaluator = MetricsEvaluator(DEVICE)
    
    results = []

    # 4. 实验循环
    for exp_cfg in SENSITIVITY_CONFIGS:
        print(f"\n================ Running: {exp_cfg['name']} ================")
        
        # A. 全量生成
        print("  -> Generating Full Synthetic Dataset...")
        syn_dataset = generate_synthetic_data(generator, tokenizer, real_texts_list, SAMPLES_TO_GENERATE, exp_cfg)
        
        # B. 评估数据指标 (FBD, Exact OT, Entropy)
        # 注意：这里内部会自动采样用于 OT 计算，避免 O(N^3) 卡死
        print("  -> Measuring Data Distribution...")
        data_metrics = evaluator.run_evaluation(real_texts_list, syn_dataset["text"])
        print(f"  -> Metrics: FBD={data_metrics['fbd_score']:.2f}, OT={data_metrics['exact_ot_distance']:.2f}, Bi-Ent={data_metrics['empirical_bigram_entropy']:.3f}")
        
        # C. 训练学生模型
        print("  -> Training Student Model on Synthetic Data...")
        student_model = train_one_round(exp_cfg["name"], syn_dataset, tokenizer)
        
        # D. 评估 PPL
        print("  -> Evaluating Student Performance...")
        ppl = compute_ppl(student_model, tokenizer, valid_dataset)
        print(f"  -> Validation PPL: {ppl:.2f}")
        
        # E. 保存
        record = {**exp_cfg, **data_metrics, "validation_ppl": ppl}
        results.append(record)
        
        with open(os.path.join(RESULT_ROOT, "final_results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        # 清理
        del student_model, syn_dataset
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()