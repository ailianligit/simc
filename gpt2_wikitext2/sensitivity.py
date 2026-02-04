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
ROUND = 7
GENERATOR_PATH = f"/home/ubuntu/data/simc/gpt2_wikitext2/model_collapse_results_v2/gen_{ROUND}_model"

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
TRAIN_BATCH_SIZE = 8
GEN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.03
USE_BF16 = True if torch.cuda.is_bf16_supported() else False
USE_FP16 = False

# 为了保证 OT 计算速度，评估时最多采样多少样本计算距离
# 注意：精确 OT (emd2) 是 O(N^3) 复杂度，N > 5000 会极慢。建议保持在 2000-5000。
OT_SAMPLE_SIZE = 5000
FBD_SAMPLE_SIZE = 10000

# ==========================================
# 1. 实验变量组 (Sensitivity Grid)
# ==========================================
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
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
        基于 POT 库计算精确的 Optimal Transport Distance
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
        计算 Fréchet BERT Distance (FBD) - 高精度数值稳定版
        """
        # 1. 强制使用双精度浮点数 (float64) 以减少累积误差
        mu1 = np.atleast_1d(mu1).astype(np.float64)
        mu2 = np.atleast_1d(mu2).astype(np.float64)
        sigma1 = np.atleast_2d(sigma1).astype(np.float64)
        sigma2 = np.atleast_2d(sigma2).astype(np.float64)

        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

        # 2. 计算均值部分的距离 (L2-norm squared)
        diff = mu1 - mu2
        mean_term = diff.dot(diff)

        # 3. 计算协方差部分的距离: Tr(∑1 + ∑2 - 2(∑1∑2)^(1/2))
        # product = Sigma1 * Sigma2
        cov_dot = sigma1.dot(sigma2)
        
        # 计算矩阵平方根
        covmean, _ = linalg.sqrtm(cov_dot, disp=False)

        # [修复] 检查计算失败 (NaN/Inf) 或 奇异矩阵
        if not np.isfinite(covmean).all():
            print(f"    | [Warning] FBD: sqrtm calculation failed. Adding epsilon {eps} to diagonal.")
            offset = np.eye(sigma1.shape[0]) * eps
            # 重新计算 (∑1 + epsI)(∑2 + epsI)
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # [修复] 处理数值误差产生的虚部
        if np.iscomplexobj(covmean):
            # 检查虚部是否真的很小
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                max_imag = np.max(np.abs(covmean.imag))
                # 仅打印警告，不中断
                print(f"    | [Warning] FBD: Imaginary component {max_imag}. Metric might be unreliable.")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        
        # 最终公式
        return float(mean_term + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def run_evaluation(self, real_texts, syn_texts):
        # ==========================================
        # 1. 动态设置采样量
        # ==========================================
        # FBD & Oracle Entropy: 
        # 建议至少 10,000 条。如果为了绝对严谨，可以将 N_FBD 设置为 len(syn_texts)
        N_FBD = min(len(real_texts), len(syn_texts), FBD_SAMPLE_SIZE) 
        
        # OT: 精确解计算极慢，必须控制在 2500 以内
        N_OT = min(len(real_texts), len(syn_texts), OT_SAMPLE_SIZE)

        # ==========================================
        # 2. 数据采样
        # ==========================================
        # A. FBD & Entropy 用的大样本
        if len(real_texts) > N_FBD:
            real_sample_fbd = random.sample(real_texts, N_FBD)
        else:
            real_sample_fbd = real_texts
            
        if len(syn_texts) > N_FBD:
            syn_sample_fbd = random.sample(syn_texts, N_FBD)
        else:
            syn_sample_fbd = syn_texts

        # ==========================================
        # 3. 计算 Real Stats (缓存)
        # ==========================================
        if self.real_stats is None or self.real_stats["n_samples"] < N_FBD:
            print(f"    | [Metrics] Computing Real Stats (N={N_FBD})...")
            
            real_emb_fbd = self.get_embeddings(real_sample_fbd)
            
            # [修改点] Real Oracle Entropy: 改用大样本 (N_FBD) 计算
            real_oracle_ent = self.get_oracle_entropy(real_sample_fbd) 
            
            # Real N-gram Entropy: 用采样的大样本 (N=10000 足够稳定)
            real_uni_ent = self.compute_empirical_entropy(real_sample_fbd, n=1)
            real_bi_ent = self.compute_empirical_entropy(real_sample_fbd, n=2)
            
            real_mu = np.mean(real_emb_fbd, axis=0)
            real_cov = np.cov(real_emb_fbd, rowvar=False)

            self.real_stats = {
                "emb": real_emb_fbd,
                "mu": real_mu,
                "cov": real_cov,
                "ent": real_oracle_ent,
                "uni": real_uni_ent,
                "bi": real_bi_ent,
                "n_samples": len(real_sample_fbd)
            }
        else:
            print("    | [Metrics] Using cached Real Stats.")

        # ==========================================
        # 4. 计算 Synthetic Stats
        # ==========================================
        print(f"    | [Metrics] Computing Synthetic Stats (N={N_FBD})...")
        
        syn_emb_fbd = self.get_embeddings(syn_sample_fbd)
        syn_mu = np.mean(syn_emb_fbd, axis=0)
        syn_cov = np.cov(syn_emb_fbd, rowvar=False)
        
        # [修改点] Synthetic Oracle Entropy: 改用大样本 (N_FBD)
        # 既然 N_FBD 已经是 10,000 条级别，这在统计上和全量没有区别，但比全量快
        syn_oracle_ent = self.get_oracle_entropy(syn_sample_fbd)
        
        # [保持原样] Synthetic N-gram: 只要计算极快，永远建议用全量 (syn_texts)
        syn_uni_ent = self.compute_empirical_entropy(syn_texts, n=1) 
        syn_bi_ent = self.compute_empirical_entropy(syn_texts, n=2)

        # ==========================================
        # 5. 计算距离
        # ==========================================
        print(f"    | [Metrics] Calculating FBD (using {N_FBD} samples)...")
        fbd_score = self.compute_fbd(
            self.real_stats['mu'], self.real_stats['cov'], 
            syn_mu, syn_cov
        )
        
        print(f"    | [Metrics] Calculating Exact OT (using subset {N_OT})...")
        # OT 依然必须切片，否则跑不完
        real_emb_ot = self.real_stats['emb'][:N_OT]
        syn_emb_ot = syn_emb_fbd[:N_OT]
        
        ot_dist = self.calculate_exact_ot_distance(
            real_emb_ot, syn_emb_ot, metric='euclidean'
        )
        
        return {
            "oracle_entropy": float(syn_oracle_ent),
            "empirical_unigram_entropy": float(syn_uni_ent),
            "empirical_bigram_entropy": float(syn_bi_ent),
            "fbd_score": float(fbd_score),
            "exact_ot_distance": float(ot_dist),
            "ref_oracle_entropy": float(self.real_stats['ent']),
            "ref_bigram_entropy": float(self.real_stats['bi'])
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
    # output_dir = os.path.join(RESULT_ROOT, config_name, "model")

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
        # output_dir=output_dir,
        # overwrite_output_dir=True,
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