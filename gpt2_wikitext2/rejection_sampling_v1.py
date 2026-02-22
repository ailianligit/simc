import os
import sys
import torch
import random
import json
import numpy as np
import gc
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

# 防止 Tokenizer 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 0. 实验配置中心
# ==========================================

# >>> 实验模式 <<<
# "BASELINE"       : 随机采样 (Temp=1.0, 无拒绝)
# "REJECT": 自适应拒绝采样 (Temp=1.2, 基于真实分布 Sigma 截断)
CURRENT_MODE = "REJECT"

# 自适应阈值参数 (仅在 REJECT 下生效)
# 接受区间 = [Mean - k*Sigma, Mean + k*Sigma]
SIGMA_K = 1.0 

# 路径设置
EXTERNAL_GEN0_PATH = "model_real_trained"
DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"
MODEL_PATH = "/home/ubuntu/data/model/gpt2_model"
OUTPUT_ROOT = f"rejection_sampling_results/{CURRENT_MODE.lower()}"

# 硬件定义
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 实验规模 ---
NUM_GENERATIONS = 10
EPOCHS = 5                   
TRAIN_BATCH_SIZE = 32        
GEN_BATCH_SIZE = 128         # [建议] 根据显存调整，256 可能导致 OOM
GRADIENT_ACCUMULATION = 1

# --- 优化器参数 ---
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"

# --- 序列与生成 ---
MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64
USE_BF16 = True if torch.cuda.is_bf16_supported() else False

# ==========================================
# 1. 核心模块: 熵判别器 (The Judge)
# ==========================================

class EntropyJudge:
    def __init__(self, model_path, device=DEVICE):
        """加载 Oracle 模型作为裁判"""
        print(f">>> [Judge] Loading Oracle from {model_path}...")
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.loss_fct = CrossEntropyLoss(reduction='none')
        
        self.ref_mu = None
        self.ref_sigma = None
        self.bounds = (None, None)

    def compute_entropy(self, texts, batch_size=64):
        """
        计算 Oracle Entropy (NLL)
        [修复] 假设输入 texts 已经是非空且有效的，不再内部过滤，保证输出长度与输入一致
        """
        if not texts: return []

        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
                logits = outputs.logits
                
                # Shift for Causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                shift_mask = inputs.attention_mask[..., 1:].contiguous()

                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())
                
                # Sequence Entropy
                seq_nll = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
                scores.extend(seq_nll.cpu().tolist())
                
        return scores

    def calibrate(self, dataset):
        """基于真实数据校准阈值"""
        print(">>> [Judge] Calibrating on Real Data...")
        # 采样最多 10000 条用于估算
        samples = [t for t in dataset["text"][:10000] if len(t.strip()) > 20]
        scores = self.compute_entropy(samples)
        
        self.ref_mu = np.mean(scores)
        self.ref_sigma = np.std(scores)
        
        lower = self.ref_mu - (SIGMA_K * self.ref_sigma)
        upper = self.ref_mu + (SIGMA_K * self.ref_sigma)
        self.bounds = (lower, upper)
        
        print(f"    -> Real Entropy: mu={self.ref_mu:.4f}, sigma={self.ref_sigma:.4f}")
        print(f"    -> Dynamic Bounds: [{lower:.4f}, {upper:.4f}] (k={SIGMA_K})")
        return self.ref_mu

    def filter_batch(self, texts):
        """执行拒绝采样"""
        if CURRENT_MODE == "BASELINE":
            return texts 

        # [修复] 第一步：先清洗无效文本，确保送入 compute_entropy 的都是合法的
        # 这样 compute_entropy 返回的 scores 就能和 valid_candidates 一一对应
        valid_candidates = [t for t in texts if len(t.strip()) >= 20]
        
        if not valid_candidates:
            return []

        scores = self.compute_entropy(valid_candidates)
        accepted = []
        low, high = self.bounds
        
        # [修复] 此时 zip 是安全的，不会错位
        for text, score in zip(valid_candidates, scores):
            if low <= score <= high:
                accepted.append(text)
                
        return accepted

# ==========================================
# 2. 生成器 (Producer)
# ==========================================

def generate_with_strategy(generator, tokenizer, prompt_dataset, target_count, judge):
    generator.eval()
    tokenizer.padding_side = "left"
    generator.config.pad_token_id = tokenizer.pad_token_id
    
    if CURRENT_MODE == "BASELINE":
        temp = 1.0 
        oversample = 1.0
    else:
        temp = 1.2 
        oversample = 2.0 
    
    raw_texts = prompt_dataset["text"]
    # 基础 Prompt 池
    prompts = [t for t in raw_texts if len(t.strip()) > 0]
    
    synthetic_data = []
    synthetic_scores = [] 
    
    pbar = tqdm(total=target_count, desc=f"Gen ({CURRENT_MODE}) T={temp}")
    prompt_idx = 0
    total_prompts = len(prompts)

    while len(synthetic_data) < target_count:
        # 计算 Batch
        needed = target_count - len(synthetic_data)
        batch_base = min(GEN_BATCH_SIZE, needed)
        actual_gen_size = min(int(batch_base * oversample), GEN_BATCH_SIZE * 2)
        
        # [优化] 使用取模索引，避免 extend 导致的内存膨胀
        batch_prompts = [prompts[(prompt_idx + i) % total_prompts] for i in range(actual_gen_size)]
        prompt_idx = (prompt_idx + actual_gen_size) % total_prompts
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN_BASE).to(DEVICE)
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float16):
                    outputs = generator.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        # [优化] 统一使用全局配置
                        max_length=min(MAX_LENGTH, inputs.input_ids.shape[1] + 256), 
                        do_sample=True,
                        temperature=temp, 
                        top_k=50, 
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id
                    )
            
            raw_batch_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 判别与筛选
            valid_texts = judge.filter_batch(raw_batch_texts)
            
            # 收集数据
            can_add = min(len(valid_texts), target_count - len(synthetic_data))
            added_texts = valid_texts[:can_add]
            synthetic_data.extend(added_texts)
            
            # 记录熵用于统计
            if added_texts:
                # 注意：这里会再次计算一次熵，有轻微性能损耗，但在接受范围内
                batch_scores = judge.compute_entropy(added_texts)
                synthetic_scores.extend(batch_scores)
            
            # 动态调整生成倍率
            acc_rate = len(valid_texts) / len(raw_batch_texts) if raw_batch_texts else 0
            if acc_rate < 0.15 and CURRENT_MODE != "BASELINE":
                oversample = min(oversample * 1.5, 6.0)
                
            pbar.update(can_add)
            pbar.set_postfix({"Acc": f"{acc_rate:.2f}"})
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                oversample = max(1.0, oversample * 0.8) 
            continue

    pbar.close()
    avg_syn_entropy = np.mean(synthetic_scores) if synthetic_scores else 0.0
    return Dataset.from_dict({"text": synthetic_data}), avg_syn_entropy

# ==========================================
# 3. 辅助函数
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_standard_ppl_with_sliding_window(model, tokenizer, dataset):
    """高精度滑动窗口 PPL"""
    model.eval()
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_len = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    limit = seq_len
    
    pbar = tqdm(range(0, limit, stride), desc="Evaluating PPL", leave=False)
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_len, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

    if prev_end_loc == 0: return float('inf')
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()

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

def train_model(model, train_dataset, tokenizer, output_dir):
    def tokenize(examples):
        return tokenizer([t + tokenizer.eos_token for t in examples["text"]])

    tokenized = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    packed = tokenized.map(group_texts, batched=True)
    
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        bf16=USE_BF16,
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
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
    trainer.save_model(output_dir)
    return model

# ==========================================
# 4. 主程序
# ==========================================

def main():
    set_seed(42)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    metrics_file = os.path.join(OUTPUT_ROOT, "metrics.json")
    
    print(f">>> [Init] Mode: {CURRENT_MODE} | Sigma_K: {SIGMA_K}")
    
    # 1. 准备数据
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    try: dataset = load_from_disk(DATA_PATH)
    except: dataset = load_dataset(DATA_PATH)
    
    real_train = dataset['train']
    real_valid = dataset['validation']
    TARGET_SIZE = len([t for t in real_train['text'] if len(t.strip()) > 0])
    
    # 逻辑：如果外部路径存在，直接用；否则才会在当前目录重新训练
    if os.path.exists(EXTERNAL_GEN0_PATH):
        print(f">>> [Step 0] Directly loading existing Gen 0 from: {EXTERNAL_GEN0_PATH}")
        gen0_path = EXTERNAL_GEN0_PATH
    else:
        # 如果找不到外部模型，回退到在当前目录下重新训练
        print(f">>> [Step 0] External path not found. Training Gen 0 Oracle locally...")
        gen0_path = os.path.join(OUTPUT_ROOT, "gen_0_oracle")
        
        if not os.path.exists(gen0_path):
            base = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
            train_model(base, real_train, tokenizer, gen0_path)
            del base; gc.collect(); torch.cuda.empty_cache()
    
    # 3. 初始化 Judge 并自适应校准
    judge = EntropyJudge(gen0_path, DEVICE)
    ref_mu = judge.calibrate(real_train)
    
    # 记录 Gen 0 PPL
    ref_ppl = compute_standard_ppl_with_sliding_window(judge.model, tokenizer, real_valid)
    metrics = [{"round": 0, "ppl": ref_ppl, "avg_entropy": ref_mu, "entropy_gap": 0.0}]
    
    current_generator = GPT2LMHeadModel.from_pretrained(gen0_path).to(DEVICE)
    
    # 4. 迭代循环
    for i in range(1, NUM_GENERATIONS + 1):
        print(f"\n=== Round {i} ===")
        
        # A. 生成
        syn_data, syn_entropy = generate_with_strategy(current_generator, tokenizer, real_train, TARGET_SIZE, judge)
        print(f"    -> Syn Entropy: {syn_entropy:.4f} (Ref: {ref_mu:.4f})")
        
        # B. 训练 (Fresh Base)
        fresh_base = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
        output_dir = os.path.join(OUTPUT_ROOT, f"gen_{i}_model")
        new_model = train_model(fresh_base, syn_data, tokenizer, output_dir)
        
        # C. 评估
        ppl = compute_standard_ppl_with_sliding_window(new_model, tokenizer, real_valid)
        print(f"    -> PPL: {ppl:.2f}")
        
        # D. 记录
        metrics.append({
            "round": i, 
            "ppl": ppl, 
            "avg_entropy": syn_entropy,
            "entropy_gap": abs(syn_entropy - ref_mu)
        })
        with open(metrics_file, "w") as f: json.dump(metrics, f, indent=4)
        
        current_generator = new_model
        del fresh_base; gc.collect(); torch.cuda.empty_cache()

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()