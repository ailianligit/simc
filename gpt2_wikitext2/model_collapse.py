import os
import sys
import torch
import random
import json
import numpy as np
import shutil
import gc  # 新增：用于强制垃圾回收
from tqdm import tqdm
from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

# ==========================================
# 0. 全局路径与硬件配置 (Global Setup)
# ==========================================

# 路径设置 (请根据你的实际环境修改这些路径)
DATA_PATH = "/home/ubuntu/data/dataset/wikitext_dataset"  # 原始真实数据路径
MODEL_PATH = "/home/ubuntu/data/model/gpt2_model"        # 原始 GPT-2 权重路径
EXPERIMENT_ROOT = "./model_collapse_results_v2"             # 实验结果输出目录

# 硬件定义 (必须在函数定义之前)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 实验配置参数 (Configuration)
# ==========================================

# --- 实验规模 ---
NUM_GENERATIONS = 10        # 观测代数
EPOCHS = 5                  # 每轮固定训练 Epoch

# --- 硬件与显存 ---
TRAIN_BATCH_SIZE = 32       # 针对 48GB 显存优化
GEN_BATCH_SIZE = 256        # 生成时的 Batch Size (越大越快)
GRADIENT_ACCUMULATION = 1

# --- 优化器参数 ---
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.03

# --- 序列设置 ---
MAX_LENGTH = 1024
PROMPT_LEN_BASE = 64

# --- 混合精度 ---
USE_BF16 = True if torch.cuda.is_bf16_supported() else False
USE_FP16 = False

# ==========================================
# 2. 工具函数 (Helper Functions)
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_standard_ppl_with_sliding_window(model, tokenizer, dataset, device=DEVICE):
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

# ==========================================
# 3. 核心功能：生成与训练 (Core Logic)
# ==========================================

def generate_synthetic_data(model, tokenizer, prompt_dataset, num_samples, 
                            batch_size=GEN_BATCH_SIZE, device=DEVICE):
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
    raw_texts = prompt_dataset["text"]
    
    # 2. 清洗 Prompt 源
    clean_prompts = [t for t in raw_texts if len(t.strip()) > 0]
    
    # 3. 填充 Prompt 池
    while len(clean_prompts) < num_samples:
        clean_prompts.extend(clean_prompts)
        
    # 4. 截取
    target_prompts = clean_prompts[:num_samples]
    
    print(f"    -> [Gen] Generating {len(target_prompts)} samples...")

    synthetic_texts = []
    
    # 5. Batch 生成
    for i in tqdm(range(0, len(target_prompts), batch_size), desc="Synthesizing"):
        batch_prompts = target_prompts[i : i + batch_size]
        
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
        ).to(device)
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float16):
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=current_max_target, 
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
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

def train_model(model, train_dataset, tokenizer, output_dir):
    """
    训练函数：只保存训练结束后的最终模型 (Last Checkpoint)。
    """
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
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=packed_dataset,
    )
    
    trainer.train()
    
    print(f"    -> Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model

# ==========================================
# 4. 主程序 (Main Execution)
# ==========================================

def main():
    set_seed(42)
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    metrics_path = os.path.join(EXPERIMENT_ROOT, "metrics.json")
    
    print(">>> [Init] Loading Data and Tokenizer...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载真实数据
    try:
        print(f"Loading from disk: {DATA_PATH}")
        dataset = load_from_disk(DATA_PATH)
    except Exception as e:
        print(f"Load from disk failed ({e}), trying generic load...")
        # Fallback 逻辑：如果是本地文本文件，可能需要指定 "text"
        try:
            dataset = load_dataset(DATA_PATH)
        except:
            dataset = load_dataset("text", data_dir=DATA_PATH)
    
    train_data_real = dataset['train']
    valid_data_real = dataset['validation'] 
    
    real_clean_samples = [t for t in train_data_real['text'] if len(t.strip()) > 0]
    SAMPLES_TO_GENERATE = len(real_clean_samples)
    print(f">>> [Config] Target Valid Samples: {SAMPLES_TO_GENERATE}")

    # 初始化记录
    if os.path.exists(metrics_path):
        print(">>> [Resume] Found existing metrics file. Loading history...")
        with open(metrics_path, "r") as f:
            metrics_history = json.load(f)
    else:
        metrics_history = []

    # ================= Generation 0: Baseline =================
    print("\n================ Generation 0 (Real Data) ================")
    gen0_model_dir = os.path.join(EXPERIMENT_ROOT, "gen_0_model")
    
    if os.path.exists(gen0_model_dir) and any(m['generation'] == 0 for m in metrics_history):
        print(">>> Gen 0 completed. Loading checkpoint...")
        model_gen0 = GPT2LMHeadModel.from_pretrained(gen0_model_dir).to(DEVICE)
    else:
        print(">>> Training Gen 0 Baseline Model...")
        base_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
        model_gen0 = train_model(base_model, train_data_real, tokenizer, gen0_model_dir)
        
        ppl = compute_standard_ppl_with_sliding_window(model_gen0, tokenizer, valid_data_real)
        
        record = {"generation": 0, "ppl": ppl, "data_source": "Real Data"}
        metrics_history.append(record)
        with open(metrics_path, "w") as f: json.dump(metrics_history, f, indent=4)
        print(f"    -> Gen 0 PPL: {ppl:.2f}")

    current_generator = model_gen0

    # ================= Recursive Generations =================
    for i in range(1, NUM_GENERATIONS + 1):
        print(f"\n================ Generation {i} ================")
        
        gen_model_dir = os.path.join(EXPERIMENT_ROOT, f"gen_{i}_model")
        gen_data_dir = os.path.join(EXPERIMENT_ROOT, f"gen_{i}_data")
        
        # Check 1: 完成检测
        if os.path.exists(gen_model_dir) and any(m['generation'] == i for m in metrics_history):
            print(f">>> Gen {i} already finished. Skipping.")
            current_generator = GPT2LMHeadModel.from_pretrained(gen_model_dir).to(DEVICE)
            continue
            
        # Check 2: 数据检测
        synthetic_dataset = None
        if os.path.exists(gen_data_dir):
            try:
                print(f">>> Found Gen {i} synthetic data. Loading...")
                synthetic_dataset = load_from_disk(gen_data_dir)
            except Exception as e:
                print(f"    | Warning: Data corrupted ({e}). Will regenerate.")
                synthetic_dataset = None
        
        # Action 1: 生成
        if synthetic_dataset is None:
            print(f">>> Gen {i}: Synthesizing Data...")
            synthetic_dataset = generate_synthetic_data(
                current_generator, tokenizer, 
                prompt_dataset=train_data_real, 
                num_samples=SAMPLES_TO_GENERATE 
            )
            synthetic_dataset.save_to_disk(gen_data_dir)
        
        # Action 2: 训练
        print(f">>> Gen {i}: Training...")
        fresh_base_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
        
        model_gen_i = train_model(fresh_base_model, synthetic_dataset, tokenizer, gen_model_dir)
        
        # Action 3: 评估
        print(f">>> Gen {i}: Evaluating...")
        ppl_score = compute_standard_ppl_with_sliding_window(model_gen_i, tokenizer, valid_data_real)
        
        # Action 4: 保存
        record = {"generation": i, "ppl": ppl_score, "data_source": "Synthetic"}
        
        metrics_history = [m for m in metrics_history if m['generation'] != i]
        metrics_history.append(record)
        
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
            
        print(f"    -> Gen {i} Finished. PPL: {ppl_score:.2f}")
        
        current_generator = model_gen_i
        
        # [关键] 强制清理显存
        del fresh_base_model
        gc.collect() 
        torch.cuda.empty_cache()

    print("\n>>> All generations completed.")
    print(json.dumps(metrics_history, indent=4))

if __name__ == "__main__":
    main()