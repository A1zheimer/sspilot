#!/usr/bin/env python3
import os; os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
"""
SFT v5 训练脚本 — 基于 v4 + 补充 logic 类数据
自动合并原始训练数据 + 新生成的 logic 数据
"""
import os, json, glob, time, torch, gc
from datetime import datetime

# ============ 配置 ============
MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4")
ORIGINAL_DATA = os.path.expanduser("~/sspilot/datasets/audit_sft_data.jsonl")
LOGIC_DATA_DIR = os.path.expanduser("~/sspilot/datasets/")
OUTPUT_DIR = os.path.expanduser("~/sspilot/checkpoints/audit_sft_v5")
BATTLE_TRACE = os.path.expanduser("~/sspilot/traces/battle_round001_20260401_113240.jsonl")

# 训练超参 (沿用 v4 最佳配置)
LR = 5e-5
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_LENGTH = 4096
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
SAVE_STEPS = 5
LOGGING_STEPS = 1
MAX_GRAD_NORM = 0.5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15

print(f"[SFT v5] 开始 - {datetime.now()}")

# ============ Step 1: 准备数据 ============
print("\n[Step 1] 准备训练数据 ...")

# 加载原始数据
all_data = []
if os.path.exists(ORIGINAL_DATA):
    with open(ORIGINAL_DATA) as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    print(f"  原始数据: {len(all_data)} 条")

# 加载新生成的 logic 数据
logic_files = glob.glob(os.path.join(LOGIC_DATA_DIR, "*logic*.jsonl"))
logic_count = 0
for lf in logic_files:
    if lf == ORIGINAL_DATA:
        continue
    with open(lf) as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    all_data.append(item)
                    logic_count += 1
                except:
                    pass
print(f"  Logic 补充数据: {logic_count} 条 (来自 {len(logic_files)} 个文件)")

# 从 Battle Round 1 提取高分样本作为额外训练数据
battle_extra = 0
if os.path.exists(BATTLE_TRACE):
    with open(BATTLE_TRACE) as f:
        for line in f:
            r = json.loads(line)
            grade = r["judge_result"]["grade"]
            if grade in ("S", "A") and r["judge_result"]["total_score"] >= 35:
                # 用高分的 audit_report 作为训练样本
                sample = r["sample"]
                report = r["audit_report"]
                item = {
                    "instruction": "请对以下代码进行安全审计，识别潜在的安全漏洞并给出修复建议。",
                    "input": sample["code"],
                    "output": json.dumps(report, ensure_ascii=False, indent=2)
                }
                all_data.append(item)
                battle_extra += 1
    print(f"  Battle 高分样本: {battle_extra} 条 (score >= 35)")

print(f"  总计: {len(all_data)} 条")

# 保存合并数据
merged_path = os.path.join(LOGIC_DATA_DIR, "sft_v5_merged.jsonl")
with open(merged_path, "w") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"  合并数据已保存: {merged_path}")

# ============ Step 2: Monkey-patch Marlin ============
print("\n[Step 2] Patch Marlin 训练模式 ...")
try:
    import gptqmodel.nn_modules.qlinear as ql
    orig_init = ql.BaseQuantLinear.__init_subclass__
except:
    pass

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
_original_train = BaseQuantLinear.train
def _patched_train(self, mode=True):
    try:
        return _original_train(mode)
    except NotImplementedError:
        return self
BaseQuantLinear.train = _patched_train
print("  ✅ Marlin train() 已 patch")

# ============ Step 3: 加载模型 ============
print("\n[Step 3] 加载基座模型 ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTQConfig

quantization_config = GPTQConfig(bits=4, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
    local_files_only=True,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

avail = torch.cuda.mem_get_info()[0] / 1024**3
print(f"  ✅ 模型已加载，可用显存: {avail:.1f}GB")

# ============ Step 4: LoRA 配置 ============
print("\n[Step 4] 配置 LoRA ...")
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ Step 5: 准备数据集 ============
print("\n[Step 5] 准备数据集 ...")
from datasets import Dataset

def format_chat(item):
    messages = [
        {"role": "system", "content": "你是一个专业的代码安全审计专家。请以JSON格式输出审计报告。"},
        {"role": "user", "content": item["instruction"] + "\n\n```python\n" + item["input"] + "\n```"},
        {"role": "assistant", "content": item["output"] if isinstance(item["output"], str) else json.dumps(item["output"], ensure_ascii=False, indent=2)}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list(all_data)
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
print(f"  ✅ 数据集: {len(dataset)} 条")

# ============ Step 6: 训练 ============
print(f"\n[Step 6] 开始训练 — {datetime.now()}")
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    max_length=MAX_LENGTH,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=5,
    fp16=True,
    max_grad_norm=MAX_GRAD_NORM,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\n  ✅ 训练完成! 耗时: {elapsed/60:.1f} 分钟")

# ============ Step 7: 保存 ============
print("\n[Step 7] 保存模型 ...")
final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"  ✅ 模型已保存: {final_dir}")

# ============ Step 8: 推理测试 ============
print("\n[Step 8] 推理测试 ...")

# 清理训练显存
del trainer
gc.collect()
torch.cuda.empty_cache()

test_code = '''
import threading, time

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        if self.balance >= amount:
            time.sleep(0.01)  # 模拟处理延迟
            self.balance -= amount
            return True
        return False
    
    def check_and_withdraw(self, amount):
        # 先检查余额
        if self.balance >= amount:
            # 检查通过后执行扣款
            return self.withdraw(amount)
        return False
'''

messages = [
    {"role": "system", "content": "你是一个专业的代码安全审计专家。请以JSON格式输出审计报告。"},
    {"role": "user", "content": f"请对以下代码进行安全审计，识别潜在的安全漏洞并给出修复建议。\n\n```python\n{test_code}\n```"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.3, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)

# 清理
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

print(f"\n[SFT v5] 全部完成 — {datetime.now()}")