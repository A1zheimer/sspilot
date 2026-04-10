#!/usr/bin/env python3
"""
from __future__ import annotations
SSPilot DPO Training Script (修复版)

修复的问题:
  原 dpo_train.py 使用 NeMo 原生加载方式，不兼容 GPTQ 量化的 nemotron 模型。
  改用 transformers AutoModelForCausalLM + GPTQConfig + trl DPOTrainer。

依赖:
  pip install transformers peft trl datasets accelerate auto-gptq

用法:
  # 在 SFT 完成后运行 (依赖 SFT checkpoint)
  python scripts/dpo_train.py \
    --base-model /home/xsuper/models/nemotron-3-nano-30b-a3b \
    --sft-adapter /home/xsuper/sspilot/checkpoints/sft-v5 \
    --dpo-data /home/xsuper/sspilot/datasets/dpo_audit.jsonl \
    --output-dir /home/xsuper/sspilot/checkpoints/dpo-v1 \
    --epochs 2 \
    --beta 0.1

硬件适配: DGX Spark GB10 (119GB UVM, ARM aarch64)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="SSPilot DPO Training (Fixed)")
    parser.add_argument("--base-model", type=str,
                        default="/home/xsuper/models/nemotron-3-nano-30b-a3b",
                        help="Base model path (nemotron)")
    parser.add_argument("--sft-adapter", type=str,
                        default=None,
                        help="SFT LoRA adapter path (optional, if already SFT'd)")
    parser.add_argument("--dpo-data", type=str,
                        default="/home/xsuper/sspilot/datasets/dpo_audit.jsonl",
                        help="DPO training data (JSONL with prompt/chosen/rejected)")
    parser.add_argument("--output-dir", type=str,
                        default="/home/xsuper/sspilot/checkpoints/dpo-v1",
                        help="Output directory for DPO adapter")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data and config without training")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  SSPilot DPO Training (Fixed Version)")
    print(f"{'='*60}")
    print(f"  Base model:  {args.base_model}")
    print(f"  SFT adapter: {args.sft_adapter or 'None (training from base)'}")
    print(f"  DPO data:    {args.dpo_data}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Beta:        {args.beta}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LoRA:        r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"{'='*60}")

    # ── Step 1: 验证数据 ──
    print("\n[Step 1/5] Validating DPO data ...")
    dpo_data = load_and_validate_data(args.dpo_data)
    print(f"  ✅ {len(dpo_data)} DPO pairs loaded")

    if args.dry_run:
        print("\n🏁 Dry run complete. Data is valid.")
        return

    # ── Step 2: 加载基础模型 ──
    print("\n[Step 2/5] Loading base model ...")
    model, tokenizer = load_base_model(args.base_model)
    print(f"  ✅ Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Step 3: 如果有 SFT adapter, 合并 ──
    if args.sft_adapter:
        print(f"\n[Step 3/5] Loading SFT adapter from {args.sft_adapter} ...")
        model = load_sft_adapter(model, args.sft_adapter)
        print(f"  ✅ SFT adapter merged. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    else:
        print("\n[Step 3/5] No SFT adapter, skipping ...")

    # ── Step 4: 配置 DPO LoRA ──
    print("\n[Step 4/5] Configuring DPO LoRA + Trainer ...")
    trainer = setup_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dpo_data=dpo_data,
        args=args,
    )
    print(f"  ✅ DPO Trainer configured")

    # ── Step 5: 训练 ──
    print(f"\n[Step 5/5] Starting DPO training ({args.epochs} epochs) ...")
    trainer.train()

    # 保存
    print(f"\n  Saving DPO adapter to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  ✅ DPO training complete!")
    print(f"  📦 Adapter saved to: {args.output_dir}")


# ──────────────────────────────────────────────────────────────────────
#  数据加载 & 验证
# ──────────────────────────────────────────────────────────────────────

def load_and_validate_data(data_path: str) -> list[dict]:
    """加载并验证 DPO JSONL 数据"""
    data = []
    required_fields = {"prompt", "chosen", "rejected"}

    with open(data_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Line {i}: Invalid JSON — {e}")
                continue

            missing = required_fields - set(item.keys())
            if missing:
                print(f"  ⚠️ Line {i}: Missing fields {missing}")
                continue

            # 基本质量检查
            if len(item["chosen"]) < 50:
                print(f"  ⚠️ Line {i}: Chosen response too short ({len(item['chosen'])} chars)")
                continue
            if len(item["rejected"]) < 20:
                print(f"  ⚠️ Line {i}: Rejected response too short ({len(item['rejected'])} chars)")
                continue

            data.append(item)

    if len(data) == 0:
        raise ValueError(f"No valid DPO pairs found in {data_path}")

    # 统计
    avg_chosen = sum(len(d["chosen"]) for d in data) / len(data)
    avg_rejected = sum(len(d["rejected"]) for d in data) / len(data)
    print(f"  Valid pairs: {len(data)}")
    print(f"  Avg chosen length:   {avg_chosen:.0f} chars")
    print(f"  Avg rejected length: {avg_rejected:.0f} chars")

    return data


# ──────────────────────────────────────────────────────────────────────
#  模型加载 — 核心修复: AutoModelForCausalLM + 正确的 dtype
# ──────────────────────────────────────────────────────────────────────

def load_base_model(model_path: str):
    """
    加载 nemotron 基础模型

    关键修复:
    - 使用 AutoModelForCausalLM (不是 NeMo 原生加载)
    - FP16 精度 (nemotron 不是 GPTQ 量化的)
    - trust_remote_code=True (Hybrid Mamba+MoE 需要)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",  # DPO 需要 left padding
    )

    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return model, tokenizer


def load_sft_adapter(model, adapter_path: str):
    """
    加载 SFT LoRA adapter 并合并到基础模型

    DPO 训练需要在 SFT 后的模型上进行，所以先合并 SFT adapter
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path)
    # 合并 LoRA 到基础模型 (DPO 需要在合并后的模型上加新 LoRA)
    model = model.merge_and_unload()
    print(f"  ✅ SFT adapter merged and unloaded")
    return model


# ──────────────────────────────────────────────────────────────────────
#  DPO Trainer 配置
# ──────────────────────────────────────────────────────────────────────

def setup_dpo_trainer(model, tokenizer, dpo_data: list[dict], args):
    """
    配置 trl DPOTrainer

    使用 trl 库的 DPOTrainer，它已经处理了:
    - DPO loss 计算
    - Reference model 管理
    - LoRA 适配器配置
    """
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from trl import DPOConfig, DPOTrainer

    # 构建 HuggingFace Dataset
    dataset = Dataset.from_list([
        {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
        for item in dpo_data
    ])

    # LoRA 配置 — 针对 nemotron 的模块
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # nemotron Hybrid Mamba+MoE 的目标模块
        # 如果自动检测失败，手动指定
        target_modules=find_target_modules(model),
    )

    # DPO 训练配置
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        bf16=False,   # DGX Spark GB10 用 FP16
        fp16=True,
        gradient_checkpointing=True,  # 节省显存
        remove_unused_columns=False,
        report_to="none",  # Hackathon 不需要 wandb
        # DPO 特定参数
        loss_type="sigmoid",  # 标准 DPO loss
    )

    # 创建 DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    return trainer


def find_target_modules(model) -> list[str]:
    """
    自动检测 LoRA 目标模块

    nemotron 是 Hybrid Mamba+MoE 架构，模块名可能与标准 transformer 不同。
    遍历模型参数名，找到 Linear 层。
    """
    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 提取最后一级名称 (e.g., "q_proj", "k_proj", "gate_proj")
            parts = name.split(".")
            if parts:
                target_names.add(parts[-1])

    # 过滤常见的 LoRA 目标
    preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = target_names & preferred

    if found:
        result = sorted(found)
        print(f"  LoRA targets (auto-detected): {result}")
        return result
    else:
        # Fallback: 使用所有 Linear 层名称（去掉 lm_head 等）
        fallback = sorted(target_names - {"lm_head", "embed_tokens"})[:6]
        print(f"  LoRA targets (fallback): {fallback}")
        return fallback


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
