#!/usr/bin/env python3
"""
SSPilot Overnight Training Pipeline — SFT v6 + DPO v1
======================================================
Phase 1: 从所有 battle traces 提取 SFT + DPO 训练数据
Phase 2: SFT v6 — 合并 v5 LoRA，在蒸馏数据上训练新 LoRA
Phase 3: DPO v1 — 合并 v6 LoRA，在偏好对上训练
Phase 4: 更新 config.py，输出结果摘要

用法:
  nohup python scripts/overnight_sft_dpo.py > overnight_sft_dpo.log 2>&1 &
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import glob
import time
import sys
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ============================================================
#  全局配置
# ============================================================
PROJECT_ROOT = Path("/home/xsuper/sspilot")
TRACE_DIR = PROJECT_ROOT / "traces"
DATASET_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BASE_MODEL = "/home/xsuper/models/Qwen2.5-Coder-32B-Int4"
PREV_LORA = str(CHECKPOINT_DIR / "audit_sft_v5" / "final")
SFT_OUTPUT = str(CHECKPOINT_DIR / "audit_sft_v6")
DPO_OUTPUT = str(CHECKPOINT_DIR / "dpo_audit_v1")

SFT_MIN_SCORE = 28       # Grade A+
DPO_SCORE_GAP = 5        # 降低阈值以获取更多 DPO 对
SFT_EPOCHS = 3
DPO_EPOCHS = 2
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LR = 5e-5
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_LENGTH = 4096

LOG_FILE = PROJECT_ROOT / "overnight_sft_dpo.log"
STARTED_AT = datetime.now()


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


_marlin_patched = False


def _apply_marlin_patch():
    global _marlin_patched
    if _marlin_patched:
        return
    try:
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear
        def _safe_train(self, mode=True):
            self.training = mode
            return self
        BaseQuantLinear.train = _safe_train
        _marlin_patched = True
        log("  Marlin train() patched")
    except Exception as e:
        log(f"  Marlin patch skipped: {e}")


def gpu_info():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        return f"GPU {used:.1f}/{total/1024**3:.1f}GB"
    return "no GPU"


# ============================================================
#  Phase 1: 数据蒸馏
# ============================================================
def phase1_distill():
    log("=" * 60)
    log("PHASE 1: 数据蒸馏 — 从 battle traces 提取训练数据")
    log("=" * 60)

    all_traces = []
    for trace_file in sorted(TRACE_DIR.glob("battle_round*.jsonl")):
        with open(trace_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_traces.append(json.loads(line))
    log(f"  加载了 {len(all_traces)} 条 trace")

    # --- SFT 数据 ---
    sft_samples = []
    for t in all_traces:
        jr = t.get("judge_result", {})
        if "scores" not in jr or jr["total_score"] < SFT_MIN_SCORE:
            continue
        sample = t.get("sample", {})
        audit = t.get("audit_report", {})
        if not sample.get("code") or audit.get("error") or audit.get("parse_error"):
            continue
        sft_samples.append({
            "instruction": "请对以下代码进行安全审计，识别潜在的安全漏洞并给出修复建议。",
            "input": sample["code"],
            "output": json.dumps(audit, ensure_ascii=False, indent=2),
            "vuln_type": t.get("vuln_type", "unknown"),
            "score": jr["total_score"],
        })

    sft_path = DATASET_DIR / "sft_distilled_v6.jsonl"
    with open(sft_path, "w") as f:
        for s in sft_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log(f"  SFT 数据: {len(sft_samples)} 条 → {sft_path}")

    # --- DPO 数据 ---
    by_type: dict[str, list] = defaultdict(list)
    for t in all_traces:
        jr = t.get("judge_result", {})
        if "scores" not in jr:
            continue
        audit = t.get("audit_report", {})
        if audit.get("error") or audit.get("parse_error"):
            continue
        by_type[t.get("vuln_type", "unknown")].append(t)

    dpo_pairs = []
    for vt, items in by_type.items():
        scored = sorted(items, key=lambda x: x["judge_result"]["total_score"], reverse=True)
        i, j = 0, len(scored) - 1
        while i < j:
            high = scored[i]
            low = scored[j]
            gap = high["judge_result"]["total_score"] - low["judge_result"]["total_score"]
            if gap >= DPO_SCORE_GAP:
                prompt_text = (
                    "你是一位顶级代码安全审计专家。请对给定代码进行全面安全审计，"
                    "找出所有潜在漏洞，并给出详细的分析和修复建议。以JSON格式输出。\n\n"
                    f"请对以下代码进行安全审计：\n\n```python\n{high['sample']['code']}\n```"
                )
                dpo_pairs.append({
                    "prompt": prompt_text,
                    "chosen": json.dumps(high["audit_report"], ensure_ascii=False),
                    "rejected": json.dumps(low["audit_report"], ensure_ascii=False),
                    "metadata": {
                        "vuln_type": vt,
                        "chosen_score": high["judge_result"]["total_score"],
                        "rejected_score": low["judge_result"]["total_score"],
                        "score_gap": gap,
                    },
                })
                i += 1
                j -= 1
            else:
                j -= 1

    dpo_path = DATASET_DIR / "dpo_distilled_v1.jsonl"
    with open(dpo_path, "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"  DPO 数据: {len(dpo_pairs)} 对 → {dpo_path}")

    # 合并所有可用 SFT 数据
    extra_data = []
    for extra_file in [
        DATASET_DIR / "logic_vuln_training.jsonl",
        DATASET_DIR / "battle_augmented_v6.jsonl",
    ]:
        if extra_file.exists():
            with open(extra_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            extra_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    log(f"  补充数据: {len(extra_data)} 条 (logic+augmented)")

    return {
        "sft_samples": sft_samples,
        "extra_data": extra_data,
        "dpo_pairs": dpo_pairs,
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
    }


# ============================================================
#  Phase 2: SFT v6
# ============================================================
def phase2_sft(sft_samples: list, extra_data: list):
    log("=" * 60)
    log("PHASE 2: SFT v6 — 合并 v5 + 蒸馏数据训练")
    log("=" * 60)
    log(f"  基座: {BASE_MODEL}")
    log(f"  前置 LoRA: {PREV_LORA}")
    log(f"  输出: {SFT_OUTPUT}")
    log(f"  数据: {len(sft_samples)} 蒸馏 + {len(extra_data)} 补充 = {len(sft_samples) + len(extra_data)} 总计")
    log(f"  配置: epochs={SFT_EPOCHS}, lr={LR}, r={LORA_R}, α={LORA_ALPHA}")

    # --- Marlin patch ---
    _apply_marlin_patch()

    # --- 加载模型 ---
    log(f"  加载基座模型... ({gpu_info()})")
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
    from peft import PeftModel

    quantization_config = GPTQConfig(bits=4, use_exllama=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  基座已加载 ({gpu_info()})")

    # --- 加载 v5 LoRA 并继续训练 (GPTQ 不支持 merge，改为 resume training) ---
    if os.path.exists(PREV_LORA):
        log(f"  加载 v5 LoRA (resume training): {PREV_LORA}")
        model = PeftModel.from_pretrained(model, PREV_LORA, is_trainable=True)
        log(f"  v5 LoRA 已加载 ({gpu_info()})")
    else:
        log(f"  ⚠️ v5 LoRA 不存在，创建新 LoRA")
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 准备数据集 (统一格式化为 chat text) ---
    from datasets import Dataset

    sys_prompt = "你是一个专业的代码安全审计专家。请以JSON格式输出审计报告。"
    default_instr = "请对以下代码进行安全审计，识别潜在的安全漏洞并给出修复建议。"

    formatted_texts = []
    for item in list(sft_samples) + list(extra_data):
        try:
            # 兼容两种格式: {instruction,input,output} 和 trace 格式 {sample,audit_report,...}
            if "input" in item:
                code = item["input"]
                instruction = item.get("instruction", default_instr)
                output = item.get("output", "")
            elif "sample" in item and isinstance(item["sample"], dict):
                code = item["sample"].get("code", "")
                instruction = default_instr
                output = item.get("audit_report", "")
            else:
                continue

            if not code:
                continue
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, indent=2)
            if len(output) < 50:
                continue

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": instruction + "\n\n```python\n" + code + "\n```"},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_texts.append({"text": text})
        except Exception:
            continue

    log(f"  格式化数据: {len(formatted_texts)} 条 (蒸馏 + 补充)")
    dataset = Dataset.from_list(formatted_texts)
    log(f"  数据集就绪: {len(dataset)} 条")

    # --- 训练 ---
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=SFT_OUTPUT,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_length=MAX_LENGTH,
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        fp16=True,
        max_grad_norm=0.5,
        weight_decay=0.01,
        warmup_ratio=0.15,
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

    log(f"  开始 SFT 训练... ({gpu_info()})")
    t0 = time.time()
    trainer.train()
    sft_elapsed = time.time() - t0
    log(f"  SFT 训练完成! 耗时 {sft_elapsed/60:.1f} 分钟")

    # --- 保存 ---
    final_dir = os.path.join(SFT_OUTPUT, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log(f"  SFT v6 已保存: {final_dir}")

    # --- 清理 ---
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    log(f"  SFT 阶段内存已释放 ({gpu_info()})")

    return {"elapsed_min": round(sft_elapsed / 60, 1), "output_dir": final_dir}


# ============================================================
#  Phase 3: DPO v1
# ============================================================
def phase3_dpo(dpo_pairs: list):
    log("=" * 60)
    log("PHASE 3: DPO v1 — 偏好优化训练")
    log("=" * 60)

    if len(dpo_pairs) < 3:
        log(f"  ⚠️ DPO 数据仅 {len(dpo_pairs)} 对，太少，跳过 DPO 阶段")
        return {"skipped": True, "reason": "insufficient data"}

    sft_v6_lora = os.path.join(SFT_OUTPUT, "final")
    log(f"  基座: {BASE_MODEL}")
    log(f"  SFT LoRA: {sft_v6_lora}")
    log(f"  输出: {DPO_OUTPUT}")
    log(f"  数据: {len(dpo_pairs)} DPO 对")
    log(f"  配置: epochs={DPO_EPOCHS}, beta=0.1, lr={LR}")

    # --- Marlin patch ---
    _apply_marlin_patch()

    # --- 加载模型 + SFT v6 LoRA ---
    log(f"  加载基座模型... ({gpu_info()})")
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
    from peft import PeftModel

    quantization_config = GPTQConfig(bits=4, use_exllama=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    log(f"  基座已加载 ({gpu_info()})")

    # 加载 SFT v6 LoRA 并继续训练 (DPO 在 SFT 基础上微调同一 adapter)
    if os.path.exists(sft_v6_lora):
        log(f"  加载 SFT v6 LoRA (DPO 继续微调)...")
        model = PeftModel.from_pretrained(model, sft_v6_lora, is_trainable=True)
        log(f"  v6 LoRA 已加载 ({gpu_info()})")
    else:
        log(f"  ⚠️ SFT v6 LoRA 未找到，跳过 DPO")
        return {"skipped": True, "reason": "no SFT v6 adapter"}

    # --- DPO 数据集 ---
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in dpo_pairs
    ])
    log(f"  DPO 数据集: {len(dataset)} 对")

    training_args = DPOConfig(
        output_dir=DPO_OUTPUT,
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-5,
        beta=0.1,
        max_length=2048,
        max_prompt_length=1024,
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        loss_type="sigmoid",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log(f"  开始 DPO 训练... ({gpu_info()})")
    t0 = time.time()
    trainer.train()
    dpo_elapsed = time.time() - t0
    log(f"  DPO 训练完成! 耗时 {dpo_elapsed/60:.1f} 分钟")

    # --- 保存 (DPO 微调后的 adapter 覆盖回 SFT v6 目录，因为是同一 adapter) ---
    dpo_final = os.path.join(DPO_OUTPUT, "final")
    os.makedirs(dpo_final, exist_ok=True)
    model.save_pretrained(dpo_final)
    tokenizer.save_pretrained(dpo_final)
    log(f"  DPO v1 已保存: {dpo_final}")

    # --- 清理 ---
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  DPO 阶段内存已释放 ({gpu_info()})")

    return {"elapsed_min": round(dpo_elapsed / 60, 1), "output_dir": final_dir}


# ============================================================
#  Phase 4: 更新 config + 结果摘要
# ============================================================
def phase4_summary(distill_result, sft_result, dpo_result):
    log("=" * 60)
    log("PHASE 4: 训练结果摘要")
    log("=" * 60)

    total_elapsed = (datetime.now() - STARTED_AT).total_seconds() / 60

    summary = {
        "started": STARTED_AT.isoformat(),
        "finished": datetime.now().isoformat(),
        "total_minutes": round(total_elapsed, 1),
        "distill": {
            "sft_samples": len(distill_result["sft_samples"]),
            "dpo_pairs": len(distill_result["dpo_pairs"]),
            "extra_data": len(distill_result["extra_data"]),
        },
        "sft_v6": sft_result,
        "dpo_v1": dpo_result,
    }

    log(f"  总耗时: {total_elapsed:.1f} 分钟")
    log(f"  SFT v6: {sft_result}")
    log(f"  DPO v1: {dpo_result}")

    # 保存摘要
    summary_path = PROJECT_ROOT / "overnight_sft_dpo_result.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  结果已保存: {summary_path}")

    # 提示下一步
    log("")
    log("=" * 60)
    log("下一步操作:")
    log(f"  1. 更新 config.py lora_path → {SFT_OUTPUT}/final")
    log(f"  2. 运行 battle round 4 验证效果")
    log(f"  3. 检查 {summary_path} 查看详细结果")
    log("=" * 60)

    return summary


# ============================================================
#  主流程
# ============================================================
def main():
    log("=" * 60)
    log("SSPilot 隔夜训练流水线 — SFT v6 + DPO v1")
    log(f"开始时间: {STARTED_AT}")
    log(f"GPU: {gpu_info()}")
    log("=" * 60)

    try:
        # Phase 1
        distill_result = phase1_distill()

        # Phase 2
        sft_result = phase2_sft(distill_result["sft_samples"], distill_result["extra_data"])

        # Phase 3
        dpo_result = phase3_dpo(distill_result["dpo_pairs"])

        # Phase 4
        phase4_summary(distill_result, sft_result, dpo_result)

        log("\n✅ 全部训练完成!")

    except Exception as e:
        log(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
