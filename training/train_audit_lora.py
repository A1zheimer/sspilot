#!/usr/bin/env python3
"""Train LoRA v5 on high-quality battle data."""
import json, os, sys

BASE_MODEL = "/models/Qwen2.5-Coder-32B-Int4"
OLD_LORA = "/workspace/sspilot/checkpoints/audit_sft_v4/final"
OUTPUT_DIR = "/workspace/sspilot/checkpoints/audit_sft_v5"
DATA_FILE = "/workspace/sspilot/training/data/audit_sft_all.json"

def main():
    if not os.path.exists(DATA_FILE):
        print(f"No data: {DATA_FILE}"); sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    print(f"Training data: {len(data)} samples")
    if len(data) < 5:
        print("Too few samples, skip"); sys.exit(1)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    print(f"Loading: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    if os.path.exists(OLD_LORA):
        print(f"Merging LoRA v4: {OLD_LORA}")
        model = PeftModel.from_pretrained(model, OLD_LORA)
        model = model.merge_and_unload()

    def format_sample(ex):
        t = "<|im_start|>system\nYou are a security audit expert.<|im_end|>\n"
        t += f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
        t += f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        return {"text": t}

    dataset = Dataset.from_list(data).map(format_sample)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM")
    training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-5, warmup_ratio=0.1, logging_steps=5,
        save_strategy="epoch", save_total_limit=2, fp16=True,
        dataloader_num_workers=0, report_to="none", max_grad_norm=1.0, lr_scheduler_type="cosine")

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=training_args, peft_config=lora_config, max_seq_length=4096, dataset_text_field="text")
    print("Training...")
    trainer.train()
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"LoRA v5 saved: {final_dir}")

if __name__ == "__main__":
    main()
