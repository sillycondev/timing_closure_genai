#!/usr/bin/env python3
"""
train_sft.py — Stage 1: Supervised Fine-Tuning (SFT Warmup) with QLoRA

Fine-tunes a base code model (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct or 7B) using 4-bit NF4 QLoRA
on `dataset/generated_dataset/sft_dataset.jsonl`. Memory footprint is optimized for 8 GB VRAM.

Usage:
    python3 scripts/train_sft.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --epochs 3
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Stage 1 SFT Training with QLoRA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model name/path")
    parser.add_argument("--dataset", type=str, default="dataset/generated_dataset/sft_dataset.jsonl", help="Path to SFT dataset")
    parser.add_argument("--output_dir", type=str, default="models/sft_lora_checkpoint", help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / args.dataset
    output_path = project_root / args.output_dir

    if not dataset_path.exists():
        print(f"Error: Dataset file {dataset_path} does not exist. Run prepare_sft_dataset.py first.")
        sys.exit(1)

    print("=" * 72)
    print("Stage 1: Supervised Fine-Tuning (SFT Warmup)")
    print(f"Base Model : {args.model}")
    print(f"Dataset    : {dataset_path}")
    print(f"Output Dir : {output_path}")
    print(f"CUDA       : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 72)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError as e:
        print(f"Error importing training libraries: {e}")
        print("Ensure transformers, peft, trl, bitsandbytes, datasets, and torch are installed.")
        sys.exit(1)

    # 1. 4-Bit Quantization Config (NF4) for 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Tokenizer & Model
    print("Loading tokenizer and 4-bit quantized base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Load Dataset
    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # 5. SFT Config
    sft_config = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataset_text_field="messages",
        max_length=2048,
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving final LoRA adapter weights to {output_path}...")
    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("SFT Training Complete!")

if __name__ == "__main__":
    main()
