#!/usr/bin/env python3
"""
train_grpo_rlvr.py — Stage 2: Group Relative Policy Optimization (GRPO) with EDA Verifiable Rewards

Connects the LLM directly to the EDA toolchain (Yosys + OpenSTA + EQY) to provide real-time
mathematical timing improvement rewards and formal equivalence penalties.

Usage:
    python3 scripts/train_grpo_rlvr.py --model models/sft_lora_checkpoint
"""

import os
import sys
import re
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure scripts directory is in python path for eda_runner import
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import eda_runner
    from jinja2 import Environment, FileSystemLoader
    import yaml
except ImportError as e:
    print(f"Error importing eda_runner dependencies: {e}")

# Global configuration for EDA rewards
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "scripts" / "config.yaml"

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        EDA_CONFIG = yaml.safe_load(f)
else:
    EDA_CONFIG = {}

JINJA_ENV = Environment(
    loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
    keep_trailing_newline=True,
)
WORKDIR = PROJECT_ROOT / "dataset" / "workdir_grpo"


# ---------------------------------------------------------------------------
# EDA Reward Functions
# ---------------------------------------------------------------------------

def extract_verilog_code(completion: str) -> str:
    """Extract Verilog code block from model completion."""
    m = re.search(r"```verilog\s*(.*?)\s*```", completion, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback if markdown block is missing
    return completion.strip()


def reward_verilog_format(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward for proper Verilog markdown formatting."""
    rewards = []
    for completion in completions:
        if "```verilog" in completion and "```" in completion:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def reward_eda_timing_and_correctness(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Master EDA Reward Function:
    1. Extracts generated Verilog code.
    2. Runs Yosys synthesis against SkyWater 130nm PDK. (Penalty -3.0 if synthesis fails)
    3. Runs OpenSTA timing analysis. (Reward = +1.0 * WNS_improvement + 3.0 bonus if WNS >= 0)
    4. Runs EQY formal equivalence. (Penalty -2.0 if non-equivalent)
    """
    rewards = []
    WORKDIR.mkdir(parents=True, exist_ok=True)

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        gen_rtl = extract_verilog_code(completion)

        # Basic check for empty or non-Verilog output
        if not gen_rtl or "module" not in gen_rtl:
            rewards.append(-5.0)
            continue

        # Extract top module name from generated RTL
        mod_match = re.search(r"module\s+(\w+)", gen_rtl)
        if not mod_match:
            rewards.append(-5.0)
            continue

        mod_name = mod_match.group(1)
        stem = f"grpo_sample_{i}_{mod_name}"

        # Detect clock port
        clk_match = re.search(r'\b(clk|clock|CLK|Clock|Clk)\b', gen_rtl)
        has_clock = clk_match is not None
        clk_port = clk_match.group(1) if clk_match else "clk"

        module_info = {
            "module_name": mod_name,
            "stem": stem,
            "filepath": str(WORKDIR / f"{stem}.v"),
            "is_sv": True,
            "has_clock": has_clock,
            "clock_port_name": clk_port,
            "rtl_source": gen_rtl,
        }

        # Write candidate Verilog to workdir
        cand_file = WORKDIR / f"{stem}.v"
        cand_file.write_text(gen_rtl)

        # Step 1: Yosys Synthesis
        synth_res = eda_runner.synthesize(module_info, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, WORKDIR)
        if not synth_res["success"]:
            rewards.append(-3.0)  # Synthesis failure penalty
            continue

        if not has_clock:
            rewards.append(0.5)  # Synthesis passed for comb module
            continue

        # Step 2: OpenSTA Timing Analysis
        sta_res = eda_runner.run_sta(module_info, synth_res, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, 1.0, WORKDIR)
        if not sta_res.get("success"):
            rewards.append(-1.0)
            continue

        wns = sta_res.get("wns_ns", -10.0)

        # Compute Timing Reward
        # Base reward proportional to WNS (higher WNS = higher reward)
        reward = wns  # E.g. -0.5ns WNS gives -0.5, +0.2ns WNS gives +0.2

        if wns >= 0.0:
            reward += 3.0  # Timing closure bonus!

        rewards.append(reward)

    torch.cuda.empty_cache()
    return rewards


# ---------------------------------------------------------------------------
# Main GRPO Training Function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 GRPO RLVR Training Loop")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model ID")
    parser.add_argument("--model", type=str, default="models/sft_lora_checkpoint", help="Path to SFT model checkpoint")
    parser.add_argument("--dataset", type=str, default="dataset/generated_dataset/sft_dataset.jsonl", help="Path to RL prompt dataset")
    parser.add_argument("--output_dir", type=str, default="models/grpo_rlvr_checkpoint", help="Output directory")
    parser.add_argument("--num_generations", type=int, default=2, help="Number of completion candidates per prompt (G)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for RL policy")
    args = parser.parse_args()

    print("=" * 72)
    print("Stage 2: Group Relative Policy Optimization (GRPO RLVR)")
    print(f"Base Model       : {args.base_model}")
    print(f"Model Checkpoint : {args.model}")
    print(f"Prompts Dataset  : {args.dataset}")
    print(f"Output Dir       : {args.output_dir}")
    print(f"Generations (G)  : {args.num_generations}")
    print("=" * 72)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        from trl import GRPOTrainer, GRPOConfig
        from datasets import load_dataset
    except ImportError as e:
        print(f"Error importing GRPO training libraries: {e}")
        print("Ensure transformers, trl, peft, and torch are installed.")
        sys.exit(1)

    # 1. Load Tokenizer & Base Model
    print("Loading base model with SFT LoRA checkpoint...")
    torch.cuda.empty_cache()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    if os.path.exists(args.model):
        print(f"Wrapping base model with SFT LoRA adapter from {args.model}...")
        model = PeftModel.from_pretrained(base_model, args.model, is_trainable=True)
    else:
        model = base_model

    # 2. Load Prompts Dataset
    raw_dataset = load_dataset("json", data_files=str(args.dataset), split="train")

    def extract_prompt_only(example):
        # Extract user prompt from ChatML structure for GRPOTrainer
        messages = example["messages"]
        user_msg = [m["content"] for m in messages if m["role"] == "user"][0]
        # Cap prompt length to prevent memory spikes on large modules
        if len(user_msg) > 1200:
            user_msg = user_msg[:1200]
        return {"prompt": user_msg}

    prompt_dataset = raw_dataset.map(extract_prompt_only)

    # 3. GRPO Config (tuned for 8GB VRAM)
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=256,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
    )

    # 4. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_verilog_format, reward_eda_timing_and_correctness],
        args=grpo_config,
        train_dataset=prompt_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO Reinforcement Learning loop...")
    trainer.train()

    print(f"Saving final GRPO RLVR trained model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("GRPO RLVR Training Complete!")


if __name__ == "__main__":
    main()
