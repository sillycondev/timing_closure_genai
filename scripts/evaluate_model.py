#!/usr/bin/env python3
"""
evaluate_model.py — Benchmark & Evaluate Fine-Tuned Model

Evaluates fine-tuned model (SFT / GRPO) on test Verilog modules:
1. Generates fixed Verilog candidate from original RTL + OpenSTA timing report.
2. Runs Yosys synthesis + OpenSTA timing analysis + EQY formal equivalence.
3. Computes metrics:
   - Syntax Pass Rate (%)
   - Synthesis Pass Rate (%)
   - Average WNS Improvement (ns)
   - Timing Closure Rate (%) (WNS >= 0)
   - Formal Equivalence Pass Rate (%)

Usage:
    python3 scripts/evaluate_model.py --model models/sft_lora_checkpoint --limit 10
"""

import os
import sys
import re
import json
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import eda_runner
    from jinja2 import Environment, FileSystemLoader
    import yaml
except ImportError as e:
    print(f"Error importing dependencies: {e}")

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
WORKDIR = PROJECT_ROOT / "dataset" / "workdir_eval"


def generate_fix(model, tokenizer, original_rtl: str, timing_report: dict) -> str:
    """Generate Verilog fix using fine-tuned model."""
    from prepare_sft_dataset import format_user_prompt, SYSTEM_PROMPT

    user_prompt = format_user_prompt(original_rtl, timing_report)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Extract verilog block
    m = re.search(r"```verilog\s*(.*?)\s*```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Model on Timing Closure")
    parser.add_argument("--model", type=str, default="models/sft_lora_checkpoint", help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="dataset/generated_dataset/timing_dataset.jsonl", help="Test dataset")
    parser.add_argument("--limit", type=int, default=10, help="Number of test samples to evaluate")
    args = parser.parse_args()

    in_path = PROJECT_ROOT / args.dataset
    if not in_path.exists():
        print(f"Error: Dataset file {in_path} does not exist.")
        sys.exit(1)

    print("=" * 72)
    print("Evaluating Fine-Tuned Model on Timing Closure Benchmark")
    print(f"Model Checkpoint : {args.model}")
    print(f"Test Dataset     : {in_path}")
    print("=" * 72)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as e:
        print(f"Error importing transformers/peft: {e}")
        sys.exit(1)

    # Load Model
    print("Loading model for evaluation...")
    torch.cuda.empty_cache()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if (Path(args.model) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, args.model)
    else:
        model = base_model

    model.eval()

    # Read samples
    samples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    if args.limit:
        samples = samples[:args.limit]

    print(f"Evaluating {len(samples)} samples...")

    total_samples = len(samples)
    syntax_passed = 0
    synth_passed = 0
    timing_closed = 0
    equiv_passed = 0
    wns_improvements = []

    WORKDIR.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        mod_name = sample.get("id", f"eval_{i}")
        orig_rtl = sample["original_rtl"]
        orig_sta = sample["timing_report"]
        orig_wns = orig_sta.get("wns_ns", 0.0)

        print(f"\n[{i+1}/{total_samples}] Evaluating {mod_name} (Original WNS: {orig_wns:+.4f}ns)...")

        # Generate fix
        try:
            gen_rtl = generate_fix(model, tokenizer, orig_rtl, orig_sta)
        except Exception as e:
            print(f"  Generation error: {e}")
            continue

        if "module" in gen_rtl:
            syntax_passed += 1
        else:
            print("  Syntax check FAILED (no module declaration found)")
            continue

        # Extract top module name
        m = re.search(r"module\s+(\w+)", gen_rtl)
        top_mod = m.group(1) if m else "top"
        stem = f"eval_{i}_{top_mod}"

        clk_match = re.search(r'\b(clk|clock|CLK|Clock|Clk)\b', gen_rtl)
        clk_port = clk_match.group(1) if clk_match else "clk"

        module_info = {
            "module_name": top_mod,
            "stem": stem,
            "filepath": str(WORKDIR / f"{stem}.v"),
            "is_sv": True,
            "has_clock": clk_match is not None,
            "clock_port_name": clk_port,
            "rtl_source": gen_rtl,
        }

        # Write generated RTL to disk for Yosys
        (WORKDIR / f"{stem}.v").write_text(gen_rtl)

        # Synthesize
        synth_res = eda_runner.synthesize(module_info, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, WORKDIR)
        if not synth_res["success"]:
            print(f"  Yosys Synthesis FAILED: {synth_res['error'][:100]}")
            continue

        synth_passed += 1

        if not clk_match:
            print("  Combinational module — synthesis passed.")
            continue

        # STA
        sta_res = eda_runner.run_sta(module_info, synth_res, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, orig_sta.get("clock_period_ns", 1.0), WORKDIR)
        if not sta_res.get("success"):
            print("  OpenSTA FAILED.")
            continue

        new_wns = sta_res.get("wns_ns", -10.0)
        wns_delta = new_wns - orig_wns
        wns_improvements.append(wns_delta)

        if new_wns >= 0.0:
            timing_closed += 1

        print(f"  STA OK: New WNS = {new_wns:+.4f}ns (Delta = {wns_delta:+.4f}ns)")

        # EQY check
        eqy_res = eda_runner.run_eqy(orig_rtl, gen_rtl, module_info, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, WORKDIR)
        if eqy_res["equivalent"]:
            equiv_passed += 1
            print("  EQY Formal Equivalence: PASSED")
        else:
            print("  EQY Formal Equivalence: FAILED or latency modified")

    # Summary
    avg_wns_imp = sum(wns_improvements) / len(wns_improvements) if wns_improvements else 0.0

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print(f"  Total Samples Evaluated  : {total_samples}")
    print(f"  Syntax Pass Rate        : {syntax_passed}/{total_samples} ({syntax_passed/total_samples*100:.1f}%)")
    print(f"  Synthesis Pass Rate     : {synth_passed}/{total_samples} ({synth_passed/total_samples*100:.1f}%)")
    print(f"  Average WNS Improvement : {avg_wns_imp:+.4f} ns")
    print(f"  Timing Closure Rate     : {timing_closed}/{total_samples} ({timing_closed/total_samples*100:.1f}%)")
    print(f"  Formal Equivalence Rate : {equiv_passed}/{total_samples} ({equiv_passed/total_samples*100:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
