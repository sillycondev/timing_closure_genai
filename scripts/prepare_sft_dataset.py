#!/usr/bin/env python3
"""
prepare_sft_dataset.py — Prepare Dataset for Supervised Fine-Tuning (SFT)

Reads `dataset/generated_dataset/timing_dataset.jsonl` and formats each entry into
ChatML standard format suitable for HuggingFace `datasets` and `trl.SFTTrainer`.

Output: `dataset/generated_dataset/sft_dataset.jsonl`
"""

import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert VLSI timing closure engineer. Given Verilog RTL and an OpenSTA "
    "static timing analysis report showing timing violations, modify the RTL to eliminate "
    "negative slack while maintaining functional correctness."
)


def format_user_prompt(original_rtl: str, timing_report: dict) -> str:
    wns = timing_report.get("wns_ns", 0.0)
    tns = timing_report.get("tns_ns", 0.0)
    period = timing_report.get("clock_period_ns", 1.0)
    cp = timing_report.get("critical_path", {})

    sp = cp.get("startpoint", "Unknown")
    ep = cp.get("endpoint", "Unknown")
    cells = " -> ".join(cp.get("path_cells", [])[:8])

    prompt = (
        f"## Original Verilog RTL\n"
        f"```verilog\n{original_rtl.strip()}\n```\n\n"
        f"## OpenSTA Static Timing Analysis Report\n"
        f"- Target Clock Period: {period:.2f} ns\n"
        f"- Worst Negative Slack (WNS): {wns:+.4f} ns\n"
        f"- Total Negative Slack (TNS): {tns:+.4f} ns\n"
        f"- Critical Path Startpoint: {sp}\n"
        f"- Critical Path Endpoint: {ep}\n"
        f"- Critical Path Gate Sequence: {cells}\n\n"
        f"Please rewrite the Verilog RTL to optimize combinational logic depth and eliminate negative slack."
    )
    return prompt


def format_assistant_response(fixed_rtl: str) -> str:
    return f"```verilog\n{fixed_rtl.strip()}\n```"


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT Dataset")
    parser.add_argument("--input", default="dataset/generated_dataset/timing_dataset.jsonl")
    parser.add_argument("--output", default="dataset/generated_dataset/sft_dataset.jsonl")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Error: Input file {in_path} does not exist.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            user_prompt = format_user_prompt(entry["original_rtl"], entry["timing_report"])
            assistant_resp = format_assistant_response(entry["fixed_rtl"])

            sft_record = {
                "id": entry.get("id", f"sample_{count}"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_resp},
                ],
                "timing_improvement_ns": entry.get("timing_improvement_ns", 0.0),
                "mutation_type": entry.get("mutation_type", "unknown"),
            }

            fout.write(json.dumps(sft_record) + "\n")
            count += 1

    print(f"Successfully converted {count} timing records into SFT ChatML dataset at {out_path}")


if __name__ == "__main__":
    main()
