#!/usr/bin/env python3
"""
prepare_sft_dataset.py — Prepare Dataset for Supervised Fine-Tuning (SFT)

Reads `dataset/generated_dataset/timing_dataset.jsonl` and formats each entry into
ChatML standard format suitable for HuggingFace `datasets` and `trl.SFTTrainer`.

Output: `dataset/generated_dataset/sft_dataset.jsonl`
"""

import json
import random
import argparse
import collections
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
    parser.add_argument("--input", default="dataset/generated_dataset/timing_dataset.jsonl",
                        help="One or more JSONL files (comma-separated) to merge")
    parser.add_argument("--output", default="dataset/generated_dataset/sft_dataset.jsonl")
    parser.add_argument("--min-improvement-ns", type=float, default=0.05,
                        help="drop records whose fix does not improve WNS by at least "
                             "this much — a non-improving 'fix' is negative training signal")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="cap how many records of the same mutation_type are kept. "
                             "A model trained on a dataset dominated by one homogeneous "
                             "family (confirmed: 78%% one family) learns to shortcut on "
                             "its surface pattern instead of the actual input it's given.")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed for --max-per-type")
    args = parser.parse_args()

    in_paths = [Path(p.strip()) for p in args.input.split(",") if p.strip()]
    missing = [p for p in in_paths if not p.exists()]
    if missing:
        print(f"Error: input file(s) not found: {', '.join(str(m) for m in missing)}")
        return
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped_nonequiv = 0
    skipped_noimprove = 0
    seen = set()
    accepted: list = []  # entries that passed the equivalence/improvement gates
    for in_path in in_paths:
        for line in in_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # 1. Never train on a fix EQY proved changes behaviour.
            if entry.get("equivalence_verified") is False:
                skipped_nonequiv += 1
                continue

            # 2. Never train on a "fix" that doesn't actually help — a
            #    neutral or timing-worsening rewrite is negative signal
            #    and is the main reason earlier models learnt nothing.
            if entry.get("timing_improvement_ns", 0.0) < args.min_improvement_ns:
                skipped_noimprove += 1
                continue

            # de-dupe identical (prompt, response) pairs across merged files
            key = (entry["original_rtl"], entry["fixed_rtl"])
            if key in seen:
                continue
            seen.add(key)
            accepted.append(entry)

    skipped_cap = 0
    if args.max_per_type:
        by_type: dict = collections.defaultdict(list)
        for e in accepted:
            by_type[e.get("mutation_type", "unknown")].append(e)
        rng = random.Random(args.seed)
        capped = []
        for mtype, group in by_type.items():
            rng.shuffle(group)
            capped.extend(group[:args.max_per_type])
            skipped_cap += max(0, len(group) - args.max_per_type)
        rng.shuffle(capped)
        accepted = capped

    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for entry in accepted:
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

    print(f"Wrote {count} SFT records to {out_path}")
    print(f"  skipped {skipped_nonequiv} (EQY proved non-equivalent), "
          f"{skipped_noimprove} (WNS improvement < {args.min_improvement_ns}ns)"
          + (f", {skipped_cap} (over --max-per-type {args.max_per_type})" if args.max_per_type else ""))


if __name__ == "__main__":
    main()
