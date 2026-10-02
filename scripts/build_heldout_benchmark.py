#!/usr/bin/env python3
"""
build_heldout_benchmark.py — a held-out benchmark that can actually move.

The evaluation set used so far was the first N lines of timing_dataset.jsonl.
Ten of those twelve records have an original WNS between -0.31ns and 0.0ns —
there is almost nothing to fix, so even a perfect model would average only
~+0.5ns improvement on it. The metric was mostly measuring "did the model
avoid outliers," not "can it close timing."

This builds a benchmark of genuinely violating modules the model has never
seen:
  * synthetic_heldout — the same template families used for training, but
    at parameter points NOT in the training GRID and with fresh seeds (so
    the module/clock names differ too). Each is verified through the real
    flow and carries its reference fix's measured improvement as an
    "oracle" upper bound.
  * real_corpus — modules from the real RTL corpus whose original WNS is
    below a threshold and whose RTL text does not appear in the current
    SFT training set.

Usage:
    python3 scripts/build_heldout_benchmark.py
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

import eda_runner
from synth_corpus_generate import TEMPLATES, GRID, verify_pair

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Parameter points deliberately absent from synth_corpus_generate.GRID.
HELDOUT_GRID: List[Tuple[str, int, int]] = [
    ("mac", 12, 3), ("mac", 20, 4), ("mac", 40, 3),
    ("adder_tree", 12, 8), ("adder_tree", 40, 12), ("adder_tree", 72, 8),
    ("adder_tree", 160, 16), ("adder_tree", 320, 8),
    ("horner", 20, 3), ("horner", 40, 4),
    ("compare_chain", 40, 4), ("compare_chain", 72, 3), ("compare_chain", 160, 5),
    ("priority_mux", 40, 16), ("priority_mux", 72, 20),
    ("shift_acc", 40, 4), ("shift_acc", 72, 5), ("shift_acc", 160, 6), ("shift_acc", 320, 4),
]
assert not (set(HELDOUT_GRID) & set(GRID)), "held-out point collides with training GRID"


def _kw(tname: str, n: int) -> Dict[str, int]:
    return {"mac": {"n_terms": n}, "adder_tree": {"n_in": n}, "horner": {"degree": n},
            "compare_chain": {"n_pairs": n}, "priority_mux": {"n_levels": n},
            "shift_acc": {"shamt_bits": n}}[tname]


def training_rtl_set(sft_path: Path) -> set:
    """Original RTL of every record in the current SFT set, for exclusion."""
    seen = set()
    if not sft_path.exists():
        return seen
    for line in sft_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        m = re.search(r"## Original Verilog RTL\n```verilog\n(.*?)\n```", user, re.DOTALL)
        if m:
            seen.add(m.group(1).strip())
    return seen


def build_synthetic(config: dict, period: float) -> List[Dict[str, Any]]:
    workdir = PROJECT_ROOT / "dataset" / "workdir_heldout"
    out = []
    for i, (tname, w, n) in enumerate(HELDOUT_GRID):
        seed = 5000 + i
        mod, clk, flat, pipe = TEMPLATES[tname](seed=seed, width=w, **_kw(tname, n))
        res = verify_pair(mod, clk, flat, pipe, config, workdir, period,
                          min_violation_ns=-0.1, min_gap_ns=0.3, keep_rtl=False,
                          family=tname)
        print(f"  {mod:<40} {res['note']}")
        if "record" not in res:
            continue
        r = res["record"]
        out.append({
            "id": f"heldout_{r['id']}",
            "source": "synthetic_heldout",
            "family": tname,
            "module_name": mod,
            "clock_port_name": clk,
            "original_rtl": r["original_rtl"],
            "timing_report": r["timing_report"],
            "reference_fixed_rtl": r["fixed_rtl"],
            "reference_improvement_ns": r["timing_improvement_ns"],
        })
    return out


def build_real(max_real: int, min_violation: float, exclude: set) -> List[Dict[str, Any]]:
    src = PROJECT_ROOT / "dataset" / "generated_dataset" / "timing_dataset.jsonl"
    by_rtl: Dict[str, Dict[str, Any]] = {}
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        rtl = e["original_rtl"].strip()
        wns = e["timing_report"].get("wns_ns", 0.0)
        if wns > min_violation or rtl in exclude or rtl in by_rtl:
            continue
        # self-contained single-module files only: no dependency resolution
        # surprises when the model's rewrite is re-synthesized
        if len(re.findall(r"^\s*module\s+\w+", rtl, re.MULTILINE)) != 1:
            continue
        mod = re.search(r"module\s+(\w+)", rtl).group(1)
        clk = eda_runner.detect_rtl_clock_port(rtl)
        if not clk:
            continue
        by_rtl[rtl] = {
            "id": f"heldout_real_{mod}_{len(by_rtl)}",
            "source": "real_corpus",
            "family": "real",
            "module_name": mod,
            "clock_port_name": clk,
            "original_rtl": e["original_rtl"],
            "timing_report": e["timing_report"],
            "reference_fixed_rtl": None,
            "reference_improvement_ns": None,
        }
    # spread across distinct module names, worst violations first
    picked, names = [], set()
    for rec in sorted(by_rtl.values(), key=lambda r: r["timing_report"]["wns_ns"]):
        if rec["module_name"] in names:
            continue
        names.add(rec["module_name"])
        picked.append(rec)
        if len(picked) >= max_real:
            break
    for r in picked:
        print(f"  {r['module_name']:<40} WNS={r['timing_report']['wns_ns']:+.2f}")
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="scripts/config.yaml")
    p.add_argument("--output", default="dataset/generated_dataset/heldout_benchmark.jsonl")
    p.add_argument("--sft", default="dataset/generated_dataset/sft_dataset.jsonl")
    p.add_argument("--period-ns", type=float, default=1.0)
    p.add_argument("--max-real", type=int, default=10)
    p.add_argument("--min-violation-ns", type=float, default=-0.5)
    args = p.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    exclude = training_rtl_set(PROJECT_ROOT / args.sft)
    print(f"{len(exclude)} training RTL texts loaded for exclusion")

    print(f"\nSynthetic held-out ({len(HELDOUT_GRID)} candidate points):")
    synth = build_synthetic(config, args.period_ns)
    print(f"\nReal corpus (WNS <= {args.min_violation_ns}ns, not in training):")
    real = build_real(args.max_real, args.min_violation_ns, exclude)

    out_path = PROJECT_ROOT / args.output
    with open(out_path, "w") as f:
        for r in synth + real:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(synth)} synthetic + {len(real)} real = {len(synth) + len(real)} "
          f"benchmark samples to {out_path}")


if __name__ == "__main__":
    main()
