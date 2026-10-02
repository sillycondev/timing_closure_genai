#!/usr/bin/env python3
"""
identity_generate.py — "already meets timing, do nothing" training examples.

Every record built by synth_corpus_generate.py / eda_runner.py teaches the
model "given a violation, transform the RTL." None teach the opposite lesson:
when the design already meets timing, the correct output is the *same* RTL,
unchanged. Confirmed missing during evaluation (see the RefModule 512-bit
feedback-function case): a checkpoint applied its learned pipeline-insert
pattern to an already-passing design and turned +0.0ns slack into -8.74ns,
purely because it had never seen a "no-op is correct" example.

For a small, deliberately-safe set of (template, width, param) instances,
this generates only the flat version, synthesizes + times it for real, and
keeps it only if it genuinely meets timing (WNS >= 0). The record's
"fixed_rtl" is byte-identical to "original_rtl" — the correct answer is
provably correct by construction, no EQY needed.

Usage:
    python3 scripts/identity_generate.py
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner
from synth_corpus_generate import TEMPLATES

logger = logging.getLogger("identity_generate")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Deliberately tiny parameter points, well under any width that violated in
# synth_corpus_generate.py's own sweep, so the flat version reliably meets
# timing at the same 1 GHz target used everywhere else in this project.
SAFE_GRID: List[Tuple[str, int, int]] = [
    ("mac", 2, 2), ("mac", 4, 2), ("mac", 2, 3), ("mac", 4, 3),
    ("adder_tree", 4, 4), ("adder_tree", 8, 4), ("adder_tree", 4, 6),
    ("adder_tree", 8, 6), ("adder_tree", 16, 4),
    ("horner", 2, 2), ("horner", 4, 2), ("horner", 2, 3),
    ("compare_chain", 8, 2), ("compare_chain", 16, 2), ("compare_chain", 24, 2),
    ("compare_chain", 8, 3), ("compare_chain", 16, 3),
    ("priority_mux", 8, 8), ("priority_mux", 8, 12), ("priority_mux", 16, 8),
    ("priority_mux", 24, 8),
    ("shift_acc", 4, 3), ("shift_acc", 8, 3), ("shift_acc", 4, 4), ("shift_acc", 8, 4),
]


def _jinja():
    return Environment(
        loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
        keep_trailing_newline=True,
    )


def verify_identity(mod: str, clk: str, flat_src: str, config: dict,
                    workdir: Path, period: float, family: str):
    jinja_env = _jinja()
    info = {"module_name": mod, "stem": mod, "is_sv": False, "has_clock": True,
           "clock_port_name": clk}
    (workdir / mod).mkdir(parents=True, exist_ok=True)
    (workdir / f"{mod}_flat.v").write_text(flat_src)
    flat_info = dict(info, rtl_source=flat_src, filepath=str(workdir / f"{mod}_flat.v"))
    fs = eda_runner.synthesize(flat_info, config, jinja_env, PROJECT_ROOT, workdir,
                               rtl_path=str(workdir / f"{mod}_flat.v"), suffix="_flat")
    if not fs["success"]:
        return {"module": mod, "note": "synth failed"}
    sta = eda_runner.run_sta(flat_info, fs, config, jinja_env, PROJECT_ROOT,
                             period, workdir, suffix="_flat")
    if not sta.get("success"):
        return {"module": mod, "note": "STA failed"}
    sta["clock_period_ns"] = period
    wns = sta.get("wns_ns", -1.0)
    if wns < 0:
        return {"module": mod, "note": f"still violates (WNS={wns:+.2f}) -- not a "
                                       "safe identity example"}

    record = {
        "id": f"{mod}__identity_000",
        "original_rtl": flat_src,
        "timing_report": {k: v for k, v in sta.items() if k != "raw_report"},
        "fixed_rtl": flat_src,  # unchanged, by construction
        "fixed_timing_report": {k: v for k, v in sta.items() if k != "raw_report"},
        "timing_improvement_ns": 0.0,
        "equivalence_verified": True,  # trivially -- identical text
        "mutation_type": "identity_no_change",
        "mutation_description": "already meets timing target; correct action is no "
                                "RTL change",
        "latency_modified": False,
        "synthesis_cell_count": {"original": fs["cell_count"], "fixed": fs["cell_count"]},
    }
    return {"module": mod, "note": f"KEPT (meets timing, WNS={wns:+.2f})", "record": record}


def main():
    p = argparse.ArgumentParser(description="Generate 'already passing, don't touch it' "
                                             "training examples")
    p.add_argument("--config", default="scripts/config.yaml")
    p.add_argument("--output", default="dataset/generated_dataset/identity_dataset.jsonl")
    p.add_argument("--period-ns", type=float, default=3.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    workdir = PROJECT_ROOT / "dataset" / "workdir_identity"

    print(f"{len(SAFE_GRID)} candidate instance(s) to generate + verify")
    kept = []
    for i, (tname, w, n) in enumerate(SAFE_GRID):
        mod, clk, flat, _pipe = TEMPLATES[tname](seed=1000 + i, width=w, **(
            {"n_terms": n} if tname == "mac" else
            {"n_in": n} if tname == "adder_tree" else
            {"degree": n} if tname == "horner" else
            {"n_pairs": n} if tname == "compare_chain" else
            {"n_levels": n} if tname == "priority_mux" else
            {"shamt_bits": n}
        ))
        res = verify_identity(mod, clk, flat, config, workdir, args.period_ns, tname)
        print(f"  {mod:<40} {res['note']}")
        if "record" in res:
            kept.append(res["record"])

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(kept)}/{len(SAFE_GRID)} kept. Output: {out_path}")


if __name__ == "__main__":
    main()
