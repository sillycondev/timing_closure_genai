#!/usr/bin/env python3
"""
deopt_generate.py — "break-then-fix" training-data generation.

Forward mutation has a low hit rate: Yosys/abc already balances expressions
during synthesis, so most source-level "fixes" change nothing (measured: only
~18% of forward-mutation records actually improve WNS). This script inverts
the problem.

For every module that *synthesizes* — whether it currently meets timing or
not — apply an INVERSE transform that Yosys cannot undo (mainly: fold two
pipeline stages into one combinational path). Re-synthesize + STA. If the
de-optimized version now genuinely violates timing and the untouched
original is meaningfully faster, emit a training record whose "fix" is the
real, human-written original source:

    original_rtl  <- de-optimized (slow) version   [what the model sees]
    fixed_rtl     <- the untouched original         [known-good, zero hallucination]
    improvement   <- measured (orig_wns - deopt_wns)

Every kept record is a genuine, tool-verified fix by construction.

Usage:
    python3 scripts/deopt_generate.py                       # whole corpus
    python3 scripts/deopt_generate.py --module fpu_add      # one module
    python3 scripts/deopt_generate.py --limit 20 --parallel 4
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner
from mutations import apply_inverse_mutations
from rejection_sample import _port_signature, _synth_is_clean, _ff_count

logger = logging.getLogger("deopt_generate")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULE_NAME = re.compile(r"\bmodule\s+(\w+)")


def _jinja():
    return Environment(
        loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
        keep_trailing_newline=True,
    )


def process_module(m: Dict[str, Any], config: dict, workdir: Path,
                   min_violation_ns: float, min_gap_ns: float,
                   max_per_module: int) -> Dict[str, Any]:
    """Synthesize the original, de-optimize it, verify, and emit records."""
    jinja_env = _jinja()
    name = m["module_name"]
    stem = m["stem"]
    out = {"records": [], "module": name, "note": ""}

    if not m["has_clock"]:
        out["note"] = "no clock"
        return out

    base = eda_runner.synthesize(m, config, jinja_env, PROJECT_ROOT, workdir)
    if not base["success"]:
        out["note"] = "orig synth failed"
        return out
    period = 1.0 / config["synthesis"].get("target_clock_ghz", [1.0])[0]
    base_sta = eda_runner.run_sta(m, base, config, jinja_env, PROJECT_ROOT,
                                  period, workdir)
    if not base_sta.get("success"):
        out["note"] = "orig STA failed"
        return out
    base_sta["clock_period_ns"] = period
    orig_wns = base_sta.get("wns_ns", 0.0)
    orig_ff = _ff_count(base.get("netlist_path"))
    orig_sig = _port_signature(m["rtl_source"])

    variants = apply_inverse_mutations(
        m["rtl_source"], config["mutations"]["enabled"],
        max_per_module, clock_port=m.get("clock_port_name") or "clk")
    if not variants:
        out["note"] = "no inverse transform applies"
        return out

    kept = 0
    for i, (tname, deopt_rtl, desc, lat_flag) in enumerate(variants):
        if kept >= max_per_module:
            break
        if _port_signature(deopt_rtl) not in (None, orig_sig):
            continue
        d_stem = f"{stem}__deopt{i}"
        dinfo = dict(m, stem=d_stem, rtl_source=deopt_rtl,
                     filepath=str(workdir / f"{d_stem}.v"))
        (workdir / d_stem).mkdir(parents=True, exist_ok=True)
        (workdir / f"{d_stem}.v").write_text(deopt_rtl)

        d_synth = eda_runner.synthesize(dinfo, config, jinja_env, PROJECT_ROOT,
                                        workdir, rtl_path=str(workdir / f"{d_stem}.v"))
        if not d_synth["success"] or not _synth_is_clean(d_synth.get("yosys_log", "")):
            continue
        d_sta = eda_runner.run_sta(dinfo, d_synth, config, jinja_env, PROJECT_ROOT,
                                   period, workdir)
        if not d_sta.get("success"):
            continue
        d_sta["clock_period_ns"] = period
        deopt_wns = d_sta.get("wns_ns", 0.0)

        # de-opt must genuinely break timing, and the original must be a
        # meaningfully better design (not just synthesis noise)
        if deopt_wns > min_violation_ns:
            continue
        gap = orig_wns - deopt_wns
        if gap < min_gap_ns:
            continue

        d_ff = _ff_count(d_synth.get("netlist_path"))
        latency_modified = bool(lat_flag) or (orig_ff >= 0 and d_ff >= 0 and d_ff != orig_ff)

        if latency_modified:
            equivalence_verified: Optional[bool] = None
        else:
            eqy = eda_runner.run_eqy(deopt_rtl, m["rtl_source"], dinfo,
                                     config, jinja_env, PROJECT_ROOT, workdir)
            if eqy["equivalent"]:
                equivalence_verified = True
            else:
                # a supposedly combinational inverse transform that fails EQY
                # is a transform bug — drop it rather than emit bad data
                continue

        out["records"].append({
            "id": f"{name}__deopt_{tname}_{i:03d}",
            "original_rtl": deopt_rtl,
            "timing_report": {k: v for k, v in d_sta.items() if k != "raw_report"},
            "fixed_rtl": m["rtl_source"],
            "fixed_timing_report": {k: v for k, v in base_sta.items() if k != "raw_report"},
            "timing_improvement_ns": gap,
            "equivalence_verified": equivalence_verified,
            "mutation_type": f"deopt_{tname}",
            "mutation_description": f"break-then-fix: {desc}; fix is the original source",
            "latency_modified": latency_modified,
            "synthesis_cell_count": {"original": d_synth["cell_count"],
                                     "fixed": base["cell_count"]},
        })
        kept += 1

    out["note"] = f"{kept} record(s)"
    return out


def main():
    p = argparse.ArgumentParser(description="Break-then-fix timing-closure data generation")
    p.add_argument("--config", default="scripts/config.yaml")
    p.add_argument("--output", default="dataset/generated_dataset/deopt_dataset.jsonl")
    p.add_argument("--module", type=str, help="only this module")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--min-violation-ns", type=float, default=-0.1,
                   help="the de-optimized version must have WNS <= this")
    p.add_argument("--min-gap-ns", type=float, default=0.3,
                   help="original must beat the de-optimized version by at least this")
    p.add_argument("--max-per-module", type=int, default=4)
    p.add_argument("--force", action="store_true", help="ignore the .progress file")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    workdir = PROJECT_ROOT / "dataset" / "workdir_deopt"
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress.json")
    progress = json.loads(progress_path.read_text()) if (progress_path.exists() and not args.force) else {}

    mods = eda_runner.discover_modules(
        str(PROJECT_ROOT / config["paths"]["raw_rtl"]),
        config["dataset"].get("skip_patterns", []))
    if args.module:
        mods = [x for x in mods if x["module_name"] == args.module or x["filename"] == args.module]
    mods = [x for x in mods if x["module_name"] not in progress]
    if args.limit:
        mods = mods[:args.limit]
    logger.info(f"{len(mods)} module(s) to process ({len(progress)} already done)")
    if not mods:
        return

    t0 = time.time()
    total = 0

    def _handle(res):
        nonlocal total
        for rec in res["records"]:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        total += len(res["records"])
        progress[res["module"]] = {"done": True, "kept": len(res["records"])}
        progress_path.write_text(json.dumps(progress, indent=2))
        logger.info(f"  {res['module']:40s} {res['note']}")

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(process_module, m, config, workdir,
                                args.min_violation_ns, args.min_gap_ns,
                                args.max_per_module): m for m in mods}
            for fut in as_completed(futs):
                try:
                    _handle(fut.result())
                except Exception as e:
                    logger.error(f"  {futs[fut]['module']}: {e}")
    else:
        for m in mods:
            try:
                _handle(process_module(m, config, workdir, args.min_violation_ns,
                                       args.min_gap_ns, args.max_per_module))
            except Exception as e:
                logger.error(f"  {m['module_name']}: {e}")

    logger.info(f"\n{'='*60}\nDone in {time.time()-t0:.0f}s. Records: {total}. "
                f"Output: {out_path}\nMerge with prepare_sft_dataset.py --input "
                f"dataset/generated_dataset/timing_dataset.jsonl,{args.output}")


if __name__ == "__main__":
    main()
