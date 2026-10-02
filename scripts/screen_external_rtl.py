#!/usr/bin/env python3
"""
screen_external_rtl.py — turn cloned open-source repos into a real-module pool.

The project's own corpus yielded only 23 self-contained, timing-violating real
modules — far too few to train or test on. This walks every .v file under
dataset/external_rtl/<repo>/, keeps the self-contained clocked ones (single
module, no submodule instantiation, no `include), synthesizes and times each
with the real flow at the project's 1 GHz target, and writes one record per
violating module in the same shape as timing_dataset.jsonl's inputs, tagged
with its source repo so train/held-out splits can be made per repo.

Usage:
    python3 scripts/screen_external_rtl.py --parallel 6
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXT = PROJECT_ROOT / "dataset" / "external_rtl"
WORKDIR = PROJECT_ROOT / "dataset" / "workdir_external"
MODULE_RE = re.compile(r"^\s*module\s+(\w+)", re.M)
INST_RE = re.compile(r"^\s*(\w+)\s*(?:#\s*\(.*?\))?\s+(\w+)\s*\(", re.M | re.S)
KEYWORDS = {"module", "input", "output", "inout", "wire", "reg", "assign", "always",
            "if", "else", "case", "for", "while", "initial", "begin", "end", "function",
            "task", "parameter", "localparam", "integer", "genvar", "generate", "logic"}


def _strip(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//.*", "", src)


def candidates() -> List[Dict[str, Any]]:
    out = []
    for repo in sorted(p for p in EXT.iterdir() if p.is_dir()):
        for f in sorted(repo.rglob("*.v")):
            parts = {x.lower() for x in f.relative_to(repo).parts}
            if parts & {"tb", "test", "tests", "testbench", "sim", "example", "examples", "fpga"}:
                continue
            src = f.read_text(encoding="utf-8", errors="replace")
            clean = _strip(src)
            mods = MODULE_RE.findall(clean)
            if len(mods) != 1 or "`include" in clean:
                continue
            body = clean[clean.find("module"):]
            insts = [m for m in INST_RE.findall(body)
                     if m[0] not in KEYWORDS and m[0] != mods[0] and not m[0].startswith("$")]
            if insts:
                continue
            clk = eda_runner.detect_rtl_clock_port(clean)
            if not clk:
                continue
            out.append({"repo": repo.name, "filepath": str(f), "filename": f.name,
                        "stem": f"{repo.name}__{f.stem}", "module_name": mods[0],
                        "is_sv": False, "has_clock": True, "clock_port_name": clk,
                        "rtl_source": src})
    return out


def _screen(args):
    info, config, period = args
    jinja = Environment(loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
                        keep_trailing_newline=True)
    synth = eda_runner.synthesize(info, config, jinja, PROJECT_ROOT, WORKDIR)
    if not synth["success"]:
        err = synth["error"].strip().splitlines()
        return info["stem"], None, "synth failed: " + (err[-1][:90] if err else "")
    sta = eda_runner.run_sta(info, synth, config, jinja, PROJECT_ROOT, period, WORKDIR)
    if not sta.get("success"):
        return info["stem"], None, "STA failed"
    sta["clock_period_ns"] = period
    wns = sta.get("wns_ns", 0.0)
    rec = {"id": f"ext_{info['stem']}", "source": "external_" + info["repo"], "repo": info["repo"],
           "family": "real", "module_name": info["module_name"],
           "clock_port_name": info["clock_port_name"], "original_rtl": info["rtl_source"],
           "timing_report": {k: v for k, v in sta.items() if k != "raw_report"},
           "cell_count": synth["cell_count"]}
    return info["stem"], rec, f"WNS {wns:+.2f}  cells {synth['cell_count']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="scripts/config.yaml")
    p.add_argument("--output", default="dataset/generated_dataset/external_real_pool.jsonl")
    p.add_argument("--period-ns", type=float, default=1.0)
    p.add_argument("--min-violation-ns", type=float, default=-0.5)
    p.add_argument("--parallel", type=int, default=6)
    args = p.parse_args()
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    config["synthesis"]["timeout_seconds"] = min(config["synthesis"].get("timeout_seconds", 300), 180)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    cands = candidates()
    by_repo: Dict[str, int] = {}
    for c in cands:
        by_repo[c["repo"]] = by_repo.get(c["repo"], 0) + 1
    print(f"{len(cands)} self-contained clocked candidates: {by_repo}", flush=True)

    kept, reasons = [], {}
    with ProcessPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(_screen, (c, config, args.period_ns)) for c in cands]
        for f in as_completed(futs):
            stem, rec, note = f.result()
            print(f"  {stem:<55} {note}", flush=True)
            if rec is None:
                reasons["fail"] = reasons.get("fail", 0) + 1
            elif rec["timing_report"]["wns_ns"] > args.min_violation_ns:
                reasons["meets timing"] = reasons.get("meets timing", 0) + 1
            else:
                kept.append(rec)
    kept.sort(key=lambda r: r["id"])
    with open(PROJECT_ROOT / args.output, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    per: Dict[str, int] = {}
    for r in kept:
        per[r["repo"]] = per.get(r["repo"], 0) + 1
    print(f"\nkept {len(kept)} violating real modules {per}; rejected {reasons}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
