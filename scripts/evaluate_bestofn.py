#!/usr/bin/env python3
"""
evaluate_bestofn.py — best-of-N generation with tool verification.

Single-shot evaluation takes whatever the model emits first and scores it,
so one bad rewrite can drag the average below zero (observed: one 512-bit
module turned +0.0ns into -8.74ns and pulled a whole 12-sample average
negative). This evaluator does what a real deployment would do:

  1. generate N candidate rewrites per module (1 greedy + N-1 sampled)
  2. reject any candidate that renames the module or changes its ports
  3. synthesize + time every surviving candidate with the real flow
  4. walk the improving candidates best-WNS-first and keep the first one
     that is FUNCTIONALLY verified — latency-tolerant simulation
     equivalence (latency_equiv.py), or EQY when the simulation is
     inconclusive; if none passes, fall back to the ORIGINAL RTL (delta 0)

Because of (4) the per-sample delta is >= 0 by construction — the outlier
class cannot occur — and the average turns positive wherever any of the N
tries succeeds. The functional gate is what makes the number trustworthy:
without it, three of four "improved" real modules in the first run were
timing-verified but functionally broken rewrites. The greedy candidate
alone is also scored without fallback (single-shot baseline), and the
ungated best-WNS choice is reported alongside for comparison.

Usage:
    python3 scripts/evaluate_bestofn.py --benchmark dataset/generated_dataset/heldout_benchmark.jsonl --n 4
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner
from latency_equiv import check_latency_equiv
from prepare_sft_dataset import format_user_prompt, SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDA_CONFIG = yaml.safe_load((PROJECT_ROOT / "scripts" / "config.yaml").read_text())
JINJA_ENV = Environment(loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
                        keep_trailing_newline=True)
WORKDIR = PROJECT_ROOT / "dataset" / "workdir_bestofn"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EPS = 0.01  # ns — an "improvement" smaller than STA noise is not accepted


def load_model(adapter: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter) if (Path(adapter) / "adapter_config.json").exists() else base
    model.eval()
    return model, tok


def _extract(response: str) -> str:
    m = re.search(r"```(?:verilog|systemverilog|v)?\s*\n(.*?)(?:```|\Z)", response, re.DOTALL)
    return (m.group(1) if m else response).strip()


def strip_header(rtl: str) -> str:
    """Drop leading license/banner comments — function-free tokens the model
    would otherwise have to reproduce inside its generation budget."""
    return re.sub(r"\A(\s*(//[^\n]*\n|/\*.*?\*/\s*))+", "", rtl, flags=re.DOTALL)


def rtl_tokens(tok, rtl: str) -> int:
    return len(tok(strip_header(rtl)).input_ids)


def generate_candidates(model, tok, original_rtl: str, timing_report: dict,
                        n: int, temperature: float) -> List[str]:
    src = strip_header(original_rtl)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_prompt(src, timing_report)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    plen = inputs.input_ids.shape[1]
    # A rewrite is the whole module plus a few registers; real modules run to
    # thousands of tokens, and a fixed 1024 cap truncated 31 of 33 real
    # held-out rewrites mid-file. Candidates are generated one at a time —
    # a 7-sequence batch on a long prompt exhausted the 8 GB GPU.
    budget = int(min(max(len(tok(src).input_ids) * 1.3 + 256, 768), 6144))
    outs: List[str] = []
    with torch.no_grad():
        greedy = model.generate(**inputs, max_new_tokens=budget, do_sample=False)
        outs.append(_extract(tok.decode(greedy[0][plen:], skip_special_tokens=True)))
        for _ in range(n - 1):
            s = model.generate(**inputs, max_new_tokens=budget, do_sample=True,
                               temperature=temperature, top_p=0.95)
            outs.append(_extract(tok.decode(s[0][plen:], skip_special_tokens=True)))
            torch.cuda.empty_cache()
    return outs


def _ports(rtl: str, mod: str) -> Optional[frozenset]:
    m = re.search(rf"module\s+{re.escape(mod)}\s*(?:#\s*\(.*?\))?\s*\((.*?)\)\s*;", rtl, re.DOTALL)
    if not m:
        return None
    names = re.findall(r"\b([A-Za-z_]\w*)\s*(?:,|$)", m.group(1).replace("\n", " ").strip() + ",")
    return frozenset(names)


def interface_matches(orig: str, cand: str, mod: str) -> bool:
    if not re.search(rf"\bmodule\s+{re.escape(mod)}\b", cand):
        return False
    po, pc = _ports(orig, mod), _ports(cand, mod)
    return po is not None and pc is not None and po == pc


def score_candidate(sample: dict, cand: str, stem: str) -> Dict[str, Any]:
    mod = sample["module_name"]
    if not interface_matches(sample["original_rtl"], cand, mod):
        return {"ok": False, "why": "module name / port list changed"}
    clk = sample.get("clock_port_name") or eda_runner.detect_rtl_clock_port(cand) or "clk"
    info = {"module_name": mod, "stem": stem, "is_sv": True, "has_clock": True,
            "clock_port_name": clk, "rtl_source": cand,
            "filepath": str(WORKDIR / f"{stem}.v")}
    (WORKDIR / f"{stem}.v").write_text(cand)
    synth = eda_runner.synthesize(info, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, WORKDIR)
    if not synth["success"]:
        return {"ok": False, "why": "synthesis failed"}
    period = sample["timing_report"].get("clock_period_ns", 1.0)
    sta = eda_runner.run_sta(info, synth, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, period, WORKDIR)
    if not sta.get("success"):
        return {"ok": False, "why": "STA failed"}
    return {"ok": True, "wns": sta["wns_ns"], "info": info, "cells": synth["cell_count"]}


def verify_function(sample: dict, cand: str, info: dict, stem: str) -> Optional[str]:
    """Returns the verification method that accepted the rewrite, or None."""
    sim = check_latency_equiv(sample["original_rtl"], cand, sample["module_name"],
                              info["clock_port_name"], WORKDIR / stem / "latency_equiv")
    if sim.get("equivalent") is True and sim["shift"] > 0:
        return f"sim_shift_{sim['shift']}"
    if sim.get("equivalent") is False:
        return None
    # A shift-0 simulation pass claims cycle-exact equivalence, which EQY can
    # prove outright — require it, since random stimulus can agree vacuously
    # (a broken 64-bit down-counter passed 3000 cycles this way).
    eqy = eda_runner.run_eqy(sample["original_rtl"], cand, info, EDA_CONFIG, JINJA_ENV,
                             PROJECT_ROOT, WORKDIR)
    return "eqy" if eqy["equivalent"] else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/sft_lora_checkpoint")
    p.add_argument("--benchmark", default="dataset/generated_dataset/heldout_benchmark.jsonl")
    p.add_argument("--n", type=int, default=4, help="candidates per module (1 greedy + n-1 sampled)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="dataset/generated_dataset/bestofn_results.jsonl")
    p.add_argument("--reuse-candidates", action="store_true",
                   help="load candidate .v files from a previous run's workdir instead of generating")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="skip modules whose (header-stripped) RTL exceeds this many tokens")
    args = p.parse_args()

    samples = [json.loads(l) for l in (PROJECT_ROOT / args.benchmark).read_text().splitlines() if l.strip()]
    WORKDIR.mkdir(parents=True, exist_ok=True)

    model = tok = None
    if not args.reuse_candidates:
        model, tok = load_model(args.model)
    if args.max_tokens and tok is not None:
        before = len(samples)
        samples = [s for s in samples if rtl_tokens(tok, s["original_rtl"]) <= args.max_tokens]
        print(f"token filter <= {args.max_tokens}: kept {len(samples)}/{before} modules")
    if args.limit:
        samples = samples[:args.limit]

    print("=" * 72)
    print(f"Best-of-{args.n} evaluation  |  model={args.model}  |  {len(samples)} samples")
    print("=" * 72)

    rows = []
    for i, s in enumerate(samples):
        orig_wns = s["timing_report"]["wns_ns"]
        print(f"\n[{i+1}/{len(samples)}] {s['id']}  ({s['source']}, original WNS {orig_wns:+.2f}ns)")
        if args.reuse_candidates:
            cands = []
            for j in range(args.n):
                f = WORKDIR / f"b{i}_c{j}_{s['module_name']}.v"
                cands.append(f.read_text() if f.exists() else "")
        else:
            cands = generate_candidates(model, tok, s["original_rtl"], s["timing_report"],
                                        args.n, args.temperature)
        scored = []
        for j, c in enumerate(cands):
            r = score_candidate(s, c, f"b{i}_c{j}_{s['module_name']}")
            tag = "greedy" if j == 0 else f"sample{j}"
            if r["ok"]:
                print(f"    {tag:<8} WNS {r['wns']:+.2f}  (delta {r['wns'] - orig_wns:+.2f})")
            else:
                print(f"    {tag:<8} rejected: {r['why']}")
            scored.append((j, c, r))

        # single-shot baseline = greedy candidate, scored as-is, no fallback
        g = scored[0][2]
        single_delta = (g["wns"] - orig_wns) if g["ok"] else None

        improving = sorted([(j, c, r) for j, c, r in scored if r["ok"] and r["wns"] > orig_wns + EPS],
                           key=lambda t: -t[2]["wns"])
        ungated_delta = (improving[0][2]["wns"] - orig_wns) if improving else 0.0
        sel = None
        for j, c, r in improving:
            tag = "greedy" if j == 0 else f"sample{j}"
            method = verify_function(s, c, r["info"], f"b{i}_c{j}_{s['module_name']}")
            if method:
                sel = {"chosen": tag, "wns": r["wns"], "delta": r["wns"] - orig_wns,
                       "verified": method, "fallback": False}
                print(f"  -> selected {tag}: delta {sel['delta']:+.2f}ns, verified by {method}")
                break
            print(f"    {tag:<8} improved timing but FAILED functional check — rejected")
        if sel is None:
            sel = {"chosen": "original", "wns": orig_wns, "delta": 0.0,
                   "verified": "identity", "fallback": True}
            print("  -> no verified improvement; keeping original RTL (delta 0)")

        rows.append({"id": s["id"], "source": s["source"], "orig_wns": orig_wns,
                     "single_shot_delta": single_delta, "ungated_delta": ungated_delta,
                     "reference_improvement_ns": s.get("reference_improvement_ns"),
                     **sel})

    with open(PROJECT_ROOT / args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    ss_scored = [r["single_shot_delta"] for r in rows if r["single_shot_delta"] is not None]
    ungated = [r["ungated_delta"] for r in rows]
    gated = [r["delta"] for r in rows]
    picked = [r for r in rows if not r["fallback"]]
    closed = sum(1 for r in rows if r["wns"] >= 0.0)
    ref = [r["reference_improvement_ns"] for r in rows if r["reference_improvement_ns"] is not None]

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  samples                                  : {n}")
    print(f"  single-shot (greedy, no fallback)        : avg delta {avg(ss_scored):+.3f} ns  "
          f"[{len(ss_scored)}/{n} synthesized; rest excluded]")
    print(f"  best-of-{args.n} timing-only (no function gate): avg delta {avg(ungated):+.3f} ns")
    print(f"  best-of-{args.n} + FUNCTION GATE + fallback    : avg delta {avg(gated):+.3f} ns  "
          f"[{len(picked)}/{n} verified improvements, {n - len(picked)} kept original]")
    print(f"  timing closure rate (final WNS >= 0)     : {closed}/{n}")
    if picked:
        methods: Dict[str, int] = {}
        for r in picked:
            methods[r["verified"]] = methods.get(r["verified"], 0) + 1
        print(f"  verification methods among selected      : {methods}")
    if ref:
        print(f"  oracle (reference fix) avg delta         : {avg(ref):+.3f} ns over {len(ref)} synthetic samples")
    by_src: Dict[str, List[float]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r["delta"])
    for k, v in by_src.items():
        imp = sum(1 for r in rows if r["source"] == k and not r["fallback"])
        print(f"    {k:<18} gated avg delta {avg(v):+.3f} ns over {len(v)}  ({imp} verified improvements)")
    print("=" * 72)


if __name__ == "__main__":
    main()
