#!/usr/bin/env python3
"""
harvest_real_fixes.py — rejection sampling on real modules, verified end to end.

For each real violating module in the train split, sample N rewrites from the
current checkpoint, and keep the best one that (a) preserves the interface,
(b) synthesizes and measurably improves WNS, and (c) passes the
latency-tolerant functional check. Kept pairs are written in the same record
shape as timing_dataset.jsonl so prepare_sft_dataset.py can merge them.

This is the only data source that teaches the model on RTL it did not
template itself; yield is expected to be low and every kept record is
tool-verified — nothing here relies on the model being right.

Usage:
    python3 scripts/harvest_real_fixes.py --pool dataset/generated_dataset/external_real_pool.jsonl \
        --exclude-repos verilog-axi zipcpu --n 8 --temperature 0.8
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_bestofn as bo

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/sft_lora_checkpoint")
    p.add_argument("--pool", default="dataset/generated_dataset/external_real_pool.jsonl")
    p.add_argument("--exclude-repos", nargs="*", default=[], help="held-out repos, never harvested")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="dataset/generated_dataset/real_harvest_dataset.jsonl")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="skip modules whose (header-stripped) RTL exceeds this many tokens")
    args = p.parse_args()

    pool = [json.loads(l) for l in (PROJECT_ROOT / args.pool).read_text().splitlines() if l.strip()]
    pool = [s for s in pool if s.get("repo") not in set(args.exclude_repos)
            and s["timing_report"]["wns_ns"] > -500]
    bo.WORKDIR = PROJECT_ROOT / "dataset" / "workdir_harvest"
    bo.WORKDIR.mkdir(parents=True, exist_ok=True)
    out_path = PROJECT_ROOT / args.output

    model, tok = bo.load_model(args.model)
    if args.max_tokens:
        before = len(pool)
        pool = [s for s in pool if bo.rtl_tokens(tok, s["original_rtl"]) <= args.max_tokens]
        print(f"token filter <= {args.max_tokens}: kept {len(pool)}/{before} modules", flush=True)
    if args.limit:
        pool = pool[:args.limit]
    print(f"harvesting from {len(pool)} real modules (excluding repos {args.exclude_repos}), "
          f"N={args.n}, T={args.temperature}", flush=True)

    kept = 0
    with open(out_path, "w") as fh:
        for i, s in enumerate(pool):
            orig_wns = s["timing_report"]["wns_ns"]
            print(f"\n[{i+1}/{len(pool)}] {s['id']} (WNS {orig_wns:+.2f})", flush=True)
            cands = bo.generate_candidates(model, tok, s["original_rtl"], s["timing_report"],
                                           args.n, args.temperature)
            scored = []
            for j, c in enumerate(cands):
                r = bo.score_candidate(s, c, f"h{i}_c{j}_{s['module_name']}")
                if r["ok"] and r["wns"] > orig_wns + bo.EPS:
                    scored.append((j, c, r))
            scored.sort(key=lambda t: -t[2]["wns"])
            print(f"    {len(scored)}/{args.n} candidates improve timing", flush=True)
            for j, c, r in scored:
                method = bo.verify_function(s, c, r["info"], f"h{i}_c{j}_{s['module_name']}")
                if not method:
                    continue
                sta = {"wns_ns": r["wns"], "clock_period_ns": s["timing_report"].get("clock_period_ns", 1.0),
                       "success": True}
                fh.write(json.dumps({
                    "id": f"{s['id']}__harvest_000",
                    "original_rtl": s["original_rtl"],
                    "timing_report": s["timing_report"],
                    "fixed_rtl": c,
                    "fixed_timing_report": sta,
                    "timing_improvement_ns": r["wns"] - orig_wns,
                    "equivalence_verified": True,
                    "equivalence_method": method,
                    "mutation_type": "real_harvest_pipeline",
                    "mutation_description": "model-proposed rewrite of a real module, kept only "
                                            "after synth+STA improvement and latency-tolerant "
                                            "simulation equivalence",
                    "latency_modified": not method.endswith("_0"),
                    "source_repo": s.get("repo"),
                }) + "\n")
                fh.flush()
                kept += 1
                print(f"    KEPT candidate {j}: delta {r['wns'] - orig_wns:+.2f}ns, {method}", flush=True)
                break
            else:
                print("    no verified fix", flush=True)
    print(f"\nkept {kept}/{len(pool)} verified real pairs -> {out_path}")


if __name__ == "__main__":
    main()
