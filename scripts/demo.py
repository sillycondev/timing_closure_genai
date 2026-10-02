#!/usr/bin/env python3
"""
demo.py — one-module, single-shot demo of the timing-closure model.

    1. Yosys + OpenSTA on the original RTL   -> baseline WNS
    2. model generates ONE rewrite (greedy)
    3. Yosys + OpenSTA on the rewrite        -> new WNS, delta
    4. print the diff

Usage:
    python3 scripts/demo.py rtl/demo_adder_chain.v
    python3 scripts/demo.py rtl/demo_adder_chain.v --model models/sft_lora_checkpoint --clock-ghz 1.0
"""

import re
import sys
import json
import time
import difflib
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner
from prepare_sft_dataset import format_user_prompt, SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDA_CONFIG = yaml.safe_load((PROJECT_ROOT / "scripts" / "config.yaml").read_text())
JINJA_ENV = Environment(loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
                        keep_trailing_newline=True)
WORKDIR = PROJECT_ROOT / "dataset" / "workdir_demo"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

BOLD, DIM, GREEN, RED, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m"


def banner(text: str):
    print(f"\n{BOLD}{CYAN}{'=' * 72}\n  {text}\n{'=' * 72}{RESET}")


def module_name(rtl: str) -> str:
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)", rtl)
    if not m:
        sys.exit("could not find a module declaration")
    return m.group(1)


def run_flow(rtl: str, mod: str, stem: str, period: float) -> dict:
    """Yosys synth + OpenSTA on one RTL string. Returns parsed timing report."""
    src = WORKDIR / f"{stem}.v"
    src.write_text(rtl)
    clk = eda_runner.detect_rtl_clock_port(rtl) or "clk"
    info = {"module_name": mod, "stem": stem, "is_sv": True, "has_clock": True,
            "clock_port_name": clk, "rtl_source": rtl, "filepath": str(src)}
    t0 = time.time()
    synth = eda_runner.synthesize(info, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, WORKDIR)
    if not synth["success"]:
        print(f"  {RED}Yosys synthesis FAILED{RESET}\n{synth['error'][-800:]}")
        return {"success": False}
    print(f"  Yosys synthesis OK   ({synth['cell_count']} Sky130 cells, {time.time() - t0:.1f}s)")
    t0 = time.time()
    sta = eda_runner.run_sta(info, synth, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, period, WORKDIR)
    if not sta.get("success"):
        print(f"  {RED}OpenSTA FAILED{RESET}: {sta.get('error')}")
        return {"success": False}
    cp = sta.get("critical_path", {})
    colour = GREEN if sta["wns_ns"] >= 0 else RED
    print(f"  OpenSTA OK           ({time.time() - t0:.1f}s)")
    print(f"  WNS = {colour}{BOLD}{sta['wns_ns']:+.3f} ns{RESET}   TNS = {sta.get('tns_ns', 0):+.2f} ns   "
          f"logic depth = {len(cp.get('path_cells', []))} cells")
    print(f"  critical path: {cp.get('startpoint', '?')}  ->  {cp.get('endpoint', '?')}")
    sta["cell_count"] = synth["cell_count"]
    return sta


def load_model(adapter: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tok


def generate_fix(model, tok, rtl: str, report: dict) -> str:
    import torch
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_prompt(rtl, report)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    plen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    m = re.search(r"```verilog\s*(.*?)\s*```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rtl", help="Verilog file to optimise")
    ap.add_argument("--model", default="models/sft_lora_checkpoint")
    ap.add_argument("--clock-ghz", type=float, default=1.0)
    ap.add_argument("--no-eqy", action="store_true", help="skip formal equivalence check")
    args = ap.parse_args()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.clock_ghz
    rtl = Path(args.rtl).read_text()
    mod = module_name(rtl)

    banner(f"STEP 1/4  Baseline timing: {args.rtl}  ({mod}) @ {args.clock_ghz:g} GHz")
    print(f"{DIM}{rtl.strip()}{RESET}\n")
    base = run_flow(rtl, mod, f"{mod}_orig", period)
    if not base["success"]:
        sys.exit(1)

    banner(f"STEP 2/4  Loading model  ({args.model})")
    t0 = time.time()
    model, tok = load_model(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    banner("STEP 3/4  Generating optimised RTL")
    t0 = time.time()
    fixed = generate_fix(model, tok, rtl, base)
    print(f"  generated in {time.time() - t0:.1f}s\n")
    print(f"{GREEN}{fixed}{RESET}")
    out_path = WORKDIR / f"{mod}_fixed.v"
    out_path.write_text(fixed + "\n")

    banner("STEP 4/4  Verifying the rewrite with Yosys + OpenSTA")
    new = run_flow(fixed, mod, f"{mod}_fixed", period)
    if not new["success"]:
        sys.exit(1)

    eqy_line = "skipped"
    if not args.no_eqy:
        info = {"module_name": mod, "stem": f"{mod}_fixed", "is_sv": True,
                "clock_port_name": eda_runner.detect_rtl_clock_port(rtl) or "clk"}
        try:
            eqy = eda_runner.run_eqy(rtl, fixed, info, EDA_CONFIG, JINJA_ENV, PROJECT_ROOT, WORKDIR)
            eqy_line = (f"{GREEN}PASS{RESET}" if eqy.get("equivalent")
                        else f"{RED}not proven{RESET} (latency changed, or eqy unavailable)")
        except Exception as e:  # eqy not installed etc.
            eqy_line = f"unavailable ({e.__class__.__name__})"

    delta = new["wns_ns"] - base["wns_ns"]
    banner("RESULT")
    print(f"  {'':<22}{'original':>12}{'optimised':>12}")
    print(f"  {'WNS (ns)':<22}{base['wns_ns']:>+12.3f}{new['wns_ns']:>+12.3f}")
    print(f"  {'TNS (ns)':<22}{base.get('tns_ns', 0):>+12.2f}{new.get('tns_ns', 0):>+12.2f}")
    print(f"  {'critical path cells':<22}{len(base.get('critical_path', {}).get('path_cells', [])):>12}"
          f"{len(new.get('critical_path', {}).get('path_cells', [])):>12}")
    print(f"  {'Sky130 cell count':<22}{base['cell_count']:>12}{new['cell_count']:>12}")
    colour = GREEN if delta > 0 else RED
    print(f"\n  WNS improvement : {colour}{BOLD}{delta:+.3f} ns{RESET}")
    print(f"  timing closed   : {'yes' if new['wns_ns'] >= 0 else 'no'}")
    print(f"  formal equiv    : {eqy_line}")
    print(f"  fixed RTL saved : {out_path.relative_to(PROJECT_ROOT)}")

    banner("DIFF  original -> optimised")
    for line in difflib.unified_diff(rtl.strip().splitlines(), fixed.splitlines(),
                                     "original", "optimised", lineterm=""):
        c = GREEN if line.startswith("+") else RED if line.startswith("-") else DIM
        print(f"{c}{line}{RESET}")


if __name__ == "__main__":
    main()
