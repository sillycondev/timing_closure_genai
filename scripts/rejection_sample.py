#!/usr/bin/env python3
"""
rejection_sample.py — Build EDA-verified timing-closure training data by
sampling candidate fixes from an LLM and keeping only the ones that real
tools confirm are (a) synthesizable, (b) genuinely faster, and (c) not a
functional regression.

This sidesteps the regex mutation engine entirely: a proposer model is
asked to rewrite each timing-violating module N times; every candidate is
run through Yosys -> OpenSTA (-> EQY) and accepted only if it clears the
bar. Output rows use the same schema as eda_runner.py, so
prepare_sft_dataset.py consumes them unchanged.

Typical use:
    # discover every violating module in the corpus, 8 samples each,
    # propose with the base model
    python3 scripts/rejection_sample.py --discover --samples-per-module 8

    # propose with the already-trained SFT adapter (iterated self-improvement)
    python3 scripts/rejection_sample.py --discover --adapter models/sft_lora_checkpoint

    # quick smoke test on one module
    python3 scripts/rejection_sample.py --module Counter --samples-per-module 2 --limit 1

    # use Anthropic as the proposer (needs `pip install anthropic` + ANTHROPIC_API_KEY)
    python3 scripts/rejection_sample.py --discover --proposer anthropic
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner
from prepare_sft_dataset import format_user_prompt, SYSTEM_PROMPT

logger = logging.getLogger("rejection_sample")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VERILOG_BLOCK = re.compile(r"```(?:verilog|systemverilog)?\s*(.*?)\s*```", re.DOTALL)
_MODULE_NAME = re.compile(r"\bmodule\s+(\w+)")
_CLK_PIN = re.compile(r"\.CLK\s*\(")


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------
class Proposer:
    """Returns up to `n` candidate completions for one (system, user) prompt."""

    def generate(self, system: str, user: str, n: int) -> List[str]:
        raise NotImplementedError


class LocalProposer(Proposer):
    """transformers text-generation, optionally with a PEFT/LoRA adapter."""

    def __init__(self, model_id: str, adapter: Optional[str],
                 temperature: float, top_p: float, max_new_tokens: int):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        self.torch = torch
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        tok_src = adapter if (adapter and (Path(adapter) / "tokenizer_config.json").exists()) else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info(f"Loading local proposer: {model_id}"
                    + (f" + adapter {adapter}" if adapter else ""))
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0},
            trust_remote_code=True,
        )
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model

    def generate(self, system: str, user: str, n: int) -> List[str]:
        torch = self.torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=4096).to(self.model.device)
        outs: List[str] = []
        # generate in small batches to cap VRAM
        remaining = n
        while remaining > 0:
            b = min(remaining, 4)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_return_sequences=b,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for row in gen:
                text = self.tokenizer.decode(
                    row[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                outs.append(text)
            remaining -= b
            torch.cuda.empty_cache()
        return outs


class AnthropicProposer(Proposer):
    def __init__(self, model: str, temperature: float, max_tokens: int):
        try:
            import anthropic
        except ImportError:
            sys.exit("anthropic SDK not installed. `pip install anthropic` and set "
                     "ANTHROPIC_API_KEY, or use --proposer local.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str, n: int) -> List[str]:
        outs = []
        for _ in range(n):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                outs.append("".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text"))
            except Exception as e:
                logger.warning(f"anthropic call failed: {e}")
        return outs


# ---------------------------------------------------------------------------
# Candidate extraction / EDA verification
# ---------------------------------------------------------------------------
def extract_rtl(completion: str) -> Optional[str]:
    m = _VERILOG_BLOCK.search(completion)
    cand = (m.group(1) if m else completion).strip()
    if "module" not in cand or not _MODULE_NAME.search(cand):
        return None
    return cand


def _ff_count(netlist_path: Optional[str]) -> int:
    if not netlist_path or not Path(netlist_path).exists():
        return -1
    try:
        return len(_CLK_PIN.findall(Path(netlist_path).read_text()))
    except Exception:
        return -1


def _strip_comments(src: str) -> str:
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def _port_signature(src: str) -> Optional[frozenset]:
    """Return {(port_name, direction)} for the first module in `src`, or None.

    A real timing fix keeps the module interface intact; a hallucinated
    rewrite almost always drops, renames, or invents ports. Direction is
    taken from input/output/inout declarations (ANSI or non-ANSI style).
    """
    clean = _strip_comments(src)
    m = re.search(r"\bmodule\s+\w+\s*(#\s*\([^;]*?\))?\s*\((.*?)\)\s*;", clean, re.DOTALL)
    if not m:
        return None
    port_blob = m.group(2)
    names = set(re.findall(r"\b([A-Za-z_]\w*)\b", port_blob))
    # keywords that can appear in an ANSI port list
    names -= {"input", "output", "inout", "wire", "reg", "logic", "signed",
              "unsigned", "parameter"}
    dirs: Dict[str, str] = {}
    for dm in re.finditer(r"\b(input|output|inout)\b([^;()]*?)(?=(?:\binput\b|\boutput\b|"
                          r"\binout\b|;|\)\s*;))", clean, re.DOTALL):
        d = dm.group(1)
        for nm in re.findall(r"\b([A-Za-z_]\w*)\b", dm.group(2)):
            if nm in ("wire", "reg", "logic", "signed", "unsigned"):
                continue
            dirs[nm] = d
    sig = {(nm, dirs.get(nm, "?")) for nm in names if nm in dirs or names == set(dirs)}
    if not sig:
        sig = {(nm, dirs.get(nm, "?")) for nm in names}
    return frozenset(sig) or None


_SYNTH_WARN_RE = re.compile(
    r"multiple driv|multiply driven|Latch inferred|implicitly declared|"
    r"identifier .* is implicitly|found unknown|Warning: Wire .* is used but",
    re.IGNORECASE)


def _synth_is_clean(yosys_log: str) -> bool:
    return not _SYNTH_WARN_RE.search(yosys_log or "")


def verify_candidate(cand_rtl: str, base_info: Dict[str, Any],
                     orig_report: Dict[str, Any], orig_ff: int,
                     config: dict, jinja_env: Environment, workdir: Path,
                     idx: int, allow_latency_changes: bool,
                     min_improvement_ns: float,
                     reject_log: Optional[list] = None) -> Optional[Dict[str, Any]]:
    """Run gate checks -> synth -> STA -> EQY on one candidate.

    Acceptance (all required):
      1. top module present; port signature (name + direction) identical to original
      2. synthesis succeeds with no multi-driver / latch-inference / implicit-decl warnings
      3. STA succeeds and (new_wns - orig_wns) >= min_improvement_ns
      4. correctness: EQY proves equivalence  -- OR, only with --allow-latency-changes,
         the candidate changed flip-flop count (a pipelining/retiming fix EQY can't
         judge cycle-accurately) while keeping ports and a clean synth.
    A weak proposer that emits broken RTL should fail (1) or (2) and produce nothing.
    """
    def _rej(reason):
        if reject_log is not None:
            reject_log.append(reason)
        return None

    top_m = _MODULE_NAME.search(cand_rtl)
    if not top_m:
        return _rej("no module")
    top = top_m.group(1)

    orig_sig = _port_signature(base_info["rtl_source"])
    cand_sig = _port_signature(cand_rtl)
    if orig_sig and cand_sig and orig_sig != cand_sig:
        return _rej("port signature changed")

    clk_m = re.search(r"\b(clk\w*|\w*clk|clock\w*|\w*clock)\b", cand_rtl, re.IGNORECASE)
    stem = f"{base_info['stem']}__rej{idx}"
    mod_info = {
        "module_name": top,
        "stem": stem,
        "filepath": str(workdir / f"{stem}.v"),
        "is_sv": True,
        "has_clock": clk_m is not None,
        "clock_port_name": (clk_m.group(1) if clk_m else base_info.get("clock_port_name")),
        "rtl_source": cand_rtl,
    }
    (workdir / stem).mkdir(parents=True, exist_ok=True)
    (workdir / f"{stem}.v").write_text(cand_rtl)

    synth = eda_runner.synthesize(mod_info, config, jinja_env, PROJECT_ROOT,
                                  workdir, rtl_path=str(workdir / f"{stem}.v"))
    if not synth["success"]:
        return _rej("synth failed")
    if not _synth_is_clean(synth.get("yosys_log", "")):
        return _rej("synth warnings (multi-driver / latch / implicit decl)")

    period = orig_report.get("clock_period_ns", 1.0)
    sta = eda_runner.run_sta(mod_info, synth, config, jinja_env, PROJECT_ROOT,
                             period, workdir)
    if not sta.get("success"):
        return _rej("STA failed")

    orig_wns = orig_report.get("wns_ns", 0.0)
    delta = sta.get("wns_ns", -99.0) - orig_wns
    if delta < min_improvement_ns:
        return _rej(f"insufficient improvement ({delta:+.3f}ns)")

    cand_ff = _ff_count(synth.get("netlist_path"))
    latency_modified = (orig_ff >= 0 and cand_ff >= 0 and cand_ff != orig_ff)

    equivalence_verified: Optional[bool]
    if latency_modified:
        if not allow_latency_changes:
            return _rej("latency changed (use --allow-latency-changes to keep)")
        equivalence_verified = None
    else:
        eqy = eda_runner.run_eqy(base_info["rtl_source"], cand_rtl, mod_info,
                                 config, jinja_env, PROJECT_ROOT, workdir)
        if eqy["equivalent"]:
            equivalence_verified = True
        else:
            return _rej("EQY not proven equivalent")

    return {
        "id": f"{base_info['module_name']}__llm_rejsample_{idx:03d}",
        "original_rtl": base_info["rtl_source"],
        "timing_report": {k: v for k, v in orig_report.items() if k != "raw_report"},
        "fixed_rtl": cand_rtl,
        "fixed_timing_report": {k: v for k, v in sta.items() if k != "raw_report"},
        "timing_improvement_ns": delta,
        "equivalence_verified": equivalence_verified,
        "mutation_type": "llm_rejection_sampling",
        "mutation_description": "LLM-proposed RTL rewrite, EDA-verified (synth + STA"
                                + ("" if latency_modified else " + EQY") + ")",
        "latency_modified": latency_modified,
        "synthesis_cell_count": {"original": base_info.get("cell_count", 0),
                                 "fixed": synth["cell_count"]},
    }


# ---------------------------------------------------------------------------
# Candidate module discovery
# ---------------------------------------------------------------------------
def discover_violating_modules(config: dict, jinja_env: Environment, workdir: Path,
                               min_violation_ns: float,
                               only_module: Optional[str]) -> List[Dict[str, Any]]:
    """Synthesize + STA every clocked module; return those with WNS <= threshold."""
    mods = eda_runner.discover_modules(
        str(PROJECT_ROOT / config["paths"]["raw_rtl"]),
        config["dataset"].get("skip_patterns", []),
    )
    if only_module:
        mods = [m for m in mods if m["module_name"] == only_module
                or m["filename"] == only_module]
    out = []
    for m in mods:
        if not m["has_clock"]:
            continue
        synth = eda_runner.synthesize(m, config, jinja_env, PROJECT_ROOT, workdir)
        if not synth["success"]:
            continue
        period = 1.0 / config["synthesis"].get("target_clock_ghz", [1.0])[0]
        sta = eda_runner.run_sta(m, synth, config, jinja_env, PROJECT_ROOT,
                                 period, workdir)
        if not sta.get("success"):
            continue
        if sta.get("wns_ns", 0.0) <= min_violation_ns:
            sta["clock_period_ns"] = period
            out.append({
                "module_name": m["module_name"],
                "stem": m["stem"],
                "clock_port_name": m.get("clock_port_name"),
                "rtl_source": m["rtl_source"],
                "cell_count": synth["cell_count"],
                "orig_report": sta,
                "orig_ff": _ff_count(synth.get("netlist_path")),
            })
        logger.info(f"  candidate: {m['filename']:40s} WNS={sta.get('wns_ns', 0):+.3f}ns"
                    f"  {'VIOLATES' if sta.get('wns_ns', 0) <= min_violation_ns else 'ok'}")
    return out


def modules_from_dataset(path: Path, only_module: Optional[str]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name = r["id"].split("__")[0]
        if name in seen:
            continue
        if only_module and name != only_module:
            continue
        seen.add(name)
        rep = r["timing_report"]
        out.append({
            "module_name": name,
            "stem": name,
            "clock_port_name": (rep.get("critical_path") or {}).get("startpoint"),
            "rtl_source": r["original_rtl"],
            "cell_count": (r.get("synthesis_cell_count") or {}).get("original", 0),
            "orig_report": rep,
            "orig_ff": -1,  # unknown from dataset; treat FF delta as unknown -> EQY still runs
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="EDA-verified LLM rejection sampling for timing closure")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--discover", action="store_true",
                     help="Synthesize+STA the whole corpus and use every violating module (default)")
    src.add_argument("--from-dataset", type=str,
                     help="Reuse modules from an existing timing_dataset.jsonl instead of discovering")
    p.add_argument("--module", type=str, help="Restrict to a single module")
    p.add_argument("--config", default="scripts/config.yaml")
    p.add_argument("--output", default="dataset/generated_dataset/rejection_sampled.jsonl")

    p.add_argument("--proposer", choices=["local", "anthropic"], default="local")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                   help="local: base model id/path")
    p.add_argument("--adapter", default=None,
                   help="local: optional PEFT/LoRA adapter dir (e.g. models/sft_lora_checkpoint)")
    p.add_argument("--anthropic-model", default="claude-sonnet-4-20250514")

    p.add_argument("--samples-per-module", type=int, default=8)
    p.add_argument("--max-keep-per-module", type=int, default=3)
    p.add_argument("--min-violation-ns", type=float, default=-0.1,
                   help="a module is a candidate only if its WNS <= this")
    p.add_argument("--min-improvement-ns", type=float, default=0.05,
                   help="a fix is kept only if (new_wns - orig_wns) >= this")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--allow-latency-changes", action="store_true",
                   help="also keep fixes that change flip-flop count (pipelining / "
                        "retiming) — EQY can't judge these cycle-accurately, so they "
                        "are marked equivalence_verified=null. Off by default.")
    p.add_argument("--limit", type=int, default=None, help="process first N candidate modules")
    p.add_argument("--force", action="store_true", help="ignore the .progress file")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    jinja_env = Environment(
        loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
        keep_trailing_newline=True,
    )
    workdir = PROJECT_ROOT / "dataset" / "workdir_rejsample"
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress.json")
    progress = {}
    if progress_path.exists() and not args.force:
        progress = json.loads(progress_path.read_text())

    # --- gather candidate modules -----------------------------------------
    logger.info("Gathering candidate (timing-violating) modules...")
    if args.from_dataset:
        modules = modules_from_dataset(PROJECT_ROOT / args.from_dataset, args.module)
    else:
        cache_path = out_path.with_suffix(".candidates.json")
        if cache_path.exists() and not args.force and not args.module:
            modules = json.loads(cache_path.read_text())
            logger.info(f"  (reusing {len(modules)} cached candidates from {cache_path.name}; "
                        f"--force to rediscover)")
        else:
            modules = discover_violating_modules(config, jinja_env, workdir,
                                                 args.min_violation_ns, args.module)
            if not args.module:
                cache_path.write_text(json.dumps(modules, default=str))
    modules = [m for m in modules if m["module_name"] not in progress]
    if args.limit:
        modules = modules[:args.limit]
    logger.info(f"{len(modules)} candidate module(s) to process "
                f"({len(progress)} already done).")
    if not modules:
        return

    # --- proposer -------------------------------------------------------------
    if args.proposer == "anthropic":
        proposer: Proposer = AnthropicProposer(args.anthropic_model, args.temperature,
                                               args.max_new_tokens)
    else:
        proposer = LocalProposer(args.model, args.adapter, args.temperature,
                                 args.top_p, args.max_new_tokens)

    t0 = time.time()
    total_kept = 0
    all_rejects: set = set()
    for i, base in enumerate(modules, 1):
        name = base["module_name"]
        orig_report = base["orig_report"]
        orig_ff = base.get("orig_ff", -1)
        logger.info(f"\n[{i}/{len(modules)}] {name}  "
                    f"(orig WNS {orig_report.get('wns_ns', 0):+.3f}ns, "
                    f"period {orig_report.get('clock_period_ns', 1.0):.3f}ns)")

        user_prompt = format_user_prompt(base["rtl_source"], orig_report)
        completions = proposer.generate(SYSTEM_PROMPT, user_prompt, args.samples_per_module)

        kept_here, seen_rtl, rejects = 0, set(), []
        for j, comp in enumerate(completions):
            if kept_here >= args.max_keep_per_module:
                break
            cand = extract_rtl(comp)
            if not cand:
                rejects.append("no verilog block / no module")
                continue
            if cand in seen_rtl:
                rejects.append("duplicate")
                continue
            seen_rtl.add(cand)
            rec = verify_candidate(cand, base, orig_report, orig_ff, config,
                                   jinja_env, workdir, total_kept + kept_here,
                                   args.allow_latency_changes, args.min_improvement_ns,
                                   reject_log=rejects)
            if rec:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                kept_here += 1
                eqv = {True: "EQY-equiv", None: "latency-modified"}[rec["equivalence_verified"]]
                logger.info(f"    kept #{kept_here}: Δwns={rec['timing_improvement_ns']:+.3f}ns  {eqv}")

        total_kept += kept_here
        all_rejects.update(rejects)
        progress[name] = {"done": True, "kept": kept_here, "samples": len(completions)}
        progress_path.write_text(json.dumps(progress, indent=2))
        rej_summary = ", ".join(sorted(set(rejects))) if rejects else "—"
        logger.info(f"    -> {kept_here} kept from {len(completions)} samples "
                    f"(rejects: {rej_summary})")

    dt = time.time() - t0
    logger.info(f"\n{'='*66}\nDone in {dt:.0f}s. Modules processed: {len(modules)}. "
                f"Records kept: {total_kept}.\nOutput: {out_path}")
    if all_rejects:
        logger.info("Reject reasons seen: " + "; ".join(sorted(all_rejects)))
    logger.info("Next: merge this file into timing_dataset.jsonl (or point "
                "prepare_sft_dataset.py --input at it) and retrain.")


if __name__ == "__main__":
    main()
