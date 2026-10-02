#!/usr/bin/env python3
"""
eda_runner.py — Automated EDA Dataset Generation Pipeline

Synthesizes RTL modules against SKY130, runs STA to extract timing violations,
applies rule-based mutations, verifies equivalence via EQY, and produces a
JSONL training dataset for GRPO RL fine-tuning.

Usage:
    python3 eda_runner.py                          # Process all modules
    python3 eda_runner.py --dry-run                # Just discover and list modules
    python3 eda_runner.py --module RefModule        # Process one module
    python3 eda_runner.py --limit 10 --parallel 4  # First 10 modules, 4 workers
    python3 eda_runner.py --skip-mutations         # Synth + STA only
"""

import os
import sys
import re
import json
import logging
import argparse
import time
import fnmatch
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import yaml
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("eda_runner")
_write_lock = threading.Lock()


def setup_logging(workdir: Path):
    """Set up dual logging: console (INFO) + file (DEBUG)."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "eda_runner.log"

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    """Walk up from this script to find the directory containing pdk/ and dataset/."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pdk").is_dir() and (current / "dataset").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
# Pass 1: exact 'clk' / 'clock' port names (fast path, covers most modules).
# Bounded by comma/semicolon/paren so it can't run away across an entire
# ANSI-style port list looking for the first stray ';' anywhere in the file.
_CLOCK_EXACT_RE = re.compile(r'\binput\b[^;,)]*\b(clk|clock)\b', re.IGNORECASE)

# Pass 2: any port whose name merely contains 'clk'/'clock' as a substring
# (axi_clk, rclk, gclk, wr_clk, sys_clk, chip_clk, MEMCLK, ref_clk, clka, ...).
# Real-world blocks rarely use the bare word 'clk' — this recovers them.
_CLOCK_TOKEN_RE = re.compile(
    r'\binput\b\s*(?:wire\s*|reg\s*|logic\s*)*(?:\[[^\]]*\]\s*)?(\w*cl(?:k|ock)\w*)\b',
    re.IGNORECASE,
)
# Exclude tokens that are clearly clock-*control* signals rather than the
# clock itself (clock enables, dividers, domain selects, gates, resets).
_CLOCK_BAD_SUBSTR = re.compile(r'(en|div|sel|mux|gate|dly|delay|rst|reset)', re.IGNORECASE)

MODULE_NAME = re.compile(r'module\s+(\w+)')


def detect_rtl_clock_port(clean_src: str) -> Optional[str]:
    """Best-effort detection of a module's clock port name from its port list.

    Tries an exact 'clk'/'clock' match first, then falls back to any input
    port whose name contains 'clk'/'clock' as a substring (covers common
    real-world naming conventions the exact match misses).
    """
    m = _CLOCK_EXACT_RE.search(clean_src)
    if m:
        return m.group(1)
    for m in _CLOCK_TOKEN_RE.finditer(clean_src):
        tok = m.group(1)
        if _CLOCK_BAD_SUBSTR.search(tok):
            continue
        return tok
    return None


def discover_modules(raw_rtl_dir: str, skip_patterns: List[str]) -> List[Dict[str, Any]]:
    """Scan raw_rtl/ for synthesizable .v/.sv files and extract metadata."""
    modules: List[Dict[str, Any]] = []
    base = Path(raw_rtl_dir)
    if not base.exists():
        logger.error(f"RTL directory not found: {raw_rtl_dir}")
        return modules

    for fpath in sorted(base.iterdir()):
        if fpath.suffix not in (".v", ".sv"):
            continue
        if any(fnmatch.fnmatch(fpath.name, p) for p in skip_patterns):
            logger.debug(f"Skipping {fpath.name} (matches skip pattern)")
            continue

        try:
            src = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {fpath}: {e}")
            continue

        clean_src = re.sub(r'//.*', '', src)
        clean_src = re.sub(r'/\*.*?\*/', '', clean_src, flags=re.DOTALL)

        m = MODULE_NAME.search(clean_src)
        if not m:
            logger.debug(f"No module found in {fpath.name}, skipping")
            continue

        # Detect clock port (comment-stripped source, to avoid matching
        # clock-sounding names inside comments)
        clock_port = detect_rtl_clock_port(clean_src)

        modules.append({
            "filepath":        str(fpath),
            "filename":        fpath.name,
            "stem":            fpath.stem,
            "module_name":     m.group(1),
            "is_sv":           fpath.suffix == ".sv",
            "has_clock":       clock_port is not None,
            "clock_port_name": clock_port,
            "rtl_source":      src,
        })

    logger.info(f"Discovered {len(modules)} modules "
                f"({sum(1 for m in modules if m['has_clock'])} with clock, "
                f"{sum(1 for m in modules if not m['has_clock'])} combinational-only)")
    return modules


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
def _run(cmd: List[str], cwd: Path, timeout: int) -> Tuple[bool, str, str]:
    """Run a command, return (ok, stdout, stderr).

    The tools are launched in their own session so a timeout kills the whole
    process group: the oss-cad-suite binaries are shell wrappers, and killing
    only the wrapper left the real yosys/abc underneath running for hours
    after every timed-out synthesis, quietly eating a core each.
    """
    import os, signal
    try:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return False, "", f"TIMEOUT after {timeout}s"
        return p.returncode == 0, out, err
    except Exception as e:
        return False, "", str(e)


# ---------------------------------------------------------------------------
# Yosys Synthesis
# ---------------------------------------------------------------------------
def synthesize(module_info: Dict, config: dict, jinja_env: Environment,
               project_root: Path, workdir: Path,
               rtl_path: Optional[str] = None,
               suffix: str = "") -> Dict[str, Any]:
    """Synthesize a single module via Yosys → SKY130 gate-level netlist."""
    mod = module_info["module_name"]
    stem = module_info.get("stem", mod)
    mdir = workdir / stem
    mdir.mkdir(parents=True, exist_ok=True)

    src = rtl_path or module_info["filepath"]
    liberty = str((project_root / config["paths"]["liberty"]).resolve())
    raw_rtl_dir = str((project_root / config["paths"]["raw_rtl"]).resolve())
    netlist = str((mdir / f"{mod}{suffix}_netlist.v").resolve())

    extra_files: List[str] = []

    def _attempt_synth(extras: List[str]):
        tcl = jinja_env.get_template("synth_sky130.tcl.j2").render(
            rtl_path=src,
            is_sv=module_info["is_sv"],
            top_module=mod,
            liberty_path=liberty,
            include_dir=raw_rtl_dir,
            extra_rtl_files=extras,
            output_netlist=netlist,
        )
        tcl_file = mdir / f"synth{suffix}.tcl"
        tcl_file.write_text(tcl)

        return _run(
            [config["tools"]["yosys"], "-s", str(tcl_file)],
            cwd=mdir,
            timeout=config["synthesis"].get("timeout_seconds", 300),
        )

    ok, stdout, stderr = _attempt_synth(extra_files)

    # Automatic submodule resolution retry — iterative, since a module can
    # depend on submodules that themselves depend on further submodules.
    # Yosys reports only the next unresolved hierarchy error per run, so a
    # single retry round only ever fixed the first missing dependency; a
    # module needing 2+ additional files would resolve the first and then
    # fail again on the second with no further retry. Loop until synthesis
    # succeeds or a round finds no new resolvable file (missing entirely, or
    # already tried), capped to guard against pathological cycles.
    raw_path = Path(raw_rtl_dir)
    seen_cells = set()
    for _round in range(40):
        if ok or "referenced in module" not in stderr and "is not part of the design" not in stderr:
            break

        missing_cells = re.findall(r"cell `\\?(\w+)' is not part of the design", stderr)
        missing_cells += re.findall(r"Module `\\?(\w+)' referenced in module", stderr)
        missing_cells = set(missing_cells) - {mod} - seen_cells
        if not missing_cells:
            break
        seen_cells |= missing_cells

        new_extras = []
        for cell in missing_cells:
            for candidate in [raw_path / f"{cell}.v", raw_path / f"{cell}.sv"]:
                if candidate.exists() and str(candidate.resolve()) != str(Path(src).resolve()):
                    new_extras.append(str(candidate.resolve()))
                    break

        if not new_extras:
            break

        extra_files = list(dict.fromkeys(extra_files + new_extras))
        logger.debug(f"Retrying synthesis for {mod} (round {_round + 1}) with submodules: {extra_files}")
        ok, stdout, stderr = _attempt_synth(extra_files)

    cell_count = 0
    if ok:
        cm = re.search(r"Number of cells:\s+(\d+)", stdout) or re.search(r"^\s*(\d+)\s+[\d.E+-]+\s+cells", stdout, re.MULTILINE)
        if cm:
            cell_count = int(cm.group(1))
        # Yosys keeps `integer` loop variables as `wire signed [31:0] i;` in the
        # netlist and OpenSTA's Verilog reader rejects `signed` with a syntax
        # error (41 of 155 external modules; the old fir_tap4 case). Signedness
        # carries no meaning in a structural gate-level netlist, so drop it.
        try:
            np = Path(netlist)
            txt = np.read_text()
            if re.search(r"\bsigned\b", txt):
                np.write_text(re.sub(r"\bsigned\s+", "", txt))
        except OSError as e:
            logger.warning(f"could not post-process netlist {netlist}: {e}")

    return {
        "success":      ok,
        "netlist_path":  netlist if ok else None,
        "cell_count":    cell_count,
        "yosys_log":     stdout + "\n" + stderr,
        "error":         stderr if not ok else None,
    }


# ---------------------------------------------------------------------------
# Clock-port detection from netlist
# ---------------------------------------------------------------------------
def detect_clock_port(netlist_path: str) -> Optional[str]:
    """Heuristic: find the clock port name from a gate-level Verilog netlist."""
    try:
        text = Path(netlist_path).read_text()
        # Check dfxtp CLK pin connections
        m = re.search(r"\.CLK\s*\(\s*(\w+)\s*\)", text)
        if m:
            return m.group(1)
        # Fallback: common names in port list
        m = re.search(r"input\s+(?:\[\d+:\d+\]\s+)?(clk|clock|CLK)\s*[;,]", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# OpenSTA
# ---------------------------------------------------------------------------
def run_sta(module_info: Dict, synth_result: Dict, config: dict,
            jinja_env: Environment, project_root: Path,
            clock_period_ns: float, workdir: Path,
            suffix: str = "") -> Dict[str, Any]:
    """Run OpenSTA on a synthesized netlist and return parsed timing report."""
    mod = module_info["module_name"]
    stem = module_info.get("stem", mod)
    mdir = workdir / stem

    netlist = synth_result["netlist_path"]
    clk_port = module_info.get("clock_port_name") or detect_clock_port(netlist)
    if not clk_port:
        return {"success": False, "error": "No clock port detected"}

    liberty = str((project_root / config["paths"]["liberty"]).resolve())
    in_delay  = clock_period_ns * config["synthesis"]["input_delay_fraction"]
    out_delay = clock_period_ns * config["synthesis"]["output_delay_fraction"]

    tcl = jinja_env.get_template("run_sta.tcl.j2").render(
        liberty_path=liberty,
        netlist_path=netlist,
        top_module=mod,
        clock_period_ns=clock_period_ns,
        clock_port=clk_port,
        input_delay_ns=in_delay,
        output_delay_ns=out_delay,
    )
    tcl_file = mdir / f"run_sta{suffix}.tcl"
    tcl_file.write_text(tcl)

    ok, stdout, stderr = _run(
        [config["tools"]["sta"], str(tcl_file)],
        cwd=mdir,
        timeout=config["synthesis"].get("timeout_seconds", 300),
    )
    if not ok:
        return {"success": False, "error": stderr, "raw_report": stdout}

    # OpenSTA can hit a fatal error partway through the script (e.g. it
    # can't parse a netlist port declared `signed` — a real, confirmed
    # failure mode) and still exit 0, since the script just runs to its
    # `exit` command with no timing commands having actually executed.
    # Trusting the exit code alone silently turns that into a fabricated
    # "wns_ns: 0.0 / meets timing" result. Require real evidence that
    # timing analysis actually ran: an explicit Error, or the total
    # absence of both a wns/tns line and any startpoint/endpoint.
    if re.search(r'^\s*Error\b', stdout, re.MULTILINE) or \
       (not re.search(r'\bwns\b', stdout, re.IGNORECASE) and
        not re.search(r'\btns\b', stdout, re.IGNORECASE)):
        return {"success": False,
                "error": "OpenSTA produced no timing report (see raw_report)",
                "raw_report": stdout}

    report = parse_sta_report(stdout)
    report["success"] = True
    report["clock_period_ns"] = clock_period_ns
    return report


def parse_sta_report(text: str) -> Dict[str, Any]:
    """Extract WNS, TNS, and critical-path details from OpenSTA output.

    OpenSTA output format:
        wns max -1.0621
        tns max -10.66
        Startpoint: _107_ (rising edge-triggered flip-flop ...)
        Endpoint: _098_ (rising edge-triggered flip-flop ...)
                    -1.0621   slack (VIOLATED)
    """
    report: Dict[str, Any] = {
        "wns_ns": 0.0,
        "tns_ns": 0.0,
        "slack_met": True,
        "critical_path": {},
        "raw_report": text,
    }

    # Number pattern matching integers or floats with optional sign
    NUM = r"([-+]?\d+(?:\.\d+)?)"

    # WNS / TNS — OpenSTA prints: "wns max -1.06" or "tns max -10.66"
    wns = re.search(r"wns\s+\w*\s*" + NUM, text)
    if wns:
        report["wns_ns"] = float(wns.group(1))
        report["slack_met"] = report["wns_ns"] >= 0.0

    tns = re.search(r"tns\s+\w*\s*" + NUM, text)
    if tns:
        report["tns_ns"] = float(tns.group(1))

    # Critical path
    sp = re.search(r"Startpoint:\s+(.+)", text)
    ep = re.search(r"Endpoint:\s+(.+)", text)
    # Slack line: "            -1.0621   slack (VIOLATED)"  — number BEFORE keyword
    sl = re.search(NUM + r"\s+slack\s+\((VIOLATED|MET)\)", text)
    ar = re.search(r"data arrival time\s+" + NUM, text)
    rq = re.search(r"data required time\s+" + NUM, text)

    # Collect cell instances along the path
    path_cells = re.findall(r"(sky130_fd_sc_hd__\w+)", text)

    if sp and ep:
        report["critical_path"] = {
            "startpoint":  sp.group(1).strip(),
            "endpoint":    ep.group(1).strip(),
            "slack_ns":    float(sl.group(1)) if sl else None,
            "arrival_ns":  float(ar.group(1)) if ar else None,
            "required_ns": float(rq.group(1)) if rq else None,
            "path_cells":  path_cells,
        }

    return report


# ---------------------------------------------------------------------------
# EQY Equivalence
# ---------------------------------------------------------------------------
def run_eqy(original_src: str, mutated_src: str, module_info: Dict,
            config: dict, jinja_env: Environment,
            project_root: Path, workdir: Path) -> Dict[str, Any]:
    """Run EQY to verify functional equivalence between original and mutated RTL."""
    mod = module_info["module_name"]
    stem = module_info.get("stem", mod)
    mdir = workdir / stem

    orig_file = mdir / f"{mod}_gold.v"
    orig_file.write_text(original_src)
    mut_file  = mdir / f"{mod}_gate.v"
    mut_file.write_text(mutated_src)

    eqy_cfg = jinja_env.get_template("verify_equiv.eqy.j2").render(
        original_rtl=str(orig_file.resolve()),
        mutated_rtl=str(mut_file.resolve()),
        is_sv=module_info["is_sv"],
        top_module=mod,
    )
    eqy_file = mdir / "verify.eqy"
    eqy_file.write_text(eqy_cfg)

    ok, stdout, stderr = _run(
        # -f: force-overwrite eqy's own "verify/" workdir. Without it, any
        # second run against the same module directory (e.g. a repeat
        # eda_runner.py pass, or evaluate_model.py run) fails immediately
        # with "Directory 'verify' already exists" rather than re-checking.
        [config["tools"]["eqy"], "-f", str(eqy_file)],
        cwd=mdir,
        timeout=config["synthesis"].get("timeout_seconds", 300),
    )

    equivalent = ok and "PASS" in stdout.upper() and "FAIL" not in stdout.upper()
    return {"equivalent": equivalent, "eqy_log": stdout + "\n" + stderr}


# ---------------------------------------------------------------------------
# Per-module pipeline
# ---------------------------------------------------------------------------
def process_module(module_info: Dict, config: dict,
                   jinja_env: Environment, project_root: Path,
                   skip_mutations: bool = False) -> Dict[str, Any]:
    """Full pipeline: synth → STA → mutate → re-synth → re-STA → EQY → records."""
    records: List[Dict] = []
    failures: List[Dict] = []
    workdir = project_root / config["paths"]["workdir"]
    mod = module_info["module_name"]
    stem = module_info.get("stem", mod)

    # --- 1. Synthesize original ------------------------------------------------
    synth = synthesize(module_info, config, jinja_env, project_root, workdir)
    if not synth["success"]:
        failures.append({"module": mod, "stage": "synthesis", "error": synth["error"]})
        return {"records": records, "failures": failures,
                "synth_ok": False, "cells": 0, "sta_report": None, "mutations_gen": 0}

    if not module_info["has_clock"]:
        return {"records": records, "failures": failures,
                "synth_ok": True, "cells": synth["cell_count"],
                "sta_report": None, "mutations_gen": 0}

    # --- 2. Run STA at each target frequency -----------------------------------
    target_clocks = config["synthesis"].get("target_clock_ghz", [1.0])
    sta_report = None
    mutations_gen = 0

    for clk_ghz in target_clocks:
        period = 1.0 / clk_ghz

        sta_report = run_sta(module_info, synth, config, jinja_env,
                             project_root, period, workdir)
        if not sta_report.get("success"):
            failures.append({"module": mod, "stage": "sta", "error": sta_report.get("error")})
            continue

        if skip_mutations:
            continue

        # --- 3. Apply mutations ------------------------------------------------
        # Import mutations lazily to support parallel execution
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from mutations import apply_mutations
        except ImportError:
            logger.warning("mutations.py not found — skipping mutation phase")
            continue

        mut_results = apply_mutations(
            module_info["rtl_source"],
            config["mutations"]["enabled"],
            config["mutations"]["max_per_module"],
            clock_port=module_info.get("clock_port_name") or "clk",
        )
        mutations_gen += len(mut_results)

        for i, (mut_name, mut_src, mut_desc, latency_mod) in enumerate(mut_results):
            suffix = f"_mut{i}_{mut_name}"

            # Write mutated RTL
            mut_dir = workdir / stem
            mut_file = mut_dir / f"{stem}{suffix}.v"
            mut_file.write_text(mut_src)

            # Synthesize mutated version (keep same module name for STA)
            mut_synth = synthesize(
                module_info, config, jinja_env, project_root, workdir,
                rtl_path=str(mut_file.resolve()), suffix=suffix,
            )
            if not mut_synth["success"]:
                logger.debug(f"  Mutation {mut_name} synth failed for {mod}")
                continue

            # STA on mutated
            mut_sta = run_sta(module_info, mut_synth, config, jinja_env,
                              project_root, period, workdir, suffix=suffix)
            if not mut_sta.get("success"):
                logger.debug(f"  Mutation {mut_name} STA failed for {mod}")
                continue

            # EQY equivalence (skip for latency-modifying mutations)
            eqy_pass = None
            if not latency_mod:
                eqy_res = run_eqy(module_info["rtl_source"], mut_src,
                                  module_info, config, jinja_env,
                                  project_root, workdir)
                eqy_pass = eqy_res["equivalent"]

            # Build training record
            orig_wns = sta_report.get("wns_ns", 0.0)
            mut_wns  = mut_sta.get("wns_ns", 0.0)

            records.append({
                "id":                    f"{mod}__{mut_name}_{i:03d}",
                "original_rtl":          module_info["rtl_source"],
                "timing_report":         {k: v for k, v in sta_report.items() if k != "raw_report"},
                "fixed_rtl":             mut_src,
                "fixed_timing_report":   {k: v for k, v in mut_sta.items() if k != "raw_report"},
                "timing_improvement_ns": mut_wns - orig_wns,
                "equivalence_verified":  eqy_pass,
                "mutation_type":         mut_name,
                "mutation_description":  mut_desc,
                "latency_modified":      latency_mod,
                "synthesis_cell_count":  {
                    "original": synth["cell_count"],
                    "fixed":    mut_synth["cell_count"],
                },
            })

    return {
        "records":       records,
        "failures":      failures,
        "synth_ok":      True,
        "cells":         synth["cell_count"],
        "sta_report":    sta_report,
        "mutations_gen": mutations_gen,
    }


# ---------------------------------------------------------------------------
# Thread-safe JSONL writer
# ---------------------------------------------------------------------------
def write_jsonl(record: dict, path: Path):
    """Append a JSON record to a JSONL file (thread-safe)."""
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EDA Dataset Generation Pipeline — Phase 1")
    parser.add_argument("--config",  default="scripts/config.yaml",
                        help="Path to config.yaml (relative to project root)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Max modules to process (for testing)")
    parser.add_argument("--module",  type=str, default=None,
                        help="Process a specific module by name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and list modules without processing")
    parser.add_argument("--skip-mutations", action="store_true",
                        help="Only run synthesis + STA, skip mutations/EQY")
    args = parser.parse_args()

    root = get_project_root()
    cfg_path = root / args.config
    if not cfg_path.exists():
        print(f"ERROR: Config not found at {cfg_path}")
        sys.exit(1)

    config = yaml.safe_load(cfg_path.read_text())
    workdir = root / config["paths"]["workdir"]
    setup_logging(workdir)

    logger.info(f"Project root : {root}")
    logger.info(f"Config       : {cfg_path}")

    jinja_env = Environment(
        loader=FileSystemLoader(str(root / "scripts" / "templates")),
        keep_trailing_newline=True,
    )

    # --- Discover modules ---------------------------------------------------
    modules = discover_modules(
        str(root / config["paths"]["raw_rtl"]),
        config["dataset"].get("skip_patterns", []),
    )
    if args.module:
        modules = [m for m in modules if m["module_name"] == args.module
                   or m["filename"] == args.module]
    if args.limit:
        modules = modules[:args.limit]

    if args.dry_run:
        for m in modules:
            clk = m["clock_port_name"] or "—"
            logger.info(f"  {m['filename']:40s}  module={m['module_name']:20s}  clk={clk}")
        return

    # --- Prepare output paths -----------------------------------------------
    out_dataset  = root / config["paths"]["output_dataset"]
    out_failures = root / config["paths"]["failures_log"]
    out_dataset.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_ok = total_fail = total_skip = total_records = 0

    # --- Process modules ----------------------------------------------------
    def _handle_result(mod: Dict, res: Dict):
        nonlocal total_ok, total_fail, total_skip, total_records
        for rec in res["records"]:
            write_jsonl(rec, out_dataset)
            total_records += 1
        for fail in res["failures"]:
            write_jsonl(fail, out_failures)

        if not res["synth_ok"]:
            total_fail += 1
            tag = "FAIL"
        elif res["sta_report"] is None:
            total_skip += 1
            tag = "SKIP (no clock)"
        else:
            total_ok += 1
            wns = res["sta_report"].get("wns_ns", 0.0)
            tag = f"synth OK ({res['cells']} cells) | WNS={wns:+.4f}ns | {res['mutations_gen']} mutations"

        n = total_ok + total_fail + total_skip
        logger.info(f"[{n}/{len(modules)}] {mod['filename']:40s} {tag}")

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(process_module, m, config, jinja_env, root,
                            args.skip_mutations): m
                for m in modules
            }
            for fut in as_completed(futures):
                mod = futures[fut]
                try:
                    _handle_result(mod, fut.result())
                except Exception as e:
                    logger.error(f"Exception processing {mod['module_name']}: {e}")
                    total_fail += 1
    else:
        for mod in modules:
            try:
                res = process_module(mod, config, jinja_env, root,
                                     args.skip_mutations)
                _handle_result(mod, res)
            except Exception as e:
                logger.error(f"Exception processing {mod['module_name']}: {e}")
                total_fail += 1

    # --- Summary ------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"  Finished in {elapsed:.1f}s")
    logger.info(f"  Modules processed : {total_ok + total_fail + total_skip}")
    logger.info(f"    Synthesized OK  : {total_ok}")
    logger.info(f"    Synth failed    : {total_fail}")
    logger.info(f"    Skipped (comb)  : {total_skip}")
    logger.info(f"  Training records  : {total_records}")
    logger.info(f"  Dataset           : {out_dataset}")
    logger.info(f"  Failures log      : {out_failures}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
