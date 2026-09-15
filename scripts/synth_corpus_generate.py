#!/usr/bin/env python3
"""
synth_corpus_generate.py — template-based training-data generation.

Every other data-scaling approach tried this session (regex mutation, AST
mutation, inverse de-optimization, external RTL, LLM rejection sampling) hit
one of two walls: Yosys/abc already does the transform (so rewriting the
source changes nothing), or the transform needs a textual pattern real
parameterized RTL doesn't expose. This script sidesteps both: it generates
the VIOLATING and FIXED RTL directly from parameterized templates, so
correctness is true by construction (same computation, staged over more
cycles) rather than discovered by pattern-matching.

For each (template, parameter) combination:
  1. Generate a single-cycle "flat" version (deep combinational path)
  2. Generate a matching multi-stage pipelined version (same computation,
     split across registers at the natural sub-expression boundaries)
  3. Synthesize + STA both against the real Sky130 flow
  4. Keep the pair only if the flat version genuinely violates timing and
     the pipelined version genuinely, measurably closes it

Every kept record is EDA-tool-verified exactly like the rest of the dataset;
templates only supply candidates, they don't get a free pass.

Usage:
    python3 scripts/synth_corpus_generate.py                    # full grid
    python3 scripts/synth_corpus_generate.py --template mac     # one family
    python3 scripts/synth_corpus_generate.py --limit 10 --keep-rtl
"""

import re
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from jinja2 import Environment, FileSystemLoader

import eda_runner

logger = logging.getLogger("synth_corpus_generate")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _jinja():
    return Environment(
        loader=FileSystemLoader(str(PROJECT_ROOT / "scripts" / "templates")),
        keep_trailing_newline=True,
    )


# ---------------------------------------------------------------------------
# Template library: each function returns (module_name, clk_port, flat_src, pipe_src)
# ---------------------------------------------------------------------------
# A model trained on a narrow, homogeneous set of module names (everything
# prefixed "syn_") learns to shortcut on that surface pattern instead of on
# the actual input it's given — confirmed: it regurgitated a memorized
# "syn_addtree_..." module verbatim in response to an unrelated real input.
# Each family therefore draws its module-name STEM from its own pool,
# picked pseudo-randomly per instance, so there's no single giveaway prefix
# to latch onto.
_CLK_NAMES = ["clk", "clk_i", "sys_clk", "core_clk", "gclk", "wr_clk", "clock"]
_NAME_POOLS = {
    "mac": ["mac_core", "dsp_accum", "conv_tap", "dot_engine", "accum_unit"],
    "adder_tree": ["sum_tree", "adder_net", "reduce_add", "vec_sum", "tree_add"],
    "horner": ["poly_eval", "horner_unit", "filt_coef", "series_eval", "poly_core"],
    "compare_chain": ["match_unit", "cmp_gate", "range_check", "thresh_bank", "eq_gate"],
    "priority_mux": ["pri_select", "arb_mux", "pick_first", "route_sel", "chain_mux"],
    "shift_acc": ["shift_add", "barrel_acc", "rot_sum", "scale_unit", "shift_mac"],
}


def _clk_for(seed: int) -> str:
    return _CLK_NAMES[seed % len(_CLK_NAMES)]


def _name_for(family: str, seed: int) -> str:
    pool = _NAME_POOLS[family]
    return pool[seed % len(pool)]


def gen_mac(seed: int, width: int, n_terms: int) -> Tuple[str, str, str, str]:
    """Multiply-accumulate with n_terms product terms, single vs 2-stage."""
    mod = f"{_name_for('mac', seed)}_w{width}_t{n_terms}_{seed}"
    clk = _clk_for(seed)
    ws = [f"a{i}" for i in range(n_terms)]
    bs = [f"b{i}" for i in range(n_terms)]
    ports = "\n".join(f"    input  wire [{width-1}:0] {a},\n"
                      f"    input  wire [{width-1}:0] {b},"
                      for a, b in zip(ws, bs))
    prods = " + ".join(f"({a} * {b})" for a, b in zip(ws, bs))
    ow = 2 * width + max(1, n_terms.bit_length())
    flat = f"""// Auto-generated: {n_terms}-term multiply-accumulate, single-cycle.
module {mod} (
    input  wire        {clk},
    input  wire        rst_n,
{ports}
    output reg  [{ow-1}:0] acc
);
    always @(posedge {clk}) begin
        if (!rst_n)
            acc <= {ow}'sd0;
        else
            acc <= acc + {prods};
    end
endmodule
"""
    prod_decls = "\n".join(f"    reg [{2*width-1}:0] p{i}_q;" for i in range(n_terms))
    prod_stage = "\n".join(f"            p{i}_q <= {a} * {b};"
                           for i, (a, b) in enumerate(zip(ws, bs)))
    sum_terms = " + ".join(f"p{i}_q" for i in range(n_terms))
    pipe = f"""// Auto-generated: {n_terms}-term multiply-accumulate, pipelined
// (products registered one cycle before the accumulate add).
module {mod} (
    input  wire        {clk},
    input  wire        rst_n,
{ports}
    output reg  [{ow-1}:0] acc
);
{prod_decls}
    always @(posedge {clk}) begin
        if (!rst_n) begin
{chr(10).join(f"            p{i}_q <= 0;" for i in range(n_terms))}
            acc <= {ow}'sd0;
        end else begin
{prod_stage}
            acc <= acc + ({sum_terms});
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


def gen_adder_tree(seed: int, width: int, n_in: int) -> Tuple[str, str, str, str]:
    """n_in-input registered adder tree, single vs 2-stage (halves split)."""
    mod = f"{_name_for('adder_tree', seed)}_w{width}_n{n_in}_{seed}"
    clk = _clk_for(seed)
    ins = [f"in{i}" for i in range(n_in)]
    ports = "\n".join(f"    input  wire [{width-1}:0] {x}," for x in ins)
    ow = width + max(1, n_in.bit_length())
    sum_all = " + ".join(ins)
    flat = f"""// Auto-generated: {n_in}-input registered adder tree, single-cycle.
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{ports}
    output reg  [{ow-1}:0] sum_o
);
    always @(posedge {clk}) begin
        if (!rst_n) sum_o <= {ow}'d0;
        else        sum_o <= {sum_all};
    end
endmodule
"""
    half = n_in // 2
    left, right = ins[:half], ins[half:]
    lw = width + max(1, half.bit_length())
    rw = width + max(1, len(right).bit_length())
    pipe = f"""// Auto-generated: {n_in}-input registered adder tree, pipelined
// (left/right partial sums registered one cycle before the final add).
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{ports}
    output reg  [{ow-1}:0] sum_o
);
    reg [{lw-1}:0] lsum_q;
    reg [{rw-1}:0] rsum_q;
    always @(posedge {clk}) begin
        if (!rst_n) begin
            lsum_q <= 0; rsum_q <= 0; sum_o <= {ow}'d0;
        end else begin
            lsum_q <= {" + ".join(left)};
            rsum_q <= {" + ".join(right)};
            sum_o  <= lsum_q + rsum_q;
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


def gen_horner(seed: int, width: int, degree: int) -> Tuple[str, str, str, str]:
    """Horner-form polynomial evaluation: (((c_n*x+c_{n-1})*x+...)*x+c_0)."""
    mod = f"{_name_for('horner', seed)}_w{width}_d{degree}_{seed}"
    clk = _clk_for(seed)
    cs = [f"c{i}" for i in range(degree + 1)]
    ports = "\n".join(f"    input  wire [{width-1}:0] {c}," for c in cs)
    expr = cs[-1]
    for c in reversed(cs[:-1]):
        expr = f"(({expr}) * x + {c})"
    ow = width * 2
    flat = f"""// Auto-generated: degree-{degree} Horner polynomial evaluation, single-cycle.
module {mod} (
    input  wire        {clk},
    input  wire        rst_n,
    input  wire [{width-1}:0] x,
{ports}
    output reg  [{ow-1}:0] y
);
    always @(posedge {clk}) begin
        if (!rst_n) y <= {ow}'sd0;
        else        y <= {expr};
    end
endmodule
"""
    # Pipeline: split the Horner fold at its midpoint. Full Horner processes
    # coefficients top-down (c_degree first); to stage it correctly, stage 1
    # must fold the HIGH-order coefficients alone (exactly like the flat
    # expression, just truncated), and stage 2 continues folding the
    # low-order coefficients starting from that partial result — not the
    # other way around, which would evaluate a different polynomial.
    mid = (degree + 1) // 2
    inner_cs, outer_cs = cs[:mid], cs[mid:]        # low coeffs, high coeffs
    stage1_expr = outer_cs[-1]
    for c in reversed(outer_cs[:-1]):
        stage1_expr = f"(({stage1_expr}) * x + {c})"
    stage2_expr = "stage1_q"
    for c in reversed(inner_cs):
        stage2_expr = f"(({stage2_expr}) * x + {c})"
    pipe = f"""// Auto-generated: degree-{degree} Horner polynomial evaluation, pipelined
// (first half of the multiply-add chain registered before the second half).
module {mod} (
    input  wire        {clk},
    input  wire        rst_n,
    input  wire [{width-1}:0] x,
{ports}
    output reg  [{ow-1}:0] y
);
    reg [{ow-1}:0] stage1_q;
    always @(posedge {clk}) begin
        if (!rst_n) begin
            stage1_q <= {ow}'sd0;
            y <= {ow}'sd0;
        end else begin
            stage1_q <= {stage1_expr};
            y <= {stage2_expr};
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


def gen_crc_xor(seed: int, width: int, n_taps: int) -> Tuple[str, str, str, str]:
    """Wide XOR-reduction (CRC/parity-style), single vs 2-stage.

    Measured across two full sweeps: XOR-only logic is fast enough on sky130
    that even a 32-bit, 12-input reduction barely violates 1GHz (WNS ~ -0.01
    to -0.08ns) — 0/18 instances kept in either run. Left defined (harmless,
    might matter at a faster target clock) but excluded from the default
    GRID in favor of the three families below, which reliably violate.
    """
    mod = f"xor_gate_w{width}_t{n_taps}_{seed}"
    clk = _clk_for(seed)
    taps = [f"d{i}" for i in range(n_taps)]
    ports = "\n".join(f"    input  wire [{width-1}:0] {t}," for t in taps)
    xor_all = " ^ ".join(taps)
    flat = f"""// Auto-generated: {n_taps}-input XOR reduction, single-cycle.
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{ports}
    output reg  [{width-1}:0] parity_o
);
    always @(posedge {clk}) begin
        if (!rst_n) parity_o <= {width}'d0;
        else        parity_o <= {xor_all};
    end
endmodule
"""
    half = n_taps // 2
    left, right = taps[:half], taps[half:]
    pipe = f"""// Auto-generated: {n_taps}-input XOR reduction, pipelined
// (left/right partial parities registered one cycle before the final XOR).
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{ports}
    output reg  [{width-1}:0] parity_o
);
    reg [{width-1}:0] lxor_q, rxor_q;
    always @(posedge {clk}) begin
        if (!rst_n) begin
            lxor_q <= 0; rxor_q <= 0; parity_o <= {width}'d0;
        end else begin
            lxor_q <= {" ^ ".join(left)};
            rxor_q <= {" ^ ".join(right)};
            parity_o <= lxor_q ^ rxor_q;
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


def gen_compare_chain(seed: int, width: int, n_pairs: int) -> Tuple[str, str, str, str]:
    """AND-reduction of n_pairs magnitude comparisons, single vs 2-stage.

    Unlike XOR (bitwise, no carry), a magnitude compare on wide operands
    needs a carry-chain-like structure, so this has real, width-dependent
    depth even with few pairs.
    """
    mod = f"{_name_for('compare_chain', seed)}_w{width}_p{n_pairs}_{seed}"
    clk = _clk_for(seed)
    xs = [f"x{i}" for i in range(n_pairs)]
    ys = [f"y{i}" for i in range(n_pairs)]
    ports = "\n".join(f"    input  wire [{width-1}:0] {x},\n"
                      f"    input  wire [{width-1}:0] {y},"
                      for x, y in zip(xs, ys))
    cmps = " && ".join(f"({x} > {y})" for x, y in zip(xs, ys))
    flat = f"""// Auto-generated: {n_pairs}-pair magnitude-compare AND-reduction, single-cycle.
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{ports}
    output reg  match_o
);
    always @(posedge {clk}) begin
        if (!rst_n) match_o <= 1'b0;
        else        match_o <= {cmps};
    end
endmodule
"""
    cmp_decls = "\n".join(f"    reg c{i}_q;" for i in range(n_pairs))
    cmp_stage = "\n".join(f"            c{i}_q <= ({x} > {y});"
                          for i, (x, y) in enumerate(zip(xs, ys)))
    and_all = " && ".join(f"c{i}_q" for i in range(n_pairs))
    pipe = f"""// Auto-generated: {n_pairs}-pair magnitude-compare AND-reduction, pipelined
// (each wide comparison registered one cycle before the final AND).
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{ports}
    output reg  match_o
);
{cmp_decls}
    always @(posedge {clk}) begin
        if (!rst_n) begin
{chr(10).join(f"            c{i}_q <= 0;" for i in range(n_pairs))}
            match_o <= 1'b0;
        end else begin
{cmp_stage}
            match_o <= {and_all};
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


def gen_priority_mux(seed: int, width: int, n_levels: int) -> Tuple[str, str, str, str]:
    """Priority-mux chain (sel0 highest priority), single vs 2-stage.

    Same fold-and-split shape as Horner (validated there), applied to a
    ternary chain instead of multiply-add: stage 1 folds the LOW-priority
    half alone (exactly like the flat chain, just truncated to a default),
    stage 2 continues folding the HIGH-priority half starting from that
    registered partial result.
    """
    mod = f"{_name_for('priority_mux', seed)}_w{width}_n{n_levels}_{seed}"
    clk = _clk_for(seed)
    sels = [f"sel{i}" for i in range(n_levels)]
    ds = [f"d{i}" for i in range(n_levels)]
    sel_ports = "\n".join(f"    input  wire {s}," for s in sels)
    d_ports = "\n".join(f"    input  wire [{width-1}:0] {d}," for d in ds)

    def fold(sel_list, d_list, base):
        acc = base
        for s, d in zip(reversed(sel_list), reversed(d_list)):
            acc = f"({s} ? {d} : {acc})"
        return acc

    full_expr = fold(sels, ds, f"{width}'d0")
    flat = f"""// Auto-generated: {n_levels}-level priority mux, single-cycle.
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{sel_ports}
{d_ports}
    output reg  [{width-1}:0] y
);
    always @(posedge {clk}) begin
        if (!rst_n) y <= {width}'d0;
        else        y <= {full_expr};
    end
endmodule
"""
    mid = n_levels // 2
    tail_expr = fold(sels[mid:], ds[mid:], f"{width}'d0")   # low-priority half
    head_expr = fold(sels[:mid], ds[:mid], "tail_q")        # high-priority half
    pipe = f"""// Auto-generated: {n_levels}-level priority mux, pipelined
// (low-priority half of the chain registered before the high-priority half).
module {mod} (
    input  wire {clk},
    input  wire rst_n,
{sel_ports}
{d_ports}
    output reg  [{width-1}:0] y
);
    reg [{width-1}:0] tail_q;
    always @(posedge {clk}) begin
        if (!rst_n) begin
            tail_q <= {width}'d0;
            y <= {width}'d0;
        end else begin
            tail_q <= {tail_expr};
            y <= {head_expr};
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


def gen_shift_acc(seed: int, width: int, shamt_bits: int) -> Tuple[str, str, str, str]:
    """Barrel-shift + accumulate: acc <= (acc << sh) + (din >> sh) + din.

    Pipelines by registering the two shifted values before the final add —
    the shift-and-add of a wide operand by a variable amount is genuinely
    deep (a real barrel shifter, not a constant shift abc can trivially
    absorb), so splitting it out is a real win.
    """
    mod = f"{_name_for('shift_acc', seed)}_w{width}_s{shamt_bits}_{seed}"
    clk = _clk_for(seed)
    flat = f"""// Auto-generated: barrel-shift accumulate, single-cycle.
module {mod} (
    input  wire {clk},
    input  wire rst_n,
    input  wire [{width-1}:0] din,
    input  wire [{shamt_bits-1}:0] sh,
    output reg  [{width-1}:0] acc
);
    always @(posedge {clk}) begin
        if (!rst_n) acc <= {width}'d0;
        else        acc <= (acc << sh) + (din >> sh) + din;
    end
endmodule
"""
    pipe = f"""// Auto-generated: barrel-shift accumulate, pipelined
// (both shifted operands registered one cycle before the final add).
module {mod} (
    input  wire {clk},
    input  wire rst_n,
    input  wire [{width-1}:0] din,
    input  wire [{shamt_bits-1}:0] sh,
    output reg  [{width-1}:0] acc
);
    reg [{width-1}:0] shl_q, shr_q;
    always @(posedge {clk}) begin
        if (!rst_n) begin
            shl_q <= 0; shr_q <= 0; acc <= {width}'d0;
        end else begin
            shl_q <= acc << sh;
            shr_q <= din >> sh;
            acc   <= shl_q + shr_q + din;
        end
    end
endmodule
"""
    return mod, clk, flat, pipe


TEMPLATES: Dict[str, Callable] = {
    "mac": gen_mac,
    "adder_tree": gen_adder_tree,
    "horner": gen_horner,
    "xor_reduce": gen_crc_xor,
    "compare_chain": gen_compare_chain,
    "priority_mux": gen_priority_mux,
    "shift_acc": gen_shift_acc,
}

# (template, width, size-param) grid — size-param means n_terms/n_in/degree/
# n_pairs/n_levels/shamt_bits depending on family. Roughly balanced instance
# counts per family so no single one can dominate the resulting dataset the
# way adder_tree/horner did in the first sweep (mac/horner involve real
# multipliers — no DSP macros on sky130 std cells, so capped at
# n_terms/degree<=6 to keep synth time reasonable; xor_reduce is excluded,
# see gen_crc_xor's docstring).
GRID: List[Tuple[str, int, int]] = []
for w in (8, 16, 24, 32):
    for n in (3, 4, 6):
        GRID.append(("mac", w, n))
for w in (8, 16, 24, 32):
    for n in (6, 8, 12, 16):
        GRID.append(("adder_tree", w, n))
for w in (8, 12, 16, 24):
    for d in (3, 4, 6):
        GRID.append(("horner", w, d))
for w in (16, 24, 32, 48):
    for n in (3, 4, 6):
        GRID.append(("compare_chain", w, n))
for w in (16, 24, 32):
    for n in (12, 16, 20, 24):  # nested ternary muxes are shallow per level on
        GRID.append(("priority_mux", w, n))  # sky130; need deep chains to violate
for w in (16, 24, 32, 48):
    for s in (4, 5):
        GRID.append(("shift_acc", w, s))


# ---------------------------------------------------------------------------
def verify_pair(mod: str, clk: str, flat_src: str, pipe_src: str,
                config: dict, workdir: Path, period: float,
                min_violation_ns: float, min_gap_ns: float,
                keep_rtl: bool, family: str = "template") -> Dict[str, Any]:
    jinja_env = _jinja()
    info = {"module_name": mod, "stem": mod, "is_sv": False, "has_clock": True,
           "clock_port_name": clk}

    flat_info = dict(info, rtl_source=flat_src, filepath=str(workdir / f"{mod}_flat.v"))
    (workdir / mod).mkdir(parents=True, exist_ok=True)
    (workdir / f"{mod}_flat.v").write_text(flat_src)
    fs = eda_runner.synthesize(flat_info, config, jinja_env, PROJECT_ROOT, workdir,
                               rtl_path=str(workdir / f"{mod}_flat.v"), suffix="_flat")
    if not fs["success"]:
        return {"module": mod, "note": "flat synth failed"}
    f_sta = eda_runner.run_sta(flat_info, fs, config, jinja_env, PROJECT_ROOT,
                               period, workdir, suffix="_flat")
    if not f_sta.get("success"):
        return {"module": mod, "note": "flat STA failed"}
    f_sta["clock_period_ns"] = period
    flat_wns = f_sta.get("wns_ns", 0.0)
    if flat_wns > min_violation_ns:
        return {"module": mod, "note": f"flat doesn't violate (WNS={flat_wns:+.2f})"}

    pipe_info = dict(info, rtl_source=pipe_src, filepath=str(workdir / f"{mod}_pipe.v"))
    (workdir / f"{mod}_pipe.v").write_text(pipe_src)
    ps = eda_runner.synthesize(pipe_info, config, jinja_env, PROJECT_ROOT, workdir,
                               rtl_path=str(workdir / f"{mod}_pipe.v"), suffix="_pipe")
    if not ps["success"]:
        return {"module": mod, "note": "pipelined synth failed"}
    p_sta = eda_runner.run_sta(pipe_info, ps, config, jinja_env, PROJECT_ROOT,
                               period, workdir, suffix="_pipe")
    if not p_sta.get("success"):
        return {"module": mod, "note": "pipelined STA failed"}
    p_sta["clock_period_ns"] = period
    pipe_wns = p_sta.get("wns_ns", 0.0)

    gap = pipe_wns - flat_wns
    if gap < min_gap_ns:
        return {"module": mod, "note": f"insufficient gap ({gap:+.2f}ns)"}

    if not keep_rtl:
        for suffix in ("_flat", "_pipe"):
            for f in (workdir / f"{mod}{suffix}.v", workdir / f"{mod}{suffix}_netlist.v"):
                f.unlink(missing_ok=True)

    record = {
        "id": f"{mod}__synth_gen_000",
        "original_rtl": flat_src,
        "timing_report": {k: v for k, v in f_sta.items() if k != "raw_report"},
        "fixed_rtl": pipe_src,
        "fixed_timing_report": {k: v for k, v in p_sta.items() if k != "raw_report"},
        "timing_improvement_ns": gap,
        "equivalence_verified": None,  # pipelining changes latency by construction
        "mutation_type": f"synthetic_{family}_pipeline",
        "mutation_description": "template-generated: flat vs. pipelined version of "
                                "the same computation; correctness by construction",
        "latency_modified": True,
        "synthesis_cell_count": {"original": fs["cell_count"], "fixed": ps["cell_count"]},
    }
    return {"module": mod, "note": f"KEPT Δ={gap:+.2f}ns (flat {flat_wns:+.2f} -> fix {pipe_wns:+.2f})",
           "record": record}


def _worker(args):
    tname, w, n, seed, config, period, min_v, min_g, keep_rtl = args
    workdir = PROJECT_ROOT / "dataset" / "workdir_synthgen"
    mod, clk, flat, pipe = TEMPLATES[tname](seed, w, n)
    return verify_pair(mod, clk, flat, pipe, config, workdir, period, min_v, min_g,
                       keep_rtl, family=tname)


def main():
    p = argparse.ArgumentParser(description="Template-based timing-closure data generation")
    p.add_argument("--config", default="scripts/config.yaml")
    p.add_argument("--output", default="dataset/generated_dataset/synthetic_dataset.jsonl")
    p.add_argument("--template", choices=list(TEMPLATES), help="only this template family")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--min-violation-ns", type=float, default=-0.1)
    p.add_argument("--min-gap-ns", type=float, default=0.3)
    p.add_argument("--period-ns", type=float, default=1.0)
    p.add_argument("--keep-rtl", action="store_true",
                   help="keep generated .v/netlist files in the workdir for inspection")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    workdir = PROJECT_ROOT / "dataset" / "workdir_synthgen"
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force and out_path.exists():
        out_path.unlink()

    grid = [g for g in GRID if not args.template or g[0] == args.template]
    if args.limit:
        grid = grid[:args.limit]
    jobs = [(tname, w, n, i, config, args.period_ns, args.min_violation_ns,
            args.min_gap_ns, args.keep_rtl) for i, (tname, w, n) in enumerate(grid)]
    logger.info(f"{len(jobs)} template instance(s) to generate + verify")

    total = 0
    def _handle(res):
        nonlocal total
        rec = res.pop("record", None)
        if rec:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            total += 1
        logger.info(f"  {res['module']:36s} {res['note']}")

    t0 = time.time()
    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(_worker, j): j[3] for j in jobs}
            for fut in as_completed(futs):
                try:
                    _handle(fut.result())
                except Exception as e:
                    logger.error(f"  job {futs[fut]}: {e}")
    else:
        for j in jobs:
            try:
                _handle(_worker(j))
            except Exception as e:
                logger.error(f"  {j}: {e}")

    logger.info(f"\n{'='*60}\nDone in {time.time()-t0:.0f}s. "
                f"{total}/{len(jobs)} kept. Output: {out_path}")


if __name__ == "__main__":
    main()
