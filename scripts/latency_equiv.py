#!/usr/bin/env python3
"""
latency_equiv.py — latency-tolerant functional equivalence via simulation.

EQY proves cycle-exact equivalence, so every pipelining fix — the one kind
of fix this project's model produces — comes back "fail / latency changed"
whether it is correct or broken. Best-of-N selection therefore had no
functional gate at all, and three of four "improved" real modules turned out
to be functionally wrong rewrites that merely happened to time better.

This check closes that gap the pragmatic way: simulate gold and gate side by
side under identical random stimulus (reset asserted during warm-up), record
both output streams, and accept iff gate(t) == gold(t - N) for some fixed
N in [0, max_shift] over every compared cycle. It is a falsifier, not a
proof — but a randomized 3000-cycle run catches the broken rewrites seen so
far trivially, and unlike SAT it does not stall on wide multipliers.

Result "equivalent" is True/False, or None when the check could not run
(unsynthesizable in iverilog, missing submodules, interface mismatch, or
outputs that never left X — vacuous agreement is not reported as a pass).
"""

import re
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

TOOLS = Path.home() / "tools" / "oss-cad-suite" / "bin"
YOSYS = str(TOOLS / "yosys")
IVERILOG = str(TOOLS / "iverilog")
VVP = str(TOOLS / "vvp")

_CLK = re.compile(r"clk|clock", re.I)
_RST = re.compile(r"rst|reset", re.I)
_ACTIVE_LOW = re.compile(r"(_n|n|_l|_b)$", re.I)


def _ports(src: str, mod: str, workdir: Path, tag: str) -> Optional[Dict[str, Tuple[str, int]]]:
    """{port: (direction, width)} via Yosys, so parameters/non-ANSI styles resolve."""
    v = workdir / f"{tag}.v"
    j = workdir / f"{tag}.json"
    v.write_text(src)
    cmd = [YOSYS, "-q", "-p", f"read_verilog -sv {v}; hierarchy -top {mod}; proc; write_json {j}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not j.exists():
        return None
    data = json.loads(j.read_text())
    m = data["modules"].get(mod)
    if not m:
        return None
    return {n: (p["direction"], len(p["bits"])) for n, p in m["ports"].items()}


def _rename(src: str, mod: str, new: str) -> str:
    return re.sub(rf"\bmodule\s+{re.escape(mod)}\b", f"module {new}", src, count=1)


def _rand(width: int) -> str:
    words = ", ".join(["$urandom"] * max(1, math.ceil(width / 32)))
    return f"{{{words}}}"


def _drive(name: str, width: int) -> str:
    """Per-cycle stimulus mixing full-random with small, zero and all-ones values.

    Pure random stimulus never lets a wide counter/timer reach its terminal
    value, so a rewrite that breaks the recurrence can agree with the original
    for the whole run (observed: a broken 64-bit down-counter passed).
    """
    return (f"      case ($urandom % 4) 0: {name} = {_rand(width)}; 1: {name} = $urandom % 8; "
            f"2: {name} = 0; default: {name} = {{{width}{{1'b1}}}}; endcase")


def _testbench(mod: str, ports: Dict[str, Tuple[str, int]], clock: str,
               max_shift: int, cycles: int, warmup: int, seed: int) -> str:
    ins = [(n, w) for n, (d, w) in ports.items() if d == "input"]
    outs = [(n, w) for n, (d, w) in ports.items() if d == "output"]
    clks = [n for n, _ in ins if n == clock or _CLK.search(n)]
    rsts = [n for n, _ in ins if n not in clks and _RST.search(n)]
    data = [(n, w) for n, w in ins if n not in clks and n not in rsts]
    ow = sum(w for _, w in outs)

    decl = "\n".join(f"  reg [{w-1}:0] {n};" for n, w in ins)
    gold_w = "\n".join(f"  wire [{w-1}:0] g_{n};" for n, w in outs)
    gate_w = "\n".join(f"  wire [{w-1}:0] t_{n};" for n, w in outs)
    conn_in = ", ".join(f".{n}({n})" for n, _ in ins)
    gold_conn = ", ".join([conn_in] + [f".{n}(g_{n})" for n, _ in outs])
    gate_conn = ", ".join([conn_in] + [f".{n}(t_{n})" for n, _ in outs])
    gold_cat = "{" + ", ".join(f"g_{n}" for n, _ in outs) + "}"
    gate_cat = "{" + ", ".join(f"t_{n}" for n, _ in outs) + "}"
    rst_on = "\n".join(f"      {n} = {0 if _ACTIVE_LOW.search(n) else 1};" for n in rsts)
    rst_off = "\n".join(f"      {n} = {1 if _ACTIVE_LOW.search(n) else 0};" for n in rsts)
    drive = "\n".join(_drive(n, w) for n, w in data)
    clk_drive = "\n".join(f"  always @(tb_clk) {n} = tb_clk;" for n in clks)

    return f"""`timescale 1ns/1ps
module tb;
  reg tb_clk = 0;
{decl}
{gold_w}
{gate_w}
  {mod}__gold gold ({gold_conn});
  {mod}__gate gate ({gate_conn});
{clk_drive}
  wire [{ow-1}:0] gold_o = {gold_cat};
  wire [{ow-1}:0] gate_o = {gate_cat};
  reg  [{ow-1}:0] hist [0:{max_shift}];
  reg  bad [0:{max_shift}];
  integer cyc, s, defined, compared, seed;
  always #5 tb_clk = ~tb_clk;
  initial begin
    seed = {seed}; s = $urandom(seed);
    for (s = 0; s <= {max_shift}; s = s + 1) begin bad[s] = 0; hist[s] = 0; end
    defined = 0; compared = 0;
{rst_on}
{drive}
    for (cyc = 0; cyc < {cycles}; cyc = cyc + 1) begin
      @(negedge tb_clk);
      if (cyc == 8) begin
{rst_off}
      end
      // hist[s] = gold output s cycles ago (hist[0] = this cycle); compare
      // BEFORE changing inputs so both designs saw identical stimulus
      for (s = {max_shift}; s > 0; s = s - 1) hist[s] = hist[s-1];
      hist[0] = gold_o;
      if (cyc >= {warmup}) begin
        compared = compared + 1;
        if (^gold_o !== 1'bx) defined = defined + 1;
        for (s = 0; s <= {max_shift}; s = s + 1)
          if (gate_o !== hist[s]) bad[s] = 1;
      end
{drive}
    end
    $write("LATENCY_EQUIV_RESULT compared=%0d defined=%0d shifts=", compared, defined);
    for (s = 0; s <= {max_shift}; s = s + 1) if (!bad[s]) $write("%0d,", s);
    $write("\\n");
    $finish;
  end
endmodule
"""


def check_latency_equiv(gold_src: str, gate_src: str, mod: str, clock: str,
                        workdir: Path, max_shift: int = 4, cycles: int = 3000,
                        warmup: int = 24, seed: int = 1, timeout: int = 180) -> Dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    pg = _ports(gold_src, mod, workdir, "le_gold")
    pt = _ports(gate_src, mod, workdir, "le_gate")
    if pg is None or pt is None:
        return {"equivalent": None, "reason": "could not elaborate ports"}
    if pg != pt:
        return {"equivalent": False, "reason": f"interface mismatch: {sorted(set(pg) ^ set(pt)) or 'widths differ'}"}
    if not any(d == "output" for d, _ in pg.values()):
        return {"equivalent": None, "reason": "no outputs to compare"}

    gold_v = workdir / "le_gold_r.v"
    gate_v = workdir / "le_gate_r.v"
    tb_v = workdir / "le_tb.v"
    gold_v.write_text(_rename(gold_src, mod, f"{mod}__gold"))
    gate_v.write_text(_rename(gate_src, mod, f"{mod}__gate"))
    tb_v.write_text(_testbench(mod, pg, clock, max_shift, cycles, warmup, seed))

    vvp = workdir / "le_tb.vvp"
    c = subprocess.run([IVERILOG, "-g2012", "-o", str(vvp), str(tb_v), str(gold_v), str(gate_v)],
                       capture_output=True, text=True, timeout=timeout)
    if c.returncode != 0:
        return {"equivalent": None, "reason": "iverilog compile failed", "log": c.stderr[-2000:]}
    try:
        r = subprocess.run([VVP, "-n", str(vvp)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"equivalent": None, "reason": "simulation timeout"}
    m = re.search(r"LATENCY_EQUIV_RESULT compared=(\d+) defined=(\d+) shifts=([\d,]*)", r.stdout)
    if not m:
        return {"equivalent": None, "reason": "no result line", "log": (r.stdout + r.stderr)[-2000:]}
    compared, defined = int(m.group(1)), int(m.group(2))
    shifts = [int(x) for x in m.group(3).split(",") if x]
    if defined < compared // 2:
        return {"equivalent": None, "reason": f"outputs mostly undefined ({defined}/{compared} cycles)",
                "shifts": shifts}
    if shifts:
        return {"equivalent": True, "shift": min(shifts), "all_shifts": shifts,
                "compared": compared, "defined": defined}
    return {"equivalent": False, "reason": f"no shift in 0..{max_shift} matches", "compared": compared}


if __name__ == "__main__":
    import sys, argparse
    p = argparse.ArgumentParser()
    p.add_argument("gold"); p.add_argument("gate"); p.add_argument("module")
    p.add_argument("--clock", default="clk"); p.add_argument("--workdir", default="/tmp/latency_equiv")
    a = p.parse_args()
    print(json.dumps(check_latency_equiv(Path(a.gold).read_text(), Path(a.gate).read_text(),
                                         a.module, a.clock, Path(a.workdir)), indent=2))
