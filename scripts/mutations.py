#!/usr/bin/env python3
"""
mutations.py — RTL transform engine for timing-closure dataset generation.

Two directions, one transform library:

  FORWARD  (make a violating module faster — used by eda_runner.py)
    - expression_balance : re-tree an associative operator chain to min depth
    - pipeline_insert    : split a deep registered assignment into two stages
    - register_retime    : convert a combinational `assign` chain node to a
                           registered stage
    - strength_reduce    : `x * 2**k` -> `x << k`

  INVERSE  (make a passing module slower on purpose, so its *original* source
            becomes a known-good, human-written "fix" — used by
            deopt_generate.py)
    - expression_unbalance : skew a balanced chain into a deep linear tree
    - pipeline_merge       : fold two pipeline stages into one combinational
                             path (the transform Yosys/abc cannot undo, so it
                             reliably creates a real violation)

Everything is source-level and byte-surgical: only the touched span changes,
the rest of the file (comments, formatting, macros) is preserved verbatim.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

try:  # allow `import mutations` from either cwd
    from rtl_expr import flatten, balance, skew, split_top_level
except ImportError:  # pragma: no cover
    from scripts.rtl_expr import flatten, balance, skew, split_top_level

logger = logging.getLogger(__name__)

# RHS of `assign L = R;`  /  `L <= R;`  /  `L = R;`  (R has no ';' — safe for exprs)
_ASSIGN_RHS = re.compile(
    r'(^|[;)\n]\s*)'                      # 1: start-of-statement anchor
    r'(assign\s+)?'                       # 2: optional 'assign'
    r'([A-Za-z_]\w*(?:\s*\[[^\]]*\])?)'   # 3: LHS (opt. bit-select)
    r'\s*(<=|=)\s*'                       # 4: assignment operator
    r'([^;]+?)'                           # 5: RHS expression
    r'\s*;'
)
_NBA = re.compile(
    r'(^|[;)\n]\s*)([A-Za-z_]\w*)\s*<=\s*([^;]+?)\s*;'
)
_OP_CHARS = re.compile(r'[+\-*/%&|^~<>!]')
_WORD = lambda w: re.compile(r'\b' + re.escape(w) + r'\b')


def _expr_weight(expr: str) -> int:
    """Cheap proxy for combinational depth: operator + paren-group count."""
    return len(_OP_CHARS.findall(expr)) + expr.count('(')


# ---------------------------------------------------------------------------
class Mutation(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def latency_modified(self) -> bool:
        return False

    def applicable(self, src: str) -> bool:
        return bool(self.apply(src))

    @abstractmethod
    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        """Forward transforms (violating -> faster)."""
        ...

    def apply_inverse(self, src: str, clock_port: str = "clk") -> List[str]:
        """Inverse transforms (passing -> slower). Default: none."""
        return []


# ---------------------------------------------------------------------------
class ExpressionBalance(Mutation):
    """Re-tree an associative operator chain (a+b+c+d, a&b&c&d, ...) to
    minimum depth. Combinationally equivalent."""

    @property
    def name(self) -> str:
        return "expression_balance"

    def _rewrite(self, src: str, mode: str) -> List[str]:
        variants, seen = [], set()
        for m in _ASSIGN_RHS.finditer(src):
            rhs = m.group(5).strip()
            rhs_start = m.start(5)
            rhs_end = m.end(5)
            for op in ("&&", "||", "+", "*", "&", "|", "^"):
                flat = flatten(rhs, op)
                if not flat or len(flat) < 3:
                    continue
                new_rhs = (balance(flat, op) if mode == "balance"
                           else skew(flat, op))
                if "".join(new_rhs.split()) == "".join(rhs.split()):
                    continue
                cand = src[:rhs_start] + new_rhs + src[rhs_end:]
                key = "".join(cand.split())
                if key not in seen:
                    seen.add(key)
                    variants.append(cand)
                break  # one operator level per RHS
        return variants

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        return self._rewrite(src, "balance")

    def apply_inverse(self, src: str, clock_port: str = "clk") -> List[str]:
        return self._rewrite(src, "skew")


# ---------------------------------------------------------------------------
class PipelineInsert(Mutation):
    """Split one deep registered assignment into two pipeline stages.
    Forward changes latency; inverse (`pipeline_merge`) removes a stage."""

    @property
    def name(self) -> str:
        return "pipeline_insert"

    @property
    def latency_modified(self) -> bool:
        return True

    MIN_WEIGHT = 3  # only target RHS expressions with real combinational depth

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        variants = []
        for m in _NBA.finditer(src):
            anchor, lhs, rhs = m.group(1), m.group(2), m.group(3).strip()
            if _expr_weight(rhs) < self.MIN_WEIGHT:
                continue
            pipe = f"{lhs}_pipe_stg"
            block = (f"{anchor}begin\n"
                     f"    {pipe} <= {rhs};\n"
                     f"    {lhs} <= {pipe};\n"
                     f"  end")
            mutated = src[:m.start()] + block + src[m.end():]
            decl = f"  reg [63:0] {pipe}; // auto-inserted pipeline register\n"
            pos = mutated.find("always")
            if pos < 0:
                continue
            mutated = mutated[:pos] + decl + mutated[pos:]
            variants.append(mutated)
        return variants

    def apply_inverse(self, src: str, clock_port: str = "clk") -> List[str]:
        """Fold `A <= <exprA>;  B <= ...A...;` into one combinational path.

        This is the transform Yosys/abc can't undo (it won't re-pipeline),
        so it reliably lengthens the critical path -> a real violation whose
        known-good fix is the untouched original source.
        """
        variants = []
        # each NBA statement: (lhs, rhs, stmt_start, stmt_end) where the span
        # covers exactly `LHS <= RHS ;` (not the leading anchor char)
        nbas = []
        for m in _NBA.finditer(src):
            nbas.append((m.group(2), m.group(3).strip(),
                         m.start(2), m.end(0)))

        for ai, (a_lhs, a_rhs, a0, a1) in enumerate(nbas):
            if _expr_weight(a_rhs) < 1:   # A's RHS must do real work (>=1 operator)
                continue
            users = [j for j, (b_lhs, b_rhs, *_) in enumerate(nbas)
                     if j != ai and _WORD(a_lhs).search(b_rhs)]
            if len(users) != 1:
                continue
            bj = users[0]
            b_lhs, b_rhs, b0, b1 = nbas[bj]
            # A is safe to fold into B only if A is *read* nowhere except
            # inside B's RHS. A may be *written* (NBA LHS) any number of times
            # (e.g. a reset branch + the datapath branch) and declared once.
            total = len(_WORD(a_lhs).findall(src))
            as_lhs = len(re.findall(_WORD(a_lhs).pattern + r'\s*<=', src))
            declared = 1 if re.search(
                r'\b(?:reg|wire|logic)\b[^;=]*' + _WORD(a_lhs).pattern, src) else 0
            read_refs = total - as_lhs - declared
            refs_in_b = len(_WORD(a_lhs).findall(b_rhs))
            if read_refs != refs_in_b or refs_in_b == 0:
                continue
            merged = _WORD(a_lhs).sub(f"({a_rhs})", b_rhs)
            b_stmt_new = f"{b_lhs} <= {merged};"

            pieces = sorted([(a0, a1, ""), (b0, b1, b_stmt_new)])
            out, cur = [], 0
            for s, e, repl in pieces:
                out.append(src[cur:s])
                out.append(repl)
                cur = e
            out.append(src[cur:])
            new_src = "".join(out)
            # tidy: drop a line that is now only whitespace
            new_src = re.sub(r'\n[ \t]*\n', '\n', new_src, count=1)
            if new_src != src:
                variants.append(new_src)
        return variants


# ---------------------------------------------------------------------------
class RegisterRetime(Mutation):
    """Convert one intermediate `assign` in a dependency chain into a
    registered stage. Changes latency of that path."""

    @property
    def name(self) -> str:
        return "register_retime"

    @property
    def latency_modified(self) -> bool:
        return True

    _ASSIGN = re.compile(r'assign\s+(\w+)\s*=\s*([^;]+);')

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        assigns = self._ASSIGN.findall(src)
        if len(assigns) < 3:
            return []
        variants, seen = [], set()
        for lhs, rhs in assigns:
            if not any(o_lhs != lhs and _WORD(lhs).search(o_rhs)
                       for o_lhs, o_rhs in assigns):
                continue
            old = f"assign {lhs} = {rhs};"
            new = f"always @(posedge {clock_port}) {lhs} <= {rhs};"
            mutated = src.replace(old, new, 1)
            mutated = re.sub(
                rf'(\bwire\b\s*(?:\[[^\]]*\]\s*)?){re.escape(lhs)}\b',
                lambda mm: mm.group(0).replace("wire", "reg", 1), mutated, count=1)
            key = "".join(mutated.split())
            if mutated != src and key not in seen:
                seen.add(key)
                variants.append(mutated)
        return variants


# ---------------------------------------------------------------------------
class StrengthReduce(Mutation):
    """`x * 2**k` -> `x << k`. Combinationally equivalent (two's complement,
    same result width). Low yield — abc often does this anyway — but free."""

    @property
    def name(self) -> str:
        return "strength_reduce"

    _MUL_POW2 = re.compile(r'([A-Za-z_]\w*(?:\s*\[[^\]]*\])?)\s*\*\s*(\d+)')

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        variants = []
        for m in self._MUL_POW2.finditer(src):
            try:
                k = int(m.group(2))
            except ValueError:
                continue
            if k < 2 or (k & (k - 1)) != 0:      # not a power of two
                continue
            shift = k.bit_length() - 1
            repl = f"({m.group(1)} << {shift})"
            variants.append(src[:m.start()] + repl + src[m.end():])
        return variants


# ---------------------------------------------------------------------------
_ALL_MUTATIONS = [ExpressionBalance(), PipelineInsert(), RegisterRetime(),
                  StrengthReduce()]


def get_all_mutations() -> List[Mutation]:
    return list(_ALL_MUTATIONS)


def _select(enabled: Optional[List[str]]) -> List[Mutation]:
    if not enabled:
        return _ALL_MUTATIONS
    return [m for m in _ALL_MUTATIONS if m.name in enabled]


def _run(muts: List[Mutation], src: str, clock_port: str, max_per_module: int,
         inverse: bool) -> List[Tuple[str, str, str, bool]]:
    results: List[Tuple[str, str, str, bool]] = []
    for mut in muts:
        if len(results) >= max_per_module:
            break
        try:
            fn = mut.apply_inverse if inverse else mut.apply
            for variant in fn(src, clock_port):
                if len(results) >= max_per_module:
                    break
                tag = mut.name + ("_inv" if inverse else "")
                results.append((tag, variant,
                                f"{'inverse ' if inverse else ''}{mut.name}",
                                mut.latency_modified))
        except Exception as e:
            logger.error(f"{mut.name}{'(inv)' if inverse else ''} failed: {e}")
    return results


def apply_mutations(rtl_source: str, enabled: Optional[List[str]] = None,
                    max_per_module: int = 3, clock_port: str = "clk"
                    ) -> List[Tuple[str, str, str, bool]]:
    """Forward transforms. Returns (name, mutated_src, description, latency_modified)."""
    return _run(_select(enabled), rtl_source, clock_port, max_per_module, inverse=False)


def apply_inverse_mutations(rtl_source: str, enabled: Optional[List[str]] = None,
                            max_per_module: int = 3, clock_port: str = "clk"
                            ) -> List[Tuple[str, str, str, bool]]:
    """Inverse transforms for break-then-fix data generation."""
    return _run(_select(enabled), rtl_source, clock_port, max_per_module, inverse=True)
