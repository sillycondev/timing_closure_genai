#!/usr/bin/env python3
"""
rtl_expr.py — minimal Verilog-expression helpers for the mutation engine.

Not a full parser. It does one thing well: given the text of a Verilog
expression, find maximal chains of a single associative/commutative binary
operator at the top nesting level (or inside any parenthesised sub-group),
and rewrite that chain into a balanced or a skewed tree — leaving the rest
of the expression, and the rest of the file, byte-for-byte untouched.

Bit-selects `x[a:b]`, concatenations `{...}`, function calls `f(...)`,
ternaries and everything inside parentheses are treated as opaque operands;
we never restructure their internals, only the chain they sit in.
"""

from typing import List, Optional, Tuple

# Associative + commutative operators we are allowed to re-tree.
# '-' and '/' are NOT associative; '<<'/'>>' handled elsewhere.
ASSOC_OPS = ["&&", "||", "+", "*", "&", "|", "^"]

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = set(")]}")


def _matching_op(op: str) -> str:
    return _OPEN[op]


def split_top_level(expr: str, op: str) -> Optional[List[str]]:
    """Split `expr` at every top-level (depth-0) binary occurrence of `op`.

    Returns the list of operand substrings (trimmed) if `op` occurs at least
    once as a *binary* operator at depth 0, else None. A leading/unary use
    of the operator (e.g. `-a`, `~b`, `&bus`) is not treated as a split
    point: an operand is only accepted between two non-empty sides.
    """
    parts: List[str] = []
    depth = 0
    i = 0
    n = len(expr)
    last = 0
    oplen = len(op)
    while i < n:
        c = expr[i]
        if c in _OPEN:
            depth += 1
            i += 1
            continue
        if c in _CLOSE:
            depth -= 1
            i += 1
            continue
        if c in ('"', "'"):
            # skip string / char literal (rare in exprs, but be safe for '"..."')
            if c == '"':
                j = expr.find('"', i + 1)
                i = (j + 1) if j != -1 else n
                continue
        if depth == 0 and expr[i:i + oplen] == op:
            # must not be part of a longer operator token
            prevc = expr[i - 1] if i > 0 else ""
            nextc = expr[i + oplen] if i + oplen < n else ""
            longer = set("+-*/%&|^~!<>=")
            if prevc in longer or nextc in longer:
                i += 1
                continue
            left = expr[last:i].strip()
            right_rest = expr[i + oplen:].strip()
            if not left or not right_rest:
                i += 1
                continue
            parts.append(left)
            last = i + oplen
            i += oplen
            continue
        i += 1

    if not parts:
        return None
    parts.append(expr[last:].strip())
    # every part must be a non-empty, balanced fragment
    for p in parts:
        if not p or _depth_delta(p) != 0:
            return None
    return parts


def _depth_delta(s: str) -> int:
    d = 0
    for c in s:
        if c in _OPEN:
            d += 1
        elif c in _CLOSE:
            d -= 1
        if d < 0:
            return -1
    return d


def flatten(expr: str, op: str) -> Optional[List[str]]:
    """Fully flatten a nested same-operator tree into a flat operand list.

    `((a+b)+(c+d))` with op '+' -> ['a','b','c','d'].
    Returns None if `expr` is not an `op`-rooted chain at all.
    """
    parts = split_top_level(expr, op)
    if parts is None:
        # maybe wrapped in one redundant paren layer
        s = expr.strip()
        if s.startswith("(") and s.endswith(")") and _depth_delta(s[1:-1]) == 0:
            parts = split_top_level(s[1:-1].strip(), op)
        if parts is None:
            return None
    flat: List[str] = []
    for p in parts:
        sub = flatten(p, op)
        if sub is None:
            flat.append(p.strip())
        else:
            flat.extend(sub)
    return flat


def balance(parts: List[str], op: str) -> str:
    """Combine operands into a minimum-depth binary tree."""
    parts = [p.strip() for p in parts]
    if len(parts) == 1:
        return parts[0]
    mid = len(parts) // 2
    left = balance(parts[:mid], op)
    right = balance(parts[mid:], op)
    return f"({left} {op} {right})"


def skew(parts: List[str], op: str) -> str:
    """Combine operands into a maximum-depth (left-leaning) binary tree."""
    parts = [p.strip() for p in parts]
    acc = parts[0]
    for p in parts[1:]:
        acc = f"({acc} {op} {p})"
    return acc


def _linear_depth(parts_count: int) -> int:
    return max(0, parts_count - 1)


def _balanced_depth(parts_count: int) -> int:
    d, w = 0, 1
    while w < parts_count:
        w *= 2
        d += 1
    return d


def find_rebalanceable(expr: str, min_operands: int = 3
                       ) -> List[Tuple[str, str, List[str], bool]]:
    """Find associative chains in `expr` worth rewriting.

    Returns a list of (op, original_chain_text, operands, already_balanced).
    `original_chain_text` is exactly `expr` here — callers pass in the single
    RHS/operand text they want considered, and recurse into parts themselves
    if desired. We report the *outermost* chain for each operator so the
    replacement is a clean whole-expression swap.
    """
    out = []
    stripped = expr.strip()
    # peel one layer of redundant wrapping parens
    core = stripped
    while core.startswith("(") and _depth_delta(core[1:-1]) == 0 and core.endswith(")"):
        inner = core[1:-1].strip()
        if _depth_delta(inner) != 0:
            break
        core = inner
    for op in ASSOC_OPS:
        parts = split_top_level(core, op)
        if parts and len(parts) >= min_operands:
            bal = balance(parts, op)
            already = _norm(bal) == _norm(core) or \
                _linear_depth(len(parts)) == _balanced_depth(len(parts))
            out.append((op, expr, parts, already))
            break  # one operator level per expression is enough
    return out


def _norm(s: str) -> str:
    return "".join(s.split())
