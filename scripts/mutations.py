#!/usr/bin/env python3
"""
mutations.py — RTL Mutation Engine for Timing Closure Dataset Generation

Applies structural transformations to Verilog/SystemVerilog source code to
produce timing-optimized variants. Works at the text/regex level (not AST).

Mutation types:
  - PipelineInsert:     Insert pipeline registers to break long combo paths (changes latency)
  - RegisterRetime:     Convert assign chains to registered stages (preserves function)
  - ExpressionBalance:  Restructure chained ops into balanced trees (preserves function)
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Mutation(ABC):
    """Base class for all RTL structural mutations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical mutation name (snake_case, matches config keys)."""
        ...

    @property
    def latency_modified(self) -> bool:
        """True if this mutation changes the pipeline depth / cycle latency."""
        return False

    @abstractmethod
    def applicable(self, src: str) -> bool:
        """Return True if this mutation can meaningfully be applied to *src*."""
        ...

    @abstractmethod
    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        """Return a list of mutated-source variants (may be empty on failure).

        clock_port: the module's actual clock port name, used by mutations
        that insert new `always @(posedge ...)` blocks (e.g. RegisterRetime).
        Mutations that don't add clocked logic may ignore it.
        """
        ...


# ---------------------------------------------------------------------------
# Pipeline Insertion
# ---------------------------------------------------------------------------
class PipelineInsert(Mutation):
    """
    Find registered blocks with complex combinational RHS expressions and
    split them by inserting an intermediate pipeline register.

    Example:
        q <= (a + b) * (c + d);
    Becomes:
        pipe_0 <= (a + b) * (c + d);   // stage 1
        q      <= pipe_0;               // stage 2
    """

    @property
    def name(self) -> str:
        return "pipeline_insert"

    @property
    def latency_modified(self) -> bool:
        return True

    # Match non-blocking assignments with 2+ operators on the RHS
    _COMPLEX_NBA = re.compile(
        r'(\s*)(\w+)\s*<=\s*'            # lhs <=
        r'((?:[^;]*[+\-*/&|^]){2,}[^;]*);'  # rhs with ≥2 operators
    )

    def applicable(self, src: str) -> bool:
        return bool(self._COMPLEX_NBA.search(src))

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        variants: List[str] = []
        try:
            m = self._COMPLEX_NBA.search(src)
            if not m:
                return variants

            indent, lhs, rhs = m.group(1), m.group(2), m.group(3)
            pipe_reg = f"{lhs}_pipe_stg"

            # Replace the single assignment with a two-stage version, wrapped
            # in begin/end: the original statement may be the sole body of an
            # if/else branch (e.g. `else acc <= expr;`), and without an
            # explicit block the second inserted statement would fall outside
            # the else and execute unconditionally every cycle, racing reset.
            new_block = (
                f"{indent}begin\n"
                f"{indent}    {pipe_reg} <= {rhs};\n"
                f"{indent}    {lhs} <= {pipe_reg};\n"
                f"{indent}end"
            )
            mutated = src[:m.start()] + new_block + src[m.end():]

            # Inject a reg declaration before the first always block
            decl = f"  reg [31:0] {pipe_reg};  // pipeline register (auto-inserted)\n"
            always_pos = mutated.find("always")
            if always_pos >= 0:
                mutated = mutated[:always_pos] + decl + mutated[always_pos:]

            variants.append(mutated)
        except Exception as e:
            logger.warning(f"PipelineInsert.apply failed: {e}")
        return variants


# ---------------------------------------------------------------------------
# Register Retiming
# ---------------------------------------------------------------------------
class RegisterRetime(Mutation):
    """
    Identify chains of combinational `assign` statements and convert one
    intermediate assign into a registered stage to balance path delays.

    Only converts an internal wire — preserves module I/O interface.
    """

    @property
    def name(self) -> str:
        return "register_retime"

    _ASSIGN = re.compile(r'assign\s+(\w+)\s*=\s*([^;]+);')

    def applicable(self, src: str) -> bool:
        return len(self._ASSIGN.findall(src)) >= 3

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        variants: List[str] = []
        try:
            assigns = self._ASSIGN.findall(src)
            if len(assigns) < 3:
                return variants

            # Build a dependency graph of assign LHS → signals used in RHS
            lhs_set = {a[0] for a in assigns}
            # Find an intermediate signal (used in another assign's RHS)
            target = None
            for lhs, rhs in assigns:
                for other_lhs, other_rhs in assigns:
                    if lhs != other_lhs and re.search(rf'\b{re.escape(lhs)}\b', other_rhs):
                        target = (lhs, rhs)
                        break
                if target:
                    break

            if not target:
                return variants

            sig, expr = target
            mutated = src

            # Convert `assign sig = expr;` → `always @(posedge <clock_port>) sig <= expr;`
            # Must reference the module's actual clock port — hardcoding "clk"
            # here would emit a reference to an undeclared signal for any
            # module whose clock isn't literally named "clk".
            old = f"assign {sig} = {expr};"
            new = f"always @(posedge {clock_port}) {sig} <= {expr};"
            mutated = mutated.replace(old, new, 1)

            # Change wire declaration to reg (if present)
            mutated = re.sub(
                rf'(\bwire\b\s*(?:\[[^\]]*\]\s*)?){re.escape(sig)}\b',
                lambda m: m.group(0).replace("wire", "reg", 1),
                mutated,
                count=1,
            )

            if mutated != src:
                variants.append(mutated)
        except Exception as e:
            logger.warning(f"RegisterRetime.apply failed: {e}")
        return variants


# ---------------------------------------------------------------------------
# Expression Balancing
# ---------------------------------------------------------------------------
class ExpressionBalance(Mutation):
    """
    Restructure chained binary operations into balanced trees to reduce
    combinational logic depth.

    Example:  a + b + c + d  →  (a + b) + (c + d)
    """

    @property
    def name(self) -> str:
        return "expression_balance"

    # Match 4+ operands chained with the same operator
    _CHAIN4 = re.compile(
        r'(\w+)\s*([+&|^])\s*(\w+)\s*\2\s*(\w+)\s*\2\s*(\w+)'
    )
    # Match 3 operands
    _CHAIN3 = re.compile(
        r'(\w+)\s*([+&|^])\s*(\w+)\s*\2\s*(\w+)'
    )

    def applicable(self, src: str) -> bool:
        return bool(self._CHAIN3.search(src))

    def apply(self, src: str, clock_port: str = "clk") -> List[str]:
        variants: List[str] = []
        try:
            mutated = src

            # First try 4-operand balancing
            def balance4(m):
                a, op, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
                return f"(({a} {op} {b}) {op} ({c} {op} {d}))"

            result = self._CHAIN4.sub(balance4, mutated)
            if result != mutated:
                variants.append(result)
                return variants

            # Fallback: 3-operand balancing
            def balance3(m):
                a, op, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
                return f"(({a} {op} {b}) {op} {c})"

            result = self._CHAIN3.sub(balance3, mutated)
            if result != mutated:
                variants.append(result)
        except Exception as e:
            logger.warning(f"ExpressionBalance.apply failed: {e}")
        return variants


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------
_ALL_MUTATIONS = [PipelineInsert(), RegisterRetime(), ExpressionBalance()]


def get_all_mutations() -> List[Mutation]:
    """Return instances of all available mutation strategies."""
    return list(_ALL_MUTATIONS)


def apply_mutations(
    rtl_source: str,
    enabled: Optional[List[str]] = None,
    max_per_module: int = 3,
    clock_port: str = "clk",
) -> List[Tuple[str, str, str, bool]]:
    """
    Apply enabled mutations to RTL source code.

    Args:
        rtl_source:     Original Verilog/SystemVerilog source.
        enabled:        List of mutation names to try (None = all).
        max_per_module: Max number of variants to return.
        clock_port:     The module's actual clock port name (used by
                        mutations that insert new registered logic).

    Returns:
        List of (mutation_name, mutated_source, description, latency_modified)
    """
    mutations = _ALL_MUTATIONS
    if enabled:
        mutations = [m for m in mutations if m.name in enabled]

    results: List[Tuple[str, str, str, bool]] = []

    for mut in mutations:
        if len(results) >= max_per_module:
            break
        try:
            if not mut.applicable(rtl_source):
                continue
            for variant in mut.apply(rtl_source, clock_port):
                if len(results) >= max_per_module:
                    break
                results.append((
                    mut.name,
                    variant,
                    f"Applied {mut.name}: {mut.__class__.__doc__.strip().split(chr(10))[0]}",
                    mut.latency_modified,
                ))
        except Exception as e:
            logger.error(f"Error applying {mut.name}: {e}")

    return results
