#!/usr/bin/env bash
# test_setup.sh — Non-destructive smoke test for the timing-closure GenAI pipeline.
#
# Verifies:
#   1. Required EDA tools are on PATH (yosys, sta; eqy/sby are optional)
#   2. Python deps needed by eda_runner.py are importable (pyyaml, jinja2)
#   3. The Liberty PDK file and raw RTL corpus exist
#   4. Module discovery works (--dry-run)
#   5. A full synth -> STA -> mutate -> re-synth -> re-STA -> EQY pass runs end to end
#      on one small module (Counter.v), writing to a scratch dir so the real
#      dataset/generated_dataset/*.jsonl files are never touched
#   6. prepare_sft_dataset.py can convert the resulting records to ChatML
#
# Usage:
#   ./scripts/test_setup.sh            # run + clean up scratch dir
#   ./scripts/test_setup.sh --keep     # run + leave scratch dir for inspection
#
# Exit code is 0 only if every hard-required check passes.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

SCRATCH="$(mktemp -d /tmp/timing_closure_test.XXXXXX)"
TEST_MODULE="Counter"
PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }
section() { echo; echo "== $1 =="; }

cleanup() {
    if [[ $KEEP -eq 0 ]]; then
        rm -rf "$SCRATCH"
    else
        echo
        echo "Scratch artifacts kept at: $SCRATCH"
    fi
}
trap cleanup EXIT

echo "Timing Closure GenAI — Setup Smoke Test"
echo "Project root : $PROJECT_ROOT"
echo "Scratch dir  : $SCRATCH"

# ---------------------------------------------------------------------------
section "1. EDA toolchain on PATH"
if command -v yosys >/dev/null 2>&1; then
    pass "yosys found: $(yosys -V 2>&1 | head -1)"
else
    fail "yosys not found on PATH (required for synthesis)"
fi

if command -v sta >/dev/null 2>&1; then
    pass "sta (OpenSTA) found: $(sta -version 2>&1 | head -1)"
else
    fail "sta not found on PATH (required for timing analysis)"
fi

if command -v eqy >/dev/null 2>&1; then
    pass "eqy found"
else
    warn "eqy not found on PATH — equivalence checks will silently report as non-equivalent." \
         " Install the OSS CAD Suite (bundles eqy/sby) if you need EQY verification."
fi

# ---------------------------------------------------------------------------
section "2. Python dependencies for eda_runner.py"
if python3 -c "import yaml, jinja2" >/dev/null 2>&1; then
    pass "pyyaml + jinja2 importable via system python3"
else
    fail "pyyaml/jinja2 missing — run: pip3 install pyyaml jinja2"
fi

# ---------------------------------------------------------------------------
section "3. Required project files"
if [[ -f "pdk/sky130_fd_sc_hd__tt_025C_1v80.lib" ]]; then
    pass "Liberty PDK file present"
else
    fail "Missing pdk/sky130_fd_sc_hd__tt_025C_1v80.lib"
fi

RTL_COUNT=$(find dataset/raw_rtl -maxdepth 1 -name '*.v' -o -name '*.sv' 2>/dev/null | wc -l)
if [[ "$RTL_COUNT" -gt 0 ]]; then
    pass "dataset/raw_rtl contains $RTL_COUNT RTL files"
else
    fail "dataset/raw_rtl is empty or missing"
fi

if [[ -f "dataset/raw_rtl/${TEST_MODULE}.v" ]]; then
    pass "Test module dataset/raw_rtl/${TEST_MODULE}.v present"
else
    fail "Test module dataset/raw_rtl/${TEST_MODULE}.v not found — cannot run pipeline test"
    echo
    echo "Aborting further tests (no test module available)."
    echo "Summary: $PASS passed, $FAIL failed, $WARN warnings."
    exit 1
fi

# ---------------------------------------------------------------------------
section "4. Module discovery (--dry-run)"
DRYRUN_OUT="$SCRATCH/dryrun.log"
if python3 scripts/eda_runner.py --dry-run --limit 5 > "$DRYRUN_OUT" 2>&1; then
    N=$(grep -c "module=" "$DRYRUN_OUT" || true)
    if [[ "$N" -gt 0 ]]; then
        pass "Discovered $N module(s) in dry-run listing"
    else
        fail "Dry-run produced no module listing (see $DRYRUN_OUT)"
    fi
else
    fail "eda_runner.py --dry-run exited non-zero (see $DRYRUN_OUT)"
fi

# ---------------------------------------------------------------------------
section "5. Full pipeline run on '${TEST_MODULE}' (synth -> STA -> mutate -> EQY)"
# Build an isolated config so this never touches the real generated_dataset/ files.
TMP_CONFIG="$SCRATCH/test_config.yaml"
cat > "$TMP_CONFIG" <<EOF
paths:
  liberty: pdk/sky130_fd_sc_hd__tt_025C_1v80.lib
  raw_rtl: dataset/raw_rtl/
  output_dataset: $SCRATCH/timing_dataset_test.jsonl
  failures_log: $SCRATCH/failures_test.jsonl
  workdir: $SCRATCH/workdir_test/

tools:
  yosys: yosys
  sta: sta
  eqy: eqy

synthesis:
  target_clock_ghz: [1.0]
  input_delay_fraction: 0.2
  output_delay_fraction: 0.2
  timeout_seconds: 120

mutations:
  enabled:
    - pipeline_insert
    - register_retime
    - expression_balance
  max_per_module: 3

dataset:
  skip_patterns:
    - "*_mon.v"
    - "*_test.v"
    - "*_test_top.v"
    - "sas_*.v"
    - "playback_*.v"
    - "*.t.v"
EOF

RUN_OUT="$SCRATCH/run.log"
if python3 scripts/eda_runner.py --config "$TMP_CONFIG" --module "$TEST_MODULE" > "$RUN_OUT" 2>&1; then
    pass "eda_runner.py completed without a fatal error"
else
    fail "eda_runner.py exited non-zero — see $RUN_OUT"
fi

if grep -q "synth OK" "$RUN_OUT"; then
    pass "Synthesis + STA succeeded for ${TEST_MODULE} ($(grep 'synth OK' "$RUN_OUT" | sed 's/^\[.*\] //'))"
else
    fail "No 'synth OK' line in run log — synthesis or STA likely failed (see $RUN_OUT)"
fi

RECORD_COUNT=0
if [[ -f "$SCRATCH/timing_dataset_test.jsonl" ]]; then
    RECORD_COUNT=$(wc -l < "$SCRATCH/timing_dataset_test.jsonl" | tr -d ' ')
fi
if [[ "$RECORD_COUNT" -gt 0 ]]; then
    pass "$RECORD_COUNT training record(s) written (mutation + re-synth + re-STA succeeded)"
else
    warn "0 training records produced — mutations may not apply cleanly to ${TEST_MODULE}," \
         " or downstream synth/STA of the mutated variant failed. Check $RUN_OUT and" \
         " $SCRATCH/failures_test.jsonl (if present)."
fi

if [[ "$RECORD_COUNT" -gt 0 ]]; then
    if python3 -c "
import json
with open('$SCRATCH/timing_dataset_test.jsonl') as f:
    rec = json.loads(f.readline())
assert 'original_rtl' in rec and 'fixed_rtl' in rec and 'timing_report' in rec
assert 'wns_ns' in rec['timing_report']
print('wns_before =', rec['timing_report']['wns_ns'], '| wns_after =', rec['fixed_timing_report']['wns_ns'], '| eqy =', rec['equivalence_verified'])
" > "$SCRATCH/record_check.log" 2>&1; then
        pass "Record schema valid ($(cat "$SCRATCH/record_check.log"))"
    else
        fail "Record schema check failed (see $SCRATCH/record_check.log)"
    fi
fi

# ---------------------------------------------------------------------------
section "6. SFT dataset formatting (prepare_sft_dataset.py)"
if [[ "$RECORD_COUNT" -gt 0 ]]; then
    SFT_OUT="$SCRATCH/sft_test.jsonl"
    if python3 scripts/prepare_sft_dataset.py \
        --input "$SCRATCH/timing_dataset_test.jsonl" \
        --output "$SFT_OUT" > "$SCRATCH/prepare_sft.log" 2>&1; then
        SFT_LINES=$(wc -l < "$SFT_OUT" 2>/dev/null | tr -d ' ')
        if [[ "${SFT_LINES:-0}" -gt 0 ]] && python3 -c "
import json
rec = json.loads(open('$SFT_OUT').readline())
roles = [m['role'] for m in rec['messages']]
assert roles == ['system', 'user', 'assistant']
" > /dev/null 2>&1; then
            pass "ChatML SFT record(s) generated with correct system/user/assistant structure"
        else
            fail "prepare_sft_dataset.py output malformed (see $SFT_OUT)"
        fi
    else
        fail "prepare_sft_dataset.py exited non-zero (see $SCRATCH/prepare_sft.log)"
    fi
else
    warn "Skipped — no timing records from step 5 to convert"
fi

# ---------------------------------------------------------------------------
section "7. ML training stack (informational only — not executed by this script)"
if [[ -x "venv_phase2/bin/python3" ]] && venv_phase2/bin/python3 -c "import torch, transformers, peft, trl, bitsandbytes" >/dev/null 2>&1; then
    pass "venv_phase2 has torch/transformers/peft/trl/bitsandbytes installed"
else
    warn "venv_phase2 does not have the ML training stack installed (or pip itself is missing)." \
         " Training/evaluation scripts (train_sft.py, train_grpo_rlvr.py, evaluate_model.py)" \
         " will not run until you install: torch transformers peft trl bitsandbytes datasets accelerate"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    pass "GPU detected: $(nvidia-smi -L | head -1)"
else
    warn "No GPU detected via nvidia-smi — training scripts assume CUDA is available"
fi

# ---------------------------------------------------------------------------
section "Summary"
echo "  Passed   : $PASS"
echo "  Failed   : $FAIL"
echo "  Warnings : $WARN"
echo
if [[ "$FAIL" -eq 0 ]]; then
    echo "RESULT: EDA pipeline (Stage 0/1) is working end to end."
    exit 0
else
    echo "RESULT: One or more hard checks failed — see [FAIL] lines above."
    exit 1
fi
