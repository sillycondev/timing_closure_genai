# Testing This Setup

This checks whether the timing-closure GenAI pipeline actually works on your
machine, in increasing order of cost: tool availability → EDA pipeline →
(optional) ML training stack.

## What's already verified on this machine

| Component | Status |
|---|---|
| `yosys` (synthesis) | ✅ on PATH (0.33) |
| `sta` / OpenSTA (timing analysis) | ✅ on PATH (3.1.0) |
| `eqy` / `sby` (formal equivalence) | ❌ not installed — equivalence checks will report `False` instead of erroring |
| `pyyaml`, `jinja2` (needed by `eda_runner.py`) | ✅ available via system `python3` |
| `venv_phase2` (ML training env) | ⚠️ exists but empty — no `pip`, no `torch`/`transformers`/`peft`/`trl` |
| GPU | ✅ NVIDIA RTX 4060 Laptop detected |

So: **the EDA data-generation pipeline (Stage 0/1) is testable right now.**
The ML training stages (SFT, GRPO) need the venv populated first — see Step 4.

---

## Step 1 — Run the automated smoke test

A non-destructive script has been added at [scripts/test_setup.sh](scripts/test_setup.sh).
It never touches your real `dataset/generated_dataset/*.jsonl` files — it
builds a throwaway config pointing at a `/tmp` scratch directory.

```bash
./scripts/test_setup.sh
```

It checks, in order:
1. `yosys` / `sta` on PATH (hard requirement), `eqy` (soft — warns if missing)
2. Python deps (`pyyaml`, `jinja2`) importable
3. `pdk/sky130_fd_sc_hd__tt_025C_1v80.lib` and `dataset/raw_rtl/` exist
4. `eda_runner.py --dry-run` discovers modules
5. A full pipeline run on `Counter.v` — synth → STA → mutate → re-synth →
   re-STA → EQY — and validates the resulting JSON record schema
6. `prepare_sft_dataset.py` converts that record into a valid ChatML triple
7. (Informational) whether the ML training stack + GPU are ready

Exit code is `0` only if every **hard** check (steps 1, 2, 4, 5, 6) passes.

To keep the generated scratch files for inspection instead of deleting them:

```bash
./scripts/test_setup.sh --keep
```

It will print the scratch directory path (e.g. `/tmp/timing_closure_test.XXXXXX`)
containing `timing_dataset_test.jsonl`, `sft_test.jsonl`, Yosys/OpenSTA logs,
and the netlists — useful for debugging a `[FAIL]` line.

**Expected result on a healthy setup:** all steps pass except the `eqy`
warning (unless you've installed OSS CAD Suite) and the ML-stack warning
(unless you've done Step 4 below).

---

## Step 2 — Run the real pipeline on a handful of modules (manual)

Once the smoke test passes, try it against the real dataset in dry-run first,
then for real on a small `--limit`, so you don't wait for all ~650 RTL files:

```bash
# See what would be processed, no EDA tools invoked
python3 scripts/eda_runner.py --dry-run --limit 10

# Actually run synth + STA + mutations + EQY on 5 modules
python3 scripts/eda_runner.py --limit 5

# Inspect what it produced
tail -n 5 dataset/generated_dataset/timing_dataset.jsonl
tail -n 5 dataset/generated_dataset/failures.jsonl
cat dataset/workdir/eda_runner.log
```

Note: this **appends** to the real `timing_dataset.jsonl` / `failures.jsonl`,
unlike the smoke test. If you want to reset and regenerate from scratch:

```bash
rm dataset/generated_dataset/timing_dataset.jsonl dataset/generated_dataset/failures.jsonl
python3 scripts/eda_runner.py --parallel 4   # full corpus, parallelized
```

Then rebuild the ChatML dataset:

```bash
python3 scripts/prepare_sft_dataset.py
```

---

## Step 3 — Sanity-check a single module by hand (optional, deepest check)

If a `[FAIL]` shows up and you want to see raw tool output rather than parsed
JSON, run the module in isolation and read the generated TCL/log files
directly:

```bash
python3 scripts/eda_runner.py --module Counter --skip-mutations
cat dataset/workdir/Counter/synth.tcl        # generated Yosys script
cat dataset/workdir/Counter/run_sta.tcl      # generated OpenSTA script
```

`dataset/workdir/<module>/` will contain the netlist, the TCL scripts, and
Yosys/OpenSTA stdout — this is the fastest way to tell whether a failure is a
Yosys elaboration error, a missing clock port, or an OpenSTA constraint
problem.

---

## Step 4 — (Optional) Set up and test the ML training stack

Only needed if you want to actually test `train_sft.py` / `train_grpo_rlvr.py`
/ `evaluate_model.py`. `venv_phase2` currently has no packages — either
populate it or use system Python:

```bash
python3 -m ensurepip --upgrade   # venv_phase2/bin/python3 has no pip yet
# or, simpler: recreate the venv with pip included
python3 -m venv --upgrade-deps venv_phase2
source venv_phase2/bin/activate
pip install torch transformers peft trl bitsandbytes datasets accelerate pyyaml jinja2
```

Then, once `dataset/generated_dataset/sft_dataset.jsonl` has real records
(from Step 2):

```bash
# Stage 1: SFT (quick sanity run — 1 epoch on a small dataset)
python3 scripts/train_sft.py --epochs 1

# Stage 2: GRPO RLVR (uses the live Yosys/OpenSTA reward function)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_grpo_rlvr.py \
    --model models/sft_lora_checkpoint --epochs 1 --num_generations 2

# Evaluate
python3 scripts/evaluate_model.py --model models/sft_lora_checkpoint --limit 5
```

Each of these will download `Qwen/Qwen2.5-Coder-1.5B-Instruct` from Hugging
Face on first run (~3 GB) and needs the GPU — expect this to take minutes,
not seconds, unlike Steps 1–3.

---

## Known limitations surfaced by testing

- **`eqy`/`sby` are not installed** on this machine, so every
  `equivalence_verified` field in generated records will be `false` even for
  mutations that are actually equivalent. Install the
  [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) if you need
  real formal verification signal.
- **Mutations don't always change timing.** E.g. `Counter.v` only has a
  single-operator increment (`q <= q + 1`), so `register_retime` /
  `pipeline_insert` find nothing complex to restructure, and the "fixed"
  variant's WNS equals the original. This is a real property of the
  regex-based mutation engine, not a test-script bug — pick modules with
  wider combinational expressions (e.g. `fpu_add.v`, `mul64.v`) to exercise
  mutations that actually move the needle.
- **`venv_phase2` is an empty shell** — `pip` itself isn't installed inside
  it. Step 4 above recreates it properly.
