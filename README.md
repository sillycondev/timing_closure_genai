# LLM-Driven RTL Timing Closure, Verified by a Real ASIC Flow

A small code LLM (Qwen2.5-Coder-1.5B, 4-bit QLoRA) is fine-tuned to rewrite Verilog that fails timing. Nothing it produces is trusted: every candidate is re-synthesised with **Yosys**, timed with **OpenSTA** on the **SkyWater 130nm** (`sky130_fd_sc_hd`) library, and checked for functional equivalence before it can count as an improvement.

The main result is about the verification as much as the model. Timing-only scoring makes the model look much better than it is. Once a functional gate is added, the gains shrink and become honest, and the same gate exposed bad training data that had been assumed correct.

---

## Pipeline

```
 violating RTL + OpenSTA report
            │
            ▼
   LLM (Qwen2.5-Coder-1.5B + QLoRA)  ──►  N candidate rewrites (1 greedy + N-1 sampled)
                                                  │
                    ┌─────────────────────────────┼──────────────────────────────┐
                    ▼                             ▼                              ▼
          interface check               Yosys + OpenSTA (sky130)        functional check
       (same module name/ports)        must synthesise and improve WNS   EQY (cycle-exact) or
                                                                         latency-tolerant simulation
                    └─────────────────────────────┬──────────────────────────────┘
                                                  ▼
                        best verified candidate, else fall back to the original RTL
```

The fallback means the reported result can never be worse than the input design: zero regressions by construction.

## Results

**Held-out benchmark:** 20 modules the model never saw, all violating timing at 1 GHz: 10 synthetic (from the template generator) and 10 real modules from the project's RTL corpus. The run logs behind every number are in `logs/`.

Two SFT runs on the same 134-record training set and the same benchmark:

| Metric (avg WNS gain, 20 modules) | Run A (`bestofn_v134_gated.log`) | Run B (`bestofn_v4_gated.log`) |
|---|---|---|
| Single-shot, greedy, no checks | +7.23 ns (13/20 synthesise) | +2.27 ns (12/20 synthesise) |
| Best-of-4, timing only (no functional gate) | +4.73 ns | +8.14 ns |
| **Best-of-4 + functional gate + fallback** | **+4.10 ns (11/20 improved)** | **+1.39 ns (3/20 improved)** |
| — synthetic half | +7.50 ns (10/10) | +2.09 ns (2/10) |
| — real half | +0.69 ns (1/10) | +0.69 ns (1/10) |
| Timing closure (final WNS ≥ 0) | 1/20 | 1/20 |
| Reference-fix oracle, synthetic half | +7.48 ns | +7.48 ns |

How to read this:
- **The functional gate matters.** In run B, timing-only selection reports +8.14 ns, but only +1.39 ns of that survives a functional check. Most of the "faster" rewrites were wrong.
- **Run-to-run variance is large.** Identical data and settings gave +4.10 ns and +1.39 ns. Both numbers are reported here; neither should be quoted alone.
- **Almost all of the gain is on synthetic modules.** Real RTL improved on 1 module of 10 in both runs.

**Real-RTL generalisation:** `screen_external_rtl.py` screened 10 open-source repositories (verilog-ethernet, verilog-pcie, verilog-axi, verilog-i2c, verilog-wishbone, verilog-uart, wb2axip, picorv32, zipcpu, cores) and found **114 self-contained modules that violate timing**. The model produced **no verified fix on any of them** (`logs/bestofn_realheldout_v4.log`: 0/13 on a held-out subset). On 200–400-line designs it re-transcribes the whole module and introduces copy errors instead of applying a local transform. The verification stack rejects all of it, which is the intended behaviour. The proposed next step is a **targeted-edit formulation**: the model emits a small patch to the critical path rather than regenerating the file.

## What the verification caught

- **EQY cannot judge pipelining.** A pipelining fix changes latency, so cycle-exact formal equivalence always reports a failure, whether the rewrite is right or wrong. `latency_equiv.py` simulates the original and the rewrite side by side (Icarus Verilog, ~3000 cycles of mixed random and boundary stimulus) and accepts only if the outputs match at one fixed latency shift. It is a falsifier, not a proof.
- **Bad training data.** Three of the six synthetic template families (`horner`, `priority_mux`, `shift_acc`) generated "fixed" pipelined variants that were not functionally equivalent to the originals. That was over half of the training data. The generator now gates every pair on the simulation check.
- **Plausible but wrong rewrites.** Before the functional gate existed, three of four "improved" real modules turned out to be functionally broken.
- **Pipeline bugs.** Generation was capped at 1024 tokens, which truncated 31 of 33 real-module rewrites mid-file; the budget now scales with module size. OpenSTA's netlist reader rejected `signed`, silently dropping 41 of 155 external modules. Orphaned Yosys processes are now killed on timeout.

## Model and training data

- **Base model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`, 4-bit NF4 QLoRA (r = 16, α = 32), 3 epochs, fits in 8 GB of VRAM (trained on an RTX 4060 laptop GPU).
- **Training data:** template-generated pairs of violating and fixed RTL across operand widths from 8 to 512 bits. Every pair is synthesised and timed, and kept only if the fix improves WNS and passes the functional check. Generator v3 kept 162 of 261 pairs; the final SFT set has 134 records (36 removed as non-equivalent, 154 for improving WNS by less than 0.05 ns, 38 by a per-family cap).
- **Tried and reverted:** "already meets timing, leave it alone" examples (`identity_generate.py`) made the model regurgitate memorised modules. Forward regex mutation (`eda_runner.py`, `mutations.py`) has a low hit rate because Yosys/ABC already performs most source-level rebalancing.
- **Not used for the results above:** `train_grpo_rlvr.py` implements a GRPO reinforcement-learning stage with EDA rewards, but it has not been trained or evaluated.

## Repository layout

```
scripts/
  synth_corpus_generate.py   template-based violating/fixed RTL pairs (the main data source)
  prepare_sft_dataset.py     filter and format pairs into ChatML for SFT
  train_sft.py               QLoRA supervised fine-tuning
  build_heldout_benchmark.py held-out benchmark of genuinely violating, unseen modules
  evaluate_bestofn.py        best-of-N with interface, timing and functional gates + fallback
  latency_equiv.py           latency-tolerant simulation equivalence check
  screen_external_rtl.py     screen cloned open-source repos for violating modules
  harvest_real_fixes.py      rejection sampling on real modules, verified end to end
  demo.py                    single-module demo: baseline WNS, one rewrite, new WNS, diff
  eda_runner.py, mutations.py, rtl_expr.py, deopt_generate.py, rejection_sample.py
                             earlier data-generation approaches (kept for reference)
  train_grpo_rlvr.py         GRPO stage (implemented, not yet used)
  templates/                 Yosys, OpenSTA and EQY script templates
  config.yaml                tool paths, clock targets, dataset options
rtl/                         demo designs
pdk/                         sky130_fd_sc_hd typical-corner Liberty file
logs/                        run logs for the results above
```

The datasets, cloned external RTL and LoRA weights are not included in the repository (see `.gitignore`).

## Setup

Linux x86_64. A GPU with 8 GB of VRAM is enough for training and inference.

1. **EDA tools:** Yosys and OpenSTA on your `PATH`, plus the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) for EQY, SymbiYosys and Icarus Verilog. Set the `eqy` path in `scripts/config.yaml`. `latency_equiv.py` expects the suite at `~/tools/oss-cad-suite/bin`; edit `TOOLS` there if yours is elsewhere.
2. **Python:**
   ```bash
   pip install torch transformers peft trl bitsandbytes datasets accelerate pyyaml jinja2
   ```

## Usage

```bash
# 1. Generate verified training pairs and format them for SFT
python3 scripts/synth_corpus_generate.py --output dataset/generated_dataset/synthetic_dataset.jsonl
python3 scripts/prepare_sft_dataset.py --input dataset/generated_dataset/synthetic_dataset.jsonl --max-per-type 30

# 2. Fine-tune
python3 scripts/train_sft.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --epochs 3

# 3. Build the held-out benchmark and evaluate with tool-verified best-of-4
python3 scripts/build_heldout_benchmark.py
python3 scripts/evaluate_bestofn.py --model models/sft_lora_checkpoint --n 4

# Single-module demo
python3 scripts/demo.py rtl/demo_adder_chain.v --model models/sft_lora_checkpoint --clock-ghz 1.0

# Check a rewrite for latency-tolerant equivalence on its own
python3 scripts/latency_equiv.py original.v rewritten.v <module_name> --clock clk
```

## License

Apache License 2.0 (see `LICENSE`). The Liberty file in `pdk/` is from the SkyWater SKY130 open-source PDK and is covered by that project's Apache 2.0 terms.
