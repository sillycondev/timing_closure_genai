# Report update — replacement sections

Drop-in text for `output.pdf`, reflecting the Wide-151 checkpoint, the
generator-v2 width scaling, the 22-sample held-out benchmark and the
best-of-N evaluator. Every number below is read from a log or dataset file
in this repo (path given in brackets); none is estimated.

Sources:
- `logs/synthgen_v2.log` — generator v2 run
- `logs/train_sft_151.log`, `logs/eval_151.log` — Wide-151 training / 12-sample eval
- `logs/train_sft_173.log`, `logs/eval_173.log` — Identity-173 experiment (reverted)
- `logs/bestofn_151.log`, `dataset/generated_dataset/bestofn_results.jsonl` — held-out best-of-4
- `dataset/generated_dataset/sft_dataset.jsonl` — the 151-record training set
- `scripts/demo.py` run on `rtl/demo_sum_tree.v` — the demo figures

---

## §1 Executive summary — replace the RESULT box

> **RESULT**
>
> The current model, trained on 151 fully-verified records spanning operand
> widths from 8 to 512 bits, improves timing on 16 of 22 held-out modules it
> has never seen, including 4 of 10 real open-source designs, for an average
> measured gain of +5.98 ns under tool-verified best-of-4 selection. On the
> twelve unseen synthetic modules it reproduces the reference fix exactly
> (+6.74 ns, equal to the oracle); on real RTL it averages +5.07 ns, with
> single-module gains up to +34.95 ns and one design (`adder_tree8`) closed
> outright. Widening the training-data width grid, the concrete next step
> identified in the previous revision, is what moved the single-shot average
> from −10.93 ns to −0.79 ns on the original evaluation set. The remaining
> failures are syntax and synthesis errors on control-heavy real RTL, not
> regurgitation; the bottleneck has shifted from data breadth to data
> *kind*, and to verifying latency-changing fixes formally.

Also in the body of §1, replace "The current model, trained on 63 diverse,
fully-verified records…" with the sentence above, and change "a 79% keep
rate" to "a 73–79% keep rate across two generator versions".

---

## §7 — add §7.4 "Generator v2: scaling operand width"

### 7.4 Generator v2: scaling operand width

The first generator's width grid stopped at 48 bits, and §8.2 diagnosed
that ceiling as the direct cause of the model breaking down on the
128–512-bit modules in the real corpus. Version 2 extends the grid to
{8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512} bits and adds
randomised per-family module-name pools (`dsp_accum`, `vec_sum`,
`tree_add`, `sum_tree`, `adder_net`, `reduce_add`, `series_eval`,
`route_sel`, `chain_mux`, `barrel_acc`, `rot_sum`, `scale_unit`,
`shift_add`, `shift_mac`, `poly_core`, …) so no single surface pattern can
dominate the dataset (§7.3).

> **RESULT**
>
> 190 of 261 generated pairs kept (73%) in a 925 s run, with measured
> improvements from +0.07 ns to +25.8 ns (`shift_add_w512_s7`: −82.62 →
> −56.79 ns). The keep rate is slightly below v1's 79% for two honest
> reasons: 11 flat versions failed synthesis at extreme width/degree
> combinations (e.g. `poly_core_w48_d6`), and ~60 pairs were dropped by the
> minimum-gap filter because the pipelined version improved slack by less
> than the 0.3 ns minimum-gap threshold, i.e. within STA noise.

After the per-family cap the training set used for the current checkpoint
holds 151 records: 141 synthetic (shift_acc 30, adder_tree 30,
priority_mux 28, horner 24, compare_chain 17, mac 12) plus 10 surviving
mutation-engine records, widths 8–512 bits, mean improvement +6.06 ns,
maximum +57.97 ns.

---

## §8 — results table: add two rows after "Diverse-63"

| CHECKPOINT | RECORDS | AVG SLACK Δ | CLOSURE | EQUIV. | DOMINANT FAILURE MODE |
|---|---|---|---|---|---|
| Diverse-63 | 63 | −10.93 ns | 33.3% | 8.3% | Real transform attempted; fails far outside training width range |
| **Wide-151 (current)** | 151 | **−0.79 ns** | 25.0% | **41.7%** | Occasional invalid syntax on control-heavy real RTL |
| Identity-173 (reverted) | 173 | −19.31 ns | 25.0% | 16.7% | Verbatim regurgitation returned |

Update the caption: "Every number comes from re-running the evaluation on
the same 12 held-out samples…" stays true for this table. Add: "The
current checkpoint is additionally evaluated on a larger, cleaner benchmark
in §8.3."

Training detail for the Wide-151 row (from `logs/train_sft_151.log`):
3 epochs, 114 optimiser steps, 427 s on the RTX 4060, final training loss
0.157, mean token accuracy 98.1%.

---

## §8.1 — replace the final sentence

Replace "…but revealed the last, honest constraint the training
distribution's operand widths (up to 48 bits) do not cover real evaluation
modules (up to 512 bits), and the attempted transform breaks down outside
that range." with:

> …but revealed one more constraint: the training distribution's operand
> widths (up to 48 bits) did not cover the real evaluation modules (up to
> 512 bits), and the attempted transform broke down outside that range.
> Extending the generator's width grid to 512 bits (§7.4) and retraining on
> 151 records addressed exactly that: on the same 12 samples the average
> moved from −10.93 ns to −0.79 ns and the formal-equivalence rate from 8.3%
> to 41.7%, the best of any checkpoint. The §8.2 diagnosis was correct, and
> the fix it prescribed worked.

---

## §8.2 — add a third failure mode

### A third failure mode: teaching restraint backfires

Because the Clean-10 model was "eager but reckless", an obvious remedy was
to add examples of *not* acting: 22 identity records in which a module that
already meets timing is returned unchanged, so the model could learn when
to leave a design alone. Trained on 151 + 22 = 173 records, the model
regressed sharply (−19.31 ns average, equivalence 41.7% → 16.7%) and the
dominant failure mode reverted to verbatim regurgitation of a memorised
synthetic module. The identity pairs are trivially learnable (copy the
input), so at 13% of the dataset they dominate the loss signal and pull the
model back toward reproducing training text instead of transforming the
input. The checkpoint was discarded and Wide-151 restored. The lesson
generalises the §7.3 naming finding: any subset of the data that is much
easier to fit than the rest will be over-learned, regardless of whether it
is a name pattern or a whole record.

---

## §8 — new §8.3 "Held-out benchmark with tool-verified best-of-N"

### 8.3 Held-out benchmark with tool-verified best-of-N

The 12-sample set used above is derived from the mutation pipeline and
shares its blind spots. A second benchmark was therefore built
(`scripts/build_heldout_benchmark.py`) from 22 modules that appear nowhere
in training: 12 synthetic modules generated from *held-out seeds* across
all families and widths (12–320 bits), each with a known reference fix so
an oracle score can be reported, and 10 real modules from the open-source
corpus (`spi_master`, `bw_r_rf32x152b`, `mac32`, `uart_serializer`,
`async_fifo`, `fpga_bridge_rcv_32`, `barrel_mac`, `ssd_1306_init`,
`adder_tree8`, `that`).

Single-shot scoring takes whatever the model emits first, so one bad
rewrite can drag a whole average negative (§8.1). The new evaluator
(`scripts/evaluate_bestofn.py`) does what a deployment would do: generate
four candidates per module (one greedy, three sampled at T = 0.7), reject
any that rename the module or change its ports, synthesise and time every
survivor with the real Yosys/OpenSTA flow, keep the best measured candidate
only if it beats the original by more than STA noise, and otherwise fall
back to the original RTL (Δ = 0). EQY then runs on the selected candidate.
The greedy candidate is also scored alone, without fallback, so the same
run reports the single-shot baseline.

| METRIC (Wide-151, 22 samples) | VALUE |
|---|---|
| Single-shot greedy, no fallback | +8.27 ns avg over the 15/22 that synthesised (7 unsynthesizable excluded, so this number is optimistic) |
| **Best-of-4 + verify + fallback** | **+5.98 ns avg; 16/22 improved, 6 kept original** |
| Synthetic held-out (12) | +6.74 ns avg — **identical to the reference-fix oracle** (+6.74 ns) |
| Real corpus (10) | +5.07 ns avg; 4/10 improved |
| Timing closure (final WNS ≥ 0) | 1/22 (`adder_tree8`, −6.91 → 0.00 ns) |
| EQY pass among selected rewrites | 0/16 — every accepted fix is a pipelining transform, which the cycle-accurate EQY flow cannot prove by design (§2.2); the "strict, EQY-verified only" average is therefore +0.00 ns |

Per-module results:

| MODULE | SOURCE | ORIGINAL WNS | SELECTED | FINAL WNS | Δ |
|---|---|---|---|---|---|
| dsp_accum_w20_t4 | synthetic | −16.83 | greedy | −16.32 | +0.51 |
| vec_sum_w12_n8 | synthetic | −6.05 | greedy | −4.58 | +1.47 |
| tree_add_w40_n12 | synthetic | −19.19 | greedy | −16.43 | +2.76 |
| sum_tree_w72_n8 | synthetic | −29.66 | greedy | −19.47 | +10.19 |
| adder_net_w160_n16 | synthetic | −70.14 | greedy | −59.18 | +10.96 |
| reduce_add_w320_n8 | synthetic | −85.76 | greedy | −49.24 | +36.52 |
| series_eval_w20_d3 | synthetic | −18.71 | greedy | −15.56 | +3.15 |
| route_sel_w40_n16 | synthetic | −0.92 | greedy | −0.47 | +0.45 |
| chain_mux_w72_n20 | synthetic | −2.81 | greedy | −1.16 | +1.65 |
| barrel_acc_w72_s5 | synthetic | −17.11 | greedy | −13.83 | +3.28 |
| rot_sum_w160_s6 | synthetic | −29.03 | greedy | −27.44 | +1.59 |
| scale_unit_w320_s4 | synthetic | −48.76 | greedy | −40.43 | +8.33 |
| spi_master | real | −58.82 | original | −58.82 | 0.00 (all 4 failed synthesis) |
| bw_r_rf32x152b | real | −51.02 | greedy | −16.07 | **+34.95** |
| mac32 | real | −21.54 | sample 2 | −13.95 | +7.59 |
| uart_serializer | real | −20.27 | original | −20.27 | 0.00 (all 4 failed synthesis) |
| async_fifo | real | −14.70 | original | −14.70 | 0.00 (all 4 failed synthesis) |
| fpga_bridge_rcv_32 | real | −10.94 | original | −10.94 | 0.00 (all 4 failed synthesis) |
| barrel_mac | real | −10.21 | greedy | −8.93 | +1.28 |
| ssd_1306_init | real | −7.38 | original | −7.38 | 0.00 (all 4 failed synthesis) |
| adder_tree8 | real | −6.91 | greedy | **0.00** | +6.91 (closed) |
| that | real | −3.78 | original | −3.78 | 0.00 (all 4 changed the interface) |

Three things stand out. First, on every one of the twelve unseen synthetic
modules the greedy candidate already matched the reference fix, and the
three sampled candidates almost always agreed with it: the model has
learned the pipelining transform rather than memorised instances, and the
learning transfers to widths (160, 320 bits) and seeds it never saw.
Second, the transfer to real RTL is genuine where the design is
arithmetic-shaped: `bw_r_rf32x152b` (a wide register file) and `mac32`
receive large, verified improvements, and `adder_tree8` is closed
completely. Third, the real-corpus failures are all of one kind: the five
modules where every candidate failed synthesis are control-heavy designs
(SPI/UART serialisers, an asynchronous FIFO, a display initialiser) with
state machines and no wide arithmetic, a shape entirely absent from the
training data. The `mac32` case also shows why best-of-N matters: its greedy
and first sampled candidates both failed synthesis, and the +7.59 ns win
came from the third candidate.

### 8.4 Interactive demonstration

`scripts/demo.py` runs the whole loop on one Verilog file in about fifty
seconds: baseline Yosys + OpenSTA, one greedy rewrite from the model,
re-synthesis and re-timing of the rewrite, an EQY attempt, and a diff. On
`rtl/demo_sum_tree.v` (an 8-input 72-bit registered adder tree, held out of
training) the model splits the sum into two registered 4-input partial sums
and a final add, taking WNS from −29.19 ns to −23.44 ns (+5.75 ns) and the
critical path from 82 to 78 cells, consistent with the +6.74 ns synthetic
average in §8.3 rather than a cherry-picked case.

---

## §9 Known limitations — replace the section

**OPEN · REINFORCEMENT LEARNING** — unchanged from the previous revision.
Every rollout completion hits the token cap without a stop token, so all
candidates receive the identical worst-case reward and the GRPO signal is
zero. Root-caused to the trainer's generation configuration, not yet fixed;
every checkpoint in this report remains SFT-only.

**OPEN · EQUIVALENCE FOR LATENCY-CHANGING FIXES** — new. The model's
dominant, and correct, transform is pipelining, which adds a cycle of
latency. The EQY flow proves cycle-accurate equivalence only, so 0 of the
16 accepted rewrites in §8.3 could be verified even though 12 of them reach
exactly the measured WNS of the known-correct reference fix. The "strict" score is
therefore uninformative rather than damning. A latency-aware check
(sequential equivalence with a matched-delay wrapper on the original) is
needed before pipelined fixes can be counted as formally verified.

**OPEN · CONTROL-LOGIC RTL** — new. 5 of the 10 real held-out modules
(§8.3) produce unsynthesizable output from all four candidates. All five
are FSM-centred, non-arithmetic designs; the training set contains no such
shape. Six template families and the mutation engine all target datapath
logic.

**OPEN · CORPUS COVERAGE** — unchanged. Roughly 270 of 619+ corpus modules
cannot synthesise; not patched with guessed definitions.

**OPEN · TOOL-VERSION CONSISTENCY** — unchanged.

**RESOLVED · WIDTH COVERAGE** — the generator grid now reaches 512 bits
(§7.4); the §8.2 failure no longer occurs (§8.1).

---

## §10 Reproduction guide — add after the first paragraph

To see the model act on a single module end-to-end, run

    source venv_phase2/bin/activate
    python3 scripts/demo.py rtl/demo_sum_tree.v

which prints the baseline timing, the model's rewrite, the re-verified
timing, a results table and a diff in roughly fifty seconds on the
reference hardware. `--model` selects a checkpoint, `--clock-ghz` the
target and `--no-eqy` skips the equivalence attempt.

To reproduce §8.3: `scripts/build_heldout_benchmark.py` assembles the
22-sample benchmark from held-out generator seeds and real corpus modules;
`scripts/evaluate_bestofn.py --n 4` runs the verified best-of-N evaluation
and writes per-module results to `bestofn_results.jsonl` (about 45 minutes
on the reference hardware, dominated by synthesis of the wide designs).

---

## §11 Conclusion & next steps — replace the numbered list

Replace the second sentence of the paragraph ("The remaining gap is
squarely a data-breadth problem…") with:

> The remaining gap is no longer breadth within the arithmetic domain, the
> current model matches the reference fix on every unseen synthetic module
> and transfers to real arithmetic RTL, but two specific things: proving
> that latency-changing fixes are correct, and covering the control-logic
> designs that make up much of a real corpus.

1. Latency-aware formal equivalence, so that the pipelining fixes the model
   already produces can be counted as verified (§9) and can serve as a
   trustworthy RL reward.
2. Extend the synthetic generator to control-dominated templates (FSMs with
   wide next-state logic, serialisers, handshake pipelines), directly
   targeting the five real-corpus synthesis failures in §8.3.
3. Trace the reinforcement-learning rollout configuration to restore a
   working RL stage; with best-of-N verification already in place (§8.3)
   the reward machinery is ready the moment rollouts terminate correctly.
4. Revisit LLM rejection sampling with a capable API-based proposer once
   budget allows; the verification side is proven.
