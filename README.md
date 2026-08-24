# Automated RTL Timing Closure GenAI Framework using GRPO RLVR & Open-Source EDA Tools

An end-to-end Machine Learning and Electronic Design Automation (EDA) framework for automated Verilog RTL timing closure. This project fine-tunes Large Language Models (LLMs) using **Group Relative Policy Optimization (GRPO RLVR)** with real-time feedback from open-source CAD tools (**Yosys Synthesis**, **OpenSTA Static Timing Analysis**, and **EQY Formal Equivalence Verification**) mapped to the **SkyWater 130nm Standard Cell Library**.

---

## 📌 Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                 LLM Policy Generation                       │
                        │        (Qwen2.5-Coder-1.5B-Instruct + QLoRA)               │
                        └──────────────────────────────┬──────────────────────────────┘
                                                       │  Generates RTL Candidates
                                                       ▼
                        ┌─────────────────────────────────────────────────────────────┐
                        │             Automated Real-Time EDA Feedback                │
                        │ ┌──────────────────────┬──────────────────────────────────┐ │
                        │ │   Yosys Synthesis    │  SkyWater 130nm Cell Library     │ │
                        │ ├──────────────────────┼──────────────────────────────────┤ │
                        │ │   OpenSTA Timing     │  Worst Negative Slack (WNS/TNS)   │ │
                        │ ├──────────────────────┼──────────────────────────────────┤ │
                        │ │ EQY Equivalence Check│  Formal Equivalence Proof        │ │
                        │ └──────────────────────┴──────────────────────────────────┘ │
                        └──────────────────────────────┬──────────────────────────────┘
                                                       │ Compute RL Rewards:
                                                       │   + Verilog Format (+1.0)
                                                       │   + Synthesis Pass / Fail (-3.0)
                                                       │   + ΔWNS Improvement (+1.0 x ΔWNS)
                                                       │   + Timing Closure Bonus (+3.0)
                                                       ▼
                        ┌─────────────────────────────────────────────────────────────┐
                        │                 GRPO RLVR Policy Update                     │
                        │       (Relative Advantage Estimation across G Candidates)  │
                        └─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
timing_closure_genai/
├── dataset/
│   ├── raw_rtl/                 # 176+ benchmark Verilog/SystemVerilog source files
│   ├── generated_dataset/
│   │   ├── timing_dataset.jsonl # 36 extracted timing violation records
│   │   └── sft_dataset.jsonl    # ChatML formatted training dataset
│   ├── workdir/                 # Workspace directory for synthesis & STA runs
│   └── workdir_eval/            # Workspace directory for model evaluation
├── models/
│   ├── sft_lora_checkpoint/     # Stage 1 SFT QLoRA adapter weights
│   └── grpo_rlvr_checkpoint/    # Stage 2 GRPO RLVR trained adapter weights
├── pdk/
│   └── sky130_fd_sc_hd__tt_025C_1v80.lib  # SkyWater 130nm Liberty Cell Library
├── rtl/
│   ├── gcd.v                    # Reference RTL designs
│   └── gcd_synth.v              # Synthesized gate-level netlists
├── scripts/
│   ├── config.yaml              # Centralized EDA toolchain configuration
│   ├── eda_runner.py            # Yosys, OpenSTA & EQY orchestration script
│   ├── mutations.py             # Structural Verilog mutation engine
│   ├── prepare_sft_dataset.py   # Dataset ChatML formatting tool
│   ├── train_sft.py             # Stage 1 4-bit QLoRA SFT training script
│   ├── train_grpo_rlvr.py       # Stage 2 GRPO RLVR training loop
│   ├── evaluate_model.py        # Benchmark evaluation & metrics script
│   └── templates/               # Jinja2 TCL & EQY script templates
│       ├── synth_sky130.tcl.j2
│       ├── run_sta.tcl.j2
│       └── verify_equiv.eqy.j2
└── README.md                    # Project documentation
```

---

## 🛠️ Prerequisites & Setup

### 1. Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8 GB VRAM (e.g. NVIDIA GeForce RTX 4060 / RTX 3060 / T4).
- **OS**: Linux x86_64.

### 2. Open-Source EDA Toolchain
Install the **OSS CAD Suite** (Yosys, OpenSTA, EQY, SymbiYosys):
```bash
# Sourced binaries must be accessible in your system PATH:
which yosys sta eqy sby
```

### 3. Python ML Dependencies
Install the required PyTorch and Hugging Face libraries:
```bash
pip3 install torch transformers peft trl bitsandbytes datasets accelerate pyyaml jinja2
```

---

## 🚀 Step-by-Step Instructions

### Step 1: Run EDA Pipeline & Generate Timing Dataset
Synthesize raw RTL benchmark modules, extract OpenSTA timing violations, apply structural mutations (`register_retime`, `pipeline_insert`, `expression_balance`), and produce training records:

```bash
# Run EDA dataset generation
python3 scripts/eda_runner.py

# Format into ChatML prompt-response pairs
python3 scripts/prepare_sft_dataset.py
```
*Output generated*: `dataset/generated_dataset/sft_dataset.jsonl`

---

### Step 2: Stage 1 Supervised Fine-Tuning (SFT Warmup)
Train the base model (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) with 4-bit NF4 QLoRA on ChatML timing fix pairs:

```bash
python3 scripts/train_sft.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --epochs 3
```
*Output generated*: `models/sft_lora_checkpoint`  
*Metrics achieved*: Loss = **0.3093**, Token Accuracy = **92.24%**.

---

### Step 3: Stage 2 GRPO RLVR Reinforcement Learning
Train the SFT policy model using Group Relative Policy Optimization with automated Yosys and OpenSTA reward feedback:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_grpo_rlvr.py \
    --model models/sft_lora_checkpoint \
    --epochs 1 \
    --num_generations 2
```
*Output generated*: `models/grpo_rlvr_checkpoint`  
*Reward Signal*: Verilog formatting, Yosys synthesis pass/fail, and OpenSTA Worst Negative Slack improvement ($\Delta\text{WNS}$).

---

### Step 4: Evaluate & Benchmark the Model
Evaluate the trained GRPO RLVR model across test RTL modules:

```bash
# Evaluate GRPO RLVR checkpoint
python3 scripts/evaluate_model.py --model models/grpo_rlvr_checkpoint --limit 10

# Compare against SFT baseline checkpoint
python3 scripts/evaluate_model.py --model models/sft_lora_checkpoint --limit 10
```

---

## 📊 Benchmark Results

Evaluated on test modules from the benchmark timing dataset:

| Benchmark Metric | Measured Score |
| :--- | :--- |
| **Syntax Pass Rate** | **100.0%** (10/10) |
| **Synthesis Pass Rate (Yosys + Sky130)** | **90.0%** (9/10) |
| **Timing Closure Rate (WNS $\ge$ 0)** | **40.0%** (4/10) |
| **Max Single-Path WNS Improvement** | **+0.0700 ns** |

---

## 💡 Running Inference on Custom Verilog Modules

To optimize a custom Verilog module for timing closure in Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_dir = "models/grpo_rlvr_checkpoint"

# Load tokenizer and 4-bit base model with LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)

# Define prompt with original Verilog and OpenSTA timing violation details
prompt = """<|im_start|>system
You are an expert IC digital design assistant specializing in RTL timing closure optimization for ASIC designs using SkyWater 130nm PDK. Your goal is to rewrite the input Verilog RTL module to fix timing violations (Worst Negative Slack WNS < 0) while maintaining functional correctness.
<|im_end|>
<|im_start|>user
Target Clock Frequency: 1.0 GHz (Period: 1.0000 ns)
Current Timing Violation:
- Worst Negative Slack (WNS): -2.6000 ns
- Critical Path Startpoint: q
- Critical Path Endpoint: out

Original Verilog Code:
```verilog
module Counter (
    input wire clk,
    input wire reset,
    output reg [7:0] q
);
    always @(posedge clk or posedge reset) begin
        if (reset) q <= 8'd0;
        else q <= q + 1'b1;
    end
endmodule
```
Provide the optimized Verilog code enclosed in ```verilog ... ```.
<|im_end|>
<|im_start|>assistant
"""

# Generate fix
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📜 License
Licensed under the Apache License 2.0. Standard Cell Library parameters subject to SkyWater 130nm Open Source PDK terms.
