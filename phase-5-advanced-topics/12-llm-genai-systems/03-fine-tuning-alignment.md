# Fine-Tuning & Alignment

## Overview

Fine-tuning adapts pretrained LLMs to specific tasks, formats, and behaviors. This document covers when to fine-tune versus prompt engineering or RAG, supervised fine-tuning (SFT), parameter-efficient methods (LoRA, QLoRA), RLHF, DPO, instruction tuning, evaluation, data quality, and production trade-offs.

---

## 1. When to Fine-Tune vs Prompt Engineering vs RAG

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  DECISION: Fine-Tune vs Prompt vs RAG                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  Do you need EXTERNAL / PRIVATE / UPDATED knowledge?
       │
       ├─ YES ──▶ RAG (index docs, retrieve at query time)
       │
       └─ NO
            │
            Can you achieve the behavior with better prompts / few-shot?
            │
            ├─ YES ──▶ Prompt engineering (cheapest, fastest)
            │
            └─ NO
                 │
                 Do you need to CHANGE MODEL BEHAVIOR (format, style, task)?
                 │
                 ├─ YES ──▶ Fine-tune (SFT, LoRA, RLHF/DPO)
                 │          - Custom output format (JSON, code)
                 │          - Brand voice, safety
                 │          - Task-specific performance
                 │
                 └─ NO ──▶ Use base model + prompts
```

| Approach | Use When | Cost | Latency | Updatability |
|----------|----------|------|---------|--------------|
| **Prompt engineering** | Simple tasks, few examples | $ | Instant | High |
| **RAG** | External/private knowledge, citations | $$ | +retrieval | High (re-index) |
| **Fine-tuning** | Output format, style, task performance | $$$ | Training time | Medium (retrain) |
| **RLHF/DPO** | Preference alignment, safety | $$$$ | Training time | Low |

---

## 2. Supervised Fine-Tuning (SFT)

### What Is SFT?

Train on (input, output) pairs to teach the model a task or format. Standard causal language modeling loss (next-token prediction) on the output tokens; input can be masked (no loss) or included (optional).

### Data Preparation

```
┌─────────────────────────────────────────────────────────────────┐
│  SFT Data Format (Chat)                                           │
│                                                                  │
│  {"messages": [                                                   │
│    {"role": "system", "content": "You are a helpful assistant."},│
│    {"role": "user", "content": "What is 2+2?"},                  │
│    {"role": "assistant", "content": "2+2 equals 4."}              │
│  ]}                                                               │
│                                                                  │
│  Loss computed only on assistant tokens (input + system masked)  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Setup

- **Optimizer:** AdamW, lr ~1e-5 to 5e-5
- **Epochs:** 1–3 typical; more risks overfitting
- **Batch size:** 4–32 per device (gradient accumulation for larger effective batch)
- **Sequence length:** 2048–8192; truncate or pack
- **Warmup:** 3–10% of steps

### Common Pitfalls

- **Overfitting:** Too many epochs, small dataset → use validation, early stopping
- **Catastrophic forgetting:** Over-train on new task → lose general abilities → use lower lr, more diverse data
- **Format collapse:** Model learns to output boilerplate → diversify examples, check format diversity

---

## 3. Parameter-Efficient Fine-Tuning (PEFT)

### LoRA (Low-Rank Adaptation)

Instead of updating all weights, add low-rank matrices that adapt specific layers (usually attention `q_proj`, `v_proj`):

```
┌─────────────────────────────────────────────────────────────────┐
│  LoRA: W' = W + ΔW  where  ΔW = A × B                            │
│                                                                  │
│  W: d × k (frozen)                                               │
│  A: d × r,  B: r × k   (r << d, k; typically r=8, 16, 32)       │
│                                                                  │
│  Trainable params: 2 × d × r (per layer)                        │
│  For 7B model: ~0.1% trainable (few MB vs several GB)             │
└─────────────────────────────────────────────────────────────────┘
```

| LoRA rank (r) | Trainable % | Quality | Speed |
|---------------|-------------|---------|-------|
| 8 | ~0.05% | Good for simple tasks | Fastest |
| 16 | ~0.1% | Balanced | Fast |
| 32–64 | ~0.2–0.4% | Best quality | Slower |

### QLoRA (Quantized LoRA)

- Base model in 4-bit (e.g., NF4); LoRA adapters in full precision
- Enables 70B fine-tuning on a single 48GB GPU
- Slight quality drop vs full LoRA; large gains in accessibility

### Adapters (Houlsby, etc.)

- Small modules inserted between layers
- Similar idea to LoRA: few parameters, modular

| Method | Trainable % | Memory | Quality |
|--------|-------------|--------|---------|
| Full fine-tune | 100% | Highest | Best |
| LoRA | ~0.1% | Low | Near full |
| QLoRA | ~0.1% + 4-bit base | Lowest | Good |
| Adapters | ~1–4% | Medium | Good |

---

## 4. RLHF (Reinforcement Learning from Human Feedback)

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  RLHF Pipeline (3 stages)                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  1. SFT model on (prompt, response) pairs
           │
           ▼
  2. Reward model: train on (prompt, response_A, response_B, preference)
     - Predict which response human preferred
     - Loss: Bradley-Terry or similar
           │
           ▼
  3. PPO: Optimize policy (LLM) to maximize reward
     - Sample responses from current policy
     - Score with reward model
     - PPO update to increase reward while staying close to ref policy (KL penalty)
```

### Reward Model

- Input: (prompt, response)
- Output: scalar reward
- Training: binary preference labels → cross-entropy or ranking loss
- Avoid reward hacking: keep KL penalty to reference model

### PPO Optimization

- Policy = current LLM; reference = SFT model
- Reward = RM score − β × KL(ref || policy)
- β balances reward vs deviation from reference

---

## 5. DPO (Direct Preference Optimization)

### Idea

DPO removes the explicit reward model and PPO loop. It optimizes the policy directly on preference data using a closed-form objective derived from the RLHF setup.

### Comparison to RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Stages | 3 (SFT, RM, PPO) | 2 (SFT, DPO) |
| Reward model | Explicit, separate | Implicit in loss |
| Stability | PPO can be unstable | Often more stable |
| Compute | Higher (PPO rollouts) | Lower |
| Quality | Slightly better in some setups | Comparable, simpler |

### When to Use

- **DPO:** Simpler pipeline, preference data available, want to avoid PPO tuning
- **RLHF:** Need nuanced reward (e.g., multiple criteria), already have RM

---

## 6. Instruction Tuning & Chat Formatting

### Instruction Tuning

- Train on diverse (instruction, response) pairs
- Makes model follow instructions (e.g., Alpaca, ShareGPT, OpenAssistant)
- Often done before task-specific fine-tuning

### Chat Formatting

Models expect specific formats; mismatches hurt performance:

| Model | Format |
|-------|--------|
| **ChatML** (OpenAI-style) | `<|im_start|>role\ncontent<|im_end|>` |
| **Llama** | `[INST] ... [/INST]` |
| **Mistral** | `[INST] ... [/INST]` |
| **ChatML** | `{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>{% endfor %}` |

Use the format matching your base model; keep it consistent at inference.

---

## 7. Evaluation

### Perplexity

- Log probability of held-out data
- Lower = better next-token prediction
- Does not always match human preference

### Human Evaluation

- A/B test: model A vs B on same prompts
- Dimensions: helpfulness, harmlessness, faithfulness, etc.
- Gold standard but expensive

### LLM-as-Judge

- Use an LLM to score or compare responses
- Cheaper than humans; reasonable correlation when calibrated
- Use stronger models (e.g., GPT-4) as judge

### Benchmark Suites

| Benchmark | Focus |
|-----------|-------|
| **MMLU** | Broad knowledge (57 subjects) |
| **HumanEval** | Code generation |
| **TruthfulQA** | Factual accuracy, avoiding misconceptions |
| **MT-Bench** | Multi-turn chat |
| **BIG-Bench** | Diverse reasoning |

---

## 8. Python Code: LoRA Fine-Tuning with PEFT

```python
# lora_finetuning_example.py
"""
LoRA fine-tuning with HuggingFace PEFT and TRL
Requires: pip install peft transformers datasets torch trl
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Config ---
MODEL_ID = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "alpaca"  # or your custom dataset
OUTPUT_DIR = "./llama2-7b-lora"
LORA_R = 16
LORA_ALPHA = 32  # alpha = 2 * r common for scaling
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # or ["q_proj","k_proj","v_proj","o_proj"]

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Model: optional 4-bit for QLoRA ---
use_4bit = True  # Set False for full LoRA
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

# --- LoRA ---
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Expect: trainable params: ~4M / 7B = 0.06%

# --- Dataset: Alpaca format ---
# Format: instruction, input, output
def format_alpaca(example):
    if example.get("input", "").strip():
        text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": text}

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# --- Training ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- Inference ---
model.eval()
prompt = "### Instruction:\nWhat is 2+2?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 9. Data Quality and Curation for Fine-Tuning

### Quality Over Quantity

- 1K–10K high-quality examples often beat 100K noisy ones
- Avoid duplicates, near-duplicates, and format inconsistencies
- Diverse prompts and response styles

### Data Sources

| Source | Pros | Cons |
|--------|------|------|
| Human writers | Best quality | Expensive |
| LLM-generated (with human review) | Scalable | Can inherit base model biases |
| Logs (user feedback) | Real distribution | Noisy, may need filtering |
| Public instruction sets | Free, diverse | Variable quality |

### Curated Datasets

- **Alpaca:** 52K instruction-following (from self-instruct)
- **ShareGPT:** Human chat
- **OpenAssistant:** Multilingual conversations
- **Dolly:** Enterprise-style instructions

### Data Mix for Generalization

- Mix task-specific + general instructions to reduce catastrophic forgetting
- Typical: 80% task-specific, 20% general

---

## 10. Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| Fine-tune vs RAG | Fine-tune | RAG | Fine-tune: behavior change. RAG: knowledge injection, easier updates |
| Full vs LoRA | Full | LoRA | Full: best quality, expensive. LoRA: 0.1% params, near-full quality |
| LoRA vs QLoRA | LoRA | QLoRA | QLoRA: 4-bit base, 70B on 1 GPU; small quality drop |
| RLHF vs DPO | RLHF | DPO | DPO: simpler, stable. RLHF: more control, RM reusable |
| Epochs | 1 | 3 | More epochs: overfit risk. Use validation, early stop |
| Data size | 1K | 10K | More data: better if quality maintained; diminishing returns |

---

## 11. Interview Tips

1. **When to fine-tune?** "When I need to change output format, style, or task behavior that prompts can't achieve."
2. **LoRA:** "Low-rank adapters on attention layers; 0.1% trainable params, near full fine-tune quality."
3. **QLoRA:** "4-bit base + LoRA; run 70B on single 48GB GPU."
4. **RLHF:** "SFT → reward model on preferences → PPO to maximize reward with KL penalty."
5. **DPO:** "Direct optimization on preferences; no explicit reward model or PPO; often simpler and as good."
6. **Data quality:** "Prioritize quality over quantity; 1K–10K curated examples can be enough."
7. **Evaluation:** "Use perplexity, human eval, and LLM-as-judge; benchmark suites like MMLU, HumanEval."

---

## 12. Related Topics

- [02 - Retrieval-Augmented Generation](./02-retrieval-augmented-generation.md) – When to use RAG instead
- [04 - Cost & Latency Optimization](./04-cost-latency-optimization.md) – Quantization for serving
- [01 - LLM Serving Infrastructure](./01-llm-serving-infrastructure.md) – Deploying fine-tuned models
- [04-model-training/01-training-infrastructure.md](../../phase-2-core-components/04-model-training/01-training-infrastructure.md) – Training infrastructure
