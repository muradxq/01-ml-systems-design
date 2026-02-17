# Cost & Latency Optimization

## Overview

Production LLM systems must balance cost, latency, and quality. Token-based pricing, large context windows, and autoregressive generation create unique optimization challenges. This document covers token economics, model selection, quantization, distillation, prompt optimization, model routing, caching strategies, cost modeling, and Python implementations.

---

## 1. Token Economics

### Cost per Token (Approximate, 2024)

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Note |
|-------|--------------------|----------------------|-----|
| GPT-4o | $2.50 | $10 | Fast, capable |
| GPT-4o-mini | $0.15 | $0.60 | 10–20x cheaper |
| Claude 3.5 Sonnet | $3 | $15 | Strong reasoning |
| Claude 3.5 Haiku | $0.25 | $1.25 | Cheap |
| Llama-3 70B (self-hosted) | GPU cost | GPU cost | Amortize over utilization |
| Mistral 7B (self-hosted) | GPU cost | GPU cost | Lower GPU needs |

### Cost at Scale

```
┌─────────────────────────────────────────────────────────────────┐
│  Example: 1M requests/day                                        │
│                                                                  │
│  Assume: 500 input + 200 output tokens/request = 700 tokens     │
│  GPT-4o: 1M × (500×2.5 + 200×10)/1M = $3.25M/day (!!)          │
│  GPT-4o-mini: 1M × (500×0.15 + 200×0.6)/1M = $195K/day         │
│  Claude Haiku: ~$875/day                                        │
│                                                                  │
│  Self-hosted Llama-70B (4×A100):                                │
│  - GPU: ~$3–4/hour × 24 = ~$96/day (if fully utilized)          │
│  - At 50% utilization: ~$192/day for unlimited tokens          │
│  - Break-even vs API depends on volume; high volume → self-host │
└─────────────────────────────────────────────────────────────────┘
```

### Input vs Output Cost

- Output is typically 2–10x more expensive than input
- Long outputs (summaries, code) dominate cost
- Optimize: shorter prompts, structured outputs (fewer tokens), caching

---

## 2. Model Selection: When to Use Large vs Small

| Factor | Small (7B) | Medium (13–34B) | Large (70B+) |
|--------|------------|-----------------|--------------|
| **Latency** | Low | Medium | High |
| **Cost** | Low | Medium | High |
| **Quality** | Good for simple tasks | Better reasoning | Best quality |
| **GPU** | 1×24GB | 1–2×40GB | 4×80GB+ |
| **Use case** | Classification, routing, simple QA | General chat, RAG | Complex reasoning, code |

### Decision Tree

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Selection                                                 │
│                                                                  │
│  Need complex reasoning / code / math?  → Large (70B+)           │
│  General chat, RAG, most use cases?     → Medium (13–34B)        │
│  Routing, simple classification, cheap? → Small (7B)             │
│  Ultra-low latency / edge?              → Tiny (1–3B) or distill │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Quantization

### Methods

| Method | Bits | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| **FP16** | 16 | Baseline | Baseline | Training, best quality |
| **INT8** | 8 | ~99% | ~1.5x | Production, good balance |
| **INT4 (GPTQ)** | 4 | ~95–98% | ~2x | Memory constrained |
| **INT4 (AWQ)** | 4 | ~97–99% | ~2x | Better quality than GPTQ |
| **GGUF (llama.cpp)** | 2–8 | Varies | Good | Local, flexible |
| **NF4 (QLoRA)** | 4 | ~95% | 2x | Training, inference |

### GPTQ vs AWQ vs GGUF

- **GPTQ:** Post-training quantization; good compression; some quality loss on difficult tasks
- **AWQ:** Activations-aware; preserves important weights; often better quality at same bit width
- **GGUF:** File format for llama.cpp; supports 2–8 bit; flexible for local deployment

### Quality vs Speed Trade-offs

```
┌─────────────────────────────────────────────────────────────────┐
│  Quantization Trade-offs                                         │
│                                                                  │
│  INT8:  Minimal quality loss, 2x less memory, ~1.2–1.5x faster  │
│  INT4:  2–5% quality drop possible, 4x less memory, ~2x faster   │
│  INT2:  Significant quality drop; rarely used for production     │
│                                                                  │
│  Rule of thumb: Use INT8 first; try INT4 if memory-bound        │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Savings

| Precision | Size (7B model) | Size (70B model) |
|-----------|-----------------|------------------|
| FP16 | ~14 GB | ~140 GB |
| INT8 | ~7 GB | ~70 GB |
| INT4 | ~3.5 GB | ~35 GB |

---

## 4. Model Distillation

### Idea

Train a smaller "student" model to imitate a larger "teacher" model's outputs. Student learns from soft labels (logits) or hard labels (generated text).

```
┌─────────────────────────────────────────────────────────────────┐
│  Distillation Pipeline                                           │
│                                                                  │
│  Teacher (70B) ──▶ Generate outputs on unlabeled data           │
│         │                                                         │
│         ▼                                                         │
│  Student (7B) ──▶ Train to match teacher outputs                 │
│         │         (or distill logits)                             │
│         ▼                                                         │
│  Smaller, faster model with ~70–80% of teacher quality            │
└─────────────────────────────────────────────────────────────────┘
```

| Aspect | Teacher | Student |
|--------|---------|---------|
| Size | 70B | 7B |
| Latency | High | 5–10x lower |
| Quality | 100% | 70–85% |
| Cost | High | Low |

---

## 5. Prompt Optimization

### Shorter Prompts

- Remove redundant instructions
- Use concise system prompts
- Example: 500 → 200 tokens = 60% input cost reduction

### Prompt Caching (Anthropic, OpenAI)

- Cache common prefix (e.g., system prompt, RAG context) across requests
- First request pays full cost; subsequent requests pay only for unique suffix
- APIs: `cache_control` or similar; check provider docs

### Structured Outputs

- Request JSON, specific format → reduces retries and wasted tokens
- Use `response_format` / JSON mode when available
- Fewer tokens from constrained formats

### Few-Shot Efficiency

- 1–3 examples often enough; more may not help
- Short examples preferred
- Dynamic few-shot: retrieve similar examples (expensive but effective)

---

## 6. Model Routing (Cascading)

### Idea

Use a cheap/fast model first; escalate to a larger model only for harder queries.

```
┌─────────────────────────────────────────────────────────────────┐
│  Cascading / Router Architecture                                 │
│                                                                  │
│  Query ──▶ Router (cheap 7B or classifier)                       │
│              │                                                     │
│              ├─ Easy (80%) ──▶ Small model (7B) ──▶ Response      │
│              │                                                     │
│              └─ Hard (20%) ──▶ Large model (70B) ──▶ Response    │
│                                                                  │
│  Cost: 0.8 × cheap + 0.2 × expensive << 1.0 × expensive           │
│  Example: 0.8×$0.01 + 0.2×$0.10 = $0.028 vs $0.10 (72% savings)  │
└─────────────────────────────────────────────────────────────────┘
```

### Router Design

- **Classifier:** Train binary/multi-class "difficulty" on historical data
- **Self-evaluate:** Small model scores its own confidence; escalate if low
- **Heuristics:** Query length, keyword presence, domain
- **LLM-based:** "Is this query complex? Yes/No" with small model

---

## 7. Caching

### Exact Match Caching

- Cache (prompt_hash, response) for identical prompts
- Hit rate: 5–30% for chat; higher for repetitive APIs
- Storage: Redis, in-memory; TTL based on data freshness

### Semantic Caching

- Embed query; if similar query exists in cache (cosine > threshold), return cached response
- Handles paraphrasing; higher hit rate than exact match
- Risk: Slightly different query may get stale/wrong answer → similarity threshold tuning

### Prompt / Prefix Caching

- Cache KV cache for common prefix (e.g., system prompt, RAG context)
- Saves prefill compute for repeated prefixes
- Supported by vLLM, some APIs

```
┌─────────────────────────────────────────────────────────────────┐
│  Cache Hit Scenarios                                             │
│                                                                  │
│  Exact: "What is our refund policy?" → cache hit                 │
│  Semantic: "How do I get my money back?" → similar to above     │
│  Prefix: Same 2K system+RAG context, different user question     │
│         → skip prefill for shared prefix                         │
└─────────────────────────────────────────────────────────────────┘
```

| Cache Type | Hit Rate | Latency Saving | Complexity |
|------------|----------|----------------|------------|
| Exact | Low–Medium | Full request | Low |
| Semantic | Medium–High | Full request | Medium |
| Prefix | Depends | Prefill only | Medium |

---

## 8. Python Code Examples

### Quantization Example (with `bitsandbytes`)

```python
# quantization_example.py
"""
Quantization for inference - load model in 8-bit or 4-bit
Requires: pip install transformers accelerate bitsandbytes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_quantized_model(model_id: str, use_4bit: bool = True):
    """Load model with 4-bit or 8-bit quantization."""
    if use_4bit:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

# Memory: 7B FP16 ≈ 14GB → 4-bit ≈ 3.5GB
model, tokenizer = load_quantized_model("meta-llama/Llama-2-7b-chat-hf", use_4bit=True)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Router Implementation

```python
# router_example.py
"""
Model router: route to cheap or expensive model based on query difficulty
"""

from typing import Literal
from dataclasses import dataclass
import hashlib
import json

@dataclass
class RouterConfig:
    """Configuration for model router."""
    complexity_threshold: float = 0.7  # Above this = hard
    cache_ttl_seconds: int = 3600
    use_semantic_cache: bool = True

class LLMRouter:
    """Route queries to cheap (7B) or expensive (70B) model."""
    
    def __init__(
        self,
        cheap_client,   # e.g., OpenAI client for gpt-4o-mini
        expensive_client,  # e.g., OpenAI client for gpt-4o
        cache=None,  # Redis or dict
        config: RouterConfig = None,
    ):
        self.cheap = cheap_client
        self.expensive = expensive_client
        self.cache = cache or {}
        self.config = config or RouterConfig()
    
    def _query_complexity(self, query: str) -> float:
        """Estimate complexity 0-1. In production: use classifier or heuristics."""
        score = 0.0
        if len(query) > 200:
            score += 0.2
        if "?" in query:
            score += 0.1
        if query.count(" and ") + query.count(" or ") > 1:
            score += 0.2
        if any(w in query.lower() for w in ["compare", "explain", "why", "how"]):
            score += 0.3
        return min(score + 0.2, 1.0)
    
    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()
    
    def _get_cached(self, key: str):
        if not self.cache:
            return None
        if isinstance(self.cache, dict):
            return self.cache.get(key)
        return self.cache.get(key)
    
    def _set_cached(self, key: str, value: str):
        if not self.cache:
            return
        if isinstance(self.cache, dict):
            self.cache[key] = value
        else:
            self.cache.setex(key, self.config.cache_ttl_seconds, value)
    
    def route(self, query: str) -> Literal["cheap", "expensive"]:
        """Decide which model to use."""
        complexity = self._query_complexity(query)
        if complexity >= self.config.complexity_threshold:
            return "expensive"
        return "cheap"
    
    def generate(self, query: str, model_override: str = None) -> str:
        """Generate response, using cache and router."""
        key = self._cache_key(query)
        cached = self._get_cached(key)
        if cached:
            return cached
        
        model_type = model_override or self.route(query)
        client = self.cheap if model_type == "cheap" else self.expensive
        model_name = "gpt-4o-mini" if model_type == "cheap" else "gpt-4o"
        
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": query}],
            max_tokens=512,
        )
        text = resp.choices[0].message.content
        self._set_cached(key, text)
        return text
```

### Semantic Cache Example

```python
# semantic_cache_example.py
"""
Semantic cache: cache by embedding similarity
"""

import numpy as np
from typing import Optional, List

class SemanticCache:
    def __init__(
        self,
        embedder,
        threshold: float = 0.92,
        max_size: int = 10000,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self.queries: List[str] = []
        self.embeddings: np.ndarray = None
        self.responses: List[str] = []
    
    def _embed(self, text: str) -> np.ndarray:
        emb = self.embedder.embed_single(text)
        return np.array(emb, dtype=np.float32).reshape(1, -1)
    
    def get(self, query: str) -> Optional[str]:
        """Return cached response if similar query exists."""
        if len(self.queries) == 0:
            return None
        q_emb = self._embed(query)
        sims = np.dot(self.embeddings, q_emb.T).flatten()
        best_idx = np.argmax(sims)
        if sims[best_idx] >= self.threshold:
            return self.responses[best_idx]
        return None
    
    def set(self, query: str, response: str):
        """Add to cache."""
        if len(self.queries) >= self.max_size:
            self.queries.pop(0)
            self.responses.pop(0)
            self.embeddings = np.vstack([self.embeddings[1:], self._embed(query)])
        else:
            new_emb = self._embed(query)
            if self.embeddings is None:
                self.embeddings = new_emb
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb])
        self.queries.append(query)
        self.responses.append(response)
```

---

## 9. Cost Modeling for Production

### Cost Formula

```
Cost per request = (input_tokens × input_price + output_tokens × output_price) / 1e6
Monthly cost = requests_per_month × cost_per_request
```

### Cost Optimization Levers

| Lever | Impact |
|-------|--------|
| **Smaller model** | 5–20x cost reduction |
| **Quantization** | 2–4x memory reduction, enables smaller GPUs |
| **Shorter prompts** | Linear in input length |
| **Caching** | (1 - hit_rate) × cost |
| **Router** | 50–80% savings if most queries are easy |
| **Batch size** | Better GPU utilization → lower $/token (self-hosted) |

### Break-Even: API vs Self-Hosted

- Compare: `monthly_api_cost` vs `monthly_gpu_cost` + engineering
- At ~10M+ tokens/day, self-hosted often wins for 70B-class models
- Use spot/preemptible GPUs for batch workloads

---

## 10. Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| Model size | 7B | 70B | 7B: cheap, fast. 70B: best quality, expensive |
| Quantization | FP16 | INT4 | INT4: 4x less memory, 2–5% quality drop |
| Caching | None | Semantic | Semantic: higher hit rate, risk of wrong cache |
| Routing | Single model | Cascade | Cascade: 50–80% savings, routing logic needed |
| Prompt | Long | Short | Short: cheaper, may lose context |
| API vs self-host | API | Self-host | Self-host: break-even at high volume |

---

## 11. Interview Tips

1. **Token economics:** "Output is 2–10x input cost. At 1M req/day, GPT-4 can be $3M+/day; GPT-4o-mini or self-hosted much cheaper."
2. **Quantization:** "INT8 for minimal quality loss; INT4 when memory-bound. AWQ often better quality than GPTQ at 4-bit."
3. **Caching:** "Exact for identical prompts; semantic for paraphrases. Prefix caching saves prefill for shared context."
4. **Routing:** "Use cheap model first; escalate 20% hard queries to expensive. Saves 50–80% with good router."
5. **Cost modeling:** "Formula: input_tok×in_price + output_tok×out_price. Levers: smaller model, quantization, caching, routing."
6. **Break-even:** "At 10M+ tokens/day, self-hosted 70B often cheaper than API; factor in engineering and ops."

---

## 12. Related Topics

- [01 - LLM Serving Infrastructure](./01-llm-serving-infrastructure.md) – Batching, quantization in serving
- [02 - Retrieval-Augmented Generation](./02-retrieval-augmented-generation.md) – RAG and context length
- [03 - Fine-Tuning & Alignment](./03-fine-tuning-alignment.md) – QLoRA, distillation
- [07-scalability-performance/02-caching-strategies.md](../../phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md) – General caching
