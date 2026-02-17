# LLM Serving Infrastructure

## Overview

Serving LLMs at production scale is fundamentally different from serving traditional ML models. Autoregressive generation produces one token at a time, creating a sequential bottleneck. Models are enormous (7B–70B+ parameters), requiring multi-GPU setups. This document covers the key concepts: KV cache, batching strategies, speculative decoding, serving frameworks, GPU memory management, and latency vs throughput trade-offs.

---

## 1. LLM Serving Challenges

### Why LLMs Are Hard to Serve

```
┌─────────────────────────────────────────────────────────────────┐
│  Traditional ML vs LLM Serving                                   │
│                                                                  │
│  Traditional (e.g., BERT classification):                        │
│  - Single forward pass → prediction                              │
│  - Batch = parallelizable                                        │
│  - Latency ~10–50ms                                              │
│                                                                  │
│  LLM (autoregressive):                                           │
│  - Forward pass → 1 token → repeat N times                       │
│  - Decode phase = inherently sequential                          │
│  - Latency = prefill + N × decode_step                           │
│  - 100 output tokens @ 50 TPS = 2+ seconds                       │
└─────────────────────────────────────────────────────────────────┘
```

| Challenge | Impact |
|-----------|--------|
| **Large models** | 7B = ~14GB FP16, 70B = ~140GB → requires tensor/pipeline parallelism |
| **Autoregressive** | Each token needs full attention over all previous tokens |
| **Variable length** | Input/output lengths vary widely → inefficient static batching |
| **Memory bound** | KV cache grows with sequence length; memory bandwidth limits throughput |

---

## 2. KV Cache

### What Is the KV Cache?

During autoregressive generation, the model computes attention over all previous tokens. The Key and Value tensors for each layer are reused across steps. Caching them avoids recomputation.

```
┌─────────────────────────────────────────────────────────────────┐
│  Attention without KV cache (naive):                             │
│  Step 1: Attend over [t1]                                        │
│  Step 2: Attend over [t1, t2]  ← recompute t1's K,V               │
│  Step 3: Attend over [t1, t2, t3]  ← recompute t1,t2's K,V        │
│  ... O(n²) recomputation                                         │
│                                                                  │
│  With KV cache:                                                  │
│  Step 1: Compute K1,V1, store, output t1                         │
│  Step 2: Compute K2,V2, append to cache, attend, output t2       │
│  Step 3: Compute K3,V3, append, attend, output t3                │
│  ... O(n) per step                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Implications

For L layers, H heads, head_dim D, sequence length S:

```
KV cache size per token = 2 × L × H × D × 2 bytes (FP16)
Example: L=32, H=32, D=128 → 2 × 32 × 32 × 128 × 2 = 524KB per token
For 4096 tokens: ~2GB per request (Llama-2 7B)
```

| Model | Layers | Heads | Head Dim | KV/token | 4K seq |
|-------|--------|-------|----------|----------|--------|
| Llama-2 7B | 32 | 32 | 128 | ~524KB | ~2GB |
| Llama-2 70B | 80 | 64 | 128 | ~1.3MB | ~5.2GB |
| GPT-4 class | 120+ | 96+ | 128 | ~2MB+ | ~8GB+ |

### PagedAttention (vLLM)

Traditional KV cache allocates contiguous memory per request, causing fragmentation when requests finish at different times. **PagedAttention** allocates KV cache in fixed-size blocks (pages), similar to OS virtual memory.

```
┌─────────────────────────────────────────────────────────────────┐
│  Traditional KV Cache (wasteful)                                 │
│                                                                  │
│  Request A: [=======]  (allocated 4K, used 2K → 2K wasted)       │
│  Request B: [=================]  (allocated 4K, used 4K)         │
│  Request C: [===]  (allocated 4K, used 1K → 3K wasted)           │
│                                                                  │
│  PagedAttention (efficient)                                      │
│                                                                  │
│  Block pool: [P0][P1][P2][P3][P4][P5]...                        │
│  Request A: P0, P1  (2 blocks)                                   │
│  Request B: P2, P3, P4  (3 blocks)                               │
│  Request C: P5  (1 block)                                         │
│  → Near-zero fragmentation, 2–4x higher throughput                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Batching Strategies

### Static Batching (Naive)

- Fixed batch size; wait until batch is full
- All requests in batch must wait for slowest to complete
- **Problem:** One long request blocks entire batch

### Continuous / Dynamic Batching (vLLM, TGI)

- New requests join batch as soon as they arrive
- Finished requests leave; new ones join
- Batch composition changes every decode step

```
┌─────────────────────────────────────────────────────────────────┐
│  Continuous Batching (Simplified)                               │
│                                                                  │
│  Step 0: [A, B, C]  (all in prefill)                             │
│  Step 1: [A, B, C]  (all decoding)                               │
│  Step 2: [A, B]     (C finished, D joins)                        │
│  Step 3: [A, B, D]  (C out, D in)                                │
│  Step 4: [B, D]     (A finished, E joins)                        │
│  Step 5: [B, D, E]  (A out, E in)                                │
│                                                                  │
│  GPU utilization stays high; no blocking on slowest request      │
└─────────────────────────────────────────────────────────────────┘
```

| Strategy | Latency | Throughput | Use Case |
|----------|---------|------------|----------|
| No batching | Best | Worst | Single user |
| Static batch=8 | Worst | Good | Batch jobs |
| Continuous | Good | Best | Production API |

---

## 4. Speculative Decoding

### Idea

Use a small "draft" model to generate K tokens quickly; verify all K in one forward pass with the large "target" model. If draft is correct, we get K tokens for 1 large-model pass instead of K passes.

```
┌─────────────────────────────────────────────────────────────────┐
│  Speculative Decoding                                             │
│                                                                  │
│  Draft model (e.g., 125M): t1, t2, t3, t4, t5  (fast, maybe wrong)│
│  Target model: Verify all 5 in ONE forward pass                  │
│                                                                  │
│  Accept until first mismatch:                                     │
│  - If t1,t2,t3 correct, t4 wrong → accept t1,t2,t3, resample t4  │
│  - Typically 2–3x speedup with good draft model                  │
└─────────────────────────────────────────────────────────────────┘
```

| Factor | Impact |
|--------|--------|
| Draft model quality | Higher acceptance → more speedup |
| K (candidate count) | Larger K → more speedup if draft good; more waste if bad |
| Draft model cost | Should be much cheaper than target |

---

## 5. Serving Frameworks

### vLLM (UC Berkeley)

- **Strengths:** PagedAttention, continuous batching, easy to use, OpenAI-compatible API
- **Best for:** General production serving, high throughput
- **Example:** `pip install vllm && python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf`

### TGI (Text Generation Inference, Hugging Face)

- **Strengths:** HuggingFace integration, Flash Attention, dynamic batching
- **Best for:** HuggingFace model hub, streaming
- **Example:** `docker run -p 8080:80 ghcr.io/huggingface/text-generation-inference --model-id meta-llama/Llama-2-7b-chat-hf`

### TensorRT-LLM (NVIDIA)

- **Strengths:** Kernel fusion, quantization, maximum performance on NVIDIA GPUs
- **Best for:** Lowest latency when GPU perf is critical
- **Trade-off:** More complex compilation and deployment

### Triton (OpenAI)

- **Strengths:** Custom kernel development, used in production by OpenAI
- **Best for:** Research and custom optimizations
- **Trade-off:** Requires kernel-level expertise

| Framework | Setup | Throughput | Latency | HuggingFace |
|-----------|-------|------------|---------|-------------|
| vLLM | Easy | ⭐⭐⭐ | ⭐⭐ | Good |
| TGI | Easy | ⭐⭐ | ⭐⭐ | Best |
| TensorRT-LLM | Hard | ⭐⭐⭐ | ⭐⭐⭐ | Manual |
| Triton | Hard | Custom | Custom | Manual |

---

## 6. GPU Memory Management

### Tensor Parallelism (TP)

Split weight matrices across GPUs; each GPU holds 1/TP of each layer. All GPUs participate in every operation.

```
┌─────────────────────────────────────────────────────────────────┐
│  Tensor Parallelism (TP=4)                                       │
│                                                                  │
│  Linear layer: Y = XW                                            │
│  W split along columns: W = [W0|W1|W2|W3]                       │
│  GPU 0: X @ W0   GPU 1: X @ W1   GPU 2: X @ W2   GPU 3: X @ W3  │
│  → All-reduce to combine partial outputs                         │
│                                                                  │
│  Memory: Model / 4 per GPU                                        │
│  Communication: All-reduce every layer                           │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Parallelism (PP)

Split model layers across GPUs. Micro-batches flow through the pipeline. Early GPUs process new batches while later GPUs process older batches.

```
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Parallelism (PP=4)                                     │
│                                                                  │
│  GPU0: Layers 0-7    GPU1: Layers 8-15   GPU2: Layers 16-23      │
│  GPU3: Layers 24-31                                               │
│                                                                  │
│  Batch 1: [G0]→[G1]→[G2]→[G3]                                    │
│  Batch 2:     [G0]→[G1]→[G2]→[G3]                                │
│  Batch 3:         [G0]→[G1]→[G2]→[G3]                            │
│                                                                  │
│  Bubble: Idle time when pipeline fills/drains                    │
│  More micro-batches → smaller bubble                              │
└─────────────────────────────────────────────────────────────────┘
```

### Offloading (CPU/NVMe)

Move some weights to CPU or NVMe when not in use. Reduces GPU memory at the cost of transfer latency. Common for inference on limited GPU memory.

| Strategy | Memory | Communication | Use Case |
|----------|--------|---------------|----------|
| TP | Model/N | High (layer-wise) | Same-node multi-GPU |
| PP | Model/N | Medium (activation) | Cross-node |
| Offload | Reduced | CPU↔GPU transfer | Single GPU, large model |

---

## 7. Latency vs Throughput Trade-offs

### TTFT vs TPS

| Metric | Definition | What Affects It |
|--------|------------|-----------------|
| **TTFT** | Time to first token | Prefill time, queue wait, batch size |
| **TPS** | Tokens per second | Decode kernel speed, batch size |

### Trade-off: Larger Batches

- **Throughput ↑** (better GPU utilization)
- **TTFT ↑** (more requests wait for batch)
- **Memory ↑** (more KV cache)

### Typical Numbers (Llama-2 7B, A100)

| Config | TTFT (p50) | TPS | Throughput |
|--------|------------|-----|------------|
| Batch=1 | ~100ms | ~80 | Low |
| Batch=8 | ~400ms | ~200 total | 8x higher |
| Batch=32 | ~1.2s | ~400 total | 32x higher |

---

## 8. Python Code Examples

### vLLM Serving Example

```python
# vllm_serve_example.py
"""
vLLM serving example - production-style setup
Requires: pip install vllm fastapi uvicorn
"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
import asyncio
from contextlib import asynccontextmanager

# Global LLM instance (load once at startup)
llm_instance = None
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

@asynccontextmanager
async def lifespan(app):
    """Load model once when server starts."""
    global llm_instance
    llm_instance = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=2,  # 2 GPUs
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )
    yield
    llm_instance = None

async def generate(prompts: list[str]) -> list[str]:
    """Batch generate with vLLM."""
    outputs = llm_instance.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]

# Alternative: Use vLLM's built-in OpenAI-compatible server
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --tensor-parallel-size 2
# Then call via OpenAI client:
# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
# response = client.chat.completions.create(model="meta-llama/...", messages=[...])
```

### Load Balancing Setup

```python
# load_balancer_example.py
"""
Load balancer for multi-replica LLM serving
Use with: nginx, HAProxy, or custom round-robin
"""

import random
from typing import List
from dataclasses import dataclass

@dataclass
class LLMReplica:
    host: str
    port: int
    health: bool
    load: float  # 0-1, current utilization

class LLMLoadBalancer:
    """Simple load-aware load balancer for LLM replicas."""
    
    def __init__(self, replicas: List[LLMReplica]):
        self.replicas = replicas
    
    def get_replica(self, strategy: str = "least_loaded") -> LLMReplica:
        healthy = [r for r in self.replicas if r.health]
        if not healthy:
            raise RuntimeError("No healthy replicas")
        
        if strategy == "round_robin":
            return random.choice(healthy)
        elif strategy == "least_loaded":
            return min(healthy, key=lambda r: r.load)
        elif strategy == "random":
            return random.choice(healthy)
        else:
            return healthy[0]
    
    def url_for(self, replica: LLMReplica) -> str:
        return f"http://{replica.host}:{replica.port}"

# Production: Use health checks + metrics
# Example replicas
replicas = [
    LLMReplica("10.0.1.1", 8000, True, 0.3),
    LLMReplica("10.0.1.2", 8000, True, 0.7),
    LLMReplica("10.0.1.3", 8000, True, 0.5),
]
lb = LLMLoadBalancer(replicas)
selected = lb.get_replica("least_loaded")
print(f"Routing to {lb.url_for(selected)}")
```

---

## 9. Production Architecture for Multi-GPU Serving

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  PRODUCTION MULTI-GPU LLM SERVING                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  Load Balancer  │
                         │  (Nginx/LB)     │
                         └────────┬────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  vLLM Instance 1 │   │  vLLM Instance 2 │   │  vLLM Instance 3 │
│  TP=4 (4x A100)  │   │  TP=4 (4x A100)  │   │  TP=4 (4x A100)  │
│  Llama-70B       │   │  Llama-70B       │   │  Llama-70B       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                          ┌───────┴───────┐
                          │   Monitoring  │
                          │ Prometheus    │
                          └───────────────┘
```

---

## 10. Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| Batching | Static | Continuous | Static: simpler, blocks. Continuous: complex, better utilization |
| KV cache | Contiguous | PagedAttention | Paged: less fragmentation, 2–4x throughput |
| Parallelism | Tensor | Pipeline | TP: lower latency, needs same-node. PP: cross-node, pipeline bubble |
| Framework | vLLM | TensorRT-LLM | vLLM: easy. TensorRT: max perf, harder |
| Batch size | Small | Large | Small: low TTFT. Large: high throughput |

---

## 11. Interview Tips

1. **Start with constraints:** "What's the expected QPS? Budget? Latency target?"
2. **KV cache:** Explain why it exists and how PagedAttention reduces fragmentation.
3. **Batching:** Describe continuous batching and why it matters for throughput.
4. **Scaling:** "For 70B model, I'd use tensor parallelism across 4 A100s."
5. **Latency:** "TTFT matters for streaming UX; TPS matters for total time. Larger batches help TPS but hurt TTFT."
6. **Speculative decoding:** "Use a small draft model to propose tokens, verify with large model in one pass for 2–3x speedup."

---

## 12. Related Topics

- [02 - Retrieval-Augmented Generation](./02-retrieval-augmented-generation.md) – RAG for context injection
- [04 - Cost & Latency Optimization](./04-cost-latency-optimization.md) – Quantization, caching
- [05-model-serving/01-serving-patterns.md](../../phase-2-core-components/05-model-serving/01-serving-patterns.md) – General serving patterns
- [07-scalability-performance/01-horizontal-scaling.md](../../phase-3-operations-and-reliability/07-scalability-performance/01-horizontal-scaling.md) – Scaling strategies
