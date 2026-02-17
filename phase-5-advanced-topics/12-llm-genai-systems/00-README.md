# 🤖 LLM & GenAI Systems

## Overview

Large Language Models (LLMs) and Generative AI systems have become central to ML systems design interviews in 2025–2026. Unlike traditional ML systems (classification, recommendation), LLM systems face unique challenges: **autoregressive generation**, **massive model sizes**, **token-level latency**, **hallucination control**, and **cost at scale**. This section covers production-ready architectures, serving infrastructure, RAG, fine-tuning, and cost optimization—the core topics increasingly asked in system design interviews.

---

## 🏛️ High-Level LLM System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION LLM SYSTEM ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   Clients    │
                                    │ (API, Chat)  │
                                    └──────┬───────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY / LOAD BALANCER                                  │
│  - Rate limiting  - Auth  - Request routing  - Prompt/result caching  - Cost tracking     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│   ROUTER / CASCADE      │  │   RAG PREPROCESSOR      │  │   SEMANTIC CACHE        │
│ - Model selection       │  │ - Query understanding   │  │ - Similar query hit      │
│ - Hard/easy routing     │  │ - Retrieval triggers     │  │ - Exact match bypass     │
│ - Fallback chains       │  │ - Context assembly       │  │ - TTL management         │
└────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
             │                           │                            │
             │         Cache miss        │                            │ Cache hit
             │                           ▼                            │
             │  ┌────────────────────────────────────────────────────┴────────────┐
             │  │                     RETRIEVAL LAYER (RAG)                        │
             │  │  - Vector DB (Pinecone, Milvus, pgvector)                        │
             │  │  - Hybrid: BM25 + dense retrieval                                │
             │  │  - Reranker (cross-encoder)                                      │
             │  └─────────────────────────────────────────────────────────────────┘
             │                                      │
             │                                      ▼
             │  ┌─────────────────────────────────────────────────────────────────┐
             │  │                    PROMPT ASSEMBLY                               │
             │  │  System prompt + Retrieved context + User query + Chat history   │
             │  └─────────────────────────────────────────────────────────────────┘
             │                                      │
             ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           LLM SERVING INFRASTRUCTURE                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                         │
│  │   vLLM      │ │   TGI       │ │ TensorRT-LLM│ │ Triton      │                         │
│  │ (Inference) │ │ (Inference) │ │ (Inference) │ │ (Kernel)    │                         │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘                         │
│                                                                                           │
│  - Continuous batching  - KV cache (PagedAttention)  - Speculative decoding              │
│  - Multi-GPU tensor/pipeline parallelism  - Quantization (INT4/INT8)                     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           POST-PROCESSING & SAFETY                                         │
│  - Output parsing (JSON, structured)  - Safety filters  - Guardrails  - Citation check   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           OBSERVABILITY & MONITORING                                       │
│  - TTFT, TPS, latency percentiles  - Token usage  - Cost per request  - Hallucination     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Table of Contents

| # | Topic | Key Concepts |
|---|-------|--------------|
| 1 | [LLM Serving Infrastructure](./01-llm-serving-infrastructure.md) | KV cache, batching, speculative decoding, vLLM/TGI, multi-GPU |
| 2 | [Retrieval-Augmented Generation (RAG)](./02-retrieval-augmented-generation.md) | Chunking, embeddings, hybrid retrieval, reranking, hallucination mitigation |
| 3 | [Fine-Tuning & Alignment](./03-fine-tuning-alignment.md) | SFT, LoRA, RLHF, DPO, when to fine-tune vs RAG |
| 4 | [Cost & Latency Optimization](./04-cost-latency-optimization.md) | Quantization, caching, model routing, token economics |

---

## ⚠️ Key Challenges Unique to LLM Systems

### 1. Cost at Scale

| Factor | Impact |
|--------|--------|
| **Token economics** | GPT-4: ~$0.03/1K input, ~$0.06/1K output (as of 2024). At 1M requests/day with 500 tokens avg → **$15K–45K/day** |
| **GPU utilization** | LLMs underutilize GPUs during decoding (sequential). Need high batching to amortize cost |
| **Long contexts** | 128K context = 128K KV cache entries per request. Memory cost dominates |

**Interview tip:** Always discuss cost as a first-class constraint. Ask: "What's the budget per 1M tokens?" and design accordingly.

---

### 2. Latency (TTFT vs TPS)

```
┌─────────────────────────────────────────────────────────────────┐
│  Latency Breakdown (Typical LLM Request)                         │
│                                                                  │
│  TTFT (Time To First Token): ~200–800ms                           │
│  ├── Prefill: Process all input tokens (parallel)                │
│  ├── KV cache population                                         │
│  └── First token decode                                          │
│                                                                  │
│  TPS (Tokens Per Second): ~20–80 tokens/sec                       │
│  ├── Decode: One token at a time (sequential)                    │
│  └── Autoregressive bottleneck                                   │
│                                                                  │
│  Total latency = TTFT + (output_length / TPS)                     │
│  Example: TTFT=400ms, 100 tokens @ 50 TPS → 400 + 2000 = 2.4s    │
└─────────────────────────────────────────────────────────────────┘
```

- **Chat UX** depends on low TTFT (streaming feels responsive)
- **Throughput** depends on TPS and batch size
- Trade-off: Batching improves throughput but can hurt TTFT

---

### 3. Hallucination & Safety

| Issue | Mitigation |
|-------|------------|
| **Factual hallucination** | RAG (ground in documents), citation, verification |
| **Adversarial prompts** | Input filtering, prompt injection detection |
| **Harmful output** | Output filters, RLHF/DPO alignment |
| **Stale knowledge** | RAG over updated docs, fine-tune on recent data |

---

### 4. Autoregressive Bottleneck

- Each output token depends on all previous tokens
- Cannot parallelize decode phase across tokens
- Solutions: speculative decoding (draft + verify), better batching, smaller/faster models

---

## 🎯 Interview Framework for LLM System Design

1. **Clarify scale:** QPS, tokens per request, budget
2. **Clarify latency:** TTFT target? Streaming or batch?
3. **Clarify accuracy:** Need citations? Domain-specific? Fresh data?
4. **Choose stack:** vLLM/TGI, RAG vs fine-tune, quantization level
5. **Discuss trade-offs:** Cost vs latency, quality vs speed

---

## 📐 Component Deep Dive

### API Gateway Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│  API Gateway Functions                                            │
│                                                                  │
│  Traffic Management:                                             │
│  - Rate limiting (per user, per API key, global)                 │
│  - Request queuing during overload                               │
│  - Load balancing across LLM replicas                            │
│                                                                  │
│  Observability:                                                   │
│  - Token usage tracking (input/output per request)               │
│  - Latency percentiles (p50, p95, p99)                           │
│  - Cost attribution per customer/team                            │
│                                                                  │
│  Optimization:                                                    │
│  - Exact match cache lookup before LLM call                       │
│  - Prompt caching (pass cache IDs to provider)                  │
│  - Request deduplication                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Router / Cascade Logic

| Query Type | % of Traffic | Model | Cost/1K tokens | Example |
|------------|--------------|-------|----------------|---------|
| Simple (routing, classification) | 60% | 7B/API-cheap | $0.001 | "Is this support or sales?" |
| General (chat, QA) | 30% | 34B/API-mid | $0.01 | "Explain how X works" |
| Complex (reasoning, code) | 10% | 70B/API-premium | $0.05 | "Refactor this and explain" |

Weighted cost: 0.6×0.001 + 0.3×0.01 + 0.1×0.05 = $0.0086 vs $0.05 single-model → **83% savings**

### RAG Integration Points

- **When to retrieve:** Every query, or only when query suggests external knowledge?
- **Context budget:** Typical 4–8 chunks × 512 tokens = 2–4K tokens
- **Citation:** Require [N] references; verify against retrieved chunks

---

## 🔢 Concrete Numbers Reference

### Model Sizes & GPU Requirements (FP16)

| Model | Parameters | VRAM (FP16) | VRAM (INT4) | Typical GPU |
|-------|------------|-------------|-------------|-------------|
| Llama-3 8B | 8B | ~16 GB | ~4 GB | 1× A100 40G |
| Llama-3 70B | 70B | ~140 GB | ~35 GB | 4× A100 80G |
| Mistral 7B | 7B | ~14 GB | ~3.5 GB | 1× A100 24G |
| GPT-4 class | ~1T (est) | N/A | N/A | API only |

### Latency Targets by Use Case

| Use Case | TTFT Target | TPS Target | Total (100 tok) |
|----------|-------------|------------|-----------------|
| Chat (streaming) | < 200ms | 40+ | ~2.5s |
| Code completion | < 100ms | 80+ | ~1.2s |
| Batch summarization | < 2s | 20+ | ~7s |
| RAG QA | < 500ms | 30+ | ~3.8s |

### Cost per 1M Tokens (API, 2024 Approx)

| Tier | Input | Output | Use Case |
|------|-------|--------|----------|
| Premium (GPT-4o, Claude Opus) | $2.50–5 | $10–15 | Complex tasks |
| Mid (GPT-4o-mini, Claude Sonnet) | $0.15–0.5 | $0.6–2 | General |
| Budget (Claude Haiku, local) | $0.05–0.25 | $0.25–1 | High volume, simple |

---

## 🧩 Technology Stack for LLM Systems

### Serving Layer

| Component | Options | When to Use |
|-----------|---------|-------------|
| **Inference** | vLLM, TGI, TensorRT-LLM | Self-hosted models |
| **API** | LangServe, FastAPI, OpenAI-compatible | Custom endpoints |
| **Orchestration** | Kubernetes, Ray Serve | Multi-replica deployment |

### Retrieval Layer (RAG)

| Component | Options | When to Use |
|-----------|---------|-------------|
| **Vector DB** | Pinecone, Milvus, pgvector, Weaviate | Dense retrieval |
| **Search** | Elasticsearch, OpenSearch | BM25, hybrid |
| **Embeddings** | sentence-transformers, OpenAI, Cohere | Query/doc encoding |
| **Reranker** | BAAI/bge-reranker, Cohere | Precision boost |

### Observability

| Component | Options | Metrics |
|-----------|---------|---------|
| **APM** | Datadog, New Relic, Prometheus | Latency, errors |
| **LLM-specific** | LangSmith, Helicone, Arize | Token usage, cost, feedback |
| **Logging** | ELK, Loki | Request/response audit |

---

## 🎤 Common Interview Scenarios

### Scenario 1: "Design a RAG system for internal docs"

- Clarify: Doc volume, update frequency, QPS, citation requirements
- Components: Chunking (512 tok, 50 overlap), embedding (bge-large), vector DB (pgvector if small, Pinecone if large), reranker, LLM with citation instructions
- Evaluation: Recall@K, faithfulness, citation accuracy

### Scenario 2: "Reduce cost of our LLM API by 70%"

- Levers: Model downgrade (GPT-4 → mini), caching (exact + semantic), router (easy queries to cheap model), shorter prompts, structured outputs
- Quantify: Measure cost per request before/after each lever

### Scenario 3: "Serve Llama-70B at 100 QPS"

- GPUs: 70B INT4 ≈ 35GB; need 4× A100 80G per instance
- Batching: Continuous batching (vLLM); target batch 16–32
- Replicas: 100 QPS ÷ (32 batch × 20 TPS) ≈ 2–4 replicas
- Load balancer + health checks

### Scenario 4: "How do you prevent hallucination?"

- RAG: Ground in retrieved docs
- Citation: Require [N] refs; verify support
- Lower temperature, top_p
- Verification: NLI or LLM check
- Explicit instructions: "Answer only from context"

---

## ✅ Production Checklist for LLM Systems

| Area | Checklist |
|------|-----------|
| **Serving** | [ ] Continuous batching enabled [ ] KV cache optimized [ ] Health checks |
| **Cost** | [ ] Token tracking [ ] Cost per customer [ ] Budget alerts |
| **Latency** | [ ] TTFT/TPS monitored [ ] p95 targets [ ] Streaming tested |
| **Quality** | [ ] Faithfulness eval [ ] Human/LLM-as-judge [ ] A/B testing |
| **Safety** | [ ] Input filters [ ] Output guards [ ] Rate limits |
| **RAG** | [ ] Index refresh pipeline [ ] Retrieval eval [ ] Citation verification |

---

## 📖 Interview Tips Summary

1. **Start with scale:** "What's the QPS? Tokens per request? Budget?"
2. **Draw the diagram:** Gateway → Router/Cache → RAG (if needed) → LLM → Post-process
3. **Quantify:** Use concrete numbers (7B vs 70B, $0.01 vs $0.05/1K tokens)
4. **Trade-offs:** Every choice has trade-off; state it explicitly
5. **RAG vs fine-tune:** RAG for knowledge; fine-tune for behavior/format
6. **Cost levers:** Smaller model, cache, router, quantization, shorter prompts
7. **Latency:** TTFT for UX; TPS for throughput; batching improves TPS, hurts TTFT

---

## 📎 Related Topics

| Topic | Link |
|-------|------|
| Model serving patterns | [05-model-serving/01-serving-patterns.md](../../phase-2-core-components/05-model-serving/01-serving-patterns.md) |
| Horizontal scaling | [07-scalability-performance/01-horizontal-scaling.md](../../phase-3-operations-and-reliability/07-scalability-performance/01-horizontal-scaling.md) |
| Caching strategies | [07-scalability-performance/02-caching-strategies.md](../../phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md) |
| Model monitoring | [06-monitoring-observability/01-model-monitoring.md](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md) |
| NLP systems | [10-end-to-end-systems/05-nlp-systems.md](../../phase-4-end-to-end-systems/10-end-to-end-systems/05-nlp-systems.md) |

---

## 🔄 LLM Systems vs Traditional ML Systems

| Dimension | Traditional ML | LLM / GenAI |
|-----------|----------------|-------------|
| **Input** | Fixed feature vector | Variable-length text (tokens) |
| **Output** | Single prediction (class, score) | Variable-length sequence (tokens) |
| **Inference** | One forward pass | Autoregressive (N passes for N tokens) |
| **Model size** | MB–GB | GB–100s GB |
| **Latency** | ms | TTFT + token-by-token |
| **Cost** | Per prediction | Per token (input + output) |
| **Hallucination** | N/A (deterministic) | Key risk; need RAG, citation |
| **Updates** | Retrain, deploy | Prompt, RAG, fine-tune |
| **Scaling** | More replicas | More GPUs, batching, quantization |

---

## 📊 Quick Reference: LLM Serving Frameworks (2025)

| Framework | Developer | Best For | Key Feature |
|-----------|-----------|----------|-------------|
| **vLLM** | Berkeley | High throughput, easy setup | PagedAttention, continuous batching |
| **TGI** | Hugging Face | HuggingFace models | Dynamic batching, streaming |
| **TensorRT-LLM** | NVIDIA | NVIDIA GPUs, max perf | Kernel fusion, quantization |
| **Triton** | OpenAI | Custom kernels | Low-level optimization |

---

**Continue to:** [01 - LLM Serving Infrastructure](./01-llm-serving-infrastructure.md)
