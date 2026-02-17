# ML Systems Estimation Reference

## Overview

This document is a **quick reference** for back-of-envelope calculations in ML Systems Design interviews. Memorize key numbers and formulas to reason quickly about QPS, latency, storage, and cost. Use it for rapid sanity checks and to demonstrate systems thinking.

---

## Latency Numbers Every ML Engineer Should Know

### Hardware & Network (Jeff Dean's Numbers, approx.)

| Operation | Latency | Notes |
|-----------|---------|-------|
| L1 cache reference | 0.5 ns | ~1× baseline |
| L2 cache reference | 7 ns | ~14× L1 |
| Main memory reference | 100 ns | ~200× L1 |
| SSD random read | 150 μs | ~300K× L1 |
| HDD seek | 10 ms | ~20M× L1 |
| Same rack, same DC | 0.25–0.5 ms | Low latency |
| Cross-DC (same region) | 1–5 ms | Regional |
| Cross-continent | 50–150 ms | High latency |

```
Relative scale (L1 = 1):
  L1:     1
  L2:     ~14
  RAM:    ~200
  SSD:    ~300,000
  DC RTT: ~1,000,000
  Cross-continent: ~100,000,000
```

### ML-Specific Latencies

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Feature store (Redis)** | 0.5–2 ms | In-memory, single key |
| **Feature store (DynamoDB)** | 5–10 ms | Single-digit ms |
| **Feature store (batch)** | 20–100 ms | Multi-key, batch |
| **Logistic regression** | 0.1–1 ms | Tiny model |
| **GBDT (XGBoost/LightGBM)** | 1–5 ms | 100–500 trees |
| **Neural network (small)** | 5–20 ms | MLP, small CNN |
| **Neural network (BERT-base)** | 10–50 ms | Depends on hardware |
| **Transformer (large)** | 50–200 ms | GPT-style |
| **LLM inference** | 100 ms–10 s | Token-by-token or batched |
| **ANN search (FAISS/HNSW)** | 1–10 ms | 1M–100M vectors |
| **Kafka produce** | 1–5 ms | Per message |
| **Kafka consume** | 1–10 ms | Per batch |
| **S3 get object** | 10–50 ms | First byte |
| **BigQuery scan** | 1–30 s | Depends on size |

---

## Model Size Reference

| Model | Parameters | Size (FP32) | Size (FP16) | Use Case |
|-------|------------|-------------|-------------|----------|
| Logistic regression | 10K–1M | KB–MB | KB–MB | CTR, simple classification |
| Random forest | 1K trees | MB | MB | Tabular, interpretable |
| XGBoost/LightGBM | 100–1K trees | MB | MB | Tabular, ranking |
| ResNet-50 | 25M | ~100 MB | ~50 MB | Image classification |
| BERT-base | 110M | ~440 MB | ~220 MB | NLP, embeddings |
| BERT-large | 340M | ~1.3 GB | ~650 MB | NLP |
| GPT-2 | 1.5B | ~6 GB | ~3 GB | Text generation |
| LLaMA-7B | 7B | ~28 GB | ~14 GB | LLM |
| LLaMA-13GB (FP16) | 7B | — | ~13 GB | Common deployment |
| LLaMA-70B | 70B | ~280 GB | ~140 GB | Large LLM |
| LLaMA-70B (FP16) | 70B | — | ~130 GB | Typical |

### Size Formula

$$
\text{Model Size (bytes)} = \text{params} \times \text{bytes\_per\_param}
$$

- FP32: 4 bytes/param  
- FP16: 2 bytes/param  
- INT8: 1 byte/param  

---

## Embedding Table Sizing

### Formula

$$
\text{Size (bytes)} = \text{num\_entities} \times \text{embedding\_dim} \times \text{bytes\_per\_float}
$$

### Examples

| Entities | Dim | Bytes/float | Size |
|----------|-----|-------------|------|
| 1M users | 64 | 4 | 256 MB |
| 10M users | 128 | 4 | 5 GB |
| 100M users | 256 | 4 | 100 GB |
| 1B users | 64 | 4 | 256 GB |
| 1B users | 256 | 4 | 1 TB |
| 10B items | 64 | 4 | 2.5 TB |

### Strategies to Reduce Size

| Strategy | Idea | Trade-off |
|----------|------|-----------|
| **Hashing** | Hash entity → bucket; share embedding | Collisions, quality loss |
| **Bucketing** | Group similar entities | Less granular |
| **Mixed-dimension** | Low dim for tail, high for head | More complexity |
| **Quantization** | FP16 or INT8 | Slight quality loss |
| **Pruning** | Sparse embeddings | Complex implementation |

---

## Feature Store Sizing

### Online Store

$$
\text{Storage} = \text{num\_entities} \times \text{num\_features} \times \text{avg\_feature\_size}
$$

| Scenario | Entities | Features | Bytes/feature | Storage |
|---------|----------|----------|---------------|---------|
| Small | 1M | 100 | 4 | 400 MB |
| Medium | 50M | 200 | 4 | 40 GB |
| Large | 500M | 200 | 4 | 400 GB |
| Very large | 1B | 300 | 4 | 1.2 TB |

### QPS Estimation

- **Read QPS** = prediction QPS × features per request (before caching)
- With 95% cache hit: backend QPS = 0.05 × prediction QPS × features_per_request
- Example: 100K pred/s × 50 features × 0.05 = 250K feature reads/s

---

## Common Scaling Breakpoints

| Scale | Approx. QPS | Architecture | Notes |
|-------|-------------|--------------|-------|
| **Single machine** | < 100 | 1 server | Simple deployment |
| **Load balanced** | 100–1K | LB + 2–5 replicas | Basic HA |
| **Distributed** | 1K–10K | LB, many replicas, caching | Need caching |
| **High scale** | 10K–100K | Sharding, Redis, CDN | Feature/prediction cache |
| **Very high** | 100K–1M | Multi-region, aggressive cache | Precomputation, edge |
| **Massive** | > 1M | Full distributed, regional | Ad systems, big tech |

```
    100 QPS    1K QPS    10K QPS    100K QPS    1M QPS
       │          │          │           │          │
       ▼          ▼          ▼           ▼          ▼
    ┌─────┐   ┌─────┐    ┌─────┐     ┌─────┐    ┌─────┐
    │ 1   │   │ LB  │    │Cache│     │Shard│    │Multi│
    │ box │   │+few │    │ +   │     │ +   │    │region│
    │     │   │repli│    │repli│     │cache│    │     │
    └─────┘   └─────┘    └─────┘     └─────┘    └─────┘
```

---

## GPU Throughput Reference

### BERT-base (110M params) Inference

| GPU | Batch 1 | Batch 8 | Batch 32 | Notes |
|-----|---------|---------|----------|-------|
| T4 | ~100 | ~300 | ~400 | Common cloud GPU |
| A10G | ~200 | ~600 | ~800 | Newer cloud |
| A100 | ~400 | ~1000 | ~1500 | High throughput |
| H100 | ~800 | ~2000 | ~3000 | ~2× A100 |

### LLM Inference (7B, 1 token)

| GPU | Throughput (tokens/s) | Notes |
|-----|------------------------|-------|
| A100 40GB | 50–100 | Single GPU |
| A100 80GB | 80–150 | Larger context |
| H100 | 100–200 | Faster |
| 8× A100 | 400–800 | Multi-GPU |

### Batch Size Impact

- Larger batch → higher throughput, higher latency
- Sweet spot: balance latency SLO vs utilization
- Rule of thumb: 2–4× latency for 4× batch often acceptable

---

## Storage Growth Estimation

### Event Logging

$$
\text{Storage} = \text{events\_per\_second} \times \text{avg\_event\_size} \times \text{retention\_seconds}
$$

| Parameter | Typical | Example |
|-----------|---------|---------|
| Event size | 100–500 B | 200 B |
| Retention | 7–90 days | 30 days |
| Seconds in 30 days | 2.59M | — |

**Example:** 100K events/s × 200 B × 2.59M s ≈ 5.2 PB (30 days)

### Training Data

- Historical data: GB–PB depending on domain
- Image: ~1 MB/image; 1M images ≈ 1 TB
- Text: ~1 KB/document; 1B docs ≈ 1 TB
- Tabular: varies widely; 1B rows × 100 cols × 4 B ≈ 400 GB

---

## Bandwidth Estimation

### Request/Response

| Component | Size | Notes |
|-----------|------|-------|
| Request (features) | 1–50 KB | Depends on feature count |
| Response (prediction) | 100 B–1 KB | Score, class, metadata |
| Full request+response | ~2–100 KB | Round trip |

**Bandwidth** = QPS × (request_size + response_size)

- 100K QPS × 10 KB ≈ 1 GB/s ≈ 8 Gbps
- 1M QPS × 2 KB ≈ 2 GB/s ≈ 16 Gbps

### Feature Store

- Read: QPS × features × bytes_per_feature
- Example: 50K × 50 × 4 = 10 MB/s per direction

### Model Updates

- Model size / update frequency
- 440 MB model, daily: ~5 KB/s average
- 14 GB model, weekly: ~23 KB/s average

---

## Powers of 2 Quick Reference

| Power | Value | Bytes |
|-------|-------|-------|
| 2¹⁰ | 1,024 | 1 KB |
| 2²⁰ | ~1.05M | 1 MB |
| 2³⁰ | ~1.07B | 1 GB |
| 2⁴⁰ | ~1.1T | 1 TB |
| 2⁵⁰ | ~1.1P | 1 PB |
| 2⁶⁰ | ~1.15E | 1 EB |

### Useful Approximations

- 1 million = 10⁶
- 1 billion = 10⁹
- Seconds/day = 86,400 ≈ 10⁵
- 1 GB = 10⁹ bytes
- 1 TB = 10¹² bytes

---

## Quick Reference Tables

### QPS Formulas

| Estimate | Formula |
|----------|---------|
| Avg QPS | DAU × actions_per_user_per_day ÷ 86,400 |
| Peak QPS | Avg QPS × 2–3 (typical) |
| Peak QPS (event) | Avg QPS × 5–10 |
| Backend QPS (cached) | QPS × (1 − cache_hit_rate) |

### Storage Formulas

| Estimate | Formula |
|----------|---------|
| Event storage | events_per_day × bytes_per_event × retention_days |
| Embedding storage | entities × dim × 4 bytes |
| Model size (FP32) | params × 4 |
| Feature store | entities × features × bytes_per_feature |

### Compute Formulas

| Estimate | Formula |
|----------|---------|
| Servers (CPU) | Peak QPS ÷ (QPS per server × headroom) |
| QPS per server | 500–5K (model-dependent) |
| GPUs (inference) | Peak QPS ÷ (inferences_per_GPU_per_sec × utilization) |
| Batch slots | Concurrent_requests ÷ batch_size |

---

## Latency Budget (Real-time ML)

```
┌────────────────────────────────────────────────────────────┐
│  Total budget: 50–100 ms (recommendation, ads)             │
├────────────────────────────────────────────────────────────┤
│  Component           │  Typical    │  Budget               │
├──────────────────────┼─────────────┼───────────────────────┤
│  LB / routing        │  1–2 ms     │  ██                   │
│  Auth                │  1–2 ms     │  ██                   │
│  Feature fetch       │  5–20 ms    │  ██████████           │
│  Model inference     │  10–50 ms   │  ████████████████████  │
│  Post-process        │  1–5 ms     │  ███                  │
│  Network             │  2–5 ms     │  ████                 │
└────────────────────────────────────────────────────────────┘
```

---

## Worked Example: Recommendation for 500M DAU

### Assumptions

| Parameter | Value |
|-----------|-------|
| DAU | 500M |
| Sessions/user/day | 5 |
| Requests/session | 1 |
| Peak factor | 2.5 |
| Cache hit rate | 0 (conservative) |

### QPS

- Daily actions: 500M × 5 = 2.5B
- Avg QPS: 2.5B ÷ 86,400 ≈ 29K
- Peak QPS: 29K × 2.5 ≈ 72K

### Storage (30-day events)

- 2.5B events/day × 300 B = 750 GB/day
- 30 days ≈ 22 TB

### Servers

- ~2K QPS per server (small model)
- 72K ÷ 2K × 1.5 headroom ≈ 54 servers

### Feature Store

- 500M users × 200 features × 4 B = 400 GB
- Read QPS: 72K × 50 features = 3.6M reads/s (before cache)

---

## Python: Quick Estimation Helpers

```python
"""
Quick estimation helpers for ML systems design interviews.
"""

def qps_from_dau(dau: int, actions_per_user: float, peak_factor: float = 2.5) -> tuple[float, float]:
    """Returns (avg_qps, peak_qps)."""
    daily = dau * actions_per_user
    avg = daily / 86400
    peak = avg * peak_factor
    return avg, peak

def storage_events(events_per_day: int, bytes_per_event: int = 200, days: int = 30) -> float:
    """Storage in bytes."""
    return events_per_day * bytes_per_event * days

def embedding_size(entities: int, dim: int = 256, bytes_per_float: int = 4) -> float:
    """Embedding table size in bytes."""
    return entities * dim * bytes_per_float

def model_size_params(params: int, fp16: bool = False) -> float:
    """Model size in bytes."""
    return params * (2 if fp16 else 4)

def servers_for_qps(peak_qps: float, qps_per_server: float = 2000, headroom: float = 1.5) -> int:
    """Number of servers."""
    return int(peak_qps / qps_per_server * headroom)

# Quick checks
if __name__ == "__main__":
    avg, peak = qps_from_dau(500_000_000, 5)
    print(f"500M DAU, 5 actions: avg={avg:,.0f} QPS, peak={peak:,.0f} QPS")
    
    emb = embedding_size(1_000_000_000, 64)
    print(f"1B entities, 64 dim: {emb/1e9:.1f} GB")
    
    size = model_size_params(110_000_000)
    print(f"BERT-base: {size/1e6:.0f} MB")
```

---

## Interview Tips

1. **State assumptions**: "Assume 500M DAU and 5 sessions per user..."
2. **Round aggressively**: 86,400 → 100K; 2.5B/100K → 25K
3. **Sanity check**: 1B users, 5 actions/day → 5B/86400 ≈ 58K QPS
4. **Use latency tiers**: L1 < L2 < RAM < SSD < network; cite approximate ratios
5. **Mention caveats**: "This assumes no cache; with 80% hit, backend load is 20%."
6. **Know scale**: Startup ~1K QPS; Mid ~10K–100K; Big tech ~1M+
7. **Model sizes**: BERT ~440MB, LLaMA-7B ~14GB, LLaMA-70B ~130GB
8. **Quick embedding math**: 1B × 64 × 4 = 256 GB

---

## Related Topics

- [Back-of-Envelope Estimation](../../phase-5-advanced-topics/15-capacity-cost-planning/01-back-of-envelope-estimation.md) - Full methodology
- [Horizontal Scaling](./01-horizontal-scaling.md) - Scaling at these breakpoints
- [Caching Strategies](./02-caching-strategies.md) - Reducing backend QPS
- [Optimization Techniques](./04-optimization-techniques.md) - Improving latency and throughput
