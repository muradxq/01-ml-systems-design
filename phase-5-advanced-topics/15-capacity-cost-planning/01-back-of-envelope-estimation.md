# Back-of-Envelope Estimation

## Overview

Back-of-envelope estimation is a **core interview skill** for ML Systems Design. You'll be asked to size systems ("How many servers? How much storage? What's the QPS?") and must reason quickly with approximate numbers. This guide provides reference tables, methodology, and worked examples with Python code.

---

## Powers of 2 Reference Table

Memorize these for quick mental math:

| Power | Value | Bytes | Common Use |
|-------|-------|-------|------------|
| 2¹⁰ | 1,024 | 1 KB | Cache line, small object |
| 2²⁰ | ~1M | 1 MB | Small model, image |
| 2³⁰ | ~1B | 1 GB | Embedding table, large model |
| 2⁴⁰ | ~1T | 1 TB | Daily logs, dataset |
| 2⁵⁰ | ~1P | 1 PB | Data lake, warehouse |
| 2⁶⁰ | ~1E | 1 EB | Global scale data |

### Useful Approximations

- **1 million** = 10⁶
- **1 billion** = 10⁹
- **1 KB** ≈ 10³ bytes
- **1 MB** ≈ 10⁶ bytes
- **1 GB** ≈ 10⁹ bytes
- **1 TB** ≈ 10¹² bytes
- **1 PB** ≈ 10¹⁵ bytes

### Sizing Examples

| Item | Size | Calculation |
|------|------|-------------|
| Single embedding (256-dim float32) | 1 KB | 256 × 4 = 1024 B |
| 1M embeddings (256-dim) | 1 GB | 1M × 1 KB |
| BERT-base (110M params) | ~440 MB | 110M × 4 bytes |
| GPT-2 (1.5B params) | ~6 GB | 1.5B × 4 bytes |
| Log event (100 bytes) | 100 B | User ID, timestamp, action |

---

## Latency Numbers Every Engineer Should Know

Reference: Jeff Dean's "Numbers Everyone Should Know" (approximate, 2020s):

| Operation | Latency | Relative to L1 |
|-----------|---------|----------------|
| L1 cache reference | 0.5 ns | 1× |
| L2 cache reference | 7 ns | 14× |
| Main memory reference | 100 ns | 200× |
| SSD random read | 150 μs | 300,000× |
| HDD seek | 10 ms | 20,000,000× |
| Round trip same datacenter | 0.5 ms | 1,000,000× |
| Round trip cross-continent | 100–150 ms | 200–300M× |

### Implications for ML Systems

| Scenario | Latency Budget | Constraint |
|----------|----------------|------------|
| Real-time recommendation | 50–100 ms | Must include feature fetch + model + network |
| Ad prediction | 10–50 ms | Often < 50 ms for auction |
| Search ranking | 100–200 ms | User waits for results |
| Batch training | Hours | Latency less critical |

---

## QPS Estimation Methodology

### Step 1: Define the Action

What triggers a request? (e.g., feed refresh, search query, ad impression)

### Step 2: Count Users and Frequency

$$
\text{QPS} = \frac{\text{DAU} \times \text{Actions per user per day}}{\text{Seconds per day}}
$$

Seconds per day = 86,400.

### Step 3: Peak vs Average

Traffic is rarely uniform:
- **Peak QPS** ≈ 2–3× average for consumer apps
- **Peak** ≈ 5–10× for events (product launch, Black Friday)

$$
\text{Peak QPS} = \text{Avg QPS} \times \text{Peak factor}
$$

### Step 4: Account for Caching

- Cache hit rate 80% → only 20% of QPS hits backend
- **Backend QPS** = QPS × (1 - cache_hit_rate)

---

## Storage Estimation

### Components

| Data Type | Retention | Size per Record | Formula |
|-----------|-----------|-----------------|---------|
| Event logs | 30–90 days | 100–500 B | events/day × bytes × days |
| Features | Real-time + batch | Varies | users × features × bytes |
| Model artifacts | Versioned | GB per model | Checkpoints, ONNX |
| Embeddings | Refreshed daily | users × dims × 4 | e.g., 1B × 256 × 4 |

### Example: Event Logs

- 1B DAU, 10 events/user/day = 10B events/day
- 200 bytes/event = 2 TB/day
- 90-day retention = 180 TB

---

## Bandwidth Estimation

### Inbound (Ingest)

- Events: QPS × bytes/event
- Example: 100K QPS × 200 B = 20 MB/s ≈ 160 Mbps

### Outbound (Serving)

- Predictions: QPS × response size
- Example: 100K QPS × 1 KB = 100 MB/s ≈ 800 Mbps

### Cross-Region

- Often the most expensive (egress fees)
- Minimize by regional deployment

---

## Worked Example 1: Recommendation System for 1B Users

### Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| DAU | 500M | 50% of 1B |
| Sessions/user/day | 5 | Feed refreshes |
| Requests/session | 1 | One recommendation call per session |
| Peak factor | 2.5 | |
| Cache hit rate | 0 | Conservative |

### QPS

$$
\text{Actions/day} = 500M \times 5 = 2.5B
$$
$$
\text{Avg QPS} = \frac{2.5B}{86400} \approx 29K
$$
$$
\text{Peak QPS} = 29K \times 2.5 \approx 72K
$$

### Storage (30-day events)

- 2.5B events/day × 300 bytes ≈ 750 GB/day
- 30 days ≈ **22 TB**

### Servers (Ballpark)

- 1 CPU server ≈ 1–5K QPS (recommendation inference with small model)
- 72K / 2K ≈ **36 servers** (with headroom: 50–70)

---

## Worked Example 2: Ad Prediction at 10M QPS

### Assumptions

| Parameter | Value |
|-----------|-------|
| QPS | 10M |
| Latency budget | 20 ms |
| Model | 100M params, 10 ms inference |

### Compute

- 10M QPS, 10 ms per request → 10M × 0.01 = 100K concurrent requests
- With batching (e.g., 64): 100K / 64 ≈ 1,600 batch slots
- 1 GPU ≈ 1–2K batch inferences/sec (depending on model)
- GPUs: 10M / (1.5K × 3600) ≈ **~2K GPUs** (simplified; real systems use massive parallelism)

*Refined*: Ad systems use highly optimized serving (TensorRT, batching, many small models). Real systems at 10M QPS might use hundreds to low thousands of GPUs across many smaller model instances.

### Bandwidth

- 10M QPS × 500 B response ≈ 5 GB/s ≈ 40 Gbps

---

## Worked Example 3: Search Ranking System

### Assumptions

| Parameter | Value |
|-----------|-------|
| DAU | 200M |
| Searches/user/day | 3 |
| Peak factor | 3 |

### QPS

$$
\text{Avg QPS} = \frac{200M \times 3}{86400} \approx 7K
$$
$$
\text{Peak QPS} \approx 21K
$$

### Latency Budget

- User expects results in < 200 ms
- Retrieval: 20 ms
- Ranking (neural): 50 ms
- Reranking: 30 ms
- Total: ~100 ms (within budget)

---

## Worked Example 4: Feature Store Sizing

### Assumptions

| Parameter | Value |
|-----------|-------|
| Users | 500M |
| Features per user | 200 |
| Bytes per feature | 4 (float32) |
| Refresh | Daily batch + real-time for 10% |

### Storage

- 500M × 200 × 4 = **400 GB** (batch)
- Real-time: 50M users × 50 features (hot) × 4 ≈ 10 GB
- Total: ~**500 GB** (before replication)

### QPS (Read)

- 100K recommendation QPS × 50 features = 5M feature reads/sec
- With caching (95% hit): 250K backend reads/sec

---

## Python: Estimation Calculator

```python
"""
Back-of-envelope estimation calculator for ML systems.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QPSEstimate:
    """QPS estimation result."""
    avg_qps: float
    peak_qps: float
    daily_actions: float


def estimate_qps(
    dau: int,
    actions_per_user_per_day: float,
    peak_factor: float = 2.5,
) -> QPSEstimate:
    """
    Estimate QPS from DAU and user behavior.
    
    Args:
        dau: Daily active users
        actions_per_user_per_day: e.g., 5 feed refreshes
        peak_factor: Peak QPS / Avg QPS (typically 2-3)
    """
    daily_actions = dau * actions_per_user_per_day
    seconds_per_day = 86400
    avg_qps = daily_actions / seconds_per_day
    peak_qps = avg_qps * peak_factor
    return QPSEstimate(avg_qps=avg_qps, peak_qps=peak_qps, daily_actions=daily_actions)


def estimate_storage(
    events_per_day: int,
    bytes_per_event: int = 200,
    retention_days: int = 30,
) -> float:
    """Storage in bytes for event logs."""
    return events_per_day * bytes_per_event * retention_days


def estimate_servers(
    peak_qps: float,
    qps_per_server: float = 2000,
    headroom: float = 1.5,
) -> int:
    """Number of servers for given QPS."""
    return int(peak_qps / qps_per_server * headroom)


def estimate_embedding_storage(
    num_entities: int,
    embedding_dim: int = 256,
    bytes_per_float: int = 4,
) -> float:
    """Storage in bytes for embedding table."""
    return num_entities * embedding_dim * bytes_per_float


# Example
if __name__ == "__main__":
    # Recommendation for 1B users
    qps = estimate_qps(dau=500_000_000, actions_per_user_per_day=5)
    print(f"Avg QPS: {qps.avg_qps:,.0f}, Peak: {qps.peak_qps:,.0f}")
    
    storage = estimate_storage(qps.daily_actions, bytes_per_event=300, retention_days=30)
    print(f"Storage: {storage / 1e12:.1f} TB")
    
    servers = estimate_servers(qps.peak_qps)
    print(f"Servers (est): {servers}")
    
    emb_storage = estimate_embedding_storage(500_000_000, 256)
    print(f"Embedding storage: {emb_storage / 1e9:.1f} GB")
```

---

## Interview Tips

1. **State assumptions** aloud: "Let's say 500M DAU and 5 sessions per user..."
2. **Round aggressively**: 86,400 → 100K; 2.5B/100K → 25K QPS
3. **Sanity check**: 1B users, 5 actions/day → 5B actions → 5B/86400 ≈ 60K QPS (ballpark)
4. **Mention caveats**: "This assumes no caching; with 80% cache we'd need 20% of the compute"
5. **Know the scale**: Meta/Google scale = 100B+ events/day; startup = 1M–10M

---

## Quick Reference: Estimation Formulas

| Estimate | Formula |
|----------|---------|
| QPS | DAU × actions/day ÷ 86,400 |
| Peak QPS | Avg QPS × 2–3 |
| Event storage | events/day × bytes × days |
| Embedding storage | entities × dims × 4 bytes |
| Servers | Peak QPS ÷ QPS per server × headroom |

---

## Latency Budget Breakdown (Real-time ML)

```
User Request
    │
    ├─ Load balancer:        1–2 ms
    ├─ Auth / routing:       1–2 ms
    ├─ Feature fetch:        5–20 ms  (cache vs DB)
    ├─ Model inference:      5–50 ms  (depends on model)
    ├─ Post-processing:      1–5 ms
    └─ Response:             1–2 ms
    ─────────────────────────────────
    Total:                   14–81 ms
```

Target: Keep P99 under 100 ms for most interactive ML (recommendations, ads).

---

## Additional Worked Examples

### Example 5: Real-Time Fraud Detection

- 1M transactions/day = 1M / 86400 ≈ 12 QPS avg
- Peak 3× = 36 QPS
- Model: 1 ms inference → 1 GPU can handle 1000 QPS → 1 GPU sufficient (with headroom: 2–3)

### Example 6: Voice Assistant (ASR + NLU)

- 100M DAU, 5 requests/user/day = 500M requests/day ≈ 5.8K QPS avg
- Peak 2.5× = 14.5K QPS
- ASR: 100 ms/request → 10 concurrent per core → ~1500 cores for ASR leg
- NLU: 50 ms → similar scale
- Total: hundreds of CPU instances (voice often CPU-bound for inference)