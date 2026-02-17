# Cost Modeling for ML Systems

## Overview

ML systems are expensive. Training large models costs millions in GPU time; inference at scale consumes thousands of servers. Interviewers expect you to build cost models, compare build vs buy, and optimize spend. This guide covers GPU costs, storage tiers, network, feature stores, labeling, TCO, and optimization strategies.

---

## GPU Costs: Training and Inference

### Cloud Pricing (2024–2025, Approximate)

| GPU | Memory | FP16 TFLOPS | Training $/hr | Inference $/hr | Notes |
|-----|--------|-------------|--------------|----------------|------|
| T4 | 16 GB | 65 | $0.50–0.70 | $0.35–0.50 | Budget inference |
| L4 | 24 GB | 121 | $0.80–1.20 | $0.60–0.90 | Good inference/$ |
| A10G | 24 GB | 125 | $1.00–1.50 | $0.75–1.00 | Common inference |
| A100 40GB | 40 GB | 312 | $2.50–3.50 | $2.00–2.50 | Workhorse training |
| A100 80GB | 80 GB | 312 | $3.50–4.50 | $2.50–3.50 | Large model training |
| H100 | 80 GB | 989 | $3–5+ | $3–4 | Supply constrained |

### Spot vs On-Demand

| Type | Discount | Use Case |
|------|----------|----------|
| On-demand | 0% | Production inference; must be reliable |
| Spot / Preemptible | 60–90% | Training; can checkpoint and restart |
| Reserved (1 yr) | 30–50% | Steady inference workload |
| Reserved (3 yr) | 50–65% | Committed capacity |

### Training Cost Example

- **BERT-large** training: ~1 day on 8× A100 → 8 × 24 × $3 ≈ **$576**
- **LLM 7B** training: ~1 week on 64× A100 → 64 × 168 × $3 ≈ **$32K**
- **LLM 70B** training: Weeks on hundreds of GPUs → **$1M+**

### Inference Cost Example

- 10K QPS, 10 ms/request → 100 concurrent inferences
- 1 A100 can do ~100–500 inferences/sec (model dependent)
- GPUs needed: 10K / 300 ≈ 34 → 34 × $2.50 × 720 hr/mo ≈ **$61K/mo** (simplified; real systems use batching, CPU for small models)

---

## Storage Costs by Tier

| Tier | $/GB/month | Latency | Use Case |
|------|------------|---------|----------|
| **Hot (SSD)** | $0.10–0.25 | ms | Feature store, hot data |
| **Warm** | $0.02–0.05 | seconds | Recent logs, checkpoints |
| **Cold (object)** | $0.01–0.02 | minutes | Archives, old data |
| **Glacier** | $0.004–0.01 | hours | Compliance, long-term |

### Sizing Impact

- 1 PB hot: $100K–250K/mo
- 1 PB cold: $10K–20K/mo
- Move cold data to warm only when needed

---

## Network / Data Transfer Costs

| Transfer | $/GB (Approx) | Notes |
|----------|---------------|------|
| Ingress | $0 | Usually free |
| Same region | $0.01–0.02 | Cheap |
| Cross-region | $0.02–0.08 | Expensive |
| Internet egress | $0.08–0.12 | Most expensive |

### ML Implications

- **Training**: Data in same region as GPU → minimal transfer
- **Inference**: Serve from region close to users → reduce egress
- **Feature store**: Co-locate with inference

---

## Feature Store Costs

| Component | Cost Driver | Typical Range |
|-----------|-------------|---------------|
| **Compute** | Batch jobs, real-time serving | $/QPS or $/hour |
| **Storage** | Feature tables, embeddings | $/GB/month |
| **Serving** | Low-latency reads | Cache + DB costs |

### Example: Feast / Tecton

- Managed feature store: ~$5K–50K/mo for mid-scale (10M features, 100K QPS)
- Self-hosted: GPU/CPU for compute + storage + ops

---

## Human Labeling Costs

| Task | $/label (Approx) | Scale |
|------|------------------|------|
| Simple (binary) | $0.01–0.05 | Millions |
| Classification (10 classes) | $0.05–0.15 | Hundreds of K |
| Bounding boxes | $0.10–0.50 | Tens of K |
| Segmentation | $0.50–2.00 | Tens of K |
| LLM alignment (RLHF) | $1–10+ | Thousands |

### Budget Example

- 1M labels at $0.02 = **$20K**
- 10K RLHF examples at $5 = **$50K**

---

## Total Cost of Ownership (TCO)

TCO = **Compute + Storage + Network + People + Tooling + Downtime**

| Category | What to Include |
|----------|-----------------|
| **Compute** | Training + inference GPUs/CPUs |
| **Storage** | Data lake, features, checkpoints |
| **Network** | Egress, cross-region |
| **People** | ML engineers, MLOps, data engineers |
| **Tooling** | Feature store, experiment platform, monitoring |
| **Risk** | Downtime, incidents |

### Simple TCO Formula

$$
\text{TCO}_{year} = \text{Cloud}_{year} + \text{People}_{year} + \text{Tooling}_{year}
$$

---

## Build vs Buy Decision Framework

| Factor | Build | Buy |
|--------|-------|-----|
| **Volume** | Very high (10M+ QPS) | Low–mid |
| **Customization** | Need deep customization | Standard workflows OK |
| **Team** | Large eng team | Small team |
| **Timeline** | 6–12+ months | 1–3 months |
| **TCO** | Lower at scale | Lower at small scale |

### When to Build

- Unique requirements (e.g., sub-ms latency, custom feature logic)
- Scale where vendor pricing doesn't fit (e.g., 100B features)
- Strategic differentiator

### When to Buy

- Standard use case (feature store, MLOps)
- Small team, need speed
- Uncertain scale

---

## Python: ML System Cost Estimator

```python
"""
ML system cost estimator for interviews and planning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CostEstimate:
    """Cost breakdown."""
    training_monthly: float
    inference_monthly: float
    storage_monthly: float
    network_monthly: float
    total_monthly: float


# Default prices (approximate, 2024)
GPU_A100_HR = 3.0
GPU_T4_HR = 0.5
STORAGE_HOT_PER_GB = 0.15
STORAGE_COLD_PER_GB = 0.02
NETWORK_EGRESS_PER_GB = 0.09


def estimate_training_cost(
    gpu_hours_per_day: float,
    gpu_price_per_hr: float = GPU_A100_HR,
    days_per_month: int = 30,
) -> float:
    """Monthly training cost."""
    return gpu_hours_per_day * days_per_month * gpu_price_per_hr


def estimate_inference_cost(
    qps: float,
    ms_per_request: float = 10,
    gpus_per_10k_qps: float = 30,  # Model-dependent
    gpu_price_per_hr: float = GPU_A100_HR,
    hours_per_month: int = 720,
) -> float:
    """Monthly inference cost (simplified)."""
    gpus = (qps / 10_000) * gpus_per_10k_qps
    return gpus * gpu_price_per_hr * hours_per_month


def estimate_storage_cost(
    storage_gb: float,
    hot_fraction: float = 0.2,
) -> float:
    """Monthly storage cost."""
    hot = storage_gb * hot_fraction * STORAGE_HOT_PER_GB
    cold = storage_gb * (1 - hot_fraction) * STORAGE_COLD_PER_GB
    return hot + cold


def estimate_network_cost(
    egress_gb_per_month: float,
) -> float:
    """Monthly network egress cost."""
    return egress_gb_per_month * NETWORK_EGRESS_PER_GB


def estimate_ml_system_cost(
    training_gpu_hr_per_day: float = 100,
    inference_qps: float = 50_000,
    storage_tb: float = 100,
    egress_tb_per_month: float = 50,
) -> CostEstimate:
    """Full ML system cost estimate."""
    training = estimate_training_cost(training_gpu_hr_per_day)
    inference = estimate_inference_cost(inference_qps)
    storage = estimate_storage_cost(storage_tb * 1024)
    network = estimate_network_cost(egress_tb_per_month * 1024)
    return CostEstimate(
        training_monthly=training,
        inference_monthly=inference,
        storage_monthly=storage,
        network_monthly=network,
        total_monthly=training + inference + storage + network,
    )


# Example
if __name__ == "__main__":
    cost = estimate_ml_system_cost(
        training_gpu_hr_per_day=200,
        inference_qps=100_000,
        storage_tb=500,
    )
    print(f"Training: ${cost.training_monthly:,.0f}/mo")
    print(f"Inference: ${cost.inference_monthly:,.0f}/mo")
    print(f"Storage: ${cost.storage_monthly:,.0f}/mo")
    print(f"Total: ${cost.total_monthly:,.0f}/mo")
```

---

## Optimization Strategies

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **Spot instances** | 60–90% on training | Preemption; need checkpointing |
| **Reserved capacity** | 30–50% on inference | Commitment |
| **Auto-scaling** | 20–40% | Complexity; scale-down delay |
| **Model compression** | 2–4× fewer GPUs | Slight accuracy loss |
| **Batch inference** | 5–10× throughput | Higher latency |
| **Cold storage** | 5–10× vs hot | Retrieval delay |

---

## Trade-offs and Interview Tips

### Interview Tips

1. **"What's the monthly cost?"** → Break into training, inference, storage, network; use ballpark numbers.
2. **"How do you reduce costs?"** → Spot for training, reserved for inference, auto-scaling, model compression.
3. **"Build vs buy feature store?"** → TCO over 3 years; team size; customization needs.
4. **"Our inference cost doubled"** → Check QPS growth, model size, batching, spot vs on-demand mix.

---

## Real-World Cost Ranges (2024)

| System Type | Monthly Cost (Ballpark) | Scale |
|-------------|-------------------------|-------|
| Small startup (1M users) | $10K–50K | Single region, modest models |
| Mid (100M users) | $500K–2M | Multi-region, feature store |
| Large (1B+ users) | $5M–50M+ | Full ML infra, many models |

---

## Cost Optimization Checklist

- [ ] Use spot/preemptible for training (with checkpointing)
- [ ] Reserved instances for steady inference (1–3 yr)
- [ ] Auto-scale down during low traffic
- [ ] Model quantization (FP16, int8) to reduce GPU count
- [ ] Batch inference where latency allows
- [ ] Move cold data to cheaper storage tier
- [ ] Minimize cross-region egress
- [ ] Right-size instances (avoid over-provisioning)

---

## Build vs Buy: Feature Store TCO Example

**Build (2 engineers, 1 year)**:
- Engineers: 2 × $200K = $400K
- Cloud (dev): ~$50K
- Total year 1: ~$450K

**Buy (Tecton, Feathr)**:
- $50K–150K/year for mid-scale
- Break-even: Build wins if scale is huge and team already exists; Buy wins for small teams.

---

## Detailed GPU Cost Scenarios

### Scenario 1: Fine-tuning BERT for 1M samples

- 8× A100, ~4 hours
- Cost: 8 × 4 × $3 = **$96**

### Scenario 2: Training ResNet-50 from scratch (ImageNet)

- 8× A100, ~2 days
- Cost: 8 × 48 × $3 ≈ **$1,150**

### Scenario 3: LLM 7B pretraining (1T tokens)

- 256× A100, ~2 weeks
- Cost: 256 × 336 × $3 ≈ **$258K** (on-demand)
- With spot (70% off): ~**$77K**

### Scenario 4: Inference at 100K QPS

- Assume 50 ms/request, batch size 32
- Effective throughput: 32 / 0.05 = 640 req/s per batch
- Batches needed: 100K / 640 ≈ 157
- With 2 batch slots per GPU: 157/2 ≈ 80 GPUs
- Cost: 80 × $2.50 × 720 ≈ **$144K/month**

---

## Storage Tier Decision Tree

```
Data access pattern?
    │
    ├─ Sub-second (feature serving)  → Hot (SSD)
    ├─ Daily/weekly (batch jobs)     → Warm
    └─ Rare (compliance, archive)    → Cold / Glacier
```

---

## Network Cost Optimization

| Strategy | Savings |
|----------|---------|
| Co-locate training data with GPUs | Eliminate cross-region transfer |
| Regional deployment (serve from user region) | Reduce egress |
| Compression (gzip, protobuf) | 50–80% less bytes |
| CDN for model artifacts | Reduce origin egress |

---

## Interview: Cost Discussion Framework

1. **Break down components**: Training, inference, storage, network
2. **State assumptions**: "Assuming 100K QPS, 10 ms inference..."
3. **Use round numbers**: $3/hr for A100, 720 hr/mo
4. **Mention optimizations**: Spot, reserved, quantization
5. **Compare to scale**: "At 1M users this is ~$X; at 100M it scales to ~$Y"