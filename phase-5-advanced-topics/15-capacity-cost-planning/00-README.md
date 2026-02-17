# 📊 Capacity & Cost Planning for ML Systems

## Overview

Capacity and cost planning are **critical skills** for ML engineers at L5/E5+ at Meta, Google, Amazon, and similar companies. ML systems consume vast compute (GPUs, CPUs), storage, and network resources—and costs scale nonlinearly with model size and traffic. Interviewers expect you to estimate requirements, model costs, and plan for growth.

This section covers back-of-envelope estimation, cost modeling for ML workloads, and capacity planning strategies. You'll learn the numbers every ML engineer should know and how to apply them in system design interviews.

---

## 🎯 Why This Matters at Meta/Google (L5/E5+)

At senior levels, you're expected to:

| Expectation | What It Means |
|-------------|---------------|
| **Sizing systems** | Estimate QPS, storage, bandwidth for a given product |
| **Cost awareness** | Understand GPU costs, TCO, build vs buy |
| **Capacity planning** | Plan for peak traffic, headroom, multi-region |
| **Trade-off decisions** | Spot instances vs reserved, batch vs realtime |

**Interview reality**: "Design a recommendation system for 1B users" quickly turns into "How many servers? What's the monthly cost? How do you handle Black Friday traffic?"

---

## 🏗️ Cost Components of an ML System

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    ML SYSTEM COST ARCHITECTURE                                            │
│                                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  TRAINING (Batch / Periodic)                                                      │   │
│   │  • GPU cluster (A100, H100)          →  $2–5/hr per GPU (cloud)                  │   │
│   │  • Storage for datasets, checkpoints  →  $0.02–0.10/GB/mo                         │   │
│   │  • Data pipeline (Spark, etc.)       →  Compute + storage                        │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                                  │
│                                        ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  FEATURE STORE                                                                     │   │
│   │  • Compute (real-time + batch)        →  $/QPS                                    │   │
│   │  • Storage (features, embeddings)     →  $/GB                                    │   │
│   │  • Serving (low-latency reads)         →  Cache + DB costs                       │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                                  │
│                                        ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  INFERENCE (Real-time)                                                             │   │
│   │  • CPU/GPU servers                    →  $/instance/hr                            │   │
│   │  • Model serving (TensorFlow, etc.)   →  Throughput-dependent                     │   │
│   │  • Load balancer, networking          →  $/GB egress                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                           │
│   ADDITIONAL: Data transfer, human labeling, experiment infra, monitoring                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Typical Cost Split (Industry)

| Component | % of ML Budget (Typical) | Notes |
|-----------|--------------------------|-------|
| **Training** | 15–30% | Bursty; can use spot |
| **Inference** | 40–60% | Steady; must be always-on |
| **Storage** | 5–15% | Data lake, features, checkpoints |
| **Network** | 5–15% | Egress, cross-region |
| **Other** | 10–20% | Labeling, tooling, experiments |

---

## 📚 Table of Contents

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Back-of-Envelope Estimation](./01-back-of-envelope-estimation.md) | Powers of 2, latency numbers, QPS/storage/bandwidth estimation, worked examples (recommendation, ads, search, feature store), Python calculator |
| 2 | [Cost Modeling](./02-cost-modeling.md) | GPU costs (A100, H100), storage tiers, network, feature store, labeling, TCO, build vs buy, Python cost estimator |
| 3 | [Capacity Planning](./03-capacity-planning.md) | Peak vs average, headroom, autoscaling, multi-region, load testing, training capacity, disaster recovery, Python capacity planner |

---

## 🔑 Key Numbers Every ML Engineer Should Know

### Powers of 2

| Prefix | Bytes | Example |
|--------|-------|---------|
| 1 KB | 2¹⁰ | Small cache entry |
| 1 MB | 2²⁰ | Model weight (small) |
| 1 GB | 2³⁰ | Embedding table (1M × 256 float) |
| 1 TB | 2⁴⁰ | Day of events (10B × 100B) |
| 1 PB | 2⁵⁰ | Data lake (weeks/months) |

### Latency (Order of Magnitude)

| Operation | Latency |
|-----------|---------|
| L1 cache | ~1 ns |
| L2 cache | ~4 ns |
| RAM | ~100 ns |
| SSD | ~100 μs |
| HDD | ~10 ms |
| Same-region network | ~1 ms |
| Cross-continent | ~100 ms |

### GPU / Inference (2024–2025)

| GPU | FP16 TFLOPS | Memory | Cloud $/hr (approx) |
|-----|-------------|--------|---------------------|
| A100 40GB | 312 | 40 GB | ~$2.50–3.50 |
| A100 80GB | 312 | 80 GB | ~$3.50–4.50 |
| H100 | 989 | 80 GB | ~$3–4 (often supply constrained) |
| T4 | 65 | 16 GB | ~$0.50–0.70 |

---

## 🚨 Interview Scenarios

- **"Estimate the infrastructure for a recommendation system serving 1B users"** → QPS, storage, bandwidth; back-of-envelope.
- **"What's the monthly cost of running this model at 10M QPS?"** → Cost modeling; GPU vs CPU; spot vs on-demand.
- **"How do you plan capacity for Black Friday?"** → Peak vs average; headroom; autoscaling.
- **"Build vs buy for a feature store?"** → TCO, team size, timeline.

---

## 📋 Pre-Interview Checklist

- [ ] Powers of 2 (KB to PB) memorized
- [ ] Latency numbers (L1 to cross-continent)
- [ ] Can estimate QPS from DAU and actions per user
- [ ] Know GPU pricing (A100, H100 ballpark)
- [ ] Can explain headroom and autoscaling trade-offs
- [ ] Can discuss TCO and build vs buy

---

## 🚀 Next Steps

1. **[Back-of-Envelope Estimation](./01-back-of-envelope-estimation.md)** – Master the numbers and estimation methodology.
2. **[Cost Modeling](./02-cost-modeling.md)** – Build cost models for ML systems.
3. **[Capacity Planning](./03-capacity-planning.md)** – Plan for peak, scale, and disasters.

---

## Related Topics

| Topic | Link | Connection |
|-------|------|------------|
| Horizontal Scaling | [07-scalability-performance/01-horizontal-scaling](../../phase-3-operations-and-reliability/07-scalability-performance/01-horizontal-scaling.md) | Scaling strategies |
| Batch vs Realtime | [07-scalability-performance/03-batch-vs-realtime](../../phase-3-operations-and-reliability/07-scalability-performance/03-batch-vs-realtime.md) | Cost implications |
| Recommendation Systems | [10-end-to-end-systems/01-recommendation-systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) | Primary sizing use case |
| Feature Stores | [03-feature-engineering/01-feature-stores](../../phase-2-core-components/03-feature-engineering/01-feature-stores.md) | Storage and serving costs |

---

## Estimation Workflow (Interview)

1. **Clarify scope**: Users, use case, SLA
2. **Estimate QPS**: DAU × actions ÷ 86,400; apply peak factor
3. **Estimate storage**: Events, features, models
4. **Estimate compute**: QPS ÷ throughput per instance; add headroom
5. **Estimate cost**: Compute × $/hr × hours; storage × $/GB
6. **Sanity check**: Compare to known systems (e.g., "1B users ≈ Netflix scale")

---

## Common Estimation Mistakes

| Mistake | Fix |
|---------|-----|
| **Ignoring peak** | Always use 2–3× average for peak |
| **Forgetting headroom** | Add 20–50% for growth and spikes |
| **Using on-demand only** | Spot for training; reserved for steady inference |
| **Underestimating network** | Egress can be 10–20% of bill |
| **Single region** | Multi-region adds cost but needed for latency/HA |
