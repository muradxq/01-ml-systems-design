# Capacity Planning for ML Systems

## Overview

Capacity planning ensures ML systems can handle **peak load** without outage or degradation. Traffic is rarely flat—diurnal patterns, weekly cycles, seasonal spikes, and Black Friday events require headroom, autoscaling, and multi-region strategies. This guide covers peak vs average traffic, headroom, autoscaling, multi-region distribution, load testing, training capacity, and disaster recovery.

---

## Peak vs Average Traffic Patterns

### Diurnal (Daily) Pattern

Traffic typically peaks in evening (local time):

```
QPS
  ▲
  │        ●●●
  │      ●     ●
  │    ●         ●
  │  ●             ●
  │ ●               ●
  └──────────────────────▶ Time (24h)
     00  06  12  18  24
```

- **Peak / Average** ≈ 2–3× for global products
- **Peak / Average** ≈ 1.5–2× for single-region

### Weekly Pattern

- **B2C**: Weekends often 20–30% higher
- **B2B**: Weekdays dominate; weekends low

### Seasonal

| Event | Typical Lift | Planning |
|-------|--------------|----------|
| Black Friday | 3–5× average | Plan 2–3 months ahead |
| Product launch | 2–10× | Depends on marketing |
| Back to school | 1.5–2× | Moderate headroom |
| Super Bowl (ad-heavy) | 10×+ | Special planning |

---

## Headroom Planning

### What Is Headroom?

**Headroom** = spare capacity beyond expected peak. Buffers for:
- Traffic spikes
- Failures (other regions down)
- Growth before next scaling cycle

### Typical Headroom

| Scenario | Headroom | Rationale |
|----------|----------|-----------|
| **Stable product** | 20–30% | Normal variation |
| **Growing product** | 50–100% | Growth + spikes |
| **Critical (e.g., payments)** | 100–200% | Failover, safety |

### Formula

$$
\text{Provisioned capacity} = \text{Peak QPS} \times (1 + \text{headroom fraction})
$$

Example: Peak 100K QPS, 50% headroom → provision for 150K QPS.

---

## Autoscaling Strategies for ML Workloads

### What to Scale On

| Signal | Pros | Cons |
|--------|------|------|
| **QPS** | Direct demand | Lag during sudden spike |
| **Latency (P99)** | Quality SLO | Reactive |
| **CPU/GPU utilization** | Resource-based | May miss queue buildup |
| **Queue depth** | Prevents overload | Needs queue in front |

### Scaling Policies

| Policy | Use Case |
|--------|----------|
| **Scale up fast, down slow** | Avoid overload; don't thrash |
| **Step scaling** | Add N instances when threshold breached |
| **Target tracking** | Maintain target utilization (e.g., 70% CPU) |

### ML-Specific Considerations

- **Cold start**: Model loading takes 30–60+ seconds; scale proactively
- **GPU**: Scaling GPU nodes is slower than CPU; need longer lead time
- **Batch jobs**: Scale training cluster independently; use spot

---

## Multi-Region Capacity Distribution

### Why Multi-Region?

- **Latency**: Users served from nearest region
- **Resilience**: One region down → others absorb
- **Compliance**: Data residency (GDPR, etc.)

### Capacity Distribution

| Strategy | How | Use Case |
|----------|-----|----------|
| **Proportional to traffic** | Each region: capacity ∝ local traffic | Latency-sensitive |
| **Active-active** | All regions serve traffic | High availability |
| **Active-passive** | Primary + DR; failover on failure | Cost optimization |

### Failover Planning

- **RTO** (Recovery Time Objective): How fast to recover (e.g., 1 hour)
- **RPO** (Recovery Point Objective): How much data loss acceptable (e.g., 0)
- Capacity in DR region: Typically 50–100% of primary to absorb failover traffic

---

## Load Testing ML Systems

### Approaches

| Approach | How | When |
|----------|-----|------|
| **Synthetic load** | Generate requests from test harness | Pre-launch |
| **Shadow traffic** | Replay production traffic to new model | Canary |
| **Traffic mirroring** | Duplicate % of prod to test cluster | Staging validation |
| **Chaos** | Kill nodes, inject latency | Resilience testing |

### Load Test Metrics

- **Throughput**: Max QPS before latency degrades
- **Latency**: P50, P95, P99 at various load levels
- **Error rate**: 4xx, 5xx as load increases
- **Saturation point**: When system collapses

---

## Capacity Planning for Model Training

### GPU Cluster Sizing

$$
\text{GPUs needed} = \frac{\text{Training time (hours)} \times \text{Target wall time (hours)}}{\text{Target wall time (hours)}}
$$

Or: based on model size and target throughput (samples/sec).

### Considerations

| Factor | Impact |
|--------|--------|
| **Checkpointing** | Enables spot; adds storage |
| **Data pipeline** | Must feed GPUs; storage/network bandwidth |
| **Communication** | All-reduce across nodes; NVLink vs network |
| **Spot** | 60–90% cheaper; need checkpoint/restart |

### Example

- 70B model, 1T tokens, 8K H100s → ~2 weeks (order of magnitude)
- 8K × 336 hr × $4 ≈ **$10M+** (simplified)

---

## Disaster Recovery Capacity

### DR Capacity

- **Hot standby**: Full capacity in DR region (2× cost)
- **Warm standby**: 50% capacity; scale up on failover
- **Cold standby**: Minimal; provision on demand (slow)

### ML-Specific DR

- **Model artifacts**: Replicated to DR region (storage)
- **Feature store**: Replicate or rebuild from source
- **Inference**: Need GPU capacity in DR

---

## Python: Capacity Planner and Auto-Scaling Simulator

```python
"""
Capacity planner and autoscaling simulator.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class CapacityPlan:
    """Capacity planning result."""
    peak_qps: float
    headroom: float
    provisioned_qps: float
    instances: int
    qps_per_instance: float


def plan_capacity(
    avg_qps: float,
    peak_to_avg: float = 2.5,
    headroom: float = 0.5,
    qps_per_instance: float = 2000,
) -> CapacityPlan:
    """Compute capacity plan."""
    peak_qps = avg_qps * peak_to_avg
    provisioned_qps = peak_qps * (1 + headroom)
    instances = int(np.ceil(provisioned_qps / qps_per_instance))
    return CapacityPlan(
        peak_qps=peak_qps,
        headroom=headroom,
        provisioned_qps=provisioned_qps,
        instances=instances,
        qps_per_instance=qps_per_instance,
    )


def simulate_autoscale(
    qps_trace: List[float],
    initial_instances: int,
    qps_per_instance: float = 2000,
    scale_up_threshold: float = 0.8,
    scale_down_threshold: float = 0.3,
    min_instances: int = 1,
    max_instances: int = 100,
) -> List[int]:
    """
    Simulate autoscaling decisions over a QPS trace.
    Uses utilization-based scaling.
    """
    instances = initial_instances
    history = [instances]
    
    for qps in qps_trace[1:]:
        utilization = qps / (instances * qps_per_instance) if instances > 0 else 1
        
        if utilization > scale_up_threshold:
            instances = min(
                max_instances,
                int(np.ceil(qps / (qps_per_instance * scale_up_threshold)))
            )
        elif utilization < scale_down_threshold:
            instances = max(
                min_instances,
                int(np.ceil(qps / (qps_per_instance * scale_down_threshold)))
            )
        
        history.append(instances)
    
    return history


# Example
if __name__ == "__main__":
    plan = plan_capacity(avg_qps=30_000, peak_to_avg=2.5, headroom=0.5)
    print(f"Peak QPS: {plan.peak_qps:,.0f}")
    print(f"Instances: {plan.instances}")
    
    # Simulate diurnal pattern
    t = np.linspace(0, 24, 240)
    qps_trace = 30_000 * (1 + 0.8 * np.sin(2 * np.pi * (t - 6) / 24))
    qps_trace = np.maximum(qps_trace, 5_000).tolist()
    
    inst_history = simulate_autoscale(
        qps_trace, initial_instances=25,
        scale_up_threshold=0.8, scale_down_threshold=0.3,
    )
    print(f"Instance range: {min(inst_history)} - {max(inst_history)}")
```

---

## Trade-offs and Interview Tips

### Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **Headroom** | More = safer, more cost |
| **Autoscale** | Elastic vs cold start lag |
| **Multi-region** | Resilience vs complexity and cost |
| **Spot for training** | Cost vs preemption risk |

### Interview Tips

1. **"How do you plan for Black Friday?"** → 3–5× normal peak; plan 2–3 months ahead; load test; 50–100% headroom.
2. **"When do you scale?"** → On QPS, latency, utilization; scale up fast, down slow.
3. **"How many regions?"** → Latency + resilience; typically 3+ for global.
4. **"DR for ML?"** → Model replication; feature store failover; GPU capacity in DR.

---

## Capacity Planning Timeline

| Horizon | Activities |
|---------|-------------|
| **Weekly** | Monitor utilization; trigger scale-up if approaching limit |
| **Monthly** | Review trends; adjust headroom |
| **Quarterly** | Re-forecast; plan for seasonal events |
| **Annually** | TCO review; reserved capacity decisions |

---

## Quick Reference: Headroom and Scaling

| Scenario | Headroom | Scale-Up Speed |
|----------|----------|----------------|
| Stable B2B | 20–30% | Minutes |
| Growing B2C | 50–100% | Minutes |
| Black Friday | 100–200% | Weeks ahead |
| New launch | 200%+ | Provision before launch |

---

## Autoscaling Configuration Example

```yaml
# Example: Kubernetes HPA for ML inference
minReplicas: 5
maxReplicas: 100
targetCPUUtilization: 70%
scaleUp:
  stabilizationWindow: 60s
  policies:
    - type: Percent
      value: 100
      periodSeconds: 60
scaleDown:
  stabilizationWindow: 300s   # Scale down slowly
  policies:
    - type: Percent
      value: 10
      periodSeconds: 60
```

---

## Multi-Region Failover Flow

```
Primary Region (us-east-1)
    │
    │  Health check fails
    ▼
Route 53 / Global LB detects failure
    │
    ▼
Traffic shifted to DR (eu-west-1)
    │
    ▼
DR region scales if warm standby (50% → 100%)
```

---

## Training Capacity: Data Parallelism Sizing

| Model Size | GPUs (A100) | Training Time (1 epoch, 1B samples) |
|------------|-------------|-------------------------------------|
| 100M params | 1–4 | Hours |
| 1B params | 8–32 | Hours–days |
| 7B params | 64–256 | Days |
| 70B params | 256–1024 | Weeks |

---

## Load Test Checklist

- [ ] Define target QPS and latency SLO
- [ ] Ramp up gradually (find breaking point)
- [ ] Test with production-like data
- [ ] Include cache cold start
- [ ] Test failover (kill nodes)