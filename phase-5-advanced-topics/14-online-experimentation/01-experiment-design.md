# Experiment Design

## Overview

A well-designed experiment produces **valid, actionable results**. Poor design leads to false positives (shipping bad changes), false negatives (killing good changes), or uninterpretable results. This guide covers hypothesis formulation, randomization units, sample size calculation, guardrail metrics, and treatment vs. control setup—with production-oriented examples and Python code.

---

## Hypothesis Formulation for ML Experiments

### Structure of a Good Hypothesis

A hypothesis should be **specific, measurable, and directional**:

```
If [we change X], then [metric Y] will [increase/decrease] by [Z%] because [mechanism].
```

| Element | Example (Recommendation) | Example (Search) |
|---------|-------------------------|------------------|
| **Change (X)** | New two-tower model with 256-dim embeddings | BERT reranker replacing hand-tuned features |
| **Metric (Y)** | Session watch time per user | Click-through rate on top 5 |
| **Direction** | Increase | Increase |
| **Magnitude (Z)** | ≥2% | ≥1.5% |
| **Mechanism** | Better personalization from richer embeddings | Better semantic matching |

### Null vs. Alternative Hypothesis

| Hypothesis | Definition | What we're testing |
|------------|------------|---------------------|
| **H₀ (Null)** | No difference between treatment and control | Default assumption |
| **H₁ (Alternative)** | Treatment differs from control by at least MDE | What we want to detect |

**Statistical framing**: We reject H₀ only when the observed difference is unlikely under H₀ (p < α). We want high power (probability of rejecting H₀ when H₁ is true).

---

## Randomization Units

The **randomization unit** is the entity to which we assign a variant. It must align with:
1. **Analysis unit**: We aggregate metrics per randomization unit
2. **Independence**: Units shouldn't influence each other's outcomes

### Unit Types and When to Use

| Unit | Use Case | Pros | Cons |
|------|----------|------|------|
| **User-level** | Recommendations, personalization, UI | Stable assignment; user sees consistent experience | Large sample needed; slow to collect |
| **Session-level** | Search, e-commerce checkout | Faster accumulation; lower correlation across sessions | User may see both variants; potential carryover |
| **Device-level** | Mobile apps, multi-device users | Clean for device-specific features | Same user or household may have multiple devices |
| **Page-level** | A/B test on a single page | Very fast; good for UI tweaks | Limited scope; can't measure downstream impact |
| **Request-level** | Latency, caching experiments | Millions of units quickly | High variance; often not independent |

### Production Examples

```
┌────────────────────────────────────────────────────────────────────────────┐
│  RANDOMIZATION UNIT DECISION TREE                                           │
│                                                                             │
│  Does the intervention affect user-level behavior (e.g., recommendations)?   │
│       YES ──▶ User-level                                                    │
│       NO                                                                     │
│           │                                                                 │
│  Does the intervention affect a single session (e.g., search results)?     │
│       YES ──▶ Session-level                                                 │
│       NO                                                                     │
│           │                                                                 │
│  Are there network effects (e.g., marketplace, social feed)?                │
│       YES ──▶ Cluster-level or Switchback (see 02-advanced-testing)        │
│       NO                                                                     │
│           │                                                                 │
│  Is it a UI/display change on one page?                                     │
│       YES ──▶ Page-level or Request-level                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

**Critical rule**: Randomization unit = analysis unit. If you assign at user level, compute metrics per user. Mixing causes inflated significance (pseudo-replication).

---

## Sample Size Calculation

### Inputs

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| **Significance level** | α | 0.05 | Probability of false positive (Type I error) |
| **Power** | 1 − β | 0.80 | Probability of detecting true effect (avoid Type II) |
| **Minimum detectable effect** | MDE | 1–5% | Smallest effect we care about (relative or absolute) |
| **Baseline rate** | p | From data | Control metric value (e.g., CTR = 0.02) |

### Formulas

**For binary metrics (CTR, CVR)** — two proportions:

$$
n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2}
$$

Where:
- $p_1$ = baseline (control) proportion
- $p_2$ = $p_1 \cdot (1 + \text{MDE})$ for relative MDE
- $z_{\alpha/2}$ ≈ 1.96 for α = 0.05 (two-tailed)
- $z_{\beta}$ ≈ 0.84 for power = 80%

**For continuous metrics (revenue, watch time)**:

$$
n = \frac{2 \cdot (z_{\alpha/2} + z_{\beta})^2 \cdot \sigma^2}{\delta^2}
$$

Where σ = standard deviation, δ = MDE (absolute).

### Rule of Thumb

For 80% power, 5% significance, detecting a **2% relative lift** on a binary metric with baseline ~2%:
- **~50,000 users per variant** (100K total for 50/50 split)

---

## Experiment Duration

### How Long to Run

| Factor | Recommendation |
|--------|----------------|
| **Minimum** | 2 full weeks (capture weekday + weekend) |
| **Calendar effects** | Avoid starting/ending near holidays, promotions |
| **Seasonality** | Run at least one full cycle (e.g., month for monthly patterns) |
| **SRM check** | Sample ratio mismatch (unequal traffic) suggests bugs—check early |

### Calendar Effects (Real Numbers)

- **Black Friday**: E-commerce CTR can 2–3x; experiments during this period are noisy
- **Post-holiday** (Jan 2–7): Drop in engagement; experiments may show false negatives
- **Weekly cycle**: B2B products vary Mon–Fri vs weekend

**Best practice**: Run experiments for a fixed, pre-specified duration. Don't extend because "we're almost significant."

---

## Guardrail Metrics

Guardrail metrics **must not regress**. If treatment wins on the primary metric but fails a guardrail, we don't ship.

| Category | Examples | Threshold (Typical) |
|----------|----------|---------------------|
| **Revenue** | Revenue per user, ARPU | No statistically significant decrease |
| **Latency** | P50, P99, P99.9 response time | < 10% increase |
| **Reliability** | Error rate, crash rate | No increase |
| **Quality** | Negative feedback, report rate | No increase |
| **Equity** | Per-segment parity | No disproportionate harm |

### Treatment vs. Control Setup

```
                    TRAFFIC SPLIT
                    
    ┌─────────────────────────────────────────────────────┐
    │  50% Control (Baseline)  │  50% Treatment (New)     │
    │  model_v1                │  model_v2                │
    │  current production      │  candidate for rollout  │
    └─────────────────────────────────────────────────────┘
    
    PRIMARY METRIC:     Treatment > Control (stat sig)
    GUARDRAIL METRICS:  Treatment ≥ Control (no stat sig degradation)
```

---

## Python Implementation: Power Analysis & Experiment Config

```python
"""
Power analysis and experiment configuration for ML A/B tests.
Production-oriented: handles binary and continuous metrics.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class PowerAnalysisResult:
    """Result of sample size calculation."""
    n_per_variant: int
    total_n: int
    power: float
    mde_relative: float
    baseline: float


def sample_size_binary(
    baseline_rate: float,
    mde_relative: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> PowerAnalysisResult:
    """
    Calculate sample size for binary metric (CTR, CVR).
    
    Args:
        baseline_rate: Control proportion (e.g., 0.02 for 2% CTR)
        mde_relative: Minimum detectable effect as relative lift (e.g., 0.02 = 2%)
        alpha: Significance level
        power: Statistical power (1 - beta)
        two_tailed: If True, two-tailed test
        
    Returns:
        PowerAnalysisResult with n_per_variant and total_n
        
    Example:
        >>> r = sample_size_binary(0.02, 0.02)
        >>> print(f"Need {r.n_per_variant} users per variant")
    """
    from scipy import stats
    
    p1 = baseline_rate
    p2 = p1 * (1 + mde_relative)
    
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_tailed else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
    effect_sq = (p2 - p1) ** 2
    
    n_per = ((z_alpha + z_beta) ** 2 * pooled_var) / effect_sq
    
    return PowerAnalysisResult(
        n_per_variant=int(math.ceil(n_per)),
        total_n=int(math.ceil(n_per * 2)),
        power=power,
        mde_relative=mde_relative,
        baseline=baseline_rate,
    )


def sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde_absolute: Optional[float] = None,
    mde_relative: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerAnalysisResult:
    """
    Calculate sample size for continuous metric (revenue, watch time).
    
    Args:
        baseline_mean: Control mean
        baseline_std: Control standard deviation
        mde_absolute: MDE in absolute units (use one of mde_absolute or mde_relative)
        mde_relative: MDE as relative (e.g., 0.02 = 2%)
    """
    from scipy import stats
    
    delta = mde_absolute if mde_absolute is not None else baseline_mean * (mde_relative or 0.02)
    
    z_alpha = stats.norm.ppf(0.975)
    z_beta = stats.norm.ppf(power)
    
    n_per = (2 * (z_alpha + z_beta) ** 2 * baseline_std ** 2) / (delta ** 2)
    
    return PowerAnalysisResult(
        n_per_variant=int(math.ceil(n_per)),
        total_n=int(math.ceil(n_per * 2)),
        power=power,
        mde_relative=delta / baseline_mean if baseline_mean else 0,
        baseline=baseline_mean,
    )


def experiment_duration_days(
    n_per_variant: int,
    daily_active_users: int,
    traffic_fraction: float = 0.5,
) -> int:
    """
    Estimate experiment duration in days.
    
    Args:
        n_per_variant: Required sample per variant
        daily_active_users: DAU in the experiment population
        traffic_fraction: Fraction of traffic to each variant (0.5 = 50%)
    
    Returns:
        Minimum number of days to reach required sample
    """
    users_per_variant_per_day = daily_active_users * traffic_fraction
    days = n_per_variant / users_per_variant_per_day
    return max(1, int(math.ceil(days)))


# --- Experiment Configuration ---

@dataclass
class ExperimentConfig:
    """Production experiment configuration."""
    experiment_id: str
    name: str
    variants: list[dict]  # [{"name": "control", "traffic": 0.5}, {"name": "treatment", "traffic": 0.5}]
    randomization_unit: str  # "user", "session", "page"
    primary_metric: str
    guardrail_metrics: list[str]
    min_sample_size_per_variant: int
    min_days: int = 14
    
    def to_assignment_config(self) -> dict:
        """Export for assignment service."""
        return {
            "experiment_id": self.experiment_id,
            "variants": [v["name"] for v in self.variants],
            "traffic": [v["traffic"] for v in self.variants],
            "unit": self.randomization_unit,
        }


# --- Example usage ---
if __name__ == "__main__":
    # Recommendation system: 2% CTR baseline, detect 2% lift
    result = sample_size_binary(baseline_rate=0.02, mde_relative=0.02)
    print(f"Binary metric: need {result.n_per_variant:,} per variant (total {result.total_n:,})")
    
    # Duration: 10M DAU, 50/50 split
    days = experiment_duration_days(result.n_per_variant, daily_active_users=10_000_000)
    print(f"At 10M DAU: ~{days} days minimum")
    
    # Continuous: $5 mean revenue, $20 std, detect 3% lift
    result2 = sample_size_continuous(5.0, 20.0, mde_relative=0.03)
    print(f"Continuous: need {result2.n_per_variant:,} per variant")
```

---

## Trade-offs and Interview Tips

### Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **User vs session randomization** | User: cleaner, slower. Session: faster, possible carryover. |
| **Larger MDE** | Smaller sample, shorter run—but may miss smaller real effects |
| **More guardrails** | Safer rollout—but more metrics to pass, higher chance of false guardrail triggers |
| **50/50 vs 90/10 split** | 50/50: faster. 90/10: lower risk if treatment is worse |

### Interview Tips

1. **Always state the hypothesis first** before diving into design.
2. **Justify the randomization unit**: "We use user-level because the model affects the whole session and we want consistent experience."
3. **Do back-of-envelope sample size**: "2% CTR, 2% MDE → ~50K per variant; with 1M DAU that's 50 days, so we'd run 2 weeks and maybe not have enough power—we might need to increase MDE or traffic."
4. **Name guardrails**: "We'd guard on revenue, P99 latency, and crash rate."
5. **Mention calendar effects**: "We'd avoid starting right before Black Friday."

---

## Quick Reference Table

| Metric Type | Key Formula | Example (2% MDE) |
|-------------|-------------|------------------|
| Binary (p≈2%) | n ≈ 2 × (1.96+0.84)² × 0.02×0.98 / 0.0004² | ~24,000 per variant |
| Binary (p≈50%) | n ≈ 2 × (1.96+0.84)² × 0.25 / 0.01² | ~3,800 per variant |
| Continuous | n ≈ 2 × (1.96+0.84)² × σ² / δ² | Depends on σ, δ |

---

## Appendix: Assignment Service Logic

Production assignment typically uses **consistent hashing**:

```python
def assign_variant(user_id: str, experiment_id: str, variants: list[str], traffic: list[float]) -> str:
    """
    Deterministic assignment: same user + experiment → same variant.
    Uses hash for reproducibility.
    """
    import hashlib
    key = f"{experiment_id}:{user_id}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    bucket = (h % 10000) / 10000.0  # 0 to 1
    
    cumulative = 0
    for v, t in zip(variants, traffic):
        cumulative += t
        if bucket < cumulative:
            return v
    return variants[-1]
```

**Interview note**: Explaining this shows you understand sticky assignment (users don't flip between variants) and scalability (stateless, no DB lookup per request).

---

## Pre-Experiment Checklist

- [ ] Hypothesis written (X → Y by Z%)
- [ ] Randomization unit chosen and justified
- [ ] Sample size calculated (power, MDE)
- [ ] Experiment duration estimated
- [ ] Primary and guardrail metrics defined
- [ ] Assignment logic implemented
- [ ] Event logging verified
