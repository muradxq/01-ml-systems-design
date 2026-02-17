# Metric Design for ML Experiments

## Overview

The right metrics determine whether we ship the right changes. Poor metric design leads to **Goodhart's law** ("When a measure becomes a target, it ceases to be a good measure")—optimizing engagement that hurts quality, or clicks that don't convert. This guide covers north star metrics, proxy metrics, counter metrics, OEC (Overall Evaluation Criterion), metric sensitivity (CUPED), and metric hierarchy.

---

## North Star Metrics

### What They Are

A **north star metric** is the single metric that best captures long-term product success. It should be:
- **Actionable**: Teams can influence it
- **Understandable**: Non-experts can grasp it
- **Leading indicator**: Predicts long-term health
- **Balanced**: Doesn't encourage gaming

### Examples by Product Type

| Product Type | North Star | Rationale |
|--------------|------------|-----------|
| **Netflix** | Streaming hours per user | Revenue and retention correlate; engagement is the driver |
| **YouTube** | Watch time | Ad revenue and satisfaction; aligns with quality |
| **Facebook/Meta** | DAU/MAU, time spent | Engagement drives ad revenue |
| **Uber** | Weekly active trips | Frequency = retention and revenue |
| **Spotify** | Listen time, retention | Predicts subscription and churn |
| **E-commerce** | Revenue, GMV | Direct business value |
| **Search** | Success rate, task completion | Quality of results |
| **Ads** | Revenue per impression (RPM) | Monetization efficiency |

### Why North Star Alone Isn't Enough

- **Too slow**: May take weeks or months to measure (e.g., retention at 30 days)
- **Too noisy**: Hard to detect 1–2% changes in short experiments
- **Too coarse**: Doesn't guide day-to-day decisions

We need **proxy metrics** for faster feedback.

---

## Proxy Metrics

### Definition

Proxy metrics are **faster-to-measure** substitutes that correlate with the north star. We optimize them during experiments and trust they track the north star.

| North Star | Proxy Metrics | Lag |
|------------|---------------|-----|
| 30-day retention | Day-1 retention, session count | Days vs weeks |
| Revenue | CTR, add-to-cart, conversion | Hours vs weeks |
| Watch time | CTR on recommendations, session length | Immediate |
| Task success | Click on top result, zero-result rate | Immediate |

### Proxy Quality

A good proxy has:
- **High correlation** with north star (typically > 0.5)
- **Sensitivity**: Detects changes faster
- **Causal relationship**: Change in proxy → change in north star (not just correlation)

```
                    PROXY METRIC QUALITY
                    
    North Star (e.g., 30-day retention)
            ▲
            │  Strong correlation
            │     ●●●●●
            │   ●      ●
            │  ●        ●
            │ ●          ●
            └────────────────────────▶ Proxy (e.g., D1 retention)
```

### Caution: Proxy Drift

Over time, optimizing a proxy can **decouple** it from the north star (Goodhart's law). Example: Optimizing "clicks" can lead to clickbait; optimizing "likes" to engagement hacks. **Counter metrics** guard against this.

---

## Counter Metrics

### Purpose

Counter metrics **prevent gaming** and **capture unintended harm**. When the primary metric improves, we check that counter metrics don't regress.

| Primary Metric | Counter Metrics | What They Guard |
|----------------|-----------------|-----------------|
| CTR | Revenue, dwell time | Clickbait |
| Watch time | Completion rate, dislikes | Padding, low quality |
| Engagement | Negative feedback, report rate | Addictive but harmful content |
| Clicks | Bounce rate, pages per session | Click-through without value |
| Conversion | Return rate, support tickets | Low-quality conversions |

### Goodhart's Law Illustrated

```
                    GOODHART'S LAW
                    
    Target: "Maximize clicks"
    
    Without counter metrics:
        ┌─────────────────────────────────────┐
        │  Clicks ↗  Revenue →  Satisfaction ↘ │
        │  (clickbait, low intent)             │
        └─────────────────────────────────────┘
    
    With counter metrics (revenue, dwell time):
        ┌─────────────────────────────────────┐
        │  Clicks ↗  Revenue must ≥ baseline  │
        │  Dwell time must ≥ baseline          │
        │  → Constrained optimization         │
        └─────────────────────────────────────┘
```

---

## Overall Evaluation Criterion (OEC)

### Definition

An **OEC** combines multiple metrics into a **single score** for experiment decision-making. Used at Google, Meta, and elsewhere.

### OEC Formula (Common Form)

$$
\text{OEC} = w_1 \cdot \text{norm}(M_1) + w_2 \cdot \text{norm}(M_2) + \cdots
$$

Where:
- $M_i$ = metric $i$ (e.g., clicks, revenue, latency)
- $\text{norm}(M_i)$ = normalized to [0, 1] or standardized
- $w_i$ = weight (from business priorities)

### Example: Recommendation OEC

| Metric | Weight | Normalization |
|--------|--------|---------------|
| Session watch time | 0.5 | Z-score or percentile |
| Click-through rate | 0.2 | Z-score |
| Diversity (catalog coverage) | 0.15 | Inverse of concentration |
| Latency (negative) | 0.15 | Penalty if P99 > threshold |

### Python: OEC Computation

```python
"""
Overall Evaluation Criterion (OEC) computation.
Combines multiple metrics into a single score for experiment comparison.
"""

from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np


@dataclass
class OECConfig:
    """OEC configuration with metric weights and normalizers."""
    metrics: List[str]
    weights: List[float]
    normalizers: Dict[str, Callable[[np.ndarray], np.ndarray]]
    
    def __post_init__(self):
        assert len(self.metrics) == len(self.weights)
        assert abs(sum(self.weights) - 1.0) < 1e-6


def z_score_normalize(x: np.ndarray) -> np.ndarray:
    """Standardize to mean=0, std=1; then scale to [0,1] for positive values."""
    if np.std(x) == 0:
        return np.ones_like(x) * 0.5
    z = (x - np.mean(x)) / np.std(x)
    # Map to [0, 1] using sigmoid-like transform
    return 1 / (1 + np.exp(-z))


def percentile_normalize(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalize x to percentile rank relative to reference distribution."""
    return np.array([np.mean(ref < xi) for xi in x])


def compute_oec(
    control_metrics: Dict[str, np.ndarray],
    treatment_metrics: Dict[str, np.ndarray],
    config: OECConfig,
) -> tuple[float, float]:
    """
    Compute OEC for control and treatment.
    
    Returns:
        (oec_control, oec_treatment)
    """
    oec_c = 0.0
    oec_t = 0.0
    
    for met, w in zip(config.metrics, config.weights):
        c_vals = control_metrics[met]
        t_vals = treatment_metrics[met]
        norm_fn = config.normalizers.get(met, z_score_normalize)
        
        # Use pooled data for normalization (fair comparison)
        pooled = np.concatenate([c_vals, t_vals])
        n_c = len(c_vals)
        
        c_norm = norm_fn(c_vals) if c_vals.size else 0
        t_norm = norm_fn(t_vals) if t_vals.size else 0
        
        if isinstance(c_norm, np.ndarray):
            oec_c += w * np.mean(c_norm)
            oec_t += w * np.mean(t_norm)
        else:
            oec_c += w * c_norm
            oec_t += w * t_norm
    
    return oec_c, oec_t


# Example usage
if __name__ == "__main__":
    config = OECConfig(
        metrics=["watch_time", "ctr", "diversity"],
        weights=[0.5, 0.3, 0.2],
        normalizers={
            "watch_time": z_score_normalize,
            "ctr": z_score_normalize,
            "diversity": z_score_normalize,
        },
    )
    
    np.random.seed(42)
    control = {
        "watch_time": np.random.exponential(5, 1000),
        "ctr": np.random.beta(2, 100, 1000),
        "diversity": np.random.uniform(0.3, 0.8, 1000),
    }
    treatment = {
        "watch_time": np.random.exponential(5.2, 1000),  # Slightly higher
        "ctr": np.random.beta(2.1, 100, 1000),
        "diversity": np.random.uniform(0.3, 0.8, 1000),
    }
    
    oec_c, oec_t = compute_oec(control, treatment, config)
    print(f"OEC Control: {oec_c:.4f}, Treatment: {oec_t:.4f}")
    print(f"Lift: {(oec_t - oec_c) / oec_c * 100:.2f}%")
```

---

## Metric Sensitivity: Variance Reduction

### Problem

Many north star and proxy metrics are **high variance** (e.g., revenue per user—few big spenders, many zeros). Detecting 1–2% effects requires huge sample sizes.

### CUPED (Controlled-experiment Using Pre-Experiment Data)

**Idea**: Use a **covariate** (e.g., pre-experiment value of the metric) to reduce variance. If users with high pre-experiment revenue tend to have high experiment revenue, we can adjust and shrink variance.

$$
Y_{i,\text{adj}} = Y_i - \theta (X_i - \mathbb{E}[X])
$$

Where:
- $Y_i$ = experiment metric for user $i$
- $X_i$ = pre-experiment covariate (e.g., revenue in prior period)
- $\theta$ = chosen to minimize $\text{Var}(Y_{i,\text{adj}})$

**Optimal** $\theta = \text{Cov}(Y,X) / \text{Var}(X)$. With this, variance reduction is $1 - \rho^2$ where $\rho$ is correlation between $Y$ and $X$.

**Typical variance reduction**: 20–50% for engagement/revenue metrics.

### Python: CUPED Variance Reduction

```python
"""
CUPED: Variance reduction for experiment metrics using pre-experiment data.
"""

import numpy as np
from typing import Tuple


def cuped_adjust(
    y: np.ndarray,
    x: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    CUPED adjustment: Y_adj = Y - theta * (X - E[X])
    
    Args:
        y: Experiment metric (e.g., revenue per user)
        x: Pre-experiment covariate (e.g., revenue in prior period)
    
    Returns:
        (y_adjusted, variance_reduction_ratio)
    """
    x_mean = np.mean(x)
    theta = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) > 0 else 0
    y_adj = y - theta * (x - x_mean)
    
    var_original = np.var(y)
    var_adjusted = np.var(y_adj)
    reduction = 1 - var_adjusted / var_original if var_original > 0 else 0
    
    return y_adj, reduction


def cuped_ttest(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,
    x_treatment: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample t-test with CUPED adjustment.
    
    Returns:
        (t_statistic, p_value)
    """
    y_c_adj, _ = cuped_adjust(y_control, x_control)
    y_t_adj, _ = cuped_adjust(y_treatment, x_treatment)
    
    n_c, n_t = len(y_c_adj), len(y_t_adj)
    mean_c, mean_t = np.mean(y_c_adj), np.mean(y_t_adj)
    var_c, var_t = np.var(y_c_adj, ddof=1), np.var(y_t_adj, ddof=1)
    
    pooled_std = np.sqrt(var_c / n_c + var_t / n_t)
    if pooled_std == 0:
        return 0.0, 1.0
    
    t = (mean_t - mean_c) / pooled_std
    df = n_c + n_t - 2
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, p


# Example
if __name__ == "__main__":
    np.random.seed(42)
    n = 10000
    # Simulate: experiment metric correlated with pre-experiment
    x_c = np.random.exponential(10, n)
    y_c = 0.7 * x_c + np.random.exponential(3, n)  # Treatment has 5% lift
    x_t = np.random.exponential(10, n)
    y_t = 0.7 * x_t * 1.05 + np.random.exponential(3, n)
    
    _, reduction = cuped_adjust(y_c, x_c)
    print(f"Variance reduction: {reduction*100:.1f}%")
    
    t, p = cuped_ttest(y_c, y_t, x_c, x_t)
    print(f"t={t:.3f}, p={p:.4f}")
```

---

## Metric Hierarchy

### Structure

```
                    METRIC HIERARCHY
                    
    Company North Star (e.g., Revenue, MAU)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Product 1    Product 2    Product 3
    (e.g., Feed) (e.g., Reels) (e.g., Marketplace)
        │           │           │
        ▼           ▼           ▼
    Team metrics (e.g., Feed ranking, Feed discovery)
        │
        ▼
    Experiment metrics (primary, guardrails, counter)
```

### Alignment

- **Experiment primary** should be a proxy for **team metric**
- **Team metric** should ladder up to **product** and **company** north star
- **Guardrails** and **counter metrics** protect company goals (revenue, trust, safety)

---

## Trade-offs and Interview Tips

### Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **Single north star** | Clear focus vs. incomplete picture |
| **Many proxy metrics** | Fast feedback vs. conflicting signals |
| **OEC** | Single decision vs. loss of nuance |
| **CUPED** | Sample size reduction vs. covariate availability |

### Interview Tips

1. **"What metric would you use for this recommendation system?"** → North star (watch time, retention) + proxy (CTR, session length) + counter (diversity, negative feedback).
2. **"How do we detect small effects with limited users?"** → CUPED; use pre-experiment engagement/revenue as covariate.
3. **"Our engagement is up but revenue is down"** → Counter metrics; engagement was a bad proxy; need revenue as guardrail.
4. **"How do we combine multiple metrics?"** → OEC with weights from product priorities; or guardrails with single primary.

---

## Metric Sensitivity: Additional Techniques

Beyond CUPED:

| Technique | Idea | When to Use |
|-----------|------|-------------|
| **CUPED** | Covariate adjustment | Pre-experiment data available |
| **Stratification** | Analyze within strata (e.g., by cohort) | Heterogeneous population |
| **Clustering** | Reduce effective n by clustering users | Network/spatial correlation |
| **Ratio metrics** | Y/X instead of Y (e.g., revenue per session) | X correlates with opportunity |

---

## Real-World Metric Examples

| Company | North Star | Proxy | Counter |
|---------|------------|-------|---------|
| **Netflix** | Streaming hours | CTR, session length | Completion rate, thumbs down |
| **YouTube** | Watch time | CTR, session length | Dislikes, reports |
| **Meta Feed** | Time spent, DAU | CTR, likes, comments | Report rate, well-being |
| **Google Search** | Task success | Clicks, queries per session | Zero-result rate |
| **Uber** | Trips per week | ETA accuracy, match rate | Cancellation, safety |