# Analysis Pitfalls in A/B Testing

## Overview

Even well-designed experiments can produce **misleading results** if analysis goes wrong. Simpson's paradox, novelty effects, multiple testing, peeking, network effects, survivorship bias, and confusion between short- and long-term impact are common traps. This guide covers each with worked examples and Python code for correct approaches.

---

## Simpson's Paradox

### Definition

**Simpson's paradox**: A trend appears in each subgroup but **reverses** when groups are combined. Caused by confounding variable (often segment or cohort) that differs between treatment and control.

### Worked Example

| Segment | Control CTR | Treatment CTR | Control N | Treatment N |
|---------|-------------|---------------|-----------|-------------|
| Mobile | 2% | 2.5% | 8,000 | 2,000 |
| Desktop | 4% | 4.2% | 2,000 | 8,000 |

**Combined (naive)**:
- Control: (0.02 × 8000 + 0.04 × 2000) / 10000 = **2.4%**
- Treatment: (0.025 × 2000 + 0.042 × 8000) / 10000 = **3.86%**

Treatment wins overall. But **within each segment**, treatment wins by the same relative amount. The paradox: treatment got more desktop traffic (higher baseline CTR), so the overall average is skewed.

**Correct analysis**: Stratify by segment. Report segment-level effects. If effects are consistent, report weighted average or run segment-specific experiments.

### Python: Simpson's Paradox Detection

```python
"""
Simpson's paradox: aggregate vs stratified analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def simpsons_paradox_check(
    df: pd.DataFrame,
    segment_col: str,
    variant_col: str,
    metric_col: str,
) -> Tuple[dict, bool]:
    """
    Check for Simpson's paradox: aggregate and stratified results.
    
    Returns:
        (results_dict, paradox_detected)
    """
    results = {}
    
    # Aggregate
    agg = df.groupby(variant_col)[metric_col].agg(["mean", "count"])
    results["aggregate"] = agg.to_dict()
    agg_control = agg.loc[df[variant_col].unique()[0], "mean"]
    agg_treat = agg.loc[df[variant_col].unique()[1], "mean"]
    agg_lift = (agg_treat - agg_control) / agg_control if agg_control else 0
    
    # Stratified
    stratified = df.groupby([segment_col, variant_col])[metric_col].agg(["mean", "count"]).reset_index()
    segments = df[segment_col].unique()
    stratified_lifts = []
    
    for seg in segments:
        seg_df = stratified[(stratified[segment_col] == seg)]
        if len(seg_df) == 2:
            c = seg_df[seg_df[variant_col] == df[variant_col].unique()[0]]["mean"].values[0]
            t = seg_df[seg_df[variant_col] == df[variant_col].unique()[1]]["mean"].values[0]
            lift = (t - c) / c if c else 0
            stratified_lifts.append((seg, lift))
    
    results["stratified"] = stratified_lifts
    
    # Paradox: aggregate direction differs from some stratum
    paradox = False
    for seg, lift in stratified_lifts:
        if (agg_lift > 0 and lift < -0.01) or (agg_lift < 0 and lift > 0.01):
            paradox = True
            break
    
    return results, paradox


# Worked example
if __name__ == "__main__":
    data = []
    # Mobile: 8000 control, 2000 treatment
    data.extend([{"segment": "mobile", "variant": "control", "ctr": 0.02} for _ in range(8000)])
    data.extend([{"segment": "mobile", "variant": "treatment", "ctr": 0.025} for _ in range(2000)])
    # Desktop: 2000 control, 8000 treatment
    data.extend([{"segment": "desktop", "variant": "control", "ctr": 0.04} for _ in range(2000)])
    data.extend([{"segment": "desktop", "variant": "treatment", "ctr": 0.042} for _ in range(8000)])
    
    df = pd.DataFrame(data)
    # Add some noise
    df["ctr"] = df["ctr"] + np.random.normal(0, 0.001, len(df))
    
    res, paradox = simpsons_paradox_check(df, "segment", "variant", "ctr")
    print("Paradox detected:", paradox)
```

---

## Novelty and Primacy Effects

### Novelty Effect

Users react positively to **something new** temporarily. After a few days or weeks, the effect decays.

### Primacy Effect

Users react to **first impression** (e.g., new UI). Early sessions bias the result.

### Mitigation

| Approach | How |
|----------|-----|
| **Exclude early data** | Drop first 3–7 days from analysis |
| **Pre-specify analysis window** | e.g., days 7–21 only |
| **Run longer** | 4+ weeks to dilute early effects |
| **Holdout for long-term** | Measure 30–90 day retention separately |

**Typical numbers**: Novelty can inflate treatment effect by 20–50% in first week.

---

## Multiple Testing Correction

### Problem

With **m** metrics (or segments), each tested at α = 0.05, the probability of at least one false positive is:
$$P(\text{at least one FP}) = 1 - (1 - 0.05)^m$$

For m = 10: ~40% chance of false positive.

### Bonferroni Correction

Use α′ = α / m for each test. Guarantees family-wise error rate ≤ α.

**Cost**: Very conservative; hard to reject. For m = 20, α′ = 0.0025.

### Benjamini-Hochberg (FDR)

Controls **False Discovery Rate**: among rejected hypotheses, expected proportion of false positives.

1. Order p-values: p₁ ≤ p₂ ≤ … ≤ pₘ
2. Find largest k such that pₖ ≤ (k/m) × α
3. Reject H₁, …, Hₖ

**Less conservative** than Bonferroni; appropriate when you're exploring many metrics and accept some false discoveries.

### Python: Multiple Testing Correction

```python
"""
Multiple testing correction: Bonferroni and Benjamini-Hochberg.
"""

import numpy as np
from typing import List, Tuple


def bonferroni_correct(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Reject hypotheses where p < alpha / m."""
    m = len(p_values)
    threshold = alpha / m
    return [p < threshold for p in p_values]


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Benjamini-Hochberg FDR correction.
    
    Returns:
        (reject_list, adjusted_p_values)
    """
    p = np.array(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    
    # BH critical values
    crit = (np.arange(1, m + 1) / m) * alpha
    reject_sorted = sorted_p <= crit
    
    # Find largest k
    k = np.where(reject_sorted)[0]
    k_max = k[-1] + 1 if len(k) > 0 else 0
    
    reject = np.zeros(m, dtype=bool)
    reject[order[:k_max]] = True
    
    # Adjusted p-values (conservative)
    adjusted = np.minimum.accumulate(np.minimum(1, sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    inv_order = np.argsort(order)
    adjusted_orig = adjusted[inv_order]
    
    return list(reject), list(adjusted_orig)


# Example
if __name__ == "__main__":
    # Simulate 10 p-values (some true nulls, some real effects)
    np.random.seed(42)
    p_vals = [
        0.001, 0.03, 0.05, 0.12, 0.20,
        0.001, 0.04, 0.15, 0.50, 0.80,
    ]
    
    rej_bf = bonferroni_correct(p_vals)
    rej_bh, adj = benjamini_hochberg(p_vals)
    
    print("Bonferroni rejects:", sum(rej_bf))
    print("BH rejects:", sum(rej_bh))
    print("BH adjusted p-values:", [f"{p:.3f}" for p in adj])
```

---

## The Peeking Problem

### What It Is

**Peeking**: Checking experiment results before planned end and stopping when significant. This **inflates Type I error** dramatically.

### Why It's Bad

If you peek every day and stop when p < 0.05, the probability of eventual false positive can exceed 20–30% (depending on sample size and peeking schedule) instead of 5%.

### Solutions

| Approach | How | When to Use |
|----------|-----|--------------|
| **Fixed sample size** | Pre-specify n, don't look until reached | Standard practice |
| **Sequential testing** | Use stopping boundaries (e.g., O'Brien-Fleming) | Need early stopping |
| **Hold analysis** | Lock analysis plan; analyze only at end | Regulatory, high-stakes |
| **Bonferroni over peeks** | α / (# of peeks) | Ad-hoc |

### Python: Sequential Testing (Simplified)

```python
"""
Sequential probability ratio test (SPRT) for early stopping.
Simplified: use alpha-spending (O'Brien-Fleming style).
"""

import numpy as np
from scipy import stats
from typing import List, Tuple


def obrien_fleming_boundaries(
    n_analyses: int,
    alpha: float = 0.05,
) -> List[float]:
    """
    O'Brien-Fleming alpha-spending: stricter early, looser late.
    Returns z-critical values for each look.
    """
    # Simplified: linear alpha spending
    # Actual OB-F uses specific formula
    alphas = [alpha * (i + 1) / n_analyses for i in range(n_analyses)]
    z_crit = [stats.norm.ppf(1 - a / 2) for a in alphas]
    return z_crit


def sequential_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    batch_size: int = 1000,
) -> Tuple[bool, int]:
    """
    Sequential t-test: stop when significant or data exhausted.
    Uses Bonferroni over batches for simplicity.
    
    Returns:
        (rejected, batch_at_stop)
    """
    n = min(len(control), len(treatment))
    n_batches = n // batch_size
    alpha_per_batch = 0.05 / n_batches if n_batches > 0 else 0.05
    
    for i in range(n_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        c_batch = control[start:end]
        t_batch = treatment[start:end]
        t_stat, p_val = stats.ttest_ind(c_batch, t_batch)
        if p_val < alpha_per_batch:
            return True, i + 1
    
    return False, n_batches
```

---

## Network Effects and Interference

### Problem

When **treatment and control users interact**, independence is violated:
- **Social**: Treatment user's feed change affects control user (viral content)
- **Marketplace**: More treatment drivers → faster pickups for everyone
- **Two-sided**: Treatment advertisers get more impressions → control advertisers get fewer

### Impact

- **Bias**: Treatment effect can be over- or underestimated
- **Variance**: Often inflated (effective n smaller)

### Solutions

- **Switchback** (time or cluster randomization)
- **Cluster randomization**: Assign clusters (cities, subgraphs)
- **Reduce spillover**: Limit interaction (e.g., separate markets)

---

## Survivorship Bias

### Definition

**Survivorship bias**: Analyzing only units that "survived" (e.g., users who returned, sessions that didn't error). This can skew results if survival differs between treatment and control.

### Example

- **Control**: 1000 users, 500 churned. Analyze 500. Revenue = $10/user.
- **Treatment**: 1000 users, 700 churned (worse retention). Analyze 300. Revenue = $12/user.

Naive: Treatment wins ($12 > $10). Correct: Treatment has *worse* retention; we're comparing survivors who are a different (perhaps higher-value) subset.

### Mitigation

- **Intent-to-treat (ITT)**: Analyze all assigned users, including churned. Use revenue per *assigned* user, not per *active* user.
- **Include churned in metric**: e.g., revenue per assigned user = 0 for churned.

---

## Long-Term vs Short-Term Effects

### Tension

| Horizon | Typical Result | Risk |
|---------|----------------|------|
| **Short (1–2 weeks)** | Fast to measure | Novelty effect; misses decay |
| **Long (3–12 months)** | True impact | Slow; high variance; confounded by other changes |

### Holdout for Long-Term

Permanent holdout (1–2% of users) never receiving new models. Compare:
- **Short-term**: Treatment vs control (temporary)
- **Long-term**: Ever-upgraded vs permanent holdout

---

## Trade-offs and Interview Tips

### Trade-offs Summary

| Pitfall | Mitigation | Cost |
|---------|------------|------|
| Simpson's | Stratify | More segments to analyze |
| Novelty | Exclude early data | Lose sample; longer runs |
| Multiple testing | Bonferroni/BH | Lower power |
| Peeking | Fixed sample / sequential | Can't stop early |
| Network effects | Switchback | Lower power; complexity |

### Interview Tips

1. **"Results differ by segment"** → Simpson's paradox; stratify and report per segment.
2. **"We checked results daily and stopped when significant"** → Peeking problem; inflation of false positives.
3. **"We have 50 metrics"** → Multiple testing; use FDR (BH) or pre-specify primary.
4. **"Treatment wins at 1 week but not at 4 weeks"** → Novelty effect; exclude early or run longer.

---

## Appendix: Common p-Value Mistakes

### Interpreting p-Values

| p-value | Meaning | Common Misinterpretation |
|---------|---------|--------------------------|
| p = 0.03 | Under H₀, 3% chance of observing this or more extreme | "97% chance treatment works" (wrong) |
| p = 0.05 | Threshold for significance | "Effect is real" (maybe; need replication) |
| p > 0.05 | Not significant | "No effect" (wrong; could be underpowered) |

### Correct Interpretation

- **p < α**: Reject H₀; evidence against null. Does NOT mean effect size is large.
- **p ≥ α**: Fail to reject; inconclusive. Does NOT prove null is true.