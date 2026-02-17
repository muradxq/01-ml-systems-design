# Advanced Testing: Bandits, Interleaving, Switchbacks

## Overview

Standard A/B tests work when you have **two variants** and **independent units**. But ML systems often need more: many model variants, ranking comparisons without full traffic split, marketplace experiments with network effects, or long-term impact measurement. This guide covers multi-armed bandits, interleaving, switchback experiments, cluster randomization, and holdout groups—with Python implementations.

---

## Multi-Armed Bandits (MABs)

### When MABs Beat A/B Tests

| Scenario | A/B Test | Multi-Armed Bandit |
|----------|----------|---------------------|
| **Variants** | 2 | 2–100+ |
| **Adaptation** | None until end | Continuously shift traffic to winners |
| **Regret** | Fixed allocation | Minimizes cumulative regret |
| **Use case** | Final validation | Exploration + exploitation |

**Key insight**: MABs reduce the **cost of experimentation** by giving more traffic to better-performing arms earlier. At Netflix, bandits are used for artwork selection (millions of combinations) and recommendation exploration.

### Epsilon-Greedy

```
With probability ε:  Explore (random arm)
With probability 1-ε: Exploit (best arm so far)
```

| ε | Trade-off |
|---|------------|
| 0.1 | 10% exploration, fast convergence |
| 0.05 | Less waste, slower to find winner |
| Decaying | Start high, decay over time |

### Upper Confidence Bound (UCB)

Choose arm that maximizes: **mean + bonus** (uncertainty)

$$
\text{UCB}(a) = \hat{\mu}_a + c \sqrt{\frac{\ln t}{n_a}}
$$

- $\hat{\mu}_a$ = observed mean for arm $a$
- $n_a$ = number of pulls for arm $a$
- $t$ = total pulls
- $c$ = exploration constant (e.g., 2)

**Property**: Automatically balances exploration (high uncertainty → high bonus) and exploitation (low uncertainty → rely on mean).

### Thompson Sampling

**Bayesian approach**: Maintain a posterior over each arm's reward; sample from posterior and pick the arm with highest sample.

For binary rewards (Bernoulli):
- Prior: Beta(α, β)
- Posterior after observations: Beta(α + successes, β + failures)
- Sample θ ~ Beta(α, β) for each arm; pick arm with max θ

**Why it works**: Naturally explores uncertain arms (wide posterior) and exploits confident winners (narrow posterior around high mean).

---

## Python: Thompson Sampling Implementation

```python
"""
Thompson Sampling for multi-armed bandit.
Suitable for binary rewards (click, conversion).
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Arm:
    """Bernoulli arm with Beta prior."""
    alpha: float  # successes + prior
    beta: float   # failures + prior
    name: str = ""


class ThompsonSamplingBandit:
    """Thompson Sampling MAB for binary rewards."""
    
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.arms = [
            Arm(alpha=prior_alpha, beta=prior_beta, name=f"arm_{i}")
            for i in range(n_arms)
        ]
        self.n_pulls = [0] * n_arms
    
    def select_arm(self) -> int:
        """Sample from each arm's posterior; return arm with highest sample."""
        samples = [
            np.random.beta(arm.alpha, arm.beta)
            for arm in self.arms
        ]
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float):
        """Update posterior with observed reward (0 or 1)."""
        reward = 1.0 if reward > 0.5 else 0.0
        self.arms[arm].alpha += reward
        self.arms[arm].beta += (1 - reward)
        self.n_pulls[arm] += 1
    
    def get_empirical_means(self) -> List[float]:
        """Current mean estimates (for comparison)."""
        return [
            (a.alpha - 1) / (a.alpha + a.beta - 2) if (a.alpha + a.beta) > 2 else 0.5
            for a in self.arms
        ]


def run_bandit_simulation(
    true_means: List[float],
    n_rounds: int = 10000,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> dict:
    """
    Simulate Thompson Sampling vs random baseline.
    
    Returns:
        regret (cumulative), arm selection counts
    """
    n_arms = len(true_means)
    optimal_reward = max(true_means)
    
    bandit = ThompsonSamplingBandit(n_arms, prior_alpha, prior_beta)
    total_reward = 0
    regret = []
    
    for t in range(n_rounds):
        arm = bandit.select_arm()
        reward = np.random.random() < true_means[arm]
        bandit.update(arm, reward)
        total_reward += reward
        regret.append((t + 1) * optimal_reward - total_reward)
    
    return {
        "cumulative_regret": regret[-1],
        "n_pulls": bandit.n_pulls,
        "empirical_means": bandit.get_empirical_means(),
        "total_reward": total_reward,
    }


# Example
if __name__ == "__main__":
    # 4 arms: best = 0.4, others 0.2, 0.25, 0.3
    result = run_bandit_simulation([0.2, 0.25, 0.3, 0.4], n_rounds=5000)
    print("Pulls per arm:", result["n_pulls"])
    print("Empirical means:", [f"{m:.3f}" for m in result["empirical_means"]])
    print("Cumulative regret:", result["cumulative_regret"])
```

---

## Interleaving Experiments

### Problem with A/B for Ranking

To compare two ranking algorithms with A/B: each user sees *either* algorithm A *or* B. We measure CTR. But:
- Different users have different intents
- Position bias: rank 1 gets more clicks than rank 5 regardless of relevance

**Interleaving** blends results from both rankers in a single list; we infer which ranker "won" based on which result was clicked.

### Team-Draft Interleaving

1. For each position, randomly choose team A or B
2. That team contributes its top unshown result
3. Track which team's results get clicked
4. Winner = team with more winning sessions (or higher CTR on their results)

```
Position  1  2  3  4  5  6  7  8
Team      A  B  A  B  B  A  A  B
Result    a1 b1 a2 b2 b3 a3 a4 b4
```

**Advantage**: Same user sees both rankers; position bias largely cancels. Much faster to get signal than full A/B.

### Python: Interleaving Evaluator

```python
"""
Team-draft interleaving for ranking comparison.
"""

from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class InterleavedResult:
    """Single result in interleaved list."""
    doc_id: str
    team: str  # "A" or "B"
    position: int


def team_draft_interleave(
    ranking_a: List[str],
    ranking_b: List[str],
    k: int = 10,
) -> List[InterleavedResult]:
    """
    Create interleaved list from two rankers.
    
    Args:
        ranking_a: Ordered doc IDs from ranker A
        ranking_b: Ordered doc IDs from ranker B
        k: Number of results to show
    """
    used_a, used_b = set(), set()
    result = []
    i_a, i_b = 0, 0
    
    while len(result) < k and (i_a < len(ranking_a) or i_b < len(ranking_b)):
        # Alternate which team picks first (fair)
        if random.random() < 0.5:
            # A picks first
            while i_a < len(ranking_a) and ranking_a[i_a] in used_a:
                i_a += 1
            if i_a < len(ranking_a):
                doc = ranking_a[i_a]
                used_a.add(doc)
                used_b.add(doc)  # Can't be used by B
                result.append(InterleavedResult(doc, "A", len(result) + 1))
                i_a += 1
            # Then B
            while i_b < len(ranking_b) and ranking_b[i_b] in used_b:
                i_b += 1
            if i_b < len(ranking_b) and len(result) < k:
                doc = ranking_b[i_b]
                used_a.add(doc)
                used_b.add(doc)
                result.append(InterleavedResult(doc, "B", len(result) + 1))
                i_b += 1
        else:
            # B picks first (symmetric)
            while i_b < len(ranking_b) and ranking_b[i_b] in used_b:
                i_b += 1
            if i_b < len(ranking_b):
                doc = ranking_b[i_b]
                used_a.add(doc)
                used_b.add(doc)
                result.append(InterleavedResult(doc, "B", len(result) + 1))
                i_b += 1
            while i_a < len(ranking_a) and ranking_a[i_a] in used_a:
                i_a += 1
            if i_a < len(ranking_a) and len(result) < k:
                doc = ranking_a[i_a]
                used_a.add(doc)
                used_b.add(doc)
                result.append(InterleavedResult(doc, "A", len(result) + 1))
                i_a += 1
    
    return result[:k]


def interleaving_winner(
    impressions: List[List[InterleavedResult]],
    clicks: List[List[str]],
) -> Tuple[float, float]:
    """
    Compute which team won based on clicks.
    
    Returns:
        (ctr_a, ctr_b) - click-through rate for each team's results
    """
    clicks_a, impressions_a = 0, 0
    clicks_b, impressions_b = 0, 0
    
    for imp, clk in zip(impressions, clicks):
        for r in imp:
            if r.team == "A":
                impressions_a += 1
                if r.doc_id in clk:
                    clicks_a += 1
            else:
                impressions_b += 1
                if r.doc_id in clk:
                    clicks_b += 1
    
    ctr_a = clicks_a / impressions_a if impressions_a else 0
    ctr_b = clicks_b / impressions_b if impressions_b else 0
    return ctr_a, ctr_b
```

---

## Switchback Experiments

### Problem: Network Effects

In marketplaces (Uber, Airbnb), social networks (Facebook), or two-sided platforms, **treatment and control interact**:
- More drivers in treatment → faster ETAs → more riders → more drivers (spillover to control)
- Feed algorithm change in treatment → users invite friends → control users affected

**Standard A/B** assumes independence. Violation inflates variance and can bias estimates.

### Switchback: Time-Based Randomization

Instead of assigning users to treatment/control, assign **time periods** or **clusters**:

| Design | How It Works | Use Case |
|--------|--------------|----------|
| **Time-based** | Odd hours = treatment, even = control (or days) | Marketplace, global effects |
| **Cluster-based** | Geographic clusters (e.g., cities) or subgraphs | Social network, regional marketplace |
| **Staggered** | Roll out by region over time | Phased launch |

```
TIME-BASED SWITCHBACK (e.g., Uber ETA)

Week 1:  Mon=T, Tue=C, Wed=T, Thu=C, Fri=T ...
Week 2:  Mon=C, Tue=T, Wed=C, Thu=T, Fri=C ...

Each user sees both treatment and control across days.
Network effects are diluted because whole system flips.
```

### Cluster-Based Randomization

```
CLUSTER RANDOMIZATION (e.g., Social Feed)

Cluster 1 (City A):  Control
Cluster 2 (City B):  Treatment
Cluster 3 (City C):  Control
...

Analysis: Compare clusters, not users. Fewer units → need larger clusters.
```

**Sample size**: With clusters, effective n = number of clusters. Need ~20+ clusters per variant for reasonable power.

---

## Holdout Groups for Long-Term Impact

### Problem: Short-Term Wins, Long-Term Losses

- **Engagement** might increase in week 1 (novelty) but decay (fatigue)
- **Recommendation** change might increase clicks but reduce diversity → long-term churn
- **Ads** might increase CTR but hurt brand perception

### Holdout Design

| Group | Traffic | Purpose |
|-------|---------|---------|
| **Control** | 10% | Baseline; never receive new model |
| **Treatment** | 90% | New model |
| **Long-term holdout** | 1–2% | Permanent control; measure 6–12 month impact |

```
                    HOLDOUT ARCHITECTURE
                    
    ┌─────────────────────────────────────────────────────────┐
    │  90% Treatment (new model)  │  10% Control (current)   │
    │  - Gets all improvements   │  - Temporary; for experiment
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │  1% Long-term Holdout (permanent control)               │
    │  - Never gets new models                                 │
    │  - Compare retention, revenue, satisfaction over 6–12mo │
    └─────────────────────────────────────────────────────────┘
```

**Netflix, Meta**: Run permanent holdouts to measure long-term impact of algorithm changes.

---

## Trade-offs and Interview Tips

### Trade-offs Summary

| Method | Pros | Cons |
|--------|------|------|
| **MAB** | Adapts quickly; less regret | Harder to do rigorous significance testing; can't easily analyze subgroups |
| **Interleaving** | Fast signal; same user sees both | Only for ranking; sensitive to interleaving algorithm |
| **Switchback** | Handles network effects | Lower power (fewer units); calendar confounds |
| **Holdout** | Captures long-term effects | Slow; 1% holdout has high variance |

### Interview Tips

1. **"We have 20 model variants"** → MAB (Thompson Sampling) to explore and converge to best, then A/B finalist vs baseline.
2. **"How do we compare two rankers quickly?"** → Interleaving; same user, position bias mitigated.
3. **"Our marketplace has network effects"** → Switchback (time or cluster); explain why user-level A/B is problematic.
4. **"We're worried about long-term engagement"** → Holdout group; 1% permanent control for retention/revenue over 6+ months.

---

## Comparison Table: When to Use What

| Scenario | Best Method | Fallback |
|----------|-------------|----------|
| 2 variants, independent units | A/B test | — |
| 5+ variants, want to converge | MAB (Thompson) | Phased A/B |
| Ranking comparison | Interleaving | A/B with enough users |
| Marketplace / network effects | Switchback | Cluster A/B |
| Long-term impact | Holdout | Cohort analysis |
| Many segments to analyze | A/B (MAB harder) | MAB + final A/B |

---

## UCB Implementation (Alternative to Thompson)

```python
import numpy as np

class UCBBandit:
    """Upper Confidence Bound bandit."""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.means = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.t = 0
    
    def select_arm(self) -> int:
        # Ensure each arm pulled at least once
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        ucb = self.means + self.c * np.sqrt(np.log(self.t + 1) / self.counts)
        return int(np.argmax(ucb))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] = self.means[arm] + (reward - self.means[arm]) / n
        self.t += 1
```

---

## Epsilon-Greedy with Decay

```python
class EpsilonGreedyBandit:
    """Epsilon-greedy with optional decay."""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: float = 0.9999):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.decay = decay
        self.means = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
    
    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.means))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] = self.means[arm] + (reward - self.means[arm]) / n
        self.epsilon *= self.decay
```

---

## Real-World Scale Examples

| Company | Method | Scale |
|---------|--------|-------|
| **Netflix** | MAB for artwork; A/B for algo | Billions of impressions |
| **Google** | Interleaving for search | Every search result |
| **Uber** | Switchback for ETA, pricing | City-level clusters |
| **Meta** | Holdout for feed, ads | 1% permanent holdout |
| **LinkedIn** | Cluster randomization | Member graph clusters |
