# 🧪 Online Experimentation in ML Systems

## Overview

Online experimentation is the **gatekeeper of production ML**. At Meta, Google, Uber, Netflix, and other tech companies, no ML change ships without passing controlled experiments. Offline metrics (AUC, NDCG, RMSE) often **fail to correlate** with real-world impact—the only way to know if a model improves the product is to run an A/B test.

This section covers the complete experimentation lifecycle: designing valid experiments, choosing randomization units, calculating sample sizes, defining metrics, and avoiding the many pitfalls that inflate false positives or hide real effects. You'll learn production patterns used by companies running millions of experiments per year.

---

## 🎯 Why Experimentation Is Critical for ML

### The Offline vs. Online Gap

ML engineers optimize offline metrics during development. But the relationship between offline and online metrics is often **weak or inverted**:

| Offline Metric | Online Metric | Typical Correlation | Real-World Example |
|----------------|---------------|---------------------|--------------------|
| **AUC** | CTR, CVR | 0.3–0.6 | Netflix: Better AUC doesn't always mean more streaming hours |
| **NDCG** | User satisfaction | 0.2–0.5 | Search: Ranking by relevance doesn't capture diversity preferences |
| **RMSE** | Engagement | Often negative | Recommendations: Lower RMSE can mean overfitting to past behavior |
| **F1** | Revenue | Unpredictable | Ads: Precision/recall trade-off differs by user segment |

**Why the gap exists:**
1. **Feedback loops**: The model shapes user behavior; offline data reflects past model outputs
2. **Metric mismatch**: We optimize proxy metrics (clicks) that don't align with true goals (satisfaction, revenue)
3. **Distribution shift**: Production traffic differs from training data
4. **Indirect effects**: Engagement up might mean more addictive behavior, not better quality

### Industry Reality: Experiment-First Culture

- **Meta (Facebook)**: 10,000+ A/B tests per year; every feature change is experimented
- **Google**: Experimentation platform runs billions of assignments; 1% of experiments show statistically significant positive impact
- **Netflix**: Every recommendation algorithm change goes through A/B tests; 90% of tested ideas fail
- **Uber**: ETA, pricing, matching—all experimented before rollout

**Interview insight**: When asked "How would you validate this ML system?", the answer always starts with **online experimentation**. Offline metrics inform what to test; online experiments decide what to ship.

---

## 🏗️ Architecture of an Experimentation Platform

A production experimentation platform integrates with the ML serving stack at multiple points:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTATION PLATFORM ARCHITECTURE                                           │
│                                                                                                   │
│  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐ │
│  │   Client/Edge   │     │  Assignment      │     │   ML Serving     │     │   Event         │ │
│  │   (App/Web)     │────▶│  Service         │────▶│   Infrastructure │────▶│   Pipeline      │ │
│  │                 │     │                  │     │                  │     │                 │ │
│  │ • User ID       │     │ • Hash(user_id + │     │ • Feature store  │     │ • Click/impress │ │
│  │ • Request ctx   │     │   exp_id) → var  │     │ • Model A/B      │     │ • Conversion    │ │
│  └─────────────────┘     │ • Sticky assign │     │ • Prediction     │     │ • Revenue       │ │
│                           └──────────────────┘     └──────────────────┘     └────────┬────────┘ │
│                                    │                           │                      │         │
│                                    │                           │                      ▼         │
│                                    │                           │             ┌─────────────────┐ │
│                                    │                           │             │  Analytics /    │ │
│                                    │                           │             │  Data Warehouse │ │
│                                    │                           │             │  (BigQuery,     │ │
│                                    │                           │             │   Druid, etc.)  │ │
│                                    │                           │             └────────┬────────┘ │
│                                    │                           │                      │         │
│                                    ▼                           ▼                      ▼         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  EXPERIMENT CONFIGURATION & ANALYSIS                                                         ││
│  │  • Create experiments (variants, traffic %, targeting)                                       ││
│  │  • Compute metrics (daily, by segment)                                                       ││
│  │  • Statistical significance (t-test, sequential, CUPED)                                     ││
│  │  │                                                                                           ││
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      ││
│  │  │ Experiment  │   │ Metric      │   │ Results     │   │ Decision    │                      ││
│  │  │ Design UI   │   │ Calculator  │   │ Dashboard   │   │ (Ship/Hold) │                      ││
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Responsibility | Latency SLO | Scale (Meta-scale) |
|-----------|----------------|-------------|---------------------|
| **Assignment Service** | Map user → experiment variant (deterministic hash) | < 5ms P99 | 10B+ assignments/day |
| **Event Pipeline** | Ingest impressions, clicks, conversions, revenue | End-to-end < 15 min | 100B+ events/day |
| **Metric Computation** | Aggregate by experiment, variant, segment | Daily batch | Petabytes scanned |
| **Analysis Engine** | Statistical tests, multiple testing correction | Sub-second | 10K+ experiments active |

### Integration Points with ML Systems

1. **Model Serving**: The assignment service tells the inference layer which model variant to invoke (e.g., `model_v2` vs `model_v1`)
2. **Feature Store**: Experiment ID and variant are features passed to the model (for conditioning or logging)
3. **Monitoring**: Experiment metrics feed into ML monitoring dashboards

---

## 📚 Table of Contents

| # | Topic | Description |
|---|-------|--------------|
| 1 | [Experiment Design](./01-experiment-design.md) | Hypothesis formulation, randomization units, sample size calculation, power analysis, guardrail metrics, treatment vs control, Python power analysis |
| 2 | [Advanced Testing](./02-advanced-testing.md) | Multi-armed bandits (epsilon-greedy, UCB, Thompson sampling), interleaving for ranking, switchback experiments, cluster randomization, holdout groups, Python implementations |
| 3 | [Metric Design](./03-metric-design.md) | North star metrics, proxy metrics, counter metrics, OEC, metric sensitivity (CUPED), metric hierarchy, Python variance reduction |
| 4 | [Analysis Pitfalls](./04-analysis-pitfalls.md) | Simpson's paradox, novelty/primacy effects, multiple testing, peeking problem, network effects, survivorship bias, sequential testing, Python implementations |

---

## 🔑 Key Concepts at a Glance

### Experiment Types by Use Case

| Use Case | Experiment Type | Example |
|----------|-----------------|---------|
| **Model comparison** | A/B test | New recommendation model vs baseline |
| **Many variants** | Multi-armed bandit | 10 ranking algorithms, adapt traffic to winners |
| **Ranking evaluation** | Interleaving | Compare two ranking systems without full A/B |
| **Network effects** | Switchback | Time-based or cluster-based randomization |
| **Long-term impact** | Holdout | 1% permanently on control to measure delayed effects |

### Statistical Rigor Checklist

- [ ] **Randomization unit** matches analysis unit (user-level assign → user-level analyze)
- [ ] **Sample size** computed for desired power (typically 80%) and MDE
- [ ] **Guardrail metrics** defined (latency, crash rate, revenue—must not regress)
- [ ] **Multiple testing** correction if checking many metrics or segments
- [ ] **Peeking** avoided or sequential testing used
- [ ] **Interference** considered (network effects, marketplace dynamics)

---

## 🚨 Why This Matters for Interviews

ML Systems Design interviews at L5/E5+ frequently probe experimentation:

- **"How would you validate that this new ranking model is better?"** → A/B test design, metrics, sample size
- **"We have 20 model variants. How do we choose?"** → Multi-armed bandits or phased rollout
- **"Offline AUC improved 2% but online metrics didn't. Why?"** → Offline-online gap, metric design
- **"How do you run experiments in a marketplace with network effects?"** → Switchback, cluster randomization
- **"We're seeing contradictory results in different user segments. What's going on?"** → Simpson's paradox, segment-specific analysis

**Winning strategy**: Demonstrate you think like a data scientist *and* a systems engineer. Know the stats (power, MDE, randomization) and the infrastructure (assignment service, event pipeline, metric computation).

---

## 📋 Pre-Interview Checklist

- [ ] Can explain why offline metrics don't correlate with online impact
- [ ] Can design an A/B test: hypothesis, randomization unit, sample size
- [ ] Can compute sample size given power (80%), significance (5%), MDE
- [ ] Can explain when to use MAB vs A/B test
- [ ] Can define north star, proxy, and counter metrics
- [ ] Can explain Simpson's paradox with a concrete example
- [ ] Can describe the peeking problem and solutions (sequential testing, hold analysis)
- [ ] Can discuss network effects and switchback experiments

---

## 🚀 Next Steps

1. **[Experiment Design](./01-experiment-design.md)** – Learn to design statistically valid experiments from hypothesis to sample size.
2. **[Advanced Testing](./02-advanced-testing.md)** – Multi-armed bandits, interleaving, switchbacks for complex scenarios.
3. **[Metric Design](./03-metric-design.md)** – Define metrics that align with business goals and detect real effects.
4. **[Analysis Pitfalls](./04-analysis-pitfalls.md)** – Avoid Simpson's paradox, peeking, and other analysis traps.

---

## Related Topics

| Topic | Link | Connection |
|-------|------|------------|
| A/B Testing | [05-model-serving/03-ab-testing](../../phase-2-core-components/05-model-serving/03-ab-testing.md) | Basic A/B testing implementation |
| Model Monitoring | [06-monitoring-observability](../../phase-3-operations-and-reliability/06-monitoring-observability) | Experiment metrics feed monitoring |
| Recommendation Systems | [10-end-to-end-systems/01-recommendation-systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) | Primary use case for experimentation |
| Search Systems | [10-end-to-end-systems/02-search-systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/02-search-systems.md) | Interleaving, ranking experiments |

---

## Production Scale Reference

### Experimentation Platform Scale (Industry)

| Company | Experiments/Year | Assignments/Day | Events/Day |
|---------|------------------|-----------------|------------|
| **Meta** | 10,000+ | 10B+ | 100B+ |
| **Google** | 10,000+ | 10B+ | 100B+ |
| **Netflix** | 1,000+ | 1B+ | 10B+ |
| **Uber** | 500+ | 100M+ | 1B+ |
| **Airbnb** | 500+ | 100M+ | 1B+ |

### Key Metrics for Experiment Design

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| **Power** | 80% | Industry standard |
| **Significance (α)** | 5% | Two-tailed |
| **MDE (relative)** | 1–5% | Depends on metric variance |
| **Minimum run** | 2 weeks | Capture weekly cycle |
| **SRM threshold** | p < 0.001 | Sample ratio mismatch alert |
