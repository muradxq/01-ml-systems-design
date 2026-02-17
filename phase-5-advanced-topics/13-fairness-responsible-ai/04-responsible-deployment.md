# Responsible Deployment

## Overview

Responsible deployment completes the fairness lifecycle: ensuring that ML systems behave ethically and fairly *in production*. This includes managing **feedback loops**, enforcing **content policies**, designing **human-in-the-loop** workflows, providing **appeals processes**, **monitoring** fairness metrics, and **red teaming** for safety. Real-world case studies illustrate what goes wrong when these practices are neglected.

---

## 🔄 Feedback Loops and Bias Amplification

### The Problem

In many ML systems, model outputs influence future inputs. This creates feedback loops that can **amplify initial bias**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FEEDBACK LOOP IN RECOMMENDATION SYSTEMS                        │
│                                                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│   │  Model      │     │  Rank/Rec   │     │  User       │     │  Log       │     │
│   │  predicts   │────▶│  items A,B  │────▶│  clicks A   │────▶│  clicks    │     │
│   │  engagement │     │  higher     │     │  more       │     │  (training)│     │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     │
│          ▲                                                           │           │
│          │                                                           │           │
│          └────────────────────── Retrain ◀──────────────────────────┘           │
│                                                                                   │
│   Result: Model learns "A gets clicks" → shows more A → more clicks on A        │
│           → echo chamber, filter bubble, homogeneity                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Real-World Examples

| System | Feedback Loop | Amplification |
|--------|---------------|---------------|
| **Recommendation** | Show X → click X → train on X → show more X | Filter bubbles; minority content under-recommended |
| **Search** | Rank result 1st → more clicks → future model ranks it 1st | Position bias; rich get richer |
| **Hiring** | Recommend candidates → only they get interviewed → only they appear in hire data | Representation of non-recommended groups shrinks |
| **Content moderation** | Flag content → reviewers see it → labeling bias | Over-moderation of certain communities |
| **Ad delivery** | Show ads to group A → A clicks → show more to A | Discriminatory ad delivery |

### Mitigation Strategies

| Strategy | Description |
|----------|-------------|
| **Diversity injection** | Explicitly surface diverse content; don't rely only on engagement |
| **Exploration vs exploitation** | Use bandits/exploration to show less-engaged content occasionally |
| **Debiased logging** | Correct for position/selection bias in training labels |
| **Holdout evaluation** | Evaluate on data not influenced by the model (e.g., random exploration) |
| **Periodic resets** | Occasionally train from a less biased prior to reduce drift |

---

## 📜 Content Policy Enforcement

ML systems that moderate content (hate speech, misinformation, violence) must align with **content policies**. Fairness here means consistent enforcement across groups and contexts.

### Challenges

| Challenge | Description |
|-----------|-------------|
| **Imbalanced training data** | Some languages/communities have more labeled examples |
| **Context dependence** | Same words mean different things in different contexts (reclaimed slurs, satire) |
| **Policy ambiguity** | Policies are nuanced; labels are subjective |
| **Adversarial evasion** | Bad actors adapt to evade detection (e.g., misspellings, coded language) |
| **Over-moderation** | Model may over-flag certain communities due to training bias |

### Design Principles

1. **Sliced evaluation** for moderation: TPR, FPR, precision by language, region, community
2. **Human review** for edge cases and appeals
3. **Policy versioning** with model versioning—track which model enforces which policy
4. **Red team** with adversarial examples (coded language, boundary cases)

---

## 👤 Human-in-the-Loop Review

**When to use**: High-stakes decisions (content removal, account actions, hiring, lending) where errors have serious consequences.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HUMAN-IN-THE-LOOP WORKFLOW                                     │
│                                                                                   │
│   Input ──▶ Model ──▶ Score/Decision ──▶ Queue (by priority)                     │
│                                                │                                  │
│                    ┌───────────────────────────┼───────────────────────────┐     │
│                    │                           ▼                           │     │
│                    │   Auto-approve (high confidence)   Human review (low)  │     │
│                    │   Auto-reject (high confidence)    Human review (low)  │     │
│                    └───────────────────────────────────────────────────────┘     │
│                                                │                                  │
│                                                ▼                                  │
│                                    Reviewer UI + Decision                         │
│                                                │                                  │
│                                                ▼                                  │
│                                    Log decisions → Retrain / Feedback             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Queue Prioritization

| Priority | Criteria | Example |
|----------|----------|---------|
| **High** | User appeals, legal/policy flags | User claims wrongful ban |
| **Medium** | Low model confidence, ambiguous policy | Score near threshold |
| **Low** | Random sample for QA | Audit review sampling |
| **Bias monitoring** | Over-sample protected groups | Ensure fair review coverage |

### Reviewer Training

- **Calibration**: Reviewers aligned on policy definitions; periodic calibration sessions
- **Bias awareness**: Training on disparate impact, stereotype pitfalls
- **Audit**: Track reviewer agreement, flag outliers for re-training
- **Feedback**: Reviewer corrections flow back to model improvement

---

## 📋 Appeals Processes

Users affected by automated decisions need a path to contest them.

### User-Facing Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **In-product appeal** | "Disagree with this decision" button |
| **Form-based** | Structured form with context (e.g., "Why do you believe this was wrong?") |
| **Evidence upload** | User can provide additional context (screenshots, documents) |
| **Status tracking** | User can see appeal status and outcome |

### Escalation Paths

```
    User Appeal
         │
         ▼
    ┌─────────────┐
    │  Automated  │  (e.g., re-run with different threshold, check for errors)
    │  re-review  │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  Human      │  (trained reviewer)
    │  review     │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  Specialist │  (complex cases, legal)
    │  escalation │
    └─────────────┘
```

### Design Principles

1. **Transparency**: User understands what was decided and why (to the extent possible)
2. **Timeliness**: SLA for appeal resolution (e.g., 24–72 hours)
3. **Outcome communication**: Clear explanation of appeal outcome
4. **Feedback loop**: Appeal outcomes inform model retraining and policy updates
5. **No punishment**: Appeals should not lead to worse outcomes for the user

---

## 📊 Monitoring for Fairness in Production

### Drift in Fairness Metrics

Model performance and fairness can degrade over time due to:
- **Data drift**: Population shifts (demographics, behavior)
- **Concept drift**: Relationship between features and target changes
- **Feedback loops**: Model behavior alters the data distribution

### What to Monitor

| Metric Type | Examples |
|-------------|----------|
| **Performance by slice** | AUC, precision, recall per demographic group |
| **Fairness metrics** | Demographic parity ratio, equalized odds gap |
| **Volume by slice** | Ensure sufficient data for reliable estimates |
| **Outcome distribution** | Positive rate by group; alert on significant changes |

### Alerting

```
    Threshold-based:  parity_ratio < 0.8  →  PagerDuty
    Trend-based:      parity_ratio dropping 5% week-over-week  →  Slack
    Anomaly:          Sudden shift in slice performance  →  Investigate
```

---

## 🐍 Python: Fairness Monitoring Service

```python
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FairnessMetrics:
    demographic_parity_ratio: float
    tpr_gap: float
    fpr_gap: float
    positive_rate_by_group: Dict[str, float]
    sample_size_by_group: Dict[str, int]
    timestamp: str

class FairnessMonitoringService:
    """
    Production fairness monitoring with drift detection and alerting.
    """
    
    def __init__(
        self,
        protected_col: str,
        parity_ratio_threshold: float = 0.8,
        odds_gap_threshold: float = 0.1,
        min_sample_size: int = 100
    ):
        self.protected_col = protected_col
        self.parity_threshold = parity_ratio_threshold
        self.odds_threshold = odds_gap_threshold
        self.min_sample_size = min_sample_size
        self.baseline_metrics: Optional[FairnessMetrics] = None
        self.history: list = []
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: np.ndarray,
        group_names: Optional[Dict[int, str]] = None
    ) -> FairnessMetrics:
        """Compute fairness metrics for current batch."""
        groups = np.unique(group)
        if len(groups) != 2:
            raise ValueError("Binary protected attribute required")
        
        g0, g1 = groups[0], groups[1]
        m0, m1 = group == g0, group == g1
        
        rate_0 = y_pred[m0].mean()
        rate_1 = y_pred[m1].mean()
        ratio = rate_1 / rate_0 if rate_0 > 0 else 0.0
        
        tpr_0 = y_pred[m0 & (y_true == 1)].mean() if (y_true[m0] == 1).any() else 0
        tpr_1 = y_pred[m1 & (y_true == 1)].mean() if (y_true[m1] == 1).any() else 0
        fpr_0 = y_pred[m0 & (y_true == 0)].mean() if (y_true[m0] == 0).any() else 0
        fpr_1 = y_pred[m1 & (y_true == 0)].mean() if (y_true[m1] == 0).any() else 0
        
        tpr_gap = abs(tpr_0 - tpr_1)
        fpr_gap = abs(fpr_0 - fpr_1)
        
        names = group_names or {g0: "group_0", g1: "group_1"}
        
        metrics = FairnessMetrics(
            demographic_parity_ratio=ratio,
            tpr_gap=tpr_gap,
            fpr_gap=fpr_gap,
            positive_rate_by_group={names[g0]: rate_0, names[g1]: rate_1},
            sample_size_by_group={names[g0]: m0.sum(), names[g1]: m1.sum()},
            timestamp=datetime.now().isoformat()
        )
        return metrics
    
    def check_alerts(self, metrics: FairnessMetrics) -> list:
        """Return list of active alerts."""
        alerts = []
        
        if metrics.demographic_parity_ratio < self.parity_threshold:
            alerts.append({
                "severity": "high",
                "type": "parity_ratio",
                "message": f"Demographic parity ratio {metrics.demographic_parity_ratio:.3f} "
                           f"below threshold {self.parity_threshold}",
                "metrics": metrics
            })
        
        if metrics.tpr_gap > self.odds_threshold or metrics.fpr_gap > self.odds_threshold:
            alerts.append({
                "severity": "medium",
                "type": "equalized_odds",
                "message": f"TPR/FPR gap exceeds {self.odds_threshold}: "
                           f"TPR_gap={metrics.tpr_gap:.3f}, FPR_gap={metrics.fpr_gap:.3f}",
                "metrics": metrics
            })
        
        for grp, n in metrics.sample_size_by_group.items():
            if n < self.min_sample_size:
                alerts.append({
                    "severity": "low",
                    "type": "sample_size",
                    "message": f"Group {grp} has only {n} samples (min: {self.min_sample_size})",
                    "metrics": metrics
                })
        
        return alerts
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray):
        """Process batch and check for alerts."""
        metrics = self.compute_metrics(y_true, y_pred, group)
        self.history.append(metrics)
        
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
        
        alerts = self.check_alerts(metrics)
        return metrics, alerts
```

---

## 🔴 Red Teaming and Adversarial Testing

**Red teaming**: Deliberately trying to find failures, biases, and safety issues before they occur in production.

### What to Test

| Category | Examples |
|----------|----------|
| **Fairness** | Test with inputs from different demographics; measure disparity |
| **Safety** | Adversarial inputs (jailbreaks, prompt injection for LLMs) |
| **Robustness** | Out-of-distribution inputs, edge cases |
| **Alignment** | Does the system do what we intend? (e.g., reward hacking) |
| **Representation** | Are minority groups represented? Stereotyping? |

### Process

1. **Scenarios**: Define scenarios of concern (e.g., "model underperforms for non-native speakers")
2. **Test cases**: Create or curate test sets (hand-crafted, crowd-sourced, synthetic)
3. **Evaluation**: Run model; measure failure rates by scenario
4. **Remediation**: Fix model, policy, or pipeline; re-test
5. **Documentation**: Record findings in model card and audit log

---

## 🐍 Python: Bias Alert System

```python
import logging
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class BiasAlert:
    alert_id: str
    severity: str
    message: str
    metrics: dict
    recommended_actions: List[str]

class BiasAlertSystem:
    """
    Centralized bias alert handling with configurable handlers.
    """
    
    def __init__(self):
        self.handlers: List[Callable[[BiasAlert], None]] = []
        self.alert_log: List[BiasAlert] = []
    
    def add_handler(self, handler: Callable[[BiasAlert], None]):
        self.handlers.append(handler)
    
    def emit(self, alert: BiasAlert):
        self.alert_log.append(alert)
        for h in self.handlers:
            try:
                h(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def slack_handler(webhook_url: str):
        """Factory for Slack webhook handler."""
        def handler(alert: BiasAlert):
            # In production: requests.post(webhook_url, json={...})
            logging.info(f"Slack alert: {alert.message}")
        return handler
    
    def pagerduty_handler(service_key: str):
        """Factory for PagerDuty handler (high severity only)."""
        def handler(alert: BiasAlert):
            if alert.severity == "high":
                # In production: trigger PagerDuty incident
                logging.warning(f"PagerDuty: {alert.message}")
        return handler
```

---

## 📚 Case Studies: Real-World Fairness Failures

### 1. Amazon Hiring Tool (2018)

| What happened | Recruiting tool trained on 10 years of resumes; learned to prefer male candidates (e.g., penalized "women's" in resume text) |
|---------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Bias type** | Historical bias (past hiring was male-dominated) |
| **Lesson** | Removing explicit gender features is insufficient; model learned proxies. Need fairness constraints and rigorous evaluation. |

### 2. COMPAS Recidivism (ProPublica, 2016)

| What happened | COMPAS predicted recidivism; higher false positive rate for Black defendants than White defendants |
|---------------|-----------------------------------------------------------------------------------------------------|
| **Bias type** | Measurement bias (arrests as proxy), evaluation bias (different base rates) |
| **Lesson** | Calibration vs. equalized odds trade-off. No single metric; must choose based on values. Need transparency and external audit. |

### 3. Facial Recognition (NIST, Buolamwini)

| What happened | Higher error rates for darker-skinned and female faces; some systems 100x worse for Black women than White men |
|---------------|----------------------------------------------------------------------------------------------------------------|
| **Bias type** | Representation bias (training data skewed), intersectionality |
| **Lesson** | Evaluate across demographic intersections. Diverse training data is necessary but not sufficient—explicit fairness testing required. |

### 4. Google Ads (Sweeney, 2013)

| What happened | Search for "Black names" more likely to show ad suggesting arrest record |
|---------------|-----------------------------------------------------------------------|
| **Bias type** | Deployment/ad delivery bias; feedback from ad performance |
| **Lesson** | Monitor ad delivery and outcomes by demographic slice; audit ad targeting systems. |

### 5. Microsoft Tay (2016)

| What happened | Chatbot learned abusive language from user interactions within hours |
|---------------|---------------------------------------------------------------------|
| **Bias type** | Feedback loop; adversarial users; insufficient safeguards |
| **Lesson** | Need content filters, human oversight, rate limiting, and red teaming before release. |

---

## 📋 Trade-offs Summary

| Topic | Trade-off |
|-------|-----------|
| **Feedback loops** | Engagement optimization vs. diversity; exploitation vs. exploration |
| **Human-in-the-loop** | Speed/scale vs. accuracy; cost vs. risk |
| **Appeals** | Volume vs. quality of review; automation vs. human judgment |
| **Monitoring** | Alert fatigue vs. missed issues; slice granularity vs. sample size |
| **Red teaming** | Coverage vs. cost; finding issues vs. fixing them |

---

## 📋 Interview Tips

1. **Feedback loops**: "Recommendation systems create feedback loops—we show what's popular, users click it, we train on that. We'd add diversity injection and exploration to avoid echo chambers."
2. **Human-in-the-loop**: "For high-stakes decisions like content removal, we'd route low-confidence cases to human review, prioritize appeals, and use reviewer feedback to improve the model."
3. **Appeals**: "Users need a clear path to contest decisions. We'd provide an in-product appeal, SLA for resolution, and feed outcomes back into model improvement."
4. **Monitoring**: "We'd track fairness metrics by slice in production—parity ratio, TPR/FPR gaps—and alert when they drift beyond thresholds."
5. **Case studies**: Reference specific failures (Amazon hiring, COMPAS, facial recognition) to show you've thought about real-world consequences.
6. **Red teaming**: "Before launch, we'd red team—test adversarial inputs, minority group scenarios, edge cases—and document findings in the model card."

---

## Related Topics

- [Bias Detection](./01-bias-detection.md) – Metrics to monitor in production
- [Fairness Constraints](./02-fairness-constraints.md) – Mitigations before deployment
- [Model Auditing](./03-model-auditing.md) – Pre-launch review and documentation
- [Model Monitoring](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md) – General monitoring infrastructure
- [Recommendation Systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) – Feedback loops in practice

---

## 📋 Responsible Deployment Checklist

Before launching an ML system with potential fairness impact:

- [ ] Model card completed and reviewed
- [ ] Sliced evaluation across relevant protected groups
- [ ] Fairness metrics (parity, equalized odds, calibration) documented
- [ ] Human-in-the-loop defined for high-stakes decisions
- [ ] Appeals process designed and communicated
- [ ] Fairness monitoring and alerting configured
- [ ] Red team or adversarial testing conducted
- [ ] Legal/compliance sign-off (if applicable)
- [ ] Stakeholder alignment on fairness-accuracy trade-off
