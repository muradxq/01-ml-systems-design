# Model Auditing

## Overview

Model auditing ensures ML systems are transparent, accountable, and compliant. It includes **documentation** (model cards, datasheets), **explainability** (SHAP, LIME, Integrated Gradients), and **process** (pre-launch reviews, periodic audits, regulatory alignment). In ML Systems Design interviews, demonstrating that you think about auditability—before deployment—signals maturity and responsibility.

---

## 🎯 Model Cards

### What Are Model Cards?

**Model cards** are short documents that summarize a model's intended use, limitations, performance across relevant slices, and ethical considerations. Google introduced the concept in 2019 ("Model Cards for Model Reporting" by Mitchell et al.).

### Google's Model Card Framework

| Section | Content |
|---------|---------|
| **Model Details** | Model type, version, date, developers, contact |
| **Intended Use** | Primary use case, intended users, out-of-scope use |
| **Factors** | Relevant factors (demographics, geography, etc.) for evaluation |
| **Metrics** | Performance metrics, evaluation approach, variance across factors |
| **Evaluation Data** | Datasets used, demographics, limitations |
| **Ethical Considerations** | Known biases, fairness trade-offs, risks |
| **Caveats and Recommendations** | When not to use, recommended monitoring |

### What to Include in a Model Card

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MODEL CARD TEMPLATE                                       │
│                                                                                   │
│  1. MODEL IDENTITY                                                                │
│     - Name, version, date                                                         │
│     - Training framework, hardware                                                │
│                                                                                   │
│  2. INTENDED USE                                                                  │
│     - Primary use case                                                            │
│     - Intended users (internal, external, automated systems)                      │
│     - Out-of-scope: "Do not use for X"                                            │
│                                                                                   │
│  3. TRAINING DATA                                                                 │
│     - Dataset(s), size, date range                                                │
│     - Demographics / slice composition                                            │
│     - Known gaps or biases in data                                                │
│                                                                                   │
│  4. EVALUATION                                                                    │
│     - Metrics (accuracy, AUC, fairness metrics)                                    │
│     - Sliced evaluation results (table)                                            │
│     - Test set characteristics                                                    │
│                                                                                   │
│  5. ETHICAL CONSIDERATIONS                                                        │
│     - Known limitations                                                           │
│     - Fairness-accuracy trade-offs made                                           │
│     - Recommendations for monitoring                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 Datasheets for Datasets

**Idea**: Document datasets like product datasheets—provenance, composition, use, and limitations.

### Datasheet Sections (Gebru et al.)

| Section | Questions |
|---------|-----------|
| **Motivation** | Why was the dataset created? Who funded it? |
| **Composition** | What does it contain? Instances, features, labels? |
| **Collection** | How was it acquired? Sampling, time range? |
| **Preprocessing** | Cleaning, labeling process, who did it? |
| **Uses** | Recommended uses, prohibited uses |
| **Distribution** | How is it distributed? License? |
| **Maintenance** | Who maintains it? Update frequency? |

**Why it matters for fairness**: Datasheets force you to document demographics, labeling subjectivity, and potential biases upfront.

---

## 🔍 Explainability Methods

Explainability helps auditors and users understand *why* a model made a decision. Different methods suit different model types and use cases.

### SHAP (SHapley Additive exPlanations)

**Idea**: Assign each feature a contribution (Shapley value) to the prediction. SHAP values sum to the difference between the model output and the baseline (e.g., mean prediction).

**Properties**: 
- **Additive**: $\phi_0 + \sum_i \phi_i = f(x) - E[f]$ 
- **Consistency**: If a feature's effect increases, its SHAP value doesn't decrease
- **Local accuracy**: Sum of SHAP values equals the prediction

**When to use**: Tree models (TreeExplainer), general models (KernelExplainer, slower).

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP for tree models (fast)
explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values may be list [class0, class1]
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use positive class

# Summary plot (global importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for single prediction (local explanation)
shap.force_plot(
    explainer.expected_value[1],
    shap_values[0],
    X_test.iloc[0],
    feature_names=feature_names
)

# Bar plot: mean |SHAP| per feature
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)
```

---

### LIME (Local Interpretable Model-agnostic Explanations)

**Idea**: Approximate the black-box model *locally* with an interpretable model (e.g., linear) on perturbed samples. The linear coefficients become the explanation.

**When to use**: Any model; image, text, tabular. Good when you need local explanations and SHAP is too slow.

```python
import lime
import lime.lime_tabular

# Create explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['negative', 'positive'],
    mode='classification'
)

# Explain single instance
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10,
    top_labels=1
)

# Show explanation
exp.show_in_notebook(show_table=True)

# Get feature contributions
exp.as_list()
# [(feature_name, weight), ...]
```

---

### Integrated Gradients

**Idea**: Attribute the prediction to input features by integrating gradients along a path from a baseline (e.g., all zeros) to the input. Satisfies **sensitivity** (nonzero input change → nonzero attribution) and **implementation invariance**.

**When to use**: Deep neural networks; when you need theoretically grounded attributions.

```python
import torch
from torch.nn import functional as F

def integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = 50
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution.
    
    Args:
        model: The model
        input_tensor: Input to explain (requires_grad=True)
        baseline: Baseline input (e.g., zeros)
        steps: Number of interpolation steps
    """
    model.eval()
    input_tensor.requires_grad_(True)
    
    # Interpolate between baseline and input
    scaled_inputs = [
        baseline + (i / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]
    
    grads = []
    for scaled_in in scaled_inputs:
        scaled_in.requires_grad_(True)
        output = model(scaled_in)
        output.backward(retain_graph=True)
        grads.append(scaled_in.grad.clone())
    
    # Average gradients and multiply by (input - baseline)
    avg_grad = torch.stack(grads).mean(dim=0)
    integrated_grads = (input_tensor - baseline) * avg_grad
    
    return integrated_grads
```

---

### Attention Visualization

**Idea**: For attention-based models (e.g., Transformers), visualize which tokens/regions the model attended to.

**When to use**: NLP, vision transformers. Complements but doesn't replace attribution—attention is not the same as causation.

---

## 📊 Explainability Trade-offs

| Method | Model Type | Speed | Local/Global | Interpretability |
|--------|------------|-------|--------------|------------------|
| **SHAP** | Any (TreeExplainer for trees) | Fast for trees, slow for others | Both | High |
| **LIME** | Any | Medium | Local | High |
| **Integrated Gradients** | Differentiable (DNNs) | Medium | Local | Medium |
| **Attention** | Attention models | Fast | Local | Medium (attention ≠ cause) |
| **Feature importance** | Tree models | Fast | Global | Medium |

---

## 📋 Audit Process

### Pre-Launch Review

Before deploying a new model or significant update:

1. **Model card** completed and reviewed
2. **Fairness evaluation** across relevant slices
3. **Explainability** verified for sample predictions
4. **Red team** or adversarial review (for high-stakes systems)
5. **Stakeholder sign-off** (legal, product, policy)

### Periodic Audits

- **Quarterly** for high-stakes systems (lending, hiring, content moderation)
- **Annually** for lower-stakes systems
- Re-run fairness metrics on production data distribution
- Check for drift in slice performance

### Triggered Audits

Audit when:
- User complaints or appeals spike
- Media or regulatory inquiry
- Significant data or model change
- New protected attribute or slice of concern

---

## 📜 Regulatory Requirements

### EU AI Act

| Risk Tier | Examples | Requirements |
|-----------|----------|--------------|
| **Unacceptable** | Social scoring, manipulation | Ban |
| **High** | Credit, hiring, critical infra | Conformity assessment, human oversight, transparency |
| **Limited** | Chatbots | Transparency (disclose AI) |
| **Minimal** | Spam filters | No specific requirements |

**Fairness implications**: High-risk AI must undergo conformity assessment including fundamental rights impact; bias monitoring and mitigation expected.

---

### NIST AI Risk Management Framework (RMF)

- **Govern**: Establish policies and culture
- **Map**: Understand context and risk
- **Manage**: Allocate resources, monitor
- **Measure**: Assess and evaluate

**Fairness**: Addressed in "Measure" (testing for bias) and "Manage" (mitigation, monitoring).

---

### Company-Specific Policies

Meta, Google, etc. have internal AI principles and review boards (e.g., Meta's Responsible AI, Google's AI Principles). Expect questions about how you'd design for internal review and compliance.

---

## 🐍 Python: Model Card Generation

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json

@dataclass
class ModelCard:
    """Structured model card for documentation and audit."""
    
    model_name: str
    version: str
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    # Intended use
    intended_use: str = ""
    intended_users: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    
    # Training
    training_data: str = ""
    training_data_demographics: Dict[str, str] = field(default_factory=dict)
    
    # Evaluation
    metrics: Dict[str, float] = field(default_factory=dict)
    sliced_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Fairness
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    known_limitations: List[str] = field(default_factory=list)
    
    # Recommendations
    monitoring_recommendations: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate model card as Markdown."""
        md = f"# Model Card: {self.model_name} v{self.version}\n\n"
        md += f"**Date**: {self.date}\n\n"
        md += "## Intended Use\n\n" + self.intended_use + "\n\n"
        md += "## Metrics\n\n"
        for k, v in self.metrics.items():
            md += f"- **{k}**: {v:.4f}\n"
        md += "\n## Sliced Evaluation\n\n"
        md += "| Slice | " + " | ".join(self.metrics.keys()) + " |\n"
        md += "|" + "---|" * (len(self.metrics) + 1) + "\n"
        for slice_name, slice_metrics in self.sliced_metrics.items():
            md += f"| {slice_name} | " + " | ".join(f"{v:.4f}" for v in slice_metrics.values()) + " |\n"
        md += "\n## Fairness Metrics\n\n"
        for k, v in self.fairness_metrics.items():
            md += f"- **{k}**: {v:.4f}\n"
        md += "\n## Known Limitations\n\n"
        for lim in self.known_limitations:
            md += f"- {lim}\n"
        md += "\n## Monitoring Recommendations\n\n"
        for rec in self.monitoring_recommendations:
            md += f"- {rec}\n"
        return md
    
    def to_json(self) -> str:
        """Serialize to JSON for programmatic use."""
        return json.dumps(self.__dict__, indent=2)
```

---

## 🐍 Python: SHAP Explanation Wrapper

```python
import shap
import pandas as pd
import numpy as np
from typing import Optional, List

def generate_shap_report(
    model,
    X: pd.DataFrame,
    sample_size: int = 100,
    model_type: str = "tree"
) -> dict:
    """
    Generate SHAP-based explanation report for auditing.
    
    Returns dict with:
    - global_importance: mean |SHAP| per feature
    - sample_explanations: list of (instance_idx, top_features)
    """
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    if model_type == "tree":
        explainer = shap.TreeExplainer(model, X_sample)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class for binary
    
    # Global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = dict(zip(X_sample.columns, mean_abs_shap))
    global_importance = dict(sorted(global_importance.items(), 
                                   key=lambda x: x[1], reverse=True))
    
    # Sample local explanations
    sample_explanations = []
    for i in range(min(5, len(X_sample))):
        sv = shap_values[i]
        top_indices = np.argsort(np.abs(sv))[-5:][::-1]
        top_features = [(X_sample.columns[j], float(sv[j])) for j in top_indices]
        sample_explanations.append({"instance_idx": i, "top_contributors": top_features})
    
    return {
        "global_importance": global_importance,
        "sample_explanations": sample_explanations,
        "base_value": float(explainer.expected_value[1]) if isinstance(
            explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    }
```

---

## 📋 Interview Tips

1. **Proactively mention model cards**: "We'd create a model card before launch documenting intended use, evaluation across slices, and known limitations."
2. **Datasheets for datasets**: "We'd document our training data with a datasheet—composition, demographics, labeling process—to surface bias risks early."
3. **SHAP vs LIME**: "SHAP has stronger theoretical properties (Shapley values) and is fast for tree models. LIME is model-agnostic and good for one-off explanations. For production auditing of tree models, we'd use SHAP."
4. **Explainability limits**: "Explanations are helpful for debugging and trust but don't prove causality. We'd use them as one input to audits, not the only one."
5. **Regulation**: "For EU deployment, we'd need to map to the AI Act risk tier. High-risk use cases require conformity assessment and ongoing monitoring."
6. **Audit triggers**: "We'd do pre-launch review, quarterly audits for high-stakes systems, and triggered audits on complaints or significant changes."

---

## Related Topics

- [Bias Detection](./01-bias-detection.md) – Metrics to include in model cards
- [Fairness Constraints](./02-fairness-constraints.md) – Document constraints in model cards
- [Responsible Deployment](./04-responsible-deployment.md) – Monitoring and triggered audits in production
- [Compliance](../../phase-3-operations-and-reliability/09-security-privacy/04-compliance.md) – GDPR, AI Act, NIST RMF

---

## 📝 Model Card Example (Snippet)

```markdown
# Model Card: Loan Approval v2.1

## Intended Use
- Primary: Automate initial creditworthiness screening for personal loan applications
- Users: Internal underwriters; not for final approval without human review
- Out-of-scope: Commercial loans; automated denial without appeal path

## Training Data
- Source: Historical applications 2018–2023, ~2M records
- Demographics: 62% White, 18% Black, 12% Hispanic, 8% Other (self-reported)
- Known gaps: Underrepresentation of applicants from rural areas; thin-file applicants

## Evaluation (Sliced)
| Slice     | AUC   | Demographic Parity Ratio | TPR Gap |
|-----------|-------|---------------------------|---------|
| Overall   | 0.82  | 0.85                      | 0.06    |
| White     | 0.84  | —                         | —       |
| Black     | 0.78  | —                         | —       |
| Hispanic  | 0.80  | —                         | —       |

## Known Limitations
- Lower AUC for Black applicants; ongoing mitigation
- Model not validated for non-US applicants
- Calibration may drift; monitor quarterly

## Monitoring
- Track parity ratio weekly; alert if < 0.8
- Human review for 10% random sample + all appeals
```

---

## 🧪 Explainability Best Practices

1. **Use the right tool**: SHAP for trees and global importance; LIME for local, model-agnostic; IG for DNNs.
2. **Don't over-interpret**: Attributions suggest correlation, not causation. Use for debugging and trust, not legal defense.
3. **Test for fairness**: If a protected attribute (or proxy) has high SHAP importance, investigate.
4. **Version explanations**: Store explanation method and parameters with model version for reproducibility.
