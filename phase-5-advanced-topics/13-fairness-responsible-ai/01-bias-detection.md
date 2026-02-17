# Bias Detection

## Overview

Bias detection is the foundation of responsible ML. Before you can fix bias, you must identify its source and quantify its impact. This topic covers the taxonomy of bias in ML systems, fairness metrics with formal definitions, sliced evaluation strategies, intersectional analysis, and Python implementations you can use in practice and discuss in interviews.

---

## 🎯 Types of Bias

Bias can enter ML systems at multiple stages. Understanding *where* bias originates helps you choose the right mitigation strategy.

### 1. Historical Bias

**Definition**: The world itself reflects past discrimination. Training data captures biased outcomes from human decisions.

| Example | What Happens |
|---------|--------------|
| **Resume screening** | Historical hiring favored men → training data has more positive labels for male candidates → model learns to prefer male names, male-coded language |
| **Loan approval** | Redlining and discriminatory lending → minority neighborhoods have fewer approved loans in data → model learns to reject similar applications |
| **Healthcare risk scores** | Past underinvestment in minority communities → lower healthcare utilization in data → model underestimates needs for Black patients |

**Interview tip**: "Historical bias is the most common—we're training on the world as it is, including its injustices. We need to either correct the labels, use causal methods, or constrain the model."

---

### 2. Representation Bias

**Definition**: Some groups are under- or over-represented in the training data relative to the population or use case.

| Example | What Happens |
|---------|--------------|
| **Facial recognition** | Training data disproportionately from lighter-skinned individuals → lower accuracy for dark-skinned faces (NIST studies, Joy Buolamwini's work) |
| **Speech recognition** | Training data from mostly American English speakers → worse performance for accents, dialects |
| **Medical imaging** | Datasets from US/Europe → models don't generalize to populations with different disease prevalence or anatomy |

```
                    Data Distribution vs. Population

    Training Data              Population (Production)
    ┌─────────────┐            ┌─────────────┐
    │  Group A    │ 70%        │  Group A    │ 50%
    │  Group B    │ 20%        │  Group B    │ 30%
    │  Group C    │ 10%        │  Group C    │ 20%
    └─────────────┘            └─────────────┘
    
    Model optimized for Group A → Poor performance on B, C in production
```

---

### 3. Measurement Bias

**Definition**: The way we measure or label the target variable is systematically different across groups.

| Example | What Happens |
|---------|--------------|
| **Recidivism prediction (COMPAS)** | Arrests used as proxy for crime—but arrest rates differ by neighborhood and policing—so labels are biased |
| **Teacher evaluation** | Student test scores used to evaluate teachers—but baseline student ability varies by school demographics |
| **Credit scores** | Traditional credit data (e.g., FICO) underrepresents thin-file populations (immigrants, young people) → alternative data may help or introduce new bias |

---

### 4. Aggregation Bias

**Definition**: A single model is trained for all groups, but optimal decision boundaries differ by group. One size doesn't fit all.

| Example | What Happens |
|---------|--------------|
| **Healthcare** | A single risk model may work well for the majority group but mis-calibrate for minorities with different disease patterns |
| **Recommendation** | A global collaborative filter may recommend items popular with the majority, under-serving niche groups |
| **Natural language** | One sentiment model for all languages/dialects—slang and cultural context differ |

---

### 5. Evaluation Bias

**Definition**: Evaluation methodology fails to surface disparity. Aggregate metrics hide subgroup performance gaps.

| Example | What Happens |
|---------|--------------|
| **Overall accuracy 95%** | But 70% for Group A and 99% for Group B—aggregate looks fine |
| **Single-group validation** | Only validating on majority group |
| **Wrong metric** | Optimizing AUC when the deployment threshold affects groups differently |

**Fix**: Sliced evaluation—report metrics per subgroup.

---

### 6. Deployment Bias

**Definition**: How and where the model is deployed creates disparate impact, independent of model predictions.

| Example | What Happens |
|---------|--------------|
| **Healthcare** | Model recommended for affluent hospitals first → lower access for underserved populations |
| **Language** | Product only in English → non-English speakers never benefit |
| **Device/connectivity** | Mobile-heavy users in developing countries get worse experience if model is desktop-optimized |

---

### 7. Selection Bias

**Definition**: The data we observe is not representative because of how it was selected (missing data, non-random sampling).

| Example | What Happens |
|---------|--------------|
| **User feedback** | Only users who engage provide labels → passive users (who may be different demographically) are under-represented |
| **A/B test analysis** | Dropping "ineligible" users can bias treatment effect estimates |
| **Survival bias** | Training on "survivors" (e.g., employees who stayed) excludes those who left—different characteristics |

---

### 8. Feedback Loop Bias

**Definition**: Model outputs influence future training data, amplifying initial bias.

| Example | What Happens |
|---------|--------------|
| **Recommendation systems** | Show more of X → users click more on X → next model trained on more X → show even more X (echo chamber) |
| **Search** | Rank certain results higher → users click them → future model learns to rank them higher (reinforcement) |
| **Hiring** | Model recommends candidates → only those get interviewed → only those get hired → future data has more of that type |

```
                    Feedback Loop Diagram

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Model     │     │   User      │     │   New       │
    │   ranks A   │────▶│   clicks A  │────▶│   Training  │
    │   higher    │     │   more      │     │   Data      │
    └─────────────┘     └─────────────┘     └─────────────┘
           ▲                                        │
           │                                        │
           └────────────────────────────────────────┘
                    Model retrained on biased data
```

---

## 📐 Fairness Metrics (Formulas)

Different fairness definitions capture different normative principles. **No single metric is universally correct**—context matters.

### Demographic Parity (Statistical Parity)

**Idea**: The positive prediction rate should be equal across groups.

**Formula**:
$$P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b) \quad \forall a, b$$

**Relaxed (ratio)**:
$$\frac{P(\hat{Y}=1 \mid A=\text{protected})}{P(\hat{Y}=1 \mid A=\text{reference})} \geq 0.8 \quad \text{(80% rule)}$$

**When to use**: Hiring, lending—when equal *opportunity to be selected* is the goal.

**Limitation**: Doesn't account for differing base rates (qualification). Can require accepting unqualified from one group or rejecting qualified from another.

---

### Equalized Odds (True Positive Rate Parity + False Positive Rate Parity)

**Idea**: Among those who *should* get the positive outcome (Y=1), equal chance of being correctly predicted; among those who shouldn't (Y=0), equal chance of being incorrectly predicted.

**Formula**:
- **TPR parity**: $P(\hat{Y}=1 \mid Y=1, A=a) = P(\hat{Y}=1 \mid Y=1, A=b)$
- **FPR parity**: $P(\hat{Y}=1 \mid Y=0, A=a) = P(\hat{Y}=1 \mid Y=0, A=b)$

**When to use**: Criminal justice, healthcare—when we care about *accuracy of decisions* conditional on ground truth.

**Limitation**: May conflict with demographic parity when base rates differ.

---

### Calibration (Predictive Parity)

**Idea**: Among those predicted positive, the actual positive rate should be the same across groups. A score of 0.7 should mean 70% chance of positive outcome for all groups.

**Formula**:
$$P(Y=1 \mid \hat{Y}=1, A=a) = P(Y=1 \mid \hat{Y}=1, A=b)$$

**When to use**: Risk scores (credit, health)—when the *interpretation of the score* must be consistent.

**Limitation**: Calibration and equalized odds can't both hold in general when base rates differ (Chouldechova, 2017).

---

### Individual Fairness

**Idea**: Similar individuals should receive similar predictions.

**Formula**: For a distance metric $d$:
$$d(x_i, x_j) \text{ small } \Rightarrow |\hat{y}_i - \hat{y}_j| \text{ small}$$

**When to use**: When you can define "similar" well (e.g., in ranking, two users with same preferences should get similar rankings).

**Limitation**: Defining similarity is hard; can perpetuate existing bias if "similar" encodes protected attributes.

---

### Counterfactual Fairness

**Idea**: Would the prediction change if the individual's protected attribute were different, holding everything else equal?

**Formula**: For protected attribute $A$:
$$P(\hat{Y}_{A\leftarrow a}(U) = y \mid X=x, A=a) = P(\hat{Y}_{A\leftarrow b}(U) = y \mid X=x, A=a)$$

**When to use**: Causal settings—when we want to remove *direct* effect of protected attribute.

**Limitation**: Requires causal model; hard to validate in practice.

---

## 📊 Trade-offs: Which Metric to Use When

| Scenario | Preferred Metric(s) | Rationale |
|----------|-------------------|-----------|
| **Hiring** | Demographic parity, equalized odds | Equal opportunity; avoid disparate impact |
| **Lending** | Calibration, equalized odds | Score must be interpretable; avoid discrimination in approvals |
| **Criminal justice** | Equalized odds | Balance FPR (wrongful detention) and FNR (wrongful release) |
| **Healthcare** | Calibration | Risk score must mean same thing across groups |
| **Content moderation** | Equalized odds | Similar error rates across groups; avoid over/under-moderation |
| **Recommendations** | Individual fairness, diversity metrics | Similar users get similar recs; avoid filter bubbles |

---

## 🔬 Sliced Evaluation

**Principle**: Never evaluate only on aggregate metrics. Always slice by protected attributes and other relevant subgroups.

```
                    Sliced Evaluation Report

    Metric: AUC-ROC

    Overall:     0.82
    ─────────────────────────────────────────
    By Gender:
      Male:      0.84  ✓
      Female:    0.79  ⚠ (gap: 0.05)
      Other:     0.81
    ─────────────────────────────────────────
    By Race:
      White:     0.85  ✓
      Black:     0.76  ⚠ (gap: 0.09)
      Asian:     0.83
      Hispanic:  0.80
    ─────────────────────────────────────────
    Intersectional (Gender × Race):
      Male/White:   0.86
      Female/Black: 0.72  ⚠ (largest gap)
      ...
```

---

## 🔗 Intersectional Bias Analysis

**Intersectionality**: Disparity can be strongest at the *intersection* of multiple protected attributes (e.g., Black women may face unique bias not explained by "Black" or "woman" alone).

**Approach**: Evaluate across combinations of attributes. Be aware of sample size—some intersections may have few examples.

---

## 🐍 Python Implementation: Fairness Metrics

```python
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, roc_auc_score

def demographic_parity_ratio(
    y_pred: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray
) -> float:
    """
    Demographic parity: P(ŷ=1|A) / P(ŷ=1|B).
    Returns ratio; 1.0 = perfect parity.
    """
    rate_a = y_pred[group_a].mean()
    rate_b = y_pred[group_b].mean()
    if rate_b == 0:
        return 0.0
    return rate_a / rate_b


def equalized_odds_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    Compute TPR and FPR for each group.
    Returns dict: {'group_0': (tpr, fpr), 'group_1': (tpr, fpr)}
    """
    results = {}
    
    for group_id, mask in enumerate([~group_mask, group_mask]):
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        results[f'group_{group_id}'] = (tpr, fpr)
    
    return results


def calibration_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_mask: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Calibration: among those with score in bin k, what fraction has Y=1?
    Returns bin means and fractions per group.
    """
    results = {}
    
    for group_id, mask in enumerate([~group_mask, group_mask]):
        y_t = y_true[mask]
        y_s = y_score[mask]
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_s, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        cal_vals = []
        for b in range(n_bins):
            in_bin = bin_indices == b
            if in_bin.sum() > 0:
                cal_vals.append(y_t[in_bin].mean())
            else:
                cal_vals.append(np.nan)
        
        results[f'group_{group_id}'] = np.array(cal_vals)
    
    return results


def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    protected_attr: np.ndarray,
    attr_name: str = "protected"
) -> pd.DataFrame:
    """
    Generate a fairness report across binary protected attribute.
    """
    group_0 = protected_attr == 0
    group_1 = protected_attr == 1
    
    report = []
    
    # Demographic parity
    for g, name in [(group_0, "group_0"), (group_1, "group_1")]:
        rate = y_pred[g].mean()
        report.append({"metric": "positive_rate", "group": name, "value": rate})
    
    dp_ratio = demographic_parity_ratio(y_pred, group_1, group_0)
    report.append({"metric": "demographic_parity_ratio", "group": "ratio", "value": dp_ratio})
    
    # Equalized odds
    eo = equalized_odds_metrics(y_true, y_pred, group_1)
    for grp, (tpr, fpr) in eo.items():
        report.append({"metric": "TPR", "group": grp, "value": tpr})
        report.append({"metric": "FPR", "group": grp, "value": fpr})
    
    # AUC per group
    if y_score is not None:
        for g, name in [(group_0, "group_0"), (group_1, "group_1")]:
            if y_true[g].sum() > 0 and (1 - y_true[g]).sum() > 0:
                auc = roc_auc_score(y_true[g], y_score[g])
            else:
                auc = np.nan
            report.append({"metric": "AUC", "group": name, "value": auc})
    
    return pd.DataFrame(report)
```

---

## 🐍 Python Implementation: Sliced Evaluation

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sliced_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slices: Dict[str, np.ndarray],
    metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
) -> pd.DataFrame:
    """
    Evaluate model performance across predefined slices.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        slices: Dict mapping slice_name -> boolean mask for that slice
        metrics: List of metric names to compute
    
    Returns:
        DataFrame with columns: slice_name, metric, value, sample_size
    """
    results = []
    
    for slice_name, mask in slices.items():
        y_t = y_true[mask]
        y_p = y_pred[mask]
        n = mask.sum()
        
        if n == 0:
            continue
            
        row = {"slice": slice_name, "sample_size": n}
        
        if "accuracy" in metrics:
            row["accuracy"] = accuracy_score(y_t, y_p)
        if "precision" in metrics:
            row["precision"] = precision_score(y_t, y_p, zero_division=0)
        if "recall" in metrics:
            row["recall"] = recall_score(y_t, y_p, zero_division=0)
        if "f1" in metrics:
            row["f1"] = f1_score(y_t, y_p, zero_division=0)
        
        results.append(row)
    
    return pd.DataFrame(results)


def intersectional_slices(
    df: pd.DataFrame,
    protected_cols: List[str]
) -> Dict[str, np.ndarray]:
    """
    Create slices for intersectional analysis.
    Each slice is a combination of protected attribute values.
    """
    slices = {}
    
    for combo, sub_df in df.groupby(protected_cols):
        if isinstance(combo, tuple):
            name = "_".join(str(c) for c in combo)
        else:
            name = str(combo)
        
        # Create boolean mask (assuming we have index alignment)
        mask = np.zeros(len(df), dtype=bool)
        mask[sub_df.index] = True
        slices[name] = mask
    
    return slices


def detect_fairness_anomalies(
    report: pd.DataFrame,
    metric: str,
    threshold_gap: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Flag slices where metric deviates significantly from overall.
    """
    overall = report[report["slice"] == "overall"]
    if len(overall) == 0:
        return []
    
    overall_val = overall[metric].values[0]
    anomalies = []
    
    for _, row in report.iterrows():
        if row["slice"] == "overall":
            continue
        gap = abs(row[metric] - overall_val)
        if gap > threshold_gap:
            anomalies.append({
                "slice": row["slice"],
                "metric": metric,
                "value": row[metric],
                "overall": overall_val,
                "gap": gap
            })
    
    return anomalies


# Example usage
def example_sliced_eval():
    np.random.seed(42)
    n = 1000
    
    # Simulated data: group 1 has worse base rate
    group = np.random.binomial(1, 0.3, n)  # 30% in protected group
    y_true = np.random.binomial(1, 0.3 + 0.2 * (1 - group), n)
    
    # Model has some bias
    y_pred = (np.random.rand(n) < 0.4 + 0.15 * (1 - group)).astype(int)
    
    slices = {
        "overall": np.ones(n, dtype=bool),
        "group_0": group == 0,
        "group_1": group == 1
    }
    
    report = sliced_evaluation(y_true, y_pred, slices)
    print(report)
    
    anomalies = detect_fairness_anomalies(report, "accuracy", 0.05)
    print("Anomalies:", anomalies)
```

---

## 📋 Interview Tips

1. **Lead with the bias taxonomy**: "Bias can enter at multiple stages—historical, representation, measurement, etc. We need to identify which we're dealing with to choose the right fix."
2. **Know the metrics cold**: Be able to write demographic parity, equalized odds, and calibration formulas.
3. **Acknowledge trade-offs**: "Demographic parity and equalized odds can conflict when base rates differ. We need to decide which principle matters more for our use case."
4. **Sliced evaluation is non-negotiable**: "We'll evaluate across all relevant subgroups, not just aggregate. That includes intersectional slices where sample size allows."
5. **Connect to real failures**: COMPAS, Amazon hiring tool, facial recognition disparities—reference specific cases to show depth.
6. **Proxies**: "Removing race doesn't help if zip code, name, or other features act as proxies. We need to test for disparate impact even without sensitive features."

---

## Related Topics

- [Fairness Constraints](./02-fairness-constraints.md) – How to mitigate bias once detected
- [Model Auditing](./03-model-auditing.md) – Documentation and explainability for accountability
- [Responsible Deployment](./04-responsible-deployment.md) – Monitoring and feedback loops in production
