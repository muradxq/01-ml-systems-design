# Fairness Constraints

## Overview

Once bias is detected, the next step is intervention. Fairness constraints can be applied at three stages: **pre-processing** (data), **in-processing** (training), and **post-processing** (predictions). Each stage has trade-offs in terms of flexibility, control, and compatibility with existing pipelines. This topic covers techniques at each stage, Python implementations, the fairness-accuracy Pareto frontier, and when to use which approach.

---

## 🎯 Three Stages of Intervention

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    FAIRNESS INTERVENTION POINTS                                           │
│                                                                                          │
│   DATA                  TRAINING                PREDICTIONS                              │
│   ┌─────────┐           ┌─────────┐            ┌─────────┐                               │
│   │ Raw     │  ───────▶ │ Model   │  ───────▶  │ Output  │                               │
│   │ Data    │           │ Training│            │ Scores  │                               │
│   └─────────┘           └─────────┘            └─────────┘                               │
│        │                      │                      │                                   │
│        ▼                      ▼                      ▼                                   │
│   PRE-PROCESSING        IN-PROCESSING         POST-PROCESSING                           │
│   • Resampling          • Constrained         • Threshold                                │
│   • Reweighting           optimization         adjustment                               │
│   • Augmentation        • Adversarial         • Calibration                             │
│   • Remove sensitive      debiasing           • Reject option                            │
│     (limited value)     • Fairness reg.         classification                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Stage 1: Pre-Processing

**Goal**: Modify the training data so that the model learns a fairer representation, without changing the model architecture.

### 1.1 Resampling

**Oversampling**: Duplicate examples from underrepresented groups to balance the dataset.
**Undersampling**: Remove examples from overrepresented groups.

| Strategy | Pros | Cons |
|----------|------|------|
| **Oversample minority** | Preserves all data | Can overfit to minority patterns; duplicates don't add information |
| **Undersample majority** | Faster training | Loses data; may hurt overall accuracy |
| **SMOTE (Synthetic Minority Oversampling)** | Creates synthetic examples | Can create unrealistic points; doesn't address label bias |

```python
import numpy as np
from collections import Counter

def resample_for_parity(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    strategy: str = "oversample"
) -> tuple:
    """
    Resample so that positive rate per group is balanced.
    
    strategy: 'oversample' | 'undersample' | 'hybrid'
    """
    n = len(y)
    indices = np.arange(n)
    
    # Group indices by (group, label)
    groups_idx = {}
    for g in np.unique(group):
        for lb in [0, 1]:
            mask = (group == g) & (y == lb)
            groups_idx[(g, lb)] = indices[mask]
    
    # Target: same number per (group, label) as max
    if strategy == "oversample":
        max_per = max(len(v) for v in groups_idx.values())
        new_idx = []
        for k, idx in groups_idx.items():
            if len(idx) < max_per:
                # Oversample with replacement
                add = np.random.choice(idx, max_per - len(idx), replace=True)
                new_idx.extend(idx.tolist() + add.tolist())
            else:
                new_idx.extend(idx.tolist())
    elif strategy == "undersample":
        min_per = min(len(v) for v in groups_idx.values())
        new_idx = []
        for idx in groups_idx.values():
            sel = np.random.choice(idx, min(min_per, len(idx)), replace=False)
            new_idx.extend(sel.tolist())
    else:
        new_idx = indices.tolist()
    
    new_idx = np.array(new_idx)
    np.random.shuffle(new_idx)
    return X[new_idx], y[new_idx], group[new_idx]
```

---

### 1.2 Reweighting

**Idea**: Assign higher weights to underrepresented (group, label) combinations so the model pays more attention to them during training.

**Formula**: For group $a$ and label $y$, weight $w_{a,y} \propto \frac{1}{P(A=a, Y=y)}$ (inverse frequency), normalized.

```python
def compute_fairness_weights(
    y: np.ndarray,
    group: np.ndarray
) -> np.ndarray:
    """
    Compute sample weights for fairness (inverse frequency weighting).
    """
    n = len(y)
    weights = np.ones(n)
    
    for g in np.unique(group):
        for lb in [0, 1]:
            mask = (group == g) & (y == lb)
            count = mask.sum()
            if count > 0:
                # Weight inversely proportional to frequency
                weights[mask] = n / (4 * count)  # 4 = 2 groups * 2 labels
    
    return weights


# Usage with sklearn
from sklearn.linear_model import LogisticRegression

def train_fair_logistic(X, y, group):
    weights = compute_fairness_weights(y, group)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y, sample_weight=weights)
    return model
```

---

### 1.3 Data Augmentation

**Idea**: Generate synthetic examples for underrepresented groups (e.g., counterfactual augmentation—flip protected attribute and create "counterfactual" examples with adjusted labels).

**Caution**: Requires care—synthetic data can introduce new bias if the augmentation is poorly designed.

---

### 1.4 Removing Sensitive Features

**Idea**: Drop protected attributes (race, gender, etc.) from features.

**Why it often fails**:
- **Proxies**: Zip code, name, school, language—all correlate with protected attributes
- **Indirect encoding**: The model can infer protected attributes from other features
- **No guarantee**: Fairness is not achieved by blindness—you must explicitly measure and constrain

| Approach | Result |
|----------|--------|
| Remove race only | Model uses zip code, name, etc. → proxy discrimination |
| Remove all possible proxies | May lose legitimate signal; model may still find new proxies |
| Remove + constrain | Better—use removal as one step, but also enforce fairness metrics |

**Interview tip**: "Removing sensitive features is necessary but not sufficient. We need to test for disparate impact and use constraints or post-processing to achieve fairness."

---

## 📦 Stage 2: In-Processing

**Goal**: Modify the training objective or procedure so the model inherently satisfies fairness constraints.

### 2.1 Constrained Optimization

**Idea**: Add fairness as a constraint to the optimization problem.

$$\min_\theta \mathcal{L}(\theta) \quad \text{s.t.} \quad \text{parity}(\theta) \geq \tau$$

**Implementation**: Use Lagrangian relaxation—convert to $\min_\theta \mathcal{L}(\theta) + \lambda \cdot \text{parity\_violation}(\theta)$ and tune $\lambda$.

---

### 2.2 Adversarial Debiasing

**Idea**: Train the predictor to be good at the task while an adversary tries to predict the protected attribute from the model's representations. The predictor learns to hide information about the protected attribute.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Adversarial Debiasing Architecture             │
│                                                                   │
│   Input X ──▶ Predictor ──▶ Prediction ŷ (task)                  │
│                   │                                               │
│                   │   Representation h                            │
│                   │         │                                      │
│                   │         ▼                                      │
│                   │   Adversary ──▶ Â (predict protected attr)    │
│                   │         │                                       │
│                   │   Loss: Predictor minimizes task loss          │
│                   │         Adversary maximizes Â accuracy         │
│                   │         Predictor ALSO tries to fool Adversary │
│                   │         (gradient reversal)                    │
└─────────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import numpy as np

class GradientReversal(torch.autograd.Function):
    """Reverse gradient in backward pass."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class AdversarialDebiasing(nn.Module):
    """
    Predictor + Adversary for debiasing.
    Predictor: X -> y (main task)
    Adversary: h(X) -> protected (tries to predict A from representations)
    Predictor gets gradient reversed when updating representations to fool adversary.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_groups: int = 2):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups)
        )
        self.repr_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.alpha = 1.0
    
    def forward(self, x, return_repr=False):
        h = self.repr_layer(x)
        h_rev = GradientReversal.apply(h, self.alpha)
        
        pred = self.predictor(h)
        adv_pred = self.adversary(h_rev)
        
        if return_repr:
            return pred, adv_pred, h
        return pred, adv_pred
    
    def set_alpha(self, alpha: float):
        self.alpha = alpha


def train_adversarial_debiasing(
    model: AdversarialDebiasing,
    X: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01
):
    opt_pred = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    opt_adv = torch.optim.Adam(model.adversary.parameters(), lr=lr)
    
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        
        # Forward
        pred, adv_pred = model(X)
        loss_task = bce(pred.squeeze(), y.float())
        loss_adv = ce(adv_pred, a.long())
        
        # Update adversary
        opt_adv.zero_grad()
        loss_adv.backward(retain_graph=True)
        opt_adv.step()
        
        # Update predictor (task loss - adversary success, to fool adversary)
        opt_pred.zero_grad()
        (-loss_adv + loss_task).backward()
        opt_pred.step()
        
        # Gradually increase alpha (gradient reversal strength)
        model.set_alpha(min(1.0, 0.01 + epoch / 100))
    
    return model
```

---

### 2.3 Fairness Regularization

**Idea**: Add a penalty term for fairness violation to the loss.

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{fair}$$

Examples of $\mathcal{L}_{fair}$:
- **Demographic parity penalty**: $\left( P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1) \right)^2$
- **Equalized odds penalty**: Sum of (TPR difference)² and (FPR difference)²

```python
def demographic_parity_loss(y_pred: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Penalty for demographic parity violation."""
    g0 = (group == 0)
    g1 = (group == 1)
    
    if g0.sum() == 0 or g1.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    
    rate_0 = y_pred[g0].mean()
    rate_1 = y_pred[g1].mean()
    return (rate_0 - rate_1).pow(2)


# In training loop:
# loss = bce(pred, y) + lambda_fair * demographic_parity_loss(pred, group)
```

---

### 2.4 Multi-Objective Optimization

**Idea**: Treat accuracy and fairness as multiple objectives; find Pareto-optimal models.

**Approach**: Train multiple models with different $\lambda$ values; select from the Pareto frontier based on product requirements.

---

## 📦 Stage 3: Post-Processing

**Goal**: Adjust model outputs (predictions or scores) after inference to satisfy fairness constraints. No retraining required.

### 3.1 Threshold Adjustment Per Group

**Idea**: Use different classification thresholds for different groups so that positive rates (or TPR/FPR) are equalized.

```python
import numpy as np
from typing import Tuple

def find_fair_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group: np.ndarray,
    target_metric: str = "demographic_parity",
    target_value: float = None
) -> Tuple[float, float]:
    """
    Find thresholds (tau_0, tau_1) for group 0 and 1 to achieve fairness.
    
    target_metric: 'demographic_parity' | 'equalized_odds'
    """
    g0 = group == 0
    g1 = group == 1
    
    scores_0 = y_score[g0]
    scores_1 = y_score[g1]
    
    # Grid search over possible thresholds
    best_tau0, best_tau1 = 0.5, 0.5
    best_violation = float('inf')
    
    for tau0 in np.linspace(0.1, 0.9, 17):
        for tau1 in np.linspace(0.1, 0.9, 17):
            pred_0 = (scores_0 >= tau0).astype(int)
            pred_1 = (scores_1 >= tau1).astype(int)
            
            rate_0 = pred_0.mean()
            rate_1 = pred_1.mean()
            
            if target_metric == "demographic_parity":
                violation = abs(rate_0 - rate_1)
            elif target_metric == "equalized_odds":
                tpr_0 = pred_0[y_true[g0] == 1].mean() if (y_true[g0] == 1).any() else 0
                tpr_1 = pred_1[y_true[g1] == 1].mean() if (y_true[g1] == 1).any() else 0
                fpr_0 = pred_0[y_true[g0] == 0].mean() if (y_true[g0] == 0).any() else 0
                fpr_1 = pred_1[y_true[g1] == 0].mean() if (y_true[g1] == 0).any() else 0
                violation = abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)
            else:
                violation = abs(rate_0 - rate_1)
            
            if violation < best_violation:
                best_violation = violation
                best_tau0, best_tau1 = tau0, tau1
    
    return best_tau0, best_tau1


def apply_group_thresholds(
    y_score: np.ndarray,
    group: np.ndarray,
    tau_0: float,
    tau_1: float
) -> np.ndarray:
    """Apply group-specific thresholds to scores."""
    pred = np.zeros_like(y_score)
    pred[group == 0] = (y_score[group == 0] >= tau_0).astype(int)
    pred[group == 1] = (y_score[group == 1] >= tau_1).astype(int)
    return pred
```

---

### 3.2 Calibration

**Idea**: Calibrate scores so that a score of 0.7 means 70% probability of positive outcome for all groups. Use Platt scaling or isotonic regression per group.

---

### 3.3 Reject Option Classification

**Idea**: For predictions near the decision boundary (e.g., score in [0.4, 0.6]), give the favorable outcome to the protected group. Reduces disparity by being "generous" in uncertain cases.

```python
def reject_option_classification(
    y_score: np.ndarray,
    group: np.ndarray,
    protected_group_id: int,
    margin: float = 0.1
) -> np.ndarray:
    """
    For scores in [0.5-margin, 0.5+margin], assign positive to protected group.
    """
    pred = (y_score >= 0.5).astype(int)
    uncertain = (y_score >= 0.5 - margin) & (y_score <= 0.5 + margin)
    protected = (group == protected_group_id)
    
    # In uncertain region: give benefit to protected
    pred[uncertain & protected] = 1
    pred[uncertain & ~protected] = 0
    
    return pred
```

---

## 📊 Fairness-Accuracy Trade-off: Pareto Frontier

```
                    Accuracy
                        ▲
                        │
                  0.90  │     ● Unconstrained (high acc, low fairness)
                        │    /
                        │   /
                  0.85  │  ●──●──●  Pareto-optimal models
                        │ /
                        │/
                  0.80  ● Constrained (lower acc, higher fairness)
                        │
                        └─────────────────────────────────▶ Fairness
                              0.6    0.8    1.0  (parity ratio)
```

**Process**:
1. Train models with varying $\lambda$ (or different post-processing thresholds).
2. Plot (fairness metric, accuracy) for each.
3. Pareto frontier = models where you can't improve one without harming the other.
4. Product/business chooses the operating point: e.g., "We need parity ratio ≥ 0.9; pick the highest-accuracy model that meets that."

---

## 📋 When to Use Which Approach

| Situation | Recommended Approach | Rationale |
|-----------|---------------------|-----------|
| **Limited control over training** | Post-processing | Can be applied to any black-box model |
| **Need interpretability** | Pre-processing (reweighting) | Simple; doesn't change model logic |
| **High-stakes, need strong guarantees** | In-processing (constrained opt) | Direct control over fairness-accuracy trade-off |
| **Black-box model (e.g., neural net)** | Adversarial debiasing or post-processing | Compatible with complex models |
| **Small dataset** | Pre-processing (reweighting) | Doesn't reduce data like undersampling |
| **Large dataset, representation bias** | Pre-processing (resampling) | Balance the data |
| **Multiple protected attributes** | In-processing | Post-processing becomes combinatorially complex |

---

## 📋 Trade-offs Summary

| Approach | Pros | Cons |
|----------|------|------|
| **Pre-processing** | Model-agnostic; easy to add to pipeline | Doesn't guarantee fairness; can hurt accuracy |
| **In-processing** | Direct optimization; flexible | Requires model access; can be complex |
| **Post-processing** | No retraining; works with any model | Needs group membership at inference; can feel "unfair" (different thresholds) |

---

## 📋 Interview Tips

1. **"We'd use a combination"**: "Pre-processing to balance data, in-processing for constrained optimization if we have control, and post-processing as a fallback or for quick fixes."
2. **Acknowledge the accuracy cost**: "Fairness constraints typically reduce accuracy. We'll measure the Pareto frontier and set a fairness floor—e.g., parity ratio ≥ 0.8—then pick the most accurate model that meets it."
3. **Post-processing and group membership**: "Post-processing requires knowing the protected attribute at inference, which has privacy and legal implications. We need to ensure we're compliant."
4. **Adversarial debiasing**: "The predictor learns representations that are predictive of the task but uninformative for the adversary. Gradient reversal is the key trick."
5. **Reject option**: "Useful when we have high uncertainty—give the benefit of the doubt to the protected group. Works best when the uncertain region is meaningful."

---

## Related Topics

- [Bias Detection](./01-bias-detection.md) – Identify bias before applying constraints
- [Model Auditing](./03-model-auditing.md) – Verify that constraints are satisfied in production
- [Responsible Deployment](./04-responsible-deployment.md) – Monitor fairness over time post-deployment
