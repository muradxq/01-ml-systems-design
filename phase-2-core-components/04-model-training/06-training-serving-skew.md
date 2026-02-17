# Training-Serving Skew

## Overview

Training-serving skew occurs when the data or environment used during model training differs from what the model encounters during inference. It is one of the most common and insidious production ML bugs—models can show excellent offline metrics while failing silently in production. Unlike traditional software where bugs cause crashes, skew causes gradual degradation that's difficult to diagnose. Google's ML team reports that training-serving skew accounts for a significant portion of production ML incidents.

**Why it matters in interviews:** Interviewers love this topic because it reveals whether you understand the full ML lifecycle, not just model training. It demonstrates production experience and awareness of operational challenges.

---

## 🎯 Sources of Training-Serving Skew

### 1. Feature Computation Differences

**Problem:** Different code paths or implementations compute the same feature during training vs serving.

```
┌─────────────────────────────────────────────────────────────────┐
│  Feature Computation Skew                                        │
│                                                                  │
│  Training Pipeline:                    Serving Pipeline:        │
│  ┌─────────────────────┐               ┌─────────────────────┐  │
│  │ Python/Pandas       │               │ Java/Go/C++          │  │
│  │ sklearn.preprocess  │               │ Custom implementation│  │
│  │ Different defaults  │     ≠         │ Different rounding   │  │
│  │ Batch computation   │               │ Real-time lookup     │  │
│  └─────────────────────┘               └─────────────────────┘  │
│                                                                  │
│  Same feature name → Different values → Model confusion          │
└─────────────────────────────────────────────────────────────────┘
```

**Concrete examples:**
- **Rounding:** Training uses `round(x, 2)` in Python, serving uses `Math.floor(x * 100) / 100` in Java—produces different values for edge cases.
- **String normalization:** Training lowercases with `.lower()`, serving uses different Unicode normalization.
- **Date/time handling:** Training computes "days since signup" in local timezone; serving uses UTC—off by hours for boundary cases.
- **Feature ordering:** Training expects `[age, income, score]`; serving accidentally sends `[income, age, score]`.

---

### 2. Data Processing Bugs

**Problem:** Different preprocessing in batch (training) vs real-time (serving) pipelines.

| Training (Batch) | Serving (Real-time) | Result |
|------------------|---------------------|--------|
| Impute missing with median from full dataset | Impute with 0 or drop | Distribution shift |
| Normalize using global min/max | Normalize using request min/max | Invalid scaling |
| Apply smoothing over 30-day window | No window (single point) | Different temporal context |
| Aggregation at user level | Aggregation at session level | Mismatched granularity |

---

### 3. Time-Travel & Data Leakage

**Problem:** Using future information during training that won't be available at inference time.

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Leakage - The "Future Information" Problem                  │
│                                                                  │
│  Training (WRONG):                                               │
│  User at t=0 → Features include "purchases next 30 days"         │
│  (Model learns from future it won't have at inference)          │
│                                                                  │
│  Inference:                                                      │
│  User at t=0 → No future data available!                        │
│  Model expects features that don't exist → poor predictions     │
│                                                                  │
│  Correct: Point-in-time joins - only use data available at t    │
└─────────────────────────────────────────────────────────────────┘
```

**Examples:**
- **Churn prediction:** Including "days until churn" or post-churn behavior as features
- **Fraud detection:** Using "was_fraudulent" from investigation outcome before model runs
- **Recommendations:** Including "item was purchased" when predicting purchase probability
- **Time-window leakage:** Using 30-day rolling window that includes future data in training

---

### 4. Feature Freshness

**Problem:** Stale features in serving vs fresh features in training.

| Scenario | Training | Serving | Impact |
|----------|----------|---------|--------|
| Batch features | Computed yesterday | Served from 7-day-old snapshot | User behavior changed |
| Aggregation lag | Real-time at train time | 5-minute delayed stream | Missed recent activity |
| Embedding drift | Fresh embeddings | Cached for 24 hours | Stale item representations |
| External APIs | Success during batch | Timeout/fallback at serving | Different fallback values |

---

### 5. Feature Store Inconsistency

**Problem:** Offline store (training) and online store (serving) diverge.

```
┌─────────────────────────────────────────────────────────────────┐
│  Offline vs Online Store Divergence                               │
│                                                                  │
│  Offline Store (Training)        Online Store (Serving)          │
│  ┌──────────────────────┐       ┌──────────────────────┐        │
│  │ Parquet/Data Lake    │       │ Redis/DynamoDB       │        │
│  │ Point-in-time joins  │       │ Latest value lookup  │        │
│  │ Full history         │  ≠    │ Current state only   │        │
│  │ Batch sync (hourly)  │       │ Real-time writes     │        │
│  └──────────────────────┘       └──────────────────────┘        │
│                                                                  │
│  Sync lag, different backends, schema drift → skew              │
└─────────────────────────────────────────────────────────────────┘
```

**Common causes:**
- **Sync latency:** Offline store updated hourly; online store receives real-time updates
- **Backfill differences:** Offline has historical backfills; online only has incremental
- **Schema version mismatch:** Training uses v2 schema; serving still on v1
- **Key format:** Offline uses `user_id|timestamp`; online uses `user_id` only

---

### 6. Numerical Precision Differences

**Problem:** float32 vs float64, different libraries produce slightly different results.

| Source | Training | Serving | Example |
|--------|----------|---------|---------|
| Default precision | float64 (NumPy) | float32 (TensorFlow) | 1e-7 vs 1e-5 relative error |
| Library differences | scikit-learn StandardScaler | Custom C++ implementation | Different handling of constant features |
| Quantization | Full precision | INT8 quantized | Activation range mismatch |
| Order of operations | (a+b)+c | a+(b+c) | Floating point non-associativity |

---

### 7. Missing Data Handling

**Problem:** Different imputation strategies between training and serving.

**Training:** Drop rows with missing values (10% of data)  
**Serving:** Must predict for all requests—missing values imputed with 0, median, or mode

**Result:** Model never saw the imputation pattern during training; predictions on imputed data are unreliable.

---

## 🔍 Detection Methods

### Feature Distribution Comparison

Compare training feature distributions with serving feature distributions using statistical tests.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

@dataclass
class SkewReport:
    """Report of detected training-serving skew."""
    feature_name: str
    skew_detected: bool
    test_name: str
    statistic: float
    p_value: float
    severity: str  # 'none', 'minor', 'moderate', 'severe'
    recommendation: str

class SkewDetector:
    """
    Detects training-serving skew by comparing feature distributions.
    Run during CI/CD and/or as scheduled production monitoring.
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.1,
        ks_pvalue_threshold: float = 0.05,
        max_drift_ratio: float = 0.2
    ):
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.max_drift_ratio = max_drift_ratio
    
    def compute_psi(self, reference: np.ndarray, current: np.ndarray, 
                    n_bins: int = 10) -> float:
        """Population Stability Index - common skew metric."""
        _, bins = np.histogram(np.concatenate([reference, current]), bins=n_bins)
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        
        ref_pct = np.clip(ref_hist / len(reference), 1e-10, 1.0)
        cur_pct = np.clip(cur_hist / len(current), 1e-10, 1.0)
        
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    
    def detect_feature_skew(
        self,
        feature_name: str,
        training_values: np.ndarray,
        serving_values: np.ndarray,
        feature_type: str = 'continuous'
    ) -> SkewReport:
        """Detect skew for a single feature."""
        
        if feature_type == 'continuous':
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(training_values, serving_values)
            
            # PSI
            psi = self.compute_psi(training_values, serving_values)
            
            skew_detected = (ks_pvalue < self.ks_pvalue_threshold or 
                           psi >= self.psi_threshold)
            
            # Severity
            if psi >= 0.25:
                severity = 'severe'
                recommendation = 'Immediate investigation required'
            elif psi >= 0.15:
                severity = 'moderate'
                recommendation = 'Alert and investigate before next deploy'
            elif psi >= 0.1:
                severity = 'minor'
                recommendation = 'Monitor closely, plan fix'
            else:
                severity = 'none'
                recommendation = 'No action needed'
            
            return SkewReport(
                feature_name=feature_name,
                skew_detected=skew_detected,
                test_name='ks_test+psi',
                statistic=psi,
                p_value=ks_pvalue,
                severity=severity,
                recommendation=recommendation
            )
        else:
            # Chi-square for categorical
            categories = np.unique(np.concatenate([training_values, serving_values]))
            ref_counts = np.array([np.sum(training_values == c) for c in categories])
            cur_counts = np.array([np.sum(serving_values == c) for c in categories])
            expected = ref_counts / len(training_values) * len(serving_values)
            expected = np.clip(expected, 1e-10, None)
            
            chi2, p_value = stats.chisquare(cur_counts, expected)
            skew_detected = p_value < self.ks_pvalue_threshold
            
            return SkewReport(
                feature_name=feature_name,
                skew_detected=skew_detected,
                test_name='chi_square',
                statistic=float(chi2),
                p_value=float(p_value),
                severity='moderate' if skew_detected else 'none',
                recommendation='Check categorical encoding consistency' if skew_detected else 'OK'
            )
    
    def detect_all(
        self,
        training_features: Dict[str, np.ndarray],
        serving_features: Dict[str, np.ndarray],
        feature_types: Optional[Dict[str, str]] = None
    ) -> List[SkewReport]:
        """Detect skew across all features."""
        reports = []
        feature_types = feature_types or {}
        
        for name in training_features.keys():
            if name not in serving_features:
                reports.append(SkewReport(
                    feature_name=name, skew_detected=True,
                    test_name='missing', statistic=0, p_value=0,
                    severity='severe',
                    recommendation=f'Feature {name} missing in serving!'
                ))
                continue
            
            ftype = feature_types.get(name, 'continuous')
            report = self.detect_feature_skew(
                name,
                training_features[name],
                serving_features[name],
                ftype
            )
            reports.append(report)
        
        return reports
```

---

### Prediction Distribution Monitoring

Monitor the distribution of model predictions in production vs training validation set. Sudden shifts indicate skew.

```python
class PredictionDistributionMonitor:
    """
    Monitor prediction distribution drift - indirect skew indicator.
    If input features skew, output distribution will shift.
    """
    
    def __init__(self, reference_predictions: np.ndarray, 
                 alert_threshold_psi: float = 0.2):
        self.reference = reference_predictions
        self.alert_threshold = alert_threshold_psi
    
    def check_drift(self, current_predictions: np.ndarray) -> dict:
        psi = self._compute_psi(self.reference, current_predictions)
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference, current_predictions)
        
        return {
            'psi': psi,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'alert': psi >= self.alert_threshold,
            'mean_shift': np.mean(current_predictions) - np.mean(self.reference)
        }
    
    def _compute_psi(self, ref: np.ndarray, cur: np.ndarray, bins: int = 20) -> float:
        _, edges = np.histogram(np.concatenate([ref, cur]), bins=bins)
        r = np.histogram(ref, bins=edges)[0] / len(ref)
        c = np.histogram(cur, bins=edges)[0] / len(cur)
        r, c = np.clip(r, 1e-10, 1), np.clip(c, 1e-10, 1)
        return float(np.sum((c - r) * np.log(c / r)))
```

---

### Integration Tests: Training vs Serving Feature Consistency

```python
class FeatureConsistencyTest:
    """
    Integration test: same input → same features from training and serving code.
    Run in CI before every deploy.
    """
    
    def __init__(
        self,
        training_feature_fn,  # Function that computes features (training code path)
        serving_feature_fn,    # Function that computes features (serving code path)
        tolerance: float = 1e-5
    ):
        self.training_fn = training_feature_fn
        self.serving_fn = serving_feature_fn
        self.tolerance = tolerance
    
    def run_test(self, test_cases: List[dict]) -> dict:
        """Run consistency test on sample inputs."""
        results = {'passed': 0, 'failed': 0, 'failures': []}
        
        for i, case in enumerate(test_cases):
            train_features = self.training_fn(case)
            serve_features = self.serving_fn(case)
            
            for feat_name in train_features:
                if feat_name not in serve_features:
                    results['failed'] += 1
                    results['failures'].append({
                        'case': i, 'feature': feat_name,
                        'error': 'Missing in serving'
                    })
                    continue
                
                t_val = train_features[feat_name]
                s_val = serve_features[feat_name]
                
                if isinstance(t_val, (int, float)) and isinstance(s_val, (int, float)):
                    if abs(t_val - s_val) > self.tolerance:
                        results['failed'] += 1
                        results['failures'].append({
                            'case': i, 'feature': feat_name,
                            'training': t_val, 'serving': s_val,
                            'diff': abs(t_val - s_val)
                        })
                elif t_val != s_val:
                    results['failed'] += 1
                    results['failures'].append({
                        'case': i, 'feature': feat_name,
                        'training': str(t_val), 'serving': str(s_val)
                    })
            
            if not any(f['case'] == i for f in results['failures'][-results['failed']:]):
                results['passed'] += 1
        
        results['total'] = len(test_cases)
        results['pass_rate'] = results['passed'] / results['total'] if results['total'] > 0 else 0
        return results
```

---

### Logging Serving Features & Replaying Through Training Pipeline

```python
class TrainingServingComparator:
    """
    Log serving-time features, then replay through training pipeline.
    Identifies skew from different computation logic.
    """
    
    def __init__(self, training_pipeline, feature_log_store):
        self.training_pipeline = training_pipeline
        self.feature_log_store = feature_log_store
    
    def compare_logged_vs_training(self, request_ids: List[str]) -> dict:
        """
        For each logged request: take RAW input, run through training pipeline,
        compare computed features to what was served.
        """
        discrepancies = []
        
        for req_id in request_ids:
            log_entry = self.feature_log_store.get(req_id)
            if not log_entry:
                continue
            
            raw_input = log_entry['raw_input']
            served_features = log_entry['served_features']
            
            # Recompute with training pipeline
            training_features = self.training_pipeline.compute(raw_input)
            
            for feat in served_features:
                if feat not in training_features:
                    discrepancies.append({
                        'request_id': req_id,
                        'feature': feat,
                        'issue': 'In serving but not in training output'
                    })
                else:
                    s, t = served_features[feat], training_features[feat]
                    if isinstance(s, (int, float)) and isinstance(t, (int, float)):
                        if abs(s - t) > 1e-6:
                            discrepancies.append({
                                'request_id': req_id,
                                'feature': feat,
                                'served': s, 'training': t,
                                'diff': abs(s - t)
                            })
                    elif s != t:
                        discrepancies.append({
                            'request_id': req_id,
                            'feature': feat,
                            'served': s, 'training': t
                        })
        
        return {
            'total_checked': len(request_ids),
            'discrepancies': discrepancies,
            'discrepancy_rate': len(discrepancies) / max(len(request_ids), 1)
        }
```

---

## 🛡️ Prevention Strategies

### 1. Shared Feature Computation Code

**Principle:** One definition, two execution contexts. Same code runs in training and serving.

```
┌─────────────────────────────────────────────────────────────────┐
│  Shared Feature Code Architecture                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           feature_compute/ (shared library)              │    │
│  │  - feature_definitions.py                                 │    │
│  │  - transforms.py                                          │    │
│  │  - validation.py                                          │    │
│  └────────────────────┬────────────────────────────────────┘    │
│                       │                                          │
│         ┌─────────────┴─────────────┐                            │
│         │                           │                            │
│         ▼                           ▼                            │
│  ┌─────────────┐             ┌─────────────┐                    │
│  │  Training   │             │   Serving    │                    │
│  │  (Python)   │             │ (Python/    │                    │
│  │             │             │  Java/C++)  │                    │
│  └─────────────┘             └─────────────┘                    │
│                                                                  │
│  Same logic, same outputs, no skew                              │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:** Use Feathr, Feast, or custom shared lib. Export to multiple runtimes if needed (e.g., Python → ONNX for serving).

---

### 2. Feature Store as Single Source of Truth

- Single feature definition
- Offline and online stores populated from same pipeline
- Point-in-time correctness in offline; low-latency lookup in online
- Validated sync between stores

---

### 3. Point-in-Time Correct Training Data

Always join training labels with features using timestamp—only use data available at prediction time.

```python
# Wrong: Uses all data for user
df = user_features.join(labels)

# Correct: Point-in-time join
df = point_in_time_join(
    entities=users,
    features=feature_table,
    timestamp_col='event_time',
    feature_timestamp_col='created_at'  # Feature must be created before event
)
```

---

### 4. Serving-Time Feature Logging

Log raw inputs and computed features at serving time. Use for:
- Next training cycle (ensure training data matches production)
- Skew detection (replay through training pipeline)
- Debugging production issues

---

### 5. CI/CD with Skew Detection Tests

- Run `FeatureConsistencyTest` on every PR
- Run `SkewDetector` comparing last training run features vs recent serving sample
- Block deploy if skew detected above threshold

---

### 6. Shadow Mode for New Models

Deploy new model in shadow mode: run alongside production, compare predictions. Large discrepancies may indicate skew in new model's feature expectations.

---

## 📊 Trade-Offs Table

| Strategy | Effort | Effectiveness | Latency Impact | When to Use |
|----------|--------|---------------|----------------|-------------|
| **Shared feature code** | High | Very High | None | Greenfield, critical systems |
| **Feature store** | High | High | Low (if well-architected) | Medium+ team, many models |
| **Point-in-time joins** | Medium | Critical for leakage | None | All tabular ML |
| **Serving feature logging** | Medium | High (detection) | Minimal | All production systems |
| **Skew detection in CI** | Medium | High | None | All systems |
| **Shadow mode** | High | High (validation) | None (async) | Major model changes |
| **Integration tests** | Low-Medium | Medium | None | All systems |

---

## 📖 Real-World Examples

### Case 1: Recommendation Model at Scale

**Symptom:** Model showed 15% CTR improvement offline, 0% in A/B test.

**Root cause:** Training used batch-computed "user_engagement_score" (24h lag). Serving used real-time score. User behavior shifted within 24 hours—training saw stale data, serving saw fresh. Model learned patterns that didn't generalize.

**Fix:** Aligned training to use same freshness as serving (or accepted lag and retrained with lagged labels).

---

### Case 2: Fraud Detection

**Symptom:** Production model had higher false positive rate than validation.

**Root cause:** Training dropped rows with missing "device_fingerprint." In production, 5% of requests had missing fingerprint (new devices, API clients). Model never learned to handle missing; defaulted to suspicious.

**Fix:** Retrained with explicit missing-value handling matching serving (e.g., separate "missing" bucket or learned imputation).

---

### Case 3: Ad Click Prediction

**Symptom:** Predictions drifted downward over weeks with no code changes.

**Root cause:** Feature store offline/online sync had 6-hour lag. Training used offline (fresh). Serving used online (stale for some features). As sync lag grew during scale-up, skew increased.

**Fix:** Reduced sync lag, added skew monitoring on high-impact features.

---

## 💡 Interview Tips

### Why Interviewers Love This Topic

1. **Reveals production experience** — Candidates who've debugged skew have battle scars
2. **Tests systems thinking** — Not just "train a model" but "deploy and maintain"
3. **Common interview question** — "Your model works offline but fails in production. Why?"
4. **No single answer** — Tests ability to reason through multiple possibilities

### How to Bring It Up Proactively

- When discussing **feature engineering:** "We need to ensure feature computation is identical in training and serving—training-serving skew is a major risk."
- When discussing **deployment:** "Before deploying, we run skew detection between our last training run and a sample of production requests."
- When discussing **monitoring:** "Beyond accuracy, we monitor feature distributions and prediction distributions for skew."
- When discussing **feature stores:** "The main benefit is eliminating offline/online skew by having one source of truth."

### Sample Interview Answer Framework

**Q: "Your model has great offline metrics but performs poorly in production. How do you debug?"**

**A:** "I'd systematically check for training-serving skew:

1. **Feature consistency** — Log production features and compare distributions to training. Use PSI or KS tests. Replay raw requests through training pipeline—do we get same features?
2. **Data leakage** — Audit training data for future information. Are we using point-in-time joins?
3. **Freshness** — Are we serving stale features? Is offline/online store in sync?
4. **Code paths** — Is the same code computing features in training vs serving? Different languages or libraries?
5. **Missing data** — How does serving handle missing values vs training?

I'd also check for concept drift and evaluate whether the problem is skew vs the world actually changing."

---

## 🔗 Related Topics

- [Feature Stores](../03-feature-engineering/01-feature-stores.md) - Single source of truth for features
- [Online vs Offline Features](../03-feature-engineering/02-online-vs-offline-features.md) - Feature computation contexts
- [Data Drift Detection](../../phase-3-operations-and-reliability/06-monitoring-observability/02-data-drift-detection.md) - Statistical drift monitoring
- [Model Monitoring](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md) - Production monitoring
- [Shadow Mode](../05-model-serving/04-model-updates.md) - Safe model validation
- [Feature Monitoring](../03-feature-engineering/04-feature-monitoring.md) - Feature quality and consistency
