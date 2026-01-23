# Data Drift Detection

## Overview

Data drift occurs when the statistical properties of input data change over time, potentially degrading model performance. Unlike software bugs that are introduced by code changes, drift happens naturally as the world changes. Early detection enables proactive model updates before business metrics are impacted.

---

## 🎯 Types of Drift

### 1. Data Drift (Covariate Shift)

**Definition:** Input feature distributions change, but the relationship between features and target remains the same.

**Examples:**
- User demographics shift (younger users start using app)
- Seasonal changes (holiday shopping patterns)
- New data sources with different characteristics

**Impact:** May not immediately affect accuracy, but model sees unfamiliar inputs.

```
Reference: P(X) → Current: P'(X)   [P(Y|X) unchanged]
```

### 2. Concept Drift

**Definition:** The relationship between features and target changes over time.

**Examples:**
- User preferences change (fashion trends)
- Fraud patterns evolve (new attack vectors)
- Economic conditions shift (recession affects purchase behavior)

**Impact:** Model becomes systematically wrong, even on familiar inputs.

```
Reference: P(Y|X) → Current: P'(Y|X)   [Different relationship]
```

### 3. Label Drift

**Definition:** The distribution of target labels changes over time.

**Examples:**
- Fraud rate increases/decreases
- Customer churn rate changes
- Class imbalance shifts

**Impact:** Thresholds and decision boundaries may need adjustment.

```
Reference: P(Y) → Current: P'(Y)   [Label distribution changes]
```

### 4. Prediction Drift

**Definition:** Model output distribution changes over time.

**Examples:**
- Model predicts higher/lower scores on average
- Confidence distribution shifts
- Class prediction ratios change

**Impact:** May indicate underlying data or concept drift.

---

## 📊 Drift Detection Methods

### Statistical Tests

| Test | Feature Type | What It Measures |
|------|--------------|------------------|
| **Kolmogorov-Smirnov (KS)** | Continuous | Maximum difference between CDFs |
| **Chi-Square** | Categorical | Difference in category frequencies |
| **Population Stability Index (PSI)** | Both | Shift in population distribution |
| **Mann-Whitney U** | Continuous | Difference in rankings |
| **T-Test** | Continuous | Difference in means |
| **Wasserstein Distance** | Continuous | "Earth mover's" distance |

### Implementation

```python
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

class DriftDetector:
    """Comprehensive drift detection for ML features."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov test for continuous features.
        Good for detecting any distribution change.
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            'test': 'ks_test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': p_value < self.significance_level
        }
    
    def chi_square_test(self, reference: np.ndarray, 
                        current: np.ndarray) -> Dict[str, Any]:
        """
        Chi-square test for categorical features.
        Compares observed vs expected frequencies.
        """
        # Get unique categories from both
        categories = np.unique(np.concatenate([reference, current]))
        
        # Count frequencies
        ref_counts = np.array([np.sum(reference == c) for c in categories])
        cur_counts = np.array([np.sum(current == c) for c in categories])
        
        # Normalize to expected frequencies
        ref_freq = ref_counts / len(reference)
        expected = ref_freq * len(current)
        
        # Handle zero expected frequencies
        mask = expected > 0
        statistic, p_value = stats.chisquare(
            cur_counts[mask], 
            expected[mask]
        )
        
        return {
            'test': 'chi_square',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': p_value < self.significance_level
        }
    
    def psi(self, reference: np.ndarray, current: np.ndarray, 
            n_bins: int = 10) -> Dict[str, Any]:
        """
        Population Stability Index.
        Common in finance, interpretable thresholds:
        - PSI < 0.1: No significant drift
        - 0.1 <= PSI < 0.2: Moderate drift
        - PSI >= 0.2: Significant drift
        """
        # Create bins from reference data
        _, bins = np.histogram(reference, bins=n_bins)
        
        # Calculate frequencies in each bin
        ref_freq, _ = np.histogram(reference, bins=bins)
        cur_freq, _ = np.histogram(current, bins=bins)
        
        # Normalize
        ref_freq = ref_freq / len(reference)
        cur_freq = cur_freq / len(current)
        
        # Avoid division by zero
        ref_freq = np.clip(ref_freq, 1e-10, 1)
        cur_freq = np.clip(cur_freq, 1e-10, 1)
        
        # Calculate PSI
        psi_value = np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq))
        
        return {
            'test': 'psi',
            'statistic': float(psi_value),
            'drift_detected': psi_value >= 0.1,
            'severity': 'none' if psi_value < 0.1 else 
                       'moderate' if psi_value < 0.2 else 'significant'
        }
    
    def wasserstein_distance(self, reference: np.ndarray, 
                            current: np.ndarray) -> Dict[str, Any]:
        """
        Wasserstein (Earth Mover's) distance.
        Measures the "work" needed to transform one distribution into another.
        """
        distance = stats.wasserstein_distance(reference, current)
        
        # Normalize by reference std for interpretability
        ref_std = np.std(reference)
        normalized_distance = distance / ref_std if ref_std > 0 else distance
        
        return {
            'test': 'wasserstein',
            'distance': float(distance),
            'normalized_distance': float(normalized_distance),
            'drift_detected': normalized_distance > 0.1  # Threshold
        }
    
    def detect_drift(self, reference: np.ndarray, current: np.ndarray,
                    feature_type: str = 'continuous') -> Dict[str, Any]:
        """Run appropriate drift tests based on feature type."""
        if feature_type == 'continuous':
            return {
                'ks_test': self.ks_test(reference, current),
                'psi': self.psi(reference, current),
                'wasserstein': self.wasserstein_distance(reference, current)
            }
        else:  # categorical
            return {
                'chi_square': self.chi_square_test(reference, current),
                'psi': self.psi(reference, current)
            }

# Usage example
detector = DriftDetector(significance_level=0.05)

# Compare reference (training) and current (production) data
reference_data = np.load('reference_features.npy')
current_data = get_recent_production_features()

for feature_name in feature_columns:
    result = detector.detect_drift(
        reference_data[feature_name],
        current_data[feature_name],
        feature_type='continuous'
    )
    
    if any(test.get('drift_detected', False) for test in result.values()):
        alert_drift(feature_name, result)
```

### Using Evidently AI

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric
)
import pandas as pd

def create_drift_report(reference_df: pd.DataFrame, 
                       current_df: pd.DataFrame,
                       column_mapping: dict = None) -> dict:
    """
    Create comprehensive drift report using Evidently.
    """
    # Data drift report
    data_drift_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable()
    ])
    
    data_drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    
    # Extract results
    results = data_drift_report.as_dict()
    
    # Parse drift results
    drift_summary = {
        'dataset_drift': results['metrics'][0]['result']['dataset_drift'],
        'drift_share': results['metrics'][0]['result']['drift_share'],
        'drifted_features': []
    }
    
    # Get per-feature drift
    for col, drift_data in results['metrics'][1]['result']['drift_by_columns'].items():
        if drift_data['drift_detected']:
            drift_summary['drifted_features'].append({
                'column': col,
                'drift_score': drift_data['drift_score'],
                'stattest': drift_data['stattest_name']
            })
    
    return drift_summary

# Scheduled drift check
def scheduled_drift_check():
    """Run as daily/hourly job."""
    # Load reference data (from training)
    reference_df = load_reference_data()
    
    # Get recent production data
    current_df = get_recent_production_data(hours=24)
    
    # Run drift detection
    drift_results = create_drift_report(reference_df, current_df)
    
    # Take action
    if drift_results['dataset_drift']:
        send_alert(
            severity='warning',
            message=f"Data drift detected: {drift_results['drift_share']:.1%} of features drifted",
            details=drift_results
        )
        
        if drift_results['drift_share'] > 0.5:
            trigger_model_retraining()
    
    # Log results
    log_drift_metrics(drift_results)
```

---

## 🏗️ Drift Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Data Stream                        │
│  (Features from model serving)                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Aggregation                              │
│  - Hourly/Daily windows                                          │
│  - Feature statistics                                            │
│  - Sample storage                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Statistical     │ │  ML-based    │ │  Rule-based      │
│  Tests           │ │  Detection   │ │  Checks          │
│  - KS test       │ │  - Drift     │ │  - Range checks  │
│  - PSI           │ │    classifer │ │  - Null rates    │
│  - Chi-square    │ │  - Anomaly   │ │  - Cardinality   │
└──────────────────┘ └──────────────┘ └──────────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reference Data                                │
│  (Training data statistics, historical baselines)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Drift Decision Engine                         │
│  - Aggregate signals                                             │
│  - Apply thresholds                                              │
│  - Generate alerts                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Alerts          │ │  Dashboards  │ │  Auto Actions    │
│  (PagerDuty)     │ │  (Grafana)   │ │  (Retrain)       │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

---

## 🎯 When to Take Action

| Drift Level | PSI Value | % Features Drifted | Action |
|-------------|-----------|-------------------|--------|
| **None** | <0.1 | <10% | Monitor |
| **Minor** | 0.1-0.2 | 10-25% | Investigate |
| **Moderate** | 0.2-0.3 | 25-50% | Alert, investigate urgently |
| **Severe** | >0.3 | >50% | Alert, consider retraining |

### Response Flowchart

```
Drift Detected?
    │
    ├─ No → Continue monitoring
    │
    └─ Yes → Is model performance degraded?
              │
              ├─ No → Log and monitor closely
              │
              └─ Yes → Is it fixable with data update?
                        │
                        ├─ Yes → Update features/retrain
                        │
                        └─ No → Investigate root cause
                                 │
                                 ├─ Concept drift → Major retrain/redesign
                                 │
                                 └─ Data quality issue → Fix data pipeline
```

---

## ✅ Best Practices

1. **Establish baselines** - Store reference distributions from training data
2. **Monitor continuously** - Real-time or at least hourly/daily checks
3. **Use multiple methods** - Combine statistical tests for robustness
4. **Set appropriate thresholds** - Tune based on your use case
5. **Correlate with performance** - Drift without performance drop may be acceptable
6. **Automate responses** - Trigger retraining pipelines automatically
7. **Track drift history** - Understand patterns over time
8. **Monitor high-importance features** - Focus on features with high model impact

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Ignoring expected drift** | Alert fatigue from seasonal patterns | Seasonal baselines |
| **Single test reliance** | Missing drift caught by other tests | Multiple tests |
| **Static thresholds** | Too sensitive or too lenient | Adaptive thresholds |
| **No root cause analysis** | Retraining doesn't fix issue | Investigate before retraining |
| **Delayed detection** | Drift detected too late | Increase monitoring frequency |

---

## 🔗 Related Topics

- [Model Monitoring](./model-monitoring.md) - Overall model health monitoring
- [Performance Metrics](./performance-metrics.md) - Measuring model performance
- [Alerting Systems](./alerting-systems.md) - Setting up drift alerts
