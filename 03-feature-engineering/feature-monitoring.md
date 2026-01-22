# Feature Monitoring

## Overview

Feature monitoring is essential for maintaining ML system health. Features can degrade over time, leading to poor model performance. Continuous monitoring helps detect and fix issues early.

---

## 🎯 Why Monitor Features?

### 1. Feature Drift
- Input distribution changes
- Feature values shift
- Model performance degrades

### 2. Data Quality Issues
- Missing values increase
- Invalid values appear
- Schema changes

### 3. Pipeline Failures
- Computation errors
- Pipeline crashes
- Data source issues

### 4. Performance Issues
- Slow feature computation
- High latency
- Resource constraints

---

## 📊 What to Monitor

### 1. Feature Statistics

**Metrics:**
- Mean, median, std dev
- Min, max values
- Percentiles
- Value distributions

**Purpose:**
- Detect distribution shifts
- Identify outliers
- Track trends

**Example:**
```python
def monitor_feature_stats(feature_name, current_data, reference_data):
    # Compare statistics
    current_mean = current_data.mean()
    reference_mean = reference_data.mean()
    
    # Detect drift
    drift_score = abs(current_mean - reference_mean) / reference_mean
    if drift_score > 0.1:  # 10% threshold
        alert(f"Feature {feature_name} mean drifted by {drift_score:.2%}")
```

---

### 2. Feature Completeness

**Metrics:**
- Missing value rate
- Null percentage
- Completeness score

**Purpose:**
- Detect data quality issues
- Identify pipeline failures
- Track data availability

**Example:**
```python
def monitor_completeness(feature_name, data):
    completeness = (data.notna().sum() / len(data)) * 100
    
    if completeness < 95:  # Threshold
        alert(f"Feature {feature_name} completeness: {completeness:.2f}%")
    
    return completeness
```

---

### 3. Feature Freshness

**Metrics:**
- Last update time
- Update frequency
- Staleness

**Purpose:**
- Ensure features are current
- Detect pipeline delays
- Track update frequency

**Example:**
```python
def monitor_freshness(feature_name, last_update_time, max_age_hours=24):
    age_hours = (datetime.now() - last_update_time).total_seconds() / 3600
    
    if age_hours > max_age_hours:
        alert(f"Feature {feature_name} is stale: {age_hours:.2f} hours old")
```

---

### 4. Feature Distributions

**Metrics:**
- Distribution comparisons
- KS test statistics
- Chi-square tests

**Purpose:**
- Detect distribution shifts
- Identify concept drift
- Compare train vs inference

**Example:**
```python
from scipy.stats import ks_2samp

def monitor_distribution(feature_name, current_data, reference_data):
    # Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(reference_data, current_data)
    
    if p_value < 0.05:  # Significant difference
        alert(f"Feature {feature_name} distribution shifted (p={p_value:.4f})")
```

---

### 5. Feature Correlations

**Metrics:**
- Correlation coefficients
- Correlation changes
- Feature relationships

**Purpose:**
- Detect relationship changes
- Identify feature interactions
- Track stability

**Example:**
```python
def monitor_correlations(current_data, reference_data):
    current_corr = current_data.corr()
    reference_corr = reference_data.corr()
    
    # Compare correlations
    corr_diff = abs(current_corr - reference_corr)
    
    # Alert on significant changes
    high_diff = corr_diff[corr_diff > 0.2]
    if len(high_diff) > 0:
        alert(f"Feature correlations changed: {high_diff}")
```

---

## 🛠️ Monitoring Tools

### 1. Evidently AI

**Purpose:** ML monitoring and drift detection

**Features:**
- Data drift detection
- Model performance monitoring
- Feature monitoring
- Dashboards

**Usage:**
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Define column mapping
column_mapping = ColumnMapping(
    target=None,
    prediction=None,
    numerical_features=['age', 'total_purchases'],
    categorical_features=['category']
)

# Create report
report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=column_mapping
)

# Get results
drift_detected = report.as_dict()['metrics'][0]['result']['drift_detected']
```

---

### 2. Fiddler

**Purpose:** ML monitoring platform

**Features:**
- Data drift detection
- Model monitoring
- Explainability
- Alerting

**Pros:**
- Comprehensive platform
- Good UI
- Enterprise features

**Cons:**
- Commercial
- Less flexible

---

### 3. Custom Monitoring

**Purpose:** Build custom monitoring

**Approach:**
- Statistical tests
- Threshold-based alerts
- Custom dashboards

**Example:**
```python
class FeatureMonitor:
    def __init__(self, reference_data, thresholds):
        self.reference_data = reference_data
        self.thresholds = thresholds
    
    def check_drift(self, feature_name, current_data):
        # Statistical tests
        ks_stat, p_value = ks_2samp(
            self.reference_data[feature_name],
            current_data[feature_name]
        )
        
        # Check threshold
        if p_value < self.thresholds['p_value']:
            return {
                'drift_detected': True,
                'p_value': p_value,
                'statistic': ks_stat
            }
        return {'drift_detected': False}
    
    def check_completeness(self, feature_name, current_data):
        completeness = current_data[feature_name].notna().sum() / len(current_data)
        
        if completeness < self.thresholds['completeness']:
            return {
                'issue_detected': True,
                'completeness': completeness
            }
        return {'issue_detected': False}
```

---

## 🏗️ Monitoring Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Feature Computation                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Storage                             │
└───────┬───────────────────────────────┬─────────────────┘
        │                               │
        ▼                               ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Offline Features    │    │   Online Features         │
└───────┬──────────────┘    └───────────┬──────────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Monitoring                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Statistics  │  │   Drift      │  │   Quality    │ │
│  │  Tracking    │  │   Detection  │  │   Checks     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Alerting & Dashboards                        │
│  (Alerts, Metrics, Visualizations)                      │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Best Practices

### 1. Set Baselines
- Establish reference distributions
- Define expected ranges
- Set quality thresholds

### 2. Monitor Continuously
- Real-time monitoring for critical features
- Batch monitoring for others
- Regular reviews

### 3. Alert Appropriately
- Set meaningful thresholds
- Avoid alert fatigue
- Escalate critical issues

### 4. Track Trends
- Monitor over time
- Identify patterns
- Predict issues

### 5. Automate Responses
- Auto-retrain on drift
- Auto-fix common issues
- Auto-scale resources

---

## 🎯 Monitoring by Feature Type

### Numerical Features
- **Monitor**: Mean, std, distribution, outliers
- **Tests**: KS test, t-test, z-score
- **Thresholds**: Statistical significance

### Categorical Features
- **Monitor**: Value counts, distribution, new categories
- **Tests**: Chi-square, JS divergence
- **Thresholds**: Category frequency changes

### Time Series Features
- **Monitor**: Trends, seasonality, anomalies
- **Tests**: Stationarity tests, anomaly detection
- **Thresholds**: Trend changes, anomaly scores

---

## 🔑 Key Takeaways

1. **Monitor continuously** - detect issues early
2. **Track multiple metrics** - statistics, completeness, freshness
3. **Use appropriate tools** - Evidently AI, Fiddler, custom
4. **Set meaningful thresholds** - avoid false positives
5. **Automate responses** - fix issues automatically when possible

---

## 📚 Further Reading

- [Evidently AI Documentation](https://docs.evidently.ai/)
- [Feature Monitoring Best Practices](https://www.oreilly.com/library/view/building-machine-learning/9781492045100/)

---

## 🔗 Related Topics

- [Feature Stores](./feature-stores.md)
- [Online vs Offline Features](./online-vs-offline-features.md)
- [Feature Pipelines](./feature-pipelines.md)
