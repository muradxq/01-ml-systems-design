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

## 🛠️ Production Feature Monitoring System

### Complete Monitoring Implementation

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import json

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MonitoringAlert:
    feature_name: str
    metric_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FeatureMetrics:
    feature_name: str
    timestamp: datetime
    completeness: float
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    p5: float
    p95: float
    unique_count: int
    null_count: int
    total_count: int

class FeatureMonitoringService:
    """Production feature monitoring with drift detection and alerting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reference_stats: Dict[str, FeatureMetrics] = {}
        self.alert_handlers: List[Callable] = []
        self.metrics_store = []
        
    def set_reference(self, reference_df: pd.DataFrame):
        """Set reference statistics for drift detection."""
        for column in reference_df.columns:
            if reference_df[column].dtype in ['int64', 'float64']:
                self.reference_stats[column] = self._compute_metrics(
                    reference_df[column], column
                )
    
    def monitor_features(self, current_df: pd.DataFrame) -> Dict:
        """Run comprehensive feature monitoring."""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'features': {},
            'alerts': [],
            'summary': {}
        }
        
        alerts = []
        
        for column in current_df.columns:
            if column not in self.reference_stats:
                continue
            
            # Compute current metrics
            current_metrics = self._compute_metrics(current_df[column], column)
            reference_metrics = self.reference_stats[column]
            
            # Run all checks
            feature_alerts = []
            
            # 1. Completeness check
            completeness_alert = self._check_completeness(
                current_metrics, column
            )
            if completeness_alert:
                feature_alerts.append(completeness_alert)
            
            # 2. Distribution drift check
            drift_alert = self._check_distribution_drift(
                current_df[column], column
            )
            if drift_alert:
                feature_alerts.append(drift_alert)
            
            # 3. Statistics drift check
            stats_alerts = self._check_statistics_drift(
                current_metrics, reference_metrics, column
            )
            feature_alerts.extend(stats_alerts)
            
            # 4. Range check
            range_alert = self._check_range(
                current_metrics, reference_metrics, column
            )
            if range_alert:
                feature_alerts.append(range_alert)
            
            # Store results
            results['features'][column] = {
                'current_metrics': vars(current_metrics),
                'reference_metrics': vars(reference_metrics),
                'alerts': [vars(a) for a in feature_alerts],
                'drift_detected': len(feature_alerts) > 0
            }
            
            alerts.extend(feature_alerts)
        
        # Summary
        results['alerts'] = [vars(a) for a in alerts]
        results['summary'] = {
            'total_features': len(current_df.columns),
            'monitored_features': len(self.reference_stats),
            'features_with_alerts': len(set(a.feature_name for a in alerts)),
            'total_alerts': len(alerts),
            'critical_alerts': sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL),
            'health_score': 1 - (len(alerts) / max(len(self.reference_stats), 1))
        }
        
        # Trigger alert handlers
        for alert in alerts:
            self._trigger_alerts(alert)
        
        # Store metrics
        self.metrics_store.append(results)
        
        return results
    
    def _compute_metrics(self, series: pd.Series, feature_name: str) -> FeatureMetrics:
        """Compute comprehensive metrics for a feature."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        return FeatureMetrics(
            feature_name=feature_name,
            timestamp=datetime.utcnow(),
            completeness=numeric_series.notna().mean(),
            mean=numeric_series.mean() if numeric_series.notna().any() else np.nan,
            std=numeric_series.std() if numeric_series.notna().any() else np.nan,
            min_val=numeric_series.min() if numeric_series.notna().any() else np.nan,
            max_val=numeric_series.max() if numeric_series.notna().any() else np.nan,
            median=numeric_series.median() if numeric_series.notna().any() else np.nan,
            p5=numeric_series.quantile(0.05) if numeric_series.notna().any() else np.nan,
            p95=numeric_series.quantile(0.95) if numeric_series.notna().any() else np.nan,
            unique_count=numeric_series.nunique(),
            null_count=int(numeric_series.isna().sum()),
            total_count=len(numeric_series)
        )
    
    def _check_completeness(self, metrics: FeatureMetrics, 
                           feature_name: str) -> Optional[MonitoringAlert]:
        """Check feature completeness."""
        threshold = self.config.get('completeness_threshold', 0.95)
        
        if metrics.completeness < threshold:
            severity = AlertSeverity.CRITICAL if metrics.completeness < 0.8 else AlertSeverity.WARNING
            return MonitoringAlert(
                feature_name=feature_name,
                metric_name='completeness',
                severity=severity,
                message=f"Feature {feature_name} completeness {metrics.completeness:.2%} below threshold {threshold:.2%}",
                current_value=metrics.completeness,
                threshold=threshold
            )
        return None
    
    def _check_distribution_drift(self, current_series: pd.Series,
                                  feature_name: str) -> Optional[MonitoringAlert]:
        """Check for distribution drift using statistical tests."""
        if feature_name not in self.reference_stats:
            return None
        
        # Get reference data (in production, you'd store this)
        # For this example, we use stored statistics
        reference = self.reference_stats[feature_name]
        current = current_series.dropna()
        
        if len(current) < 30:  # Not enough samples
            return None
        
        # PSI (Population Stability Index) calculation
        psi = self._calculate_psi(reference, current)
        
        threshold = self.config.get('psi_threshold', 0.2)
        
        if psi > threshold:
            severity = AlertSeverity.CRITICAL if psi > 0.25 else AlertSeverity.WARNING
            return MonitoringAlert(
                feature_name=feature_name,
                metric_name='psi',
                severity=severity,
                message=f"Distribution drift detected for {feature_name}: PSI={psi:.4f}",
                current_value=psi,
                threshold=threshold
            )
        return None
    
    def _calculate_psi(self, reference: FeatureMetrics, 
                       current: pd.Series) -> float:
        """Calculate Population Stability Index."""
        # Simplified PSI calculation
        # In production, you'd bin the distributions properly
        n_bins = 10
        
        # Create bins based on reference percentiles
        bins = np.linspace(reference.min_val, reference.max_val, n_bins + 1)
        
        # Calculate expected and actual percentages
        current_hist, _ = np.histogram(current, bins=bins)
        current_pct = current_hist / len(current)
        
        # Expected (uniform for simplicity, use actual reference in production)
        expected_pct = np.ones(n_bins) / n_bins
        
        # Calculate PSI
        psi = np.sum((current_pct - expected_pct) * np.log(
            (current_pct + 0.0001) / (expected_pct + 0.0001)
        ))
        
        return abs(psi)
    
    def _check_statistics_drift(self, current: FeatureMetrics,
                               reference: FeatureMetrics,
                               feature_name: str) -> List[MonitoringAlert]:
        """Check for drift in statistical measures."""
        alerts = []
        
        # Mean drift
        mean_change = abs(current.mean - reference.mean) / (reference.std + 0.0001)
        if mean_change > self.config.get('mean_drift_threshold', 2.0):
            alerts.append(MonitoringAlert(
                feature_name=feature_name,
                metric_name='mean_drift',
                severity=AlertSeverity.WARNING,
                message=f"Mean drift for {feature_name}: {mean_change:.2f} std deviations",
                current_value=current.mean,
                threshold=reference.mean
            ))
        
        # Std drift
        std_ratio = current.std / (reference.std + 0.0001)
        if std_ratio < 0.5 or std_ratio > 2.0:
            alerts.append(MonitoringAlert(
                feature_name=feature_name,
                metric_name='std_drift',
                severity=AlertSeverity.WARNING,
                message=f"Variance change for {feature_name}: ratio={std_ratio:.2f}",
                current_value=current.std,
                threshold=reference.std
            ))
        
        return alerts
    
    def _check_range(self, current: FeatureMetrics,
                    reference: FeatureMetrics,
                    feature_name: str) -> Optional[MonitoringAlert]:
        """Check for values outside expected range."""
        # Check if current range exceeds reference range significantly
        range_expansion = (current.max_val - current.min_val) / (
            reference.max_val - reference.min_val + 0.0001
        )
        
        if range_expansion > 2.0:
            return MonitoringAlert(
                feature_name=feature_name,
                metric_name='range_expansion',
                severity=AlertSeverity.WARNING,
                message=f"Range expanded for {feature_name}: {range_expansion:.2f}x",
                current_value=range_expansion,
                threshold=2.0
            )
        return None
    
    def _trigger_alerts(self, alert: MonitoringAlert):
        """Trigger registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler failed: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """Register a function to handle alerts."""
        self.alert_handlers.append(handler)
    
    def get_health_dashboard(self) -> Dict:
        """Get current health status of all features."""
        if not self.metrics_store:
            return {'status': 'no_data'}
        
        latest = self.metrics_store[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'overall_health': latest['summary']['health_score'],
            'features_monitored': latest['summary']['monitored_features'],
            'active_alerts': latest['summary']['total_alerts'],
            'critical_alerts': latest['summary']['critical_alerts'],
            'features_with_drift': latest['summary']['features_with_alerts']
        }

# Usage
config = {
    'completeness_threshold': 0.95,
    'psi_threshold': 0.2,
    'mean_drift_threshold': 2.0
}

monitor = FeatureMonitoringService(config)

# Set reference from training data
monitor.set_reference(training_df)

# Register alert handlers
def slack_alert(alert: MonitoringAlert):
    if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        send_slack_message(f"🚨 Feature Alert: {alert.message}")

def pagerduty_alert(alert: MonitoringAlert):
    if alert.severity == AlertSeverity.CRITICAL:
        trigger_pagerduty(alert.feature_name, alert.message)

monitor.register_alert_handler(slack_alert)
monitor.register_alert_handler(pagerduty_alert)

# Run monitoring
results = monitor.monitor_features(production_df)
dashboard = monitor.get_health_dashboard()
```

---

### Automated Drift Response

```python
class DriftResponseSystem:
    """Automated responses to feature drift."""
    
    def __init__(self, feature_monitor: FeatureMonitoringService,
                 model_registry, retraining_pipeline):
        self.monitor = feature_monitor
        self.model_registry = model_registry
        self.retraining_pipeline = retraining_pipeline
        self.drift_history = []
    
    def handle_drift(self, alerts: List[MonitoringAlert]):
        """Handle drift alerts with appropriate responses."""
        critical_features = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        warning_features = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        
        # Track drift history
        self.drift_history.append({
            'timestamp': datetime.utcnow(),
            'critical_count': len(critical_features),
            'warning_count': len(warning_features)
        })
        
        # Determine response
        if len(critical_features) > 3:
            return self._emergency_response(critical_features)
        elif len(critical_features) > 0:
            return self._trigger_retraining(critical_features)
        elif self._is_sustained_drift():
            return self._scheduled_retraining()
        else:
            return self._log_and_monitor(warning_features)
    
    def _emergency_response(self, alerts: List[MonitoringAlert]):
        """Emergency response for severe drift."""
        print("🚨 Emergency: Severe drift detected")
        
        # Option 1: Fallback to previous model
        self.model_registry.rollback_to_previous()
        
        # Option 2: Switch to rule-based fallback
        # self.model_registry.enable_fallback_mode()
        
        # Notify team
        self._alert_team(alerts, priority='high')
        
        return {'action': 'rollback', 'alerts': alerts}
    
    def _trigger_retraining(self, alerts: List[MonitoringAlert]):
        """Trigger model retraining."""
        print("⚠️ Triggering retraining due to drift")
        
        # Start retraining pipeline
        job_id = self.retraining_pipeline.trigger(
            reason='drift_detected',
            affected_features=[a.feature_name for a in alerts]
        )
        
        return {'action': 'retrain', 'job_id': job_id}
    
    def _is_sustained_drift(self) -> bool:
        """Check if drift has been sustained over multiple checks."""
        if len(self.drift_history) < 5:
            return False
        
        recent = self.drift_history[-5:]
        return all(h['warning_count'] > 0 for h in recent)
    
    def _scheduled_retraining(self):
        """Schedule retraining for sustained drift."""
        print("📅 Scheduling retraining for sustained drift")
        
        job_id = self.retraining_pipeline.schedule(
            delay_hours=24,
            reason='sustained_drift'
        )
        
        return {'action': 'schedule_retrain', 'job_id': job_id}
    
    def _log_and_monitor(self, alerts: List[MonitoringAlert]):
        """Log warnings and continue monitoring."""
        for alert in alerts:
            print(f"📝 Logging warning: {alert.message}")
        
        return {'action': 'monitor', 'logged_alerts': len(alerts)}
```

---

## 🎯 Interview Questions

**Q1: How do you set drift detection thresholds to avoid false positives?**

**Answer:**
```
Threshold setting strategy:

1. Historical analysis:
   - Analyze feature variance over time
   - Set thresholds based on historical fluctuations
   - Use percentile-based thresholds (e.g., 99th percentile)

2. Business impact:
   - Higher thresholds for less critical features
   - Lower thresholds for features known to impact model performance
   - A/B test threshold sensitivity

3. Statistical approach:
   - PSI < 0.1: No action
   - PSI 0.1-0.2: Monitor closely
   - PSI > 0.2: Investigate
   - PSI > 0.25: Alert

4. Adaptive thresholds:
   - Start with defaults
   - Adjust based on alert feedback
   - Seasonal adjustments
```

**Q2: Design a monitoring system for 10,000 features across 100 models.**

**Answer:**
```
Architecture:

1. Sampling strategy:
   - Full monitoring: Critical features (top 100)
   - Sampled monitoring: Important features (next 1000)
   - Aggregated monitoring: Remaining features

2. Hierarchical alerting:
   - Feature-level: Individual alerts
   - Feature-group level: Aggregated health
   - Model-level: Overall model health score

3. Efficient computation:
   - Streaming statistics (incremental updates)
   - Distributed computation (Spark/Flink)
   - Pre-aggregation at ingestion

4. Storage:
   - Time-series DB for metrics (InfluxDB/TimescaleDB)
   - Alert history in PostgreSQL
   - Reference stats in Redis
```

**Q3: How do you differentiate between data drift and concept drift?**

| Aspect | Data Drift | Concept Drift |
|--------|-----------|---------------|
| Definition | Input distribution changes | Input-output relationship changes |
| Detection | Feature statistics | Model performance on fresh labels |
| Example | Users age distribution shifts | Same age, different buying behavior |
| Response | May not need retraining | Definitely needs retraining |
| Monitoring | Feature monitoring | Requires ground truth labels |

---

## 🔑 Key Takeaways

1. **Monitor continuously** - detect issues early
2. **Track multiple metrics** - statistics, completeness, freshness
3. **Use appropriate tools** - Evidently AI, Fiddler, custom
4. **Set meaningful thresholds** - avoid false positives
5. **Automate responses** - fix issues automatically when possible
6. **Differentiate drift types** - data vs concept drift
7. **Scale monitoring** - sampling and aggregation for large feature sets

---

## 📚 Further Reading

- [Evidently AI Documentation](https://docs.evidently.ai/)
- [Feature Monitoring Best Practices](https://www.oreilly.com/library/view/building-machine-learning/9781492045100/)
- [ML Monitoring in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
- [Detecting Data Drift](https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766)

---

## 🔗 Related Topics

- [Feature Stores](./feature-stores.md)
- [Online vs Offline Features](./online-vs-offline-features.md)
- [Feature Pipelines](./feature-pipelines.md)
