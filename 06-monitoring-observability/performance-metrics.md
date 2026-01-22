# Performance Metrics

## Overview

Performance metrics track both technical and business performance of ML systems. The right metrics enable optimization, demonstrate business value, and provide early warning of issues. ML systems require tracking metrics at multiple levels: infrastructure, model quality, and business impact.

---

## 📊 Metric Categories

### 1. Technical/System Metrics

| Metric | Description | Target | How to Measure |
|--------|-------------|--------|----------------|
| **Latency (p50)** | Median response time | <50ms | Timer around prediction |
| **Latency (p95)** | 95th percentile response | <100ms | Timer around prediction |
| **Latency (p99)** | 99th percentile response | <200ms | Timer around prediction |
| **Throughput** | Requests per second | Varies | Request counter |
| **Error rate** | % of failed requests | <0.1% | Error counter / total |
| **Availability** | % uptime | >99.9% | Uptime monitoring |
| **CPU utilization** | % CPU used | <70% | System metrics |
| **Memory usage** | % memory used | <80% | System metrics |
| **GPU utilization** | % GPU used | >50% | nvidia-smi |

### 2. Model Quality Metrics

#### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | Cost of false positives high |
| **Recall** | TP / (TP + FN) | Cost of false negatives high |
| **F1 Score** | 2 * (P * R) / (P + R) | Balance precision/recall |
| **AUC-ROC** | Area under ROC curve | Ranking quality |
| **AUC-PR** | Area under PR curve | Imbalanced classes |
| **Log Loss** | -Σ(y*log(p)) | Calibration matters |

#### Regression Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **MAE** | Σ\|y - ŷ\| / n | Robust to outliers |
| **MSE** | Σ(y - ŷ)² / n | Penalize large errors |
| **RMSE** | √MSE | Same unit as target |
| **MAPE** | Σ\|(y - ŷ)/y\| / n | Percentage errors |
| **R²** | 1 - SS_res/SS_tot | Explained variance |

#### Ranking Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **NDCG** | Normalized Discounted Cumulative Gain | Search, recommendations |
| **MRR** | Mean Reciprocal Rank | First relevant item matters |
| **MAP** | Mean Average Precision | Multiple relevant items |
| **Hit Rate@K** | % with hit in top K | Recommendation systems |

### 3. Business Metrics

| Metric | Description | Example |
|--------|-------------|---------|
| **CTR** | Click-through rate | Ads, recommendations |
| **Conversion Rate** | % who complete action | E-commerce |
| **Revenue per User** | Average revenue impact | Pricing models |
| **Customer Lifetime Value** | Long-term value | Churn prediction |
| **Cost Savings** | Operational savings | Fraud detection |
| **Time Saved** | Efficiency gains | Automation |

---

## 🏗️ Metrics Collection Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML System                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Prediction  │  │  Feature    │  │    Data     │            │
│  │   Service   │  │   Store     │  │  Pipeline   │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                     │
│    ┌────┴────────────────┴────────────────┴────┐              │
│    │           Instrumentation Layer           │              │
│    │  - Prometheus client                      │              │
│    │  - Custom metrics                         │              │
│    │  - Trace context                          │              │
│    └────────────────────┬──────────────────────┘              │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Metrics Aggregation                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Prometheus  │  │ CloudWatch  │  │   Custom    │            │
│  │  (Pull)     │  │   (Push)    │  │  Collector  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage & Analysis                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Time Series DB (Prometheus, InfluxDB, TimescaleDB)     │   │
│  │  + Business Metrics (Analytics DB, Snowflake)           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
│   Dashboards     │ │  Alerts  │ │     Reports      │
│   (Real-time)    │ │          │ │   (Weekly/Daily) │
└──────────────────┘ └──────────┘ └──────────────────┘
```

---

## 📝 Implementation Examples

### Comprehensive Metrics Collection

```python
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps
from typing import Dict, Any
import numpy as np

class MLMetrics:
    """Comprehensive ML metrics collection."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        # System metrics
        self.request_latency = Histogram(
            f'{service_name}_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint', 'model_version'],
            buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0]
        )
        
        self.request_count = Counter(
            f'{service_name}_requests_total',
            'Total requests',
            ['endpoint', 'model_version', 'status']
        )
        
        # Model metrics
        self.prediction_value = Histogram(
            f'{service_name}_prediction_value',
            'Distribution of prediction values',
            ['model_version'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.prediction_confidence = Histogram(
            f'{service_name}_prediction_confidence',
            'Distribution of confidence scores',
            ['model_version'],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
        )
        
        self.feature_count = Histogram(
            f'{service_name}_feature_count',
            'Number of features in request',
            ['model_version'],
            buckets=[5, 10, 25, 50, 100, 250, 500]
        )
        
        # Batch metrics
        self.batch_size = Histogram(
            f'{service_name}_batch_size',
            'Batch size for predictions',
            ['model_version'],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        )
        
        # Business proxy metrics
        self.high_confidence_predictions = Counter(
            f'{service_name}_high_confidence_total',
            'Predictions with high confidence (>0.9)',
            ['model_version']
        )
        
        self.low_confidence_predictions = Counter(
            f'{service_name}_low_confidence_total',
            'Predictions with low confidence (<0.5)',
            ['model_version']
        )
    
    def record_prediction(self, model_version: str, prediction: float,
                         confidence: float, latency: float,
                         feature_count: int, status: str = 'success'):
        """Record all metrics for a single prediction."""
        
        # System metrics
        self.request_latency.labels(
            endpoint='/predict',
            model_version=model_version
        ).observe(latency)
        
        self.request_count.labels(
            endpoint='/predict',
            model_version=model_version,
            status=status
        ).inc()
        
        # Model metrics
        self.prediction_value.labels(model_version=model_version).observe(prediction)
        self.prediction_confidence.labels(model_version=model_version).observe(confidence)
        self.feature_count.labels(model_version=model_version).observe(feature_count)
        
        # Business proxy metrics
        if confidence > 0.9:
            self.high_confidence_predictions.labels(model_version=model_version).inc()
        elif confidence < 0.5:
            self.low_confidence_predictions.labels(model_version=model_version).inc()

# Decorator for automatic metrics
def track_prediction(metrics: MLMetrics, model_version: str):
    def decorator(func):
        @wraps(func)
        def wrapper(features: Dict[str, Any], *args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(features, *args, **kwargs)
                
                metrics.record_prediction(
                    model_version=model_version,
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    latency=time.time() - start_time,
                    feature_count=len(features),
                    status=status
                )
                
                return result
            except Exception as e:
                status = 'error'
                metrics.request_count.labels(
                    endpoint='/predict',
                    model_version=model_version,
                    status=status
                ).inc()
                raise
        return wrapper
    return decorator
```

### Model Quality Metrics Calculator

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np
from typing import Dict, Any, Optional

class ModelMetricsCalculator:
    """Calculate and track model quality metrics."""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError:
                pass  # Binary classification with single class
            
            metrics['log_loss'] = log_loss(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    @staticmethod
    def ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                       k: int = 10) -> Dict[str, float]:
        """Calculate ranking metrics for top-K recommendations."""
        # Sort by predicted scores
        sorted_indices = np.argsort(y_scores)[::-1]
        top_k = sorted_indices[:k]
        
        # Hit rate at K
        hits = np.sum(y_true[top_k])
        hit_rate = hits / k
        
        # NDCG at K
        relevance = y_true[top_k]
        dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
        ideal_relevance = np.sort(y_true)[::-1][:k]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        ranks = np.where(y_true[sorted_indices] > 0)[0]
        mrr = 1 / (ranks[0] + 1) if len(ranks) > 0 else 0
        
        return {
            f'hit_rate@{k}': hit_rate,
            f'ndcg@{k}': ndcg,
            'mrr': mrr
        }

# Usage
calculator = ModelMetricsCalculator()

def evaluate_model_performance(model, test_data, test_labels):
    """Evaluate model and log metrics."""
    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data) if hasattr(model, 'predict_proba') else None
    
    metrics = calculator.classification_metrics(
        test_labels, predictions, probabilities
    )
    
    # Log metrics
    for name, value in metrics.items():
        log_metric(f'model_{name}', value)
    
    return metrics
```

### Business Metrics Correlation

```python
import pandas as pd
from scipy import stats

def correlate_ml_business_metrics(ml_metrics: pd.DataFrame, 
                                  business_metrics: pd.DataFrame) -> Dict[str, float]:
    """
    Correlate ML metrics with business outcomes.
    
    Args:
        ml_metrics: DataFrame with columns [timestamp, accuracy, confidence, latency]
        business_metrics: DataFrame with columns [timestamp, ctr, conversion, revenue]
    
    Returns:
        Correlation coefficients between ML and business metrics
    """
    # Merge on timestamp
    merged = pd.merge(ml_metrics, business_metrics, on='timestamp', how='inner')
    
    correlations = {}
    
    # Correlate each ML metric with each business metric
    ml_cols = ['accuracy', 'confidence', 'latency']
    business_cols = ['ctr', 'conversion', 'revenue']
    
    for ml_col in ml_cols:
        for biz_col in business_cols:
            corr, p_value = stats.pearsonr(merged[ml_col], merged[biz_col])
            correlations[f'{ml_col}_vs_{biz_col}'] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return correlations

# Example usage
def weekly_metrics_report():
    """Generate weekly metrics report."""
    ml_metrics = get_ml_metrics_last_week()
    business_metrics = get_business_metrics_last_week()
    
    correlations = correlate_ml_business_metrics(ml_metrics, business_metrics)
    
    # Find significant correlations
    significant = {k: v for k, v in correlations.items() if v['significant']}
    
    report = {
        'period': 'last_7_days',
        'ml_metrics_summary': ml_metrics.describe().to_dict(),
        'business_metrics_summary': business_metrics.describe().to_dict(),
        'correlations': correlations,
        'insights': generate_insights(significant)
    }
    
    return report
```

---

## 📊 Dashboard Design

### Essential Dashboard Panels

| Panel | Metrics | Purpose |
|-------|---------|---------|
| **Request Overview** | QPS, latency percentiles, errors | System health |
| **Prediction Distribution** | Histogram of predictions | Detect output drift |
| **Confidence Distribution** | Histogram of confidence | Model certainty |
| **Model Accuracy** | Accuracy over time | Performance trend |
| **Business Impact** | CTR, conversion, revenue | Business value |
| **Resource Usage** | CPU, memory, GPU | Cost optimization |

---

## ✅ Best Practices

1. **Track both technical and business** - demonstrate ML value
2. **Establish baselines** - compare against historical performance
3. **Monitor trends, not just thresholds** - gradual degradation matters
4. **Correlate ML and business metrics** - understand impact
5. **Report regularly** - weekly summaries for stakeholders
6. **Set SLOs** - define acceptable performance levels
7. **Use percentiles, not averages** - p95/p99 catch tail latency

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Average-only tracking** | Miss tail latency | Track percentiles |
| **No business correlation** | Can't show value | Track business metrics |
| **Static thresholds** | Don't adapt to growth | Use relative thresholds |
| **Too many metrics** | Dashboard overload | Focus on key metrics |
| **No historical context** | Can't spot trends | Store and compare history |

---

## 🔗 Related Topics

- [Model Monitoring](./model-monitoring.md) - What to monitor
- [Alerting Systems](./alerting-systems.md) - Alert on metric thresholds
- [Data Drift Detection](./data-drift-detection.md) - Input distribution changes
