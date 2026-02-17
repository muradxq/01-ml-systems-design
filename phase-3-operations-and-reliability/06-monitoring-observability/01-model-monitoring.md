# Model Monitoring

## Overview

Model monitoring tracks model performance, predictions, and system health in production. Unlike traditional software where bugs are explicit, ML models can degrade silently due to data drift, concept drift, or changing user behavior. Comprehensive monitoring enables early detection of issues before they impact business metrics.

---

## 🎯 What to Monitor

### 1. Prediction Metrics

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| **Prediction distribution** | Output distribution shift | >2 std from baseline |
| **Mean confidence** | Model certainty changes | <80% of baseline |
| **Low confidence rate** | Increasing uncertainty | >2x baseline |
| **Prediction latency** | Performance degradation | p99 > 2x baseline |
| **Throughput** | Capacity issues | <80% expected |
| **Error rate** | System failures | >0.1% |

### 2. Model Performance Metrics

| Metric | When Available | Purpose |
|--------|----------------|---------|
| **Accuracy** | With ground truth | Direct quality measure |
| **Precision/Recall** | With ground truth | Class-specific performance |
| **AUC-ROC** | With ground truth | Ranking quality |
| **Proxy metrics** | Always | Indirect quality signals |
| **Business metrics** | With delay | Real-world impact |

### 3. System Health Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| **Request latency (p50)** | <50ms | >100ms |
| **Request latency (p99)** | <200ms | >500ms |
| **Error rate** | <0.1% | >1% |
| **CPU utilization** | <70% | >90% |
| **Memory usage** | <80% | >95% |
| **GPU utilization** | >50% | <20% (waste) |

---

## 🏗️ Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Serving Layer                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Model Server (TensorFlow Serving, TorchServe, etc.)    │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Instrumentation Layer                          │    │   │
│  │  │  - Request/Response logging                     │    │   │
│  │  │  - Latency tracking                             │    │   │
│  │  │  - Prediction capture                           │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Metrics Collection                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Prometheus    │  │  CloudWatch    │  │  Custom        │   │
│  │  (Pull-based)  │  │  (Push-based)  │  │  Collector     │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis & Storage                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Time Series DB  │  ML Monitoring Platform             │    │
│  │  (InfluxDB,      │  (Evidently, Fiddler, Arize)       │    │
│  │   TimescaleDB)   │                                     │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│   Dashboards     │ │   Alerts     │ │   Reports        │
│   (Grafana)      │ │   (PagerDuty)│ │   (Weekly)       │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

---

## 📝 Implementation Examples

### Complete Monitoring Setup

```python
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
import numpy as np
from typing import Dict, Any

# Define metrics
class ModelMetrics:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        
        # Request metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total predictions made',
            ['model', 'version', 'status']
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model', 'version'],
            buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5]
        )
        
        # Prediction distribution metrics
        self.prediction_value = Histogram(
            'ml_prediction_value',
            'Distribution of prediction values',
            ['model', 'version'],
            buckets=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
        )
        
        self.prediction_confidence = Histogram(
            'ml_prediction_confidence',
            'Distribution of confidence scores',
            ['model', 'version'],
            buckets=[.5, .6, .7, .8, .85, .9, .95, .99, 1.0]
        )
        
        # Resource metrics
        self.model_load_time = Gauge(
            'ml_model_load_time_seconds',
            'Time to load model',
            ['model', 'version']
        )
        
        self.feature_count = Gauge(
            'ml_features_count',
            'Number of features in request',
            ['model', 'version']
        )
    
    def track_prediction(self, latency: float, prediction: float, 
                        confidence: float, status: str = 'success'):
        """Track a single prediction."""
        labels = {'model': self.model_name, 'version': self.model_version}
        
        self.prediction_counter.labels(**labels, status=status).inc()
        self.prediction_latency.labels(**labels).observe(latency)
        self.prediction_value.labels(**labels).observe(prediction)
        self.prediction_confidence.labels(**labels).observe(confidence)

# Decorator for monitoring
def monitor_prediction(metrics: ModelMetrics):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                
                # Track metrics
                latency = time.time() - start_time
                metrics.track_prediction(
                    latency=latency,
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    status=status
                )
                
                return result
            except Exception as e:
                status = 'error'
                metrics.prediction_counter.labels(
                    model=metrics.model_name,
                    version=metrics.model_version,
                    status=status
                ).inc()
                raise
        return wrapper
    return decorator

# Usage
metrics = ModelMetrics(model_name="recommendation", model_version="v1.2.3")

@monitor_prediction(metrics)
def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    prediction = model.predict(features)
    return {
        'prediction': float(prediction[0]),
        'confidence': float(prediction.max())
    }
```

### Prediction Logging for Analysis

```python
import json
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging

@dataclass
class PredictionLog:
    """Structured prediction log for analysis."""
    timestamp: str
    model_name: str
    model_version: str
    request_id: str
    user_id: Optional[str]
    features: Dict[str, Any]
    prediction: float
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any]

class PredictionLogger:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.logger = logging.getLogger('prediction_logger')
        
    def log_prediction(self, request_id: str, features: Dict, 
                      prediction: float, confidence: float,
                      latency_ms: float, user_id: str = None,
                      metadata: Dict = None):
        """Log prediction for analysis."""
        log = PredictionLog(
            timestamp=datetime.datetime.utcnow().isoformat(),
            model_name=self.model_name,
            model_version=self.model_version,
            request_id=request_id,
            user_id=user_id,
            features=features,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        # Log as JSON for easy parsing
        self.logger.info(json.dumps(asdict(log)))
        
        # Also send to analytics pipeline
        send_to_analytics(log)

# Streaming logs to analytics
def send_to_analytics(log: PredictionLog):
    """Send prediction log to analytics pipeline."""
    # Send to Kafka, Kinesis, or similar
    kafka_producer.send('predictions', asdict(log))
```

### Real-time Distribution Monitoring

```python
import numpy as np
from collections import deque
from scipy import stats

class DistributionMonitor:
    """Monitor prediction distribution for drift."""
    
    def __init__(self, window_size: int = 1000, 
                 reference_data: np.ndarray = None):
        self.window_size = window_size
        self.recent_predictions = deque(maxlen=window_size)
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data) if reference_data is not None else None
        self.reference_std = np.std(reference_data) if reference_data is not None else None
    
    def add_prediction(self, prediction: float):
        """Add prediction to monitoring window."""
        self.recent_predictions.append(prediction)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current distribution statistics."""
        if len(self.recent_predictions) < 100:
            return {}
        
        predictions = np.array(self.recent_predictions)
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'median': float(np.median(predictions)),
            'p5': float(np.percentile(predictions, 5)),
            'p95': float(np.percentile(predictions, 95)),
            'count': len(predictions)
        }
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for distribution drift against reference."""
        if self.reference_data is None or len(self.recent_predictions) < 100:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        current = np.array(self.recent_predictions)
        
        # KS test for distribution comparison
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference_data, current)
        
        # Mean shift check
        mean_shift = abs(np.mean(current) - self.reference_mean) / self.reference_std
        
        drift_detected = ks_pvalue < 0.01 or mean_shift > 2.0
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'mean_shift_std': float(mean_shift),
            'current_mean': float(np.mean(current)),
            'reference_mean': float(self.reference_mean)
        }

# Usage
reference_predictions = np.load('reference_predictions.npy')
monitor = DistributionMonitor(window_size=1000, reference_data=reference_predictions)

@app.post("/predict")
async def predict(features: dict):
    prediction = model.predict(features)
    
    # Add to distribution monitor
    monitor.add_prediction(prediction)
    
    # Periodically check drift
    if len(monitor.recent_predictions) % 100 == 0:
        drift_result = monitor.check_drift()
        if drift_result['drift_detected']:
            alert_drift(drift_result)
    
    return {'prediction': prediction}
```

---

## 🛠️ Monitoring Tools Comparison

| Tool | Type | Strengths | Best For |
|------|------|-----------|----------|
| **Evidently AI** | ML-specific | Easy drift detection, reports | Quick setup |
| **Fiddler** | ML-specific | Explainability, monitoring | Enterprise |
| **Arize** | ML-specific | Embeddings, NLP support | Advanced ML |
| **Prometheus** | General metrics | Scalable, flexible | Custom metrics |
| **Grafana** | Visualization | Dashboards, alerts | Visualization |
| **Datadog** | Full stack | APM + ML monitoring | All-in-one |
| **Custom** | DIY | Full control | Specific needs |

---

## 📊 Dashboard Design

### Essential Dashboard Panels

1. **Request Volume & Latency**
   - Requests per second
   - Latency percentiles (p50, p95, p99)
   - Error rate

2. **Prediction Distribution**
   - Histogram of prediction values
   - Confidence distribution
   - Class distribution (for classification)

3. **Model Performance** (when ground truth available)
   - Accuracy over time
   - Precision/Recall
   - Business metrics correlation

4. **Resource Usage**
   - CPU/Memory/GPU utilization
   - Model load time
   - Cache hit rate

5. **Drift Indicators**
   - Feature drift scores
   - Prediction drift
   - Distribution comparisons

---

## ⚠️ The Ground Truth Problem

In production, you often can't measure accuracy directly because:
- Ground truth is delayed (fraud discovered weeks later)
- Ground truth is partial (only some predictions verified)
- Ground truth is never available (recommendations never tested)

### Solutions

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Proxy metrics** | Use correlated metrics (CTR, conversion) | Always |
| **Sampled evaluation** | Human review of samples | High-stakes |
| **Delayed labels** | Wait for ground truth | When available |
| **A/B testing** | Compare against baseline | Model updates |
| **Shadow mode** | Run alongside production | New models |

---

## ✅ Best Practices

1. **Start monitoring before launch** - establish baselines
2. **Monitor predictions, not just system** - ML-specific metrics
3. **Use multiple detection methods** - statistical tests + business metrics
4. **Set appropriate thresholds** - avoid alert fatigue
5. **Automate responses** - auto-rollback, auto-retrain triggers
6. **Review regularly** - weekly model performance reviews
7. **Correlate with business** - connect ML metrics to business outcomes

---

## 🔗 Related Topics

- [Data Drift Detection](./02-data-drift-detection.md) - detecting input distribution changes
- [Performance Metrics](./03-performance-metrics.md) - technical and business metrics
- [Alerting Systems](./04-alerting-systems.md) - setting up effective alerts
