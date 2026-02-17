# Alerting Systems

## Overview

Alerting systems notify teams of issues and anomalies in ML systems. Well-designed alerts enable quick response before issues impact users or business metrics. The challenge is balancing sensitivity (catching real issues) with specificity (avoiding alert fatigue).

---

## 🎯 Alert Design Principles

### 1. Severity Levels

| Level | Response Time | Example | Channel |
|-------|--------------|---------|---------|
| **Critical (P1)** | <15 min | Model completely down | PagerDuty + SMS |
| **High (P2)** | <1 hour | Accuracy dropped >20% | PagerDuty |
| **Medium (P3)** | <4 hours | Latency degraded | Slack |
| **Low (P4)** | Next business day | Minor drift detected | Email |
| **Info** | When convenient | Weekly report | Email digest |

### 2. Alert Criteria Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Threshold** | Fixed value exceeded | Latency > 200ms |
| **Rate of change** | Rapid change detected | Accuracy dropped 10% in 1 hour |
| **Anomaly** | Statistical anomaly | 3 std from baseline |
| **Composite** | Multiple conditions | High error rate + high latency |
| **Absence** | Expected event missing | No predictions in 5 minutes |

### 3. Good Alert Characteristics

- **Actionable**: Clear steps to investigate/resolve
- **Relevant**: Alert on symptoms that affect users
- **Timely**: Alert before impact, not after
- **Context-rich**: Include relevant data for debugging
- **Deduplicated**: Don't spam with repeated alerts

---

## 🏗️ Alerting Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Metrics Sources                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Prometheus  │  │ CloudWatch  │  │   Custom    │            │
│  │  Metrics    │  │   Metrics   │  │   Metrics   │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Alert Manager                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Alert Rules Engine                                      │   │
│  │  - Threshold checks                                      │   │
│  │  - Anomaly detection                                     │   │
│  │  - Rate of change                                        │   │
│  │  - Composite rules                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Alert Processing                                        │   │
│  │  - Deduplication                                         │   │
│  │  - Grouping                                              │   │
│  │  - Silencing                                             │   │
│  │  - Escalation                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│   PagerDuty      │ │    Slack     │ │      Email       │
│   (Critical)     │ │   (Medium)   │ │      (Low)       │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

---

## 📝 Implementation Examples

### Prometheus Alert Rules

```yaml
# alerting_rules.yml
groups:
  - name: ml_model_alerts
    rules:
      # Critical: Model server down
      - alert: ModelServerDown
        expr: up{job="model-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model server is down"
          description: "Model server {{ $labels.instance }} has been down for more than 1 minute."
          runbook: "https://wiki.company.com/runbooks/model-server-down"
      
      # High: Latency degradation
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, rate(ml_prediction_latency_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "High prediction latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 0.2s)"
          dashboard: "https://grafana.company.com/d/model-latency"
      
      # High: Error rate spike
      - alert: HighErrorRate
        expr: |
          sum(rate(ml_predictions_total{status="error"}[5m])) 
          / sum(rate(ml_predictions_total[5m])) > 0.01
        for: 2m
        labels:
          severity: high
        annotations:
          summary: "High prediction error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      # Medium: Accuracy drop
      - alert: AccuracyDrop
        expr: |
          ml_model_accuracy < 0.8 * ml_model_accuracy offset 1d
        for: 30m
        labels:
          severity: medium
        annotations:
          summary: "Model accuracy dropped significantly"
          description: "Current accuracy {{ $value }} is 20% below yesterday"
      
      # Medium: Data drift detected
      - alert: DataDriftDetected
        expr: ml_drift_score > 0.2
        for: 1h
        labels:
          severity: medium
        annotations:
          summary: "Data drift detected"
          description: "Drift score {{ $value }} exceeds threshold (0.2)"
          features: "{{ $labels.drifted_features }}"
      
      # Low: Low throughput
      - alert: LowThroughput
        expr: |
          sum(rate(ml_predictions_total[5m])) < 10
        for: 30m
        labels:
          severity: low
        annotations:
          summary: "Unusually low prediction throughput"
          description: "Current throughput: {{ $value }} predictions/second"

  - name: ml_infrastructure_alerts
    rules:
      # Critical: GPU memory exhaustion
      - alert: GPUMemoryExhaustion
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory near exhaustion"
          description: "GPU {{ $labels.gpu }} is at {{ $value | humanizePercentage }} memory usage"
      
      # High: Feature store latency
      - alert: FeatureStoreLatency
        expr: histogram_quantile(0.99, rate(feature_store_latency_seconds_bucket[5m])) > 0.05
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Feature store latency is high"
          description: "P99 latency is {{ $value }}s"
```

### Python Alert Manager

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Callable, Optional
import time
from datetime import datetime, timedelta

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    description: str
    labels: Dict[str, str]
    value: float
    threshold: float
    timestamp: datetime
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None

class AlertRule:
    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        condition: Callable[[Dict[str, float]], bool],
        message_template: str,
        for_duration: timedelta = timedelta(minutes=5),
        runbook_url: str = None
    ):
        self.name = name
        self.severity = severity
        self.condition = condition
        self.message_template = message_template
        self.for_duration = for_duration
        self.runbook_url = runbook_url
        self.firing_since: Optional[datetime] = None
    
    def evaluate(self, metrics: Dict[str, float]) -> Optional[Alert]:
        """Evaluate rule against current metrics."""
        is_firing = self.condition(metrics)
        
        if is_firing:
            if self.firing_since is None:
                self.firing_since = datetime.utcnow()
            
            # Check if firing long enough
            if datetime.utcnow() - self.firing_since >= self.for_duration:
                return Alert(
                    name=self.name,
                    severity=self.severity,
                    message=self.message_template.format(**metrics),
                    description=f"Rule {self.name} triggered",
                    labels={'rule': self.name},
                    value=metrics.get('value', 0),
                    threshold=metrics.get('threshold', 0),
                    timestamp=datetime.utcnow(),
                    runbook_url=self.runbook_url
                )
        else:
            self.firing_since = None
        
        return None

class MLAlertManager:
    """Manage ML-specific alerts."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.silenced_rules: Dict[str, datetime] = {}
        self.notification_handlers: Dict[AlertSeverity, Callable] = {}
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def register_handler(self, severity: AlertSeverity, handler: Callable):
        """Register notification handler for severity level."""
        self.notification_handlers[severity] = handler
    
    def silence_rule(self, rule_name: str, duration: timedelta):
        """Silence a rule for a duration."""
        self.silenced_rules[rule_name] = datetime.utcnow() + duration
    
    def evaluate(self, metrics: Dict[str, float]):
        """Evaluate all rules against current metrics."""
        for rule in self.rules:
            # Skip silenced rules
            if rule.name in self.silenced_rules:
                if datetime.utcnow() < self.silenced_rules[rule.name]:
                    continue
                else:
                    del self.silenced_rules[rule.name]
            
            alert = rule.evaluate(metrics)
            
            if alert:
                # New or updated alert
                if rule.name not in self.active_alerts:
                    self.active_alerts[rule.name] = alert
                    self._notify(alert)
            else:
                # Alert resolved
                if rule.name in self.active_alerts:
                    self._notify_resolved(self.active_alerts[rule.name])
                    del self.active_alerts[rule.name]
    
    def _notify(self, alert: Alert):
        """Send notification for alert."""
        handler = self.notification_handlers.get(alert.severity)
        if handler:
            handler(alert)
    
    def _notify_resolved(self, alert: Alert):
        """Send notification for resolved alert."""
        # Send to same channel with resolved status
        pass

# Define rules
alert_manager = MLAlertManager()

# High latency rule
alert_manager.add_rule(AlertRule(
    name="high_latency",
    severity=AlertSeverity.HIGH,
    condition=lambda m: m.get('p99_latency', 0) > 0.2,
    message_template="P99 latency is {p99_latency:.3f}s (threshold: 0.2s)",
    for_duration=timedelta(minutes=5),
    runbook_url="https://wiki.company.com/runbooks/high-latency"
))

# Error rate rule
alert_manager.add_rule(AlertRule(
    name="high_error_rate",
    severity=AlertSeverity.HIGH,
    condition=lambda m: m.get('error_rate', 0) > 0.01,
    message_template="Error rate is {error_rate:.2%} (threshold: 1%)",
    for_duration=timedelta(minutes=2)
))

# Accuracy drop rule
alert_manager.add_rule(AlertRule(
    name="accuracy_drop",
    severity=AlertSeverity.MEDIUM,
    condition=lambda m: m.get('accuracy', 1) < 0.8 * m.get('baseline_accuracy', 1),
    message_template="Accuracy dropped to {accuracy:.1%} from baseline {baseline_accuracy:.1%}",
    for_duration=timedelta(minutes=30)
))

# Drift rule
alert_manager.add_rule(AlertRule(
    name="data_drift",
    severity=AlertSeverity.MEDIUM,
    condition=lambda m: m.get('drift_score', 0) > 0.2,
    message_template="Data drift detected: score {drift_score:.3f}",
    for_duration=timedelta(hours=1)
))

# Register handlers
def send_pagerduty(alert: Alert):
    """Send alert to PagerDuty."""
    # PagerDuty API call
    pass

def send_slack(alert: Alert):
    """Send alert to Slack."""
    slack_client.send_message(
        channel="#ml-alerts",
        text=f"⚠️ *{alert.name}*\n{alert.message}",
        attachments=[{
            "color": "danger" if alert.severity == AlertSeverity.CRITICAL else "warning",
            "fields": [
                {"title": "Severity", "value": alert.severity.value, "short": True},
                {"title": "Value", "value": str(alert.value), "short": True},
                {"title": "Runbook", "value": alert.runbook_url or "N/A"}
            ]
        }]
    )

alert_manager.register_handler(AlertSeverity.CRITICAL, send_pagerduty)
alert_manager.register_handler(AlertSeverity.HIGH, send_slack)
alert_manager.register_handler(AlertSeverity.MEDIUM, send_slack)
```

### Alert Context Template

```python
def create_alert_context(alert_name: str, metrics: Dict) -> Dict[str, Any]:
    """Create rich context for alert investigation."""
    return {
        'alert': alert_name,
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': {
            'current': metrics,
            'baseline': get_baseline_metrics(),
            'historical': get_historical_metrics(hours=24)
        },
        'system': {
            'model_version': get_current_model_version(),
            'last_deployment': get_last_deployment_time(),
            'recent_changes': get_recent_changes()
        },
        'investigation': {
            'dashboard_url': f"https://grafana.company.com/d/ml-dashboard?time={int(time.time())}",
            'logs_url': f"https://kibana.company.com/app/logs?query=model-server",
            'runbook_url': f"https://wiki.company.com/runbooks/{alert_name}"
        },
        'suggested_actions': get_suggested_actions(alert_name),
        'recent_alerts': get_recent_alerts(hours=24)
    }
```

---

## 📊 ML-Specific Alert Examples

### Complete Alert Ruleset

| Alert Name | Condition | Severity | For Duration |
|------------|-----------|----------|--------------|
| `ModelServerDown` | Server unreachable | Critical | 1 min |
| `HighErrorRate` | Error rate > 1% | Critical | 2 min |
| `HighLatencyP99` | P99 > 500ms | High | 5 min |
| `AccuracyDrop` | Accuracy < 80% of baseline | High | 30 min |
| `DataDriftSignificant` | PSI > 0.2 | Medium | 1 hour |
| `PredictionDistributionShift` | KS test p < 0.01 | Medium | 1 hour |
| `LowConfidenceSurge` | Low conf > 20% of predictions | Medium | 30 min |
| `FeatureStoreSlow` | Feature latency > 50ms | Medium | 5 min |
| `LowThroughput` | QPS < expected * 0.5 | Low | 30 min |
| `GPUUnderutilized` | GPU usage < 20% | Low | 1 hour |

---

## ⚠️ Avoiding Alert Fatigue

### Symptoms of Alert Fatigue
- Team ignores or auto-resolves alerts
- High alert volume (>10/day per person)
- Low signal-to-noise ratio
- Alert storms during incidents

### Solutions

| Problem | Solution |
|---------|----------|
| **Too sensitive** | Increase thresholds, extend for_duration |
| **Too many alerts** | Consolidate, group related alerts |
| **Non-actionable** | Remove or convert to dashboards |
| **Noisy baselines** | Use statistical anomaly detection |
| **Alert storms** | Implement grouping and deduplication |

---

## ✅ Best Practices

1. **Every alert should be actionable** - if no action, it's noise
2. **Include context** - links to dashboards, logs, runbooks
3. **Use appropriate channels** - critical = page, low = email
4. **Group related alerts** - don't spam with individual alerts
5. **Test alerts regularly** - ensure they actually fire
6. **Review and tune** - weekly alert review meeting
7. **Track alert metrics** - time to acknowledge, time to resolve
8. **Document runbooks** - standard operating procedures

---

## 🔗 Related Topics

- [Model Monitoring](./01-model-monitoring.md) - What to monitor
- [Performance Metrics](./03-performance-metrics.md) - Metrics to alert on
- [Data Drift Detection](./02-data-drift-detection.md) - Drift-based alerts
