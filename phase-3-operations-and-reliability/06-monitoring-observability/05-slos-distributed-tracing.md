# SLOs, SLIs & Distributed Tracing for ML Systems

## Overview

ML systems require SLOs (Service Level Objectives) that go beyond traditional software because they involve unique failure modes: model degradation, data drift, feature staleness, and inference variability. Distributed tracing is essential since a single prediction request traverses many services—API gateway, feature store, model server, and post-processing—making latency breakdown and root cause analysis critical for on-call engineers.

---

## SLI/SLO/SLA Definitions and Relationships

### The Hierarchy

| Term | Definition | Example |
|------|------------|---------|
| **SLI** (Service Level Indicator) | What you measure; the raw metric | P99 latency = 85ms |
| **SLO** (Service Level Objective) | The target you commit to for the SLI | P99 latency < 100ms |
| **SLA** (Service Level Agreement) | The contractual commitment with business consequences | 99.9% uptime or credits |

```
┌─────────────────────────────────────────────────────────────────┐
│                         SLA (Contract)                            │
│  "99.9% uptime or customer receives service credits"              │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SLO (Target)                              │
│  "P99 prediction latency < 100ms"                                │
│  "99.9% of requests succeed"                                      │
│  "Model freshness < 24 hours"                                     │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SLI (Measurement)                         │
│  histogram_quantile(0.99, latency_bucket)                         │
│  success_count / total_count                                      │
│  now() - last_model_update_timestamp                             │
└─────────────────────────────────────────────────────────────────┘
```

### Relationship

- **SLI** → raw measurement (e.g., 0.01 error rate)
- **SLO** → target (e.g., error rate < 0.001)
- **SLA** → legal/business commitment (e.g., 99.9% uptime)

---

## ML-Specific SLIs with Examples

| SLI | Description | Example Metric | Typical Target |
|-----|-------------|----------------|----------------|
| **Prediction latency (p50)** | Median end-to-end prediction time | 35ms | < 50ms |
| **Prediction latency (p95)** | 95th percentile latency | 78ms | < 100ms |
| **Prediction latency (p99)** | 99th percentile latency | 120ms | < 200ms |
| **Prediction throughput** | Predictions per second | 5,000/s | Meet peak load |
| **Model freshness** | Time since last model update | 6 hours | < 24 hours |
| **Feature freshness** | Time since last feature update | 15 min | < 1 hour for real-time |
| **Prediction quality** | Accuracy / AUC on golden set | 0.92 | > 0.90 |
| **Data pipeline freshness** | Time since last successful run | 2 hours | < 4 hours |
| **Feature store availability** | % of feature lookups that succeed | 99.95% | > 99.9% |

### SLI Selection by Use Case

```
Recommendation system:
  Primary:   P99 latency, throughput
  Secondary: Feature freshness, model freshness

Fraud detection:
  Primary:   Latency (p99 < 100ms), accuracy on golden set
  Secondary: Feature pipeline freshness

Search ranking:
  Primary:   P95 latency, NDCG@10 on offline eval
  Secondary: Model freshness, feature store latency
```

---

## Defining SLOs for ML Systems

### Setting Realistic Targets

| Factor | Consideration | Example |
|--------|---------------|---------|
| **User expectation** | What does the user notice? | < 100ms feels instant |
| **Business impact** | Revenue, engagement sensitivity | Ads: < 50ms for auction |
| **Historical baseline** | What have you achieved? | p99 was 80ms last month |
| **Cost** | Tighter SLOs = more resources | 99.99% vs 99.9% = 10× cost |

### Error Budgets

**Error budget** = How much failure/degradation is acceptable before SLO is violated.

$$
\text{Error Budget} = 1 - \text{SLO}
$$

Example: 99.9% availability SLO → 0.1% error budget = ~43 minutes downtime/month.

| SLO | Monthly Error Budget | Approx. Downtime |
|-----|----------------------|------------------|
| 99% | 1% | 7.2 hours |
| 99.9% | 0.1% | 43 minutes |
| 99.95% | 0.05% | 22 minutes |
| 99.99% | 0.01% | 4.3 minutes |

### Error Budget Policies

| Policy | When Budget Exhausted | Use Case |
|--------|----------------------|----------|
| **Freeze releases** | No new model deployments | High-risk systems |
| **Reduce experimentation** | Pause A/B tests, new features | Core recommendations |
| **Escalate** | Senior review before changes | Critical path |
| **Burn rate alert** | Alert when burning too fast | Early warning |
| **Graceful degradation** | Serve cached/stale responses | User-facing apps |

### Multi-Tier SLOs

Different percentiles for different concerns:

| Tier | SLI | SLO | Purpose |
|------|-----|-----|---------|
| **User experience** | p50 latency | < 50ms | Typical user |
| **Tail latency** | p99 latency | < 200ms | Worst-case experience |
| **Availability** | Success rate | > 99.9% | System reliability |
| **Freshness** | Model age | < 24h | Model relevance |

---

## Distributed Tracing for ML Pipelines

### Why Tracing Matters

A single ML request touches many services:

```
Client Request
    │
    ├─► API Gateway (auth, routing)
    ├─► Feature Service (online features)
    ├─► Feature Store (Redis/DynamoDB lookup)
    ├─► Model Server (inference)
    ├─► Post-Processing (business logic)
    └─► Response
```

Without tracing, you see "P99 = 150ms" but cannot tell if the bottleneck is feature fetch (80ms) or model inference (60ms).

### OpenTelemetry for ML Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML Prediction Request                         │
│  trace_id: abc123  span_id: span_001                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
┌───────────┐           ┌───────────┐           ┌───────────┐
│  Span 1   │           │  Span 2   │           │  Span 3   │
│  feature  │   ───►    │  model    │   ───►    │  post-    │
│  fetch    │           │  infer    │           │  process  │
│  12ms     │           │  45ms     │           │  3ms      │
└───────────┘           └───────────┘           └───────────┘
```

### Trace Anatomy: Spans for ML

| Span | Service | Attributes | Typical Duration |
|------|---------|------------|------------------|
| **feature_fetch** | Feature store client | entity_ids, feature_names | 5–20ms |
| **model_inference** | Model server | model_version, batch_size | 10–100ms |
| **post_process** | API layer | transform_type | 1–5ms |
| **cache_lookup** | Redis/DynamoDB | hit/miss | 0.5–5ms |

### Correlating Traces with Model Predictions

- Store `trace_id` in prediction logs
- Link trace to `request_id`, `model_version`, `prediction`
- When accuracy drops: sample traces for failed predictions
- When latency spikes: filter traces by `model_version` to isolate bad deployments

---

## Latency Breakdown Analysis

### Typical ML Request Latency Budget

```
┌──────────────────────────────────────────────────────────────────┐
│  Total P99 Budget: 100ms                                          │
├──────────────────────────────────────────────────────────────────┤
│  Load balancer / routing     │  1–2 ms   │  ██                    │
│  Auth / API gateway         │  1–2 ms   │  ██                    │
│  Feature store lookup       │  5–20 ms  │  ████████████          │  ◄── Often bottleneck
│  Model inference            │  10–50 ms │  ████████████████████  │  ◄── Model-dependent
│  Post-processing            │  1–5 ms   │  ███                   │
│  Network / serialization    │  2–5 ms   │  ████                  │
└──────────────────────────────────────────────────────────────────┘
```

### Identifying Bottlenecks

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| High p99, low p50 | Feature store cold keys, cache misses | Feature store latency percentiles |
| Latency spike after deploy | New model slower | Compare spans by model_version |
| Intermittent spikes | GC pauses, network jitter | Correlation with GC, network metrics |
| Gradual increase | Data growth, feature count | Feature store size, model size |
| Spike during peak | Saturation, queueing | Queue depth, CPU/GPU utilization |

---

## On-Call Playbooks for ML Incidents

### 1. Model Serving Degradation

| Phase | Actions |
|-------|---------|
| **Detection** | P99 latency > threshold, error rate spike, health check failures |
| **Diagnosis** | Check traces: which span? Check model version, recent deploys, GPU memory |
| **Mitigation** | Rollback to previous model version; scale up replicas; enable fallback model |
| **Resolution** | Fix model or config; redeploy; validate |
| **Post-mortem** | Root cause, timeline, prevention (canary, load test) |

### 2. Feature Pipeline Failure

| Phase | Actions |
|-------|---------|
| **Detection** | Pipeline run failed; feature freshness alert; missing features in serving |
| **Diagnosis** | Check pipeline logs, data source availability, schema changes |
| **Mitigation** | Serve stale features if acceptable; disable dependent models; circuit breaker |
| **Resolution** | Fix pipeline; backfill if needed; validate feature store |
| **Post-mortem** | Why did pipeline fail? Add monitoring, retries |

### 3. Sudden Accuracy Drop

| Phase | Actions |
|-------|---------|
| **Detection** | Golden set accuracy alert; proxy metric (e.g., low confidence rate) spike |
| **Diagnosis** | Compare model version, feature distributions, data drift scores |
| **Mitigation** | Rollback model; increase human review; narrow model scope |
| **Resolution** | Retrain on fresh data; fix data pipeline; adjust thresholds |
| **Post-mortem** | Drift analysis, retraining cadence, monitoring gaps |

### 4. Data Drift Alert

| Phase | Actions |
|-------|---------|
| **Detection** | PSI/KS test triggered; feature distribution alert |
| **Diagnosis** | Which features drifted? Is it data pipeline or real distribution shift? |
| **Mitigation** | Log for analysis; consider disabling sensitive features; alert ML team |
| **Resolution** | Retrain if needed; fix data source if pipeline bug |
| **Post-mortem** | Root cause of drift, early detection improvements |

### 5. Traffic Spike Beyond Capacity

| Phase | Actions |
|-------|---------|
| **Detection** | QPS 2–3× normal; latency climbing; queue depth high |
| **Diagnosis** | Check traffic source (legit vs attack); which endpoints? |
| **Mitigation** | Scale up (if auto-scale); rate limit; serve cached/stale; degrade gracefully |
| **Resolution** | Adjust capacity; add caching; optimize hot path |
| **Post-mortem** | Capacity planning, auto-scaling tuning |

### Post-Mortem Template

```markdown
## Incident: [Title]
- **Start**: [Timestamp]
- **End**: [Timestamp]
- **Severity**: P1/P2/P3
- **Impact**: [User-facing, revenue, etc.]

## Timeline
- T+0: Detection
- T+15: Diagnosis
- T+30: Mitigation
- T+60: Resolution

## Root Cause
[What actually caused it]

## Prevention
- [ ] Add monitoring for X
- [ ] Change process Y
- [ ] Update runbook Z

## Action Items
- [ ] Owner, Due date
```

---

## 📝 Implementation Examples

### SLOMonitor Class

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading

class SLOStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"  # Burn rate high
    EXHAUSTED = "exhausted"  # Budget consumed

@dataclass
class SLIDefinition:
    name: str
    query: str  # e.g., Prometheus query
    target: float  # SLO target value
    slo_type: str  # "lower_is_better" | "higher_is_better"
    window_hours: int = 24

@dataclass
class SLOConfig:
    name: str
    slis: List[SLIDefinition]
    error_budget_policy: str  # "freeze" | "alert" | "degrade"
    alert_burn_rate: float = 2.0  # Alert when burning 2x budget

class SLOMonitor:
    """
    Monitor SLO compliance and error budget for ML systems.
    """
    
    def __init__(self, config: SLOConfig, metrics_client=None):
        self.config = config
        self.metrics_client = metrics_client
        self.error_budget_remaining: Dict[str, float] = {}
        self.last_check: Optional[datetime] = None
        self._lock = threading.Lock()
    
    def check_sli(self, sli: SLIDefinition) -> tuple[float, bool]:
        """
        Check SLI against SLO target.
        Returns (current_value, in_compliance).
        """
        # Simulated; in production, run metrics_client.query(sli.query)
        current_value = self._fetch_sli_value(sli)
        
        if sli.slo_type == "lower_is_better":
            in_compliance = current_value <= sli.target
        else:
            in_compliance = current_value >= sli.target
        
        return current_value, in_compliance
    
    def _fetch_sli_value(self, sli: SLIDefinition) -> float:
        """Fetch SLI value from metrics backend."""
        # Placeholder: would use Prometheus/CloudWatch
        return 0.0
    
    def compute_error_budget(self, sli: SLIDefinition) -> float:
        """
        Compute remaining error budget (0.0 to 1.0).
        1.0 = full budget, 0.0 = exhausted.
        """
        current, in_compliance = self.check_sli(sli)
        # Simplified: assume we track success rate
        # budget = 1 - (actual_errors / allowed_errors)
        return 0.85  # Placeholder
    
    def get_status(self) -> SLOStatus:
        """Overall SLO status."""
        budgets = [self.compute_error_budget(sli) for sli in self.config.slis]
        min_budget = min(budgets)
        
        if min_budget <= 0:
            return SLOStatus.EXHAUSTED
        elif min_budget < 0.1:
            return SLOStatus.WARNING
        return SLOStatus.HEALTHY
    
    def should_allow_deployment(self) -> tuple[bool, str]:
        """
        Check if deployment is allowed per error budget policy.
        """
        status = self.get_status()
        
        if status == SLOStatus.EXHAUSTED and self.config.error_budget_policy == "freeze":
            return False, "Error budget exhausted; deployments frozen"
        if status == SLOStatus.WARNING:
            return True, "Warning: low error budget; proceed with caution"
        return True, "OK"

# Example config
prediction_slo = SLOConfig(
    name="prediction_service",
    slis=[
        SLIDefinition("p99_latency", "histogram_quantile(0.99, rate(latency_bucket[5m]))", 0.1, "lower_is_better"),
        SLIDefinition("availability", "sum(rate(success_total[5m]))/sum(rate(total[5m]))", 0.999, "higher_is_better"),
    ],
    error_budget_policy="alert",
)

monitor = SLOMonitor(prediction_slo)
print(monitor.should_allow_deployment())
```

### DistributedTracer for ML

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.trace import Status, StatusCode
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Configure tracing (simplified)
trace.set_tracer_provider(TracerProvider())
# In production: add JaegerExporter, etc.

tracer = trace.get_tracer("ml-prediction-service", "1.0.0")

class MLRequestTracer:
    """
    Distributed tracer for ML prediction requests.
    Creates child spans for feature fetch, inference, post-processing.
    """
    
    def __init__(self, request_id: str, model_version: str):
        self.request_id = request_id
        self.model_version = model_version
        self.ctx = None
        self._span = None
    
    @contextmanager
    def trace_request(self):
        """Top-level span for entire request."""
        with tracer.start_as_current_span(
            "ml_prediction_request",
            attributes={
                "request_id": self.request_id,
                "model_version": self.model_version,
            }
        ) as span:
            self._span = span
            self.ctx = trace.get_current_context()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    @contextmanager
    def span_feature_fetch(self, entity_ids: list, feature_count: int):
        """Span for feature store lookup."""
        with tracer.start_as_current_span(
            "feature_fetch",
            attributes={
                "entity_count": len(entity_ids),
                "feature_count": feature_count,
            }
        ) as span:
            start = time.perf_counter()
            try:
                yield span
            finally:
                span.set_attribute("duration_ms", (time.perf_counter() - start) * 1000)
    
    @contextmanager
    def span_model_inference(self, batch_size: int):
        """Span for model inference."""
        with tracer.start_as_current_span(
            "model_inference",
            attributes={
                "model_version": self.model_version,
                "batch_size": batch_size,
            }
        ) as span:
            start = time.perf_counter()
            try:
                yield span
            finally:
                span.set_attribute("duration_ms", (time.perf_counter() - start) * 1000)
    
    @contextmanager
    def span_post_process(self):
        """Span for post-processing."""
        with tracer.start_as_current_span("post_process") as span:
            start = time.perf_counter()
            try:
                yield span
            finally:
                span.set_attribute("duration_ms", (time.perf_counter() - start) * 1000)

# Usage in prediction flow
def predict_with_tracing(request_id: str, model_version: str, entities: list, features: dict):
    tracer_obj = MLRequestTracer(request_id, model_version)
    
    with tracer_obj.trace_request():
        with tracer_obj.span_feature_fetch(entities, len(features)):
            # feature_store.get_features(entities)
            time.sleep(0.01)  # Simulate
        
        with tracer_obj.span_model_inference(len(entities)):
            # model.predict(features)
            time.sleep(0.05)  # Simulate
        
        with tracer_obj.span_post_process():
            # post_process(prediction)
            time.sleep(0.002)  # Simulate
        
        return {"prediction": 0.85, "trace_id": trace.get_current_span().get_span_context().trace_id}
```

### IncidentPlaybook

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class IncidentType(Enum):
    MODEL_SERVING_DEGRADATION = "model_serving_degradation"
    FEATURE_PIPELINE_FAILURE = "feature_pipeline_failure"
    ACCURACY_DROP = "accuracy_drop"

@dataclass
class PlaybookStep:
    phase: str
    action: str
    commands: List[str]
    checklist: List[str]

class IncidentPlaybook:
    PLAYBOOKS = {
        IncidentType.MODEL_SERVING_DEGRADATION: [
            PlaybookStep("diagnosis", "Check model version and traces",
                ["kubectl get pods -l app=model-server", "jaeger query trace_id"],
                ["Identify slow span", "Check recent deployments"]),
            PlaybookStep("mitigation", "Rollback model",
                ["kubectl rollout undo deployment/model-server"],
                ["Confirm rollback success", "Check latency"]),
        ],
        IncidentType.ACCURACY_DROP: [
            PlaybookStep("diagnosis", "Compare model and data",
                ["Compare model versions", "Run drift report"],
                ["Identify drifted features", "Check golden set"]),
            PlaybookStep("mitigation", "Rollback or scope down",
                ["kubectl rollout undo deployment/model-server"],
                ["Validate rollback", "Notify stakeholders"]),
        ],
    }

    @classmethod
    def get_playbook(cls, incident_type: IncidentType) -> List[PlaybookStep]:
        return cls.PLAYBOOKS.get(incident_type, [])
```

---

## Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| **SLO strictness** | 99.9% | 99.99% | Tighter = 10× more resources, fewer incidents allowed |
| **Error budget policy** | Freeze on exhaust | Alert only | Freeze = safer but blocks releases; alert = faster but riskier |
| **Tracing sampling** | 100% | 1% | Full = complete view, expensive; sampled = cheaper, may miss rare issues |
| **Multi-tier SLOs** | p50 + p99 | p99 only | Multi-tier = richer signals, more complexity; single = simpler |
| **Playbook detail** | Extensive runbooks | High-level only | Extensive = faster resolution, more upkeep; high-level = flexible, slower |
| **Golden set frequency** | Real-time | Daily batch | Real-time = faster accuracy alerts, higher cost; batch = cheaper, delayed |

---

## Interview Tips

1. **Start with user impact**: "For a recommendation API, users notice latency > 100ms, so we'd set p99 SLO at 100ms."
2. **Connect SLIs to business**: Don't just say "p99 latency"; say "p99 latency because it directly affects user engagement."
3. **Mention error budgets**: "We use a 99.9% availability SLO, which gives us ~43 minutes error budget per month; when exhausted, we freeze model deployments."
4. **Trace anatomy**: "A prediction request has spans for feature fetch, model inference, and post-processing; when p99 spikes, we filter traces by model_version to find which deploy caused it."
5. **Playbook structure**: "For model degradation: detect via latency/error alerts, diagnose via traces and model version, mitigate by rollback, resolve with fix and post-mortem."
6. **ML-specific SLIs**: Emphasize model freshness, feature freshness, and prediction quality (golden set)—not just latency and availability.
7. **Trade-offs**: Be ready to discuss SLO strictness vs cost, trace sampling vs completeness, and error budget policy choices.

---

## Related Topics

- [Model Monitoring](./01-model-monitoring.md) - What to monitor for ML systems
- [Performance Metrics](./03-performance-metrics.md) - Metrics that become SLIs
- [Alerting Systems](./04-alerting-systems.md) - Alerting on SLO violations
- [Data Drift Detection](./02-data-drift-detection.md) - Drift as SLO/SLI signal
