# 🛡️ Reliability & Fault Tolerance

## Overview

Reliability and fault tolerance ensure ML systems continue operating despite failures. ML systems are particularly vulnerable because they depend on multiple components—data pipelines, feature stores, model servers, and downstream services—any of which can fail. Designing for failure from the start is essential for production systems.

---

## 🎯 Learning Objectives

After completing this section, you should understand:
- High availability patterns for ML systems
- Graceful degradation strategies when components fail
- Circuit breaker patterns to prevent cascade failures
- Disaster recovery planning and implementation

---

## 📚 Topics Covered

1. [High Availability](./01-high-availability.md)
   - Redundancy patterns
   - Failover strategies
   - Multi-region deployment

2. [Graceful Degradation](./02-graceful-degradation.md)
   - Fallback models
   - Default predictions
   - Feature fallbacks

3. [Circuit Breakers](./03-circuit-breakers.md)
   - Preventing cascade failures
   - Implementation patterns
   - Recovery strategies

4. [Disaster Recovery](./04-disaster-recovery.md)
   - Backup strategies
   - Recovery procedures
   - RTO/RPO planning

---

## 🏗️ Reliability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Requests                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Load Balancer (Active-Active)                   │
│              [Region A]           [Region B]                     │
│                  │                    │                          │
│                  ▼                    ▼                          │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │   Health Checks     │  │   Health Checks     │              │
│  │   Circuit Breaker   │  │   Circuit Breaker   │              │
│  └─────────────────────┘  └─────────────────────┘              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Model Server    │ │  Model Server    │ │  Model Server    │
│  (Primary)       │ │  (Primary)       │ │  (Primary)       │
│       │          │ │       │          │ │       │          │
│       ▼          │ │       ▼          │ │       ▼          │
│  [Fallback       │ │  [Fallback       │ │  [Fallback       │
│   Model]         │ │   Model]         │ │   Model]         │
└──────────────────┘ └──────────────────┘ └──────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Feature Store   │ │  Feature Store   │ │  Default         │
│  (Primary)       │ │  (Replica)       │ │  Features        │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 📊 Failure Modes in ML Systems

### 1. Model Server Failures

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| **Server crash** | No predictions | Multiple replicas, auto-restart |
| **OOM error** | Server unresponsive | Memory limits, graceful degradation |
| **Model loading failure** | Can't serve | Pre-loaded fallback model |
| **Slow predictions** | Timeout errors | Circuit breaker, timeout handling |

### 2. Feature Store Failures

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| **Store unavailable** | Missing features | Cached features, defaults |
| **Stale features** | Degraded accuracy | Feature freshness monitoring |
| **Partial features** | Incomplete input | Feature fallbacks |
| **High latency** | Slow predictions | Caching, timeouts |

### 3. Data Pipeline Failures

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| **Pipeline failure** | No new data | Monitoring, auto-retry |
| **Data corruption** | Bad predictions | Validation, rollback |
| **Schema changes** | Pipeline breaks | Schema versioning |
| **Source unavailable** | No data flow | Multi-source, caching |

---

## 🔧 Reliability Patterns

### 1. Circuit Breaker Implementation

```python
from enum import Enum
from dataclasses import dataclass
import time
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 3
    
    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
    
    def call(self, func: Callable, fallback: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                return fallback(*args, **kwargs)
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            return fallback(*args, **kwargs)
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

def predict_with_circuit_breaker(features):
    return circuit_breaker.call(
        func=lambda f: primary_model.predict(f),
        fallback=lambda f: fallback_model.predict(f),
        features
    )
```

### 2. Fallback Model Strategy

```python
class FallbackPredictionService:
    def __init__(self):
        self.primary_model = load_model("primary")
        self.fallback_model = load_model("fallback")
        self.default_prediction = {"class": "unknown", "confidence": 0.0}
    
    def predict(self, features: dict) -> dict:
        """Make prediction with fallback chain."""
        
        # Try primary model
        try:
            prediction = self.primary_model.predict(features)
            if prediction['confidence'] > 0.3:  # Confidence threshold
                return prediction
        except Exception as e:
            log_error(f"Primary model failed: {e}")
        
        # Try fallback model
        try:
            prediction = self.fallback_model.predict(features)
            prediction['fallback'] = True
            return prediction
        except Exception as e:
            log_error(f"Fallback model failed: {e}")
        
        # Return default prediction
        return {
            **self.default_prediction,
            'fallback': True,
            'default': True
        }
```

### 3. Feature Fallback

```python
class FeatureService:
    def __init__(self, feature_store, cache, default_features):
        self.feature_store = feature_store
        self.cache = cache
        self.default_features = default_features
    
    def get_features(self, entity_id: str) -> dict:
        """Get features with fallback chain."""
        
        # Try feature store
        try:
            features = self.feature_store.get(entity_id)
            self.cache.set(entity_id, features)  # Update cache
            return features
        except Exception as e:
            log_warning(f"Feature store unavailable: {e}")
        
        # Try cache
        cached_features = self.cache.get(entity_id)
        if cached_features:
            return {**cached_features, 'cached': True}
        
        # Return defaults
        return {**self.default_features, 'default': True}
```

---

## 📈 Reliability Metrics

### Service Level Objectives (SLOs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Availability** | 99.9% | Uptime / Total time |
| **Latency (p99)** | <200ms | 99th percentile response |
| **Error Rate** | <0.1% | Errors / Total requests |
| **Recovery Time** | <5 min | Time to recover from failure |

### Reliability Formula

```
Availability = (Total Time - Downtime) / Total Time

99.9% availability = 8.76 hours downtime/year
99.99% availability = 52.6 minutes downtime/year
```

---

## 🧪 Chaos Engineering

### Testing Failure Scenarios

```python
import random

class ChaosMonkey:
    """Inject failures to test system resilience."""
    
    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
    
    def maybe_fail(self, component: str):
        """Randomly fail based on failure rate."""
        if random.random() < self.failure_rate:
            raise Exception(f"Chaos Monkey: {component} failed!")
    
    def slow_down(self, latency_ms: int = 1000):
        """Add artificial latency."""
        if random.random() < self.failure_rate:
            time.sleep(latency_ms / 1000)

# Usage in tests
chaos = ChaosMonkey(failure_rate=0.1)

def predict_with_chaos(features):
    chaos.maybe_fail("model_server")
    chaos.maybe_fail("feature_store")
    return model.predict(features)
```

### Failure Scenarios to Test

- [ ] Model server crash and restart
- [ ] Feature store unavailable
- [ ] Network partition between services
- [ ] Slow responses (high latency)
- [ ] Database connection failures
- [ ] Cache failures
- [ ] Partial feature availability
- [ ] Model loading failures

---

## 🔑 Key Principles

1. **Design for failure** - assume every component can and will fail
2. **Redundancy** - multiple instances, replicas, backups
3. **Graceful degradation** - provide value even when degraded
4. **Quick recovery** - fast failure detection and automatic recovery
5. **Test failures** - regularly test failure scenarios
6. **Monitor health** - proactive health checks and alerting
7. **Isolate failures** - prevent cascade failures with circuit breakers

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Single point of failure** | One failure brings down system | Redundancy at every layer |
| **Cascade failures** | One failure causes others | Circuit breakers, bulkheads |
| **Silent failures** | System fails without notice | Health checks, monitoring |
| **Slow recovery** | Long downtime | Auto-scaling, auto-healing |
| **Untested fallbacks** | Fallbacks don't work | Regular testing |

---

## 📋 Reliability Checklist

- [ ] Multiple replicas for all critical services
- [ ] Health checks configured and working
- [ ] Circuit breakers implemented
- [ ] Fallback models/predictions available
- [ ] Feature fallbacks configured
- [ ] Timeouts set appropriately
- [ ] Auto-scaling configured
- [ ] Disaster recovery plan documented
- [ ] Regular failure testing (chaos engineering)
- [ ] Monitoring and alerting in place

---

## 🚀 Next Steps

- Learn about [High Availability](./01-high-availability.md) - ensure your system is always available
- Understand [Graceful Degradation](./02-graceful-degradation.md) - handle partial failures
- Explore [Circuit Breakers](./03-circuit-breakers.md) - prevent cascade failures
- Study [Disaster Recovery](./04-disaster-recovery.md) - plan for the worst

Then proceed to [Security & Privacy](../09-security-privacy/00-README.md) to protect your reliable systems.
