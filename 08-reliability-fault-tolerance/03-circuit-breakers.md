# Circuit Breakers

## Overview

Circuit breakers prevent cascading failures by stopping requests to failing services, allowing them time to recover. Named after electrical circuit breakers, they "trip" when failure rates exceed a threshold, immediately rejecting requests instead of waiting for timeouts. This protects both the failing service and the calling service.

---

## 🔄 Circuit Breaker States

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│    ┌──────────────┐         failures > threshold                │
│    │              │ ─────────────────────────────────┐          │
│    │    CLOSED    │                                  │          │
│    │   (Normal)   │ ◄────────────────────────────┐   │          │
│    │              │    success_count > threshold │   │          │
│    └──────────────┘                              │   │          │
│           │                                      │   │          │
│           │ all requests pass through            │   │          │
│           │ count failures                       │   │          │
│           ▼                                      │   ▼          │
│                                           ┌──────────────┐      │
│                                           │              │      │
│                                           │     OPEN     │      │
│                                           │   (Failed)   │      │
│                                           │              │      │
│                                           └──────┬───────┘      │
│                                                  │              │
│                                                  │ timeout      │
│                                                  │ expires      │
│                                                  ▼              │
│                                           ┌──────────────┐      │
│                                           │              │      │
│                                           │  HALF-OPEN   │      │
│                                           │  (Testing)   │      │
│                                           │              │      │
│                                           └──────────────┘      │
│                                                  │              │
│                               ┌──────────────────┴───────┐      │
│                               │                          │      │
│                         success                       failure   │
│                               │                          │      │
│                               ▼                          ▼      │
│                          back to                    back to     │
│                          CLOSED                     OPEN        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Implementation

### Complete Circuit Breaker

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import time
import threading
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout: float = 30.0               # Seconds before trying half-open
    half_open_max_calls: int = 3        # Max calls in half-open state
    failure_rate_threshold: float = 0.5 # Alternative: rate-based threshold
    sliding_window_size: int = 10       # Window for rate calculation

class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        
        # Sliding window for rate-based threshold
        self.call_results = []  # List of (timestamp, success: bool)
        
        self._lock = threading.Lock()
    
    def call(self, func: Callable, fallback: Callable = None, 
             *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        with self._lock:
            if not self._can_execute():
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            if fallback:
                return fallback(*args, **kwargs)
            raise
    
    async def call_async(self, func: Callable, fallback: Callable = None,
                        *args, **kwargs) -> Any:
        """Async version of call."""
        with self._lock:
            if not self._can_execute():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {self.name} is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            if fallback:
                return await fallback(*args, **kwargs)
            raise
    
    def _can_execute(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time > self.config.timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._record_call(True)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._record_call(False)
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._transition_to(CircuitState.OPEN)
            else:
                self.failure_count += 1
                
                # Check threshold
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)
    
    def _should_open(self) -> bool:
        """Check if circuit should open."""
        # Count-based threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Rate-based threshold
        if len(self.call_results) >= self.config.sliding_window_size:
            failures = sum(1 for _, success in self.call_results if not success)
            rate = failures / len(self.call_results)
            if rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _record_call(self, success: bool):
        """Record call in sliding window."""
        now = time.time()
        self.call_results.append((now, success))
        
        # Remove old entries
        cutoff = now - 60  # 60 second window
        self.call_results = [
            (t, s) for t, s in self.call_results if t > cutoff
        ][-self.config.sliding_window_size:]
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        
        # Reset counters
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
            self.half_open_calls = 0
        
        # Log transition
        logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")
        
        # Emit metric
        metrics.gauge(f"circuit_breaker.{self.name}.state", 
                     {"closed": 0, "half_open": 1, "open": 2}[new_state.value])
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time
        }

class CircuitOpenError(Exception):
    """Raised when circuit is open."""
    pass
```

### Usage Example

```python
# Create circuit breakers for different services
feature_store_circuit = CircuitBreaker(
    "feature_store",
    CircuitBreakerConfig(
        failure_threshold=5,
        timeout=30,
        success_threshold=3
    )
)

model_service_circuit = CircuitBreaker(
    "model_service",
    CircuitBreakerConfig(
        failure_threshold=3,
        timeout=60,
        success_threshold=2
    )
)

# Use in prediction service
class PredictionService:
    async def predict(self, user_id: str) -> dict:
        # Get features with circuit breaker
        features = await feature_store_circuit.call_async(
            func=lambda: self.feature_store.get(user_id),
            fallback=lambda: self.get_default_features(user_id)
        )
        
        # Get prediction with circuit breaker
        prediction = await model_service_circuit.call_async(
            func=lambda: self.model_client.predict(features),
            fallback=lambda: self.fallback_model.predict(features)
        )
        
        return prediction
```

### Circuit Breaker Registry

```python
class CircuitBreakerRegistry:
    """Central registry for all circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(self, name: str, 
                      config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_states(self) -> List[dict]:
        """Get state of all circuit breakers."""
        return [cb.get_state() for cb in self.breakers.values()]
    
    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuits."""
        return [
            name for name, cb in self.breakers.items()
            if cb.state == CircuitState.OPEN
        ]

# Global registry
circuit_registry = CircuitBreakerRegistry()

# Health endpoint
@app.get("/health/circuits")
async def circuit_health():
    states = circuit_registry.get_all_states()
    open_circuits = circuit_registry.get_open_circuits()
    
    return {
        "healthy": len(open_circuits) == 0,
        "open_circuits": open_circuits,
        "all_circuits": states
    }
```

---

## 📊 Circuit Breaker Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **State** | Current state (0=closed, 1=half-open, 2=open) | state == 2 |
| **Failure rate** | % of failed calls | >50% |
| **Open duration** | Time circuit has been open | >5 minutes |
| **Trip count** | Number of times opened | Unusual increase |

---

## ✅ Best Practices

1. **Tune thresholds** - based on service characteristics
2. **Use with fallbacks** - always have a fallback
3. **Monitor all circuits** - track states centrally
4. **Test circuit behavior** - verify it trips correctly
5. **Set appropriate timeouts** - don't wait too long
6. **Log state transitions** - for debugging
7. **Different configs per service** - not one-size-fits-all

---

## 🔗 Related Topics

- [High Availability](./high-availability.md) - System-wide availability
- [Graceful Degradation](./graceful-degradation.md) - What to do when circuit opens
- [Disaster Recovery](./disaster-recovery.md) - Recovering from failures
