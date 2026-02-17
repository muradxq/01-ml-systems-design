# Graceful Degradation

## Overview

Graceful degradation allows ML systems to continue providing value when components fail, even if at reduced quality. Instead of returning errors, the system provides the best possible response given current constraints. This is essential for user-facing applications where a degraded response is better than no response.

---

## 🎯 Degradation Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Best Quality                                  │
│                    ┌─────────────────────────────────────┐      │
│  Level 1:          │  Full model + all features           │      │
│                    │  (100% accuracy)                     │      │
│                    └─────────────────────────────────────┘      │
│                                    │ Feature store fails         │
│                                    ▼                             │
│                    ┌─────────────────────────────────────┐      │
│  Level 2:          │  Full model + cached features        │      │
│                    │  (95% accuracy, potentially stale)   │      │
│                    └─────────────────────────────────────┘      │
│                                    │ Primary model fails         │
│                                    ▼                             │
│                    ┌─────────────────────────────────────┐      │
│  Level 3:          │  Fallback model + default features   │      │
│                    │  (80% accuracy, simpler model)       │      │
│                    └─────────────────────────────────────┘      │
│                                    │ All models fail             │
│                                    ▼                             │
│                    ┌─────────────────────────────────────┐      │
│  Level 4:          │  Cached predictions / Popular items  │      │
│                    │  (Not personalized)                  │      │
│                    └─────────────────────────────────────┘      │
│                                    │ Cache fails                 │
│                                    ▼                             │
│                    ┌─────────────────────────────────────┐      │
│  Level 5:          │  Static defaults                     │      │
│                    │  (Hardcoded fallback)                │      │
│                    └─────────────────────────────────────┘      │
│                    Worst Quality                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Implementation

### Complete Degradation Service

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

class DegradationLevel(Enum):
    FULL = "full"           # All systems working
    CACHED_FEATURES = "cached_features"
    FALLBACK_MODEL = "fallback_model"
    CACHED_PREDICTIONS = "cached_predictions"
    STATIC_DEFAULT = "static_default"

@dataclass
class PredictionResult:
    prediction: Any
    confidence: float
    degradation_level: DegradationLevel
    metadata: Dict[str, Any]

class GracefulPredictionService:
    """
    ML prediction service with graceful degradation.
    """
    
    def __init__(self):
        # Primary components
        self.primary_model = None
        self.fallback_model = None
        self.feature_store = None
        self.prediction_cache = None
        
        # Defaults
        self.default_features = self._load_default_features()
        self.popular_items = self._load_popular_items()
        self.static_default = {"items": [], "score": 0.0}
        
        # Monitoring
        self.degradation_counter = {}
        self.logger = logging.getLogger(__name__)
    
    async def predict(self, user_id: str, context: Dict) -> PredictionResult:
        """
        Make prediction with graceful degradation.
        Always returns a result, never throws to caller.
        """
        
        # Level 1: Try full model with fresh features
        try:
            features = await self._get_features(user_id)
            prediction = self.primary_model.predict(features)
            return PredictionResult(
                prediction=prediction,
                confidence=0.95,
                degradation_level=DegradationLevel.FULL,
                metadata={"features": "fresh", "model": "primary"}
            )
        except Exception as e:
            self.logger.warning(f"Primary path failed: {e}")
        
        # Level 2: Try full model with cached features
        try:
            cached_features = await self._get_cached_features(user_id)
            if cached_features:
                prediction = self.primary_model.predict(cached_features)
                return PredictionResult(
                    prediction=prediction,
                    confidence=0.85,
                    degradation_level=DegradationLevel.CACHED_FEATURES,
                    metadata={"features": "cached", "model": "primary"}
                )
        except Exception as e:
            self.logger.warning(f"Cached features path failed: {e}")
        
        # Level 3: Try fallback model with default features
        try:
            features = {**self.default_features, "user_id": user_id}
            prediction = self.fallback_model.predict(features)
            return PredictionResult(
                prediction=prediction,
                confidence=0.60,
                degradation_level=DegradationLevel.FALLBACK_MODEL,
                metadata={"features": "default", "model": "fallback"}
            )
        except Exception as e:
            self.logger.warning(f"Fallback model failed: {e}")
        
        # Level 4: Return cached prediction
        try:
            cached_pred = await self.prediction_cache.get(user_id)
            if cached_pred:
                return PredictionResult(
                    prediction=cached_pred,
                    confidence=0.40,
                    degradation_level=DegradationLevel.CACHED_PREDICTIONS,
                    metadata={"source": "prediction_cache"}
                )
        except Exception as e:
            self.logger.warning(f"Prediction cache failed: {e}")
        
        # Level 5: Return popular items (non-personalized)
        try:
            return PredictionResult(
                prediction=self.popular_items,
                confidence=0.20,
                degradation_level=DegradationLevel.CACHED_PREDICTIONS,
                metadata={"source": "popular_items"}
            )
        except Exception as e:
            self.logger.error(f"Popular items failed: {e}")
        
        # Level 6: Static default (always works)
        self._track_degradation(DegradationLevel.STATIC_DEFAULT)
        return PredictionResult(
            prediction=self.static_default,
            confidence=0.0,
            degradation_level=DegradationLevel.STATIC_DEFAULT,
            metadata={"source": "static_default"}
        )
    
    async def _get_features(self, user_id: str) -> Dict:
        """Get fresh features with timeout."""
        return await asyncio.wait_for(
            self.feature_store.get(user_id),
            timeout=0.05  # 50ms timeout
        )
    
    async def _get_cached_features(self, user_id: str) -> Optional[Dict]:
        """Get cached features."""
        return await self.prediction_cache.get(f"features:{user_id}")
    
    def _track_degradation(self, level: DegradationLevel):
        """Track degradation for monitoring."""
        if level not in self.degradation_counter:
            self.degradation_counter[level] = 0
        self.degradation_counter[level] += 1
        
        # Emit metric
        metrics.counter(
            "prediction_degradation",
            tags={"level": level.value}
        ).increment()
```

### Feature Fallbacks

```python
class FeatureService:
    """Feature service with fallbacks."""
    
    def __init__(self):
        self.feature_store = FeatureStoreClient()
        self.cache = RedisClient()
        self.default_features = {
            "user_age_bucket": "unknown",
            "user_tenure_days": 0,
            "total_purchases": 0,
            "avg_order_value": 50.0,
            "preferred_category": "general"
        }
    
    async def get_features(self, user_id: str, 
                          required: List[str],
                          optional: List[str]) -> Dict[str, Any]:
        """
        Get features with fallbacks for optional features.
        Required features will raise if unavailable.
        Optional features will use defaults.
        """
        result = {}
        
        # Try to get all features
        try:
            all_features = await self.feature_store.get(
                user_id, 
                required + optional
            )
            return all_features
        except Exception as e:
            self.logger.warning(f"Feature store failed: {e}")
        
        # Try cache
        try:
            cached = await self.cache.get(f"features:{user_id}")
            if cached:
                return cached
        except:
            pass
        
        # Build from defaults for optional features
        for feature in optional:
            result[feature] = self.default_features.get(feature)
        
        # Required features must come from somewhere
        for feature in required:
            if feature in self.default_features:
                result[feature] = self.default_features[feature]
            else:
                raise FeatureUnavailableError(f"Required feature {feature} unavailable")
        
        return result
```

### Model Fallback Chain

```python
class ModelFallbackChain:
    """Chain of models with decreasing complexity."""
    
    def __init__(self):
        self.models = [
            ("deep_learning", self._load_dl_model(), 0.95),
            ("gradient_boosting", self._load_gbm_model(), 0.85),
            ("logistic_regression", self._load_lr_model(), 0.70),
            ("rules_based", self._rules_based_predict, 0.50)
        ]
    
    def predict(self, features: Dict) -> PredictionResult:
        """Try models in order until one succeeds."""
        
        for model_name, model, expected_accuracy in self.models:
            try:
                # Try prediction with timeout
                prediction = self._predict_with_timeout(model, features)
                
                return PredictionResult(
                    prediction=prediction,
                    model_used=model_name,
                    expected_accuracy=expected_accuracy
                )
            except Exception as e:
                self.logger.warning(f"{model_name} failed: {e}")
                continue
        
        # All models failed - return default
        return PredictionResult(
            prediction=self._default_prediction(),
            model_used="default",
            expected_accuracy=0.0
        )
    
    def _rules_based_predict(self, features: Dict) -> float:
        """Simple rules-based fallback (always works)."""
        score = 0.5
        
        if features.get('total_purchases', 0) > 10:
            score += 0.2
        if features.get('days_since_last_visit', 999) < 7:
            score += 0.1
        if features.get('user_tenure_days', 0) > 365:
            score += 0.1
        
        return min(score, 1.0)
```

---

## 📊 Monitoring Degradation

```python
class DegradationMonitor:
    """Monitor and alert on degradation levels."""
    
    def __init__(self, alert_threshold: float = 0.1):
        self.alert_threshold = alert_threshold
        self.window_size = 1000
        self.recent_predictions = []
    
    def record(self, result: PredictionResult):
        """Record prediction result for monitoring."""
        self.recent_predictions.append(result.degradation_level)
        
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)
        
        self._check_alerts()
    
    def _check_alerts(self):
        """Check if degradation exceeds threshold."""
        if len(self.recent_predictions) < 100:
            return
        
        # Calculate degradation rate
        degraded = sum(
            1 for level in self.recent_predictions
            if level != DegradationLevel.FULL
        )
        rate = degraded / len(self.recent_predictions)
        
        if rate > self.alert_threshold:
            self._send_alert(rate)
    
    def get_stats(self) -> Dict:
        """Get degradation statistics."""
        from collections import Counter
        counts = Counter(self.recent_predictions)
        total = len(self.recent_predictions)
        
        return {
            level.value: {
                "count": counts.get(level, 0),
                "percentage": counts.get(level, 0) / total * 100
            }
            for level in DegradationLevel
        }
```

---

## ✅ Best Practices

1. **Plan degradation levels** - define hierarchy upfront
2. **Test each level** - verify fallbacks work
3. **Monitor degradation** - track which levels are used
4. **Communicate degradation** - return confidence/quality signals
5. **Set timeouts** - don't wait forever for failing services
6. **Cache aggressively** - more cache = better degradation
7. **Keep defaults fresh** - update popular items regularly

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Untested fallbacks** | Don't work when needed | Regular testing |
| **Silent degradation** | Don't know it's happening | Monitor and alert |
| **Stale defaults** | Poor fallback quality | Regular updates |
| **No timeouts** | Slow failures | Set aggressive timeouts |
| **Missing levels** | Gap in degradation chain | Complete fallback chain |

---

## 🔗 Related Topics

- [High Availability](./01-high-availability.md) - Prevent failures
- [Circuit Breakers](./03-circuit-breakers.md) - Fast failure detection
- [Caching Strategies](../07-scalability-performance/02-caching-strategies.md) - Cache for fallbacks
