# Caching Strategies

## Overview

Caching reduces latency and computation by storing frequently accessed data and predictions. In ML systems, caching can be applied at multiple levels: predictions, features, embeddings, and model artifacts. Effective caching can reduce latency by 10-100x for cached requests.

---

## 🏗️ Caching Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CDN / Edge Cache                              │
│  (Static responses, common predictions)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway Cache                             │
│  (Response caching, rate limiting)                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prediction Cache (Redis)                      │
│  Key: hash(features) → Value: prediction                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ (cache miss)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Server                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Local Model Cache                       │   │
│  │  (In-memory loaded models)                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Cache (Redis)                         │
│  Key: entity_id → Value: features                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │ (cache miss)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Store / Database                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Caching Patterns

### 1. Prediction Caching

**When to use:**
- Same inputs occur frequently
- Predictions don't need to be real-time fresh
- High computation cost per prediction

**Implementation:**

```python
import redis
import hashlib
import json
from typing import Dict, Any, Optional

class PredictionCache:
    def __init__(self, redis_host: str = 'localhost', 
                 ttl_seconds: int = 3600):
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.ttl = ttl_seconds
    
    def _generate_key(self, model_name: str, model_version: str, 
                     features: Dict[str, Any]) -> str:
        """Generate deterministic cache key from features."""
        # Sort features for consistent hashing
        feature_str = json.dumps(features, sort_keys=True)
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]
        return f"pred:{model_name}:{model_version}:{feature_hash}"
    
    def get(self, model_name: str, model_version: str, 
            features: Dict[str, Any]) -> Optional[Dict]:
        """Get cached prediction."""
        key = self._generate_key(model_name, model_version, features)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, model_name: str, model_version: str,
            features: Dict[str, Any], prediction: Dict):
        """Cache prediction."""
        key = self._generate_key(model_name, model_version, features)
        self.redis.setex(key, self.ttl, json.dumps(prediction))
    
    def invalidate_model(self, model_name: str, model_version: str):
        """Invalidate all predictions for a model version."""
        pattern = f"pred:{model_name}:{model_version}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

# Usage
cache = PredictionCache(ttl_seconds=3600)

def predict_with_cache(model_name: str, model_version: str, 
                       features: Dict) -> Dict:
    # Try cache first
    cached = cache.get(model_name, model_version, features)
    if cached:
        cached['cached'] = True
        return cached
    
    # Make prediction
    prediction = model.predict(features)
    
    # Cache result
    cache.set(model_name, model_version, features, prediction)
    
    return prediction
```

### 2. Feature Caching

**When to use:**
- Features are expensive to compute
- Same entities requested frequently
- Features don't change rapidly

```python
class FeatureCache:
    def __init__(self, redis_host: str = 'localhost',
                 default_ttl: int = 300):
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.default_ttl = default_ttl
        self.feature_ttls = {
            'user_profile': 3600,      # Stable, cache longer
            'user_activity': 60,        # Changes frequently
            'item_features': 86400,     # Very stable
            'real_time_features': 5     # Near real-time
        }
    
    def get_features(self, entity_type: str, entity_id: str,
                    feature_groups: list) -> Dict[str, Any]:
        """Get features for an entity, using cache where possible."""
        result = {}
        missing_groups = []
        
        # Check cache for each feature group
        for group in feature_groups:
            key = f"feat:{entity_type}:{entity_id}:{group}"
            cached = self.redis.get(key)
            if cached:
                result[group] = json.loads(cached)
            else:
                missing_groups.append(group)
        
        # Fetch missing from feature store
        if missing_groups:
            fetched = self.feature_store.get_features(
                entity_type, entity_id, missing_groups
            )
            
            # Cache fetched features
            for group, features in fetched.items():
                ttl = self.feature_ttls.get(group, self.default_ttl)
                key = f"feat:{entity_type}:{entity_id}:{group}"
                self.redis.setex(key, ttl, json.dumps(features))
                result[group] = features
        
        return result
```

### 3. Embedding Caching

**When to use:**
- Embeddings are expensive to compute
- Same items/users queried repeatedly
- Embeddings are stable over time

```python
import numpy as np

class EmbeddingCache:
    def __init__(self, redis_host: str = 'localhost',
                 embedding_dim: int = 128):
        self.redis = redis.Redis(host=redis_host)
        self.embedding_dim = embedding_dim
    
    def get_embedding(self, entity_type: str, 
                     entity_id: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = f"emb:{entity_type}:{entity_id}"
        cached = self.redis.get(key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)
        return None
    
    def set_embedding(self, entity_type: str, entity_id: str,
                     embedding: np.ndarray, ttl: int = 86400):
        """Cache embedding."""
        key = f"emb:{entity_type}:{entity_id}"
        self.redis.setex(key, ttl, embedding.tobytes())
    
    def get_batch_embeddings(self, entity_type: str,
                            entity_ids: list) -> Dict[str, np.ndarray]:
        """Get multiple embeddings efficiently."""
        keys = [f"emb:{entity_type}:{eid}" for eid in entity_ids]
        values = self.redis.mget(keys)
        
        result = {}
        missing = []
        
        for entity_id, value in zip(entity_ids, values):
            if value:
                result[entity_id] = np.frombuffer(value, dtype=np.float32)
            else:
                missing.append(entity_id)
        
        # Compute missing embeddings
        if missing:
            computed = self.compute_embeddings(entity_type, missing)
            for entity_id, embedding in computed.items():
                self.set_embedding(entity_type, entity_id, embedding)
                result[entity_id] = embedding
        
        return result
```

### 4. Model Artifact Caching

```python
import threading
from functools import lru_cache

class ModelCache:
    """Cache loaded model artifacts in memory."""
    
    def __init__(self, max_models: int = 10):
        self.max_models = max_models
        self.models = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get_model(self, model_name: str, version: str):
        """Get model, loading if necessary."""
        key = f"{model_name}:{version}"
        
        with self.lock:
            if key in self.models:
                self.access_times[key] = time.time()
                return self.models[key]
            
            # Evict LRU model if at capacity
            if len(self.models) >= self.max_models:
                self._evict_lru()
            
            # Load model
            model = self._load_model(model_name, version)
            self.models[key] = model
            self.access_times[key] = time.time()
            
            return model
    
    def _evict_lru(self):
        """Evict least recently used model."""
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.models[lru_key]
        del self.access_times[lru_key]
    
    def _load_model(self, model_name: str, version: str):
        """Load model from storage."""
        path = f"s3://models/{model_name}/{version}/model.pkl"
        return load_model(path)
```

---

## 📊 Cache Configuration Guidelines

| Cache Type | TTL | Invalidation Strategy |
|------------|-----|----------------------|
| **Predictions** | 1-60 min | Model update, TTL expiry |
| **User features** | 5-60 min | User activity, TTL |
| **Item features** | 1-24 hours | Item update events |
| **Embeddings** | 1-7 days | Model retrain |
| **Model artifacts** | Forever | Explicit invalidation |

---

## 🔧 Cache Metrics

```python
class CacheMetrics:
    def __init__(self):
        self.hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
        self.misses = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
        self.latency = Histogram('cache_latency_seconds', 'Cache latency',
                                 ['cache_type', 'operation'])
    
    def record_hit(self, cache_type: str):
        self.hits.labels(cache_type=cache_type).inc()
    
    def record_miss(self, cache_type: str):
        self.misses.labels(cache_type=cache_type).inc()
    
    def get_hit_rate(self, cache_type: str) -> float:
        hits = self.hits.labels(cache_type=cache_type)._value.get()
        misses = self.misses.labels(cache_type=cache_type)._value.get()
        total = hits + misses
        return hits / total if total > 0 else 0
```

---

## ✅ Best Practices

1. **Cache at multiple levels** - predictions, features, embeddings
2. **Use appropriate TTLs** - balance freshness vs performance
3. **Invalidate on updates** - don't serve stale data
4. **Monitor hit rates** - target >80% for high-traffic caches
5. **Handle cache failures** - degrade gracefully
6. **Warm caches** - pre-populate on deployment
7. **Use consistent hashing** - for distributed caches

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Stale predictions** | Model updated but cache not | Invalidate on deploy |
| **Cache stampede** | Many requests hit miss simultaneously | Locking, pre-warming |
| **Memory exhaustion** | Too much cached | Set max memory, LRU eviction |
| **Inconsistent keys** | Different servers generate different keys | Deterministic key generation |
| **No monitoring** | Don't know if cache is effective | Track hit rates |

---

## 🔗 Related Topics

- [Horizontal Scaling](./horizontal-scaling.md) - Scale with caching
- [Optimization Techniques](./optimization-techniques.md) - Other optimizations
- [Batch vs Real-time](./batch-vs-realtime.md) - Pre-compute for batch
