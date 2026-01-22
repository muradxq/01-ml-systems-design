# ⚡ Scalability & Performance

## Overview

Scalability and performance are critical for production ML systems. Systems must handle increasing load while maintaining low latency and high throughput. ML systems have unique scaling challenges due to compute-intensive models, large data requirements, and complex feature pipelines.

---

## 🎯 Learning Objectives

After completing this section, you should understand:
- Horizontal and vertical scaling strategies for ML systems
- Caching strategies for features, predictions, and model artifacts
- Trade-offs between batch and real-time processing
- Model optimization techniques for production

---

## 📚 Topics Covered

1. [Horizontal Scaling](./horizontal-scaling.md)
   - Load balancing and auto-scaling
   - Stateless design patterns
   - Distributed inference

2. [Caching Strategies](./caching-strategies.md)
   - Feature caching
   - Prediction caching
   - Model caching

3. [Batch vs Real-time](./batch-vs-realtime.md)
   - When to use each approach
   - Hybrid architectures
   - Cost vs latency trade-offs

4. [Optimization Techniques](./optimization-techniques.md)
   - Model compression
   - Hardware acceleration
   - Serving optimizations

---

## 🏗️ Scalability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Requests                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer / CDN                         │
│  (Route requests, SSL termination, basic caching)               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway                                 │
│  (Rate limiting, authentication, request routing)               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Prediction      │ │  Prediction      │ │  Prediction      │
│  Service         │ │  Service         │ │  Service         │
│  (Instance 1)    │ │  (Instance 2)    │ │  (Instance N)    │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Prediction      │ │  Feature         │ │  Model           │
│  Cache           │ │  Cache           │ │  Cache           │
│  (Redis)         │ │  (Redis)         │ │  (Local/CDN)     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 📊 Performance Targets

### Latency Targets by Use Case

| Use Case | Target Latency | Acceptable Range |
|----------|---------------|------------------|
| **Real-time recommendations** | <50ms | 20-100ms |
| **Search ranking** | <100ms | 50-200ms |
| **Fraud detection** | <100ms | 50-200ms |
| **Image classification** | <200ms | 100-500ms |
| **Batch predictions** | Minutes-hours | Depends on SLA |

### Throughput Targets

| System Type | Target QPS | Scaling Strategy |
|-------------|-----------|------------------|
| **Small** | <100 QPS | Single instance |
| **Medium** | 100-1000 QPS | Horizontal scaling |
| **Large** | 1000-10000 QPS | Distributed + caching |
| **Very Large** | >10000 QPS | Edge deployment + aggressive caching |

---

## 🔧 Scaling Strategies

### 1. Horizontal Scaling

```python
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: prediction_latency_p95
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
```

### 2. Prediction Caching

```python
import redis
import hashlib
import json

class PredictionCache:
    def __init__(self, redis_host='localhost', ttl=3600):
        self.redis = redis.Redis(host=redis_host)
        self.ttl = ttl
    
    def _cache_key(self, features: dict) -> str:
        """Generate cache key from features."""
        feature_str = json.dumps(features, sort_keys=True)
        return f"pred:{hashlib.md5(feature_str.encode()).hexdigest()}"
    
    def get(self, features: dict) -> dict | None:
        """Get cached prediction."""
        key = self._cache_key(features)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, features: dict, prediction: dict):
        """Cache prediction."""
        key = self._cache_key(features)
        self.redis.setex(key, self.ttl, json.dumps(prediction))

# Usage
cache = PredictionCache()

def predict_with_cache(features: dict) -> dict:
    # Check cache first
    cached = cache.get(features)
    if cached:
        return cached
    
    # Make prediction
    prediction = model.predict(features)
    
    # Cache result
    cache.set(features, prediction)
    
    return prediction
```

### 3. Model Optimization

```python
import torch
import torch.quantization as quant

def optimize_model(model, example_input):
    """Apply various optimization techniques."""
    
    # 1. Dynamic Quantization (reduce precision)
    quantized_model = quant.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8
    )
    
    # 2. TorchScript compilation (optimize execution)
    scripted_model = torch.jit.trace(quantized_model, example_input)
    
    # 3. Optimize for inference
    optimized_model = torch.jit.optimize_for_inference(scripted_model)
    
    return optimized_model

# ONNX Export for cross-platform optimization
def export_to_onnx(model, example_input, output_path):
    """Export model to ONNX format."""
    torch.onnx.export(
        model,
        example_input,
        output_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
```

---

## 📈 Optimization Comparison

| Technique | Latency Reduction | Memory Reduction | Accuracy Impact |
|-----------|------------------|------------------|-----------------|
| **Quantization (INT8)** | 2-4x faster | 4x smaller | <1% loss |
| **Pruning** | 2-3x faster | 2-10x smaller | 1-2% loss |
| **Knowledge Distillation** | 2-10x faster | Variable | 1-5% loss |
| **ONNX Runtime** | 1.5-2x faster | Same | None |
| **TensorRT** | 2-5x faster | Same | <0.5% loss |
| **Batching** | 2-10x throughput | Higher peak | None |

---

## 🔑 Key Principles

1. **Scale horizontally** - add more instances rather than bigger machines
2. **Cache aggressively** - features, predictions, model artifacts
3. **Optimize models** - quantization, pruning, distillation
4. **Monitor performance** - latency percentiles, throughput, errors
5. **Plan for growth** - design for 10x current load
6. **Measure first** - profile before optimizing
7. **Test at scale** - load test regularly

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Premature optimization** | Optimize wrong things | Profile first, optimize bottlenecks |
| **Cache invalidation** | Stale predictions | TTL strategy, invalidation events |
| **Cold start latency** | Slow first request | Model warming, keep-alive |
| **Memory leaks** | Growing memory usage | Monitor, restart policies |
| **GPU underutilization** | Expensive, slow | Batch requests, right-size |

---

## 📊 Performance Checklist

- [ ] Latency p50, p95, p99 within targets
- [ ] Throughput meets expected QPS
- [ ] Auto-scaling configured and tested
- [ ] Caching strategy implemented
- [ ] Model optimized (quantization, etc.)
- [ ] Load testing completed
- [ ] Monitoring and alerting in place
- [ ] Cold start latency acceptable
- [ ] Resource utilization optimized

---

## 🚀 Next Steps

- Learn about [Horizontal Scaling](./horizontal-scaling.md) - scale your inference infrastructure
- Understand [Caching Strategies](./caching-strategies.md) - reduce latency with caching
- Explore [Batch vs Real-time](./batch-vs-realtime.md) - choose the right processing pattern
- Study [Optimization Techniques](./optimization-techniques.md) - make models faster

Then proceed to [Reliability & Fault Tolerance](../08-reliability-fault-tolerance/README.md) to ensure your scaled systems are reliable.
