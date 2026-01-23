# Horizontal Scaling

## Overview

Horizontal scaling adds more instances to handle increased load. For ML systems, this means deploying multiple model servers that can handle predictions in parallel. It's the primary strategy for handling production traffic at scale.

---

## 🏗️ Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Traffic                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer                               │
│  - Round Robin / Least Connections / IP Hash                    │
│  - Health checks                                                 │
│  - SSL termination                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Model Server 1  │ │  Model Server 2  │ │  Model Server N  │
│  ┌────────────┐  │ │  ┌────────────┐  │ │  ┌────────────┐  │
│  │   Model    │  │ │  │   Model    │  │ │  │   Model    │  │
│  │   (v1.2)   │  │ │  │   (v1.2)   │  │ │  │   (v1.2)   │  │
│  └────────────┘  │ │  └────────────┘  │ │  └────────────┘  │
│  ┌────────────┐  │ │  ┌────────────┐  │ │  ┌────────────┐  │
│  │ Local Cache│  │ │  │ Local Cache│  │ │  │ Local Cache│  │
│  └────────────┘  │ │  └────────────┘  │ │  └────────────┘  │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Services                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Feature     │  │  Prediction │  │   Model     │            │
│  │ Store       │  │  Cache      │  │   Registry  │            │
│  │ (Redis)     │  │  (Redis)    │  │   (S3)      │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Scaling Strategies

### 1. Auto-scaling

**Metrics to scale on:**
- CPU utilization (>70%)
- Memory utilization (>80%)
- Request rate (QPS)
- Custom metrics (latency, queue depth)

**Kubernetes HPA Example:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 3
  maxReplicas: 50
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
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
        name: prediction_latency_p99_seconds
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
  - type: External
    external:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
```

### 2. Load Balancing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Round Robin** | Rotate through servers | Uniform requests |
| **Least Connections** | Route to least busy | Variable request duration |
| **IP Hash** | Consistent routing | Session affinity |
| **Weighted** | Route by server capacity | Heterogeneous servers |
| **Random** | Random selection | Simple, good distribution |

**NGINX Load Balancer Config:**

```nginx
upstream model_servers {
    least_conn;  # Use least connections
    
    server model-server-1:8080 weight=1;
    server model-server-2:8080 weight=1;
    server model-server-3:8080 weight=1;
    
    # Health checks
    keepalive 32;
}

server {
    listen 80;
    
    location /predict {
        proxy_pass http://model_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;
        
        # Retry on failure
        proxy_next_upstream error timeout http_502 http_503;
        proxy_next_upstream_tries 3;
    }
    
    location /health {
        proxy_pass http://model_servers;
    }
}
```

### 3. Stateless Design

**Principles:**
- No local state between requests
- All state in external stores
- Any instance can handle any request
- Easy to add/remove instances

**Implementation:**

```python
from fastapi import FastAPI
import redis
import boto3

app = FastAPI()

# External state stores
redis_client = redis.Redis(host='redis-cluster', port=6379)
s3_client = boto3.client('s3')

class StatelessModelServer:
    def __init__(self):
        # Load model from shared storage
        self.model = self._load_model_from_s3()
    
    def _load_model_from_s3(self):
        """Load model from shared storage - same for all instances."""
        model_path = s3_client.download_file(
            'models-bucket', 
            'production/model.pkl',
            '/tmp/model.pkl'
        )
        return load_model('/tmp/model.pkl')
    
    def predict(self, features: dict, request_id: str) -> dict:
        # Check shared cache first
        cache_key = f"pred:{hash_features(features)}"
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Cache in shared store
        redis_client.setex(cache_key, 3600, json.dumps(prediction))
        
        return prediction

server = StatelessModelServer()

@app.post("/predict")
async def predict(request: PredictionRequest):
    return server.predict(request.features, request.request_id)
```

---

## 📊 Scaling Metrics

| Metric | Scale Up When | Scale Down When |
|--------|---------------|-----------------|
| **CPU** | >70% for 2 min | <30% for 10 min |
| **Memory** | >80% | <40% |
| **Latency (p99)** | >100ms for 5 min | <50ms for 10 min |
| **Queue Depth** | >10 requests | <2 requests |
| **Error Rate** | >1% | N/A |

---

## 🔧 Implementation Patterns

### GPU Scaling

```yaml
# Kubernetes GPU deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-model-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: model-server
        image: model-server:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

### Multi-Model Serving

```python
class MultiModelServer:
    """Serve multiple models on same infrastructure."""
    
    def __init__(self):
        self.models = {}
        self.model_lock = threading.Lock()
    
    def get_model(self, model_name: str, version: str):
        """Lazy load models as needed."""
        key = f"{model_name}:{version}"
        
        if key not in self.models:
            with self.model_lock:
                if key not in self.models:
                    self.models[key] = self._load_model(model_name, version)
        
        return self.models[key]
    
    def predict(self, model_name: str, version: str, features: dict):
        model = self.get_model(model_name, version)
        return model.predict(features)
```

---

## ✅ Best Practices

1. **Design stateless from the start** - no local state
2. **Use auto-scaling** - don't manually scale
3. **Scale on the right metric** - latency > CPU for user-facing
4. **Set appropriate min/max** - ensure baseline and cap costs
5. **Handle cold starts** - pre-warm instances
6. **Test scaling behavior** - load test regularly
7. **Monitor scaling events** - track scale up/down

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Local state** | Different instances return different results | Use shared state stores |
| **Cold start latency** | New instances slow | Pre-load models, keep-alive |
| **Scaling too fast** | Oscillation, thrashing | Use stabilization windows |
| **Scaling too slow** | Requests fail during spikes | Aggressive scale-up policies |
| **Wrong metric** | Scales but latency still high | Use latency-based scaling |

---

## 🔗 Related Topics

- [Caching Strategies](./caching-strategies.md) - Reduce load with caching
- [Optimization Techniques](./optimization-techniques.md) - Make each instance faster
- [Batch vs Real-time](./batch-vs-realtime.md) - Processing patterns
