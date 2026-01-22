# High Availability

## Overview

High availability (HA) ensures ML systems remain operational despite component failures. For production ML systems, this means users continue receiving predictions even when servers crash, models fail to load, or infrastructure has issues. HA is achieved through redundancy, automatic failover, and comprehensive health monitoring.

---

## 📊 Availability Targets

| Availability | Downtime/Year | Downtime/Month | Target For |
|--------------|---------------|----------------|------------|
| **99%** | 3.65 days | 7.2 hours | Development |
| **99.9%** | 8.76 hours | 43.8 minutes | Standard production |
| **99.99%** | 52.6 minutes | 4.32 minutes | Critical systems |
| **99.999%** | 5.26 minutes | 25.9 seconds | Mission critical |

---

## 🏗️ High Availability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Global Load Balancer                          │
│  (DNS-based, geo-routing)                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│      Region A (Primary)   │    │    Region B (Secondary)   │
│  ┌────────────────────┐  │    │  ┌────────────────────┐  │
│  │   Load Balancer    │  │    │  │   Load Balancer    │  │
│  └─────────┬──────────┘  │    │  └─────────┬──────────┘  │
│            │             │    │            │             │
│  ┌─────────┴──────────┐  │    │  ┌─────────┴──────────┐  │
│  │                    │  │    │  │                    │  │
│  ▼         ▼          │  │    │  ▼         ▼          │  │
│ ┌───┐    ┌───┐       │  │    │ ┌───┐    ┌───┐       │  │
│ │ S │    │ S │  ...  │  │    │ │ S │    │ S │  ...  │  │
│ │ 1 │    │ 2 │       │  │    │ │ 1 │    │ 2 │       │  │
│ └───┘    └───┘       │  │    │ └───┘    └───┘       │  │
│                       │  │    │                       │  │
│  ┌────────────────┐   │  │    │  ┌────────────────┐   │  │
│  │  Redis Cluster │   │  │    │  │  Redis Replica │   │  │
│  │   (Primary)    │◄──┼──┼────┼──│   (Secondary)  │   │  │
│  └────────────────┘   │  │    │  └────────────────┘   │  │
└──────────────────────────┘    └──────────────────────────┘
```

---

## 🎯 HA Strategies

### 1. Redundancy at Every Layer

```yaml
# Kubernetes deployment with redundancy
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3  # Multiple replicas
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero downtime updates
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      # Spread across nodes/zones
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: model-server
      # Anti-affinity: don't put all pods on same node
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: model-server
              topologyKey: kubernetes.io/hostname
      containers:
      - name: model-server
        image: model-server:v1.2.3
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 2. Health Checks

```python
from fastapi import FastAPI, Response
from datetime import datetime
import asyncio

app = FastAPI()

# Health check state
health_state = {
    'model_loaded': False,
    'feature_store_connected': False,
    'last_prediction': None,
    'startup_time': None
}

@app.on_event("startup")
async def startup():
    """Initialize and verify all components."""
    health_state['startup_time'] = datetime.utcnow()
    
    # Load model
    try:
        global model
        model = load_model()
        health_state['model_loaded'] = True
    except Exception as e:
        health_state['model_loaded'] = False
    
    # Connect to feature store
    try:
        await feature_store.connect()
        health_state['feature_store_connected'] = True
    except Exception as e:
        health_state['feature_store_connected'] = False

@app.get("/health/live")
async def liveness():
    """
    Liveness probe: Is the process running?
    If this fails, Kubernetes restarts the pod.
    """
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness(response: Response):
    """
    Readiness probe: Can this instance handle requests?
    If this fails, traffic is routed away.
    """
    checks = {
        'model_loaded': health_state['model_loaded'],
        'feature_store_connected': health_state['feature_store_connected']
    }
    
    all_healthy = all(checks.values())
    
    if not all_healthy:
        response.status_code = 503
    
    return {
        "ready": all_healthy,
        "checks": checks
    }

@app.get("/health/startup")
async def startup_probe(response: Response):
    """
    Startup probe: Has the service finished initializing?
    Gives slow-starting containers time to initialize.
    """
    if not health_state['model_loaded']:
        response.status_code = 503
        return {"status": "starting", "message": "Model not yet loaded"}
    
    return {"status": "started", "startup_time": health_state['startup_time']}
```

### 3. Automatic Failover

```python
class FailoverClient:
    """Client with automatic failover between endpoints."""
    
    def __init__(self, primary_url: str, secondary_url: str,
                 failover_threshold: int = 3):
        self.primary_url = primary_url
        self.secondary_url = secondary_url
        self.failover_threshold = failover_threshold
        
        self.primary_failures = 0
        self.using_secondary = False
        self.last_primary_check = 0
        self.primary_check_interval = 30  # seconds
    
    async def call(self, endpoint: str, data: dict) -> dict:
        """Make request with automatic failover."""
        
        # Check if should try primary again
        if self.using_secondary:
            if time.time() - self.last_primary_check > self.primary_check_interval:
                if await self._check_primary_health():
                    self.using_secondary = False
                    self.primary_failures = 0
        
        # Determine which endpoint to use
        url = self.secondary_url if self.using_secondary else self.primary_url
        
        try:
            response = await self._make_request(url, endpoint, data)
            
            if not self.using_secondary:
                self.primary_failures = 0  # Reset on success
            
            return response
            
        except Exception as e:
            if not self.using_secondary:
                self.primary_failures += 1
                
                if self.primary_failures >= self.failover_threshold:
                    self.using_secondary = True
                    self.last_primary_check = time.time()
                    
                    # Retry with secondary
                    return await self._make_request(
                        self.secondary_url, endpoint, data
                    )
            raise
    
    async def _check_primary_health(self) -> bool:
        """Check if primary is healthy again."""
        try:
            response = await self._make_request(
                self.primary_url, "/health/ready", {}
            )
            return response.get('ready', False)
        except:
            return False
```

---

## 📝 Multi-Region Setup

```python
# Multi-region configuration
REGIONS = {
    'us-east-1': {
        'model_server': 'https://model-east.example.com',
        'feature_store': 'redis://redis-east.example.com:6379',
        'priority': 1
    },
    'us-west-2': {
        'model_server': 'https://model-west.example.com',
        'feature_store': 'redis://redis-west.example.com:6379',
        'priority': 2
    },
    'eu-west-1': {
        'model_server': 'https://model-eu.example.com',
        'feature_store': 'redis://redis-eu.example.com:6379',
        'priority': 3
    }
}

class MultiRegionClient:
    """Client that routes to nearest healthy region."""
    
    def __init__(self, regions: dict):
        self.regions = sorted(
            regions.items(), 
            key=lambda x: x[1]['priority']
        )
        self.region_health = {r: True for r in regions}
    
    async def predict(self, features: dict) -> dict:
        """Make prediction using nearest healthy region."""
        for region_name, config in self.regions:
            if not self.region_health[region_name]:
                continue
            
            try:
                return await self._call_region(config, features)
            except Exception as e:
                self.region_health[region_name] = False
                asyncio.create_task(
                    self._check_region_health(region_name, config)
                )
        
        raise Exception("All regions unavailable")
```

---

## ✅ Best Practices

1. **No single points of failure** - redundancy at every layer
2. **Health checks at all levels** - liveness, readiness, startup
3. **Automatic failover** - don't wait for manual intervention
4. **Geographic distribution** - survive regional failures
5. **Test failure scenarios** - chaos engineering
6. **Monitor availability** - track uptime metrics
7. **Document runbooks** - clear recovery procedures

---

## 📊 HA Checklist

- [ ] Multiple replicas deployed (min 3)
- [ ] Spread across availability zones
- [ ] Liveness probes configured
- [ ] Readiness probes configured
- [ ] Load balancer health checks
- [ ] Automatic failover tested
- [ ] Multi-region (for critical systems)
- [ ] Runbooks documented
- [ ] Chaos testing performed

---

## 🔗 Related Topics

- [Graceful Degradation](./graceful-degradation.md) - Handle partial failures
- [Circuit Breakers](./circuit-breakers.md) - Prevent cascade failures
- [Disaster Recovery](./disaster-recovery.md) - Recover from major failures
