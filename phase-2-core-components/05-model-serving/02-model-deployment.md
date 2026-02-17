# Model Deployment

## Overview

Model deployment moves trained models from development to production environments where they can serve predictions. Effective deployment requires containerization, infrastructure automation, safe rollout strategies, and observability. The goal is to deploy frequently with confidence while minimizing user impact from issues.

---

## 🏗️ Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Deployment Pipeline                                       │
│                                                                  │
│  Model Registry ───▶ CI/CD ───▶ Container Registry ───▶ K8s    │
│        │                │              │                  │      │
│        ▼                ▼              ▼                  ▼      │
│  Versioned Model    Build &     Docker Image      Deployment    │
│  + Metadata         Test                           + Rollout    │
│                                                                  │
│  Components:                                                     │
│  - Model artifacts + dependencies                               │
│  - Serving code (API, preprocessing)                            │
│  - Infrastructure configuration                                 │
│  - Monitoring & alerting                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Containerization

### Dockerfile for ML Model

```dockerfile
# Multi-stage build for smaller image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.onnx
ENV CONFIG_PATH=/app/config/serving.yaml

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GPU-Enabled Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements-gpu.txt

WORKDIR /app
COPY . .

# Runtime environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000
CMD ["python", "-m", "src.server"]
```

### requirements.txt

```
# Core
fastapi==0.104.0
uvicorn[standard]==0.24.0
pydantic==2.5.0

# ML
torch==2.1.0
onnxruntime==1.16.0
numpy==1.24.0
pandas==2.0.0

# Monitoring
prometheus-client==0.18.0
structlog==23.2.0

# Feature Store
redis==5.0.0

# Cloud
boto3==1.29.0
```

---

## ☸️ Kubernetes Deployment

### Complete Kubernetes Manifests

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-serving
  labels:
    name: ml-serving

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-server-config
  namespace: ml-serving
data:
  serving.yaml: |
    model:
      name: fraud-detection
      version: v1.2.0
      path: /models/model.onnx
    
    server:
      host: 0.0.0.0
      port: 8000
      workers: 4
    
    features:
      store_url: redis://redis:6379
      cache_ttl: 300
    
    logging:
      level: INFO
      format: json

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-server-secrets
  namespace: ml-serving
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: "your-access-key"
  AWS_SECRET_ACCESS_KEY: "your-secret-key"

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  namespace: ml-serving
  labels:
    app: model-server
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: model-server
        version: v1.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: model-server
      terminationGracePeriodSeconds: 60
      
      # Init container to download model
      initContainers:
      - name: model-downloader
        image: amazon/aws-cli:latest
        command:
        - /bin/sh
        - -c
        - |
          aws s3 cp s3://models/fraud-detection/v1.2.0/model.onnx /models/model.onnx
        volumeMounts:
        - name: model-volume
          mountPath: /models
        envFrom:
        - secretRef:
            name: model-server-secrets
      
      containers:
      - name: model-server
        image: myregistry/model-server:v1.2.0
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        
        # Environment variables
        env:
        - name: MODEL_VERSION
          value: "v1.2.0"
        - name: LOG_LEVEL
          value: "INFO"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        
        envFrom:
        - secretRef:
            name: model-server-secrets
        
        # Resources
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        # Probes
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 30  # 30 * 5 = 150s max startup time
        
        # Volume mounts
        volumeMounts:
        - name: model-volume
          mountPath: /models
          readOnly: true
        - name: config-volume
          mountPath: /config
          readOnly: true
        
        # Lifecycle hooks
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - sleep 10  # Allow time for load balancer to remove pod
      
      volumes:
      - name: model-volume
        emptyDir: {}
      - name: config-volume
        configMap:
          name: model-server-config
      
      # Affinity and anti-affinity
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: model-server
              topologyKey: kubernetes.io/hostname
      
      # Topology spread for HA
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: model-server

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
  namespace: ml-serving
  labels:
    app: model-server
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: model-server

---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 3
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
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60

---
# pdb.yaml - Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: model-server-pdb
  namespace: ml-serving
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: model-server

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-server-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: tls-secret
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1/predict
        pathType: Prefix
        backend:
          service:
            name: model-server
            port:
              number: 80
```

---

## 🚀 Deployment Strategies

### 1. Rolling Update

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%      # Create up to 25% extra pods
      maxUnavailable: 0   # Never reduce below desired count
```

### 2. Blue-Green Deployment

```yaml
# blue-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-blue
  labels:
    app: model-server
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      version: blue
  template:
    metadata:
      labels:
        app: model-server
        version: blue
    spec:
      containers:
      - name: model-server
        image: myregistry/model-server:v1.2.0

---
# green-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-green
  labels:
    app: model-server
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      version: green
  template:
    metadata:
      labels:
        app: model-server
        version: green
    spec:
      containers:
      - name: model-server
        image: myregistry/model-server:v1.3.0

---
# service.yaml - Switch traffic by changing selector
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  selector:
    app: model-server
    version: blue  # Change to 'green' to switch
  ports:
  - port: 80
    targetPort: 8000
```

### 3. Canary Deployment with Istio

```yaml
# virtualservice.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-server
spec:
  hosts:
  - model-server
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: model-server
        subset: canary
  - route:
    - destination:
        host: model-server
        subset: stable
      weight: 95
    - destination:
        host: model-server
        subset: canary
      weight: 5

---
# destinationrule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: model-server
spec:
  host: model-server
  subsets:
  - name: stable
    labels:
      version: v1.2.0
  - name: canary
    labels:
      version: v1.3.0
```

---

## 🛠️ Serverless Deployment

### AWS Lambda

```python
# lambda_handler.py
import json
import boto3
import numpy as np
import onnxruntime as ort

# Load model at cold start
s3 = boto3.client('s3')
s3.download_file('models-bucket', 'model.onnx', '/tmp/model.onnx')
session = ort.InferenceSession('/tmp/model.onnx')
input_name = session.get_inputs()[0].name

def handler(event, context):
    """Lambda handler for model inference."""
    try:
        # Parse input
        body = json.loads(event.get('body', '{}'))
        features = np.array([body.get('features')], dtype=np.float32)
        
        # Inference
        outputs = session.run(None, {input_name: features})
        prediction = outputs[0][0]
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': float(prediction),
                'model_version': 'v1.2.0'
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### AWS SAM Template

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    MemorySize: 1024

Resources:
  ModelFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_handler.handler
      Runtime: python3.10
      Architectures:
        - x86_64
      Events:
        Predict:
          Type: Api
          Properties:
            Path: /predict
            Method: post
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref ModelBucket
      
      # Provisioned concurrency for warm starts
      ProvisionedConcurrencyConfig:
        ProvisionedConcurrentExecutions: 5
      
      # Auto-scaling
      AutoPublishAlias: live
      DeploymentPreference:
        Type: Linear10PercentEvery1Minute
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Model Server

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'src/**'
      - 'Dockerfile'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/model-server

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest tests/ -v --cov=src
    
    - name: Model validation
      run: python scripts/validate_model.py

  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=sha,prefix=
          type=ref,event=branch
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      uses: azure/k8s-deploy@v4
      with:
        namespace: ml-serving-staging
        manifests: k8s/staging/
        images: ${{ needs.build.outputs.image_tag }}

  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
    - name: Run integration tests
      run: |
        python tests/integration/test_staging.py

  deploy-production:
    needs: [build, integration-tests]
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy canary (5%)
      uses: azure/k8s-deploy@v4
      with:
        namespace: ml-serving
        strategy: canary
        percentage: 5
        manifests: k8s/production/
        images: ${{ needs.build.outputs.image_tag }}
    
    - name: Wait and validate
      run: |
        sleep 300
        python scripts/validate_canary.py
    
    - name: Promote to full rollout
      uses: azure/k8s-deploy@v4
      with:
        namespace: ml-serving
        strategy: canary
        percentage: 100
        manifests: k8s/production/
        images: ${{ needs.build.outputs.image_tag }}
```

---

## ✅ Best Practices

### 1. Deployment Checklist

```markdown
## Pre-deployment
- [ ] Model validated on test data
- [ ] API contract unchanged (or versioned)
- [ ] Resource requirements estimated
- [ ] Rollback plan documented
- [ ] Alerts configured

## During deployment
- [ ] Canary/gradual rollout
- [ ] Monitor error rates
- [ ] Monitor latency
- [ ] Monitor business metrics

## Post-deployment
- [ ] Verify model version in logs
- [ ] Check prediction distribution
- [ ] Validate feature retrieval
- [ ] Update documentation
```

### 2. Health Endpoints

```python
from fastapi import FastAPI, Response
from datetime import datetime
import time

app = FastAPI()

startup_time = time.time()
model_loaded = False

@app.get("/health/live")
async def liveness():
    """Liveness probe - is the process running?"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness(response: Response):
    """Readiness probe - can it serve traffic?"""
    checks = {
        "model_loaded": model_loaded,
        "feature_store_connected": await check_feature_store(),
        "startup_complete": startup_time > 0
    }
    
    if not all(checks.values()):
        response.status_code = 503
    
    return {"ready": all(checks.values()), "checks": checks}

@app.get("/health/startup")
async def startup(response: Response):
    """Startup probe - is initialization complete?"""
    if not model_loaded:
        response.status_code = 503
        return {"status": "starting"}
    return {"status": "started", "startup_time_seconds": time.time() - startup_time}
```

---

## 🔗 Related Topics

- [Serving Patterns](./01-serving-patterns.md) - Choose serving approach
- [A/B Testing](./03-ab-testing.md) - Test deployments
- [Model Updates](./04-model-updates.md) - Update strategies
- [High Availability](../../phase-3-operations-and-reliability/08-reliability-fault-tolerance/01-high-availability.md) - Ensure uptime
