# Key Components of ML Systems

## Overview

ML systems consist of multiple interconnected components. Understanding these components and their interactions is essential for designing robust ML architectures.

---

## 🏗️ Core Components

### 1. Data Collection Layer

**Purpose:** Gather data from various sources

**Components:**
- Event streaming (Kafka, Kinesis)
- Database connectors
- API integrations
- File ingestion
- Web scraping

**Key Features:**
- Real-time and batch ingestion
- Data validation
- Schema enforcement
- Error handling

**Technologies:**
- Apache Kafka, AWS Kinesis
- Apache Flume, Logstash
- Custom connectors

---

### 2. Data Storage Layer

**Purpose:** Store raw and processed data efficiently

**Components:**
- Data lakes (raw data)
- Data warehouses (structured data)
- Feature stores (processed features)
- Model registries (model artifacts)

**Key Features:**
- Scalability
- Query performance
- Data versioning
- Access control

**Technologies:**
- Data Lakes: S3, Azure Data Lake, HDFS
- Data Warehouses: Snowflake, BigQuery, Redshift
- Feature Stores: Feast, Tecton, Hopsworks
- Model Registries: MLflow, Weights & Biases, SageMaker Model Registry

---

### 3. Data Processing Layer

**Purpose:** Transform raw data into features

**Components:**
- ETL/ELT pipelines
- Data validation
- Data cleaning
- Feature computation
- Data quality checks

**Key Features:**
- Batch and streaming processing
- Fault tolerance
- Scalability
- Reproducibility

**Technologies:**
- Batch: Apache Spark, Pandas, Dask
- Streaming: Apache Flink, Kafka Streams
- Orchestration: Airflow, Prefect, Dagster

---

### 4. Feature Store

**Purpose:** Centralized storage and serving of features

**Components:**
- Feature definitions
- Feature computation pipelines
- Online feature serving
- Offline feature storage
- Feature monitoring

**Key Features:**
- Consistent features (train/inference)
- Low-latency serving
- Feature versioning
- Feature discovery

**Technologies:**
- Feast, Tecton, Hopsworks
- Custom implementations

---

### 5. Model Training Infrastructure

**Purpose:** Train and validate models

**Components:**
- Training pipelines
- Experiment tracking
- Hyperparameter tuning
- Model validation
- Model registry

**Key Features:**
- Distributed training
- GPU support
- Experiment reproducibility
- Resource management

**Technologies:**
- Training: TensorFlow, PyTorch, XGBoost
- Experiment Tracking: MLflow, Weights & Biases, TensorBoard
- Hyperparameter Tuning: Optuna, Hyperopt, Ray Tune
- Orchestration: Kubeflow, SageMaker Pipelines

---

### 6. Model Serving Layer

**Purpose:** Serve model predictions in production

**Components:**
- Model servers
- API gateways
- Load balancers
- Caching layers
- A/B testing framework

**Key Features:**
- Low latency
- High throughput
- Scalability
- Model versioning

**Technologies:**
- Model Servers: TensorFlow Serving, TorchServe, Triton
- API Frameworks: FastAPI, Flask, gRPC
- Cloud Services: SageMaker, Vertex AI, Azure ML

---

### 7. Monitoring & Observability

**Purpose:** Track system and model health

**Components:**
- Metrics collection
- Logging infrastructure
- Alerting systems
- Dashboards
- Drift detection

**Key Features:**
- Real-time monitoring
- Historical analysis
- Anomaly detection
- Alerting

**Technologies:**
- Metrics: Prometheus, CloudWatch, Datadog
- Logging: ELK Stack, Splunk, CloudWatch Logs
- Monitoring: Grafana, DataDog, New Relic
- Drift Detection: Evidently AI, Fiddler, Arize

---

### 8. Orchestration Layer

**Purpose:** Coordinate workflows and pipelines

**Components:**
- Workflow schedulers
- Pipeline orchestration
- Dependency management
- Error handling
- Retry logic

**Key Features:**
- Workflow definition
- Scheduling
- Dependency resolution
- Failure handling

**Technologies:**
- Airflow, Prefect, Dagster
- Kubeflow Pipelines
- SageMaker Pipelines
- Azure ML Pipelines

---

## 🔗 Component Interactions

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Collection Layer                       │
│  (Kafka, Kinesis, Connectors)                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Storage Layer                          │
│  (Data Lakes, Warehouses, Feature Stores)               │
└───────┬───────────────────────────────┬─────────────────┘
        │                               │
        ▼                               ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Data Processing     │    │   Feature Store          │
│  (ETL Pipelines)     │───▶│   (Online/Offline)       │
└──────────────────────┘    └───────────┬──────────────┘
        │                               │
        │                               ▼
        │                    ┌──────────────────────────┐
        │                    │  Model Training         │
        │                    │  (Experiments, Tuning)   │
        │                    └───────────┬──────────────┘
        │                               │
        │                               ▼
        │                    ┌──────────────────────────┐
        │                    │  Model Registry          │
        │                    └───────────┬──────────────┘
        │                               │
        │                               ▼
        │                    ┌──────────────────────────┐
        │                    │  Model Serving           │
        │                    │  (APIs, Batch Jobs)      │
        │                    └───────────┬──────────────┘
        │                               │
        └───────────────────────────────┼──────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │  Monitoring & Observability   │
                        │  (Metrics, Logs, Alerts)      │
                        └───────────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │  Orchestration Layer          │
                        │  (Workflows, Pipelines)       │
                        └───────────────────────────────┘
```

---

## 🎯 Component Selection Criteria

### 1. Scale Requirements
- **Small Scale**: Simple tools (Pandas, Flask)
- **Medium Scale**: Managed services (SageMaker, Vertex AI)
- **Large Scale**: Distributed systems (Spark, Kubernetes)

### 2. Latency Requirements
- **Real-time (<100ms)**: In-memory feature stores, optimized model servers
- **Near real-time (<1s)**: Cached features, fast model servers
- **Batch**: Standard processing pipelines

### 3. Team Expertise
- **Data Scientists**: Higher-level tools (AutoML, managed services)
- **ML Engineers**: Flexible frameworks (MLflow, custom pipelines)
- **Platform Teams**: Infrastructure-focused (Kubernetes, custom systems)

### 4. Cost Considerations
- **Open Source**: Self-hosted (requires infrastructure)
- **Managed Services**: Pay-per-use (easier but more expensive)
- **Hybrid**: Critical components managed, others self-hosted

---

## 🛠️ Implementation Examples

### Complete Model Serving Component

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import redis
import mlflow
import numpy as np
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model', 'status'])
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')

app = FastAPI(title="ML Model Serving API")

class PredictionRequest(BaseModel):
    user_id: int
    features: Dict[str, Any] = None  # Optional pre-computed features

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    latency_ms: float

class ModelServer:
    """Production-ready model server with feature fetching and caching."""
    
    def __init__(self, model_name: str, model_version: str):
        self.model = self._load_model(model_name, model_version)
        self.model_version = model_version
        self.feature_store = redis.Redis(host='localhost', port=6379, db=0)
        self.prediction_cache = redis.Redis(host='localhost', port=6379, db=1)
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_name: str, version: str):
        """Load model from MLflow registry."""
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def _get_features(self, user_id: int) -> Dict[str, Any]:
        """Fetch features from feature store."""
        cache_key = f"features:{user_id}"
        cached = self.feature_store.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # Compute features if not cached
        features = self._compute_real_time_features(user_id)
        self.feature_store.setex(cache_key, 300, json.dumps(features))
        return features
    
    def _check_prediction_cache(self, user_id: int) -> Dict:
        """Check if recent prediction exists in cache."""
        cache_key = f"prediction:{user_id}"
        cached = self.prediction_cache.get(cache_key)
        return json.loads(cached) if cached else None
    
    @latency_histogram.time()
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction with caching and monitoring."""
        import time
        start_time = time.time()
        
        try:
            # Check cache first
            cached_prediction = self._check_prediction_cache(request.user_id)
            if cached_prediction:
                prediction_counter.labels(model=self.model_version, status='cache_hit').inc()
                return PredictionResponse(**cached_prediction)
            
            # Get features
            features = request.features or self._get_features(request.user_id)
            
            # Make prediction
            feature_vector = np.array([list(features.values())])
            prediction = self.model.predict(feature_vector)[0]
            confidence = self._get_confidence(feature_vector)
            
            # Cache result
            result = {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'model_version': self.model_version,
                'latency_ms': (time.time() - start_time) * 1000
            }
            self.prediction_cache.setex(
                f"prediction:{request.user_id}", 
                60,  # 1 minute TTL
                json.dumps(result)
            )
            
            prediction_counter.labels(model=self.model_version, status='success').inc()
            return PredictionResponse(**result)
            
        except Exception as e:
            prediction_counter.labels(model=self.model_version, status='error').inc()
            self.logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize server
model_server = ModelServer(model_name="fraud_detector", model_version="production")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return model_server.predict(request)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": model_server.model_version}

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

---

### Feature Store Integration Example

```python
from feast import FeatureStore
from datetime import datetime, timedelta

class FeatureService:
    """Service for feature retrieval with offline/online support."""
    
    def __init__(self, repo_path: str):
        self.store = FeatureStore(repo_path=repo_path)
    
    def get_online_features(self, entity_rows: List[Dict]) -> Dict:
        """Get features for real-time inference."""
        feature_vector = self.store.get_online_features(
            entity_rows=entity_rows,
            features=[
                "user_features:age",
                "user_features:total_purchases",
                "user_features:avg_order_value",
                "user_features:days_since_signup"
            ]
        )
        return feature_vector.to_dict()
    
    def get_training_features(self, entity_df, start_date: datetime, end_date: datetime):
        """Get historical features for model training."""
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=[
                "user_features:age",
                "user_features:total_purchases",
                "user_features:avg_order_value",
                "user_features:days_since_signup"
            ]
        )
        return training_df.to_df()
```

---

## 📊 Component Technology Stack Reference

### Starter Stack (Small Scale)
| Component | Technology |
|-----------|------------|
| Data Storage | PostgreSQL + S3 |
| Feature Store | Feast (local) |
| Training | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI + Docker |
| Monitoring | Prometheus + Grafana |
| Orchestration | Airflow |

### Production Stack (Medium Scale)
| Component | Technology |
|-----------|------------|
| Data Storage | Snowflake + S3 |
| Feature Store | Feast (Redis backend) |
| Training | PyTorch/TensorFlow on GPU |
| Experiment Tracking | MLflow + Weights & Biases |
| Model Serving | TorchServe/TF Serving + Kubernetes |
| Monitoring | Datadog + Evidently AI |
| Orchestration | Kubeflow Pipelines |

### Enterprise Stack (Large Scale)
| Component | Technology |
|-----------|------------|
| Data Storage | Delta Lake + Snowflake |
| Feature Store | Tecton / Databricks Feature Store |
| Training | Distributed training on Kubernetes |
| Experiment Tracking | Custom + MLflow |
| Model Serving | Triton Inference Server |
| Monitoring | Custom platform + Fiddler |
| Orchestration | Custom + Kubeflow |

---

## 🎯 Interview Questions

**Q1: Design an ML system for real-time fraud detection.**

**Answer Framework:**
```
1. Data Layer:
   - Event streaming (Kafka) for transactions
   - Feature store for user profiles
   - Data lake for historical transactions

2. Feature Layer:
   - Real-time features: current transaction amount, velocity
   - Batch features: historical patterns, user profile
   - Feature computation: Flink for streaming

3. Model Layer:
   - Model: Gradient boosting or neural network
   - Serving: Low-latency API (<100ms)
   - Updates: Hourly retraining on new fraud patterns

4. Monitoring:
   - Prediction distribution monitoring
   - Latency tracking
   - Drift detection
```

**Q2: How would you handle a 100x increase in prediction requests?**

**Answer Framework:**
- Horizontal scaling (Kubernetes HPA)
- Prediction caching
- Model optimization (quantization)
- Batch predictions where possible
- Load balancing

**Q3: What happens if your feature store goes down?**

**Answer Framework:**
- Fallback to cached features
- Default feature values
- Graceful degradation
- Circuit breaker pattern
- Multi-region deployment

---

## 🔑 Key Takeaways

1. **Each component has a specific purpose** - understand what each does
2. **Components interact** - design interfaces carefully
3. **Choose based on requirements** - scale, latency, team, cost
4. **Start simple** - add complexity as needed
5. **Monitor everything** - each component needs observability
6. **Plan for failure** - design fallbacks and redundancy

---

## 📚 Further Reading

- [MLOps Stack Canvas](https://ml-ops.org/content/mlops-stack-canvas)
- [The ML Test Score: A Rubric for ML Production Readiness](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7adfe23ed72af17065adf1df2da94.pdf)
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

---

## 🔗 Related Topics

- [ML vs Traditional Software](./ml-vs-traditional-software.md)
- [ML System Lifecycle](./ml-system-lifecycle.md)
- [Common Challenges](./common-challenges.md)
