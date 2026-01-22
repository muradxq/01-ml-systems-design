# ML System Design Cheatsheet

Quick reference for ML system design concepts, tools, and patterns.

---

## 🏗️ System Architecture Template

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data        │ ──▶ │  Feature     │ ──▶ │  Model       │
│  Collection  │     │  Engineering │     │  Training    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Monitoring  │ ◀── │  Model       │ ◀── │  Model       │
│              │     │  Serving     │     │  Registry    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📊 Metrics Quick Reference

### Classification Metrics
| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/Total | Balanced classes |
| **Precision** | TP/(TP+FP) | FP is costly |
| **Recall** | TP/(TP+FN) | FN is costly |
| **F1** | 2×P×R/(P+R) | Balance P/R |
| **AUC-ROC** | Area under ROC | Ranking quality |
| **AUC-PR** | Area under PR | Imbalanced data |

### Regression Metrics
| Metric | Use When |
|--------|----------|
| **MAE** | Robust to outliers |
| **MSE/RMSE** | Penalize large errors |
| **MAPE** | Percentage errors |
| **R²** | Explained variance |

### System Metrics
| Metric | Target |
|--------|--------|
| **Latency p50** | <50ms |
| **Latency p99** | <200ms |
| **Throughput** | Based on traffic |
| **Error rate** | <0.1% |
| **Availability** | 99.9%+ |

---

## 🛠️ Technology Stack

### Data Storage
| Use Case | Technology |
|----------|------------|
| **Raw data** | S3, GCS, HDFS |
| **Structured data** | Snowflake, BigQuery, Redshift |
| **Real-time features** | Redis, DynamoDB |
| **Model artifacts** | S3 + MLflow/W&B |

### Processing
| Use Case | Technology |
|----------|------------|
| **Batch processing** | Spark, Dask |
| **Stream processing** | Flink, Kafka Streams |
| **Orchestration** | Airflow, Prefect, Dagster |
| **Feature store** | Feast, Tecton, Hopsworks |

### Model Training
| Use Case | Technology |
|----------|------------|
| **Experiment tracking** | MLflow, W&B, Neptune |
| **Hyperparameter tuning** | Optuna, Ray Tune |
| **Distributed training** | Horovod, PyTorch DDP |
| **AutoML** | AutoGluon, H2O |

### Model Serving
| Use Case | Technology |
|----------|------------|
| **REST API** | FastAPI, Flask |
| **Model server** | TensorFlow Serving, TorchServe, Triton |
| **Managed** | SageMaker, Vertex AI |
| **Edge** | ONNX Runtime, TensorRT |

### Monitoring
| Use Case | Technology |
|----------|------------|
| **Metrics** | Prometheus, CloudWatch |
| **Visualization** | Grafana, Datadog |
| **ML monitoring** | Evidently, Fiddler, Arize |
| **Logging** | ELK Stack, Splunk |

---

## 🔄 Common Patterns

### Serving Patterns

| Pattern | Latency | Use Case |
|---------|---------|----------|
| **Real-time** | <100ms | User-facing, fraud |
| **Batch** | Hours | Daily recommendations |
| **Near real-time** | Seconds | Stream processing |
| **Precomputed** | <10ms | Cached predictions |

### Feature Patterns

| Pattern | Description |
|---------|-------------|
| **Batch features** | Computed daily/hourly |
| **Real-time features** | Computed per request |
| **Streaming features** | Computed from event stream |
| **On-demand features** | Computed when needed |

### Training Patterns

| Pattern | Description |
|---------|-------------|
| **Offline training** | Train on historical data |
| **Online learning** | Update with each example |
| **Continual learning** | Periodic retraining |
| **Federated learning** | Train on distributed data |

---

## ⚡ Optimization Techniques

### Model Optimization
| Technique | Speedup | Accuracy Loss |
|-----------|---------|---------------|
| **Quantization (INT8)** | 2-4x | <1% |
| **Pruning** | 1.5-2x | <1% |
| **Distillation** | 2-10x | 1-5% |
| **ONNX/TensorRT** | 1.5-3x | None |

### System Optimization
| Technique | Benefit |
|-----------|---------|
| **Caching** | Reduce computation |
| **Batching** | Improve throughput |
| **Horizontal scaling** | Handle more traffic |
| **CDN** | Reduce latency |

---

## 🔐 Security Checklist

- [ ] API authentication (JWT/OAuth)
- [ ] Role-based access control
- [ ] Data encryption (rest + transit)
- [ ] Input validation
- [ ] Rate limiting
- [ ] Audit logging
- [ ] Secrets in vault

---

## 📈 Reliability Checklist

- [ ] Multiple replicas
- [ ] Health checks
- [ ] Auto-scaling
- [ ] Circuit breakers
- [ ] Fallback models
- [ ] Graceful degradation
- [ ] Disaster recovery plan

---

## 🎯 Interview Response Template

### 1. Clarify (2-3 min)
- "What's the business goal?"
- "How many users/requests?"
- "What's the latency requirement?"
- "What data is available?"

### 2. Metrics (2-3 min)
- Offline: accuracy, AUC, etc.
- Online: CTR, revenue, etc.
- System: latency, availability

### 3. High-Level Design (5-10 min)
- Draw architecture diagram
- Explain data flow
- Identify key components

### 4. Deep Dive (10-15 min)
- Feature engineering
- Model choice
- Serving strategy
- Monitoring approach

### 5. Trade-offs (5 min)
- Discuss alternatives
- Explain decisions
- Scale considerations

---

## 📋 Common Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Serving** | Real-time (fresh) | Batch (cheap) |
| **Model** | Complex (accurate) | Simple (fast) |
| **Features** | Many (accurate) | Few (fast) |
| **Training** | Frequent (fresh) | Infrequent (stable) |
| **Infrastructure** | Managed (easy) | Self-hosted (flexible) |

---

## 🔢 Numbers to Know

| Metric | Typical Value |
|--------|---------------|
| **HTTP request** | 50-100ms |
| **Database query** | 5-50ms |
| **Cache lookup** | 1-5ms |
| **Model inference (small)** | 5-20ms |
| **Model inference (large)** | 50-500ms |
| **Feature store lookup** | 5-50ms |
| **S3 read** | 50-200ms |

---

## 🚨 Red Flags in Interviews

### Don't:
- Skip requirements clarification
- Ignore scale/latency requirements
- Forget monitoring/evaluation
- Design without trade-offs
- Over-engineer for small scale
- Ignore data quality issues
- Forget about cold start

### Do:
- Ask clarifying questions
- Define success metrics first
- Consider failure modes
- Discuss trade-offs explicitly
- Start simple, then scale
- Consider the full pipeline
- Mention monitoring early

---

## 📚 Quick Links

| Topic | Location |
|-------|----------|
| Data Management | [02-data-management](./02-data-management/README.md) |
| Feature Engineering | [03-feature-engineering](./03-feature-engineering/README.md) |
| Model Training | [04-model-training](./04-model-training/README.md) |
| Model Serving | [05-model-serving](./05-model-serving/README.md) |
| Monitoring | [06-monitoring-observability](./06-monitoring-observability/README.md) |
| Scalability | [07-scalability-performance](./07-scalability-performance/README.md) |
| Reliability | [08-reliability-fault-tolerance](./08-reliability-fault-tolerance/README.md) |
| Security | [09-security-privacy](./09-security-privacy/README.md) |
| End-to-End Systems | [10-end-to-end-systems](./10-end-to-end-systems/README.md) |
| Interview Questions | [interview-questions](./interview-questions.md) |
