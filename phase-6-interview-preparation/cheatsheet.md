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

## 🧠 Embeddings & Retrieval Patterns

### Embedding Types
| Type | Use Case | Dimension |
|------|----------|-----------|
| **Word2Vec/GloVe** | Text similarity | 100-300 |
| **Item2Vec** | Item recommendations | 64-256 |
| **Two-tower** | User-item retrieval | 64-128 |
| **Sentence-BERT** | Semantic search, RAG | 384-768 |
| **CLIP** | Image-text matching | 512 |

### ANN Algorithms
| Algorithm | Build Time | Query Time | Memory | Best For |
|-----------|-----------|------------|--------|----------|
| **HNSW** | Slow | Fast (<1ms) | High | Low-latency serving |
| **IVF** | Medium | Medium | Medium | Large-scale, GPU |
| **PQ** | Medium | Medium | Low | Memory-constrained |
| **ScaNN** | Fast | Fast | Medium | Google-scale retrieval |

### Two-Tower Serving Pattern
```
Offline: Compute all item embeddings -> Build ANN index
Online:  User request -> User tower (real-time) -> ANN search -> Top-K candidates
```

### Vector Databases
| System | Managed | Hybrid Search | Best For |
|--------|---------|---------------|----------|
| **Pinecone** | Yes | Yes | Quick start, serverless |
| **Milvus** | Self/Cloud | Yes | Large-scale, on-prem |
| **Weaviate** | Self/Cloud | Yes | Multi-modal |
| **pgvector** | Self | Via SQL | Postgres integration |
| **Qdrant** | Self/Cloud | Yes | Filtering-heavy |

---

## 🤖 LLM System Patterns

### LLM Serving Key Numbers
| Metric | Typical Value |
|--------|---------------|
| **TTFT (time to first token)** | 100-500ms |
| **Token generation speed** | 30-100 tokens/sec |
| **KV cache per request** | 1-10 GB |
| **GPU memory (7B model, FP16)** | ~14 GB |
| **GPU memory (70B model, FP16)** | ~140 GB |

### RAG Architecture
```
Documents -> Chunk -> Embed -> Vector Store (offline)
Query -> Embed -> Retrieve top-K -> Rerank -> Inject into prompt -> LLM -> Response
```

### When to Use What
| Approach | Latency | Cost | Best For |
|----------|---------|------|----------|
| **Prompt engineering** | Low | Low | Simple tasks, few-shot |
| **RAG** | Medium | Medium | Domain knowledge, freshness |
| **Fine-tuning (LoRA)** | Low (serve) | Medium (train) | Style, format, domain |
| **Full fine-tuning** | Low (serve) | High (train) | Major behavior change |

### LLM Cost Optimization
| Technique | Savings | Trade-off |
|-----------|---------|-----------|
| **Quantization (INT4)** | 4x memory | 1-3% quality loss |
| **Model routing** | 50-80% | Complexity |
| **Prompt caching** | 30-70% | Staleness |
| **Smaller model first** | 60-90% | Latency for hard queries |

---

## ⚖️ Fairness Checklist

### Before Launch
- [ ] Define protected attributes for the use case
- [ ] Compute fairness metrics across subgroups (demographic parity, equalized odds)
- [ ] Check for proxy features that correlate with protected attributes
- [ ] Run sliced evaluation (model performance by subgroup)
- [ ] Create model card documenting intended use, limitations, fairness evaluation
- [ ] Review for feedback loop risks (will the model amplify existing bias?)

### In Production
- [ ] Monitor fairness metrics continuously (alert on drift)
- [ ] Periodic fairness audits (quarterly minimum)
- [ ] Human review process for edge cases
- [ ] User appeal mechanism for affected decisions
- [ ] Red team testing for adversarial bias exploitation

### Key Fairness Metrics
| Metric | Definition | Use When |
|--------|-----------|----------|
| **Demographic parity** | P(Y=1\|A=0) = P(Y=1\|A=1) | Equal rates across groups |
| **Equalized odds** | TPR and FPR equal across groups | Error rates matter |
| **Calibration** | P(Y=1\|score=s) same across groups | Predicted probabilities used directly |
| **Individual fairness** | Similar individuals get similar predictions | Fine-grained fairness |

---

## 📐 Scale Estimation Quick Reference

### Key Formulas
```
QPS = DAU x actions_per_user / 86,400
Peak QPS = Average QPS x 3
Storage = num_entities x record_size x retention_days
GPUs needed = Peak QPS / inferences_per_GPU_per_sec
Feature store size = entities x features x bytes_per_feature
Embedding table = entities x dimension x 4 bytes (FP32)
```

### Quick Numbers
| Scale | Users | QPS (avg) | QPS (peak) |
|-------|-------|-----------|------------|
| **Startup** | 1M DAU | ~1K | ~3K |
| **Mid-scale** | 50M DAU | ~50K | ~150K |
| **Meta/Google** | 500M-2B DAU | ~500K-5M | ~1.5M-15M |

### GPU Throughput (inferences/sec)
| Model Type | T4 | A100 | H100 |
|------------|-----|------|------|
| **Logistic regression** | 50K | 100K | 150K |
| **GBDT** | 10K | 20K | 30K |
| **BERT-base** | 300 | 1000 | 2000 |
| **LLM 7B (batch)** | N/A | 50 | 100 |

---

## 🏢 Company-Specific Quick Reference

### Meta Interview Focus
- Product thinking (how does this help the user?)
- Real-time systems at 3B+ user scale
- Integrity and content safety
- Multi-objective optimization
- **Top questions:** Feed ranking, ad click prediction, content moderation, PYMK, notification ranking

### Google Interview Focus
- Algorithmic depth and research awareness
- Infrastructure thinking (TFX, TPUs, distributed systems)
- Measurement and experimentation rigor
- **Top questions:** YouTube recommendations, search ranking, autocomplete, spam detection, Maps ETA

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
| Data Management | [02-data-management](../phase-2-core-components/02-data-management/00-README.md) |
| Feature Engineering | [03-feature-engineering](../phase-2-core-components/03-feature-engineering/00-README.md) |
| Model Training | [04-model-training](../phase-2-core-components/04-model-training/00-README.md) |
| Model Serving | [05-model-serving](../phase-2-core-components/05-model-serving/00-README.md) |
| Monitoring | [06-monitoring-observability](../phase-3-operations-and-reliability/06-monitoring-observability/00-README.md) |
| Scalability | [07-scalability-performance](../phase-3-operations-and-reliability/07-scalability-performance/00-README.md) |
| Reliability | [08-reliability-fault-tolerance](../phase-3-operations-and-reliability/08-reliability-fault-tolerance/00-README.md) |
| Security & Privacy | [09-security-privacy](../phase-3-operations-and-reliability/09-security-privacy/00-README.md) |
| End-to-End Systems | [10-end-to-end-systems](../phase-4-end-to-end-systems/10-end-to-end-systems/00-README.md) |
| Embeddings & Retrieval | [11-embeddings-retrieval](../phase-5-advanced-topics/11-embeddings-retrieval/00-README.md) |
| LLM & GenAI Systems | [12-llm-genai-systems](../phase-5-advanced-topics/12-llm-genai-systems/00-README.md) |
| Fairness & Responsible AI | [13-fairness-responsible-ai](../phase-5-advanced-topics/13-fairness-responsible-ai/00-README.md) |
| Online Experimentation | [14-online-experimentation](../phase-5-advanced-topics/14-online-experimentation/00-README.md) |
| Capacity & Cost Planning | [15-capacity-cost-planning](../phase-5-advanced-topics/15-capacity-cost-planning/00-README.md) |
| Interview Prep | [16-interview-prep](./16-interview-prep/00-README.md) |
