# 🏗️ End-to-End ML Systems

## Overview

This section presents complete ML system designs for common industry use cases. Each system demonstrates how the concepts from previous sections (data management, feature engineering, model training, serving, monitoring, scalability, reliability, security) come together in production environments. These examples serve as templates for designing your own ML systems and preparing for system design interviews.

---

## 📚 Systems Covered

| System | Primary ML Task | Key Challenges | Industry Examples |
|--------|-----------------|----------------|-------------------|
| [Recommendation Systems](./recommendation-systems.md) | Ranking, Collaborative Filtering | Cold start, scalability | Netflix, Amazon, Spotify |
| [Search Systems](./search-systems.md) | Information Retrieval, Ranking | Latency, relevance | Google, Elasticsearch, E-commerce |
| [Fraud Detection](./fraud-detection.md) | Binary Classification | Real-time, imbalanced data | Banks, Payment processors |
| [Computer Vision](./computer-vision-systems.md) | Image Classification, Detection | GPU scaling, latency | Autonomous vehicles, Medical imaging |
| [NLP Systems](./nlp-systems.md) | Text Classification, Generation | Model size, context | Chatbots, Translation, Search |
| [Time Series Forecasting](./time-series-forecasting.md) | Regression, Sequence Modeling | Seasonality, drift | Finance, Supply chain, Weather |

---

## 🎯 Key Design Principles

### 1. Start with Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│  Requirements Framework                                          │
│                                                                  │
│  Functional Requirements:                                       │
│  - What problem are we solving?                                 │
│  - Who are the users?                                           │
│  - What actions/predictions are needed?                         │
│                                                                  │
│  Non-Functional Requirements:                                   │
│  - Latency: p50, p95, p99 targets                              │
│  - Throughput: Requests per second                             │
│  - Availability: Uptime SLA (99.9%, 99.99%)                    │
│  - Scale: Users, data volume, growth rate                      │
│                                                                  │
│  Constraints:                                                   │
│  - Budget                                                       │
│  - Timeline                                                     │
│  - Team size/expertise                                          │
│  - Existing infrastructure                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Design for Scale

```
┌─────────────────────────────────────────────────────────────────┐
│  Scale Considerations                                            │
│                                                                  │
│  Data Scale:                                                    │
│  - Training data: GBs → TBs → PBs                              │
│  - Feature store: Millions → Billions of entities              │
│  - Inference data: Real-time streams                           │
│                                                                  │
│  Traffic Scale:                                                 │
│  - QPS: 100 → 10K → 1M+                                        │
│  - Concurrent users: 1K → 100K → 10M+                          │
│  - Global distribution                                          │
│                                                                  │
│  Model Scale:                                                   │
│  - Parameters: Millions → Billions                             │
│  - Inference time: ms → seconds                                │
│  - GPU requirements                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Design for Failure

- **Redundancy:** No single points of failure
- **Graceful Degradation:** Fallback when components fail
- **Circuit Breakers:** Prevent cascade failures
- **Monitoring:** Detect issues quickly
- **Rollback:** Quick recovery from bad deployments

### 4. Iterative Development

```
┌─────────────────────────────────────────────────────────────────┐
│  Iteration Strategy                                              │
│                                                                  │
│  Phase 1: MVP                                                   │
│  - Simple model (logistic regression, basic rules)             │
│  - Essential features only                                      │
│  - Basic monitoring                                             │
│                                                                  │
│  Phase 2: Improve                                               │
│  - Better model (gradient boosting, neural networks)           │
│  - More features                                                │
│  - A/B testing infrastructure                                   │
│                                                                  │
│  Phase 3: Scale                                                 │
│  - Advanced models (deep learning, embeddings)                 │
│  - Full feature store                                           │
│  - Real-time features                                           │
│  - Multi-region deployment                                      │
│                                                                  │
│  Phase 4: Optimize                                              │
│  - Model optimization (quantization, distillation)             │
│  - Cost optimization                                            │
│  - Advanced personalization                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Common Architecture Patterns

### Pattern 1: Two-Stage Retrieval + Ranking

Used by: Recommendations, Search, Ads

```
┌─────────────────────────────────────────────────────────────────┐
│  Two-Stage Architecture                                          │
│                                                                  │
│  Request ───▶ Candidate Generation ───▶ Ranking ───▶ Response  │
│                    │                        │                    │
│                    ▼                        ▼                    │
│              Fast & Broad            Slow & Precise             │
│              (ANN, filters)          (ML ranking model)         │
│              1M → 1000              1000 → 10                   │
│              < 10ms                  < 50ms                     │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern 2: Feature Store + Real-time Inference

Used by: Fraud Detection, Personalization

```
┌─────────────────────────────────────────────────────────────────┐
│  Real-time Feature + Inference                                   │
│                                                                  │
│  Event ───▶ Feature Enrichment ───▶ Model ───▶ Action          │
│                   │                    │                         │
│                   ▼                    ▼                         │
│           ┌───────────────┐    ┌───────────────┐               │
│           │ Feature Store │    │ Model Server  │               │
│           │ (Online)      │    │ (GPU/CPU)     │               │
│           └───────────────┘    └───────────────┘               │
│                   │                                              │
│                   ▼                                              │
│           ┌───────────────┐                                     │
│           │ Feature Store │                                     │
│           │ (Offline)     │                                     │
│           └───────────────┘                                     │
│                   │                                              │
│                   ▼                                              │
│           Training Pipeline                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern 3: Batch Processing + Precomputation

Used by: Content Recommendations, Reports

```
┌─────────────────────────────────────────────────────────────────┐
│  Batch Precomputation                                            │
│                                                                  │
│  Nightly:                                                       │
│  Data Lake ───▶ Training ───▶ Batch Scoring ───▶ Cache/DB      │
│                                                                  │
│  Real-time:                                                     │
│  Request ───▶ Lookup Precomputed ───▶ Response                 │
│                      │                                           │
│                      └───▶ < 10ms latency                       │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern 4: Streaming + Continuous Learning

Used by: Fraud Detection, Anomaly Detection

```
┌─────────────────────────────────────────────────────────────────┐
│  Streaming Pipeline                                              │
│                                                                  │
│  Events ───▶ Stream Processing ───▶ Feature Updates             │
│     │              │                      │                      │
│     │              ▼                      ▼                      │
│     │       Real-time Inference    Feature Store                │
│     │              │                      │                      │
│     │              ▼                      ▼                      │
│     └────▶ Feedback Loop ──────▶ Model Retraining               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Technology Stack Reference

### Data Layer

| Component | Options | Use Case |
|-----------|---------|----------|
| **Data Lake** | S3, GCS, Azure Blob | Raw data storage |
| **Data Warehouse** | Snowflake, BigQuery, Redshift | Analytics, training data |
| **Feature Store** | Feast, Tecton, Redis | Feature serving |
| **Vector DB** | Pinecone, Milvus, Weaviate | Embedding search |

### Compute Layer

| Component | Options | Use Case |
|-----------|---------|----------|
| **Training** | SageMaker, Vertex AI, Kubeflow | Model training |
| **Batch** | Spark, Dataflow, EMR | Batch processing |
| **Streaming** | Kafka, Flink, Kinesis | Real-time processing |
| **Serving** | Kubernetes, Lambda, Cloud Run | Model serving |

### ML Layer

| Component | Options | Use Case |
|-----------|---------|----------|
| **Framework** | PyTorch, TensorFlow, scikit-learn | Model development |
| **Serving** | TorchServe, TF Serving, Triton | Model deployment |
| **Registry** | MLflow, W&B, SageMaker | Model versioning |
| **Orchestration** | Airflow, Prefect, Kubeflow | Pipeline management |

### Monitoring Layer

| Component | Options | Use Case |
|-----------|---------|----------|
| **Metrics** | Prometheus, CloudWatch, Datadog | System metrics |
| **Logging** | ELK, Splunk, CloudWatch Logs | Centralized logging |
| **ML Monitoring** | Evidently, Fiddler, Arize | Model monitoring |
| **Alerting** | PagerDuty, OpsGenie, Slack | Incident response |

---

## 🎓 System Design Interview Framework

### Step 1: Clarify Requirements (5 minutes)

```markdown
Questions to ask:
- What is the primary use case?
- Who are the users?
- What scale are we designing for?
- What are the latency requirements?
- What's the accuracy/business metric target?
- Any constraints (budget, timeline, team)?
```

### Step 2: High-Level Design (10 minutes)

```markdown
Components to cover:
- Data sources and ingestion
- Feature engineering
- Model training pipeline
- Model serving
- Monitoring
- User interface/API
```

### Step 3: Deep Dive (15 minutes)

```markdown
Pick 2-3 areas to elaborate:
- Data pipeline design
- Feature store architecture
- Model selection and training
- Serving infrastructure
- Scaling strategy
- Monitoring and alerting
```

### Step 4: Trade-offs and Extensions (5 minutes)

```markdown
Discuss:
- Why this design over alternatives?
- What would you change with more time?
- How would you handle 10x scale?
- What are the failure modes?
```

---

## 📚 System Index

Each system includes:
- **Architecture Overview:** High-level system diagram
- **Component Deep Dive:** Detailed design of each component
- **Data Flow:** How data moves through the system
- **Scale Considerations:** Handling growth
- **Code Examples:** Implementation snippets
- **Trade-offs:** Design decisions and alternatives
- **Interview Tips:** Common questions and answers

Continue to:
1. [Recommendation Systems](./recommendation-systems.md)
2. [Search Systems](./search-systems.md)
3. [Fraud Detection](./fraud-detection.md)
4. [Computer Vision Systems](./computer-vision-systems.md)
5. [NLP Systems](./nlp-systems.md)
6. [Time Series Forecasting](./time-series-forecasting.md)
