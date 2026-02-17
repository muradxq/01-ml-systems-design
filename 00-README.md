# ML System Design: Comprehensive Interview Preparation

A comprehensive guide to designing, building, and deploying production-grade machine learning systems. Covers everything you need to pass Meta/Google final-round ML system design interviews.

---

## Table of Contents

### Phase 1: Introduction & Foundations
1. [Introduction to ML Systems](./phase-1-introduction-and-foundations/01-introduction/00-README.md)
   - [ML vs Traditional Software](./phase-1-introduction-and-foundations/01-introduction/01-ml-vs-traditional-software.md)
   - [ML System Lifecycle](./phase-1-introduction-and-foundations/01-introduction/02-ml-system-lifecycle.md)
   - [Key Components](./phase-1-introduction-and-foundations/01-introduction/03-key-components.md)
   - [Common Challenges](./phase-1-introduction-and-foundations/01-introduction/04-common-challenges.md)

### Phase 2: Core Components
2. [Data Management](./phase-2-core-components/02-data-management/00-README.md)
   - [Data Collection](./phase-2-core-components/02-data-management/01-data-collection.md)
   - [Data Storage](./phase-2-core-components/02-data-management/02-data-storage.md)
   - [Data Versioning](./phase-2-core-components/02-data-management/03-data-versioning.md)
   - [Data Quality](./phase-2-core-components/02-data-management/04-data-quality.md)

3. [Feature Engineering](./phase-2-core-components/03-feature-engineering/00-README.md)
   - [Feature Stores](./phase-2-core-components/03-feature-engineering/01-feature-stores.md)
   - [Online vs Offline Features](./phase-2-core-components/03-feature-engineering/02-online-vs-offline-features.md)
   - [Feature Pipelines](./phase-2-core-components/03-feature-engineering/03-feature-pipelines.md)
   - [Feature Monitoring](./phase-2-core-components/03-feature-engineering/04-feature-monitoring.md)

4. [Model Training](./phase-2-core-components/04-model-training/00-README.md)
   - [Training Infrastructure](./phase-2-core-components/04-model-training/01-training-infrastructure.md)
   - [Model Versioning](./phase-2-core-components/04-model-training/02-model-versioning.md)
   - [Experiment Tracking](./phase-2-core-components/04-model-training/03-experiment-tracking.md)
   - [Hyperparameter Tuning](./phase-2-core-components/04-model-training/04-hyperparameter-tuning.md)
   - [Distributed Training](./phase-2-core-components/04-model-training/05-distributed-training.md)
   - [Training-Serving Skew](./phase-2-core-components/04-model-training/06-training-serving-skew.md)

5. [Model Serving](./phase-2-core-components/05-model-serving/00-README.md)
   - [Serving Patterns](./phase-2-core-components/05-model-serving/01-serving-patterns.md)
   - [Model Deployment](./phase-2-core-components/05-model-serving/02-model-deployment.md)
   - [A/B Testing](./phase-2-core-components/05-model-serving/03-ab-testing.md)
   - [Model Updates](./phase-2-core-components/05-model-serving/04-model-updates.md)
   - [Edge & Mobile Deployment](./phase-2-core-components/05-model-serving/05-edge-deployment.md)

### Phase 3: Operations & Reliability
6. [Monitoring & Observability](./phase-3-operations-and-reliability/06-monitoring-observability/00-README.md)
   - [Model Monitoring](./phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md)
   - [Data Drift Detection](./phase-3-operations-and-reliability/06-monitoring-observability/02-data-drift-detection.md)
   - [Performance Metrics](./phase-3-operations-and-reliability/06-monitoring-observability/03-performance-metrics.md)
   - [Alerting Systems](./phase-3-operations-and-reliability/06-monitoring-observability/04-alerting-systems.md)
   - [SLOs & Distributed Tracing](./phase-3-operations-and-reliability/06-monitoring-observability/05-slos-distributed-tracing.md)

7. [Scalability & Performance](./phase-3-operations-and-reliability/07-scalability-performance/00-README.md)
   - [Horizontal Scaling](./phase-3-operations-and-reliability/07-scalability-performance/01-horizontal-scaling.md)
   - [Caching Strategies](./phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md)
   - [Batch vs Real-time](./phase-3-operations-and-reliability/07-scalability-performance/03-batch-vs-realtime.md)
   - [Optimization Techniques](./phase-3-operations-and-reliability/07-scalability-performance/04-optimization-techniques.md)
   - [Estimation Reference](./phase-3-operations-and-reliability/07-scalability-performance/05-estimation-reference.md)

8. [Reliability & Fault Tolerance](./phase-3-operations-and-reliability/08-reliability-fault-tolerance/00-README.md)
   - [High Availability](./phase-3-operations-and-reliability/08-reliability-fault-tolerance/01-high-availability.md)
   - [Graceful Degradation](./phase-3-operations-and-reliability/08-reliability-fault-tolerance/02-graceful-degradation.md)
   - [Circuit Breakers](./phase-3-operations-and-reliability/08-reliability-fault-tolerance/03-circuit-breakers.md)
   - [Disaster Recovery](./phase-3-operations-and-reliability/08-reliability-fault-tolerance/04-disaster-recovery.md)

9. [Security & Privacy](./phase-3-operations-and-reliability/09-security-privacy/00-README.md)
   - [Data Privacy](./phase-3-operations-and-reliability/09-security-privacy/01-data-privacy.md)
   - [Model Security](./phase-3-operations-and-reliability/09-security-privacy/02-model-security.md)
   - [Access Control](./phase-3-operations-and-reliability/09-security-privacy/03-access-control.md)
   - [Compliance](./phase-3-operations-and-reliability/09-security-privacy/04-compliance.md)

### Phase 4: End-to-End Systems
10. [End-to-End Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/00-README.md)
    - [Recommendation Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md)
    - [Search Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/02-search-systems.md)
    - [Fraud Detection](./phase-4-end-to-end-systems/10-end-to-end-systems/03-fraud-detection.md)
    - [Computer Vision Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/04-computer-vision-systems.md)
    - [NLP Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/05-nlp-systems.md)
    - [Time Series Forecasting](./phase-4-end-to-end-systems/10-end-to-end-systems/06-time-series-forecasting.md)
    - [Ad Click Prediction](./phase-4-end-to-end-systems/10-end-to-end-systems/07-ad-click-prediction.md)
    - [Feed Ranking](./phase-4-end-to-end-systems/10-end-to-end-systems/08-feed-ranking.md)
    - [Content Moderation](./phase-4-end-to-end-systems/10-end-to-end-systems/09-content-moderation.md)
    - [People You May Know](./phase-4-end-to-end-systems/10-end-to-end-systems/10-people-you-may-know.md)
    - [Autocomplete & Typeahead](./phase-4-end-to-end-systems/10-end-to-end-systems/11-autocomplete-typeahead.md)
    - [Notification Ranking](./phase-4-end-to-end-systems/10-end-to-end-systems/12-notification-ranking.md)
    - [Chatbot / LLM System](./phase-4-end-to-end-systems/10-end-to-end-systems/13-chatbot-llm-system.md)
    - [Video Recommendation](./phase-4-end-to-end-systems/10-end-to-end-systems/14-video-recommendation.md)
    - [Entity Resolution](./phase-4-end-to-end-systems/10-end-to-end-systems/15-entity-resolution.md)

### Phase 5: Advanced Topics
11. [Embeddings & Retrieval](./phase-5-advanced-topics/11-embeddings-retrieval/00-README.md)
    - [Embedding Fundamentals](./phase-5-advanced-topics/11-embeddings-retrieval/01-embedding-fundamentals.md)
    - [Approximate Nearest Neighbors](./phase-5-advanced-topics/11-embeddings-retrieval/02-approximate-nearest-neighbors.md)
    - [Vector Databases](./phase-5-advanced-topics/11-embeddings-retrieval/03-vector-databases.md)
    - [Two-Tower Architecture](./phase-5-advanced-topics/11-embeddings-retrieval/04-two-tower-architecture.md)

12. [LLM & GenAI Systems](./phase-5-advanced-topics/12-llm-genai-systems/00-README.md)
    - [LLM Serving Infrastructure](./phase-5-advanced-topics/12-llm-genai-systems/01-llm-serving-infrastructure.md)
    - [Retrieval-Augmented Generation](./phase-5-advanced-topics/12-llm-genai-systems/02-retrieval-augmented-generation.md)
    - [Fine-Tuning & Alignment](./phase-5-advanced-topics/12-llm-genai-systems/03-fine-tuning-alignment.md)
    - [Cost & Latency Optimization](./phase-5-advanced-topics/12-llm-genai-systems/04-cost-latency-optimization.md)

13. [Fairness & Responsible AI](./phase-5-advanced-topics/13-fairness-responsible-ai/00-README.md)
    - [Bias Detection](./phase-5-advanced-topics/13-fairness-responsible-ai/01-bias-detection.md)
    - [Fairness Constraints](./phase-5-advanced-topics/13-fairness-responsible-ai/02-fairness-constraints.md)
    - [Model Auditing](./phase-5-advanced-topics/13-fairness-responsible-ai/03-model-auditing.md)
    - [Responsible Deployment](./phase-5-advanced-topics/13-fairness-responsible-ai/04-responsible-deployment.md)

14. [Online Experimentation](./phase-5-advanced-topics/14-online-experimentation/00-README.md)
    - [Experiment Design](./phase-5-advanced-topics/14-online-experimentation/01-experiment-design.md)
    - [Advanced Testing](./phase-5-advanced-topics/14-online-experimentation/02-advanced-testing.md)
    - [Metric Design](./phase-5-advanced-topics/14-online-experimentation/03-metric-design.md)
    - [Analysis Pitfalls](./phase-5-advanced-topics/14-online-experimentation/04-analysis-pitfalls.md)

15. [Capacity & Cost Planning](./phase-5-advanced-topics/15-capacity-cost-planning/00-README.md)
    - [Back-of-Envelope Estimation](./phase-5-advanced-topics/15-capacity-cost-planning/01-back-of-envelope-estimation.md)
    - [Cost Modeling](./phase-5-advanced-topics/15-capacity-cost-planning/02-cost-modeling.md)
    - [Capacity Planning](./phase-5-advanced-topics/15-capacity-cost-planning/03-capacity-planning.md)

### Interview Preparation
16. [Interview Prep](./phase-6-interview-preparation/16-interview-prep/00-README.md)
    - [Interview Framework (CLEAR Method)](./phase-6-interview-preparation/16-interview-prep/01-interview-framework.md)
    - [Company-Specific Guide (Meta vs Google)](./phase-6-interview-preparation/16-interview-prep/02-company-specific-guide.md)
    - [Common Mistakes & Anti-Patterns](./phase-6-interview-preparation/16-interview-prep/03-common-mistakes.md)
    - [Scale Estimation Guide](./phase-6-interview-preparation/16-interview-prep/04-scale-estimation-guide.md)
    - [Question Bank (25 Questions)](./phase-6-interview-preparation/16-interview-prep/05-question-bank.md)
    - [Mock Interview Walkthroughs](./phase-6-interview-preparation/16-interview-prep/06-mock-interview-walkthroughs.md)
- [Cheatsheet](./phase-6-interview-preparation/cheatsheet.md) - Quick reference for ML system design
- [Interview Questions](./phase-6-interview-preparation/interview-questions.md) - Common ML system design interview questions

---

## Learning Paths

### Beginner Path (Weeks 1-2)
1. Start with [Introduction](./phase-1-introduction-and-foundations/01-introduction/00-README.md)
2. Understand [Data Management](./phase-2-core-components/02-data-management/00-README.md)
3. Learn [Feature Engineering](./phase-2-core-components/03-feature-engineering/00-README.md)
4. Study [Model Training](./phase-2-core-components/04-model-training/00-README.md)
5. Explore [Model Serving](./phase-2-core-components/05-model-serving/00-README.md)

### Intermediate Path (Weeks 3-4)
1. Deep dive into [Monitoring](./phase-3-operations-and-reliability/06-monitoring-observability/00-README.md)
2. Master [Scalability](./phase-3-operations-and-reliability/07-scalability-performance/00-README.md)
3. Understand [Reliability](./phase-3-operations-and-reliability/08-reliability-fault-tolerance/00-README.md)
4. Study [Security](./phase-3-operations-and-reliability/09-security-privacy/00-README.md)
5. Learn [Embeddings & Retrieval](./phase-5-advanced-topics/11-embeddings-retrieval/00-README.md)

### Advanced Path (Weeks 5-6)
1. Study [LLM & GenAI Systems](./phase-5-advanced-topics/12-llm-genai-systems/00-README.md)
2. Master [Fairness & Responsible AI](./phase-5-advanced-topics/13-fairness-responsible-ai/00-README.md)
3. Learn [Online Experimentation](./phase-5-advanced-topics/14-online-experimentation/00-README.md)
4. Understand [Capacity & Cost Planning](./phase-5-advanced-topics/15-capacity-cost-planning/00-README.md)
5. Design [End-to-End Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/00-README.md)

### Interview-Focused Path (Prioritized by frequency asked)
1. [Recommendation Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) - asked everywhere
2. [Ad Click Prediction](./phase-4-end-to-end-systems/10-end-to-end-systems/07-ad-click-prediction.md) - #1 Meta question
3. [Feed Ranking](./phase-4-end-to-end-systems/10-end-to-end-systems/08-feed-ranking.md) - core Meta product
4. [Search Systems](./phase-4-end-to-end-systems/10-end-to-end-systems/02-search-systems.md) - core Google product
5. [Video Recommendation](./phase-4-end-to-end-systems/10-end-to-end-systems/14-video-recommendation.md) - YouTube/Google
6. [Content Moderation](./phase-4-end-to-end-systems/10-end-to-end-systems/09-content-moderation.md) - Meta favorite
7. [Fraud Detection](./phase-4-end-to-end-systems/10-end-to-end-systems/03-fraud-detection.md) - asked at both
8. [People You May Know](./phase-4-end-to-end-systems/10-end-to-end-systems/10-people-you-may-know.md) - Meta graph-based
9. [Autocomplete](./phase-4-end-to-end-systems/10-end-to-end-systems/11-autocomplete-typeahead.md) - classic Google
10. [Chatbot / LLM System](./phase-4-end-to-end-systems/10-end-to-end-systems/13-chatbot-llm-system.md) - increasingly asked

---

## 2-Week Interview Study Plan

### Week 1: Build Foundations
| Day | Focus | Study Material |
|-----|-------|----------------|
| **Day 1** | Interview framework & estimation | [CLEAR Method](./phase-6-interview-preparation/16-interview-prep/01-interview-framework.md), [Scale Estimation](./phase-6-interview-preparation/16-interview-prep/04-scale-estimation-guide.md) |
| **Day 2** | Embeddings & retrieval | [Embedding Fundamentals](./phase-5-advanced-topics/11-embeddings-retrieval/01-embedding-fundamentals.md), [ANN](./phase-5-advanced-topics/11-embeddings-retrieval/02-approximate-nearest-neighbors.md), [Two-Tower](./phase-5-advanced-topics/11-embeddings-retrieval/04-two-tower-architecture.md) |
| **Day 3** | Feature stores & training | [Feature Stores](./phase-2-core-components/03-feature-engineering/01-feature-stores.md), [Training-Serving Skew](./phase-2-core-components/04-model-training/06-training-serving-skew.md) |
| **Day 4** | Serving & monitoring | [Serving Patterns](./phase-2-core-components/05-model-serving/01-serving-patterns.md), [SLOs](./phase-3-operations-and-reliability/06-monitoring-observability/05-slos-distributed-tracing.md) |
| **Day 5** | Experimentation & fairness | [Experiment Design](./phase-5-advanced-topics/14-online-experimentation/01-experiment-design.md), [Bias Detection](./phase-5-advanced-topics/13-fairness-responsible-ai/01-bias-detection.md) |
| **Day 6** | LLM systems | [LLM Serving](./phase-5-advanced-topics/12-llm-genai-systems/01-llm-serving-infrastructure.md), [RAG](./phase-5-advanced-topics/12-llm-genai-systems/02-retrieval-augmented-generation.md) |
| **Day 7** | Review & cheatsheet | [Cheatsheet](./phase-6-interview-preparation/cheatsheet.md), [Common Mistakes](./phase-6-interview-preparation/16-interview-prep/03-common-mistakes.md) |

### Week 2: Practice System Designs
| Day | Focus | Practice |
|-----|-------|----------|
| **Day 8** | Recommendation + Ad prediction | Design each in 45 min, compare with [Recs](./phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md), [Ads](./phase-4-end-to-end-systems/10-end-to-end-systems/07-ad-click-prediction.md) |
| **Day 9** | Feed ranking + Content moderation | Design each in 45 min, compare with [Feed](./phase-4-end-to-end-systems/10-end-to-end-systems/08-feed-ranking.md), [Moderation](./phase-4-end-to-end-systems/10-end-to-end-systems/09-content-moderation.md) |
| **Day 10** | Search + Autocomplete | Design each in 45 min, compare with [Search](./phase-4-end-to-end-systems/10-end-to-end-systems/02-search-systems.md), [Autocomplete](./phase-4-end-to-end-systems/10-end-to-end-systems/11-autocomplete-typeahead.md) |
| **Day 11** | Video rec + PYMK | Design each in 45 min, compare with [Video](./phase-4-end-to-end-systems/10-end-to-end-systems/14-video-recommendation.md), [PYMK](./phase-4-end-to-end-systems/10-end-to-end-systems/10-people-you-may-know.md) |
| **Day 12** | Fraud + Chatbot | Design each in 45 min, compare with [Fraud](./phase-4-end-to-end-systems/10-end-to-end-systems/03-fraud-detection.md), [Chatbot](./phase-4-end-to-end-systems/10-end-to-end-systems/13-chatbot-llm-system.md) |
| **Day 13** | Mock interviews | Study [Walkthroughs](./phase-6-interview-preparation/16-interview-prep/06-mock-interview-walkthroughs.md), practice with a friend |
| **Day 14** | Final review | [Company Guide](./phase-6-interview-preparation/16-interview-prep/02-company-specific-guide.md), [Question Bank](./phase-6-interview-preparation/16-interview-prep/05-question-bank.md), [Cheatsheet](./phase-6-interview-preparation/cheatsheet.md) |

---

## Key Concepts

### ML System Components
- **Data Pipeline**: Collect, store, and process data
- **Feature Store**: Centralized feature management (online + offline)
- **Training Pipeline**: Automated model training with experiment tracking
- **Model Registry**: Version control for models
- **Serving Infrastructure**: Real-time and batch inference
- **Monitoring**: Track model performance, data drift, and system health
- **Experimentation**: A/B testing and online evaluation

### Design Principles
1. **Modularity**: Separate concerns (data, training, serving)
2. **Reproducibility**: Version everything (data, code, models, features)
3. **Scalability**: Design for growth (horizontal scaling, caching, sharding)
4. **Reliability**: Handle failures gracefully (circuit breakers, fallbacks)
5. **Observability**: Monitor everything (SLIs, SLOs, distributed tracing)
6. **Security**: Protect data and models (encryption, access control)
7. **Fairness**: Detect and mitigate bias (sliced evaluation, fairness metrics)
8. **Cost-efficiency**: Optimize for cost (quantization, caching, capacity planning)

---

## How to Use This Guide

1. **Sequential study**: Follow the phases for comprehensive understanding
2. **Interview-focused**: Use the Interview-Focused Path for targeted prep
3. **2-week plan**: Follow the study plan for intensive interview preparation
4. **Reference**: Use the [Cheatsheet](./phase-6-interview-preparation/cheatsheet.md) and [Estimation Reference](./phase-3-operations-and-reliability/07-scalability-performance/05-estimation-reference.md) as quick references
5. **Practice**: Design systems from the [Question Bank](./phase-6-interview-preparation/16-interview-prep/05-question-bank.md) before reading the solutions

---

## Additional Resources

- [MLOps Best Practices](https://ml-ops.org/)
- [Feature Store Guide](https://www.featurestore.org/)
- [Google ML System Design](https://developers.google.com/machine-learning)
- [Meta Engineering Blog](https://engineering.fb.com/)
