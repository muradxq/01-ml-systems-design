# Phase 2: Core Components

Deep dive into the four pillars of any ML system: data management, feature engineering, model training, and model serving. These are the building blocks that every ML system design interview will test.

---

## Chapters

2. [Data Management](./02-data-management/00-README.md)
   - [Data Collection](./02-data-management/01-data-collection.md) - Batch, streaming, and hybrid ingestion patterns
   - [Data Storage](./02-data-management/02-data-storage.md) - Data lakes, warehouses, and lakehouse architectures
   - [Data Versioning](./02-data-management/03-data-versioning.md) - Snapshot, delta, and lineage tracking
   - [Data Quality](./02-data-management/04-data-quality.md) - Completeness, accuracy, consistency, and validation

3. [Feature Engineering](./03-feature-engineering/00-README.md)
   - [Feature Stores](./03-feature-engineering/01-feature-stores.md) - Online/offline stores, Feast, Tecton
   - [Online vs Offline Features](./03-feature-engineering/02-online-vs-offline-features.md) - Synchronization and consistency
   - [Feature Pipelines](./03-feature-engineering/03-feature-pipelines.md) - Batch, streaming, and hybrid pipelines
   - [Feature Monitoring](./03-feature-engineering/04-feature-monitoring.md) - Drift detection and quality tracking

4. [Model Training](./04-model-training/00-README.md)
   - [Training Infrastructure](./04-model-training/01-training-infrastructure.md) - Compute, storage, orchestration
   - [Model Versioning](./04-model-training/02-model-versioning.md) - MLflow, model registry, stage management
   - [Experiment Tracking](./04-model-training/03-experiment-tracking.md) - MLflow, W&B, parameters and metrics
   - [Hyperparameter Tuning](./04-model-training/04-hyperparameter-tuning.md) - Grid, random, Bayesian, multi-objective
   - [Distributed Training](./04-model-training/05-distributed-training.md) - Data parallelism, DDP, DeepSpeed
   - [Training-Serving Skew](./04-model-training/06-training-serving-skew.md) - Sources, detection, and prevention

5. [Model Serving](./05-model-serving/00-README.md)
   - [Serving Patterns](./05-model-serving/01-serving-patterns.md) - Real-time, batch, async
   - [Model Deployment](./05-model-serving/02-model-deployment.md) - Docker, Kubernetes, blue-green, canary
   - [A/B Testing](./05-model-serving/03-ab-testing.md) - Traffic splitting and statistical tests
   - [Model Updates](./05-model-serving/04-model-updates.md) - Canary, shadow, feature flags, rollback
   - [Edge & Mobile Deployment](./05-model-serving/05-edge-deployment.md) - TFLite, Core ML, model compression, OTA updates

---

## What You'll Learn

- How to design data pipelines for ML at scale
- Feature engineering patterns and feature store architecture
- Training infrastructure from single-GPU to distributed clusters
- Production serving patterns with low-latency requirements
- How to avoid training-serving skew and deploy to edge devices

---

## Next Phase

Continue to [Phase 3: Operations & Reliability](../phase-3-operations-and-reliability/00-README.md) to learn how to monitor, scale, and make your ML systems reliable.
