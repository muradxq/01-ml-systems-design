# ML System Lifecycle

## Overview

The ML system lifecycle encompasses all stages from problem definition to production deployment and continuous improvement. Understanding this lifecycle is essential for building maintainable ML systems.

---

## 🔄 Lifecycle Stages

### 1. Problem Definition & Planning

**Activities:**
- Define business objectives
- Identify success metrics
- Assess data availability
- Estimate feasibility
- Plan resources and timeline

**Key Questions:**
- What problem are we solving?
- How will we measure success?
- Do we have the necessary data?
- Is ML the right solution?

**Outputs:**
- Problem statement
- Success criteria
- Data requirements
- Project plan

---

### 2. Data Collection & Preparation

**Activities:**
- Collect raw data from various sources
- Assess data quality
- Handle missing values and outliers
- Label data (for supervised learning)
- Create train/validation/test splits

**Key Considerations:**
- Data quality > Data quantity
- Representative sampling
- Data privacy and compliance
- Storage and access patterns

**Outputs:**
- Cleaned datasets
- Data documentation
- Data quality reports

---

### 3. Feature Engineering

**Activities:**
- Extract relevant features
- Transform features (normalization, encoding)
- Create derived features
- Select important features
- Build feature pipelines

**Key Considerations:**
- Feature consistency (train vs inference)
- Feature versioning
- Online vs offline features
- Feature store integration

**Outputs:**
- Feature definitions
- Feature pipelines
- Feature store entries

---

### 4. Model Development

**Activities:**
- Experiment with different algorithms
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Model selection

**Key Considerations:**
- Experiment tracking
- Reproducibility
- Overfitting prevention
- Model interpretability

**Outputs:**
- Trained models
- Experiment logs
- Evaluation metrics
- Model documentation

---

### 5. Model Validation & Testing

**Activities:**
- Validate on holdout test set
- Test on production-like data
- Evaluate business metrics
- Test edge cases
- Performance benchmarking

**Key Considerations:**
- Unbiased evaluation
- Real-world scenarios
- Latency requirements
- Resource constraints

**Outputs:**
- Validation reports
- Performance benchmarks
- Go/no-go decision

---

### 6. Model Deployment

**Activities:**
- Package model artifacts
- Set up serving infrastructure
- Deploy to staging environment
- Run smoke tests
- Gradual rollout (canary, A/B test)

**Key Considerations:**
- Model versioning
- Rollback strategy
- Traffic routing
- Monitoring setup

**Outputs:**
- Deployed model
- Monitoring dashboards
- Deployment documentation

---

### 7. Monitoring & Maintenance

**Activities:**
- Monitor model performance
- Track data drift
- Monitor system health
- Collect feedback
- Analyze errors

**Key Considerations:**
- Real-time monitoring
- Alert thresholds
- Performance baselines
- Feedback loops

**Outputs:**
- Monitoring dashboards
- Alert configurations
- Performance reports

---

### 8. Model Updates & Retraining

**Activities:**
- Detect performance degradation
- Collect new training data
- Retrain models
- Validate new models
- Deploy updates

**Key Considerations:**
- Retraining triggers
- Data freshness
- Update frequency
- Gradual rollout

**Outputs:**
- Updated models
- Retraining logs
- Performance improvements

---

## 🔁 Continuous Loop

```
┌─────────────────────────────────────────────────────────┐
│              Problem Definition & Planning               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Data Collection & Preparation                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Feature Engineering                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Model Development                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Model Validation & Testing                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Model Deployment                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Monitoring & Maintenance                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Model Updates & Retraining                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     └───────────────┐
                                     │
                                     ▼
                          (Loop back to Data Collection)
```

---

## 📊 Stage-Specific Metrics

### Development Stages
- **Data Quality**: Completeness, accuracy, consistency
- **Feature Quality**: Feature importance, correlation
- **Model Performance**: Accuracy, precision, recall, F1
- **Experiment Tracking**: Hyperparameters, metrics, artifacts

### Production Stages
- **Model Performance**: Prediction accuracy, latency, throughput
- **Data Drift**: Distribution shifts, feature changes
- **System Health**: Uptime, error rates, resource usage
- **Business Impact**: Revenue, user engagement, conversions

---

## 🎯 Best Practices

### 1. Start Simple
- Begin with baseline models
- Iterate and improve
- Avoid premature optimization

### 2. Version Everything
- Data versions
- Feature versions
- Model versions
- Code versions

### 3. Automate Where Possible
- Automated retraining
- Automated testing
- Automated deployment
- Automated monitoring

### 4. Monitor Continuously
- Set up monitoring from day one
- Track both technical and business metrics
- Create alerting for critical issues

### 5. Plan for Failure
- Design rollback strategies
- Implement graceful degradation
- Test failure scenarios

---

## 🛠️ End-to-End Pipeline Implementation

### Complete Pipeline Example with Python

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

@dataclass
class ExperimentConfig:
    experiment_name: str
    model_name: str
    version: str
    hyperparameters: Dict[str, Any]

class MLPipeline:
    """End-to-end ML pipeline orchestrator."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_id = None
        
    def stage_1_data_collection(self, data_source: str) -> pd.DataFrame:
        """Stage 1: Collect and validate data."""
        print("Stage 1: Data Collection")
        
        # Load data
        df = pd.read_parquet(data_source)
        
        # Validate data quality
        assert df.notna().mean().min() > 0.95, "Data completeness < 95%"
        
        # Log data stats
        mlflow.log_metric("data_rows", len(df))
        mlflow.log_metric("data_columns", len(df.columns))
        
        return df
    
    def stage_2_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Feature engineering and transformation."""
        print("Stage 2: Feature Engineering")
        
        # Feature transformations
        features = self._compute_features(df)
        
        # Validate features
        self._validate_features(features)
        
        # Log feature stats
        mlflow.log_metric("num_features", len(features.columns) - 1)
        
        return features
    
    def stage_3_model_training(self, features: pd.DataFrame, target_col: str):
        """Stage 3: Model training and hyperparameter tuning."""
        print("Stage 3: Model Training")
        
        X = features.drop(columns=[target_col])
        y = features[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = self._train_model(X_train, y_train)
        
        # Log hyperparameters
        mlflow.log_params(self.config.hyperparameters)
        
        return model, (X_test, y_test)
    
    def stage_4_validation(self, model, test_data):
        """Stage 4: Model validation and testing."""
        print("Stage 4: Validation")
        
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Check if model passes quality gate
        assert metrics['accuracy'] > 0.7, "Model accuracy below threshold"
        
        return metrics
    
    def stage_5_registration(self, model, metrics: Dict[str, float]):
        """Stage 5: Register model for deployment."""
        print("Stage 5: Model Registration")
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=self.config.model_name
        )
        
        # Add metadata
        mlflow.set_tag("model_version", self.config.version)
        mlflow.set_tag("ready_for_deployment", metrics['accuracy'] > 0.8)
        
        return mlflow.active_run().info.run_id
    
    def run_pipeline(self, data_source: str, target_col: str):
        """Execute the complete pipeline."""
        with mlflow.start_run(run_name=self.config.experiment_name):
            # Execute stages
            df = self.stage_1_data_collection(data_source)
            features = self.stage_2_feature_engineering(df)
            model, test_data = self.stage_3_model_training(features, target_col)
            metrics = self.stage_4_validation(model, test_data)
            run_id = self.stage_5_registration(model, metrics)
            
            print(f"Pipeline completed. Run ID: {run_id}")
            return run_id, metrics
```

---

## 📊 Lifecycle Automation with CI/CD

### GitHub Actions Workflow Example

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI/CD

on:
  push:
    paths:
      - 'data/**'
      - 'models/**'
      - 'src/**'
  schedule:
    - cron: '0 0 * * *'  # Daily retraining

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Data Quality
        run: |
          python -m pytest tests/test_data_quality.py
      
  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train Model
        run: |
          python src/train.py --config config/production.yaml
      - name: Upload Model Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: outputs/

  model-evaluation:
    needs: model-training
    runs-on: ubuntu-latest
    steps:
      - name: Evaluate Model
        run: |
          python src/evaluate.py --model outputs/model.pkl
      - name: Check Quality Gates
        run: |
          python src/quality_gates.py

  deployment:
    needs: model-evaluation
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: |
          python src/deploy.py --env staging
      - name: Run Smoke Tests
        run: |
          python tests/smoke_tests.py
      - name: Promote to Production
        run: |
          python src/deploy.py --env production --canary 10
```

---

## 🎯 MLOps Maturity Levels

### Level 0: Manual Process
- Manual data prep, training, deployment
- No automation or versioning
- Ad-hoc monitoring

### Level 1: ML Pipeline Automation
- Automated training pipelines
- Experiment tracking
- Model versioning

### Level 2: CI/CD Pipeline Automation
- Automated testing
- Continuous integration
- Automated deployment

### Level 3: Full MLOps
- Automated retraining on drift
- Feature stores
- A/B testing infrastructure
- Comprehensive monitoring

```
Level 0 ──────> Level 1 ──────> Level 2 ──────> Level 3
 Manual      Automated     CI/CD for ML     Full MLOps
 Process     Training      Pipelines        Automation
```

---

## 🎯 Interview Questions

**Q1: Walk me through the ML lifecycle stages. What are the key activities and outputs of each?**

**Answer Framework:**
1. Problem Definition → Success metrics, feasibility assessment
2. Data Collection → Cleaned datasets, quality reports
3. Feature Engineering → Feature pipelines, feature store entries
4. Model Training → Trained models, experiment logs
5. Validation → Performance metrics, go/no-go decision
6. Deployment → Deployed model, monitoring setup
7. Monitoring → Dashboards, alerts
8. Retraining → Updated models, performance improvements

**Q2: How do you decide when to retrain a model?**

**Answer Framework:**
- **Scheduled**: Regular intervals (daily, weekly)
- **Triggered**: Performance degradation detected
- **Data-driven**: New data available, distribution shift
- **Business**: New requirements, feedback

**Q3: What's the difference between MLOps maturity levels?**

Explain the progression from manual processes to full automation, emphasizing the value at each level.

---

## 🔑 Key Takeaways

1. **ML lifecycle is iterative** - models need continuous updates
2. **Each stage has specific outputs** - track artifacts and metrics
3. **Monitoring is critical** - detect issues early
4. **Automation accelerates** - automate repetitive tasks
5. **Plan for production** - consider production constraints early
6. **Quality gates** - establish checkpoints at each stage

---

## 📚 Further Reading

- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [The ML Test Score: A Rubric for ML Production Readiness](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7adfe23ed72af17065adf1df2da94.pdf)
- [Practical MLOps: Operationalizing Machine Learning Models](https://www.oreilly.com/library/view/practical-mlops/9781098103002/)

---

## 🔗 Related Topics

- [ML vs Traditional Software](./01-ml-vs-traditional-software.md)
- [Key Components](./03-key-components.md)
- [Common Challenges](./04-common-challenges.md)
