# Model Versioning

## Overview

Model versioning tracks model artifacts, metadata, and lineage throughout the ML lifecycle. Unlike code versioning, model versioning must handle large binary files, track training data and hyperparameters, and support rollback and comparison. Effective model versioning enables reproducibility, safe deployments, and regulatory compliance.

---

## 🎯 Why Version Models?

### 1. Reproducibility
- Reproduce model training results exactly
- Debug model performance changes
- Compare experiments fairly

### 2. Traceability
- Track which data was used for training
- Understand model lineage (data → features → model)
- Audit model changes for compliance

### 3. Deployment Safety
- Rollback to previous versions quickly
- A/B test different versions
- Blue-green deployments

### 4. Collaboration
- Share models with team members
- Track who trained what and when
- Avoid conflicts with model changes

### 5. Compliance
- Maintain audit trail for regulations
- Document model changes over time
- Support model governance

---

## 🏗️ Model Registry Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Registry                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Metadata Store (Database)                               │   │
│  │  - Model name, version, stage                            │   │
│  │  - Training parameters, metrics                          │   │
│  │  - Data lineage, feature versions                        │   │
│  │  - Tags, annotations                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Artifact Store (Object Storage)                         │   │
│  │  - Model weights (pickle, ONNX, SavedModel)              │   │
│  │  - Preprocessing artifacts                               │   │
│  │  - Configuration files                                   │   │
│  │  - Evaluation results                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  APIs & SDKs                                             │   │
│  │  - Register models                                       │   │
│  │  - Retrieve models by version/stage                      │   │
│  │  - Transition model stages                               │   │
│  │  - Search and compare                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Model Registry Tools

### 1. MLflow Model Registry

**Features:**
- Model versioning and staging
- Model lineage tracking
- API and UI for management
- Integration with MLflow Tracking

**Usage:**

```python
import mlflow
from mlflow.tracking import MlflowClient

# Initialize client
client = MlflowClient()

# Train and log model
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params({
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    })
    
    # Train model
    model = train_model(...)
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': 0.95,
        'f1_score': 0.94,
        'auc': 0.98
    })
    
    # Log model with signature
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model, 
        "model",
        signature=signature,
        input_example=X_train[:5]
    )
    
    # Register model
    model_uri = f"runs:/{run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, "fraud-detection-model")

# Model lifecycle management
def promote_model(model_name: str, version: int, stage: str):
    """
    Transition model to new stage.
    Stages: None -> Staging -> Production -> Archived
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production")
    )

# Load model by stage
def load_production_model(model_name: str):
    """Load the production model."""
    model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)

# Compare model versions
def compare_versions(model_name: str, v1: int, v2: int) -> dict:
    """Compare metrics between model versions."""
    mv1 = client.get_model_version(model_name, v1)
    mv2 = client.get_model_version(model_name, v2)
    
    run1 = client.get_run(mv1.run_id)
    run2 = client.get_run(mv2.run_id)
    
    return {
        'v1_metrics': run1.data.metrics,
        'v2_metrics': run2.data.metrics,
        'metric_diff': {
            k: run2.data.metrics.get(k, 0) - run1.data.metrics.get(k, 0)
            for k in run1.data.metrics
        }
    }
```

---

### 2. Weights & Biases (W&B)

**Features:**
- Experiment tracking + model registry
- Rich visualization
- Team collaboration
- Model lineage graphs

**Usage:**

```python
import wandb

# Initialize run
wandb.init(project="fraud-detection", name="experiment-1")

# Log config
wandb.config.update({
    'learning_rate': 0.001,
    'epochs': 100,
    'architecture': 'transformer'
})

# Train and log
for epoch in range(100):
    train_loss, val_loss = train_epoch(model)
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch
    })

# Save model artifact
artifact = wandb.Artifact(
    name='fraud-model',
    type='model',
    description='Fraud detection model v1',
    metadata={
        'accuracy': 0.95,
        'framework': 'pytorch'
    }
)
artifact.add_file('model.pt')
wandb.log_artifact(artifact)

# Link to model registry
wandb.link_artifact(artifact, 'fraud-detection/production')

# Load model from registry
def load_model_from_wandb(artifact_name: str):
    """Load model from W&B."""
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()
    return torch.load(f"{artifact_dir}/model.pt")
```

---

### 3. Custom Model Registry

```python
import boto3
import json
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib

class ModelRegistry:
    """
    Custom model registry with S3 storage and DynamoDB metadata.
    """
    
    def __init__(self, bucket: str, table_name: str):
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.bucket = bucket
        self.table = self.dynamodb.Table(table_name)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        tags: Dict[str, str] = None,
        description: str = None
    ) -> str:
        """Register a new model version."""
        
        # Generate version
        version = self._get_next_version(model_name)
        
        # Upload model artifact
        s3_key = f"models/{model_name}/{version}/model.pkl"
        self.s3.upload_file(model_path, self.bucket, s3_key)
        
        # Compute model hash for integrity
        model_hash = self._compute_hash(model_path)
        
        # Store metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.utcnow().isoformat(),
            'stage': 'None',
            's3_uri': f"s3://{self.bucket}/{s3_key}",
            'metrics': metrics,
            'params': params,
            'tags': tags or {},
            'description': description,
            'model_hash': model_hash
        }
        
        self.table.put_item(Item=metadata)
        
        return version
    
    def get_model(
        self, 
        model_name: str, 
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Dict:
        """Get model metadata."""
        
        if stage:
            # Get by stage
            response = self.table.query(
                IndexName='stage-index',
                KeyConditionExpression='model_name = :name AND stage = :stage',
                ExpressionAttributeValues={
                    ':name': model_name,
                    ':stage': stage
                }
            )
            items = response.get('Items', [])
            if not items:
                raise ValueError(f"No model in stage {stage}")
            return items[0]
        
        elif version:
            # Get by version
            response = self.table.get_item(
                Key={'model_name': model_name, 'version': version}
            )
            return response.get('Item')
        
        else:
            # Get latest
            return self.get_latest_version(model_name)
    
    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_previous: bool = True
    ):
        """Transition model to new stage."""
        
        valid_stages = ['None', 'Staging', 'Production', 'Archived']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}")
        
        # Archive previous production model
        if archive_previous and stage == 'Production':
            self._archive_current_production(model_name)
        
        # Update stage
        self.table.update_item(
            Key={'model_name': model_name, 'version': version},
            UpdateExpression='SET stage = :stage, updated_at = :time',
            ExpressionAttributeValues={
                ':stage': stage,
                ':time': datetime.utcnow().isoformat()
            }
        )
    
    def download_model(self, model_name: str, version: str, local_path: str):
        """Download model from registry."""
        metadata = self.get_model(model_name, version)
        s3_uri = metadata['s3_uri']
        
        # Parse S3 URI
        bucket, key = self._parse_s3_uri(s3_uri)
        
        # Download
        self.s3.download_file(bucket, key, local_path)
        
        # Verify integrity
        local_hash = self._compute_hash(local_path)
        if local_hash != metadata['model_hash']:
            raise ValueError("Model integrity check failed")
        
        return local_path
    
    def list_versions(self, model_name: str) -> list:
        """List all versions of a model."""
        response = self.table.query(
            KeyConditionExpression='model_name = :name',
            ExpressionAttributeValues={':name': model_name}
        )
        return sorted(response['Items'], key=lambda x: x['version'])
    
    def compare_models(self, model_name: str, v1: str, v2: str) -> Dict:
        """Compare two model versions."""
        m1 = self.get_model(model_name, v1)
        m2 = self.get_model(model_name, v2)
        
        return {
            'version_1': v1,
            'version_2': v2,
            'metrics_v1': m1['metrics'],
            'metrics_v2': m2['metrics'],
            'metric_changes': {
                k: m2['metrics'].get(k, 0) - m1['metrics'].get(k, 0)
                for k in m1['metrics']
            },
            'params_diff': self._diff_params(m1['params'], m2['params'])
        }
    
    def _get_next_version(self, model_name: str) -> str:
        """Get next version number."""
        versions = self.list_versions(model_name)
        if not versions:
            return '1'
        return str(int(versions[-1]['version']) + 1)
    
    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _archive_current_production(self, model_name: str):
        """Archive current production model."""
        try:
            current = self.get_model(model_name, stage='Production')
            self.transition_stage(
                model_name, 
                current['version'], 
                'Archived',
                archive_previous=False
            )
        except ValueError:
            pass  # No current production model

# Usage
registry = ModelRegistry(bucket='models-bucket', table_name='model-registry')

# Register model
version = registry.register_model(
    model_name='fraud-detection',
    model_path='./model.pkl',
    metrics={'accuracy': 0.95, 'auc': 0.98},
    params={'learning_rate': 0.001, 'epochs': 100},
    tags={'team': 'fraud', 'environment': 'production'}
)

# Promote to production
registry.transition_stage('fraud-detection', version, 'Production')

# Load production model
registry.download_model('fraud-detection', version, './production_model.pkl')
```

---

## 📋 Model Metadata Schema

### Essential Metadata

```python
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    
    # Identity
    model_name: str
    version: str
    stage: str  # None, Staging, Production, Archived
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime]
    
    # Lineage
    training_run_id: str
    data_version: str
    feature_version: str
    code_version: str  # Git commit
    
    # Training info
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    
    # Model info
    framework: str  # sklearn, pytorch, tensorflow
    model_type: str  # classification, regression, etc.
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    
    # Storage
    artifact_uri: str
    model_size_bytes: int
    model_hash: str  # SHA256 for integrity
    
    # Annotations
    description: Optional[str]
    tags: Dict[str, str] = {}
    
    # Ownership
    created_by: str
    team: str
```

---

## 🔄 Model Lifecycle Management

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Lifecycle States                        │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │             │     │             │     │             │      │
│  │   NONE      │────▶│  STAGING    │────▶│ PRODUCTION  │      │
│  │             │     │             │     │             │      │
│  └─────────────┘     └──────┬──────┘     └──────┬──────┘      │
│                             │                   │               │
│                             │                   │               │
│                             ▼                   ▼               │
│                      ┌─────────────┐     ┌─────────────┐      │
│                      │             │     │             │      │
│                      │  REJECTED   │     │  ARCHIVED   │      │
│                      │             │     │             │      │
│                      └─────────────┘     └─────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Stage Transitions

```python
class ModelLifecycleManager:
    """Manage model stage transitions with validation."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.valid_transitions = {
            'None': ['Staging', 'Rejected'],
            'Staging': ['Production', 'Rejected', 'None'],
            'Production': ['Archived'],
            'Rejected': ['None'],
            'Archived': []
        }
    
    def request_promotion(
        self,
        model_name: str,
        version: str,
        target_stage: str,
        requester: str
    ) -> Dict:
        """Request model promotion with validation."""
        
        model = self.registry.get_model(model_name, version)
        current_stage = model['stage']
        
        # Validate transition
        if target_stage not in self.valid_transitions.get(current_stage, []):
            raise ValueError(
                f"Invalid transition: {current_stage} -> {target_stage}"
            )
        
        # Run stage-specific validation
        if target_stage == 'Staging':
            self._validate_for_staging(model)
        elif target_stage == 'Production':
            self._validate_for_production(model)
        
        # Execute transition
        self.registry.transition_stage(model_name, version, target_stage)
        
        return {
            'model_name': model_name,
            'version': version,
            'new_stage': target_stage,
            'transitioned_by': requester,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _validate_for_staging(self, model: Dict):
        """Validate model meets staging requirements."""
        required_metrics = ['accuracy', 'auc', 'f1']
        
        for metric in required_metrics:
            if metric not in model['metrics']:
                raise ValueError(f"Missing required metric: {metric}")
        
        # Check minimum performance
        if model['metrics'].get('accuracy', 0) < 0.8:
            raise ValueError("Accuracy below minimum threshold (0.8)")
    
    def _validate_for_production(self, model: Dict):
        """Validate model meets production requirements."""
        
        # Must be in staging first
        if model['stage'] != 'Staging':
            raise ValueError("Model must be in Staging before Production")
        
        # Performance requirements
        if model['metrics'].get('auc', 0) < 0.9:
            raise ValueError("AUC below production threshold (0.9)")
        
        # Check model has been validated
        if 'validation_report' not in model.get('tags', {}):
            raise ValueError("Model requires validation report")
```

---

## ✅ Best Practices

### 1. Version Everything
- Model weights and architecture
- Training code (git hash)
- Training data version
- Feature definitions
- Configuration

### 2. Use Semantic Versioning
```
MAJOR.MINOR.PATCH

Examples:
- 1.0.0: Initial production model
- 1.1.0: Added new features
- 1.1.1: Bug fix in preprocessing
- 2.0.0: Breaking architecture change
```

### 3. Store Rich Metadata
- Training parameters
- Evaluation metrics
- Data lineage
- Environment info

### 4. Implement Stage Gates
- Validation requirements for each stage
- Automated testing before promotion
- Approval workflows

### 5. Document Changes
- Changelog for each version
- Migration notes
- Known issues

---

## 📊 Model Comparison Dashboard

```python
def generate_comparison_report(
    model_name: str,
    versions: List[str],
    registry: ModelRegistry
) -> str:
    """Generate HTML comparison report."""
    
    models = [registry.get_model(model_name, v) for v in versions]
    
    # Metric comparison table
    metrics_table = []
    all_metrics = set()
    for m in models:
        all_metrics.update(m['metrics'].keys())
    
    for metric in sorted(all_metrics):
        row = [metric]
        for m in models:
            row.append(m['metrics'].get(metric, 'N/A'))
        metrics_table.append(row)
    
    # Generate report
    report = f"""
    <h1>Model Comparison: {model_name}</h1>
    <h2>Versions: {', '.join(versions)}</h2>
    
    <h3>Metrics Comparison</h3>
    <table border="1">
        <tr>
            <th>Metric</th>
            {''.join(f'<th>v{v}</th>' for v in versions)}
        </tr>
        {''.join(f'<tr>{"".join(f"<td>{c}</td>" for c in row)}</tr>' for row in metrics_table)}
    </table>
    
    <h3>Parameter Differences</h3>
    <pre>{json.dumps(compare_params(models), indent=2)}</pre>
    """
    
    return report
```

---

## 🔗 Related Topics

- [Training Infrastructure](./01-training-infrastructure.md) - Where models are trained
- [Experiment Tracking](./03-experiment-tracking.md) - Track training experiments
- [Model Deployment](../05-model-serving/02-model-deployment.md) - Deploy versioned models
