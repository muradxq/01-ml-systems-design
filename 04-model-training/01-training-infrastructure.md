# Training Infrastructure

## Overview

Training infrastructure provides the compute resources, orchestration, and tooling needed to train ML models efficiently and reliably. Proper infrastructure enables faster iteration, reproducible experiments, and cost-effective scaling. As models grow in size and complexity, training infrastructure becomes a critical competitive advantage.

---

## 🏗️ Infrastructure Components

### 1. Compute Resources

**Types:**

| Resource | Best For | Cost | Performance |
|----------|----------|------|-------------|
| **CPUs** | Traditional ML, small models | $ | Baseline |
| **GPUs** | Deep learning, parallel ops | $$$ | 10-100x faster |
| **TPUs** | TensorFlow, large models | $$$$ | Optimized for ML |
| **Custom ASICs** | Specific workloads | $$$$$ | Maximum efficiency |

**GPU Selection Guide:**

| GPU | Memory | Use Case | Typical Cost/hour |
|-----|--------|----------|-------------------|
| **T4** | 16GB | Inference, small training | $0.35-0.75 |
| **V100** | 32GB | Medium models | $2-3 |
| **A100** | 40-80GB | Large models, multi-GPU | $3-5 |
| **H100** | 80GB | LLMs, massive scale | $5-10 |

**Resource Selection Criteria:**
- Model type (deep learning needs GPUs)
- Dataset size
- Training time requirements
- Budget constraints
- Memory requirements

---

### 2. Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Infrastructure Storage               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Hot Storage (Fast Access)                                 ││
│  │  - NVMe SSDs for active training data                      ││
│  │  - Local disk for checkpoints during training              ││
│  │  - High IOPS, low latency                                  ││
│  └────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Warm Storage (Frequent Access)                            ││
│  │  - Network-attached storage (EFS, Filestore)               ││
│  │  - Shared across training jobs                             ││
│  │  - Good throughput, moderate latency                       ││
│  └────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Cold Storage (Archive)                                    ││
│  │  - Object storage (S3, GCS)                                ││
│  │  - Training datasets, model artifacts                      ││
│  │  - Cost-effective, high durability                         ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Storage Requirements:**
- Training data access (high throughput)
- Model artifact storage (versioned)
- Checkpoint storage (fast writes)
- Log storage (append-only)

**Solutions:**
- Object storage (S3, GCS) for datasets and artifacts
- Network-attached storage for shared data
- Local SSD for temporary data and checkpoints
- Distributed file systems (HDFS) for large-scale

---

### 3. Orchestration

**Purpose:** Manage training jobs, resources, and dependencies

**Features:**
- Job scheduling and queuing
- Resource allocation and management
- Dependency tracking
- Failure handling and retries
- Auto-scaling

**Tools:**

| Tool | Type | Best For |
|------|------|----------|
| **Kubeflow** | Open source | Kubernetes-native ML |
| **SageMaker** | Managed | AWS ecosystem |
| **Vertex AI** | Managed | GCP ecosystem |
| **Azure ML** | Managed | Azure ecosystem |
| **Ray** | Open source | Distributed computing |
| **Airflow** | Open source | Pipeline orchestration |

---

## 🛠️ Training Infrastructure Patterns

### 1. Cloud Managed Services

**Examples:**
- AWS SageMaker
- Google Vertex AI
- Azure ML
- Databricks

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Managed ML Platform                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  User Interface (SDK, Console, CLI)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Training Job Manager                                    │   │
│  │  - Job submission                                        │   │
│  │  - Resource provisioning                                 │   │
│  │  - Monitoring                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Managed Compute                                         │   │
│  │  - Auto-provisioned instances                            │   │
│  │  - GPU/CPU clusters                                      │   │
│  │  - Spot/preemptible instances                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Easy to use, minimal setup
- Managed infrastructure
- Built-in features (monitoring, logging)
- Scalable

**Cons:**
- Vendor lock-in
- Less flexible
- Can be expensive at scale
- Limited customization

**Example: SageMaker Training Job**

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Configure training job
estimator = PyTorch(
    entry_point='train.py',
    source_dir='./src',
    role='arn:aws:iam::xxx:role/SageMakerRole',
    instance_count=2,
    instance_type='ml.p3.2xlarge',
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'val:accuracy', 'Regex': 'accuracy: ([0-9\\.]+)'}
    ]
)

# Start training
estimator.fit({
    'train': 's3://bucket/train',
    'validation': 's3://bucket/val'
})
```

---

### 2. Kubernetes-Based Infrastructure

**Examples:**
- Kubeflow
- MLflow on Kubernetes
- Custom operators

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Kubeflow Components                                     │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐             │   │
│  │  │ Notebooks │ │ Pipelines │ │  Katib    │             │   │
│  │  │           │ │           │ │ (HPO)     │             │   │
│  │  └───────────┘ └───────────┘ └───────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Training Operators                                      │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐             │   │
│  │  │ TFJob     │ │ PyTorchJob│ │ MPIJob    │             │   │
│  │  │           │ │           │ │           │             │   │
│  │  └───────────┘ └───────────┘ └───────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Compute Nodes                                           │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐             │   │
│  │  │ GPU Node  │ │ GPU Node  │ │ CPU Node  │             │   │
│  │  │ (Worker)  │ │ (Worker)  │ │ (PS)      │             │   │
│  │  └───────────┘ └───────────┘ └───────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Kubeflow PyTorchJob Example:**

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-training-job
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch-training:latest
            command: ["python", "train.py"]
            args:
              - "--epochs=10"
              - "--batch-size=64"
            resources:
              limits:
                nvidia.com/gpu: 2
                memory: "32Gi"
                cpu: "8"
            volumeMounts:
            - name: data
              mountPath: /data
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: training-data-pvc
    Worker:
      replicas: 4
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch-training:latest
            command: ["python", "train.py"]
            resources:
              limits:
                nvidia.com/gpu: 2
```

**Pros:**
- Flexible and customizable
- Portable across clouds
- Scalable
- Open source

**Cons:**
- Complex setup
- Requires Kubernetes expertise
- More maintenance
- Steeper learning curve

---

### 3. Hybrid Approach

**Pattern:** Use managed services for common cases, custom infrastructure for special needs

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Training Infrastructure                │
│                                                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │  Standard Training       │  │  Custom Training         │   │
│  │  (Managed Service)       │  │  (Kubernetes)            │   │
│  │                          │  │                          │   │
│  │  - Quick experiments     │  │  - Large-scale training  │   │
│  │  - Small/medium models   │  │  - Custom frameworks     │   │
│  │  - Simple workflows      │  │  - Specific hardware     │   │
│  │                          │  │  - Complex pipelines     │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Shared Services                                         │   │
│  │  - Model Registry (MLflow)                               │   │
│  │  - Feature Store (Feast)                                 │   │
│  │  - Experiment Tracking (W&B)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Training Pipeline Implementation

### Complete Training Pipeline

```python
import mlflow
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

@dataclass
class TrainingConfig:
    """Configuration for training job."""
    model_name: str
    dataset_path: str
    output_path: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    num_workers: int = 4
    checkpoint_interval: int = 1
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    distributed: bool = False

class TrainingPipeline:
    """
    Production training pipeline with MLflow tracking,
    checkpointing, and monitoring.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device()
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
    
    def run(self) -> Dict[str, Any]:
        """Execute training pipeline."""
        
        # Start MLflow run
        with mlflow.start_run(run_name=self.config.model_name):
            # Log configuration
            mlflow.log_params({
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'mixed_precision': self.config.mixed_precision
            })
            
            # Initialize components
            model = self._create_model()
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)
            train_loader, val_loader = self._create_data_loaders()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                # Train epoch
                train_metrics = self._train_epoch(
                    model, train_loader, optimizer, epoch
                )
                
                # Validate
                val_metrics = self._validate(model, val_loader)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, step=epoch)
                
                # Learning rate scheduling
                scheduler.step(val_metrics['loss'])
                
                # Checkpointing
                if epoch % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(model, optimizer, epoch, val_metrics)
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self._save_best_model(model)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}"
                )
            
            # Log final model
            mlflow.pytorch.log_model(model, "model")
            
            return {
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }
    
    def _train_epoch(self, model, train_loader, optimizer, epoch) -> Dict:
        """Train for one epoch with mixed precision."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate(self, model, val_loader) -> Dict:
        """Validate model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        path = f"{self.config.output_path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def _save_best_model(self, model):
        """Save best model."""
        path = f"{self.config.output_path}/best_model.pt"
        torch.save(model.state_dict(), path)
        self.logger.info(f"Saved best model to {path}")

# Usage
if __name__ == "__main__":
    config = TrainingConfig(
        model_name="image-classifier-v1",
        dataset_path="s3://bucket/train",
        output_path="/models/output",
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        mixed_precision=True
    )
    
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    print(f"Training complete: {results}")
```

---

## 💰 Cost Optimization

### 1. Spot/Preemptible Instances

```python
# AWS SageMaker with Spot Instances
estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p3.2xlarge',
    instance_count=2,
    use_spot_instances=True,
    max_wait=7200,  # Max wait time including spot delays
    max_run=3600,   # Max training time
    checkpoint_s3_uri='s3://bucket/checkpoints/'  # For spot interruption recovery
)
```

### 2. Auto-scaling

```yaml
# Kubernetes cluster autoscaler config
apiVersion: autoscaling.k8s.io/v1
kind: ClusterAutoscaler
metadata:
  name: training-cluster-autoscaler
spec:
  scaleDownEnabled: true
  scaleDownDelayAfterAdd: 10m
  scaleDownUnneededTime: 5m
  nodeGroups:
  - name: gpu-nodes
    minSize: 0
    maxSize: 50
    instanceType: p3.2xlarge
```

### 3. Resource Right-sizing

```python
def estimate_resource_requirements(model, dataset_size, batch_size):
    """Estimate GPU memory and compute requirements."""
    
    # Model memory
    model_params = sum(p.numel() for p in model.parameters())
    model_memory_gb = model_params * 4 / (1024**3)  # FP32
    
    # Gradient memory (same as model)
    gradient_memory_gb = model_memory_gb
    
    # Optimizer state (Adam uses 2x model size)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Activation memory (approximate)
    activation_memory_gb = batch_size * estimate_activation_size(model)
    
    total_memory_gb = (
        model_memory_gb + 
        gradient_memory_gb + 
        optimizer_memory_gb + 
        activation_memory_gb
    )
    
    # Recommend GPU
    if total_memory_gb < 8:
        return "T4 (16GB)"
    elif total_memory_gb < 16:
        return "V100 (32GB)"
    elif total_memory_gb < 40:
        return "A100-40GB"
    else:
        return "A100-80GB or multi-GPU"
```

---

## ✅ Best Practices

### 1. Use Appropriate Compute
- GPUs for deep learning
- TPUs for large TensorFlow models
- CPUs for traditional ML

### 2. Optimize Data Loading
- Use efficient data formats (TFRecord, WebDataset)
- Prefetch data during training
- Use multiple data loading workers

### 3. Implement Checkpointing
- Regular checkpoint saves
- Enable resumption from checkpoints
- Store checkpoints in durable storage

### 4. Monitor Resources
- Track GPU utilization
- Monitor memory usage
- Alert on underutilization

### 5. Automate Pipelines
- Use CI/CD for training
- Automate hyperparameter tuning
- Version everything

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **GPU underutilization** | Low GPU usage | Increase batch size, optimize data loading |
| **OOM errors** | Training crashes | Gradient checkpointing, mixed precision |
| **Slow data loading** | CPU bottleneck | More workers, efficient formats, caching |
| **No checkpointing** | Lost progress on failure | Regular checkpoints |
| **Wrong instance type** | Over/under provisioned | Profile and right-size |

---

## 📊 Resource Monitoring

```python
import GPUtil
import psutil
import time
from prometheus_client import Gauge

# Prometheus metrics
gpu_utilization = Gauge('training_gpu_utilization', 'GPU utilization %')
gpu_memory_used = Gauge('training_gpu_memory_gb', 'GPU memory used GB')
cpu_utilization = Gauge('training_cpu_utilization', 'CPU utilization %')

def monitor_resources():
    """Monitor and report resource utilization."""
    while True:
        # GPU metrics
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_utilization.set(gpu.load * 100)
            gpu_memory_used.set(gpu.memoryUsed / 1024)
        
        # CPU metrics
        cpu_utilization.set(psutil.cpu_percent())
        
        time.sleep(10)
```

---

## 🔗 Related Topics

- [Model Versioning](./model-versioning.md) - Version trained models
- [Experiment Tracking](./experiment-tracking.md) - Track experiments
- [Distributed Training](./distributed-training.md) - Scale training
- [Hyperparameter Tuning](./hyperparameter-tuning.md) - Optimize hyperparameters
