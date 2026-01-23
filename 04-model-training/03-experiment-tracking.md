# Experiment Tracking

## Overview

Experiment tracking logs experiments, hyperparameters, metrics, and artifacts to enable reproducibility, comparison, and collaboration. In ML development, running hundreds of experiments is common, and without proper tracking, it becomes impossible to understand what worked, reproduce results, or collaborate effectively.

---

## 🎯 Why Track Experiments?

### 1. Reproducibility
- Recreate any experiment exactly
- Debug model performance changes
- Share results with confidence

### 2. Comparison
- Compare different approaches
- Identify best configurations
- Track progress over time

### 3. Collaboration
- Share experiments with team
- Avoid duplicate work
- Learn from others' experiments

### 4. Documentation
- Automatic experiment documentation
- Audit trail for compliance
- Institutional knowledge

---

## 📊 What to Track

### 1. Hyperparameters

```python
# Everything that affects training
hyperparameters = {
    # Model architecture
    'model_type': 'transformer',
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,
    'dropout': 0.1,
    
    # Training
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam',
    'weight_decay': 0.01,
    
    # Data
    'train_size': 100000,
    'val_size': 10000,
    'augmentation': True,
    
    # Environment
    'random_seed': 42,
    'gpu_type': 'V100'
}
```

### 2. Metrics

```python
# Offline metrics (during training)
training_metrics = {
    # Classification
    'accuracy': 0.95,
    'precision': 0.94,
    'recall': 0.96,
    'f1_score': 0.95,
    'auc_roc': 0.98,
    'auc_pr': 0.97,
    
    # Regression
    'mae': 0.15,
    'mse': 0.03,
    'rmse': 0.17,
    'r2': 0.92,
    
    # Training metrics
    'train_loss': 0.05,
    'val_loss': 0.08,
    'training_time_minutes': 45
}

# Track metrics over time
for epoch in range(epochs):
    metrics = train_epoch()
    log_metrics(metrics, step=epoch)
```

### 3. Artifacts

```python
artifacts = {
    # Models
    'model': 'model.pkl',
    'model_onnx': 'model.onnx',
    'checkpoint': 'checkpoint_best.pt',
    
    # Visualizations
    'confusion_matrix': 'confusion_matrix.png',
    'learning_curves': 'learning_curves.png',
    'feature_importance': 'feature_importance.png',
    'roc_curve': 'roc_curve.png',
    
    # Data
    'predictions': 'predictions.csv',
    'errors': 'error_analysis.csv',
    
    # Code
    'training_script': 'train.py',
    'config': 'config.yaml'
}
```

### 4. Environment

```python
environment = {
    'python_version': '3.10.0',
    'pytorch_version': '2.0.0',
    'cuda_version': '11.8',
    'gpu_name': 'NVIDIA V100',
    'dependencies': 'requirements.txt',
    'git_commit': 'a1b2c3d4',
    'git_branch': 'feature/new-model'
}
```

### 5. Code

```python
# Track code version
code_info = {
    'git_commit': get_git_commit(),
    'git_branch': get_git_branch(),
    'git_diff': get_uncommitted_changes(),
    'entry_point': 'train.py',
    'source_dir': './src'
}
```

---

## 🛠️ Experiment Tracking Tools

### 1. MLflow

**Features:**
- Open source
- Self-hosted or managed
- Model registry integration
- Artifact storage
- Parameter and metric logging

**Complete Example:**

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Set tracking server
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("fraud-detection")

class ExperimentTracker:
    """MLflow-based experiment tracking."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
    
    def start_run(self, run_name: str, tags: dict = None):
        """Start a new run."""
        self.run = mlflow.start_run(run_name=run_name)
        
        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Log environment
        self._log_environment()
        
        return self.run
    
    def log_params(self, params: dict):
        """Log hyperparameters."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """Log model artifact."""
        if isinstance(model, nn.Module):
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        else:
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    
    def log_figure(self, fig, name: str):
        """Log matplotlib figure."""
        mlflow.log_figure(fig, name)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log arbitrary artifact."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names: list):
        """Log confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='Predicted',
               ylabel='True')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        self.log_figure(fig, "confusion_matrix.png")
        plt.close()
    
    def log_learning_curves(self, train_losses: list, val_losses: list):
        """Log learning curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Learning Curves')
        
        self.log_figure(fig, "learning_curves.png")
        plt.close()
    
    def _log_environment(self):
        """Log environment information."""
        import sys
        import platform
        
        mlflow.set_tags({
            'python_version': sys.version,
            'platform': platform.platform(),
            'pytorch_version': torch.__version__,
        })
        
        # Log GPU info
        if torch.cuda.is_available():
            mlflow.set_tag('gpu', torch.cuda.get_device_name(0))
    
    def end_run(self):
        """End the current run."""
        mlflow.end_run()

# Usage
def train_with_tracking():
    tracker = ExperimentTracker("fraud-detection")
    
    # Start run
    with tracker.start_run(run_name="experiment-001", tags={'model': 'transformer'}):
        
        # Log hyperparameters
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'hidden_size': 256
        }
        tracker.log_params(params)
        
        # Training loop
        model = create_model(params)
        train_losses, val_losses = [], []
        
        for epoch in range(params['epochs']):
            train_loss = train_epoch(model)
            val_loss, val_metrics = validate(model)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Log metrics each epoch
            tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            }, step=epoch)
        
        # Final evaluation
        y_true, y_pred = evaluate(model, test_loader)
        final_metrics = compute_metrics(y_true, y_pred)
        tracker.log_metrics(final_metrics)
        
        # Log visualizations
        tracker.log_learning_curves(train_losses, val_losses)
        tracker.log_confusion_matrix(y_true, y_pred, ['legitimate', 'fraud'])
        
        # Log model
        tracker.log_model(model, "model")
```

---

### 2. Weights & Biases (W&B)

**Features:**
- Rich visualization
- Real-time monitoring
- Team collaboration
- Hyperparameter sweeps
- System metrics

```python
import wandb
from wandb.keras import WandbCallback
import torch

class WandbExperimentTracker:
    """Weights & Biases experiment tracking."""
    
    def __init__(self, project: str, entity: str = None):
        self.project = project
        self.entity = entity
    
    def init_run(self, config: dict, name: str = None, tags: list = None):
        """Initialize a new run."""
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config,
            tags=tags
        )
        return wandb.run
    
    def log(self, metrics: dict, step: int = None):
        """Log metrics."""
        wandb.log(metrics, step=step)
    
    def log_image(self, key: str, image):
        """Log image."""
        wandb.log({key: wandb.Image(image)})
    
    def log_table(self, key: str, columns: list, data: list):
        """Log table."""
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters."""
        wandb.watch(model, log_freq=log_freq, log='all')
    
    def log_artifact(self, name: str, artifact_type: str, file_path: str):
        """Log artifact."""
        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish run."""
        wandb.finish()

# Usage with PyTorch
def train_with_wandb():
    tracker = WandbExperimentTracker(project="fraud-detection")
    
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'architecture': 'transformer',
        'hidden_size': 256
    }
    
    run = tracker.init_run(config, name="transformer-v1", tags=['production'])
    
    model = create_model(config)
    tracker.watch_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(config['epochs']):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                tracker.log({
                    'train_loss': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        # Validation
        val_loss, val_acc = validate(model)
        tracker.log({
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch': epoch
        })
        
        # Log sample predictions
        if epoch % 10 == 0:
            log_predictions_table(tracker, model, val_loader)
    
    # Save model
    torch.save(model.state_dict(), 'model.pt')
    tracker.log_artifact('fraud-model', 'model', 'model.pt')
    
    tracker.finish()

def log_predictions_table(tracker, model, val_loader):
    """Log predictions as W&B table."""
    model.eval()
    columns = ['input', 'prediction', 'ground_truth', 'correct']
    data = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            for i in range(min(10, len(inputs))):
                data.append([
                    inputs[i].tolist(),
                    preds[i].item(),
                    targets[i].item(),
                    preds[i].item() == targets[i].item()
                ])
            break
    
    tracker.log_table('predictions', columns, data)
```

---

### 3. TensorBoard

**Features:**
- Built into TensorFlow/PyTorch
- Visualization dashboards
- Scalars, images, histograms
- Model graphs

```python
from torch.utils.tensorboard import SummaryWriter
import torchvision

class TensorBoardTracker:
    """TensorBoard-based experiment tracking."""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
    
    def add_scalar(self, tag: str, value: float, step: int):
        """Add scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag: str, values: dict, step: int):
        """Add multiple scalars."""
        self.writer.add_scalars(main_tag, values, step)
    
    def add_histogram(self, tag: str, values, step: int):
        """Add histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def add_image(self, tag: str, img_tensor, step: int):
        """Add image."""
        self.writer.add_image(tag, img_tensor, step)
    
    def add_images(self, tag: str, img_tensor, step: int):
        """Add multiple images as grid."""
        grid = torchvision.utils.make_grid(img_tensor)
        self.writer.add_image(tag, grid, step)
    
    def add_graph(self, model, input_to_model):
        """Add model graph."""
        self.writer.add_graph(model, input_to_model)
    
    def add_embedding(self, mat, metadata=None, tag='embedding'):
        """Add embedding projection."""
        self.writer.add_embedding(mat, metadata=metadata, tag=tag)
    
    def add_pr_curve(self, tag: str, labels, predictions, step: int):
        """Add precision-recall curve."""
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def add_hparams(self, hparam_dict: dict, metric_dict: dict):
        """Add hyperparameters and metrics."""
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Close writer."""
        self.writer.close()

# Usage
def train_with_tensorboard():
    tracker = TensorBoardTracker(log_dir='./runs/experiment-001')
    
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Log model architecture
    sample_input = torch.randn(1, 3, 224, 224)
    tracker.add_graph(model, sample_input)
    
    for epoch in range(100):
        train_loss = train_epoch(model, optimizer)
        val_loss, val_acc = validate(model)
        
        # Log metrics
        tracker.add_scalars('Loss', {
            'train': train_loss,
            'validation': val_loss
        }, epoch)
        
        tracker.add_scalar('Accuracy/validation', val_acc, epoch)
        
        # Log model weights distribution
        for name, param in model.named_parameters():
            tracker.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                tracker.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    # Log hyperparameters
    tracker.add_hparams(
        {'lr': 0.001, 'batch_size': 32, 'epochs': 100},
        {'accuracy': val_acc, 'loss': val_loss}
    )
    
    tracker.close()
```

---

## 📊 Experiment Comparison

```python
import pandas as pd
from mlflow.tracking import MlflowClient

class ExperimentComparison:
    """Compare and analyze experiments."""
    
    def __init__(self, experiment_name: str):
        self.client = MlflowClient()
        self.experiment = self.client.get_experiment_by_name(experiment_name)
    
    def get_runs_dataframe(self, max_runs: int = 100) -> pd.DataFrame:
        """Get all runs as DataFrame."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            max_results=max_runs
        )
        
        data = []
        for run in runs:
            row = {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'duration': run.info.end_time - run.info.start_time if run.info.end_time else None,
                **run.data.params,
                **run.data.metrics
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_run(self, metric: str, maximize: bool = True) -> dict:
        """Get best run by metric."""
        order = "DESC" if maximize else "ASC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            return {
                'run_id': run.info.run_id,
                'params': run.data.params,
                'metrics': run.data.metrics
            }
        return None
    
    def compare_runs(self, run_ids: list) -> pd.DataFrame:
        """Compare specific runs."""
        rows = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            rows.append({
                'run_id': run_id,
                **run.data.params,
                **run.data.metrics
            })
        return pd.DataFrame(rows)
    
    def parameter_importance(self, metric: str) -> pd.DataFrame:
        """Analyze parameter importance for a metric."""
        df = self.get_runs_dataframe()
        
        # Get parameter columns
        param_cols = [c for c in df.columns if c not in ['run_id', 'run_name', 'status', 'start_time', 'duration', metric]]
        
        correlations = {}
        for param in param_cols:
            try:
                # Convert to numeric if possible
                df[param] = pd.to_numeric(df[param], errors='coerce')
                corr = df[param].corr(df[metric])
                if not pd.isna(corr):
                    correlations[param] = abs(corr)
            except:
                pass
        
        return pd.DataFrame([
            {'parameter': k, 'correlation': v}
            for k, v in sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        ])

# Usage
comparison = ExperimentComparison("fraud-detection")

# Get all runs
runs_df = comparison.get_runs_dataframe()
print(runs_df.head())

# Get best run
best = comparison.get_best_run(metric='auc', maximize=True)
print(f"Best AUC: {best['metrics']['auc']} (run: {best['run_id']})")

# Parameter importance
importance = comparison.parameter_importance('auc')
print(importance)
```

---

## ✅ Best Practices

### 1. Track Everything
- Parameters, metrics, artifacts
- Environment and dependencies
- Code version

### 2. Use Consistent Naming
- Clear experiment names
- Consistent parameter names
- Organized artifact paths

### 3. Compare Experiments
- Use dashboards for visualization
- Document findings
- Share insights with team

### 4. Automate Tracking
- Integrate into training pipeline
- Automatic logging
- CI/CD integration

### 5. Review Regularly
- Weekly experiment reviews
- Clean up old experiments
- Document learnings

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Inconsistent logging** | Hard to compare | Standardize logging |
| **Missing metadata** | Can't reproduce | Log everything |
| **Too many experiments** | Hard to navigate | Use tags, organize |
| **No cleanup** | Storage costs | Archive old experiments |
| **Manual tracking** | Prone to errors | Automate |

---

## 🔗 Related Topics

- [Model Versioning](./model-versioning.md) - Version trained models
- [Hyperparameter Tuning](./hyperparameter-tuning.md) - Systematic tuning
- [Training Infrastructure](./training-infrastructure.md) - Run experiments
