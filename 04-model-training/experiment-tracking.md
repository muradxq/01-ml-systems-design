# Experiment Tracking

## Overview

Experiment tracking logs experiments, metrics, and artifacts to enable reproducibility and comparison.

---

## 🎯 What to Track

1. **Hyperparameters**: Learning rate, batch size, etc.
2. **Metrics**: Accuracy, loss, F1, etc.
3. **Artifacts**: Models, plots, data samples
4. **Code**: Git commit, code version
5. **Environment**: Dependencies, system info

---

## 🛠️ Tools

### MLflow

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    
    # Train model
    model = train_model(params)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("plot.png")
```

### Weights & Biases

```python
import wandb

wandb.init(project="my_project")

# Log config
wandb.config.learning_rate = 0.01
wandb.config.batch_size = 32

# Train model
model = train_model()

# Log metrics
wandb.log({"accuracy": accuracy, "loss": loss})

# Log model
wandb.log_model(model, "model")
```

---

## ✅ Best Practices

1. **Track everything** - parameters, metrics, artifacts
2. **Use consistent naming** - clear experiment names
3. **Compare experiments** - use dashboards
4. **Document findings** - notes and conclusions
5. **Automate tracking** - integrate into pipelines

---

## 🔗 Related Topics

- [Model Versioning](./model-versioning.md)
- [Hyperparameter Tuning](./hyperparameter-tuning.md)
