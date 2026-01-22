# Hyperparameter Tuning

## Overview

Hyperparameter tuning optimizes model hyperparameters to improve performance. Automated tuning accelerates the process.

---

## 🎯 Tuning Strategies

### 1. Grid Search
- Exhaustive search over parameter grid
- Simple but expensive
- Good for small parameter spaces

### 2. Random Search
- Random sampling of parameters
- More efficient than grid search
- Good for large parameter spaces

### 3. Bayesian Optimization
- Uses prior knowledge to guide search
- Most efficient
- Best for expensive evaluations

---

## 🛠️ Tools

### Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Train model
    model = train_model(lr, batch_size)
    
    # Return metric to optimize
    return evaluate_model(model)

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Best parameters
print(study.best_params)
```

### Ray Tune

```python
from ray import tune

def train_model(config):
    # Train with config
    model = train(config["lr"], config["batch_size"])
    accuracy = evaluate(model)
    tune.report(accuracy=accuracy)

# Tune
analysis = tune.run(
    train_model,
    config={
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64])
    },
    num_samples=100
)
```

---

## ✅ Best Practices

1. **Define search space** - reasonable ranges
2. **Use appropriate strategy** - Bayesian for expensive evaluations
3. **Early stopping** - stop poor trials early
4. **Parallelize** - run multiple trials
5. **Track experiments** - log all trials

---

## 🔗 Related Topics

- [Experiment Tracking](./experiment-tracking.md)
- [Training Infrastructure](./training-infrastructure.md)
