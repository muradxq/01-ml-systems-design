# Hyperparameter Tuning

## Overview

Hyperparameter tuning optimizes model configuration for best performance. Unlike model parameters (weights) learned during training, hyperparameters are set before training and significantly impact model performance. Systematic tuning can improve model accuracy by 10-30% and is a critical step in productionizing ML models.

---

## 🎯 Types of Hyperparameters

### Model Architecture

```python
architecture_params = {
    # Neural Networks
    'hidden_layers': [1, 2, 3, 4, 5],
    'hidden_size': [64, 128, 256, 512, 1024],
    'activation': ['relu', 'tanh', 'gelu', 'silu'],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
    
    # Transformers
    'num_heads': [4, 8, 12, 16],
    'num_layers': [2, 4, 6, 12],
    'embedding_dim': [256, 512, 768, 1024],
    
    # Tree models
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'n_estimators': [100, 200, 500, 1000]
}
```

### Training Configuration

```python
training_params = {
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64, 128, 256],
    'epochs': [10, 50, 100, 200],
    'optimizer': ['adam', 'sgd', 'adamw', 'rmsprop'],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'gradient_clipping': [None, 0.5, 1.0, 5.0]
}
```

### Regularization

```python
regularization_params = {
    'l1_lambda': [0, 1e-5, 1e-4, 1e-3],
    'l2_lambda': [0, 1e-5, 1e-4, 1e-3],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3],
    'early_stopping_patience': [5, 10, 20],
    'data_augmentation': [True, False]
}
```

---

## 📊 Tuning Strategies

### 1. Grid Search

**How it works:** Exhaustively search all combinations

```
┌─────────────────────────────────────────────────────────────────┐
│  Grid Search - Exhaustive Combinatorial Search                   │
│                                                                  │
│  learning_rate:    0.001    0.01     0.1                        │
│                      │        │        │                        │
│  batch_size: 32  ────●────────●────────●                        │
│              64  ────●────────●────────●                        │
│             128  ────●────────●────────●                        │
│                                                                  │
│  Total experiments: 3 × 3 = 9                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Simple to implement
- Guaranteed to find best in search space
- Parallelizable

**Cons:**
- Computationally expensive (exponential)
- Inefficient for high dimensions
- Wastes resources on poor regions

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Total combinations: 3 × 4 × 3 × 3 = 108

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,  # Parallel execution
    verbose=2,
    return_train_score=True
)

# Run search
grid_search.fit(X_train, y_train)

# Results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best AUC: {grid_search.best_score_:.4f}")

# Analyze results
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['param_n_estimators', 'param_max_depth', 'mean_test_score']]
      .sort_values('mean_test_score', ascending=False)
      .head(10))
```

---

### 2. Random Search

**How it works:** Sample random combinations from distributions

```
┌─────────────────────────────────────────────────────────────────┐
│  Random Search - Stochastic Sampling                             │
│                                                                  │
│  learning_rate (log uniform): 1e-5 ◄──────────────────► 1e-1    │
│                                        ●      ●   ●     ●       │
│                                                                  │
│  batch_size (choice): [16, 32, 64, 128]                         │
│                        ●       ●     ●         ●                │
│                                                                  │
│  hidden_size (uniform): 64 ◄──────────────────────────► 512     │
│                             ●    ●       ●       ●              │
│                                                                  │
│  Samples: 20 random combinations                                │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:**
- More efficient than grid search
- Better coverage of continuous spaces
- Easy to add budget

**Cons:**
- No guarantee of finding optimal
- Random, no learning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint

# Define distributions
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
    'bootstrap': [True, False]
}

# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # Number of random samples
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best AUC: {random_search.best_score_:.4f}")
```

---

### 3. Bayesian Optimization

**How it works:** Build probabilistic model of objective function, use it to select promising points

```
┌─────────────────────────────────────────────────────────────────┐
│  Bayesian Optimization - Guided Search                           │
│                                                                  │
│  Objective │                    ●                               │
│  Function  │              ●           ●                         │
│            │         ●                      ●                   │
│            │    ●                                 ●             │
│            │●                                          ●        │
│            └────────────────────────────────────────────────    │
│                              Hyperparameter                      │
│                                                                  │
│  Surrogate Model (Gaussian Process):                            │
│  - Models the objective function                                │
│  - Provides uncertainty estimates                               │
│                                                                  │
│  Acquisition Function:                                          │
│  - Balances exploration vs exploitation                         │
│  - Suggests next point to evaluate                              │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Most sample-efficient
- Learns from previous evaluations
- Handles continuous and discrete

**Cons:**
- More complex
- Overhead for cheap objectives
- Can get stuck in local optima

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn

# Define objective function
def objective(trial):
    # Suggest hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'hidden_size': trial.suggest_int('hidden_size', 64, 512, step=64),
        'num_layers': trial.suggest_int('num_layers', 1, 5),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    }
    
    # Create model with suggested params
    model = create_model(params)
    
    # Training loop with pruning
    for epoch in range(100):
        train_loss = train_epoch(model, params['batch_size'])
        val_loss, val_acc = validate(model)
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_acc

# Create study
study = optuna.create_study(
    study_name='neural-network-tuning',
    direction='maximize',
    sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

# Optimize
study.optimize(
    objective,
    n_trials=100,
    timeout=3600,  # 1 hour timeout
    n_jobs=4,  # Parallel trials
    show_progress_bar=True
)

# Results
print(f"Best trial:")
print(f"  Value: {study.best_trial.value:.4f}")
print(f"  Params: {study.best_trial.params}")

# Visualization
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_param_importances(study)
fig.show()

fig = optuna.visualization.plot_contour(study, params=['learning_rate', 'hidden_size'])
fig.show()
```

---

### 4. Hyperband / ASHA

**How it works:** Allocate resources adaptively, early stop poor configurations

```
┌─────────────────────────────────────────────────────────────────┐
│  Hyperband - Successive Halving                                  │
│                                                                  │
│  Bracket 1 (many configs, few resources):                       │
│  ──●──●──●──●──●──●──●──●──●  (9 configs, 1 epoch each)        │
│       │     │     │     │                                       │
│  ──●──●──●──●  (4 best continue, 3 epochs each)                │
│       │     │                                                   │
│  ──●──●  (2 best continue, 9 epochs each)                      │
│       │                                                         │
│  ──●  (1 best continues, 27 epochs)                            │
│                                                                  │
│  Total budget: 9×1 + 4×3 + 2×9 + 1×27 = 48 epochs              │
│  vs Grid: 9 × 27 = 243 epochs for same configs                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def train_model(config):
    """Training function for Ray Tune."""
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, optimizer, config['batch_size'])
        val_loss, val_acc = validate(model)
        
        # Report metrics to Ray Tune
        tune.report(
            loss=val_loss,
            accuracy=val_acc,
            epoch=epoch
        )

# Configuration space
config = {
    'learning_rate': tune.loguniform(1e-5, 1e-1),
    'batch_size': tune.choice([16, 32, 64, 128]),
    'hidden_size': tune.choice([64, 128, 256, 512]),
    'num_layers': tune.randint(1, 6),
    'dropout': tune.uniform(0, 0.5),
    'epochs': 50
}

# ASHA scheduler for early stopping
scheduler = ASHAScheduler(
    metric='accuracy',
    mode='max',
    max_t=50,  # Max epochs
    grace_period=5,  # Min epochs before pruning
    reduction_factor=3
)

# Optuna search
search_alg = OptunaSearch(metric='accuracy', mode='max')

# Run tuning
analysis = tune.run(
    train_model,
    config=config,
    scheduler=scheduler,
    search_alg=search_alg,
    num_samples=100,
    resources_per_trial={'cpu': 2, 'gpu': 0.5},
    local_dir='./ray_results',
    name='hyperparameter_tuning'
)

# Best configuration
best_config = analysis.best_config
best_accuracy = analysis.best_result['accuracy']
print(f"Best config: {best_config}")
print(f"Best accuracy: {best_accuracy:.4f}")
```

---

## 🛠️ Hyperparameter Tuning Tools

### Tool Comparison

| Tool | Type | Distributed | Integration | Complexity |
|------|------|-------------|-------------|------------|
| **Optuna** | Bayesian | Yes | Any framework | Medium |
| **Ray Tune** | Multi-strategy | Yes | Any framework | High |
| **Keras Tuner** | Bayesian | No | Keras | Low |
| **SageMaker** | Bayesian | Yes | AWS | Medium |
| **W&B Sweeps** | Grid/Random/Bayes | Yes | Any | Medium |

### Optuna Advanced Features

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl

class OptunaTuner:
    """Advanced Optuna-based hyperparameter tuning."""
    
    def __init__(
        self,
        study_name: str,
        storage: str = None,
        direction: str = 'maximize',
        n_trials: int = 100
    ):
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,  # e.g., 'sqlite:///optuna.db' or 'postgresql://...'
            direction=direction,
            load_if_exists=True
        )
        self.n_trials = n_trials
    
    def define_search_space(self, trial) -> dict:
        """Define hyperparameter search space."""
        return {
            # Model architecture
            'hidden_size': trial.suggest_int('hidden_size', 64, 1024, log=True),
            'num_layers': trial.suggest_int('num_layers', 1, 6),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'silu']),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            
            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            
            # Learning rate scheduler
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'plateau']),
            
            # Conditional parameters
            'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000)
            if trial.params.get('scheduler') == 'cosine' else 0
        }
    
    def objective(self, trial) -> float:
        """Objective function to optimize."""
        params = self.define_search_space(trial)
        
        # Create model and train
        model = self.create_model(params)
        
        for epoch in range(50):
            train_loss = self.train_epoch(model, params)
            val_loss, val_acc = self.validate(model)
            
            # Report for pruning
            trial.report(val_acc, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return val_acc
    
    def run(self):
        """Run hyperparameter optimization."""
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=-1,
            show_progress_bar=True
        )
        
        return self.study.best_params, self.study.best_value
    
    def get_importance(self) -> dict:
        """Get hyperparameter importance."""
        return optuna.importance.get_param_importances(self.study)
    
    def visualize(self):
        """Generate visualizations."""
        figs = {
            'history': optuna.visualization.plot_optimization_history(self.study),
            'importance': optuna.visualization.plot_param_importances(self.study),
            'parallel': optuna.visualization.plot_parallel_coordinate(self.study),
            'slice': optuna.visualization.plot_slice(self.study)
        }
        return figs
```

### Weights & Biases Sweeps

```python
import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'hidden_size': {
            'distribution': 'int_uniform',
            'min': 64,
            'max': 512
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5
    }
}

def train_sweep():
    """Training function for W&B sweep."""
    # Initialize run
    run = wandb.init()
    config = wandb.config
    
    # Create model with sweep config
    model = create_model(
        hidden_size=config.hidden_size,
        dropout=config.dropout
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )
    
    # Training loop
    for epoch in range(50):
        train_loss = train_epoch(model, optimizer, config.batch_size)
        val_loss, val_acc = validate(model)
        
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch': epoch
        })
    
    run.finish()

# Create and run sweep
sweep_id = wandb.sweep(sweep_config, project='hyperparameter-tuning')
wandb.agent(sweep_id, train_sweep, count=100)
```

---

## 📊 Multi-Objective Optimization

```python
import optuna

def multi_objective(trial):
    """Optimize for both accuracy and latency."""
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 32, 512),
        'num_layers': trial.suggest_int('num_layers', 1, 6),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    }
    
    model = create_model(params)
    
    # Train model
    train_model(model, params)
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(model)
    
    # Measure latency
    latency = measure_latency(model)
    
    return accuracy, latency  # Maximize accuracy, minimize latency

# Multi-objective study
study = optuna.create_study(
    directions=['maximize', 'minimize'],
    study_name='multi-objective-tuning'
)

study.optimize(multi_objective, n_trials=100)

# Get Pareto front
pareto_trials = study.best_trials
print(f"Found {len(pareto_trials)} Pareto-optimal solutions")

for trial in pareto_trials:
    print(f"Accuracy: {trial.values[0]:.4f}, Latency: {trial.values[1]:.2f}ms")
    print(f"Params: {trial.params}")
```

---

## ✅ Best Practices

### 1. Start Simple
- Begin with random search
- Move to Bayesian if needed
- Use pruning early

### 2. Define Good Search Space
- Use domain knowledge
- Log-scale for learning rates
- Reasonable bounds

### 3. Use Cross-Validation
- Robust estimates
- Avoid overfitting to validation set
- K-fold or stratified

### 4. Track All Experiments
- Log all trials
- Save configurations
- Version control

### 5. Budget Appropriately
- Allocate time based on importance
- Use early stopping
- Parallelize when possible

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Too large search space** | Wasted resources | Narrow based on domain knowledge |
| **No early stopping** | Slow convergence | Use pruning |
| **Overfitting to val** | Poor generalization | Use nested CV |
| **Not enough trials** | Miss optimal | More trials, better sampling |
| **Ignoring interactions** | Suboptimal | Consider joint distributions |

---

## 🔗 Related Topics

- [Experiment Tracking](./03-experiment-tracking.md) - Track tuning experiments
- [Training Infrastructure](./01-training-infrastructure.md) - Run parallel trials
- [Distributed Training](./05-distributed-training.md) - Scale tuning
