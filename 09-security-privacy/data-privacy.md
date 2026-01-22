# Data Privacy

## Overview

Data privacy protects sensitive user data throughout the ML lifecycle—from collection and storage to training and inference. ML systems face unique privacy challenges because models can memorize training data, and even aggregate statistics can leak individual information. Proper privacy protection is both an ethical imperative and legal requirement.

---

## 🎯 Privacy Techniques

### 1. Data Anonymization

| Technique | Description | Strength | Use Case |
|-----------|-------------|----------|----------|
| **Suppression** | Remove sensitive fields | High | Remove names, SSNs |
| **Generalization** | Replace with ranges | Medium | Age → Age group |
| **Pseudonymization** | Replace with tokens | Medium | Reversible mapping |
| **K-Anonymity** | Ensure k identical records | Medium | Statistical privacy |
| **L-Diversity** | Ensure l diverse values | Higher | Sensitive attributes |

```python
import hashlib
from typing import Dict, Any
import pandas as pd

class DataAnonymizer:
    """Anonymization techniques for ML data."""
    
    def __init__(self, salt: str):
        self.salt = salt
    
    def suppress_fields(self, df: pd.DataFrame, 
                       fields: list) -> pd.DataFrame:
        """Remove sensitive fields entirely."""
        return df.drop(columns=fields, errors='ignore')
    
    def pseudonymize(self, value: str) -> str:
        """Replace with irreversible hash."""
        salted = f"{self.salt}:{value}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]
    
    def generalize_age(self, age: int) -> str:
        """Generalize age to buckets."""
        if age < 18: return "under_18"
        elif age < 25: return "18-24"
        elif age < 35: return "25-34"
        elif age < 45: return "35-44"
        elif age < 55: return "45-54"
        elif age < 65: return "55-64"
        else: return "65+"
    
    def generalize_location(self, zipcode: str) -> str:
        """Generalize location (keep first 3 digits)."""
        return zipcode[:3] + "XX" if len(zipcode) >= 3 else "XXXXX"
    
    def k_anonymize(self, df: pd.DataFrame, 
                   quasi_identifiers: list, k: int) -> pd.DataFrame:
        """
        Ensure each combination of quasi-identifiers
        appears at least k times.
        """
        # Group by quasi-identifiers
        groups = df.groupby(quasi_identifiers).size()
        
        # Find groups with fewer than k records
        small_groups = groups[groups < k].index.tolist()
        
        # Suppress or generalize small groups
        for group in small_groups:
            mask = True
            for col, val in zip(quasi_identifiers, group):
                mask &= (df[col] == val)
            df = df[~mask]
        
        return df

# Usage
anonymizer = DataAnonymizer(salt="secret_salt_123")

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare privacy-safe training data."""
    # Remove direct identifiers
    df = anonymizer.suppress_fields(df, ['name', 'email', 'ssn', 'phone'])
    
    # Pseudonymize user IDs
    df['user_id'] = df['user_id'].apply(anonymizer.pseudonymize)
    
    # Generalize quasi-identifiers
    df['age_group'] = df['age'].apply(anonymizer.generalize_age)
    df['location'] = df['zipcode'].apply(anonymizer.generalize_location)
    df = df.drop(columns=['age', 'zipcode'])
    
    # Apply k-anonymity
    df = anonymizer.k_anonymize(df, ['age_group', 'location', 'gender'], k=5)
    
    return df
```

### 2. Differential Privacy

```python
import numpy as np
from typing import Callable

class DifferentialPrivacy:
    """
    Differential privacy for ML training and inference.
    Provides mathematical guarantee that individual records
    cannot be identified from outputs.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy leak
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def laplace_mechanism(self, true_value: float, 
                         sensitivity: float) -> float:
        """
        Add Laplace noise for (ε)-differential privacy.
        
        Args:
            true_value: Actual value to protect
            sensitivity: Maximum change from one record
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise
    
    def gaussian_mechanism(self, true_value: float,
                          sensitivity: float) -> float:
        """
        Add Gaussian noise for (ε,δ)-differential privacy.
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        return true_value + noise
    
    def private_mean(self, values: np.ndarray, 
                    clip_bound: float) -> float:
        """
        Compute differentially private mean.
        """
        # Clip values to bound sensitivity
        clipped = np.clip(values, -clip_bound, clip_bound)
        
        # True mean
        true_mean = np.mean(clipped)
        
        # Sensitivity = 2 * clip_bound / n
        sensitivity = 2 * clip_bound / len(values)
        
        # Add noise
        return self.laplace_mechanism(true_mean, sensitivity)
    
    def private_count(self, predicate: Callable, data: list) -> int:
        """
        Differentially private count.
        """
        true_count = sum(1 for item in data if predicate(item))
        # Count sensitivity is always 1
        noisy_count = self.laplace_mechanism(true_count, sensitivity=1)
        return max(0, int(round(noisy_count)))

# Differential privacy in model training
from opacus import PrivacyEngine

def train_with_differential_privacy(model, train_loader, epochs=10,
                                   target_epsilon=1.0):
    """
    Train PyTorch model with differential privacy using Opacus.
    """
    # Initialize privacy engine
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=1e-5,
        max_grad_norm=1.0  # Gradient clipping for bounded sensitivity
    )
    
    # Train as usual
    for epoch in range(epochs):
        for batch, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Get actual privacy spent
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Training complete. Privacy budget spent: ε={epsilon:.2f}")
    
    return model
```

### 3. Federated Learning

```python
from typing import List, Dict
import numpy as np

class FederatedLearning:
    """
    Federated learning: train models without centralizing data.
    """
    
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_updates = []
    
    def client_train(self, client_data, client_id: str, 
                    local_epochs: int = 5) -> Dict:
        """
        Train on client device (runs on edge).
        Only model updates are sent, not data.
        """
        # Start with global model weights
        local_model = self.global_model.copy()
        
        # Train locally
        for epoch in range(local_epochs):
            for batch in client_data:
                local_model.train_step(batch)
        
        # Compute update (difference from global)
        update = {
            name: local_model.weights[name] - self.global_model.weights[name]
            for name in local_model.weights
        }
        
        return {
            'client_id': client_id,
            'update': update,
            'num_samples': len(client_data)
        }
    
    def aggregate_updates(self, updates: List[Dict]) -> Dict:
        """
        Federated averaging: aggregate client updates.
        Weighted by number of samples.
        """
        total_samples = sum(u['num_samples'] for u in updates)
        
        # Weighted average of updates
        aggregated = {}
        for name in updates[0]['update']:
            weighted_sum = sum(
                u['update'][name] * u['num_samples']
                for u in updates
            )
            aggregated[name] = weighted_sum / total_samples
        
        return aggregated
    
    def update_global_model(self, aggregated_update: Dict):
        """Apply aggregated update to global model."""
        for name in self.global_model.weights:
            self.global_model.weights[name] += aggregated_update[name]
    
    def federated_round(self, client_data_loaders: List):
        """
        Run one round of federated learning.
        """
        # 1. Distribute global model to clients
        # 2. Clients train locally
        updates = [
            self.client_train(loader, f"client_{i}")
            for i, loader in enumerate(client_data_loaders)
        ]
        
        # 3. Aggregate updates
        aggregated = self.aggregate_updates(updates)
        
        # 4. Update global model
        self.update_global_model(aggregated)
        
        return self.global_model
```

---

## 📊 Privacy-Preserving ML Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raw Data (PII)                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Anonymization Layer                           │
│  - Remove direct identifiers                                     │
│  - Pseudonymize user IDs                                         │
│  - Generalize quasi-identifiers                                  │
│  - Apply k-anonymity                                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Privacy-Safe Training Data                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Training                                │
│  - Differential privacy (Opacus)                                 │
│  - Gradient clipping                                             │
│  - Noise injection                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Private Model                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference                                     │
│  - Input validation                                              │
│  - Output perturbation (optional)                                │
│  - Rate limiting                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Best Practices

1. **Data minimization** - collect only what's needed
2. **Purpose limitation** - use data only for stated purpose
3. **Storage limitation** - delete when no longer needed
4. **Anonymize early** - before storage/processing
5. **Audit access** - log who accesses what
6. **Encrypt everywhere** - at rest and in transit
7. **Regular reviews** - assess privacy practices

---

## 🔗 Related Topics

- [Model Security](./model-security.md) - Protect models from attacks
- [Compliance](./compliance.md) - Meet regulatory requirements
- [Access Control](./access-control.md) - Control data access
