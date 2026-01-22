# Model Security

## Overview

Model security protects ML models from theft, adversarial attacks, and misuse. ML models are valuable intellectual property and can be exploited in ways traditional software cannot—through adversarial inputs, model extraction attacks, or data poisoning. Comprehensive model security is essential for production systems.

---

## 🚨 Attack Types

### 1. Model Extraction Attacks

**Goal:** Steal the model by querying it repeatedly

```python
class ModelExtractionDefense:
    """Defenses against model extraction attacks."""
    
    def __init__(self, max_queries_per_user: int = 1000,
                 max_queries_per_minute: int = 60):
        self.query_counts = {}
        self.rate_limiter = RateLimiter(max_queries_per_minute)
    
    def check_request(self, user_id: str, features: dict) -> bool:
        """Check if request should be allowed."""
        # Rate limiting
        if not self.rate_limiter.allow(user_id):
            raise RateLimitExceeded()
        
        # Query budget
        if user_id not in self.query_counts:
            self.query_counts[user_id] = 0
        self.query_counts[user_id] += 1
        
        if self.query_counts[user_id] > self.max_queries_per_user:
            raise QueryBudgetExceeded()
        
        # Detect suspicious patterns
        if self._is_suspicious(user_id, features):
            self._alert_security_team(user_id)
            raise SuspiciousActivityDetected()
        
        return True
    
    def _is_suspicious(self, user_id: str, features: dict) -> bool:
        """Detect extraction attack patterns."""
        # Grid search pattern detection
        # High query volume with systematic variations
        # Queries near decision boundaries
        pass
    
    def add_prediction_perturbation(self, prediction: float,
                                   confidence: float) -> float:
        """
        Add small random perturbation to predictions.
        Makes extraction less accurate while preserving utility.
        """
        if confidence > 0.9:  # High confidence predictions
            noise = np.random.uniform(-0.01, 0.01)
        else:
            noise = np.random.uniform(-0.05, 0.05)
        return np.clip(prediction + noise, 0, 1)
```

### 2. Adversarial Attacks

**Goal:** Craft inputs that cause misclassification

```python
import numpy as np
from typing import Callable

class AdversarialDefense:
    """Defenses against adversarial examples."""
    
    def __init__(self, model):
        self.model = model
    
    def detect_adversarial(self, x: np.ndarray) -> bool:
        """
        Detect potential adversarial examples.
        """
        # Method 1: Input validation
        if not self._is_valid_input(x):
            return True
        
        # Method 2: Feature squeezing
        squeezed = self._squeeze_features(x)
        pred_original = self.model.predict(x)
        pred_squeezed = self.model.predict(squeezed)
        
        if np.abs(pred_original - pred_squeezed) > 0.1:
            return True
        
        # Method 3: Prediction consistency
        predictions = []
        for _ in range(5):
            noisy_x = x + np.random.normal(0, 0.01, x.shape)
            predictions.append(self.model.predict(noisy_x))
        
        if np.std(predictions) > 0.1:
            return True
        
        return False
    
    def _squeeze_features(self, x: np.ndarray) -> np.ndarray:
        """Reduce input precision to remove adversarial perturbations."""
        # Reduce bit depth
        squeezed = np.round(x * 16) / 16
        return squeezed
    
    def _is_valid_input(self, x: np.ndarray) -> bool:
        """Validate input is within expected bounds."""
        # Check range
        if x.min() < -3 or x.max() > 3:
            return False
        
        # Check for NaN/Inf
        if np.isnan(x).any() or np.isinf(x).any():
            return False
        
        return True
    
    def adversarial_training(self, train_data, train_labels,
                            epsilon: float = 0.1):
        """
        Train model on adversarial examples for robustness.
        """
        augmented_data = []
        augmented_labels = []
        
        for x, y in zip(train_data, train_labels):
            # Original example
            augmented_data.append(x)
            augmented_labels.append(y)
            
            # Generate adversarial example (FGSM)
            x_adv = self._fgsm_attack(x, y, epsilon)
            augmented_data.append(x_adv)
            augmented_labels.append(y)
        
        # Train on augmented dataset
        self.model.fit(augmented_data, augmented_labels)
    
    def _fgsm_attack(self, x: np.ndarray, y: int, 
                    epsilon: float) -> np.ndarray:
        """Fast Gradient Sign Method attack."""
        x_tensor = torch.tensor(x, requires_grad=True)
        output = self.model(x_tensor)
        loss = criterion(output, torch.tensor([y]))
        loss.backward()
        
        # Perturb in direction of gradient sign
        perturbation = epsilon * x_tensor.grad.sign()
        x_adv = x_tensor + perturbation
        
        return x_adv.detach().numpy()
```

### 3. Data Poisoning Detection

```python
class DataPoisoningDefense:
    """Detect and mitigate data poisoning attacks."""
    
    def __init__(self, clean_baseline_data):
        self.baseline_stats = self._compute_stats(clean_baseline_data)
    
    def detect_poisoning(self, new_data: pd.DataFrame) -> dict:
        """Detect potential data poisoning."""
        results = {
            'is_poisoned': False,
            'suspicious_features': [],
            'suspicious_samples': []
        }
        
        # Check feature distributions
        for column in new_data.columns:
            if column in self.baseline_stats:
                baseline = self.baseline_stats[column]
                current = new_data[column].describe()
                
                # Check for distribution shift
                if self._significant_shift(baseline, current):
                    results['suspicious_features'].append(column)
        
        # Check for outliers
        outliers = self._detect_outliers(new_data)
        results['suspicious_samples'] = outliers
        
        # Label flip detection (if labels available)
        # Check if labels match expected distribution
        
        results['is_poisoned'] = (
            len(results['suspicious_features']) > 0 or
            len(results['suspicious_samples']) > len(new_data) * 0.01
        )
        
        return results
    
    def clean_poisoned_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove suspected poisoned samples."""
        detection = self.detect_poisoning(data)
        
        if not detection['is_poisoned']:
            return data
        
        # Remove outlier samples
        clean_data = data.drop(index=detection['suspicious_samples'])
        
        return clean_data
```

---

## 🔐 Model Protection

### Model Watermarking

```python
class ModelWatermark:
    """
    Embed watermark in model for ownership verification.
    """
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.trigger_samples = self._generate_triggers()
    
    def _generate_triggers(self) -> list:
        """Generate trigger samples from secret key."""
        np.random.seed(hash(self.secret_key) % 2**32)
        triggers = []
        for i in range(10):
            trigger = np.random.randn(input_dim)
            target = i % num_classes  # Predetermined outputs
            triggers.append((trigger, target))
        return triggers
    
    def embed_watermark(self, model, train_data, train_labels):
        """Train model with watermark triggers."""
        # Add trigger samples to training data
        watermark_data = train_data.copy()
        watermark_labels = train_labels.copy()
        
        for trigger, target in self.trigger_samples:
            watermark_data.append(trigger)
            watermark_labels.append(target)
        
        # Train model (will learn to classify triggers)
        model.fit(watermark_data, watermark_labels)
        
        return model
    
    def verify_watermark(self, model) -> bool:
        """Verify model contains watermark."""
        correct = 0
        for trigger, expected in self.trigger_samples:
            prediction = model.predict([trigger])[0]
            if prediction == expected:
                correct += 1
        
        # Watermark verified if >90% triggers correct
        return correct / len(self.trigger_samples) > 0.9
```

### Secure Model Serving

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, validator
import numpy as np

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list
    
    @validator('features')
    def validate_features(cls, v):
        # Input validation
        if len(v) != EXPECTED_FEATURE_COUNT:
            raise ValueError(f'Expected {EXPECTED_FEATURE_COUNT} features')
        
        arr = np.array(v)
        
        # Check for adversarial patterns
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError('Invalid feature values')
        
        if arr.min() < -10 or arr.max() > 10:
            raise ValueError('Feature values out of range')
        
        return v

# Security middleware
extraction_defense = ModelExtractionDefense(
    max_queries_per_user=1000,
    max_queries_per_minute=60
)

adversarial_defense = AdversarialDefense(model)

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    user: User = Depends(get_current_user)
):
    features = np.array(request.features)
    
    # Check extraction attack
    extraction_defense.check_request(user.id, features)
    
    # Check adversarial attack
    if adversarial_defense.detect_adversarial(features):
        raise HTTPException(400, "Suspicious input detected")
    
    # Make prediction
    prediction = model.predict(features)
    
    # Add perturbation to protect model
    prediction = extraction_defense.add_prediction_perturbation(
        prediction, 
        confidence=model.predict_proba(features).max()
    )
    
    return {"prediction": float(prediction)}
```

---

## ✅ Best Practices

1. **Rate limit API access** - prevent extraction attacks
2. **Validate all inputs** - reject suspicious inputs
3. **Monitor usage patterns** - detect attacks early
4. **Adversarial training** - build robust models
5. **Watermark models** - protect IP
6. **Secure model storage** - encrypt model files
7. **Audit model access** - log all interactions

---

## 🔗 Related Topics

- [Data Privacy](./data-privacy.md) - Protect training data
- [Access Control](./access-control.md) - Control model access
- [Compliance](./compliance.md) - Regulatory requirements
