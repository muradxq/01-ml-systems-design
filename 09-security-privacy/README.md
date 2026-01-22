# 🔒 Security & Privacy

## Overview

Security and privacy are critical for ML systems handling sensitive data. ML systems face unique security challenges including adversarial attacks on models, data poisoning, model theft, and privacy leakage through predictions. Proper security protects data, models, users, and your organization from breaches and regulatory penalties.

---

## 🎯 Learning Objectives

After completing this section, you should understand:
- Data privacy techniques and regulations
- Model security threats and defenses
- Access control patterns for ML systems
- Compliance requirements (GDPR, CCPA, HIPAA)

---

## 📚 Topics Covered

1. [Data Privacy](./data-privacy.md)
   - Anonymization and pseudonymization
   - Differential privacy
   - Federated learning

2. [Model Security](./model-security.md)
   - Adversarial attacks and defenses
   - Model theft prevention
   - Data poisoning detection

3. [Access Control](./access-control.md)
   - Authentication and authorization
   - Role-based access control (RBAC)
   - API security

4. [Compliance](./compliance.md)
   - GDPR requirements
   - CCPA compliance
   - Industry-specific regulations

---

## 🏗️ Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Perimeter                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    WAF / DDoS Protection                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              API Gateway (Authentication)                │   │
│  │              - API Keys / OAuth 2.0                      │   │
│  │              - Rate Limiting                             │   │
│  │              - Input Validation                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │   Model      │   │   Feature    │   │   Data       │       │
│  │   Service    │   │   Store      │   │   Pipeline   │       │
│  │   (RBAC)     │   │   (Encrypted)│   │   (Encrypted)│       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│          │                   │                   │              │
│          └───────────────────┼───────────────────┘              │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Encrypted Storage                           │   │
│  │              - Data at Rest Encryption                   │   │
│  │              - Key Management (KMS)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Audit Logging & Monitoring                  │   │
│  │              - Access Logs                               │   │
│  │              - Security Events                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚨 ML-Specific Security Threats

### 1. Adversarial Attacks

| Attack Type | Description | Defense |
|-------------|-------------|---------|
| **Evasion attacks** | Crafted inputs that cause misclassification | Adversarial training, input validation |
| **Model extraction** | Querying to steal model | Rate limiting, query monitoring |
| **Model inversion** | Recovering training data from predictions | Differential privacy, output perturbation |
| **Membership inference** | Determine if data was in training set | Differential privacy, regularization |

### 2. Data Attacks

| Attack Type | Description | Defense |
|-------------|-------------|---------|
| **Data poisoning** | Injecting malicious training data | Data validation, anomaly detection |
| **Label flipping** | Corrupting labels in training data | Label verification, consensus labeling |
| **Backdoor attacks** | Hidden triggers in training data | Data inspection, clean training |

### 3. Infrastructure Attacks

| Attack Type | Description | Defense |
|-------------|-------------|---------|
| **API abuse** | Excessive queries, DDoS | Rate limiting, WAF |
| **Data exfiltration** | Stealing sensitive data | Encryption, access controls |
| **Privilege escalation** | Gaining unauthorized access | RBAC, least privilege |

---

## 🔧 Security Implementation

### 1. API Authentication

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from datetime import datetime, timedelta

app = FastAPI()

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key."""
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# JWT Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=1))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Verify JWT and get user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return await get_user(user_id)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Protected endpoint
@app.post("/predict")
async def predict(
    features: dict,
    api_key: str = Depends(verify_api_key),
    user: dict = Depends(get_current_user)
):
    # Check user has permission
    if "predict" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    return model.predict(features)
```

### 2. Data Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, key: bytes = None):
        if key is None:
            key = Fernet.generate_key()
        self.fernet = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

# Encrypt sensitive features
encryptor = DataEncryption()

def preprocess_with_encryption(user_data: dict) -> dict:
    """Encrypt sensitive fields before processing."""
    sensitive_fields = ['ssn', 'email', 'phone']
    
    processed = user_data.copy()
    for field in sensitive_fields:
        if field in processed:
            processed[field] = encryptor.encrypt(processed[field])
    
    return processed
```

### 3. Differential Privacy

```python
import numpy as np

class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
    
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def privatize_prediction(self, prediction: np.ndarray, 
                            sensitivity: float = 1.0) -> np.ndarray:
        """Add noise to prediction for privacy."""
        noise = np.random.laplace(0, sensitivity / self.epsilon, 
                                  prediction.shape)
        return prediction + noise

# Usage
dp = DifferentialPrivacy(epsilon=1.0)

def private_predict(features: dict) -> dict:
    """Make prediction with differential privacy."""
    raw_prediction = model.predict(features)
    private_prediction = dp.privatize_prediction(
        raw_prediction, 
        sensitivity=0.1
    )
    return {"prediction": private_prediction.tolist()}
```

### 4. Input Validation

```python
from pydantic import BaseModel, validator, Field
from typing import List
import numpy as np

class PredictionInput(BaseModel):
    """Validated input for predictions."""
    
    features: List[float] = Field(..., min_items=10, max_items=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    
    @validator('features')
    def validate_features(cls, v):
        # Check for NaN or infinite values
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contain invalid values")
        
        # Check for reasonable ranges
        if any(abs(x) > 1e6 for x in v):
            raise ValueError("Feature values out of range")
        
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        # Prevent injection attacks
        if any(c in v for c in ['<', '>', '"', "'", ';', '--']):
            raise ValueError("Invalid characters in user_id")
        return v

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Prediction endpoint with validation."""
    return model.predict(input_data.features)
```

---

## 📋 Compliance Requirements

### GDPR (EU)

| Requirement | ML System Implication |
|-------------|----------------------|
| **Right to be forgotten** | Delete user data from training sets |
| **Data portability** | Export user data in standard format |
| **Consent** | Clear consent for data collection |
| **Minimization** | Collect only necessary data |
| **Explainability** | Explain automated decisions |

### CCPA (California)

| Requirement | ML System Implication |
|-------------|----------------------|
| **Right to know** | Disclose data collection practices |
| **Right to delete** | Delete user data on request |
| **Right to opt-out** | Allow opt-out of data sale |
| **Non-discrimination** | Equal service regardless of opt-out |

### HIPAA (Healthcare)

| Requirement | ML System Implication |
|-------------|----------------------|
| **PHI protection** | Encrypt all health data |
| **Access controls** | Strict RBAC for health data |
| **Audit trails** | Log all data access |
| **Business associates** | Agreements with ML vendors |

---

## 🔑 Key Principles

1. **Encrypt everywhere** - data at rest and in transit
2. **Least privilege** - minimum access needed
3. **Defense in depth** - multiple security layers
4. **Audit everything** - comprehensive logging
5. **Assume breach** - design for detection and response
6. **Privacy by design** - build privacy into architecture
7. **Regular testing** - security audits and penetration testing

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Hardcoded secrets** | Credentials in code | Use secrets management |
| **Overpermissive access** | Everyone has admin | RBAC with least privilege |
| **No input validation** | Injection attacks | Validate all inputs |
| **Unencrypted data** | Data breaches | Encrypt at rest and in transit |
| **Missing audit logs** | Can't detect breaches | Comprehensive logging |
| **Training data exposure** | Model inversion | Differential privacy |

---

## 📋 Security Checklist

- [ ] API authentication implemented (OAuth 2.0, API keys)
- [ ] Authorization (RBAC) configured
- [ ] Data encrypted at rest
- [ ] Data encrypted in transit (TLS)
- [ ] Input validation on all endpoints
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Secrets in secure storage (KMS, Vault)
- [ ] Security scanning in CI/CD
- [ ] Regular security audits
- [ ] Incident response plan documented
- [ ] Compliance requirements mapped

---

## 🚀 Next Steps

- Learn about [Data Privacy](./data-privacy.md) - protect user data
- Understand [Model Security](./model-security.md) - defend against ML attacks
- Explore [Access Control](./access-control.md) - implement proper authorization
- Study [Compliance](./compliance.md) - meet regulatory requirements

Then proceed to [End-to-End Systems](../10-end-to-end-systems/README.md) to see how all components work together.
