# Access Control

## Overview

Access control ensures only authorized users and systems can access ML data, models, and infrastructure. ML systems require fine-grained access control because different roles need different levels of access—data scientists need training data access, ML engineers need model deployment access, and applications need prediction API access.

---

## 🏗️ Access Control Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Identity Provider (IdP)                       │
│  (Okta, Auth0, Azure AD, AWS IAM)                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway                                   │
│  - Authentication (JWT validation)                               │
│  - Rate limiting                                                 │
│  - Request logging                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Authorization Service                         │
│  - RBAC (Role-Based Access Control)                             │
│  - ABAC (Attribute-Based Access Control)                        │
│  - Policy evaluation                                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Training Data   │ │     Models       │ │  Predictions     │
│  (Data Lake)     │ │  (Registry)      │ │  (API)           │
│  - Read access   │ │  - Deploy access │ │  - Invoke access │
│  - Write access  │ │  - View access   │ │  - Rate limits   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 🎯 Role-Based Access Control (RBAC)

### ML System Roles

| Role | Data Access | Model Access | API Access |
|------|-------------|--------------|------------|
| **Data Scientist** | Read training data | Read models, run experiments | Read predictions |
| **ML Engineer** | Read processed data | Full model access, deploy | Full API access |
| **Data Engineer** | Full data access | No model access | No API access |
| **Application** | No data access | No model access | Invoke predictions |
| **Admin** | Full access | Full access | Full access |

### Implementation

```python
from enum import Enum
from typing import Set, Dict
from functools import wraps

class Permission(Enum):
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    
    # Model permissions
    MODEL_READ = "model:read"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DELETE = "model:delete"
    
    # Prediction permissions
    PREDICTION_INVOKE = "prediction:invoke"
    PREDICTION_BATCH = "prediction:batch"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_AUDIT = "admin:audit"

class Role(Enum):
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    DATA_ENGINEER = "data_engineer"
    APPLICATION = "application"
    ADMIN = "admin"

# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.DATA_SCIENTIST: {
        Permission.DATA_READ,
        Permission.MODEL_READ,
        Permission.MODEL_TRAIN,
        Permission.PREDICTION_INVOKE
    },
    Role.ML_ENGINEER: {
        Permission.DATA_READ,
        Permission.MODEL_READ,
        Permission.MODEL_TRAIN,
        Permission.MODEL_DEPLOY,
        Permission.MODEL_DELETE,
        Permission.PREDICTION_INVOKE,
        Permission.PREDICTION_BATCH
    },
    Role.DATA_ENGINEER: {
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_DELETE
    },
    Role.APPLICATION: {
        Permission.PREDICTION_INVOKE
    },
    Role.ADMIN: set(Permission)  # All permissions
}

class AccessControl:
    """RBAC implementation for ML systems."""
    
    def __init__(self):
        self.user_roles: Dict[str, Set[Role]] = {}
    
    def assign_role(self, user_id: str, role: Role):
        """Assign role to user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission."""
        if user_id not in self.user_roles:
            return False
        
        for role in self.user_roles[user_id]:
            if permission in ROLE_PERMISSIONS[role]:
                return True
        return False
    
    def get_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user."""
        permissions = set()
        for role in self.user_roles.get(user_id, set()):
            permissions.update(ROLE_PERMISSIONS[role])
        return permissions

# Decorator for permission checking
def require_permission(permission: Permission):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User, **kwargs):
            if not access_control.has_permission(user.id, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission.value}"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator

# Usage
access_control = AccessControl()

@app.post("/models/{model_name}/deploy")
@require_permission(Permission.MODEL_DEPLOY)
async def deploy_model(model_name: str, user: User = Depends(get_current_user)):
    """Deploy model - requires MODEL_DEPLOY permission."""
    return await model_service.deploy(model_name)

@app.post("/predict")
@require_permission(Permission.PREDICTION_INVOKE)
async def predict(request: PredictionRequest, user: User = Depends(get_current_user)):
    """Make prediction - requires PREDICTION_INVOKE permission."""
    return await prediction_service.predict(request)
```

---

## 🔐 Authentication Implementation

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key"  # Use env variable
ALGORITHM = "HS256"

class TokenData:
    def __init__(self, user_id: str, roles: list, exp: datetime):
        self.user_id = user_id
        self.roles = roles
        self.exp = exp

def create_access_token(user_id: str, roles: list, 
                       expires_delta: timedelta = timedelta(hours=1)) -> str:
    """Create JWT access token."""
    expire = datetime.utcnow() + expires_delta
    to_encode = {
        "sub": user_id,
        "roles": roles,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials = Security(security)) -> TokenData:
    """Validate JWT and return user."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id = payload.get("sub")
        roles = payload.get("roles", [])
        exp = datetime.fromtimestamp(payload.get("exp"))
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return TokenData(user_id=user_id, roles=roles, exp=exp)
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Key authentication for services
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

class APIKey:
    def __init__(self, key_id: str, permissions: Set[Permission], 
                 rate_limit: int):
        self.key_id = key_id
        self.permissions = permissions
        self.rate_limit = rate_limit

# Store API keys (use database in production)
API_KEYS: Dict[str, APIKey] = {}

async def validate_api_key(api_key: str = Security(api_key_header)) -> APIKey:
    """Validate API key for service-to-service auth."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    if key_hash not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return API_KEYS[key_hash]
```

---

## 📝 Audit Logging

```python
import json
from datetime import datetime
from typing import Any, Dict

class AuditLogger:
    """Log all access for compliance and security."""
    
    def __init__(self, log_destination: str):
        self.destination = log_destination
    
    def log_access(self, 
                   user_id: str,
                   action: str,
                   resource: str,
                   result: str,
                   metadata: Dict[str, Any] = None):
        """Log access event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,  # "allowed" or "denied"
            "metadata": metadata or {},
            "ip_address": get_client_ip(),
            "user_agent": get_user_agent()
        }
        
        # Log to destination (CloudWatch, S3, etc.)
        self._write_log(event)
    
    def log_data_access(self, user_id: str, dataset: str, 
                       operation: str, row_count: int):
        """Log data access for compliance."""
        self.log_access(
            user_id=user_id,
            action=f"data:{operation}",
            resource=dataset,
            result="allowed",
            metadata={"row_count": row_count}
        )
    
    def log_model_access(self, user_id: str, model_name: str,
                        operation: str, version: str = None):
        """Log model access."""
        self.log_access(
            user_id=user_id,
            action=f"model:{operation}",
            resource=model_name,
            result="allowed",
            metadata={"version": version}
        )
    
    def log_prediction(self, user_id: str, model_name: str,
                      request_id: str):
        """Log prediction request."""
        self.log_access(
            user_id=user_id,
            action="prediction:invoke",
            resource=model_name,
            result="allowed",
            metadata={"request_id": request_id}
        )

# Usage with middleware
audit_logger = AuditLogger(log_destination="cloudwatch")

@app.middleware("http")
async def audit_middleware(request, call_next):
    """Log all API requests."""
    start_time = time.time()
    response = await call_next(request)
    
    # Log request
    audit_logger.log_access(
        user_id=request.state.user_id if hasattr(request.state, 'user_id') else "anonymous",
        action=f"{request.method}:{request.url.path}",
        resource=request.url.path,
        result="allowed" if response.status_code < 400 else "denied",
        metadata={
            "status_code": response.status_code,
            "duration_ms": (time.time() - start_time) * 1000
        }
    )
    
    return response
```

---

## ✅ Best Practices

1. **Least privilege** - grant minimum required access
2. **Separation of duties** - split critical operations
3. **Regular access reviews** - audit permissions quarterly
4. **MFA for humans** - require multi-factor authentication
5. **API keys for services** - separate from user auth
6. **Audit everything** - log all access
7. **Rotate credentials** - regular key rotation

---

## 🔗 Related Topics

- [Data Privacy](./01-data-privacy.md) - Protect data
- [Model Security](./02-model-security.md) - Protect models
- [Compliance](./04-compliance.md) - Meet regulations
