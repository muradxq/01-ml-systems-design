# Fraud Detection Systems

## Overview

Fraud detection systems identify fraudulent transactions, accounts, or activities in real-time. These systems are critical for financial institutions, e-commerce platforms, and payment processors. The challenge is detecting fraud with high accuracy while minimizing false positives that impact legitimate users, all within strict latency constraints.

---

## 🎯 Problem Definition

### Business Goals
- Minimize fraud losses
- Minimize customer friction (false positives)
- Meet regulatory compliance requirements
- Enable fast, automated decisions
- Adapt to evolving fraud patterns

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Latency** | < 100ms p99 (real-time decisions) |
| **Throughput** | 10K-100K transactions/second |
| **Accuracy** | > 95% precision, > 90% recall |
| **Availability** | 99.99% uptime |
| **Adaptability** | Detect new fraud patterns within hours |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Fraud Detection Architecture                          │
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Transaction   │                                                          │
│  │  Event         │                                                          │
│  └───────┬────────┘                                                          │
│          │                                                                    │
│          ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Real-Time Processing                           │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │   Validate   │──│   Enrich     │──│   Score      │           │        │
│  │  │   Event      │  │   Features   │  │   (ML + Rules│           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│          ┌───────────────────────┼───────────────────────┐                  │
│          ▼                       ▼                       ▼                  │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐            │
│  │   APPROVE    │       │   REVIEW     │       │   DECLINE    │            │
│  │   (Low Risk) │       │   (Medium)   │       │   (High Risk)│            │
│  └──────────────┘       └──────────────┘       └──────────────┘            │
│                                  │                                           │
│                                  ▼                                           │
│                         ┌──────────────┐                                    │
│                         │  Manual      │                                    │
│                         │  Review Queue│                                    │
│                         └──────────────┘                                    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Feedback Loop                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │  Label       │──│  Retrain     │──│  Deploy      │           │        │
│  │  │  Collection  │  │  Models      │  │  Updated     │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Feature Engineering

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

@dataclass
class Transaction:
    """Transaction event."""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    card_id: str
    device_id: str
    ip_address: str
    location: tuple  # (lat, lon)
    merchant_category: str

class FraudFeatureEngine:
    """Feature engineering for fraud detection."""
    
    def __init__(self, feature_store, velocity_calculator, device_fingerprinter):
        self.feature_store = feature_store
        self.velocity = velocity_calculator
        self.device_fp = device_fingerprinter
    
    async def compute_features(self, txn: Transaction) -> Dict[str, float]:
        """Compute all features for a transaction."""
        
        # Get pre-computed features from feature store
        user_features = await self.feature_store.get_user_features(txn.user_id)
        merchant_features = await self.feature_store.get_merchant_features(txn.merchant_id)
        card_features = await self.feature_store.get_card_features(txn.card_id)
        
        # Compute real-time velocity features
        velocity_features = await self._compute_velocity_features(txn)
        
        # Compute device/session features
        device_features = await self._compute_device_features(txn)
        
        # Compute transaction-specific features
        txn_features = self._compute_transaction_features(txn, user_features)
        
        # Combine all features
        features = {
            **user_features,
            **merchant_features,
            **card_features,
            **velocity_features,
            **device_features,
            **txn_features
        }
        
        return features
    
    async def _compute_velocity_features(self, txn: Transaction) -> Dict[str, float]:
        """Compute real-time velocity features."""
        
        # Transaction count in time windows
        txn_count_1h = await self.velocity.count_transactions(
            user_id=txn.user_id,
            window=timedelta(hours=1)
        )
        txn_count_24h = await self.velocity.count_transactions(
            user_id=txn.user_id,
            window=timedelta(hours=24)
        )
        
        # Amount sum in time windows
        amount_sum_1h = await self.velocity.sum_amount(
            user_id=txn.user_id,
            window=timedelta(hours=1)
        )
        amount_sum_24h = await self.velocity.sum_amount(
            user_id=txn.user_id,
            window=timedelta(hours=24)
        )
        
        # Unique merchants/devices
        unique_merchants_24h = await self.velocity.count_unique(
            user_id=txn.user_id,
            field="merchant_id",
            window=timedelta(hours=24)
        )
        unique_devices_24h = await self.velocity.count_unique(
            user_id=txn.user_id,
            field="device_id",
            window=timedelta(hours=24)
        )
        
        # Time since last transaction
        last_txn_time = await self.velocity.get_last_transaction_time(txn.user_id)
        time_since_last = (txn.timestamp - last_txn_time).total_seconds() if last_txn_time else -1
        
        return {
            "txn_count_1h": txn_count_1h,
            "txn_count_24h": txn_count_24h,
            "amount_sum_1h": amount_sum_1h,
            "amount_sum_24h": amount_sum_24h,
            "unique_merchants_24h": unique_merchants_24h,
            "unique_devices_24h": unique_devices_24h,
            "time_since_last_txn": time_since_last,
            "is_first_txn": int(last_txn_time is None)
        }
    
    async def _compute_device_features(self, txn: Transaction) -> Dict[str, float]:
        """Compute device and session features."""
        
        # Device fingerprint analysis
        device_risk = await self.device_fp.get_risk_score(txn.device_id)
        
        # IP analysis
        ip_info = await self._analyze_ip(txn.ip_address)
        
        # Device history
        device_history = await self.feature_store.get_device_history(txn.device_id)
        
        return {
            "device_risk_score": device_risk,
            "is_vpn": int(ip_info.get("is_vpn", False)),
            "is_proxy": int(ip_info.get("is_proxy", False)),
            "is_tor": int(ip_info.get("is_tor", False)),
            "ip_country_match": int(ip_info.get("country") == device_history.get("usual_country")),
            "device_age_days": device_history.get("age_days", 0),
            "device_txn_count": device_history.get("txn_count", 0),
            "device_fraud_rate": device_history.get("fraud_rate", 0)
        }
    
    def _compute_transaction_features(
        self,
        txn: Transaction,
        user_features: Dict
    ) -> Dict[str, float]:
        """Compute transaction-specific features."""
        
        avg_amount = user_features.get("avg_transaction_amount", txn.amount)
        
        return {
            "amount": txn.amount,
            "amount_log": np.log1p(txn.amount),
            "amount_zscore": (txn.amount - avg_amount) / max(user_features.get("std_transaction_amount", 1), 1),
            "is_round_amount": int(txn.amount % 100 == 0),
            "hour_of_day": txn.timestamp.hour,
            "day_of_week": txn.timestamp.weekday(),
            "is_weekend": int(txn.timestamp.weekday() >= 5),
            "is_night": int(txn.timestamp.hour < 6 or txn.timestamp.hour > 22)
        }

class VelocityCalculator:
    """Calculate real-time velocity features using Redis."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def count_transactions(
        self,
        user_id: str,
        window: timedelta
    ) -> int:
        """Count transactions in time window using sorted set."""
        key = f"velocity:txn_count:{user_id}"
        now = datetime.utcnow().timestamp()
        window_start = now - window.total_seconds()
        
        # Count items in time range
        count = await self.redis.zcount(key, window_start, now)
        return count
    
    async def sum_amount(
        self,
        user_id: str,
        window: timedelta
    ) -> float:
        """Sum transaction amounts in time window."""
        key = f"velocity:amounts:{user_id}"
        now = datetime.utcnow().timestamp()
        window_start = now - window.total_seconds()
        
        # Get amounts in range
        amounts = await self.redis.zrangebyscore(
            key, window_start, now, withscores=True
        )
        return sum(float(amount) for _, amount in amounts)
    
    async def record_transaction(self, txn: Transaction):
        """Record transaction for velocity tracking."""
        now = txn.timestamp.timestamp()
        
        # Transaction count
        await self.redis.zadd(
            f"velocity:txn_count:{txn.user_id}",
            {txn.transaction_id: now}
        )
        
        # Amount tracking
        await self.redis.zadd(
            f"velocity:amounts:{txn.user_id}",
            {txn.transaction_id: txn.amount}
        )
        
        # Unique tracking
        await self.redis.zadd(
            f"velocity:merchants:{txn.user_id}",
            {txn.merchant_id: now}
        )
        
        # Expire old data
        for key_prefix in ["txn_count", "amounts", "merchants"]:
            key = f"velocity:{key_prefix}:{txn.user_id}"
            await self.redis.zremrangebyscore(key, 0, now - 86400 * 7)
```

### 2. ML Model + Rules Engine

```python
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class Decision(Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"

@dataclass
class FraudScore:
    """Fraud scoring result."""
    ml_score: float
    rule_flags: List[str]
    final_score: float
    decision: Decision
    explanation: Dict

class FraudScoringEngine:
    """Combined ML + Rules fraud scoring."""
    
    def __init__(
        self,
        ml_model,
        rules_engine,
        thresholds: Dict[str, float]
    ):
        self.ml_model = ml_model
        self.rules = rules_engine
        self.thresholds = thresholds  # approve, review, decline thresholds
    
    def score(self, features: Dict[str, float], txn: Dict) -> FraudScore:
        """Score transaction for fraud."""
        
        # ML scoring
        ml_score = self._ml_score(features)
        
        # Rules evaluation
        rule_flags = self.rules.evaluate(features, txn)
        rule_score = self._compute_rule_score(rule_flags)
        
        # Combine scores
        final_score = self._combine_scores(ml_score, rule_score, rule_flags)
        
        # Make decision
        decision = self._make_decision(final_score, rule_flags)
        
        # Generate explanation
        explanation = self._generate_explanation(features, ml_score, rule_flags)
        
        return FraudScore(
            ml_score=ml_score,
            rule_flags=rule_flags,
            final_score=final_score,
            decision=decision,
            explanation=explanation
        )
    
    def _ml_score(self, features: Dict) -> float:
        """Get ML model fraud probability."""
        feature_vector = self._features_to_vector(features)
        probability = self.ml_model.predict_proba(feature_vector)[0][1]
        return probability
    
    def _compute_rule_score(self, flags: List[str]) -> float:
        """Compute score from rule flags."""
        if not flags:
            return 0.0
        
        # Weight critical rules higher
        weights = {
            "BLACKLISTED_DEVICE": 1.0,
            "STOLEN_CARD": 1.0,
            "VELOCITY_BREACH": 0.5,
            "HIGH_RISK_COUNTRY": 0.3,
            "UNUSUAL_AMOUNT": 0.2
        }
        
        total_weight = sum(weights.get(flag, 0.1) for flag in flags)
        return min(total_weight, 1.0)
    
    def _combine_scores(
        self,
        ml_score: float,
        rule_score: float,
        flags: List[str]
    ) -> float:
        """Combine ML and rule scores."""
        
        # Hard flags override ML
        hard_flags = {"BLACKLISTED_DEVICE", "STOLEN_CARD"}
        if any(flag in hard_flags for flag in flags):
            return 1.0
        
        # Weighted combination
        return 0.7 * ml_score + 0.3 * rule_score
    
    def _make_decision(
        self,
        score: float,
        flags: List[str]
    ) -> Decision:
        """Make approve/review/decline decision."""
        
        # Hard declines
        if any(flag in ["BLACKLISTED_DEVICE", "STOLEN_CARD"] for flag in flags):
            return Decision.DECLINE
        
        # Score-based decision
        if score >= self.thresholds["decline"]:
            return Decision.DECLINE
        elif score >= self.thresholds["review"]:
            return Decision.REVIEW
        else:
            return Decision.APPROVE
    
    def _generate_explanation(
        self,
        features: Dict,
        ml_score: float,
        flags: List[str]
    ) -> Dict:
        """Generate human-readable explanation."""
        
        reasons = []
        
        # Rule-based reasons
        for flag in flags:
            reasons.append(self._flag_to_reason(flag))
        
        # ML-based reasons (SHAP or similar)
        if ml_score > 0.5:
            ml_reasons = self._explain_ml_score(features)
            reasons.extend(ml_reasons[:3])
        
        return {
            "reasons": reasons,
            "risk_factors": self._get_risk_factors(features),
            "ml_contribution": ml_score,
            "rule_contribution": len(flags) > 0
        }

class RulesEngine:
    """Rule-based fraud detection."""
    
    def __init__(self, rules_config: Dict):
        self.rules = self._load_rules(rules_config)
    
    def evaluate(self, features: Dict, txn: Dict) -> List[str]:
        """Evaluate all rules and return flags."""
        flags = []
        
        for rule_name, rule_func in self.rules.items():
            if rule_func(features, txn):
                flags.append(rule_name)
        
        return flags
    
    def _load_rules(self, config: Dict) -> Dict:
        """Load rules from configuration."""
        rules = {}
        
        # Velocity rules
        rules["VELOCITY_BREACH"] = lambda f, t: (
            f.get("txn_count_1h", 0) > config.get("max_txn_per_hour", 10) or
            f.get("amount_sum_24h", 0) > config.get("max_amount_per_day", 10000)
        )
        
        # Device rules
        rules["BLACKLISTED_DEVICE"] = lambda f, t: (
            f.get("device_risk_score", 0) > 0.95
        )
        
        # Location rules
        rules["HIGH_RISK_COUNTRY"] = lambda f, t: (
            not f.get("ip_country_match", True) and
            f.get("is_vpn", False)
        )
        
        # Amount rules
        rules["UNUSUAL_AMOUNT"] = lambda f, t: (
            f.get("amount_zscore", 0) > 3.0
        )
        
        # Time rules
        rules["UNUSUAL_TIME"] = lambda f, t: (
            f.get("is_night", False) and
            f.get("txn_count_1h", 0) > 3
        )
        
        # First transaction rules
        rules["RISKY_FIRST_TXN"] = lambda f, t: (
            f.get("is_first_txn", False) and
            f.get("amount", 0) > config.get("first_txn_limit", 500)
        )
        
        return rules
```

### 3. Real-Time Scoring Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import asyncio

app = FastAPI()

class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    card_id: str
    device_id: str
    ip_address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    merchant_category: str

class FraudResponse(BaseModel):
    transaction_id: str
    decision: str
    score: float
    flags: List[str]
    latency_ms: float

class FraudDetectionService:
    """Real-time fraud detection service."""
    
    def __init__(
        self,
        feature_engine: FraudFeatureEngine,
        scoring_engine: FraudScoringEngine,
        velocity_calc: VelocityCalculator,
        metrics_client
    ):
        self.features = feature_engine
        self.scoring = scoring_engine
        self.velocity = velocity_calc
        self.metrics = metrics_client
    
    async def evaluate(self, request: TransactionRequest) -> FraudResponse:
        """Evaluate transaction for fraud."""
        
        start_time = time.time()
        
        try:
            # Convert to transaction object
            txn = self._to_transaction(request)
            
            # Compute features
            features = await asyncio.wait_for(
                self.features.compute_features(txn),
                timeout=0.050  # 50ms timeout for features
            )
            
            # Score transaction
            score_result = self.scoring.score(features, request.dict())
            
            # Record velocity (async, don't wait)
            asyncio.create_task(self.velocity.record_transaction(txn))
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Log metrics
            self.metrics.record_decision(
                decision=score_result.decision.value,
                score=score_result.final_score,
                latency_ms=latency_ms
            )
            
            return FraudResponse(
                transaction_id=request.transaction_id,
                decision=score_result.decision.value,
                score=score_result.final_score,
                flags=score_result.rule_flags,
                latency_ms=latency_ms
            )
        
        except asyncio.TimeoutError:
            # Fallback to rules-only on timeout
            return await self._fallback_evaluation(request)
        
        except Exception as e:
            self.metrics.record_error(str(e))
            # Safe default: approve but flag for review
            return FraudResponse(
                transaction_id=request.transaction_id,
                decision="review",
                score=0.5,
                flags=["EVALUATION_ERROR"],
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _fallback_evaluation(
        self,
        request: TransactionRequest
    ) -> FraudResponse:
        """Fallback evaluation when ML times out."""
        
        # Simple rules-only evaluation
        flags = []
        
        if request.amount > 5000:
            flags.append("HIGH_AMOUNT")
        
        decision = "review" if flags else "approve"
        
        return FraudResponse(
            transaction_id=request.transaction_id,
            decision=decision,
            score=0.5 if flags else 0.1,
            flags=flags + ["FALLBACK_MODE"],
            latency_ms=0
        )

# API Endpoint
service = FraudDetectionService(...)

@app.post("/evaluate", response_model=FraudResponse)
async def evaluate_transaction(request: TransactionRequest):
    return await service.evaluate(request)

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 4. Model Retraining Pipeline

```python
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import mlflow

class FraudModelTrainer:
    """Automated fraud model retraining."""
    
    def __init__(self, data_warehouse, model_registry, feature_store):
        self.dw = data_warehouse
        self.registry = model_registry
        self.features = feature_store
    
    def train_model(self, lookback_days: int = 90) -> str:
        """Train new fraud detection model."""
        
        with mlflow.start_run():
            # 1. Gather labeled data
            data = self._get_training_data(lookback_days)
            
            # 2. Feature engineering
            X, y = self._prepare_features(data)
            
            # 3. Handle class imbalance
            X_balanced, y_balanced = self._balance_data(X, y)
            
            # 4. Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, stratify=y_balanced
            )
            
            # 5. Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8
            )
            model.fit(X_train, y_train)
            
            # 6. Evaluate
            metrics = self._evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            
            # 7. Check if better than production
            if self._is_better_than_production(metrics):
                # Register new model
                model_version = self.registry.register(model, metrics)
                return model_version
            
            return None
    
    def _get_training_data(self, lookback_days: int) -> pd.DataFrame:
        """Get labeled transaction data."""
        
        query = f"""
        SELECT 
            t.*,
            COALESCE(f.is_fraud, 0) as is_fraud
        FROM transactions t
        LEFT JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        WHERE t.timestamp > CURRENT_DATE - INTERVAL '{lookback_days} days'
        """
        
        return self.dw.query(query)
    
    def _balance_data(self, X, y):
        """Balance dataset using SMOTE or undersampling."""
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        return X_balanced, y_balanced
    
    def _evaluate_model(self, model, X_test, y_test) -> Dict:
        """Evaluate model performance."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob)
        }
```

---

## 📈 Metrics & Monitoring

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision** | Fraud predictions that are correct | > 80% |
| **Recall** | Actual frauds detected | > 90% |
| **False Positive Rate** | Legitimate txns declined | < 1% |
| **Decision Latency** | Time to decision | < 100ms |
| **Loss Rate** | $ fraud / $ total | < 0.1% |

### Monitoring Dashboard

```python
from prometheus_client import Counter, Histogram, Gauge

# Decision metrics
decisions_total = Counter(
    'fraud_decisions_total',
    'Total fraud decisions',
    ['decision']
)

# Score distribution
score_histogram = Histogram(
    'fraud_score',
    'Fraud score distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Latency
latency_histogram = Histogram(
    'fraud_decision_latency_ms',
    'Decision latency in milliseconds',
    buckets=[10, 25, 50, 75, 100, 150, 200, 500]
)

# Model performance (updated by batch job)
precision_gauge = Gauge('fraud_model_precision', 'Model precision')
recall_gauge = Gauge('fraud_model_recall', 'Model recall')
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Precision vs Recall** | High precision (less friction) | High recall (catch more fraud) |
| **Rules vs ML** | Rules (interpretable, fast) | ML (adaptive, accurate) |
| **Real-time vs Batch** | Real-time (immediate) | Batch + review (thorough) |
| **Speed vs Features** | Fewer features (fast) | More features (accurate) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you handle class imbalance?
2. How do you detect new fraud patterns?
3. How do you minimize false positives?
4. How would you ensure < 100ms latency?
5. How do you handle adversarial adaptation?

**Key Points:**
- Combine ML with rules
- Real-time velocity features are critical
- Feedback loops for model improvement
- Explainability for compliance
- Graceful degradation on failures

---

## 🔗 Related Topics

- [Feature Engineering](../03-feature-engineering/README.md)
- [Real-time Serving](../05-model-serving/serving-patterns.md)
- [Data Drift Detection](../06-monitoring-observability/data-drift-detection.md)
