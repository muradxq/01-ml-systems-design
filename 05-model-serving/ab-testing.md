# A/B Testing for ML Models

## Overview

A/B testing (also called split testing or online experimentation) compares different model versions in production to measure their impact on real users. Unlike offline evaluation, A/B tests capture the true effect of models on business metrics. It's the gold standard for validating ML improvements before full rollout.

---

## 🎯 Why A/B Test Models?

### Offline vs Online Metrics

| Metric Type | Offline | Online (A/B Test) |
|-------------|---------|-------------------|
| **Measures** | Model accuracy | Business impact |
| **Data** | Historical | Real-time user behavior |
| **Feedback loops** | Ignored | Captured |
| **User experience** | Not measured | Directly measured |
| **Confidence** | Limited | High (statistical) |

### When to A/B Test

- New model version with significant changes
- Different model architectures
- Feature additions/removals
- Threshold changes
- Algorithm changes

---

## 🏗️ A/B Testing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  A/B Testing Architecture                                        │
│                                                                  │
│  User Request ───▶ Experiment Assignment ───▶ Model Selection   │
│                           │                         │            │
│                           ▼                         ▼            │
│                    Assignment Store          Model A or B        │
│                    (User → Variant)               │              │
│                                                   ▼              │
│                                              Prediction          │
│                                                   │              │
│                                                   ▼              │
│                                          Event Logging           │
│                                          (Predictions,           │
│                                           Outcomes)              │
│                                                   │              │
│                                                   ▼              │
│                                        Statistical Analysis      │
│                                        (Significance Testing)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Implementation

### 1. Experiment Configuration

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import hashlib
import json

class VariantType(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

@dataclass
class ExperimentVariant:
    """A variant in an experiment."""
    name: str
    model_version: str
    traffic_percentage: float  # 0-100
    parameters: Dict = None

@dataclass
class Experiment:
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    
    # Targeting
    target_audience: Optional[str] = None  # e.g., "new_users", "premium"
    
    # Metrics
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = None
    guardrail_metrics: List[str] = None  # Metrics that should not degrade
    
    # Statistical settings
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    mde: float = 0.02  # Minimum detectable effect
    
    # Lifecycle
    status: str = "draft"  # draft, running, paused, completed
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def validate(self):
        """Validate experiment configuration."""
        # Check traffic percentages sum to 100
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")
        
        # Check at least 2 variants
        if len(self.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")

# Example experiment
experiment = Experiment(
    experiment_id="exp_20240115_model_v2",
    name="Fraud Detection Model V2",
    description="Test new transformer-based fraud detection model",
    variants=[
        ExperimentVariant(
            name="control",
            model_version="v1.0.0",
            traffic_percentage=50
        ),
        ExperimentVariant(
            name="treatment",
            model_version="v2.0.0",
            traffic_percentage=50
        )
    ],
    primary_metric="fraud_detection_rate",
    secondary_metrics=["precision", "false_positive_rate"],
    guardrail_metrics=["user_friction_rate", "latency_p99"],
    min_sample_size=10000,
    mde=0.05
)
```

### 2. Assignment Service

```python
import hashlib
from typing import Optional
import redis
import json

class ExperimentAssignmentService:
    """Assigns users to experiment variants deterministically."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.experiments_cache = {}
    
    def get_assignment(
        self,
        user_id: str,
        experiment_id: str,
        context: Dict = None
    ) -> Optional[ExperimentVariant]:
        """Get variant assignment for a user."""
        
        # Load experiment
        experiment = self._get_experiment(experiment_id)
        if not experiment or experiment.status != "running":
            return None
        
        # Check targeting
        if not self._matches_targeting(experiment, context):
            return None
        
        # Check for existing assignment (sticky)
        existing = self._get_existing_assignment(user_id, experiment_id)
        if existing:
            return existing
        
        # Deterministic assignment using hash
        variant = self._assign_variant(user_id, experiment)
        
        # Store assignment
        self._store_assignment(user_id, experiment_id, variant)
        
        return variant
    
    def _assign_variant(
        self,
        user_id: str,
        experiment: Experiment
    ) -> ExperimentVariant:
        """Deterministically assign user to variant using hash."""
        
        # Create hash for consistent assignment
        hash_input = f"{experiment.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100  # 0-99
        
        # Assign based on traffic percentages
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                return variant
        
        # Fallback to last variant
        return experiment.variants[-1]
    
    def _get_existing_assignment(
        self,
        user_id: str,
        experiment_id: str
    ) -> Optional[ExperimentVariant]:
        """Get existing assignment for sticky bucketing."""
        key = f"assignment:{experiment_id}:{user_id}"
        data = self.redis.get(key)
        
        if data:
            variant_name = json.loads(data)["variant"]
            experiment = self._get_experiment(experiment_id)
            for variant in experiment.variants:
                if variant.name == variant_name:
                    return variant
        
        return None
    
    def _store_assignment(
        self,
        user_id: str,
        experiment_id: str,
        variant: ExperimentVariant
    ):
        """Store assignment for sticky bucketing."""
        key = f"assignment:{experiment_id}:{user_id}"
        self.redis.setex(
            key,
            60 * 60 * 24 * 30,  # 30 days TTL
            json.dumps({"variant": variant.name})
        )
    
    def _matches_targeting(
        self,
        experiment: Experiment,
        context: Dict
    ) -> bool:
        """Check if user matches experiment targeting."""
        if not experiment.target_audience:
            return True
        
        # Example targeting logic
        if experiment.target_audience == "new_users":
            return context.get("days_since_signup", 0) < 7
        elif experiment.target_audience == "premium":
            return context.get("subscription_tier") == "premium"
        
        return True
    
    def _get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment configuration."""
        # Cache lookup
        if experiment_id in self.experiments_cache:
            return self.experiments_cache[experiment_id]
        
        # Load from storage
        data = self.redis.get(f"experiment:{experiment_id}")
        if data:
            experiment = Experiment(**json.loads(data))
            self.experiments_cache[experiment_id] = experiment
            return experiment
        
        return None
```

### 3. Model Router

```python
from fastapi import FastAPI, Depends, Request
from typing import Dict, Any
import time

app = FastAPI()

class ModelRouter:
    """Routes predictions to appropriate model based on experiment."""
    
    def __init__(
        self,
        assignment_service: ExperimentAssignmentService,
        model_registry: Dict[str, Any]  # version -> model
    ):
        self.assignment_service = assignment_service
        self.models = model_registry
    
    async def predict(
        self,
        user_id: str,
        features: Dict,
        experiment_id: str,
        context: Dict = None
    ) -> Dict:
        """Make prediction using assigned model variant."""
        
        start_time = time.time()
        
        # Get assignment
        variant = self.assignment_service.get_assignment(
            user_id=user_id,
            experiment_id=experiment_id,
            context=context
        )
        
        # Default to control if no experiment
        if variant is None:
            model_version = "v1.0.0"  # Default
            variant_name = "default"
        else:
            model_version = variant.model_version
            variant_name = variant.name
        
        # Get model
        model = self.models.get(model_version)
        if model is None:
            raise ValueError(f"Model version {model_version} not found")
        
        # Make prediction
        prediction = model.predict(features)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log for analysis
        result = {
            "prediction": prediction,
            "model_version": model_version,
            "experiment_id": experiment_id,
            "variant": variant_name,
            "latency_ms": latency_ms,
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        # Async log to event store
        await self._log_prediction(result)
        
        return result
    
    async def _log_prediction(self, result: Dict):
        """Log prediction for experiment analysis."""
        # Send to Kafka, BigQuery, etc.
        pass

# API endpoint
router = ModelRouter(
    assignment_service=ExperimentAssignmentService("redis://localhost"),
    model_registry={
        "v1.0.0": load_model("v1.0.0"),
        "v2.0.0": load_model("v2.0.0")
    }
)

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    return await router.predict(
        user_id=body["user_id"],
        features=body["features"],
        experiment_id="exp_20240115_model_v2"
    )
```

### 4. Event Logging

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict
import json
from kafka import KafkaProducer

@dataclass
class ExperimentEvent:
    """Event for experiment analysis."""
    event_type: str  # "assignment", "prediction", "outcome"
    experiment_id: str
    variant: str
    user_id: str
    timestamp: str
    
    # For predictions
    model_version: str = None
    prediction: Any = None
    latency_ms: float = None
    features: Dict = None
    
    # For outcomes
    outcome_type: str = None  # "conversion", "click", "fraud_detected"
    outcome_value: Any = None
    
    # Metadata
    session_id: str = None
    device_type: str = None
    
class ExperimentEventLogger:
    """Log events for experiment analysis."""
    
    def __init__(self, kafka_servers: str, topic: str):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic
    
    def log_assignment(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        **kwargs
    ):
        """Log experiment assignment."""
        event = ExperimentEvent(
            event_type="assignment",
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
        self._send(event)
    
    def log_prediction(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        model_version: str,
        prediction: Any,
        latency_ms: float,
        **kwargs
    ):
        """Log model prediction."""
        event = ExperimentEvent(
            event_type="prediction",
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            model_version=model_version,
            prediction=prediction,
            latency_ms=latency_ms,
            **kwargs
        )
        self._send(event)
    
    def log_outcome(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        outcome_type: str,
        outcome_value: Any,
        **kwargs
    ):
        """Log business outcome."""
        event = ExperimentEvent(
            event_type="outcome",
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            outcome_type=outcome_type,
            outcome_value=outcome_value,
            **kwargs
        )
        self._send(event)
    
    def _send(self, event: ExperimentEvent):
        """Send event to Kafka."""
        self.producer.send(self.topic, asdict(event))
```

### 5. Statistical Analysis

```python
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ExperimentResults:
    """Results of experiment analysis."""
    experiment_id: str
    control_variant: str
    treatment_variant: str
    
    # Sample sizes
    control_size: int
    treatment_size: int
    
    # Primary metric
    control_rate: float
    treatment_rate: float
    relative_lift: float  # (treatment - control) / control
    absolute_lift: float  # treatment - control
    
    # Statistical significance
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    
    # Recommendation
    recommendation: str

class ExperimentAnalyzer:
    """Analyze A/B test results."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze_conversion_rate(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int
    ) -> Dict:
        """Analyze conversion rate experiment (binomial)."""
        
        # Calculate rates
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        
        # Relative lift
        if control_rate > 0:
            relative_lift = (treatment_rate - control_rate) / control_rate
        else:
            relative_lift = float('inf') if treatment_rate > 0 else 0
        
        absolute_lift = treatment_rate - control_rate
        
        # Two-proportion z-test
        pooled_rate = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))
        
        if se > 0:
            z_score = (treatment_rate - control_rate) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
        else:
            z_score = 0
            p_value = 1.0
        
        # Confidence interval for difference
        ci_margin = stats.norm.ppf(1 - self.alpha/2) * se
        ci_lower = absolute_lift - ci_margin
        ci_upper = absolute_lift + ci_margin
        
        is_significant = p_value < self.alpha
        
        # Recommendation
        if not is_significant:
            recommendation = "No significant difference. Continue testing or conclude no effect."
        elif treatment_rate > control_rate:
            recommendation = f"Treatment wins! {relative_lift*100:.1f}% improvement."
        else:
            recommendation = f"Control wins. Treatment is {abs(relative_lift)*100:.1f}% worse."
        
        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "relative_lift": relative_lift,
            "absolute_lift": absolute_lift,
            "p_value": p_value,
            "z_score": z_score,
            "confidence_interval": (ci_lower, ci_upper),
            "is_significant": is_significant,
            "recommendation": recommendation,
            "sample_sizes": {
                "control": control_total,
                "treatment": treatment_total
            }
        }
    
    def analyze_continuous_metric(
        self,
        control_values: List[float],
        treatment_values: List[float]
    ) -> Dict:
        """Analyze continuous metric experiment (e.g., revenue)."""
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_values)-1)*control_std**2 + (len(treatment_values)-1)*treatment_std**2) /
            (len(control_values) + len(treatment_values) - 2)
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Relative lift
        relative_lift = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
        
        # Confidence interval
        se = np.sqrt(control_std**2/len(control_values) + treatment_std**2/len(treatment_values))
        t_crit = stats.t.ppf(1 - self.alpha/2, len(control_values) + len(treatment_values) - 2)
        ci_margin = t_crit * se
        
        is_significant = p_value < self.alpha
        
        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "relative_lift": relative_lift,
            "absolute_lift": treatment_mean - control_mean,
            "p_value": p_value,
            "t_statistic": t_stat,
            "cohens_d": cohens_d,
            "confidence_interval": (
                treatment_mean - control_mean - ci_margin,
                treatment_mean - control_mean + ci_margin
            ),
            "is_significant": is_significant,
            "sample_sizes": {
                "control": len(control_values),
                "treatment": len(treatment_values)
            }
        }
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        mde: float,  # Minimum detectable effect (relative)
        power: float = 0.8
    ) -> int:
        """Calculate required sample size per variant."""
        
        treatment_rate = baseline_rate * (1 + mde)
        
        # Pooled rate
        p_pooled = (baseline_rate + treatment_rate) / 2
        
        # Effect size
        effect_size = abs(treatment_rate - baseline_rate)
        
        # Standard error
        se = np.sqrt(2 * p_pooled * (1 - p_pooled))
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size per group
        n = ((z_alpha + z_beta) * se / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def check_guardrails(
        self,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        thresholds: Dict[str, float]  # metric -> max acceptable degradation
    ) -> Dict[str, bool]:
        """Check if guardrail metrics are violated."""
        
        results = {}
        for metric, threshold in thresholds.items():
            control_val = control_metrics.get(metric, 0)
            treatment_val = treatment_metrics.get(metric, 0)
            
            if control_val > 0:
                degradation = (treatment_val - control_val) / control_val
            else:
                degradation = 0
            
            # For latency, error_rate: higher is worse
            # Guardrail violated if degradation > threshold
            results[metric] = {
                "violated": degradation > threshold,
                "control": control_val,
                "treatment": treatment_val,
                "degradation": degradation,
                "threshold": threshold
            }
        
        return results

# Usage
analyzer = ExperimentAnalyzer(confidence_level=0.95)

# Analyze conversion rate
results = analyzer.analyze_conversion_rate(
    control_conversions=1200,
    control_total=10000,
    treatment_conversions=1350,
    treatment_total=10000
)

print(f"Control rate: {results['control_rate']:.2%}")
print(f"Treatment rate: {results['treatment_rate']:.2%}")
print(f"Relative lift: {results['relative_lift']:.1%}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['is_significant']}")
print(f"Recommendation: {results['recommendation']}")

# Calculate required sample size
sample_size = analyzer.calculate_sample_size(
    baseline_rate=0.12,  # 12% baseline conversion
    mde=0.05,  # Detect 5% relative improvement
    power=0.8
)
print(f"Required sample size per variant: {sample_size:,}")
```

---

## 📊 Multi-Armed Bandits

For scenarios requiring faster convergence, consider bandits over A/B tests:

```python
import numpy as np
from typing import List, Dict

class ThompsonSamplingBandit:
    """Multi-armed bandit using Thompson Sampling."""
    
    def __init__(self, variants: List[str]):
        self.variants = variants
        # Beta distribution parameters (successes, failures)
        self.alpha = {v: 1 for v in variants}  # Prior successes
        self.beta = {v: 1 for v in variants}   # Prior failures
    
    def select_variant(self) -> str:
        """Select variant using Thompson Sampling."""
        samples = {
            v: np.random.beta(self.alpha[v], self.beta[v])
            for v in self.variants
        }
        return max(samples, key=samples.get)
    
    def update(self, variant: str, reward: int):
        """Update beliefs based on observed reward."""
        if reward == 1:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get current probability each variant is best."""
        n_samples = 10000
        wins = {v: 0 for v in self.variants}
        
        for _ in range(n_samples):
            samples = {
                v: np.random.beta(self.alpha[v], self.beta[v])
                for v in self.variants
            }
            winner = max(samples, key=samples.get)
            wins[winner] += 1
        
        return {v: wins[v]/n_samples for v in self.variants}

# Usage
bandit = ThompsonSamplingBandit(["control", "treatment_a", "treatment_b"])

# Simulate
for _ in range(1000):
    variant = bandit.select_variant()
    # Simulate reward (in practice, this is the observed outcome)
    reward = np.random.binomial(1, {"control": 0.10, "treatment_a": 0.12, "treatment_b": 0.09}[variant])
    bandit.update(variant, reward)

print("Probability each variant is best:")
print(bandit.get_probabilities())
```

---

## ✅ Best Practices

1. **Define metrics upfront** - Primary, secondary, guardrails
2. **Calculate sample size** - Don't peek early
3. **Use sticky bucketing** - Consistent user experience
4. **Log everything** - Predictions, outcomes, metadata
5. **Run power analysis** - Know when you have enough data
6. **Check guardrails** - Don't regress on key metrics
7. **Document decisions** - Why experiment was run and outcome

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Peeking** | Inflated false positives | Pre-calculate duration, use sequential testing |
| **Multiple comparisons** | False discoveries | Bonferroni correction, control FDR |
| **Selection bias** | Skewed results | Random assignment, stratification |
| **Novelty effects** | Temporary changes | Run longer, segment by user tenure |
| **Network effects** | Contamination | Cluster randomization |

---

## 🔗 Related Topics

- [Model Updates](./model-updates.md) - Safe rollout strategies
- [Model Deployment](./model-deployment.md) - Deploy variants
- [Performance Metrics](../06-monitoring-observability/performance-metrics.md) - Track experiment metrics
