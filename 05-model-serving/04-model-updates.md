# Model Updates

## Overview

Model updates replace or augment production models with new versions. Safe model updates minimize user impact, enable quick rollbacks, and provide confidence that new models improve (or at least don't degrade) system performance. This is critical because ML models, unlike traditional software, can fail silently with degraded predictions.

---

## 🎯 Update Strategies

### Strategy Comparison

| Strategy | Risk | Complexity | Rollback Time | Use Case |
|----------|------|------------|---------------|----------|
| **Big Bang** | High | Low | Minutes-Hours | Dev/test only |
| **Blue-Green** | Medium | Medium | Seconds | Fast cutover needed |
| **Canary** | Low | High | Seconds | Production critical |
| **Shadow Mode** | Very Low | High | N/A (no impact) | High-risk changes |
| **Feature Flags** | Low | Medium | Instant | Gradual rollout |

---

## 🔵 1. Blue-Green Deployment

**How it works:** Two identical environments, instant traffic switch

```
┌─────────────────────────────────────────────────────────────────┐
│  Blue-Green Deployment                                           │
│                                                                  │
│  Before:                                                        │
│  ┌────────────┐     Load      ┌────────────┐                   │
│  │   Blue     │◀───Balancer───│  Users     │                   │
│  │  (v1.0)    │               │            │                   │
│  └────────────┘               └────────────┘                   │
│  ┌────────────┐                                                 │
│  │   Green    │  (Standby, preparing v2.0)                     │
│  │  (v2.0)    │                                                 │
│  └────────────┘                                                 │
│                                                                  │
│  After Switch:                                                  │
│  ┌────────────┐                                                 │
│  │   Blue     │  (Standby for rollback)                        │
│  │  (v1.0)    │                                                 │
│  └────────────┘                                                 │
│  ┌────────────┐     Load      ┌────────────┐                   │
│  │   Green    │◀───Balancer───│  Users     │                   │
│  │  (v2.0)    │               │            │                   │
│  └────────────┘               └────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import boto3
import time

class Environment(Enum):
    BLUE = "blue"
    GREEN = "green"

@dataclass
class DeploymentStatus:
    active_environment: Environment
    blue_version: str
    green_version: str
    last_switch_time: float

class BlueGreenDeployer:
    """Blue-green deployment manager for ML models."""
    
    def __init__(
        self,
        cluster_name: str,
        blue_service: str,
        green_service: str,
        load_balancer_arn: str
    ):
        self.ecs = boto3.client('ecs')
        self.elbv2 = boto3.client('elbv2')
        self.cluster = cluster_name
        self.blue_service = blue_service
        self.green_service = green_service
        self.lb_arn = load_balancer_arn
        
        self.status = DeploymentStatus(
            active_environment=Environment.BLUE,
            blue_version="v1.0.0",
            green_version="v1.0.0",
            last_switch_time=time.time()
        )
    
    def deploy_to_inactive(self, new_version: str, image_uri: str) -> bool:
        """Deploy new version to inactive environment."""
        
        # Determine inactive environment
        inactive = (Environment.GREEN 
                   if self.status.active_environment == Environment.BLUE 
                   else Environment.BLUE)
        service = (self.green_service 
                  if inactive == Environment.GREEN 
                  else self.blue_service)
        
        print(f"Deploying {new_version} to {inactive.value} environment")
        
        # Update ECS service with new image
        try:
            # Get current task definition
            response = self.ecs.describe_services(
                cluster=self.cluster,
                services=[service]
            )
            task_def_arn = response['services'][0]['taskDefinition']
            
            # Get task definition details
            task_def = self.ecs.describe_task_definition(
                taskDefinition=task_def_arn
            )['taskDefinition']
            
            # Create new task definition with updated image
            container_defs = task_def['containerDefinitions']
            container_defs[0]['image'] = image_uri
            
            new_task_def = self.ecs.register_task_definition(
                family=task_def['family'],
                containerDefinitions=container_defs,
                cpu=task_def.get('cpu'),
                memory=task_def.get('memory'),
                networkMode=task_def.get('networkMode'),
                requiresCompatibilities=task_def.get('requiresCompatibilities', [])
            )
            
            # Update service
            self.ecs.update_service(
                cluster=self.cluster,
                service=service,
                taskDefinition=new_task_def['taskDefinition']['taskDefinitionArn']
            )
            
            # Wait for deployment
            self._wait_for_stable(service)
            
            # Update status
            if inactive == Environment.BLUE:
                self.status.blue_version = new_version
            else:
                self.status.green_version = new_version
            
            print(f"Deployment to {inactive.value} complete")
            return True
        
        except Exception as e:
            print(f"Deployment failed: {e}")
            return False
    
    def switch_traffic(self) -> bool:
        """Switch traffic to inactive environment."""
        
        new_active = (Environment.GREEN 
                     if self.status.active_environment == Environment.BLUE 
                     else Environment.BLUE)
        
        print(f"Switching traffic from {self.status.active_environment.value} to {new_active.value}")
        
        try:
            # Get target groups
            listeners = self.elbv2.describe_listeners(
                LoadBalancerArn=self.lb_arn
            )['Listeners']
            
            # Update listener rules to point to new target group
            for listener in listeners:
                rules = self.elbv2.describe_rules(
                    ListenerArn=listener['ListenerArn']
                )['Rules']
                
                for rule in rules:
                    if not rule['IsDefault']:
                        # Update target group
                        new_tg = self._get_target_group(new_active)
                        self.elbv2.modify_rule(
                            RuleArn=rule['RuleArn'],
                            Actions=[{
                                'Type': 'forward',
                                'TargetGroupArn': new_tg
                            }]
                        )
            
            self.status.active_environment = new_active
            self.status.last_switch_time = time.time()
            
            print(f"Traffic switched to {new_active.value}")
            return True
        
        except Exception as e:
            print(f"Traffic switch failed: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous environment."""
        return self.switch_traffic()
    
    def _wait_for_stable(self, service: str, timeout: int = 300):
        """Wait for ECS service to become stable."""
        waiter = self.ecs.get_waiter('services_stable')
        waiter.wait(
            cluster=self.cluster,
            services=[service],
            WaiterConfig={'Delay': 10, 'MaxAttempts': timeout // 10}
        )
    
    def _get_target_group(self, env: Environment) -> str:
        """Get target group ARN for environment."""
        # Implementation depends on your setup
        pass

# Usage
deployer = BlueGreenDeployer(
    cluster_name="ml-cluster",
    blue_service="model-server-blue",
    green_service="model-server-green",
    load_balancer_arn="arn:aws:elasticloadbalancing:..."
)

# Deploy new version
deployer.deploy_to_inactive("v2.0.0", "registry/model-server:v2.0.0")

# Validate new version
if validate_model_health():
    deployer.switch_traffic()
else:
    print("Validation failed, keeping current version")
```

---

## 🐤 2. Canary Deployment

**How it works:** Gradually shift traffic from old to new version

```
┌─────────────────────────────────────────────────────────────────┐
│  Canary Deployment Stages                                        │
│                                                                  │
│  Stage 1 (5% canary):                                           │
│  ┌────────────┐                                                 │
│  │  v1.0.0    │◀──── 95% traffic                               │
│  └────────────┘                                                 │
│  ┌────────────┐                                                 │
│  │  v2.0.0    │◀──── 5% traffic (canary)                       │
│  └────────────┘                                                 │
│                                                                  │
│  Stage 2 (25% canary):                                          │
│  ┌────────────┐                                                 │
│  │  v1.0.0    │◀──── 75% traffic                               │
│  └────────────┘                                                 │
│  ┌────────────┐                                                 │
│  │  v2.0.0    │◀──── 25% traffic                               │
│  └────────────┘                                                 │
│                                                                  │
│  Stage 3 (100%):                                                │
│  ┌────────────┐                                                 │
│  │  v2.0.0    │◀──── 100% traffic                              │
│  └────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Callable
import time
import asyncio

@dataclass
class CanaryStage:
    """Canary deployment stage configuration."""
    percentage: int  # Traffic percentage to canary
    duration_minutes: int  # How long to stay at this stage
    metrics_thresholds: dict  # Metrics to check

@dataclass  
class CanaryConfig:
    """Canary deployment configuration."""
    stages: List[CanaryStage]
    rollback_on_failure: bool = True
    notification_webhook: Optional[str] = None

class CanaryDeployer:
    """Progressive canary deployment for ML models."""
    
    def __init__(
        self,
        traffic_splitter: 'TrafficSplitter',
        metrics_client: 'MetricsClient',
        config: CanaryConfig
    ):
        self.traffic_splitter = traffic_splitter
        self.metrics = metrics_client
        self.config = config
        self.current_stage = 0
        self.deployment_active = False
    
    async def deploy(
        self,
        baseline_version: str,
        canary_version: str
    ) -> bool:
        """Execute canary deployment."""
        
        self.deployment_active = True
        self.current_stage = 0
        
        print(f"Starting canary deployment: {baseline_version} -> {canary_version}")
        
        for i, stage in enumerate(self.config.stages):
            self.current_stage = i
            
            print(f"Stage {i+1}: {stage.percentage}% traffic to canary")
            
            # Update traffic split
            self.traffic_splitter.set_split(
                baseline_version=baseline_version,
                canary_version=canary_version,
                canary_percentage=stage.percentage
            )
            
            # Wait and monitor
            success = await self._monitor_stage(
                stage=stage,
                baseline_version=baseline_version,
                canary_version=canary_version
            )
            
            if not success:
                print(f"Canary failed at stage {i+1}")
                
                if self.config.rollback_on_failure:
                    await self.rollback(baseline_version)
                
                return False
        
        # Full rollout
        print("Canary successful, completing rollout")
        self.traffic_splitter.set_split(
            baseline_version=canary_version,
            canary_version=canary_version,
            canary_percentage=100
        )
        
        self.deployment_active = False
        return True
    
    async def _monitor_stage(
        self,
        stage: CanaryStage,
        baseline_version: str,
        canary_version: str
    ) -> bool:
        """Monitor canary during a stage."""
        
        check_interval = 60  # seconds
        checks_remaining = (stage.duration_minutes * 60) // check_interval
        
        while checks_remaining > 0:
            # Get metrics for both versions
            baseline_metrics = await self.metrics.get_metrics(
                version=baseline_version,
                window_minutes=5
            )
            canary_metrics = await self.metrics.get_metrics(
                version=canary_version,
                window_minutes=5
            )
            
            # Check thresholds
            for metric, threshold in stage.metrics_thresholds.items():
                baseline_val = baseline_metrics.get(metric, 0)
                canary_val = canary_metrics.get(metric, 0)
                
                if isinstance(threshold, dict):
                    # Relative threshold
                    if threshold.get('type') == 'relative':
                        max_degradation = threshold.get('max_degradation', 0.1)
                        if baseline_val > 0:
                            degradation = (canary_val - baseline_val) / baseline_val
                            # For error_rate, latency: higher is worse
                            if metric in ['error_rate', 'latency_p99']:
                                if degradation > max_degradation:
                                    print(f"Canary failed: {metric} degraded by {degradation:.1%}")
                                    return False
                            # For accuracy, success_rate: lower is worse
                            else:
                                if degradation < -max_degradation:
                                    print(f"Canary failed: {metric} dropped by {abs(degradation):.1%}")
                                    return False
                else:
                    # Absolute threshold
                    if metric in ['error_rate', 'latency_p99']:
                        if canary_val > threshold:
                            print(f"Canary failed: {metric}={canary_val} > {threshold}")
                            return False
                    else:
                        if canary_val < threshold:
                            print(f"Canary failed: {metric}={canary_val} < {threshold}")
                            return False
            
            print(f"Canary healthy, {checks_remaining} checks remaining")
            await asyncio.sleep(check_interval)
            checks_remaining -= 1
        
        return True
    
    async def rollback(self, baseline_version: str):
        """Rollback canary deployment."""
        print(f"Rolling back to {baseline_version}")
        
        self.traffic_splitter.set_split(
            baseline_version=baseline_version,
            canary_version=baseline_version,
            canary_percentage=0
        )
        
        self.deployment_active = False

class TrafficSplitter:
    """Manage traffic splitting between model versions."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def set_split(
        self,
        baseline_version: str,
        canary_version: str,
        canary_percentage: int
    ):
        """Set traffic split configuration."""
        config = {
            'baseline_version': baseline_version,
            'canary_version': canary_version,
            'canary_percentage': canary_percentage
        }
        self.redis.set('traffic_split', json.dumps(config))
    
    def get_version_for_request(self, user_id: str) -> str:
        """Get model version for a specific request."""
        config = json.loads(self.redis.get('traffic_split') or '{}')
        
        if not config:
            return 'default'
        
        # Deterministic assignment based on user_id
        bucket = hash(user_id) % 100
        
        if bucket < config.get('canary_percentage', 0):
            return config['canary_version']
        return config['baseline_version']

# Usage
config = CanaryConfig(
    stages=[
        CanaryStage(
            percentage=5,
            duration_minutes=15,
            metrics_thresholds={
                'error_rate': {'type': 'relative', 'max_degradation': 0.1},
                'latency_p99': {'type': 'relative', 'max_degradation': 0.2},
                'accuracy': {'type': 'relative', 'max_degradation': 0.05}
            }
        ),
        CanaryStage(
            percentage=25,
            duration_minutes=30,
            metrics_thresholds={
                'error_rate': 0.01,  # Absolute: max 1%
                'latency_p99': 200,  # Absolute: max 200ms
                'accuracy': 0.90     # Absolute: min 90%
            }
        ),
        CanaryStage(
            percentage=50,
            duration_minutes=60,
            metrics_thresholds={
                'error_rate': 0.01,
                'latency_p99': 200,
                'accuracy': 0.90
            }
        )
    ],
    rollback_on_failure=True
)

deployer = CanaryDeployer(
    traffic_splitter=TrafficSplitter(redis_client),
    metrics_client=MetricsClient(),
    config=config
)

# Deploy
success = await deployer.deploy(
    baseline_version="v1.0.0",
    canary_version="v2.0.0"
)
```

---

## 👻 3. Shadow Mode

**How it works:** Run new model in parallel, compare outputs without affecting users

```
┌─────────────────────────────────────────────────────────────────┐
│  Shadow Mode Deployment                                          │
│                                                                  │
│                    ┌─────────────────┐                          │
│                    │   User Request  │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │   API Gateway   │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│              ▼                             ▼                    │
│     ┌─────────────────┐          ┌─────────────────┐           │
│     │  Primary (v1.0) │          │  Shadow (v2.0)  │           │
│     │  Returns to     │          │  Logs only      │           │
│     │  user           │          │  (no response)  │           │
│     └────────┬────────┘          └────────┬────────┘           │
│              │                             │                    │
│              ▼                             ▼                    │
│     ┌─────────────────┐          ┌─────────────────┐           │
│     │    Response     │          │   Comparison    │           │
│     │    to User      │          │   & Analysis    │           │
│     └─────────────────┘          └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import json

@dataclass
class ShadowComparison:
    """Comparison between primary and shadow predictions."""
    request_id: str
    user_id: str
    features: Dict
    
    primary_prediction: Any
    shadow_prediction: Any
    
    primary_latency_ms: float
    shadow_latency_ms: float
    
    predictions_match: bool
    difference: Optional[float]
    
    timestamp: float

class ShadowModeServer:
    """Model server with shadow mode support."""
    
    def __init__(
        self,
        primary_model,
        shadow_model,
        comparison_logger,
        shadow_percentage: int = 100
    ):
        self.primary_model = primary_model
        self.shadow_model = shadow_model
        self.comparison_logger = comparison_logger
        self.shadow_percentage = shadow_percentage
    
    async def predict(
        self,
        request_id: str,
        user_id: str,
        features: Dict
    ) -> Dict:
        """Make prediction with optional shadow comparison."""
        
        # Always run primary
        primary_start = time.time()
        primary_result = await self._run_primary(features)
        primary_latency = (time.time() - primary_start) * 1000
        
        # Conditionally run shadow (non-blocking)
        if self._should_shadow(user_id):
            asyncio.create_task(
                self._run_shadow_and_compare(
                    request_id=request_id,
                    user_id=user_id,
                    features=features,
                    primary_result=primary_result,
                    primary_latency=primary_latency
                )
            )
        
        # Return only primary result
        return primary_result
    
    async def _run_primary(self, features: Dict) -> Dict:
        """Run primary model."""
        prediction = self.primary_model.predict(features)
        return {
            'prediction': prediction,
            'model_version': self.primary_model.version
        }
    
    async def _run_shadow_and_compare(
        self,
        request_id: str,
        user_id: str,
        features: Dict,
        primary_result: Dict,
        primary_latency: float
    ):
        """Run shadow model and log comparison."""
        try:
            shadow_start = time.time()
            shadow_prediction = self.shadow_model.predict(features)
            shadow_latency = (time.time() - shadow_start) * 1000
            
            # Compare predictions
            primary_pred = primary_result['prediction']
            
            if isinstance(primary_pred, (int, float)):
                predictions_match = abs(primary_pred - shadow_prediction) < 0.01
                difference = shadow_prediction - primary_pred
            else:
                predictions_match = primary_pred == shadow_prediction
                difference = None
            
            # Log comparison
            comparison = ShadowComparison(
                request_id=request_id,
                user_id=user_id,
                features=features,
                primary_prediction=primary_pred,
                shadow_prediction=shadow_prediction,
                primary_latency_ms=primary_latency,
                shadow_latency_ms=shadow_latency,
                predictions_match=predictions_match,
                difference=difference,
                timestamp=time.time()
            )
            
            await self.comparison_logger.log(comparison)
        
        except Exception as e:
            # Shadow errors should never affect primary
            print(f"Shadow prediction error: {e}")
    
    def _should_shadow(self, user_id: str) -> bool:
        """Determine if request should be shadowed."""
        return hash(user_id) % 100 < self.shadow_percentage

class ShadowAnalyzer:
    """Analyze shadow mode results."""
    
    def __init__(self, data_store):
        self.data_store = data_store
    
    def get_comparison_stats(
        self,
        start_time: float,
        end_time: float
    ) -> Dict:
        """Get statistics on shadow comparisons."""
        
        comparisons = self.data_store.query(
            start_time=start_time,
            end_time=end_time
        )
        
        if not comparisons:
            return {}
        
        total = len(comparisons)
        matching = sum(1 for c in comparisons if c.predictions_match)
        
        differences = [c.difference for c in comparisons if c.difference is not None]
        
        return {
            'total_comparisons': total,
            'match_rate': matching / total,
            'mismatch_rate': (total - matching) / total,
            
            # Difference statistics
            'mean_difference': np.mean(differences) if differences else None,
            'std_difference': np.std(differences) if differences else None,
            'max_difference': max(differences) if differences else None,
            'min_difference': min(differences) if differences else None,
            
            # Latency comparison
            'mean_primary_latency': np.mean([c.primary_latency_ms for c in comparisons]),
            'mean_shadow_latency': np.mean([c.shadow_latency_ms for c in comparisons]),
            
            # Distribution of differences
            'difference_percentiles': {
                'p50': np.percentile(differences, 50) if differences else None,
                'p90': np.percentile(differences, 90) if differences else None,
                'p99': np.percentile(differences, 99) if differences else None
            }
        }
    
    def find_discrepancies(
        self,
        start_time: float,
        end_time: float,
        threshold: float = 0.1
    ) -> List[ShadowComparison]:
        """Find significant prediction discrepancies."""
        
        comparisons = self.data_store.query(
            start_time=start_time,
            end_time=end_time
        )
        
        discrepancies = []
        for c in comparisons:
            if c.difference is not None and abs(c.difference) > threshold:
                discrepancies.append(c)
        
        return discrepancies

# Usage
server = ShadowModeServer(
    primary_model=load_model("v1.0.0"),
    shadow_model=load_model("v2.0.0"),
    comparison_logger=ComparisonLogger(),
    shadow_percentage=100  # Shadow all requests
)

# After running shadow mode for a period
analyzer = ShadowAnalyzer(data_store)
stats = analyzer.get_comparison_stats(
    start_time=time.time() - 86400,  # Last 24 hours
    end_time=time.time()
)

print(f"Match rate: {stats['match_rate']:.1%}")
print(f"Mean difference: {stats['mean_difference']:.4f}")
print(f"Latency comparison: {stats['mean_shadow_latency']/stats['mean_primary_latency']:.1%} of primary")
```

---

## 🚩 4. Feature Flags

**How it works:** Control model rollout via configuration

```python
from typing import Any, Dict, Optional
import json

class ModelFeatureFlags:
    """Feature flag management for model versions."""
    
    def __init__(self, config_store):
        self.config_store = config_store
    
    def get_model_version(
        self,
        feature_name: str,
        user_id: str,
        context: Dict = None
    ) -> str:
        """Get model version based on feature flag rules."""
        
        flag = self.config_store.get(f"flag:{feature_name}")
        if not flag:
            return "default"
        
        rules = json.loads(flag)
        
        # Check targeting rules in order
        for rule in rules.get('rules', []):
            if self._matches_rule(rule, user_id, context):
                return rule.get('model_version')
        
        # Check percentage rollout
        percentage = rules.get('percentage', 0)
        if hash(user_id) % 100 < percentage:
            return rules.get('treatment_version', 'default')
        
        return rules.get('control_version', 'default')
    
    def _matches_rule(
        self,
        rule: Dict,
        user_id: str,
        context: Dict
    ) -> bool:
        """Check if user matches targeting rule."""
        
        conditions = rule.get('conditions', [])
        
        for condition in conditions:
            attribute = condition.get('attribute')
            operator = condition.get('operator')
            value = condition.get('value')
            
            # Get attribute value
            if attribute == 'user_id':
                actual = user_id
            else:
                actual = context.get(attribute) if context else None
            
            # Evaluate condition
            if not self._evaluate_condition(actual, operator, value):
                return False
        
        return True
    
    def _evaluate_condition(
        self,
        actual: Any,
        operator: str,
        expected: Any
    ) -> bool:
        """Evaluate a single condition."""
        
        if operator == 'equals':
            return actual == expected
        elif operator == 'not_equals':
            return actual != expected
        elif operator == 'in':
            return actual in expected
        elif operator == 'not_in':
            return actual not in expected
        elif operator == 'greater_than':
            return actual > expected
        elif operator == 'less_than':
            return actual < expected
        elif operator == 'contains':
            return expected in actual if actual else False
        
        return False

# Example flag configuration
flag_config = {
    "feature_name": "fraud_model_v2",
    "rules": [
        {
            # Beta users always get new model
            "conditions": [
                {"attribute": "user_tier", "operator": "equals", "value": "beta"}
            ],
            "model_version": "v2.0.0"
        },
        {
            # High-value merchants get new model
            "conditions": [
                {"attribute": "merchant_volume", "operator": "greater_than", "value": 10000}
            ],
            "model_version": "v2.0.0"
        }
    ],
    # 20% rollout for everyone else
    "percentage": 20,
    "treatment_version": "v2.0.0",
    "control_version": "v1.0.0"
}
```

---

## ✅ Best Practices

### Update Checklist

```markdown
## Pre-Update
- [ ] Model validated offline
- [ ] Deployment plan documented
- [ ] Rollback procedure tested
- [ ] Monitoring dashboards ready
- [ ] On-call team notified

## During Update
- [ ] Start with small traffic percentage
- [ ] Monitor key metrics (latency, errors, accuracy)
- [ ] Watch guardrail metrics
- [ ] Gradual traffic increase

## Post-Update
- [ ] Full rollout verified
- [ ] Documentation updated
- [ ] Post-mortem if issues
- [ ] Clean up old version resources
```

### Monitoring During Updates

```python
def create_update_dashboard_query():
    """Metrics to monitor during model updates."""
    return {
        "panels": [
            # Error rate by version
            {
                "title": "Error Rate by Version",
                "query": 'sum(rate(predictions_errors_total[5m])) by (model_version)'
            },
            # Latency by version
            {
                "title": "P99 Latency by Version",
                "query": 'histogram_quantile(0.99, sum(rate(prediction_latency_bucket[5m])) by (le, model_version))'
            },
            # Traffic split
            {
                "title": "Traffic Distribution",
                "query": 'sum(rate(predictions_total[5m])) by (model_version)'
            },
            # Prediction distribution
            {
                "title": "Prediction Distribution",
                "query": 'histogram_quantile(0.5, sum(rate(prediction_value_bucket[5m])) by (le, model_version))'
            }
        ]
    }
```

---

## 🔗 Related Topics

- [A/B Testing](./ab-testing.md) - Compare model versions
- [Model Deployment](./model-deployment.md) - Deploy models
- [Monitoring & Observability](../06-monitoring-observability/README.md) - Monitor updates
- [High Availability](../08-reliability-fault-tolerance/high-availability.md) - Safe rollouts
