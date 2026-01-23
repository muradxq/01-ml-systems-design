# Compliance

## Overview

Compliance ensures ML systems meet legal and regulatory requirements. ML systems face unique compliance challenges around automated decision-making, data usage, and model explainability. Non-compliance can result in significant fines (up to 4% of global revenue for GDPR) and reputational damage.

---

## 📋 Regulatory Requirements

### GDPR (EU) - ML Specific

| Requirement | ML Implication | Implementation |
|-------------|----------------|----------------|
| **Right to explanation** | Explain automated decisions | Model interpretability |
| **Right to human review** | Allow appeal of ML decisions | Human-in-the-loop |
| **Data minimization** | Use only necessary data | Feature selection audit |
| **Purpose limitation** | Use data only for stated purpose | Data lineage tracking |
| **Right to erasure** | Delete user data on request | Data deletion pipeline |
| **Data portability** | Export user data | Data export API |

### CCPA (California)

| Requirement | ML Implication | Implementation |
|-------------|----------------|----------------|
| **Right to know** | Disclose data collection | Privacy policy |
| **Right to delete** | Delete data on request | Deletion workflow |
| **Opt-out of sale** | Allow opt-out | Consent management |
| **Non-discrimination** | Equal service for opt-out | Model fairness |

### HIPAA (Healthcare)

| Requirement | ML Implication | Implementation |
|-------------|----------------|----------------|
| **PHI protection** | Encrypt health data | Encryption + access controls |
| **Minimum necessary** | Access only needed data | Role-based access |
| **Audit controls** | Log all access | Comprehensive logging |
| **BAA with vendors** | Contracts with ML services | Legal agreements |

---

## 🏗️ Compliance Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Compliance Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Consent Management                       │   │
│  │  - Track user consent                                    │   │
│  │  - Respect opt-outs                                      │   │
│  │  - Version consent                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Data Governance                          │   │
│  │  - Data lineage                                          │   │
│  │  - Purpose tracking                                      │   │
│  │  - Retention policies                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Model Governance                         │   │
│  │  - Explainability                                        │   │
│  │  - Fairness monitoring                                   │   │
│  │  - Bias detection                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Audit & Reporting                        │   │
│  │  - Access logs                                           │   │
│  │  - Compliance reports                                    │   │
│  │  - Incident tracking                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Implementation

### GDPR Right to Explanation

```python
import shap
import lime
from typing import Dict, Any

class ModelExplainer:
    """
    Provide explanations for model predictions (GDPR Article 22).
    """
    
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(model)
    
    def explain_prediction(self, features: dict) -> Dict[str, Any]:
        """
        Generate human-readable explanation for prediction.
        Required for automated decisions with significant effects.
        """
        # Make prediction
        feature_array = [features[f] for f in self.feature_names]
        prediction = self.model.predict([feature_array])[0]
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values([feature_array])
        
        # Create explanation
        feature_importance = sorted(
            zip(self.feature_names, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            prediction, feature_importance[:5]
        )
        
        return {
            "prediction": prediction,
            "explanation": explanation,
            "feature_contributions": {
                f: float(v) for f, v in feature_importance
            },
            "confidence": float(self.model.predict_proba([feature_array]).max())
        }
    
    def _generate_explanation(self, prediction, top_features) -> str:
        """Generate natural language explanation."""
        reasons = []
        for feature, importance in top_features:
            direction = "increased" if importance > 0 else "decreased"
            reasons.append(f"{feature} {direction} the score")
        
        return f"The prediction was {prediction}. " + \
               f"Main factors: {', '.join(reasons[:3])}."

# Usage
explainer = ModelExplainer(model, feature_names)

@app.post("/predict")
async def predict_with_explanation(request: PredictionRequest):
    result = explainer.explain_prediction(request.features)
    
    # Log explanation for compliance
    audit_logger.log_prediction(
        user_id=request.user_id,
        prediction=result['prediction'],
        explanation=result['explanation']
    )
    
    return result
```

### Right to Erasure (GDPR/CCPA)

```python
class DataDeletionService:
    """
    Handle right to erasure requests.
    Must delete all user data from training data, features, and logs.
    """
    
    def __init__(self):
        self.data_stores = []
        self.deletion_log = []
    
    async def process_deletion_request(self, user_id: str) -> Dict:
        """
        Process GDPR/CCPA deletion request.
        """
        deletion_report = {
            "user_id": user_id,
            "request_time": datetime.utcnow().isoformat(),
            "deleted_from": [],
            "status": "completed"
        }
        
        # 1. Delete from feature store
        await self._delete_from_feature_store(user_id)
        deletion_report["deleted_from"].append("feature_store")
        
        # 2. Delete from prediction logs
        await self._delete_from_logs(user_id)
        deletion_report["deleted_from"].append("prediction_logs")
        
        # 3. Delete from training data (if possible)
        # Note: May need to retrain model
        if await self._can_delete_from_training(user_id):
            await self._delete_from_training_data(user_id)
            deletion_report["deleted_from"].append("training_data")
            deletion_report["retrain_required"] = True
        
        # 4. Delete from caches
        await self._delete_from_caches(user_id)
        deletion_report["deleted_from"].append("caches")
        
        # 5. Log deletion for compliance
        self._log_deletion(deletion_report)
        
        return deletion_report
    
    async def _delete_from_feature_store(self, user_id: str):
        """Delete user's features."""
        await feature_store.delete(f"user:{user_id}")
    
    async def _delete_from_logs(self, user_id: str):
        """Delete user's prediction history."""
        await log_store.delete_by_user(user_id)
    
    def _log_deletion(self, report: Dict):
        """Log deletion for compliance audit."""
        self.deletion_log.append(report)
        audit_logger.log_access(
            user_id="system",
            action="gdpr:deletion",
            resource=report["user_id"],
            result="completed",
            metadata=report
        )
```

### Fairness and Bias Monitoring

```python
from fairlearn.metrics import MetricFrame
import pandas as pd

class FairnessMonitor:
    """
    Monitor model fairness for compliance.
    Many regulations require non-discrimination.
    """
    
    def __init__(self, sensitive_features: list):
        self.sensitive_features = sensitive_features
    
    def compute_fairness_metrics(self, y_true, y_pred, 
                                 sensitive_data: pd.DataFrame) -> Dict:
        """
        Compute fairness metrics across sensitive groups.
        """
        metrics = {}
        
        for feature in self.sensitive_features:
            # Compute metrics by group
            metric_frame = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "selection_rate": selection_rate,
                    "false_positive_rate": false_positive_rate
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_data[feature]
            )
            
            # Calculate disparities
            metrics[feature] = {
                "by_group": metric_frame.by_group.to_dict(),
                "ratio": metric_frame.ratio().to_dict(),
                "difference": metric_frame.difference().to_dict()
            }
            
            # Check for violations (e.g., 80% rule)
            selection_rates = metric_frame.by_group["selection_rate"]
            min_rate = selection_rates.min()
            max_rate = selection_rates.max()
            
            if min_rate / max_rate < 0.8:
                metrics[feature]["violation"] = "Disparate impact detected"
        
        return metrics
    
    def generate_fairness_report(self, y_true, y_pred,
                                sensitive_data: pd.DataFrame) -> str:
        """Generate compliance report for fairness."""
        metrics = self.compute_fairness_metrics(y_true, y_pred, sensitive_data)
        
        report = "# Model Fairness Report\n\n"
        report += f"Generated: {datetime.utcnow().isoformat()}\n\n"
        
        for feature, data in metrics.items():
            report += f"## {feature}\n"
            report += f"Selection rate ratio: {data['ratio'].get('selection_rate', 'N/A')}\n"
            if "violation" in data:
                report += f"⚠️ {data['violation']}\n"
            report += "\n"
        
        return report
```

---

## 📊 Compliance Checklist

### GDPR Checklist
- [ ] Privacy policy updated for ML usage
- [ ] Consent mechanism for data collection
- [ ] Data Processing Agreement with vendors
- [ ] Right to explanation implemented
- [ ] Data deletion process
- [ ] Data portability API
- [ ] Data protection impact assessment
- [ ] DPO appointed (if required)

### CCPA Checklist
- [ ] Privacy policy with data categories
- [ ] "Do Not Sell" mechanism
- [ ] Deletion request process
- [ ] Consumer request verification
- [ ] 12-month data access provision

### HIPAA Checklist (Healthcare ML)
- [ ] BAA with all vendors
- [ ] PHI encryption at rest and transit
- [ ] Access controls implemented
- [ ] Audit logs enabled
- [ ] Incident response plan
- [ ] Employee training

---

## ✅ Best Practices

1. **Privacy by design** - build compliance into architecture
2. **Document everything** - maintain compliance evidence
3. **Regular audits** - verify compliance quarterly
4. **Train team** - ensure awareness of requirements
5. **Stay updated** - regulations evolve
6. **Legal review** - involve legal in ML decisions
7. **Automate compliance** - reduce human error

---

## 🔗 Related Topics

- [Data Privacy](./data-privacy.md) - Technical privacy measures
- [Access Control](./access-control.md) - Access management
- [Model Security](./model-security.md) - Protect ML systems
