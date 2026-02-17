# Disaster Recovery

## Overview

Disaster recovery (DR) plans ensure ML systems can recover from major failures—region outages, data corruption, critical bugs, or security breaches. For ML systems, DR is uniquely complex because you need to recover not just code and infrastructure, but also models, training data, feature pipelines, and prediction history.

---

## 📊 Key Metrics

| Metric | Definition | Typical Target |
|--------|------------|----------------|
| **RTO** (Recovery Time Objective) | Maximum acceptable downtime | 1 hour - 24 hours |
| **RPO** (Recovery Point Objective) | Maximum acceptable data loss | 1 hour - 24 hours |
| **MTTR** (Mean Time To Recovery) | Average actual recovery time | <4 hours |
| **MTBF** (Mean Time Between Failures) | Average time between disasters | >1 year |

---

## 🏗️ ML System DR Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Primary Region (Active)                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Model Serving  │  Feature Store  │  Training Pipeline    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              │ Continuous Replication            │
│                              ▼                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DR Region (Standby)                           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Model Serving  │  Feature Store  │  Training Pipeline    │ │
│  │   (Standby)     │   (Replica)     │   (Cold)              │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backup Storage (S3/GCS)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Models    │  │  Training   │  │   Feature   │            │
│  │  (Versioned)│  │    Data     │  │ Definitions │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What to Backup

### ML-Specific Components

| Component | Backup Frequency | Retention | Priority |
|-----------|-----------------|-----------|----------|
| **Production models** | On each deployment | 90 days | Critical |
| **Model registry** | Continuous | 1 year | Critical |
| **Training data** | Daily | 1 year | High |
| **Feature definitions** | On change | Forever | Critical |
| **Feature store data** | Continuous | 30 days | High |
| **Experiment tracking** | Daily | 1 year | Medium |
| **Prediction logs** | Daily | 90 days | Medium |
| **Configuration** | On change | Forever | Critical |

### Backup Strategy Implementation

```python
import boto3
from datetime import datetime, timedelta
import json

class MLDisasterRecoveryManager:
    """
    Manages backups and recovery for ML systems.
    """
    
    def __init__(self, primary_bucket: str, dr_bucket: str):
        self.s3 = boto3.client('s3')
        self.primary_bucket = primary_bucket
        self.dr_bucket = dr_bucket
    
    def backup_model(self, model_name: str, model_version: str, 
                     model_path: str):
        """Backup model to DR storage."""
        backup_key = f"models/{model_name}/{model_version}/{datetime.utcnow().isoformat()}"
        
        # Copy model artifacts
        self.s3.copy_object(
            CopySource={'Bucket': self.primary_bucket, 'Key': model_path},
            Bucket=self.dr_bucket,
            Key=backup_key
        )
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_version': model_version,
            'backup_time': datetime.utcnow().isoformat(),
            'source_path': model_path
        }
        self.s3.put_object(
            Bucket=self.dr_bucket,
            Key=f"{backup_key}/metadata.json",
            Body=json.dumps(metadata)
        )
        
        return backup_key
    
    def backup_feature_store(self, feature_store_client):
        """Backup feature store definitions and recent data."""
        backup_time = datetime.utcnow().isoformat()
        
        # Backup feature definitions
        definitions = feature_store_client.get_all_feature_definitions()
        self.s3.put_object(
            Bucket=self.dr_bucket,
            Key=f"feature_store/definitions/{backup_time}.json",
            Body=json.dumps(definitions)
        )
        
        # Backup recent feature data (last 24 hours)
        for feature_group in feature_store_client.list_feature_groups():
            data = feature_store_client.export_features(
                feature_group,
                start_time=datetime.utcnow() - timedelta(days=1)
            )
            self.s3.put_object(
                Bucket=self.dr_bucket,
                Key=f"feature_store/data/{feature_group}/{backup_time}.parquet",
                Body=data
            )
    
    def backup_training_data(self, data_path: str, dataset_name: str):
        """Backup training data with versioning."""
        backup_key = f"training_data/{dataset_name}/{datetime.utcnow().isoformat()}"
        
        # For large datasets, use multipart upload
        self._multipart_copy(
            source_bucket=self.primary_bucket,
            source_key=data_path,
            dest_bucket=self.dr_bucket,
            dest_key=backup_key
        )
        
        return backup_key
    
    def restore_model(self, model_name: str, version: str = "latest") -> str:
        """Restore model from DR backup."""
        # Find backup
        if version == "latest":
            backups = self._list_backups(f"models/{model_name}/")
            backup_key = sorted(backups)[-1]  # Most recent
        else:
            backup_key = f"models/{model_name}/{version}"
        
        # Restore to primary
        restore_path = f"restored/{model_name}/{datetime.utcnow().isoformat()}"
        self.s3.copy_object(
            CopySource={'Bucket': self.dr_bucket, 'Key': backup_key},
            Bucket=self.primary_bucket,
            Key=restore_path
        )
        
        return restore_path
    
    def full_system_restore(self):
        """
        Full DR procedure - restore entire ML system.
        """
        restore_log = []
        
        # 1. Restore latest models
        for model in self._list_production_models():
            path = self.restore_model(model)
            restore_log.append(f"Restored model {model} to {path}")
        
        # 2. Restore feature definitions
        self._restore_feature_definitions()
        restore_log.append("Restored feature definitions")
        
        # 3. Restore feature data
        self._restore_feature_data()
        restore_log.append("Restored feature data")
        
        # 4. Verify system health
        health = self._verify_system_health()
        restore_log.append(f"System health: {health}")
        
        return restore_log
```

---

## 📋 DR Runbook Template

### Model Rollback Procedure

```markdown
## Model Rollback Runbook

### Trigger Conditions
- Model accuracy dropped >10%
- Model latency increased >100%
- Critical bug in model logic

### Steps

1. **Assess Impact**
   - [ ] Check error rates in monitoring
   - [ ] Identify affected users/traffic percentage
   - [ ] Determine rollback model version

2. **Execute Rollback**
   ```bash
   # Get previous stable version
   MODEL_VERSION=$(kubectl get deployment model-server -o jsonpath='{.metadata.annotations.previous-version}')
   
   # Update deployment
   kubectl set image deployment/model-server model=model-server:$MODEL_VERSION
   
   # Verify rollout
   kubectl rollout status deployment/model-server
   ```

3. **Verify Recovery**
   - [ ] Check prediction latency
   - [ ] Check error rates
   - [ ] Verify accuracy metrics
   - [ ] Confirm with stakeholders

4. **Post-Incident**
   - [ ] Document incident
   - [ ] Root cause analysis
   - [ ] Update deployment process
```

### Full DR Procedure

```markdown
## Full Disaster Recovery Runbook

### Trigger Conditions
- Primary region completely unavailable
- Data corruption in primary
- Security breach requiring clean environment

### Phase 1: Assessment (Target: 15 minutes)
- [ ] Confirm disaster scope
- [ ] Notify stakeholders
- [ ] Activate DR team

### Phase 2: Failover to DR Region (Target: 1 hour)
1. [ ] Update DNS to DR region
2. [ ] Verify DR infrastructure is running
3. [ ] Restore latest model backups
4. [ ] Restore feature store from backup
5. [ ] Verify prediction service health

### Phase 3: Validation (Target: 30 minutes)
- [ ] Run smoke tests
- [ ] Verify prediction accuracy
- [ ] Check latency metrics
- [ ] Monitor error rates

### Phase 4: Communication
- [ ] Update status page
- [ ] Notify customers if applicable
- [ ] Internal status updates every 30 minutes

### Recovery Verification Checklist
- [ ] All models loaded and serving
- [ ] Feature store accessible
- [ ] Prediction latency <100ms
- [ ] Error rate <0.1%
- [ ] All monitoring systems operational
```

---

## 🧪 DR Testing

```python
class DRTestFramework:
    """Framework for testing disaster recovery procedures."""
    
    def __init__(self, dr_manager: MLDisasterRecoveryManager):
        self.dr_manager = dr_manager
    
    def test_model_restore(self, model_name: str) -> dict:
        """Test model restore procedure."""
        results = {'passed': True, 'steps': []}
        
        # Backup current model
        backup_key = self.dr_manager.backup_model(
            model_name, 
            "test_backup",
            f"models/{model_name}/current"
        )
        results['steps'].append(f"Backup created: {backup_key}")
        
        # Restore model
        restore_path = self.dr_manager.restore_model(model_name, "test_backup")
        results['steps'].append(f"Model restored to: {restore_path}")
        
        # Verify model works
        try:
            model = load_model(restore_path)
            prediction = model.predict(get_test_input())
            results['steps'].append(f"Model prediction successful")
        except Exception as e:
            results['passed'] = False
            results['steps'].append(f"Model verification failed: {e}")
        
        return results
    
    def test_full_dr(self) -> dict:
        """Test full disaster recovery procedure."""
        results = {'passed': True, 'rto_minutes': 0, 'steps': []}
        
        start_time = time.time()
        
        # Simulate disaster (in test environment)
        self._simulate_primary_failure()
        
        # Execute DR
        restore_log = self.dr_manager.full_system_restore()
        results['steps'].extend(restore_log)
        
        # Measure RTO
        results['rto_minutes'] = (time.time() - start_time) / 60
        
        # Verify system
        health = self._verify_system_health()
        results['passed'] = health['all_checks_passed']
        
        return results
    
    def scheduled_dr_drill(self):
        """Quarterly DR drill."""
        # Run all tests
        model_test = self.test_model_restore("production_model")
        feature_test = self.test_feature_store_restore()
        
        # Generate report
        report = {
            'date': datetime.utcnow().isoformat(),
            'model_restore': model_test,
            'feature_restore': feature_test,
            'overall_passed': model_test['passed'] and feature_test['passed']
        }
        
        # Send report
        self._send_dr_report(report)
        
        return report
```

---

## ✅ Best Practices

1. **Automate backups** - never rely on manual processes
2. **Test DR regularly** - quarterly at minimum
3. **Document everything** - runbooks, contacts, procedures
4. **Define RTO/RPO** - and measure against them
5. **Multi-region backups** - don't keep backups in same region
6. **Version everything** - models, data, configs
7. **Practice drills** - involve the whole team
8. **Update after incidents** - improve based on learnings

---

## 📊 DR Checklist

- [ ] RTO and RPO defined and documented
- [ ] All critical components backed up automatically
- [ ] Backups stored in separate region
- [ ] Recovery procedures documented
- [ ] DR drills conducted quarterly
- [ ] Team trained on procedures
- [ ] Monitoring for backup failures
- [ ] Contact list for DR activation

---

## 🔗 Related Topics

- [High Availability](./01-high-availability.md) - Prevent disasters
- [Graceful Degradation](./02-graceful-degradation.md) - Handle partial failures
- [Circuit Breakers](./03-circuit-breakers.md) - Prevent cascade failures
