# Data Quality

## Overview

Data quality is fundamental to ML system success. Poor data quality leads to poor model performance, unreliable predictions, and production issues.

---

## 📊 Data Quality Dimensions

### 1. Completeness

**Definition:** Percentage of non-null values

**Metrics:**
- Missing value rate
- Record completeness
- Field completeness

**Issues:**
- Missing values
- Incomplete records
- Null fields

**Solutions:**
- Data validation
- Imputation strategies
- Default values
- Remove incomplete records

**Example:**
```python
# Check completeness
completeness = (df.notna().sum() / len(df)) * 100
print(f"Completeness: {completeness}%")

# Handle missing values
df = df.ffill()  # Forward fill
df = df.fillna(df.mean())  # Mean imputation
```

---

### 2. Accuracy

**Definition:** Data correctly represents real-world values

**Metrics:**
- Error rate
- Correctness percentage
- Validation against ground truth

**Issues:**
- Incorrect values
- Data entry errors
- Measurement errors

**Solutions:**
- Data validation rules
- Cross-validation
- Manual review
- Automated checks

**Example:**
```python
# Validate accuracy
def validate_age(age):
    return 0 <= age <= 120

def validate_email(email):
    return '@' in email and '.' in email.split('@')[1]

df['age_valid'] = df['age'].apply(validate_age)
df['email_valid'] = df['email'].apply(validate_email)
```

---

### 3. Consistency

**Definition:** Data is consistent across sources and over time

**Metrics:**
- Consistency score
- Duplicate rate
- Cross-source agreement

**Issues:**
- Duplicate records
- Conflicting values
- Inconsistent formats

**Solutions:**
- Deduplication
- Standardization
- Cross-validation
- Master data management

**Example:**
```python
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicates: {duplicates}")

# Remove duplicates
df = df.drop_duplicates()

# Standardize formats
df['phone'] = df['phone'].str.replace(r'[^\d]', '')
```

---

### 4. Validity

**Definition:** Data conforms to defined schema and rules

**Metrics:**
- Validation pass rate
- Schema compliance
- Rule violations

**Issues:**
- Schema violations
- Type mismatches
- Constraint violations

**Solutions:**
- Schema validation
- Type checking
- Constraint enforcement
- Data transformation

**Example:**
```python
from pydantic import BaseModel, validator

class UserData(BaseModel):
    user_id: int
    email: str
    age: int
    
    @validator('age')
    def validate_age(cls, v):
        if not 0 <= v <= 120:
            raise ValueError('Age must be between 0 and 120')
        return v

# Validate data
try:
    user = UserData(**row)
except ValidationError as e:
    print(f"Validation error: {e}")
```

---

### 5. Timeliness

**Definition:** Data is up-to-date and available when needed

**Metrics:**
- Data freshness
- Update frequency
- Latency

**Issues:**
- Stale data
- Delayed updates
- Outdated information

**Solutions:**
- Real-time updates
- Scheduled refreshes
- Data freshness monitoring
- TTL policies

**Example:**
```python
# Check data freshness
from datetime import datetime, timedelta

def check_freshness(timestamp, max_age_hours=24):
    age = datetime.now() - timestamp
    return age < timedelta(hours=max_age_hours)

df['is_fresh'] = df['updated_at'].apply(
    lambda x: check_freshness(x)
)
```

---

### 6. Uniqueness

**Definition:** No duplicate records

**Metrics:**
- Duplicate rate
- Unique record count
- Primary key violations

**Issues:**
- Duplicate records
- Non-unique keys
- Repeated entries

**Solutions:**
- Deduplication
- Unique constraints
- Primary key enforcement
- Merge strategies

**Example:**
```python
# Check uniqueness
unique_count = df['user_id'].nunique()
total_count = len(df)
uniqueness = unique_count / total_count
print(f"Uniqueness: {uniqueness}")

# Enforce uniqueness
df = df.drop_duplicates(subset=['user_id'])
```

---

## 🛠️ Data Quality Frameworks

### 1. Great Expectations

**Purpose:** Data validation and testing

**Features:**
- Expectation-based validation
- Data profiling
- Documentation
- Integration with pipelines

**Usage:**
```python
import great_expectations as ge

# Create expectation suite
df = ge.read_csv("data/train.csv")

# Add expectations
df.expect_column_values_to_not_be_null("user_id")
df.expect_column_values_to_be_between("age", 0, 120)
df.expect_column_values_to_match_regex("email", r".+@.+\..+")

# Validate
results = df.validate()
```

**Pros:**
- Comprehensive expectations
- Good documentation
- Pipeline integration
- Data profiling

**Cons:**
- Learning curve
- Setup complexity
- Performance overhead

---

### 2. Pandera

**Purpose:** Statistical data validation

**Features:**
- Schema validation
- Statistical checks
- Type validation
- Custom validators

**Usage:**
```python
import pandera as pa
from pandera import Column, Check

schema = pa.DataFrameSchema({
    "user_id": Column(int, checks=Check.greater_than(0)),
    "age": Column(int, checks=Check.in_range(0, 120)),
    "email": Column(str, checks=Check.str_matches(r".+@.+\..+")),
})

# Validate
df_validated = schema.validate(df)
```

**Pros:**
- Simple API
- Statistical validation
- Type checking
- Good performance

**Cons:**
- Less comprehensive than Great Expectations
- Fewer integrations

---

### 3. Deequ

**Purpose:** Data quality measurement (Spark)

**Features:**
- Constraint checking
- Data profiling
- Anomaly detection
- Metrics computation

**Usage:**
```scala
import com.amazon.deequ.checks.{Check, CheckLevel}
import com.amazon.deequ.{VerificationSuite, VerificationResult}

val verificationResult = VerificationSuite()
  .onData(df)
  .addCheck(
    Check(CheckLevel.Error, "Data Quality Check")
      .isComplete("user_id")
      .isNonNegative("age")
      .hasUniqueness("user_id", _ >= 0.95)
  )
  .run()
```

**Pros:**
- Spark integration
- Scalable
- Constraint-based
- Metrics computation

**Cons:**
- Scala/Java only
- Less flexible
- Learning curve

---

## 🏗️ Data Quality Architecture

```
┌─────────────────────────────────────────┐
│         Data Ingestion                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Schema Validation                   │
│  (Structure, types, required fields)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Rule Validation                     │
│  (Business rules, constraints)          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Statistical Validation              │
│  (Distributions, outliers, anomalies)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Quality Scoring                    │
│  (Compute quality metrics)               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Quality Monitoring                 │
│  (Track quality over time)               │
└─────────────────────────────────────────┘
```

---

## 📊 Data Quality Monitoring

### 1. Real-Time Monitoring

**Purpose:** Detect quality issues immediately

**Metrics:**
- Validation pass rate
- Error rates
- Anomaly detection
- Data freshness

**Tools:**
- Streaming validation
- Real-time alerts
- Dashboards

---

### 2. Batch Monitoring

**Purpose:** Track quality trends over time

**Metrics:**
- Quality scores
- Trend analysis
- Historical comparison
- Reports

**Tools:**
- Scheduled quality checks
- Quality dashboards
- Reports

---

### 3. Anomaly Detection

**Purpose:** Detect unusual quality issues

**Methods:**
- Statistical outliers
- Distribution shifts
- Pattern changes
- Threshold violations

**Example:**
```python
from scipy import stats

# Detect outliers using z-score
z_scores = stats.zscore(df['age'])
outliers = df[abs(z_scores) > 3]

# Detect distribution shifts
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(reference_data, new_data)
if p_value < 0.05:
    print("Distribution shift detected")
```

---

## ✅ Best Practices

### 1. Validate Early
- Validate at ingestion
- Catch issues before processing
- Fail fast on critical errors

### 2. Define Quality Standards
- Set quality thresholds
- Document expectations
- Establish SLAs

### 3. Monitor Continuously
- Track quality metrics
- Set up alerts
- Review regularly

### 4. Automate Validation
- Automated checks
- CI/CD integration
- Prevent bad data

### 5. Document Issues
- Log quality issues
- Track resolutions
- Learn from problems

---

## 🎯 Quality by Data Type

### Structured Data
- **Focus**: Schema, types, constraints
- **Tools**: Great Expectations, Pandera
- **Metrics**: Completeness, validity, consistency

### Unstructured Data
- **Focus**: Format, content, metadata
- **Tools**: Custom validators
- **Metrics**: Format compliance, content quality

### Time Series Data
- **Focus**: Timestamps, continuity, gaps
- **Tools**: Custom validators
- **Metrics**: Completeness, timeliness, consistency

---

## 🛠️ Production Data Quality Pipeline

### Complete Data Quality System

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

class QualitySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityRule:
    name: str
    check_fn: Callable
    severity: QualitySeverity
    description: str

@dataclass
class QualityIssue:
    rule_name: str
    severity: QualitySeverity
    description: str
    affected_rows: int
    details: Dict

class DataQualityEngine:
    """Production-grade data quality validation engine."""
    
    def __init__(self):
        self.rules: List[QualityRule] = []
        self.issues: List[QualityIssue] = []
        
    def add_rule(self, rule: QualityRule):
        """Register a quality rule."""
        self.rules.append(rule)
    
    def add_completeness_check(self, column: str, min_completeness: float = 0.95):
        """Add completeness check for a column."""
        def check(df: pd.DataFrame) -> Optional[QualityIssue]:
            completeness = df[column].notna().mean()
            if completeness < min_completeness:
                return QualityIssue(
                    rule_name=f"completeness_{column}",
                    severity=QualitySeverity.HIGH if completeness < 0.8 else QualitySeverity.MEDIUM,
                    description=f"Column {column} completeness {completeness:.2%} < {min_completeness:.2%}",
                    affected_rows=int(df[column].isna().sum()),
                    details={'completeness': completeness, 'threshold': min_completeness}
                )
            return None
        
        self.add_rule(QualityRule(
            name=f"completeness_{column}",
            check_fn=check,
            severity=QualitySeverity.HIGH,
            description=f"Check {column} completeness >= {min_completeness:.2%}"
        ))
    
    def add_range_check(self, column: str, min_val: float = None, max_val: float = None):
        """Add range validation for numeric column."""
        def check(df: pd.DataFrame) -> Optional[QualityIssue]:
            violations = pd.Series([False] * len(df))
            if min_val is not None:
                violations |= df[column] < min_val
            if max_val is not None:
                violations |= df[column] > max_val
            
            if violations.any():
                return QualityIssue(
                    rule_name=f"range_{column}",
                    severity=QualitySeverity.MEDIUM,
                    description=f"Column {column} has {violations.sum()} values outside range [{min_val}, {max_val}]",
                    affected_rows=int(violations.sum()),
                    details={'min': min_val, 'max': max_val, 'violation_rate': violations.mean()}
                )
            return None
        
        self.add_rule(QualityRule(
            name=f"range_{column}",
            check_fn=check,
            severity=QualitySeverity.MEDIUM,
            description=f"Check {column} in range [{min_val}, {max_val}]"
        ))
    
    def add_distribution_check(self, column: str, reference_data: pd.Series, 
                               threshold: float = 0.05):
        """Add distribution drift check."""
        def check(df: pd.DataFrame) -> Optional[QualityIssue]:
            stat, p_value = stats.ks_2samp(reference_data, df[column].dropna())
            
            if p_value < threshold:
                return QualityIssue(
                    rule_name=f"distribution_{column}",
                    severity=QualitySeverity.HIGH,
                    description=f"Column {column} distribution shifted (p={p_value:.4f})",
                    affected_rows=len(df),
                    details={'ks_statistic': stat, 'p_value': p_value}
                )
            return None
        
        self.add_rule(QualityRule(
            name=f"distribution_{column}",
            check_fn=check,
            severity=QualitySeverity.HIGH,
            description=f"Check {column} distribution matches reference"
        ))
    
    def add_uniqueness_check(self, columns: List[str], expected_unique: float = 1.0):
        """Add uniqueness check for column(s)."""
        def check(df: pd.DataFrame) -> Optional[QualityIssue]:
            uniqueness = df[columns].drop_duplicates().shape[0] / len(df)
            
            if uniqueness < expected_unique:
                return QualityIssue(
                    rule_name=f"uniqueness_{'_'.join(columns)}",
                    severity=QualitySeverity.CRITICAL if uniqueness < 0.9 else QualitySeverity.HIGH,
                    description=f"Columns {columns} uniqueness {uniqueness:.2%} < {expected_unique:.2%}",
                    affected_rows=int(len(df) - df[columns].drop_duplicates().shape[0]),
                    details={'uniqueness': uniqueness, 'expected': expected_unique}
                )
            return None
        
        self.add_rule(QualityRule(
            name=f"uniqueness_{'_'.join(columns)}",
            check_fn=check,
            severity=QualitySeverity.CRITICAL,
            description=f"Check {columns} uniqueness >= {expected_unique:.2%}"
        ))
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks and return report."""
        self.issues = []
        
        for rule in self.rules:
            try:
                issue = rule.check_fn(df)
                if issue:
                    self.issues.append(issue)
            except Exception as e:
                self.issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=QualitySeverity.CRITICAL,
                    description=f"Rule execution failed: {str(e)}",
                    affected_rows=0,
                    details={'error': str(e)}
                ))
        
        # Calculate summary
        critical_issues = sum(1 for i in self.issues if i.severity == QualitySeverity.CRITICAL)
        high_issues = sum(1 for i in self.issues if i.severity == QualitySeverity.HIGH)
        
        return {
            'passed': len(self.issues) == 0,
            'total_rules': len(self.rules),
            'failed_rules': len(self.issues),
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'issues': [vars(i) for i in self.issues],
            'quality_score': 1 - (len(self.issues) / max(len(self.rules), 1)),
            'can_proceed': critical_issues == 0
        }

# Usage Example
engine = DataQualityEngine()
engine.add_completeness_check('user_id', min_completeness=0.99)
engine.add_completeness_check('purchase_amount', min_completeness=0.95)
engine.add_range_check('age', min_val=0, max_val=120)
engine.add_range_check('purchase_amount', min_val=0)
engine.add_uniqueness_check(['transaction_id'], expected_unique=1.0)
engine.add_distribution_check('purchase_amount', reference_data=training_df['purchase_amount'])

report = engine.validate(new_data)
if not report['can_proceed']:
    raise DataQualityException(f"Critical data quality issues: {report['issues']}")
```

---

### Great Expectations Integration

```python
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import Checkpoint

class GreatExpectationsValidator:
    """Production Great Expectations setup for ML pipelines."""
    
    def __init__(self, context_path: str):
        self.context = ge.get_context(context_root_dir=context_path)
        
    def create_ml_expectation_suite(self, suite_name: str, schema: Dict):
        """Create expectation suite for ML data."""
        suite = self.context.create_expectation_suite(
            expectation_suite_name=suite_name,
            overwrite_existing=True
        )
        
        # Add expectations based on schema
        for column, config in schema.items():
            expectations = []
            
            # Type expectation
            if 'type' in config:
                expectations.append({
                    'expectation_type': 'expect_column_values_to_be_of_type',
                    'kwargs': {'column': column, 'type_': config['type']}
                })
            
            # Completeness
            if config.get('required', False):
                expectations.append({
                    'expectation_type': 'expect_column_values_to_not_be_null',
                    'kwargs': {'column': column}
                })
            
            # Range
            if 'min' in config or 'max' in config:
                expectations.append({
                    'expectation_type': 'expect_column_values_to_be_between',
                    'kwargs': {
                        'column': column,
                        'min_value': config.get('min'),
                        'max_value': config.get('max')
                    }
                })
            
            # Categorical
            if 'allowed_values' in config:
                expectations.append({
                    'expectation_type': 'expect_column_values_to_be_in_set',
                    'kwargs': {'column': column, 'value_set': config['allowed_values']}
                })
            
            for exp in expectations:
                suite.add_expectation(ge.core.ExpectationConfiguration(**exp))
        
        self.context.save_expectation_suite(suite)
        return suite
    
    def validate_training_data(self, df: pd.DataFrame, suite_name: str) -> Dict:
        """Validate training data against expectations."""
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector",
            data_asset_name="training_data",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"}
        )
        
        checkpoint = self.context.add_or_update_checkpoint(
            name="training_data_checkpoint",
            config_version=1,
            class_name="SimpleCheckpoint",
            validations=[{
                "batch_request": batch_request,
                "expectation_suite_name": suite_name
            }]
        )
        
        results = checkpoint.run()
        
        return {
            'success': results.success,
            'statistics': results.run_results[list(results.run_results.keys())[0]]['validation_result']['statistics'],
            'results_url': results.run_results[list(results.run_results.keys())[0]]['actions_results'].get('store_validation_result', {}).get('validation_result_url')
        }
```

---

## 🎯 Interview Questions

**Q1: How would you implement data quality checks in a real-time ML pipeline?**

**Answer:**
```
Strategy: Streaming validation with fallbacks

1. Pre-validation (< 1ms):
   - Schema validation
   - Type checking
   - Required field checks

2. Statistical validation (async):
   - Sample-based distribution checks
   - Aggregated anomaly detection
   - Update rolling statistics

3. Fallback handling:
   - Default values for minor issues
   - Previous valid value for features
   - Rejection + alert for critical issues

Implementation:
Request → Schema Check → Feature Check → Model
              ↓               ↓
          DLQ (reject)    Use defaults
```

**Q2: A model's performance degraded but all traditional metrics look fine. How do you investigate?**

**Answer:**
1. Check **label quality** - labeling accuracy may have degraded
2. Check **feature correlations** - relationships may have changed
3. Check **subgroup performance** - issues in specific segments
4. Check **temporal patterns** - time-based changes
5. Check **upstream data sources** - silent schema changes
6. Check **feature computation** - pipeline bugs

**Q3: How do you balance data quality with data freshness?**

**Answer:**
- Define **quality SLAs** by data type
- Implement **quality gates** with different thresholds
- Use **sampling** for expensive checks
- **Async validation** for non-blocking checks
- **Graceful degradation** - use cached data if quality check fails

---

## 📊 Data Quality Metrics Dashboard

```python
class DataQualityDashboard:
    """Track and visualize data quality metrics over time."""
    
    def __init__(self, metrics_store):
        self.metrics = metrics_store
        
    def record_quality_metrics(self, dataset: str, report: Dict):
        """Record quality metrics for trending."""
        timestamp = datetime.utcnow()
        
        metrics = {
            'dataset': dataset,
            'timestamp': timestamp,
            'quality_score': report['quality_score'],
            'completeness': report.get('completeness', {}),
            'validity': report.get('validity', {}),
            'critical_issues': report['critical_issues'],
            'high_issues': report['high_issues']
        }
        
        self.metrics.insert(metrics)
        
        # Alert on degradation
        self._check_quality_trend(dataset, metrics)
    
    def _check_quality_trend(self, dataset: str, current_metrics: Dict):
        """Check for quality degradation trends."""
        # Get last 7 days
        recent = self.metrics.query(
            dataset=dataset,
            start_time=datetime.utcnow() - timedelta(days=7)
        )
        
        if len(recent) < 3:
            return
        
        # Check for declining trend
        scores = [m['quality_score'] for m in recent]
        if all(scores[i] > scores[i+1] for i in range(len(scores)-1)):
            alert(f"Data quality declining for {dataset}: {scores}")
        
        # Check for sudden drop
        avg_score = np.mean(scores[:-1])
        if current_metrics['quality_score'] < avg_score * 0.9:
            alert(f"Sudden quality drop for {dataset}: {current_metrics['quality_score']:.2f} vs avg {avg_score:.2f}")
```

---

## 🔑 Key Takeaways

1. **Quality dimensions matter** - completeness, accuracy, consistency, etc.
2. **Validate early** - catch issues at ingestion
3. **Use frameworks** - Great Expectations, Pandera, Deequ
4. **Monitor continuously** - track quality over time
5. **Automate validation** - prevent bad data from entering system
6. **Define SLAs** - set clear quality thresholds
7. **Trend analysis** - detect gradual degradation

---

## 📚 Further Reading

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Data Quality Best Practices](https://www.oreilly.com/library/view/fundamentals-of-data/9781492082099/)
- [Data Quality at Scale](https://www.oreilly.com/library/view/data-quality-fundamentals/9781098112035/)

---

## 🔗 Related Topics

- [Data Collection](./01-data-collection.md)
- [Data Storage](./02-data-storage.md)
- [Data Versioning](./03-data-versioning.md)
