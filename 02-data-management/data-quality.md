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
df = df.fillna(method='forward')  # Forward fill
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

## 🔑 Key Takeaways

1. **Quality dimensions matter** - completeness, accuracy, consistency, etc.
2. **Validate early** - catch issues at ingestion
3. **Use frameworks** - Great Expectations, Pandera, Deequ
4. **Monitor continuously** - track quality over time
5. **Automate validation** - prevent bad data from entering system

---

## 📚 Further Reading

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Data Quality Best Practices](https://www.oreilly.com/library/view/fundamentals-of-data/9781492082099/)

---

## 🔗 Related Topics

- [Data Collection](./data-collection.md)
- [Data Storage](./data-storage.md)
- [Data Versioning](./data-versioning.md)
