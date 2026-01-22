# Data Versioning

## Overview

Data versioning is critical for ML reproducibility. It allows you to track data changes, reproduce experiments, and debug issues.

---

## 🎯 Why Version Data?

### 1. Reproducibility
- Reproduce model training results
- Debug model performance changes
- Compare experiments fairly

### 2. Traceability
- Track which data was used for training
- Understand model lineage
- Audit data changes

### 3. Collaboration
- Share datasets with team
- Work with consistent data versions
- Avoid data conflicts

### 4. Debugging
- Identify data-related issues
- Compare data versions
- Isolate problems

---

## 📊 Versioning Strategies

### 1. Snapshot Versioning

**Approach:** Store complete copies of datasets

**How it Works:**
- Each version is a complete snapshot
- Versions are immutable
- Access by version tag or timestamp

**Pros:**
- Simple to implement
- Easy to access any version
- No dependencies

**Cons:**
- Storage intensive
- Slow for large datasets
- Expensive

**Tools:**
- DVC (Data Version Control)
- Git LFS
- S3 versioning

**Example:**
```
data/
  v1/
    train.csv
    test.csv
  v2/
    train.csv
    test.csv
```

---

### 2. Delta Versioning

**Approach:** Store only changes between versions

**How it Works:**
- Store base version
- Store deltas (additions, deletions, modifications)
- Reconstruct versions from base + deltas

**Pros:**
- Storage efficient
- Fast for small changes
- Scales well

**Cons:**
- More complex
- Slower reconstruction
- Requires delta management

**Tools:**
- Delta Lake
- Apache Hudi
- Custom implementations

**Example:**
```
data/
  base/
    train.csv
  deltas/
    v1_to_v2.parquet  (new rows)
    v2_to_v3.parquet  (modified rows)
```

---

### 3. Time-Travel Versioning

**Approach:** Use timestamps for versioning

**How it Works:**
- Data stored with timestamps
- Query by time point
- Automatic versioning

**Pros:**
- Natural versioning
- Easy to query
- Built into some systems

**Cons:**
- Requires timestamp support
- May not capture all changes
- Timezone considerations

**Tools:**
- Delta Lake (time travel)
- BigQuery (time travel)
- Databricks (time travel)

**Example:**
```sql
-- Query data as of specific time
SELECT * FROM table
FOR SYSTEM_TIME AS OF '2024-01-15 10:00:00'
```

---

### 4. Hash-Based Versioning

**Approach:** Use content hashes for versioning

**How it Works:**
- Compute hash of data content
- Use hash as version identifier
- Deduplicate identical content

**Pros:**
- Automatic deduplication
- Content-based identification
- Efficient storage

**Cons:**
- Hash collisions (rare)
- Requires hash computation
- Less human-readable

**Tools:**
- DVC (uses content hashes)
- Git (uses content hashes)
- Custom implementations

**Example:**
```
data/
  a1b2c3d4/  (hash of content)
    train.csv
  e5f6g7h8/  (different hash)
    train.csv
```

---

## 🛠️ Versioning Tools

### 1. DVC (Data Version Control)

**Purpose:** Version control for data and ML models

**Features:**
- Git-like interface
- Content-addressable storage
- Remote storage support
- Pipeline versioning

**Usage:**
```bash
# Initialize DVC
dvc init

# Add data
dvc add data/train.csv

# Commit to Git
git add data/train.csv.dvc
git commit -m "Add training data"

# Push to remote
dvc push
```

**Pros:**
- Git integration
- Efficient storage
- Pipeline support
- Remote storage

**Cons:**
- Requires Git
- Learning curve
- Setup complexity

---

### 2. MLflow

**Purpose:** ML lifecycle management

**Features:**
- Experiment tracking
- Model registry
- Data versioning (via artifacts)
- Model versioning

**Usage:**
```python
import mlflow

with mlflow.start_run():
    # Log data
    mlflow.log_artifact("data/train.csv", "data")
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

**Pros:**
- Integrated with ML workflow
- Model versioning
- Experiment tracking
- UI for visualization

**Cons:**
- Less focused on data
- Requires MLflow server
- Learning curve

---

### 3. Delta Lake

**Purpose:** ACID transactions and versioning for data lakes

**Features:**
- ACID transactions
- Time travel
- Schema evolution
- Upserts and deletes

**Usage:**
```python
from delta import DeltaTable

# Write data
df.write.format("delta").save("data/train")

# Read specific version
df = spark.read.format("delta").option("versionAsOf", 0).load("data/train")

# Time travel
df = spark.read.format("delta").option("timestampAsOf", "2024-01-15").load("data/train")
```

**Pros:**
- ACID guarantees
- Time travel queries
- Schema evolution
- Efficient updates

**Cons:**
- Requires Spark
- Less flexible than DVC
- Learning curve

---

## 🏗️ Versioning Architecture

### Simple Versioning
```
Data Source → Versioned Storage → ML Pipeline
```

### Advanced Versioning
```
┌─────────────────────────────────────────┐
│         Data Sources                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Versioning Layer                    │
│  (DVC, MLflow, Delta Lake)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Versioned Storage                   │
│  (S3, GCS, Azure Blob)                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Metadata Store                      │
│  (Database, Registry)                    │
└─────────────────────────────────────────┘
```

---

## ✅ Best Practices

### 1. Version Everything
- Training data
- Test data
- Validation data
- Features
- Models

### 2. Use Semantic Versioning
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

**Example:**
- v1.0.0: Initial dataset
- v1.1.0: Added new features
- v1.1.1: Fixed data quality issues

### 3. Document Changes
- Changelog for each version
- Data quality reports
- Schema changes
- Known issues

### 4. Automate Versioning
- Automatic versioning on changes
- CI/CD integration
- Version tagging

### 5. Store Metadata
- Version number
- Timestamp
- Author
- Description
- Data quality metrics

---

## 📝 Versioning Workflow

### 1. Development
```bash
# Create new version
dvc add data/train.csv

# Commit version
git add data/train.csv.dvc
git commit -m "Version 1.0.0: Initial training data"

# Tag version
git tag -a v1.0.0 -m "Initial training data"
```

### 2. Production
```bash
# Use specific version
dvc checkout data/train.csv.dvc@v1.0.0

# Or in code
import dvc.api
data = dvc.api.get_url('data/train.csv', rev='v1.0.0')
```

### 3. Updates
```bash
# Update data
# ... modify data ...

# Create new version
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "Version 1.1.0: Added new features"
git tag -a v1.1.0 -m "Added new features"
```

---

## 🎯 Versioning by Data Type

### Training Data
- **Strategy**: Snapshot or delta versioning
- **Frequency**: Per training run
- **Storage**: Versioned storage (S3, GCS)
- **Tool**: DVC, MLflow

### Features
- **Strategy**: Hash-based or snapshot
- **Frequency**: Per feature pipeline run
- **Storage**: Feature store
- **Tool**: Feature store versioning

### Models
- **Strategy**: Snapshot versioning
- **Frequency**: Per model training
- **Storage**: Model registry
- **Tool**: MLflow, SageMaker Model Registry

---

## 🔑 Key Takeaways

1. **Version everything** - data, features, models
2. **Choose the right strategy** - snapshot vs delta vs time-travel
3. **Use appropriate tools** - DVC, MLflow, Delta Lake
4. **Document changes** - changelogs and metadata
5. **Automate versioning** - integrate into workflows

---

## 📚 Further Reading

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Delta Lake Documentation](https://delta.io/)

---

## 🔗 Related Topics

- [Data Collection](./data-collection.md)
- [Data Storage](./data-storage.md)
- [Data Quality](./data-quality.md)
