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

## 🛠️ Production Data Versioning System

### Complete DVC Implementation

```python
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class DataVersion:
    version_id: str
    dataset_name: str
    created_at: datetime
    size_bytes: int
    row_count: int
    schema_hash: str
    dvc_hash: str
    metadata: Dict

class DataVersionManager:
    """Production data versioning with DVC and metadata tracking."""
    
    def __init__(self, repo_path: str, remote_storage: str):
        self.repo_path = Path(repo_path)
        self.remote_storage = remote_storage
        self.metadata_file = self.repo_path / "data_versions.json"
        self._init_dvc()
    
    def _init_dvc(self):
        """Initialize DVC in repository."""
        if not (self.repo_path / ".dvc").exists():
            subprocess.run(["dvc", "init"], cwd=self.repo_path, check=True)
            subprocess.run(
                ["dvc", "remote", "add", "-d", "storage", self.remote_storage],
                cwd=self.repo_path, check=True
            )
    
    def version_dataset(self, data_path: str, dataset_name: str, 
                       metadata: Optional[Dict] = None) -> DataVersion:
        """Create a new version of a dataset."""
        import pandas as pd
        
        full_path = self.repo_path / data_path
        
        # Load data for metadata
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(full_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(full_path)
        else:
            df = None
        
        # Add to DVC
        subprocess.run(["dvc", "add", data_path], cwd=self.repo_path, check=True)
        
        # Get DVC hash
        dvc_file = full_path.with_suffix(full_path.suffix + '.dvc')
        with open(dvc_file) as f:
            dvc_content = f.read()
            dvc_hash = self._extract_dvc_hash(dvc_content)
        
        # Create version metadata
        version = DataVersion(
            version_id=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dvc_hash[:8]}",
            dataset_name=dataset_name,
            created_at=datetime.now(),
            size_bytes=full_path.stat().st_size,
            row_count=len(df) if df is not None else -1,
            schema_hash=self._compute_schema_hash(df) if df is not None else "",
            dvc_hash=dvc_hash,
            metadata=metadata or {}
        )
        
        # Save version metadata
        self._save_version_metadata(version)
        
        # Push to remote
        subprocess.run(["dvc", "push"], cwd=self.repo_path, check=True)
        
        # Commit to git
        subprocess.run(
            ["git", "add", str(dvc_file), str(self.metadata_file)],
            cwd=self.repo_path, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"Version {version.version_id}: {dataset_name}"],
            cwd=self.repo_path, check=True
        )
        subprocess.run(
            ["git", "tag", version.version_id],
            cwd=self.repo_path, check=True
        )
        
        return version
    
    def get_version(self, version_id: str) -> DataVersion:
        """Retrieve specific data version."""
        metadata = self._load_all_metadata()
        for v in metadata['versions']:
            if v['version_id'] == version_id:
                return DataVersion(**v)
        raise ValueError(f"Version {version_id} not found")
    
    def checkout_version(self, version_id: str):
        """Checkout a specific data version."""
        subprocess.run(
            ["git", "checkout", version_id],
            cwd=self.repo_path, check=True
        )
        subprocess.run(["dvc", "checkout"], cwd=self.repo_path, check=True)
    
    def list_versions(self, dataset_name: Optional[str] = None) -> List[DataVersion]:
        """List all versions, optionally filtered by dataset."""
        metadata = self._load_all_metadata()
        versions = [DataVersion(**v) for v in metadata['versions']]
        
        if dataset_name:
            versions = [v for v in versions if v.dataset_name == dataset_name]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def compare_versions(self, v1_id: str, v2_id: str) -> Dict:
        """Compare two versions of data."""
        v1 = self.get_version(v1_id)
        v2 = self.get_version(v2_id)
        
        return {
            'version_1': v1_id,
            'version_2': v2_id,
            'size_change': v2.size_bytes - v1.size_bytes,
            'row_change': v2.row_count - v1.row_count,
            'schema_changed': v1.schema_hash != v2.schema_hash,
            'time_delta': (v2.created_at - v1.created_at).total_seconds()
        }
    
    def _compute_schema_hash(self, df) -> str:
        """Compute hash of dataframe schema."""
        schema_str = str(sorted([(col, str(dtype)) for col, dtype in df.dtypes.items()]))
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _extract_dvc_hash(self, dvc_content: str) -> str:
        """Extract MD5 hash from DVC file."""
        import yaml
        dvc_data = yaml.safe_load(dvc_content)
        return dvc_data['outs'][0]['md5']
    
    def _save_version_metadata(self, version: DataVersion):
        """Save version metadata to JSON."""
        metadata = self._load_all_metadata()
        metadata['versions'].append({
            'version_id': version.version_id,
            'dataset_name': version.dataset_name,
            'created_at': version.created_at.isoformat(),
            'size_bytes': version.size_bytes,
            'row_count': version.row_count,
            'schema_hash': version.schema_hash,
            'dvc_hash': version.dvc_hash,
            'metadata': version.metadata
        })
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_all_metadata(self) -> Dict:
        """Load all version metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {'versions': []}

# Usage Example
version_manager = DataVersionManager(
    repo_path="/path/to/ml-project",
    remote_storage="s3://my-bucket/dvc-storage"
)

# Version new training data
version = version_manager.version_dataset(
    data_path="data/train.parquet",
    dataset_name="training_data",
    metadata={
        'source': 'production_db',
        'collection_date': '2024-01-15',
        'description': 'Training data with new fraud labels',
        'quality_score': 0.98
    }
)

# List versions
versions = version_manager.list_versions(dataset_name="training_data")

# Compare versions
comparison = version_manager.compare_versions("v20240115_120000_abc123", "v20240122_090000_def456")
```

---

### Lineage Tracking

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import networkx as nx

@dataclass
class DataLineageNode:
    node_id: str
    node_type: str  # 'source', 'transformation', 'dataset', 'model'
    name: str
    version: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class DataLineageEdge:
    source_id: str
    target_id: str
    transformation: str
    timestamp: datetime

class DataLineageTracker:
    """Track data lineage through ML pipelines."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, DataLineageNode] = {}
        self.edges: List[DataLineageEdge] = []
    
    def register_dataset(self, dataset_id: str, name: str, version: str,
                        metadata: Optional[Dict] = None) -> str:
        """Register a dataset in the lineage graph."""
        node = DataLineageNode(
            node_id=dataset_id,
            node_type='dataset',
            name=name,
            version=version,
            metadata=metadata or {}
        )
        self.nodes[dataset_id] = node
        self.graph.add_node(dataset_id, **vars(node))
        return dataset_id
    
    def register_transformation(self, transform_id: str, name: str,
                               inputs: List[str], outputs: List[str],
                               code_version: str) -> str:
        """Register a transformation in the lineage graph."""
        node = DataLineageNode(
            node_id=transform_id,
            node_type='transformation',
            name=name,
            version=code_version,
            metadata={'inputs': inputs, 'outputs': outputs}
        )
        self.nodes[transform_id] = node
        self.graph.add_node(transform_id, **vars(node))
        
        # Add edges from inputs to transformation
        for input_id in inputs:
            self._add_edge(input_id, transform_id, name)
        
        # Add edges from transformation to outputs
        for output_id in outputs:
            self._add_edge(transform_id, output_id, name)
        
        return transform_id
    
    def register_model(self, model_id: str, name: str, version: str,
                      training_data: List[str]) -> str:
        """Register a model in the lineage graph."""
        node = DataLineageNode(
            node_id=model_id,
            node_type='model',
            name=name,
            version=version,
            metadata={'training_data': training_data}
        )
        self.nodes[model_id] = node
        self.graph.add_node(model_id, **vars(node))
        
        # Add edges from training data to model
        for data_id in training_data:
            self._add_edge(data_id, model_id, 'model_training')
        
        return model_id
    
    def get_upstream(self, node_id: str) -> List[DataLineageNode]:
        """Get all upstream dependencies."""
        ancestors = nx.ancestors(self.graph, node_id)
        return [self.nodes[n] for n in ancestors]
    
    def get_downstream(self, node_id: str) -> List[DataLineageNode]:
        """Get all downstream dependents."""
        descendants = nx.descendants(self.graph, node_id)
        return [self.nodes[n] for n in descendants]
    
    def get_lineage_path(self, source_id: str, target_id: str) -> List[str]:
        """Get the lineage path between two nodes."""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def impact_analysis(self, node_id: str) -> Dict:
        """Analyze impact of changes to a node."""
        downstream = self.get_downstream(node_id)
        
        impact = {
            'affected_datasets': [],
            'affected_models': [],
            'affected_transformations': []
        }
        
        for node in downstream:
            if node.node_type == 'dataset':
                impact['affected_datasets'].append(node.name)
            elif node.node_type == 'model':
                impact['affected_models'].append(node.name)
            elif node.node_type == 'transformation':
                impact['affected_transformations'].append(node.name)
        
        return impact
    
    def _add_edge(self, source_id: str, target_id: str, transformation: str):
        """Add edge to lineage graph."""
        edge = DataLineageEdge(
            source_id=source_id,
            target_id=target_id,
            transformation=transformation,
            timestamp=datetime.utcnow()
        )
        self.edges.append(edge)
        self.graph.add_edge(source_id, target_id, **vars(edge))

# Usage
lineage = DataLineageTracker()

# Register data sources
lineage.register_dataset("raw_transactions", "Raw Transactions", "v1.0")
lineage.register_dataset("user_profiles", "User Profiles", "v2.1")

# Register transformation
lineage.register_transformation(
    "feature_pipeline_v1",
    "Feature Engineering Pipeline",
    inputs=["raw_transactions", "user_profiles"],
    outputs=["training_features"],
    code_version="commit_abc123"
)

lineage.register_dataset("training_features", "Training Features", "v1.0")

# Register model
lineage.register_model(
    "fraud_model_v1",
    "Fraud Detection Model",
    "v1.0",
    training_data=["training_features"]
)

# Analyze impact of data change
impact = lineage.impact_analysis("raw_transactions")
print(f"Changing raw_transactions affects: {impact}")
```

---

## 🎯 Interview Questions

**Q1: How do you ensure reproducibility when training ML models?**

**Answer:**
```
Four pillars of ML reproducibility:

1. Data Versioning (DVC, Delta Lake):
   - Track exact data used for training
   - Point-in-time correctness for features

2. Code Versioning (Git):
   - Tag releases with model versions
   - Include config files

3. Environment Versioning (Docker, Conda):
   - Lock dependencies
   - Reproducible environments

4. Experiment Tracking (MLflow):
   - Log hyperparameters
   - Log metrics and artifacts

Example workflow:
git commit -m "Feature changes"
dvc commit  # Track data changes
docker build -t model:v1.0 .
mlflow run . --experiment-name prod_training
```

**Q2: A model performed well 3 months ago but fails now. How do you investigate using versioning?**

**Answer:**
1. Get model version from 3 months ago
2. Compare **data versions** (what changed?)
3. Check **schema changes** in the data
4. Compare **feature distributions**
5. Check **code/pipeline changes**
6. Verify **environment differences**

**Q3: How would you implement data versioning for a 10TB dataset?**

**Answer:**
- Use **delta versioning** (only store changes)
- Implement **partitioned storage** (version partitions independently)
- Use **Delta Lake** or **Apache Iceberg** for table versioning
- Store **metadata only** in version control
- Use **content-addressable storage** for deduplication

---

## 🔑 Key Takeaways

1. **Version everything** - data, features, models
2. **Choose the right strategy** - snapshot vs delta vs time-travel
3. **Use appropriate tools** - DVC, MLflow, Delta Lake
4. **Document changes** - changelogs and metadata
5. **Automate versioning** - integrate into workflows
6. **Track lineage** - understand data flow and dependencies
7. **Enable reproducibility** - combine data, code, and environment versioning

---

## 📚 Further Reading

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Delta Lake Documentation](https://delta.io/)
- [Data Versioning Best Practices](https://dvc.org/doc/use-cases/versioning-data-and-model-files)

---

## 🔗 Related Topics

- [Data Collection](./data-collection.md)
- [Data Storage](./data-storage.md)
- [Data Quality](./data-quality.md)
