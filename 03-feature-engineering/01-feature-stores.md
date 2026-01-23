# Feature Stores

## Overview

Feature stores are centralized systems for storing, managing, and serving features for ML models. They solve the problem of feature inconsistency between training and inference.

---

## 🎯 What is a Feature Store?

A feature store is a system that:
- **Stores** feature definitions and computed features
- **Serves** features for both training and inference
- **Manages** feature versioning and metadata
- **Monitors** feature quality and drift

---

## 🏗️ Feature Store Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│              Feature Store                               │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Feature Registry                         │  │
│  │  (Definitions, Metadata, Lineage)               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Offline Store   │      │   Online Store    │       │
│  │  (Training)      │      │   (Inference)     │       │
│  │                  │      │                   │       │
│  │  - Data Lake     │      │  - Redis          │       │
│  │  - Data          │      │  - DynamoDB       │       │
│  │    Warehouse     │      │  - Cassandra      │       │
│  └──────────────────┘      └──────────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Feature Serving API                      │  │
│  │  (REST, gRPC, SDK)                               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features

### 1. Feature Registry

**Purpose:** Centralized feature definitions

**Contains:**
- Feature schemas
- Computation logic
- Data sources
- Dependencies
- Metadata

**Benefits:**
- Feature discovery
- Documentation
- Lineage tracking
- Reusability

---

### 2. Offline Store

**Purpose:** Store features for training

**Characteristics:**
- Historical data
- Point-in-time correctness
- Batch access
- Large datasets

**Storage:**
- Data lakes (S3, GCS)
- Data warehouses (Snowflake, BigQuery)
- Parquet files

**Use Cases:**
- Model training
- Feature analysis
- Backtesting
- Experimentation

---

### 3. Online Store

**Purpose:** Serve features for inference

**Characteristics:**
- Low latency (<10ms)
- Real-time access
- Current values
- Key-value access

**Storage:**
- Redis
- DynamoDB
- Cassandra
- In-memory databases

**Use Cases:**
- Real-time inference
- Online predictions
- Feature serving

---

### 4. Feature Serving API

**Purpose:** Unified interface for feature access

**APIs:**
- REST API
- gRPC
- Python SDK
- SQL interface

**Features:**
- Batch and point lookups
- Feature joins
- Versioning support
- Caching

---

## 🛠️ Feature Store Tools

### 1. Feast

**Type:** Open-source feature store

**Features:**
- Offline and online stores
- Feature registry
- Point-in-time correctness
- Multiple backends

**Pros:**
- Open source
- Flexible
- Good documentation
- Active community

**Cons:**
- Self-hosted
- Setup complexity
- Less managed

**Usage:**
```python
from feast import FeatureStore

# Initialize
store = FeatureStore(repo_path=".")

# Get online features
features = store.get_online_features(
    entity_rows=[{"user_id": 123}],
    features=["user_features:age", "user_features:total_purchases"]
)

# Get offline features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:age", "user_features:total_purchases"]
)
```

---

### 2. Tecton

**Type:** Managed feature platform

**Features:**
- Managed infrastructure
- Real-time features
- Feature monitoring
- Enterprise features

**Pros:**
- Fully managed
- Easy to use
- Good performance
- Enterprise support

**Cons:**
- Commercial
- Vendor lock-in
- Less flexible

---

### 3. Hopsworks

**Type:** Open-source feature store platform

**Features:**
- Feature store
- Model registry
- Experiment tracking
- MLOps platform

**Pros:**
- Comprehensive platform
- Open source
- Good integration
- Active development

**Cons:**
- Complex setup
- Learning curve
- Resource intensive

---

## 📊 Feature Store Workflow

### 1. Feature Definition

```python
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource

# Define entity
user = Entity(name="user_id", value_type=ValueType.INT64)

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),
    features=[
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="total_purchases", dtype=ValueType.INT64),
        Feature(name="avg_order_value", dtype=ValueType.FLOAT),
    ],
    source=user_stats_source
)
```

---

### 2. Feature Computation

```python
# Compute features
def compute_user_features(user_id):
    # Fetch raw data
    user_data = fetch_user_data(user_id)
    
    # Compute features
    features = {
        "age": calculate_age(user_data["birth_date"]),
        "total_purchases": count_purchases(user_id),
        "avg_order_value": calculate_avg_order_value(user_id)
    }
    
    return features

# Write to feature store
store.write_to_online_store("user_stats", features)
```

---

### 3. Feature Serving

```python
# Online serving (inference)
features = store.get_online_features(
    entity_rows=[{"user_id": 123}],
    features=["user_stats:age", "user_stats:total_purchases"]
)

# Offline serving (training)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_stats:age", "user_stats:total_purchases"]
)
```

---

## ✅ Best Practices

### 1. Feature Naming
- Use consistent naming conventions
- Include feature group prefix
- Document feature meanings
- Version features

**Example:**
```
user_stats:age
user_stats:total_purchases
product_features:price
product_features:category
```

---

### 2. Feature Versioning
- Version feature definitions
- Track feature changes
- Support multiple versions
- Deprecate old versions

---

### 3. Feature Documentation
- Document feature definitions
- Include data sources
- Explain computation logic
- Provide examples

---

### 4. Feature Monitoring
- Monitor feature freshness
- Track feature distributions
- Detect feature drift
- Alert on issues

---

### 5. Feature Reusability
- Share features across models
- Avoid duplication
- Create feature groups
- Document dependencies

---

## 🎯 Use Cases

### 1. Consistent Features
- Same features in training and inference
- Avoid feature skew
- Ensure reproducibility

### 2. Feature Reuse
- Share features across models
- Reduce computation
- Maintain consistency

### 3. Real-time Features
- Low-latency feature serving
- Real-time inference
- Online predictions

### 4. Feature Discovery
- Find existing features
- Understand feature definitions
- Track feature lineage

---

## 🛠️ Complete Feature Store Implementation

### Building a Custom Feature Store

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import redis
import pyarrow.parquet as pq
from abc import ABC, abstractmethod

@dataclass
class FeatureDefinition:
    name: str
    dtype: str
    description: str
    entity: str
    computation_fn: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    ttl_seconds: int = 3600
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class FeatureView:
    name: str
    entities: List[str]
    features: List[FeatureDefinition]
    source: str
    ttl: timedelta
    online: bool = True
    offline: bool = True

class OnlineStore(ABC):
    @abstractmethod
    def get(self, entity_key: str, feature_names: List[str]) -> Dict[str, Any]: pass
    
    @abstractmethod
    def put(self, entity_key: str, features: Dict[str, Any], ttl: int): pass

class OfflineStore(ABC):
    @abstractmethod
    def get_historical(self, entity_df: pd.DataFrame, feature_names: List[str], 
                      timestamp_col: str) -> pd.DataFrame: pass
    
    @abstractmethod
    def write(self, df: pd.DataFrame, feature_view: str): pass

class RedisOnlineStore(OnlineStore):
    """Redis-based online feature store."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def get(self, entity_key: str, feature_names: List[str]) -> Dict[str, Any]:
        """Get features for an entity."""
        pipeline = self.client.pipeline()
        for feature in feature_names:
            pipeline.hget(f"features:{entity_key}", feature)
        
        results = pipeline.execute()
        return {name: self._deserialize(val) for name, val in zip(feature_names, results)}
    
    def put(self, entity_key: str, features: Dict[str, Any], ttl: int = 3600):
        """Store features for an entity."""
        key = f"features:{entity_key}"
        serialized = {k: self._serialize(v) for k, v in features.items()}
        
        pipeline = self.client.pipeline()
        pipeline.hset(key, mapping=serialized)
        pipeline.expire(key, ttl)
        pipeline.execute()
    
    def _serialize(self, value: Any) -> str:
        import json
        return json.dumps(value)
    
    def _deserialize(self, value: str) -> Any:
        import json
        return json.loads(value) if value else None

class ParquetOfflineStore(OfflineStore):
    """Parquet-based offline feature store."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def get_historical(self, entity_df: pd.DataFrame, feature_names: List[str],
                      timestamp_col: str = 'event_timestamp') -> pd.DataFrame:
        """Get historical features with point-in-time correctness."""
        results = []
        
        for feature_view in self._get_feature_views(feature_names):
            path = f"{self.base_path}/{feature_view}"
            feature_df = pd.read_parquet(path)
            
            # Point-in-time join
            joined = self._asof_join(
                entity_df,
                feature_df,
                timestamp_col
            )
            results.append(joined)
        
        # Merge all feature dataframes
        final_df = entity_df.copy()
        for result_df in results:
            final_df = final_df.merge(result_df, on=['entity_id', timestamp_col])
        
        return final_df
    
    def write(self, df: pd.DataFrame, feature_view: str):
        """Write features to offline store."""
        path = f"{self.base_path}/{feature_view}"
        df.to_parquet(path, partition_cols=['date'])
    
    def _asof_join(self, left: pd.DataFrame, right: pd.DataFrame,
                   timestamp_col: str) -> pd.DataFrame:
        """Perform as-of join for point-in-time correctness."""
        return pd.merge_asof(
            left.sort_values(timestamp_col),
            right.sort_values(timestamp_col),
            on=timestamp_col,
            by='entity_id',
            direction='backward'
        )

class FeatureStore:
    """Complete feature store implementation."""
    
    def __init__(self, online_store: OnlineStore, offline_store: OfflineStore):
        self.online = online_store
        self.offline = offline_store
        self.feature_views: Dict[str, FeatureView] = {}
        self.registry: Dict[str, FeatureDefinition] = {}
    
    def register_feature_view(self, feature_view: FeatureView):
        """Register a feature view."""
        self.feature_views[feature_view.name] = feature_view
        for feature in feature_view.features:
            self.registry[f"{feature_view.name}:{feature.name}"] = feature
    
    def get_online_features(self, entity_rows: List[Dict],
                           features: List[str]) -> Dict[str, List]:
        """Get features for real-time inference."""
        results = {feat: [] for feat in features}
        
        for entity in entity_rows:
            entity_key = self._get_entity_key(entity)
            feature_names = [f.split(':')[1] for f in features]
            
            values = self.online.get(entity_key, feature_names)
            
            for feat, val in zip(features, feature_names):
                results[feat].append(values.get(val))
        
        return results
    
    def get_historical_features(self, entity_df: pd.DataFrame,
                               features: List[str]) -> pd.DataFrame:
        """Get historical features for training."""
        feature_names = [f.split(':')[1] for f in features]
        return self.offline.get_historical(entity_df, feature_names)
    
    def materialize(self, feature_view_name: str, start_date: datetime,
                   end_date: datetime):
        """Materialize features from offline to online store."""
        feature_view = self.feature_views[feature_view_name]
        
        # Read from offline store
        df = pd.read_parquet(f"{self.offline.base_path}/{feature_view_name}")
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Get latest value per entity
        latest = df.sort_values('timestamp').groupby('entity_id').last().reset_index()
        
        # Write to online store
        for _, row in latest.iterrows():
            entity_key = row['entity_id']
            features = {f.name: row[f.name] for f in feature_view.features if f.name in row}
            self.online.put(entity_key, features, int(feature_view.ttl.total_seconds()))
    
    def _get_entity_key(self, entity: Dict) -> str:
        """Generate entity key from entity dict."""
        return ':'.join(str(v) for v in entity.values())

# Usage
online = RedisOnlineStore(host='localhost', port=6379)
offline = ParquetOfflineStore(base_path='s3://bucket/features')
store = FeatureStore(online, offline)

# Register feature view
user_features = FeatureView(
    name="user_features",
    entities=["user_id"],
    features=[
        FeatureDefinition(name="age", dtype="int", description="User age", entity="user"),
        FeatureDefinition(name="total_purchases", dtype="float", description="Total purchases", entity="user"),
        FeatureDefinition(name="avg_order_value", dtype="float", description="Average order value", entity="user"),
    ],
    source="user_events",
    ttl=timedelta(hours=24)
)
store.register_feature_view(user_features)

# Get online features
features = store.get_online_features(
    entity_rows=[{"user_id": 123}, {"user_id": 456}],
    features=["user_features:age", "user_features:total_purchases"]
)

# Get historical features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:age", "user_features:total_purchases"]
)
```

---

### Feast Production Setup

```python
from feast import FeatureStore, Entity, Feature, FeatureView, ValueType, FileSource
from feast.infra.offline_stores.file_source import FileSource
from datetime import timedelta
import pandas as pd

# Define entities
user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique user identifier"
)

# Define feature sources
user_source = FileSource(
    path="s3://bucket/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Define feature views
user_profile_fv = FeatureView(
    name="user_profile",
    entities=["user_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="gender", dtype=ValueType.STRING),
        Feature(name="account_age_days", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=user_source,
    tags={"team": "user", "priority": "high"}
)

user_behavior_fv = FeatureView(
    name="user_behavior",
    entities=["user_id"],
    ttl=timedelta(hours=1),  # More frequent updates
    features=[
        Feature(name="purchases_last_7d", dtype=ValueType.INT64),
        Feature(name="avg_order_value_30d", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_purchase", dtype=ValueType.INT64),
        Feature(name="favorite_category", dtype=ValueType.STRING),
    ],
    online=True,
    batch_source=FileSource(
        path="s3://bucket/user_behavior.parquet",
        timestamp_field="event_timestamp"
    ),
    tags={"team": "behavior", "priority": "high"}
)

# Production deployment script
class FeastDeployment:
    def __init__(self, repo_path: str):
        self.store = FeatureStore(repo_path=repo_path)
    
    def apply_feature_views(self):
        """Apply feature view definitions."""
        self.store.apply([
            user_entity,
            user_profile_fv,
            user_behavior_fv
        ])
    
    def materialize_features(self, days_back: int = 7):
        """Materialize features to online store."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.store.materialize(
            feature_views=["user_profile", "user_behavior"],
            start_date=start_date,
            end_date=end_date
        )
    
    def get_training_data(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """Get training data with all features."""
        return self.store.get_historical_features(
            entity_df=entity_df,
            features=[
                "user_profile:age",
                "user_profile:gender",
                "user_profile:account_age_days",
                "user_behavior:purchases_last_7d",
                "user_behavior:avg_order_value_30d",
                "user_behavior:days_since_last_purchase",
            ]
        ).to_df()
    
    def serve_features(self, user_ids: List[int]) -> Dict:
        """Serve features for inference."""
        entity_rows = [{"user_id": uid} for uid in user_ids]
        
        return self.store.get_online_features(
            entity_rows=entity_rows,
            features=[
                "user_profile:age",
                "user_behavior:purchases_last_7d",
                "user_behavior:avg_order_value_30d",
            ]
        ).to_dict()
```

---

## 🎯 Interview Questions

**Q1: Why use a feature store instead of computing features at inference time?**

**Answer:**
```
1. Consistency: Same features in training and inference
2. Performance: Pre-computed features, low latency
3. Reusability: Share features across models
4. Governance: Central feature documentation
5. Monitoring: Track feature quality in one place

Trade-off: More infrastructure complexity

Without feature store:
- Training: SQL query → compute features → train
- Inference: API call → compute features (different code!) → predict
  ↑ Risk of training-serving skew

With feature store:
- Training: Feature store → train
- Inference: Feature store → predict
  ↑ Same features guaranteed
```

**Q2: How do you handle point-in-time correctness?**

**Answer:**
- Use **as-of joins** to get feature values at specific timestamps
- Store **event timestamps** with all features
- Avoid **data leakage** by only using features available at prediction time
- Implement **backfill** for historical features

**Q3: Design a feature store for a recommendation system with 100M users.**

**Answer:**
```
Architecture:
├── Online Store: Redis Cluster (sharded by user_id)
│   ├── User features: 100M keys × 1KB ≈ 100GB
│   ├── Item features: 10M items × 2KB ≈ 20GB
│   └── Real-time features: Session data
│
├── Offline Store: Delta Lake on S3
│   ├── Historical user features (years of data)
│   ├── Item catalog with embeddings
│   └── Interaction logs for training
│
└── Feature Computation:
    ├── Batch: Daily user aggregations (Spark)
    ├── Streaming: Real-time session features (Flink)
    └── On-demand: Similar items (cached)
```

---

## 🔑 Key Takeaways

1. **Feature stores ensure consistency** - same features in training and inference
2. **Separate offline and online stores** - different requirements
3. **Use appropriate tools** - Feast, Tecton, Hopsworks
4. **Monitor features** - track quality and drift
5. **Document features** - enable discovery and reuse
6. **Point-in-time correctness** - critical for avoiding data leakage
7. **Materialization strategy** - balance freshness vs cost

---

## 📚 Further Reading

- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store Guide](https://www.featurestore.org/)
- [Building a Feature Store](https://www.oreilly.com/library/view/building-machine-learning/9781492045100/)
- [Feature Stores for ML](https://www.tecton.ai/blog/what-is-a-feature-store/)

---

## 🔗 Related Topics

- [Online vs Offline Features](./online-vs-offline-features.md)
- [Feature Pipelines](./feature-pipelines.md)
- [Feature Monitoring](./feature-monitoring.md)
