# Online vs Offline Features

## Overview

Understanding the difference between online and offline features is crucial for building ML systems. Each serves different purposes and has different requirements.

---

## 📊 Key Differences

| Aspect | Offline Features | Online Features |
|--------|------------------|-----------------|
| **Purpose** | Model training | Model inference |
| **Latency** | Minutes to hours | Milliseconds |
| **Data** | Historical, batch | Current, real-time |
| **Access Pattern** | Bulk reads | Point lookups |
| **Storage** | Data lakes/warehouses | Key-value stores |
| **Computation** | Batch processing | Real-time computation |
| **Point-in-time** | Required | Not required |

---

## 🔄 Offline Features

### Characteristics

**Purpose:** Features for model training

**Requirements:**
- Historical data
- Point-in-time correctness
- Batch access
- Large datasets

**Computation:**
- Batch processing
- Scheduled jobs
- ETL pipelines

**Storage:**
- Data lakes (S3, GCS)
- Data warehouses (Snowflake, BigQuery)
- Parquet files

**Access:**
- SQL queries
- Batch APIs
- File reads

---

### Use Cases

1. **Model Training**
   - Historical feature values
   - Point-in-time correctness
   - Large feature sets

2. **Feature Analysis**
   - Exploratory data analysis
   - Feature importance
   - Data quality checks

3. **Backtesting**
   - Historical predictions
   - Model evaluation
   - A/B testing analysis

4. **Experimentation**
   - Feature engineering
   - Model development
   - Hyperparameter tuning

---

### Example: Offline Feature Computation

```python
# Batch feature computation
def compute_offline_features(start_date, end_date):
    # Load historical data
    events = load_events(start_date, end_date)
    
    # Compute features for each user
    features = []
    for user_id in events['user_id'].unique():
        user_events = events[events['user_id'] == user_id]
        
        # Compute aggregated features
        user_features = {
            'user_id': user_id,
            'total_purchases': len(user_events[user_events['type'] == 'purchase']),
            'avg_order_value': user_events[user_events['type'] == 'purchase']['value'].mean(),
            'days_since_last_purchase': (datetime.now() - user_events['timestamp'].max()).days,
            'feature_timestamp': datetime.now()
        }
        features.append(user_features)
    
    # Save to offline store
    save_to_data_lake(features)
    
    return features
```

---

## ⚡ Online Features

### Characteristics

**Purpose:** Features for real-time inference

**Requirements:**
- Low latency (<10ms)
- Current values
- Point lookups
- High availability

**Computation:**
- Real-time computation
- Pre-computed values
- Streaming updates

**Storage:**
- Redis
- DynamoDB
- Cassandra
- In-memory databases

**Access:**
- Key-value lookups
- REST APIs
- gRPC

---

### Use Cases

1. **Real-time Inference**
   - Online predictions
   - API serving
   - User-facing applications

2. **Low-latency Requirements**
   - Sub-100ms predictions
   - Interactive applications
   - Real-time recommendations

3. **Current State**
   - Latest feature values
   - Real-time updates
   - Current user state

---

### Example: Online Feature Serving

```python
# Online feature serving
def get_online_features(user_id):
    # Check cache first
    cached_features = redis.get(f"features:{user_id}")
    if cached_features:
        return json.loads(cached_features)
    
    # Compute real-time features
    features = {
        'user_id': user_id,
        'current_cart_value': get_current_cart_value(user_id),
        'session_duration': get_session_duration(user_id),
        'recent_clicks': get_recent_clicks(user_id, minutes=5)
    }
    
    # Cache for future requests
    redis.setex(
        f"features:{user_id}",
        300,  # 5 minutes TTL
        json.dumps(features)
    )
    
    return features
```

---

## 🔄 Feature Synchronization

### Challenge

Offline and online features must be consistent but serve different purposes.

### Solutions

#### 1. Dual-Write Pattern

**Approach:** Write to both stores simultaneously

```python
def compute_and_store_features(user_id):
    # Compute features
    features = compute_features(user_id)
    
    # Write to offline store (for training)
    write_to_offline_store(features)
    
    # Write to online store (for inference)
    write_to_online_store(features)
```

**Pros:**
- Simple
- Consistent

**Cons:**
- Two writes
- Potential inconsistency

---

#### 2. ETL Pattern

**Approach:** Compute offline, then sync to online

```python
# Batch job computes offline features
def batch_compute_features():
    features = compute_offline_features()
    save_to_offline_store(features)
    
    # Sync to online store
    sync_to_online_store(features)

# Real-time updates for online store
def update_online_features(user_id, event):
    # Update online features in real-time
    update_online_store(user_id, event)
```

**Pros:**
- Efficient
- Clear separation

**Cons:**
- Potential lag
- More complex

---

#### 3. Feature Store Pattern

**Approach:** Use feature store to manage both

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Feature store handles both offline and online
# Offline: For training
training_df = store.get_historical_features(...)

# Online: For inference
features = store.get_online_features(...)
```

**Pros:**
- Unified interface
- Consistent features
- Less code

**Cons:**
- Requires feature store
- Learning curve

---

## 🎯 When to Use Each

### Use Offline Features When:
- Training models
- Historical analysis
- Batch processing
- Point-in-time correctness needed
- Large datasets

### Use Online Features When:
- Real-time inference
- Low latency required (<100ms)
- Current state needed
- Point lookups
- High availability required

---

## 🏗️ Architecture Patterns

### Pattern 1: Separate Computation

```
Raw Data → Offline Feature Pipeline → Offline Store (Training)
Raw Data → Online Feature Pipeline → Online Store (Inference)
```

### Pattern 2: Unified with Feature Store

```
Raw Data → Feature Computation → Feature Store
                                    ├─ Offline Store
                                    └─ Online Store
```

### Pattern 3: Hybrid Approach

```
Raw Data → Batch Pipeline → Offline Store → Sync → Online Store
Raw Data → Streaming Pipeline → Online Store (Real-time updates)
```

---

## ✅ Best Practices

### 1. Consistency
- Use same feature definitions
- Ensure feature values match
- Monitor for drift

### 2. Performance
- Optimize online features for latency
- Optimize offline features for throughput
- Use appropriate storage

### 3. Monitoring
- Track feature freshness
- Monitor latency
- Alert on inconsistencies

### 4. Versioning
- Version feature definitions
- Support multiple versions
- Track changes

---

## 🛠️ Production Implementation

### Real-Time Feature Service

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as aioredis
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class FeatureRequest:
    entity_id: str
    feature_names: List[str]
    timestamp: datetime = None

@dataclass
class FeatureResponse:
    entity_id: str
    features: Dict[str, Any]
    latency_ms: float
    cache_hit: bool

class RealTimeFeatureService:
    """High-performance real-time feature serving."""
    
    def __init__(self, redis_url: str, feature_store_client):
        self.redis_url = redis_url
        self.feature_store = feature_store_client
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._local_cache = {}  # L1 cache
        self._cache_ttl = 60  # seconds
        
    async def initialize(self):
        """Initialize async Redis connection."""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def get_features(self, request: FeatureRequest) -> FeatureResponse:
        """Get features with multi-level caching."""
        import time
        start_time = time.time()
        cache_hit = False
        
        # L1: Check local cache
        cache_key = f"{request.entity_id}:{':'.join(sorted(request.feature_names))}"
        if cache_key in self._local_cache:
            cached = self._local_cache[cache_key]
            if time.time() - cached['timestamp'] < self._cache_ttl:
                return FeatureResponse(
                    entity_id=request.entity_id,
                    features=cached['features'],
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=True
                )
        
        # L2: Check Redis cache
        redis_key = f"features:{request.entity_id}"
        cached_features = await self.redis.hgetall(redis_key)
        
        if cached_features:
            features = {k.decode(): self._deserialize(v) 
                       for k, v in cached_features.items()
                       if k.decode() in request.feature_names}
            
            if len(features) == len(request.feature_names):
                cache_hit = True
                # Update L1 cache
                self._local_cache[cache_key] = {
                    'features': features,
                    'timestamp': time.time()
                }
                return FeatureResponse(
                    entity_id=request.entity_id,
                    features=features,
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=cache_hit
                )
        
        # L3: Compute features
        features = await self._compute_features(request)
        
        # Update caches
        await self._update_caches(request.entity_id, features, cache_key)
        
        return FeatureResponse(
            entity_id=request.entity_id,
            features=features,
            latency_ms=(time.time() - start_time) * 1000,
            cache_hit=cache_hit
        )
    
    async def get_features_batch(self, requests: List[FeatureRequest]) -> List[FeatureResponse]:
        """Batch feature retrieval for efficiency."""
        tasks = [self.get_features(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    async def _compute_features(self, request: FeatureRequest) -> Dict[str, Any]:
        """Compute features from various sources."""
        features = {}
        
        # Separate features by type
        precomputed_features = []
        realtime_features = []
        
        for feature_name in request.feature_names:
            if self._is_realtime_feature(feature_name):
                realtime_features.append(feature_name)
            else:
                precomputed_features.append(feature_name)
        
        # Fetch precomputed features from feature store
        if precomputed_features:
            precomputed = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.feature_store.get_online_features(
                    entity_rows=[{"entity_id": request.entity_id}],
                    features=precomputed_features
                )
            )
            features.update(precomputed)
        
        # Compute real-time features
        if realtime_features:
            realtime = await self._compute_realtime_features(
                request.entity_id, realtime_features
            )
            features.update(realtime)
        
        return features
    
    async def _compute_realtime_features(self, entity_id: str,
                                         feature_names: List[str]) -> Dict[str, Any]:
        """Compute features that must be calculated in real-time."""
        features = {}
        
        for feature_name in feature_names:
            if feature_name == 'session_duration':
                features[feature_name] = await self._get_session_duration(entity_id)
            elif feature_name == 'recent_clicks':
                features[feature_name] = await self._get_recent_clicks(entity_id)
            elif feature_name == 'current_cart_value':
                features[feature_name] = await self._get_cart_value(entity_id)
        
        return features
    
    async def _update_caches(self, entity_id: str, features: Dict[str, Any],
                            cache_key: str):
        """Update all cache levels."""
        # Update L1
        self._local_cache[cache_key] = {
            'features': features,
            'timestamp': time.time()
        }
        
        # Update L2 (Redis)
        redis_key = f"features:{entity_id}"
        serialized = {k: self._serialize(v) for k, v in features.items()}
        await self.redis.hset(redis_key, mapping=serialized)
        await self.redis.expire(redis_key, 300)  # 5 min TTL

# Async FastAPI endpoint
from fastapi import FastAPI
app = FastAPI()

feature_service = RealTimeFeatureService(
    redis_url="redis://localhost:6379",
    feature_store_client=feast_store
)

@app.on_event("startup")
async def startup():
    await feature_service.initialize()

@app.post("/features")
async def get_features(request: FeatureRequest):
    response = await feature_service.get_features(request)
    return response
```

---

### Offline Feature Pipeline with Point-in-Time Correctness

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class OfflineFeatureComputer:
    """Compute offline features with point-in-time correctness."""
    
    def __init__(self, event_store_path: str):
        self.event_store_path = event_store_path
    
    def compute_historical_features(self, 
                                    entity_df: pd.DataFrame,
                                    feature_configs: List[Dict]) -> pd.DataFrame:
        """
        Compute historical features for training data.
        
        entity_df must have:
        - entity_id: The entity identifier
        - event_timestamp: The point in time for feature lookup
        """
        result_df = entity_df.copy()
        
        for config in feature_configs:
            feature_name = config['name']
            computation_fn = config['computation']
            lookback = config.get('lookback', timedelta(days=30))
            
            # Compute feature for each row with point-in-time correctness
            features = []
            for _, row in entity_df.iterrows():
                entity_id = row['entity_id']
                timestamp = row['event_timestamp']
                
                # Get historical data up to (but not including) event_timestamp
                historical_data = self._get_historical_data(
                    entity_id, 
                    timestamp - lookback, 
                    timestamp
                )
                
                feature_value = computation_fn(historical_data)
                features.append(feature_value)
            
            result_df[feature_name] = features
        
        return result_df
    
    def _get_historical_data(self, entity_id: str, 
                            start_time: datetime, 
                            end_time: datetime) -> pd.DataFrame:
        """Get historical events for an entity within time range."""
        # In production, this would query your event store
        events = pd.read_parquet(
            self.event_store_path,
            filters=[
                ('entity_id', '=', entity_id),
                ('timestamp', '>=', start_time),
                ('timestamp', '<', end_time)  # Strict less than for point-in-time
            ]
        )
        return events
    
    def create_training_dataset(self, 
                               labels_df: pd.DataFrame,
                               feature_configs: List[Dict]) -> pd.DataFrame:
        """
        Create a training dataset with proper temporal handling.
        
        labels_df must have:
        - entity_id: The entity
        - event_timestamp: When the label was observed
        - label: The target variable
        """
        # Compute features for each labeled example
        features_df = self.compute_historical_features(labels_df, feature_configs)
        
        # Validate no data leakage
        self._validate_no_leakage(features_df)
        
        return features_df
    
    def _validate_no_leakage(self, df: pd.DataFrame):
        """Check for potential data leakage."""
        # Check that no feature values are from the future
        # This is a simplified check - production would be more thorough
        for col in df.columns:
            if 'timestamp' in col.lower():
                feature_timestamps = pd.to_datetime(df[col])
                event_timestamps = pd.to_datetime(df['event_timestamp'])
                
                if (feature_timestamps > event_timestamps).any():
                    raise DataLeakageError(
                        f"Feature {col} contains future data!"
                    )

# Feature computation functions
def compute_purchase_count(events: pd.DataFrame) -> int:
    """Count purchases in time window."""
    return len(events[events['event_type'] == 'purchase'])

def compute_avg_order_value(events: pd.DataFrame) -> float:
    """Average order value in time window."""
    purchases = events[events['event_type'] == 'purchase']
    return purchases['amount'].mean() if len(purchases) > 0 else 0

def compute_days_since_last_activity(events: pd.DataFrame) -> int:
    """Days since last activity."""
    if len(events) == 0:
        return -1
    return (datetime.now() - events['timestamp'].max()).days

# Usage
feature_configs = [
    {
        'name': 'purchase_count_30d',
        'computation': compute_purchase_count,
        'lookback': timedelta(days=30)
    },
    {
        'name': 'avg_order_value_30d',
        'computation': compute_avg_order_value,
        'lookback': timedelta(days=30)
    },
    {
        'name': 'purchase_count_7d',
        'computation': compute_purchase_count,
        'lookback': timedelta(days=7)
    }
]

computer = OfflineFeatureComputer(event_store_path='s3://bucket/events/')
training_df = computer.create_training_dataset(labels_df, feature_configs)
```

---

## 📊 Feature Latency Optimization

### Latency Budget Allocation

```
Total Budget: 100ms for ML inference

Feature Fetching:    40ms (40%)
├── L1 Cache Hit:     < 1ms
├── L2 Cache Hit:     5-10ms
├── Feature Store:    20-30ms
└── Compute:          30-40ms

Model Inference:     30ms (30%)
Post-processing:     15ms (15%)
Network/Overhead:    15ms (15%)
```

### Optimization Strategies

| Strategy | Latency Impact | Complexity | Use Case |
|----------|----------------|------------|----------|
| In-memory cache | -90% | Low | Frequently accessed entities |
| Redis cache | -70% | Low | Moderate access patterns |
| Pre-computation | -95% | Medium | Batch-computable features |
| Feature subset | -50% | Low | Reduce feature count |
| Model distillation | -60% | High | Complex models |
| Quantization | -30% | Medium | Neural networks |

---

## 🎯 Interview Questions

**Q1: How do you prevent data leakage when creating training features?**

**Answer:**
```
Data leakage occurs when training uses information not available at prediction time.

Prevention strategies:
1. Point-in-time joins: Only use data before event_timestamp
2. Strict temporal ordering: feature_time < label_time
3. Feature timestamp validation: Check all features
4. Separate pipelines: Training can't access inference data

Example of leakage:
❌ Average of ALL user purchases (includes future)
✅ Average of purchases BEFORE this transaction

Code pattern:
events_before = events[events['timestamp'] < prediction_timestamp]
feature = compute_feature(events_before)
```

**Q2: Design a system for features that need both 7-day and 30-day aggregations.**

**Answer:**
```
Option 1: Pre-compute all windows
- Daily batch job computes 7d, 30d, 90d aggregations
- Store in feature store
- Pros: Fast serving
- Cons: Storage cost, stale within day

Option 2: Sliding window with incremental updates
- Store daily aggregates
- Compute windows dynamically: sum(last_7_daily_aggs)
- Pros: Fresh, less storage
- Cons: Computation at serving

Option 3: Hybrid
- Pre-compute stable windows (30d, 90d)
- Compute short windows (1d, 7d) with streaming
- Best of both worlds
```

**Q3: Your online features are 50ms slower than required. How do you debug and fix?**

**Answer:**
```
Debugging:
1. Add timing instrumentation at each step
2. Profile: cache lookup, feature store query, computation
3. Check cache hit rates
4. Analyze feature complexity

Common fixes:
1. Cache optimization:
   - Increase cache TTL
   - Pre-warm cache for active users
   - Use local + distributed cache

2. Feature store optimization:
   - Batch multiple entity lookups
   - Reduce feature count
   - Optimize data serialization

3. Computation optimization:
   - Move to pre-computation
   - Simplify feature logic
   - Use approximate algorithms
```

---

## 🔑 Key Takeaways

1. **Offline for training** - historical, batch, point-in-time
2. **Online for inference** - current, real-time, low latency
3. **Keep consistent** - same definitions, synchronized values
4. **Use feature stores** - manage both efficiently
5. **Monitor both** - track quality and performance
6. **Prevent leakage** - strict temporal handling
7. **Optimize latency** - caching, pre-computation, batching

---

## 📚 Further Reading

- [Feast Documentation](https://docs.feast.dev/)
- [Online vs Offline Features](https://www.featurestore.org/online-vs-offline-features)
- [Time-Aware Feature Engineering](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

## 🔗 Related Topics

- [Feature Stores](./01-feature-stores.md)
- [Feature Pipelines](./03-feature-pipelines.md)
- [Feature Monitoring](./04-feature-monitoring.md)
