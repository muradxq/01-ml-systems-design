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

## 🔑 Key Takeaways

1. **Offline for training** - historical, batch, point-in-time
2. **Online for inference** - current, real-time, low latency
3. **Keep consistent** - same definitions, synchronized values
4. **Use feature stores** - manage both efficiently
5. **Monitor both** - track quality and performance

---

## 📚 Further Reading

- [Feast Documentation](https://docs.feast.dev/)
- [Online vs Offline Features](https://www.featurestore.org/online-vs-offline-features)

---

## 🔗 Related Topics

- [Feature Stores](./feature-stores.md)
- [Feature Pipelines](./feature-pipelines.md)
- [Feature Monitoring](./feature-monitoring.md)
