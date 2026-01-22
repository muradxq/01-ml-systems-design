# Feature Pipelines

## Overview

Feature pipelines transform raw data into features that ML models can use. Well-designed pipelines ensure consistent, reliable, and scalable feature computation.

---

## 🏗️ Pipeline Types

### 1. Batch Pipelines

**Purpose:** Compute features on historical data

**Characteristics:**
- Scheduled execution
- Large datasets
- Point-in-time correctness
- Offline features

**Use Cases:**
- Training data preparation
- Historical feature computation
- Daily/weekly feature updates

**Tools:**
- Apache Airflow, Prefect (orchestration)
- Apache Spark, Pandas (processing)
- Great Expectations (validation)

---

### 2. Streaming Pipelines

**Purpose:** Compute features on real-time data

**Characteristics:**
- Continuous processing
- Low latency
- Real-time updates
- Online features

**Use Cases:**
- Real-time feature computation
- Streaming aggregations
- Online feature updates

**Tools:**
- Apache Flink, Kafka Streams (streaming)
- Apache Kafka (messaging)
- Redis (caching)

---

### 3. Hybrid Pipelines

**Purpose:** Combine batch and streaming

**Characteristics:**
- Batch for historical
- Streaming for real-time
- Unified feature store

**Use Cases:**
- Complete feature sets
- Historical + real-time
- Lambda architecture

---

## 🏗️ Pipeline Architecture

### Simple Pipeline

```
Raw Data → Transform → Validate → Store Features
```

### Advanced Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    Raw Data                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Validation                             │
│  (Schema, Quality Checks)                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Computation                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Transform   │  │  Aggregate   │  │  Enrich      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Validation                          │
│  (Quality, Completeness, Consistency)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Storage                             │
│  ┌──────────────┐  ┌──────────────┐                   │
│  │   Offline    │  │    Online    │                   │
│  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Pipeline Design Patterns

### 1. ETL Pattern

**Extract → Transform → Load**

```python
def etl_pipeline():
    # Extract
    raw_data = extract_from_source()
    
    # Transform
    features = transform_data(raw_data)
    
    # Load
    load_to_feature_store(features)
```

---

### 2. ELT Pattern

**Extract → Load → Transform**

```python
def elt_pipeline():
    # Extract
    raw_data = extract_from_source()
    
    # Load (to data lake)
    load_to_data_lake(raw_data)
    
    # Transform (in place)
    features = transform_in_place(raw_data)
    
    # Store features
    store_features(features)
```

---

### 3. Lambda Pattern

**Batch + Streaming**

```python
# Batch layer
def batch_pipeline():
    historical_features = compute_batch_features()
    store_offline(historical_features)

# Streaming layer
def streaming_pipeline():
    real_time_features = compute_streaming_features()
    store_online(real_time_features)
```

---

## 🛠️ Implementation Examples

### Batch Pipeline with Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def extract_data():
    # Extract raw data
    data = fetch_from_database()
    return data

def transform_features(data):
    # Transform to features
    features = []
    for record in data:
        feature = {
            'user_id': record['user_id'],
            'total_purchases': compute_total_purchases(record['user_id']),
            'avg_order_value': compute_avg_order_value(record['user_id']),
            'days_since_last_purchase': compute_days_since(record['user_id'])
        }
        features.append(feature)
    return features

def load_features(features):
    # Load to feature store
    save_to_feature_store(features)

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'feature_pipeline',
    default_args=default_args,
    description='Daily feature pipeline',
    schedule_interval=timedelta(days=1)
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_features',
    python_callable=transform_features,
    op_args=[extract_task.output],
    dag=dag
)

load_task = PythonOperator(
    task_id='load_features',
    python_callable=load_features,
    op_args=[transform_task.output],
    dag=dag
)

extract_task >> transform_task >> load_task
```

---

### Streaming Pipeline with Flink

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

def streaming_feature_pipeline():
    env = StreamExecutionEnvironment.get_execution_environment()
    table_env = StreamTableEnvironment.create(env)
    
    # Define source
    table_env.execute_sql("""
        CREATE TABLE events (
            user_id BIGINT,
            event_type STRING,
            timestamp TIMESTAMP(3),
            properties MAP<STRING, STRING>
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'user-events',
            'properties.bootstrap.servers' = 'localhost:9092',
            'format' = 'json'
        )
    """)
    
    # Compute features
    table_env.execute_sql("""
        CREATE TABLE user_features AS
        SELECT
            user_id,
            COUNT(*) as total_events,
            COUNT(DISTINCT event_type) as unique_event_types,
            MAX(timestamp) as last_event_time
        FROM events
        GROUP BY user_id, TUMBLE(timestamp, INTERVAL '1' HOUR)
    """)
    
    # Sink to feature store
    table_env.execute_sql("""
        CREATE TABLE feature_store (
            user_id BIGINT,
            total_events BIGINT,
            unique_event_types INT,
            last_event_time TIMESTAMP(3)
        ) WITH (
            'connector' = 'redis',
            'host' = 'localhost',
            'port' = '6379',
            'format' = 'json'
        )
    """)
    
    table_env.execute_sql("""
        INSERT INTO feature_store
        SELECT * FROM user_features
    """)
```

---

## ✅ Best Practices

### 1. Idempotency
- Pipeline should be rerunnable
- Same input → same output
- Handle partial failures

```python
def idempotent_pipeline(date):
    # Check if already processed
    if is_already_processed(date):
        return
    
    # Process
    features = compute_features(date)
    
    # Mark as processed
    mark_as_processed(date)
```

---

### 2. Error Handling
- Retry on transient failures
- Dead letter queues
- Alert on failures
- Graceful degradation

```python
def robust_pipeline():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return process_features()
        except TransientError as e:
            if attempt == max_retries - 1:
                send_to_dlq(e)
                raise
            time.sleep(2 ** attempt)
```

---

### 3. Monitoring
- Track pipeline execution
- Monitor feature quality
- Alert on failures
- Track latency

**Metrics:**
- Pipeline success rate
- Processing time
- Feature quality scores
- Data freshness

---

### 4. Testing
- Unit tests for transformations
- Integration tests for pipelines
- Data quality tests
- End-to-end tests

```python
def test_feature_computation():
    # Test data
    test_data = create_test_data()
    
    # Compute features
    features = compute_features(test_data)
    
    # Assertions
    assert len(features) > 0
    assert 'user_id' in features[0]
    assert features[0]['total_purchases'] >= 0
```

---

### 5. Documentation
- Document feature definitions
- Explain transformations
- Document dependencies
- Provide examples

---

## 🎯 Pipeline Optimization

### 1. Parallelization
- Process multiple partitions
- Use distributed computing
- Parallel feature computation

### 2. Caching
- Cache intermediate results
- Reuse computed features
- Avoid redundant computation

### 3. Incremental Processing
- Process only new data
- Update existing features
- Avoid full recomputation

```python
def incremental_pipeline(last_processed_date):
    # Process only new data
    new_data = fetch_data_since(last_processed_date)
    new_features = compute_features(new_data)
    
    # Update feature store incrementally
    update_feature_store(new_features)
```

---

## 🔑 Key Takeaways

1. **Choose the right type** - batch vs streaming vs hybrid
2. **Design for reliability** - idempotency, error handling
3. **Monitor continuously** - track execution and quality
4. **Test thoroughly** - unit, integration, E2E tests
5. **Optimize performance** - parallelization, caching, incremental processing

---

## 📚 Further Reading

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Apache Flink Documentation](https://flink.apache.org/docs/)
- [Feature Engineering Best Practices](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

## 🔗 Related Topics

- [Feature Stores](./feature-stores.md)
- [Online vs Offline Features](./online-vs-offline-features.md)
- [Feature Monitoring](./feature-monitoring.md)
