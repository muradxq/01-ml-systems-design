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

## 🛠️ Feature Transformation Library

### Comprehensive Feature Transformations

```python
from typing import List, Dict, Union, Callable, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder
import hashlib

class FeatureTransformer:
    """Library of feature transformation functions."""
    
    def __init__(self):
        self.transformers = {}
        self.fitted = False
    
    # ============ Numerical Transformations ============
    
    def standardize(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Z-score standardization: (x - mean) / std"""
        if not self.fitted:
            self.transformers['standardize'] = {}
            for col in columns:
                scaler = StandardScaler()
                df[f'{col}_std'] = scaler.fit_transform(df[[col]])
                self.transformers['standardize'][col] = scaler
        else:
            for col in columns:
                scaler = self.transformers['standardize'][col]
                df[f'{col}_std'] = scaler.transform(df[[col]])
        return df
    
    def normalize(self, df: pd.DataFrame, columns: List[str],
                  range_min: float = 0, range_max: float = 1) -> pd.DataFrame:
        """Min-max normalization to [0, 1] or custom range."""
        if not self.fitted:
            self.transformers['normalize'] = {}
            for col in columns:
                scaler = MinMaxScaler(feature_range=(range_min, range_max))
                df[f'{col}_norm'] = scaler.fit_transform(df[[col]])
                self.transformers['normalize'][col] = scaler
        else:
            for col in columns:
                scaler = self.transformers['normalize'][col]
                df[f'{col}_norm'] = scaler.transform(df[[col]])
        return df
    
    def log_transform(self, df: pd.DataFrame, columns: List[str],
                      offset: float = 1) -> pd.DataFrame:
        """Log transformation for skewed distributions."""
        for col in columns:
            df[f'{col}_log'] = np.log1p(df[col] + offset)
        return df
    
    def binning(self, df: pd.DataFrame, column: str, 
                bins: Union[int, List[float]], labels: List[str] = None) -> pd.DataFrame:
        """Discretize continuous features into bins."""
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
        return df
    
    def quantile_binning(self, df: pd.DataFrame, column: str,
                         n_quantiles: int = 4) -> pd.DataFrame:
        """Binning based on quantiles."""
        df[f'{column}_quantile'] = pd.qcut(df[column], q=n_quantiles, 
                                            labels=[f'q{i}' for i in range(n_quantiles)],
                                            duplicates='drop')
        return df
    
    # ============ Categorical Transformations ============
    
    def one_hot_encode(self, df: pd.DataFrame, columns: List[str],
                       max_categories: int = 50) -> pd.DataFrame:
        """One-hot encoding for categorical features."""
        for col in columns:
            # Limit to top categories to avoid explosion
            top_categories = df[col].value_counts().nlargest(max_categories).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'OTHER')
            
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
        return df
    
    def label_encode(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Label encoding for ordinal categories."""
        if not self.fitted:
            self.transformers['label_encode'] = {}
            for col in columns:
                encoder = LabelEncoder()
                df[f'{col}_encoded'] = encoder.fit_transform(df[col].fillna('MISSING'))
                self.transformers['label_encode'][col] = encoder
        else:
            for col in columns:
                encoder = self.transformers['label_encode'][col]
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        return df
    
    def target_encode(self, df: pd.DataFrame, columns: List[str],
                      target: str, smoothing: float = 1.0) -> pd.DataFrame:
        """Target encoding with smoothing to prevent overfitting."""
        if not self.fitted:
            self.transformers['target_encode'] = {}
            for col in columns:
                encoder = TargetEncoder(smoothing=smoothing)
                df[f'{col}_target_enc'] = encoder.fit_transform(df[col], df[target])
                self.transformers['target_encode'][col] = encoder
        else:
            for col in columns:
                encoder = self.transformers['target_encode'][col]
                df[f'{col}_target_enc'] = encoder.transform(df[col])
        return df
    
    def hash_encode(self, df: pd.DataFrame, column: str,
                    n_buckets: int = 100) -> pd.DataFrame:
        """Hash encoding for high-cardinality categorical features."""
        df[f'{column}_hash'] = df[column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % n_buckets
        )
        return df
    
    # ============ Text Transformations ============
    
    def tfidf_features(self, df: pd.DataFrame, column: str,
                       max_features: int = 100) -> pd.DataFrame:
        """TF-IDF features for text columns."""
        if not self.fitted:
            vectorizer = TfidfVectorizer(max_features=max_features)
            tfidf_matrix = vectorizer.fit_transform(df[column].fillna(''))
            self.transformers['tfidf'] = {column: vectorizer}
        else:
            vectorizer = self.transformers['tfidf'][column]
            tfidf_matrix = vectorizer.transform(df[column].fillna(''))
        
        # Add TF-IDF features as columns
        feature_names = [f'{column}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        return pd.concat([df, tfidf_df], axis=1)
    
    def text_statistics(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Extract statistical features from text."""
        df[f'{column}_length'] = df[column].str.len()
        df[f'{column}_word_count'] = df[column].str.split().str.len()
        df[f'{column}_unique_words'] = df[column].apply(
            lambda x: len(set(str(x).split())) if pd.notna(x) else 0
        )
        df[f'{column}_avg_word_length'] = df[column].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) else 0
        )
        return df
    
    # ============ Time-Based Transformations ============
    
    def datetime_features(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Extract features from datetime column."""
        df[column] = pd.to_datetime(df[column])
        
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        df[f'{column}_dayofweek'] = df[column].dt.dayofweek
        df[f'{column}_hour'] = df[column].dt.hour
        df[f'{column}_is_weekend'] = df[column].dt.dayofweek >= 5
        df[f'{column}_quarter'] = df[column].dt.quarter
        df[f'{column}_is_month_start'] = df[column].dt.is_month_start
        df[f'{column}_is_month_end'] = df[column].dt.is_month_end
        
        return df
    
    def cyclical_encode(self, df: pd.DataFrame, column: str,
                        period: int) -> pd.DataFrame:
        """Cyclical encoding for periodic features (e.g., hour, day of week)."""
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / period)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / period)
        return df
    
    # ============ Aggregation Features ============
    
    def rolling_features(self, df: pd.DataFrame, column: str,
                         windows: List[int], agg_funcs: List[str]) -> pd.DataFrame:
        """Rolling window aggregation features."""
        for window in windows:
            for func in agg_funcs:
                col_name = f'{column}_rolling_{window}_{func}'
                if func == 'mean':
                    df[col_name] = df[column].rolling(window=window).mean()
                elif func == 'std':
                    df[col_name] = df[column].rolling(window=window).std()
                elif func == 'min':
                    df[col_name] = df[column].rolling(window=window).min()
                elif func == 'max':
                    df[col_name] = df[column].rolling(window=window).max()
                elif func == 'sum':
                    df[col_name] = df[column].rolling(window=window).sum()
        return df
    
    def lag_features(self, df: pd.DataFrame, column: str,
                     lags: List[int]) -> pd.DataFrame:
        """Create lag features."""
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        return df
    
    def group_aggregations(self, df: pd.DataFrame, group_col: str,
                          agg_col: str, agg_funcs: List[str]) -> pd.DataFrame:
        """Group-level aggregation features."""
        agg_dict = {agg_col: agg_funcs}
        agg_df = df.groupby(group_col).agg(agg_dict)
        agg_df.columns = [f'{group_col}_{agg_col}_{func}' for func in agg_funcs]
        return df.merge(agg_df, on=group_col, how='left')

# Usage Example
transformer = FeatureTransformer()

# Apply transformations
df = transformer.standardize(df, ['amount', 'age'])
df = transformer.log_transform(df, ['income'])
df = transformer.one_hot_encode(df, ['category'])
df = transformer.target_encode(df, ['merchant'], target='is_fraud')
df = transformer.datetime_features(df, 'transaction_time')
df = transformer.cyclical_encode(df, 'hour', period=24)
df = transformer.rolling_features(df, 'amount', windows=[7, 30], agg_funcs=['mean', 'std'])

transformer.fitted = True  # Mark as fitted for inference
```

---

### Feature Selection Methods

```python
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFE, 
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

class FeatureSelector:
    """Feature selection methods for ML pipelines."""
    
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.selected_features = []
    
    def variance_filter(self, df: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove low-variance features."""
        feature_cols = [c for c in df.columns if c != self.target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[numeric_cols])
        
        selected = numeric_cols[selector.get_support()].tolist()
        print(f"Variance filter: {len(numeric_cols)} → {len(selected)} features")
        return selected
    
    def correlation_filter(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        feature_cols = [c for c in df.columns if c != self.target_col]
        numeric_df = df[feature_cols].select_dtypes(include=[np.number])
        
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        selected = [c for c in numeric_df.columns if c not in to_drop]
        
        print(f"Correlation filter: {len(numeric_df.columns)} → {len(selected)} features")
        return selected
    
    def mutual_information(self, df: pd.DataFrame, k: int = 50) -> List[str]:
        """Select top k features by mutual information."""
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[self.target_col]
        
        selector = SelectKBest(mutual_info_classif, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        # Get importance scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': selector.scores_
        }).sort_values('mi_score', ascending=False)
        
        print(f"Mutual information: top {len(selected)} features selected")
        return selected, scores
    
    def recursive_elimination(self, df: pd.DataFrame, n_features: int = 20,
                             step: int = 5) -> List[str]:
        """Recursive feature elimination with cross-validation."""
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[self.target_col]
        
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=step)
        selector.fit(X, y)
        
        selected = X.columns[selector.support_].tolist()
        
        # Get ranking
        ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        print(f"RFE: {len(selected)} features selected")
        return selected, ranking
    
    def model_based_selection(self, df: pd.DataFrame, threshold: str = 'median') -> List[str]:
        """Select features based on model importance."""
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[self.target_col]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        # Get importance scores
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Model-based selection: {len(selected)} features selected")
        return selected, importances
    
    def comprehensive_selection(self, df: pd.DataFrame) -> Dict:
        """Run all selection methods and combine results."""
        # Run all methods
        variance_features = set(self.variance_filter(df))
        corr_features = set(self.correlation_filter(df))
        mi_features, mi_scores = self.mutual_information(df)
        mi_features = set(mi_features)
        rfe_features, rfe_ranking = self.recursive_elimination(df)
        rfe_features = set(rfe_features)
        model_features, importance_scores = self.model_based_selection(df)
        model_features = set(model_features)
        
        # Find consensus features (selected by multiple methods)
        all_features = variance_features | corr_features | mi_features | rfe_features | model_features
        
        feature_scores = {}
        for feature in all_features:
            score = sum([
                feature in variance_features,
                feature in corr_features,
                feature in mi_features,
                feature in rfe_features,
                feature in model_features
            ])
            feature_scores[feature] = score
        
        # Select features selected by at least 3 methods
        consensus_features = [f for f, s in feature_scores.items() if s >= 3]
        
        return {
            'consensus_features': consensus_features,
            'feature_scores': feature_scores,
            'mi_scores': mi_scores,
            'importance_scores': importance_scores
        }

# Usage
selector = FeatureSelector(target_col='is_fraud')
selection_results = selector.comprehensive_selection(df)
final_features = selection_results['consensus_features']
```

---

## 🎯 Interview Questions

**Q1: How do you handle feature engineering for real-time inference with strict latency requirements (<50ms)?**

**Answer:**
```
Strategy: Pre-compute + cache + optimize

1. Pre-compute aggregations (batch):
   - User history features → Feature store
   - Running statistics → Updated hourly

2. Real-time features (compute):
   - Current request features only
   - Lightweight transformations

3. Caching layers:
   - L1: In-process cache (LRU)
   - L2: Redis (feature store)
   - L3: Database (fallback)

4. Optimization:
   - Vectorized operations
   - Compiled transformations (Numba)
   - Feature subset for inference
```

**Q2: How do you ensure feature pipeline consistency between training and inference?**

**Answer:**
1. **Single codebase** - same transformation code
2. **Feature store** - centralized feature definitions
3. **Schema enforcement** - validate feature schemas
4. **Integration tests** - test train/inference parity
5. **Monitoring** - detect feature drift

**Q3: Explain the difference between target encoding and one-hot encoding.**

| Aspect | One-Hot Encoding | Target Encoding |
|--------|-----------------|-----------------|
| Output | Binary columns | Single numeric |
| Cardinality | High → Many columns | Any → 1 column |
| Information | Category presence | Target relationship |
| Leakage risk | None | Yes (needs smoothing) |
| Best for | Low cardinality | High cardinality |

---

## 🔑 Key Takeaways

1. **Choose the right type** - batch vs streaming vs hybrid
2. **Design for reliability** - idempotency, error handling
3. **Monitor continuously** - track execution and quality
4. **Test thoroughly** - unit, integration, E2E tests
5. **Optimize performance** - parallelization, caching, incremental processing
6. **Feature selection matters** - reduce noise, improve performance
7. **Consistency is critical** - same transformations in train and inference

---

## 📚 Further Reading

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Apache Flink Documentation](https://flink.apache.org/docs/)
- [Feature Engineering Best Practices](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Feature Engineering and Selection](http://www.feat.engineering/)

---

## 🔗 Related Topics

- [Feature Stores](./01-feature-stores.md)
- [Online vs Offline Features](./02-online-vs-offline-features.md)
- [Feature Monitoring](./04-feature-monitoring.md)
