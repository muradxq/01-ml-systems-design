# Batch vs Real-time Processing

## Overview

Choosing between batch and real-time processing depends on latency requirements, data volume, cost constraints, and use cases. Many production ML systems use a hybrid approach—real-time for user-facing predictions and batch for pre-computation and analytics.

---

## 📊 Detailed Comparison

| Aspect | Batch | Real-time | Streaming |
|--------|-------|-----------|-----------|
| **Latency** | Minutes-hours | <100ms | Seconds |
| **Throughput** | Very high | Medium | High |
| **Cost** | $ | $$$ | $$ |
| **Complexity** | Low | High | Medium |
| **Data freshness** | Hours old | Current | Near-current |
| **Resource usage** | Bursty | Constant | Constant |
| **Fault tolerance** | Easy | Complex | Medium |

---

## 🏗️ Architecture Patterns

### 1. Pure Real-time

```
Request → Feature Fetch → Model Inference → Response
                    (all in <100ms)
```

**Use when:**
- Predictions depend on real-time context
- Each request is unique
- Low latency is critical

**Examples:** Fraud detection, search ranking, real-time bidding

### 2. Pure Batch

```
Scheduled Job → Process All Data → Store Results → Serve from Cache
                    (runs periodically)
```

**Use when:**
- Predictions can be pre-computed
- Data doesn't change frequently
- Cost is a primary concern

**Examples:** Daily recommendations, email campaigns, reporting

### 3. Hybrid (Lambda Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Incoming Data                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│    Batch Layer       │        │   Speed Layer        │
│  - Daily processing  │        │  - Real-time updates │
│  - Historical data   │        │  - Recent data only  │
│  - High accuracy     │        │  - Approximate       │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │     Serving Layer        │
              │  - Merges batch + speed  │
              │  - Serves queries        │
              └──────────────────────────┘
```

**Use when:**
- Need both historical completeness and real-time updates
- Batch provides baseline, real-time provides freshness

---

## 📝 Implementation Examples

### Batch Processing

```python
from pyspark.sql import SparkSession
from datetime import datetime, timedelta

def batch_prediction_job():
    """Daily batch prediction job using Spark."""
    spark = SparkSession.builder \
        .appName("DailyRecommendations") \
        .getOrCreate()
    
    # Load data
    users = spark.read.parquet("s3://data/users/")
    items = spark.read.parquet("s3://data/items/")
    interactions = spark.read.parquet("s3://data/interactions/")
    
    # Load model (broadcast to all workers)
    model = load_model("recommendation_model")
    model_broadcast = spark.sparkContext.broadcast(model)
    
    # Generate predictions for all users
    def predict_for_user(user_features):
        model = model_broadcast.value
        return model.predict(user_features)
    
    predict_udf = spark.udf.register("predict", predict_for_user)
    
    # Batch predict
    predictions = users.select(
        "user_id",
        predict_udf("features").alias("recommendations")
    )
    
    # Write to serving store
    predictions.write \
        .mode("overwrite") \
        .parquet(f"s3://predictions/{datetime.now().date()}/")
    
    # Also write to Redis for serving
    predictions.foreachPartition(write_to_redis)
    
    spark.stop()

def write_to_redis(partition):
    """Write partition to Redis."""
    import redis
    r = redis.Redis(host='redis', port=6379)
    
    for row in partition:
        r.setex(
            f"recommendations:{row.user_id}",
            86400,  # 24 hour TTL
            json.dumps(row.recommendations)
        )

# Schedule with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    'daily_recommendations',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False
)

task = PythonOperator(
    task_id='generate_recommendations',
    python_callable=batch_prediction_job,
    dag=dag
)
```

### Real-time Processing

```python
from fastapi import FastAPI
import asyncio
from typing import Dict, Any

app = FastAPI()

# Pre-load model
model = load_model("fraud_detection")
feature_store = FeatureStoreClient()

@app.post("/predict/fraud")
async def predict_fraud(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Real-time fraud prediction.
    Must complete in <100ms.
    """
    start_time = time.time()
    
    # Fetch features in parallel
    user_features, merchant_features = await asyncio.gather(
        feature_store.get_async(f"user:{transaction['user_id']}"),
        feature_store.get_async(f"merchant:{transaction['merchant_id']}")
    )
    
    # Combine features
    features = {
        **transaction,
        **user_features,
        **merchant_features
    }
    
    # Make prediction
    prediction = model.predict(features)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "is_fraud": prediction > 0.5,
        "fraud_score": float(prediction),
        "latency_ms": latency_ms
    }
```

### Hybrid Approach

```python
class HybridRecommendationService:
    """
    Combines batch pre-computed recommendations with real-time signals.
    """
    
    def __init__(self):
        self.batch_cache = redis.Redis(host='redis-batch')
        self.realtime_model = load_model("realtime_ranker")
        self.feature_store = FeatureStoreClient()
    
    async def get_recommendations(self, user_id: str, 
                                  context: Dict) -> list:
        """
        Get recommendations using hybrid approach:
        1. Fetch batch-computed candidates
        2. Re-rank with real-time signals
        """
        # 1. Get batch-computed candidates (fast)
        batch_recs = self.batch_cache.get(f"recommendations:{user_id}")
        if batch_recs:
            candidates = json.loads(batch_recs)
        else:
            # Fallback to popular items
            candidates = self.get_popular_items()
        
        # 2. Get real-time features
        user_session = await self.feature_store.get_async(
            f"session:{user_id}"
        )
        
        # 3. Re-rank with real-time context
        scored_candidates = []
        for item_id in candidates[:100]:  # Top 100 candidates
            features = {
                'user_id': user_id,
                'item_id': item_id,
                'time_of_day': context.get('hour'),
                'device': context.get('device'),
                'recent_views': user_session.get('recent_views', [])
            }
            score = self.realtime_model.predict(features)
            scored_candidates.append((item_id, score))
        
        # 4. Return top re-ranked items
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scored_candidates[:20]]
```

---

## 🎯 Decision Framework

### Choose Batch When:
- [ ] Predictions can be pre-computed
- [ ] Data changes infrequently (daily/weekly)
- [ ] Cost efficiency is priority
- [ ] Some staleness is acceptable
- [ ] High volume, predictable set of entities

### Choose Real-time When:
- [ ] Predictions depend on real-time context
- [ ] Each request is unique/unpredictable
- [ ] Low latency (<100ms) is required
- [ ] Data changes frequently
- [ ] User experience requires freshness

### Choose Hybrid When:
- [ ] Need both coverage and freshness
- [ ] Batch for base, real-time for personalization
- [ ] Cost-sensitive but latency-critical
- [ ] Can pre-compute candidates, re-rank in real-time

---

## 📊 Cost Comparison

| Scenario | Batch Cost | Real-time Cost | Notes |
|----------|-----------|----------------|-------|
| **1M predictions/day** | ~$10/day | ~$100/day | Batch 10x cheaper |
| **100K unique users** | Pre-compute all | On-demand | Batch if users are known |
| **Fraud detection** | N/A | Required | Must be real-time |
| **Email recommendations** | Batch daily | N/A | No real-time need |

---

## ✅ Best Practices

1. **Start with batch** - easier, cheaper, then add real-time
2. **Use batch for candidates** - real-time for ranking
3. **Cache batch results** - serve instantly
4. **Monitor freshness** - track staleness metrics
5. **Plan for failures** - batch results as fallback for real-time
6. **Test both paths** - ensure quality at each stage

---

## 🔗 Related Topics

- [Horizontal Scaling](./01-horizontal-scaling.md) - Scale real-time systems
- [Caching Strategies](./02-caching-strategies.md) - Serve batch results
- [Feature Stores](../../phase-2-core-components/03-feature-engineering/01-feature-stores.md) - Online/offline features
