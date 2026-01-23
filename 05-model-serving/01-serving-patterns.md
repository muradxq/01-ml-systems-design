# Serving Patterns

## Overview

Model serving patterns define how ML models deliver predictions in production. The choice of serving pattern depends on latency requirements, throughput needs, cost constraints, and infrastructure capabilities. Different use cases require different patterns—fraud detection needs real-time, while batch reports can run overnight.

---

## 📊 Serving Pattern Comparison

| Pattern | Latency | Throughput | Cost | Use Case |
|---------|---------|------------|------|----------|
| **Real-time** | < 100ms | Medium | High | Fraud detection, recommendations |
| **Batch** | Hours | Very High | Low | Reports, bulk scoring |
| **Streaming** | Seconds-Minutes | High | Medium | Event processing, aggregations |
| **Async** | Seconds-Minutes | High | Medium | Non-blocking requests |

---

## 🚀 1. Real-Time Serving

**Definition:** Synchronous predictions with sub-second latency

```
┌─────────────────────────────────────────────────────────────────┐
│  Real-Time Serving Architecture                                  │
│                                                                  │
│  Client ───▶ Load Balancer ───▶ Model Servers ───▶ Response     │
│                                      │                           │
│                                      ▼                           │
│                              Feature Store (Hot)                 │
│                                                                  │
│  Latency Target: < 100ms (p99)                                  │
│  Availability Target: 99.9%                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Characteristics

- **Synchronous:** Client waits for response
- **Stateless:** Each request independent
- **Horizontally scalable:** Add more replicas for throughput
- **Feature retrieval:** Real-time feature store lookup

### Implementation

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import asyncio
import time
from prometheus_client import Counter, Histogram, generate_latest
import redis
import logging

# Metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests', ['model', 'status'])
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency', ['model'])
FEATURE_LATENCY = Histogram('feature_retrieval_latency_seconds', 'Feature retrieval latency')

# Request/Response schemas
class PredictionRequest(BaseModel):
    user_id: str
    item_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str
    latency_ms: float
    features_used: List[str]

# Model server
class RealTimeModelServer:
    """High-performance real-time model server."""
    
    def __init__(
        self,
        model_path: str,
        model_version: str,
        feature_store_url: str,
        cache_ttl: int = 300
    ):
        self.model = self._load_model(model_path)
        self.model_version = model_version
        self.feature_store = redis.Redis.from_url(feature_store_url)
        self.prediction_cache = redis.Redis.from_url(
            feature_store_url, 
            db=1,
            decode_responses=True
        )
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, path: str):
        """Load model with optimizations."""
        model = torch.jit.load(path)
        model.eval()
        
        # Enable inference optimizations
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.jit.optimize_for_inference(model)
        
        return model
    
    async def get_features(self, user_id: str, item_id: str = None) -> Dict[str, float]:
        """Retrieve features from feature store."""
        start = time.time()
        
        try:
            # Get user features
            user_key = f"user:{user_id}"
            user_features = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.feature_store.hgetall(user_key)
            )
            
            features = {k.decode(): float(v) for k, v in user_features.items()}
            
            # Get item features if item_id provided
            if item_id:
                item_key = f"item:{item_id}"
                item_features = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.feature_store.hgetall(item_key)
                )
                features.update({k.decode(): float(v) for k, v in item_features.items()})
            
            return features
        
        finally:
            FEATURE_LATENCY.observe(time.time() - start)
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check prediction cache."""
        cached = self.prediction_cache.get(cache_key)
        if cached:
            return eval(cached)  # Use json.loads in production
        return None
    
    def _cache_prediction(self, cache_key: str, result: Dict):
        """Cache prediction result."""
        self.prediction_cache.setex(
            cache_key,
            self.cache_ttl,
            str(result)
        )
    
    @torch.no_grad()
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction with caching and monitoring."""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"pred:{request.user_id}:{request.item_id}"
            cached = self._check_cache(cache_key)
            if cached:
                REQUEST_COUNT.labels(model=self.model_version, status='cache_hit').inc()
                return PredictionResponse(**cached)
            
            # Get features
            features = await self.get_features(request.user_id, request.item_id)
            
            # Add context features
            if request.context:
                features.update(request.context)
            
            # Prepare input tensor
            feature_names = sorted(features.keys())
            feature_values = [features[name] for name in feature_names]
            input_tensor = torch.tensor([feature_values], dtype=torch.float32)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Predict
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            
            # Build response
            latency_ms = (time.time() - start_time) * 1000
            response = PredictionResponse(
                prediction=prediction,
                probability=probability,
                model_version=self.model_version,
                latency_ms=latency_ms,
                features_used=feature_names
            )
            
            # Cache result
            self._cache_prediction(cache_key, response.dict())
            
            # Record metrics
            REQUEST_COUNT.labels(model=self.model_version, status='success').inc()
            REQUEST_LATENCY.labels(model=self.model_version).observe(latency_ms / 1000)
            
            return response
        
        except Exception as e:
            REQUEST_COUNT.labels(model=self.model_version, status='error').inc()
            self.logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# FastAPI application
app = FastAPI(title="Real-Time Model Server")
model_server = None

@app.on_event("startup")
async def startup():
    global model_server
    model_server = RealTimeModelServer(
        model_path="models/model.pt",
        model_version="v1.0.0",
        feature_store_url="redis://localhost:6379"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return await model_server.predict(request)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": model_server.model_version}

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

### Best Practices

1. **Optimize model loading** - Use TorchScript, ONNX, or TensorRT
2. **Cache predictions** - For identical inputs
3. **Async feature retrieval** - Don't block on I/O
4. **Connection pooling** - Reuse database/cache connections
5. **Request batching** - Batch concurrent requests

---

## 📦 2. Batch Serving

**Definition:** Asynchronous predictions on large datasets

```
┌─────────────────────────────────────────────────────────────────┐
│  Batch Serving Architecture                                      │
│                                                                  │
│  Scheduler ───▶ Data Lake ───▶ Batch Job ───▶ Output Store      │
│      │              │              │              │              │
│      │              ▼              ▼              ▼              │
│      │         Read Data     Process     Write Results          │
│      │                       in Bulk                             │
│      │                                                           │
│      └───────── Triggered by schedule or event                  │
│                                                                  │
│  Run Time: Minutes to Hours                                     │
│  Volume: Millions to Billions of records                        │
└─────────────────────────────────────────────────────────────────┘
```

### Characteristics

- **Scheduled:** Run at fixed intervals
- **High throughput:** Optimize for volume
- **Cost efficient:** Use spot instances
- **Fault tolerant:** Checkpoint progress

### Implementation with Spark

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, pandas_udf
from pyspark.sql.types import DoubleType, StructType, StructField, StringType
import pandas as pd
import mlflow.pyfunc
from datetime import datetime
import logging

class BatchPredictionPipeline:
    """Batch prediction pipeline using Spark."""
    
    def __init__(
        self,
        model_uri: str,
        spark_config: dict = None
    ):
        # Initialize Spark
        builder = SparkSession.builder.appName("BatchPredictions")
        
        if spark_config:
            for key, value in spark_config.items():
                builder = builder.config(key, value)
        
        self.spark = builder.getOrCreate()
        
        # Load model (broadcast to all workers)
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        
        self.logger = logging.getLogger(__name__)
    
    def run_predictions(
        self,
        input_path: str,
        output_path: str,
        feature_columns: list,
        partition_by: str = None,
        num_partitions: int = 200
    ):
        """Run batch predictions on input data."""
        
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Starting batch prediction run: {run_id}")
        
        # Read input data
        df = self.spark.read.parquet(input_path)
        self.logger.info(f"Loaded {df.count()} records from {input_path}")
        
        # Repartition for parallelism
        df = df.repartition(num_partitions)
        
        # Define prediction UDF using pandas for efficiency
        @pandas_udf(DoubleType())
        def predict_udf(features: pd.DataFrame) -> pd.Series:
            model = self.broadcast_model.value
            predictions = model.predict(features)
            return pd.Series(predictions)
        
        # Create feature vector column
        df = df.withColumn(
            "prediction",
            predict_udf(*[col(c) for c in feature_columns])
        )
        
        # Add metadata
        df = df.withColumn("prediction_run_id", lit(run_id))
        df = df.withColumn("prediction_timestamp", current_timestamp())
        
        # Write output
        writer = df.write.mode("overwrite")
        
        if partition_by:
            writer = writer.partitionBy(partition_by)
        
        writer.parquet(output_path)
        
        # Log statistics
        stats = df.select(
            count("*").alias("total_predictions"),
            avg("prediction").alias("avg_prediction"),
            min("prediction").alias("min_prediction"),
            max("prediction").alias("max_prediction")
        ).collect()[0]
        
        self.logger.info(f"Completed batch prediction run: {run_id}")
        self.logger.info(f"Statistics: {stats.asDict()}")
        
        return {
            "run_id": run_id,
            "total_predictions": stats.total_predictions,
            "output_path": output_path
        }
    
    def run_incremental_predictions(
        self,
        input_path: str,
        output_path: str,
        watermark_path: str,
        feature_columns: list,
        timestamp_column: str
    ):
        """Run incremental predictions on new data only."""
        
        # Read watermark (last processed timestamp)
        try:
            watermark_df = self.spark.read.parquet(watermark_path)
            last_watermark = watermark_df.select("last_processed").collect()[0][0]
        except:
            last_watermark = "1970-01-01 00:00:00"
        
        self.logger.info(f"Processing records after {last_watermark}")
        
        # Read only new data
        df = self.spark.read.parquet(input_path).filter(
            col(timestamp_column) > last_watermark
        )
        
        new_records = df.count()
        if new_records == 0:
            self.logger.info("No new records to process")
            return {"new_records": 0}
        
        self.logger.info(f"Found {new_records} new records")
        
        # Run predictions
        result = self.run_predictions(
            input_path=None,  # Use filtered df
            output_path=output_path,
            feature_columns=feature_columns
        )
        
        # Update watermark
        new_watermark = df.agg({timestamp_column: "max"}).collect()[0][0]
        self.spark.createDataFrame(
            [{"last_processed": new_watermark}]
        ).write.mode("overwrite").parquet(watermark_path)
        
        return result

# Usage with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_daily_predictions(**context):
    """Airflow task to run daily batch predictions."""
    
    pipeline = BatchPredictionPipeline(
        model_uri="models:/fraud-detection/Production",
        spark_config={
            "spark.executor.memory": "8g",
            "spark.executor.cores": "4",
            "spark.dynamicAllocation.enabled": "true"
        }
    )
    
    execution_date = context['execution_date'].strftime("%Y/%m/%d")
    
    result = pipeline.run_predictions(
        input_path=f"s3://data/transactions/{execution_date}/",
        output_path=f"s3://predictions/fraud/{execution_date}/",
        feature_columns=['amount', 'merchant_category', 'user_age', 'transaction_count'],
        partition_by="prediction_hour"
    )
    
    return result

with DAG(
    'batch_predictions',
    default_args={
        'owner': 'ml-team',
        'depends_on_past': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5)
    },
    description='Daily batch predictions',
    schedule_interval='0 6 * * *',  # 6 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    predict_task = PythonOperator(
        task_id='run_predictions',
        python_callable=run_daily_predictions,
        provide_context=True
    )
```

### Best Practices

1. **Use checkpointing** - Resume from failures
2. **Partition output** - For efficient querying
3. **Monitor progress** - Log batch statistics
4. **Optimize resources** - Use spot instances
5. **Validate output** - Check for anomalies

---

## 🌊 3. Streaming Serving

**Definition:** Continuous predictions on streaming data

```
┌─────────────────────────────────────────────────────────────────┐
│  Streaming Serving Architecture                                  │
│                                                                  │
│  Event Source ───▶ Kafka ───▶ Stream Processor ───▶ Output      │
│                                     │                            │
│                                     ▼                            │
│                              Model Inference                     │
│                                     │                            │
│                                     ▼                            │
│                              Kafka (Results)                     │
│                                                                  │
│  Latency: Seconds to Minutes                                    │
│  Characteristics: Continuous, event-driven                      │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation with Kafka and Flink

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import MapFunction, ProcessFunction
import json
import mlflow.pyfunc

class StreamingPredictionService:
    """Streaming prediction service using Flink."""
    
    def __init__(
        self,
        model_uri: str,
        kafka_bootstrap_servers: str,
        input_topic: str,
        output_topic: str
    ):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_parallelism(4)
        
        # Load model
        self.model = mlflow.pyfunc.load_model(model_uri)
        
        # Kafka source
        self.source = KafkaSource.builder() \
            .set_bootstrap_servers(kafka_bootstrap_servers) \
            .set_topics(input_topic) \
            .set_value_only_deserializer(SimpleStringSchema()) \
            .build()
        
        # Kafka sink
        self.sink = KafkaSink.builder() \
            .set_bootstrap_servers(kafka_bootstrap_servers) \
            .set_record_serializer(
                KafkaRecordSerializationSchema.builder()
                .set_topic(output_topic)
                .set_value_serialization_schema(SimpleStringSchema())
                .build()
            ) \
            .build()
    
    def run(self):
        """Run streaming prediction pipeline."""
        
        # Create prediction function
        model = self.model
        
        class PredictionMapper(MapFunction):
            def map(self, value):
                event = json.loads(value)
                
                # Extract features
                features = {
                    'amount': event.get('amount', 0),
                    'merchant_id': event.get('merchant_id', ''),
                    'user_id': event.get('user_id', '')
                }
                
                # Make prediction
                prediction = model.predict([features])[0]
                
                # Build result
                result = {
                    'event_id': event.get('event_id'),
                    'prediction': float(prediction),
                    'timestamp': event.get('timestamp'),
                    'processed_at': datetime.utcnow().isoformat()
                }
                
                return json.dumps(result)
        
        # Build pipeline
        stream = self.env.from_source(
            self.source,
            WatermarkStrategy.for_monotonous_timestamps(),
            "Kafka Source"
        )
        
        predictions = stream.map(PredictionMapper())
        predictions.sink_to(self.sink)
        
        # Execute
        self.env.execute("Streaming Predictions")

# Alternative: Kafka Streams-style with confluent-kafka
from confluent_kafka import Consumer, Producer
import threading

class KafkaStreamingPredictor:
    """Simple Kafka streaming predictor."""
    
    def __init__(
        self,
        model_path: str,
        bootstrap_servers: str,
        input_topic: str,
        output_topic: str,
        group_id: str,
        num_workers: int = 4
    ):
        self.model = mlflow.pyfunc.load_model(model_path)
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.num_workers = num_workers
        
        self.consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        }
        
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers
        }
        
        self.running = False
    
    def _worker(self, worker_id: int):
        """Worker thread for consuming and predicting."""
        consumer = Consumer(self.consumer_config)
        producer = Producer(self.producer_config)
        
        consumer.subscribe([self.input_topic])
        
        while self.running:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            
            try:
                # Parse event
                event = json.loads(msg.value().decode('utf-8'))
                
                # Make prediction
                features = self._extract_features(event)
                prediction = self.model.predict([features])[0]
                
                # Build result
                result = {
                    'event_id': event.get('event_id'),
                    'prediction': float(prediction),
                    'worker_id': worker_id,
                    'latency_ms': (time.time() - event.get('timestamp', time.time())) * 1000
                }
                
                # Produce result
                producer.produce(
                    self.output_topic,
                    key=str(event.get('event_id')),
                    value=json.dumps(result)
                )
                producer.poll(0)  # Trigger delivery reports
                
                # Commit offset
                consumer.commit(asynchronous=False)
            
            except Exception as e:
                print(f"Error processing message: {e}")
        
        consumer.close()
        producer.flush()
    
    def start(self):
        """Start worker threads."""
        self.running = True
        self.workers = []
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop worker threads."""
        self.running = False
        for worker in self.workers:
            worker.join()
```

---

## 🔄 4. Async Serving

**Definition:** Non-blocking predictions with callbacks or polling

```
┌─────────────────────────────────────────────────────────────────┐
│  Async Serving Architecture                                      │
│                                                                  │
│  Client ───▶ API ───▶ Queue ───▶ Worker ───▶ Results Store      │
│     │                                              │             │
│     │         ◀────────────────────────────────────┘             │
│     │                  Poll/Webhook                              │
│     │                                                            │
│  1. Submit request, get request_id                              │
│  2. Worker processes asynchronously                             │
│  3. Client polls or receives webhook                            │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import asyncio
from typing import Optional
import redis
import json

app = FastAPI()

class AsyncPredictionRequest(BaseModel):
    features: dict
    callback_url: Optional[str] = None

class AsyncPredictionResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[dict] = None

class AsyncPredictionService:
    """Async prediction service with queue processing."""
    
    def __init__(self):
        self.redis = redis.Redis(decode_responses=True)
        self.model = self._load_model()
    
    async def submit_prediction(
        self,
        request: AsyncPredictionRequest
    ) -> str:
        """Submit prediction request to queue."""
        request_id = str(uuid.uuid4())
        
        # Store request
        self.redis.hset(f"request:{request_id}", mapping={
            "status": "pending",
            "features": json.dumps(request.features),
            "callback_url": request.callback_url or "",
            "created_at": datetime.utcnow().isoformat()
        })
        
        # Add to processing queue
        self.redis.lpush("prediction_queue", request_id)
        
        return request_id
    
    async def get_prediction_status(
        self,
        request_id: str
    ) -> AsyncPredictionResponse:
        """Get status of prediction request."""
        data = self.redis.hgetall(f"request:{request_id}")
        
        if not data:
            raise HTTPException(status_code=404, detail="Request not found")
        
        result = None
        if data.get("status") == "completed":
            result = json.loads(data.get("result", "{}"))
        
        return AsyncPredictionResponse(
            request_id=request_id,
            status=data.get("status"),
            result=result
        )
    
    async def process_queue(self):
        """Background worker to process prediction queue."""
        while True:
            # Get next request from queue
            request_id = self.redis.brpop("prediction_queue", timeout=5)
            
            if request_id is None:
                continue
            
            request_id = request_id[1]
            
            try:
                # Update status
                self.redis.hset(f"request:{request_id}", "status", "processing")
                
                # Get request data
                data = self.redis.hgetall(f"request:{request_id}")
                features = json.loads(data.get("features"))
                
                # Make prediction
                prediction = self.model.predict([features])[0]
                
                result = {
                    "prediction": float(prediction),
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                # Store result
                self.redis.hset(f"request:{request_id}", mapping={
                    "status": "completed",
                    "result": json.dumps(result)
                })
                
                # Callback if specified
                callback_url = data.get("callback_url")
                if callback_url:
                    await self._send_callback(callback_url, request_id, result)
            
            except Exception as e:
                self.redis.hset(f"request:{request_id}", mapping={
                    "status": "failed",
                    "error": str(e)
                })
    
    async def _send_callback(self, url: str, request_id: str, result: dict):
        """Send webhook callback."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            await client.post(url, json={
                "request_id": request_id,
                "result": result
            })

# API endpoints
service = AsyncPredictionService()

@app.post("/predict/async")
async def submit_prediction(request: AsyncPredictionRequest):
    request_id = await service.submit_prediction(request)
    return {"request_id": request_id, "status": "pending"}

@app.get("/predict/{request_id}")
async def get_prediction(request_id: str):
    return await service.get_prediction_status(request_id)

@app.on_event("startup")
async def startup():
    # Start background worker
    asyncio.create_task(service.process_queue())
```

---

## 🎯 Choosing the Right Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│  Decision Framework                                              │
│                                                                  │
│  Latency requirement?                                           │
│  ├── < 100ms ──────────▶ Real-time                             │
│  ├── < 1 minute ───────▶ Streaming or Async                    │
│  └── > 1 minute ───────▶ Batch                                 │
│                                                                  │
│  Volume?                                                        │
│  ├── < 100 QPS ────────▶ Real-time (simple)                    │
│  ├── 100-10K QPS ──────▶ Real-time (scaled)                    │
│  └── > 10K QPS ────────▶ Real-time + Caching                   │
│                                                                  │
│  Data freshness?                                                │
│  ├── Real-time features ▶ Real-time or Streaming               │
│  ├── Near-real-time ───▶ Streaming                             │
│  └── Historical only ──▶ Batch                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Best Practices

1. **Match pattern to requirements** - Don't over-engineer
2. **Monitor latency distributions** - p50, p95, p99
3. **Implement fallbacks** - Graceful degradation
4. **Cache appropriately** - Reduce redundant computation
5. **Scale horizontally** - Stateless design

---

## 🔗 Related Topics

- [Model Deployment](./model-deployment.md) - Deploy serving infrastructure
- [A/B Testing](./ab-testing.md) - Test different models
- [Caching Strategies](../07-scalability-performance/caching-strategies.md) - Optimize serving
