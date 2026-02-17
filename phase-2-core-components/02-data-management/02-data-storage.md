# Data Storage

## Overview

Choosing the right storage architecture is critical for ML systems. Different storage solutions serve different purposes and have different trade-offs.

---

## 🏗️ Storage Architectures

### 1. Data Lake

**Purpose:** Store raw, unprocessed data in its native format

**Characteristics:**
- Schema-on-read (flexible schemas)
- Supports structured, semi-structured, unstructured data
- Cost-effective for large volumes
- Long-term storage

**Use Cases:**
- Raw data ingestion
- Exploratory data analysis
- Multiple data formats
- Historical data archive

**Technologies:**
- AWS S3, Azure Data Lake Storage, Google Cloud Storage
- HDFS (Hadoop Distributed File System)
- MinIO (self-hosted)

**Pros:**
- Flexible schema
- Cost-effective
- Scalable
- Supports all data types

**Cons:**
- Slower queries
- No ACID guarantees
- Requires processing layer

---

### 2. Data Warehouse

**Purpose:** Store processed, structured data optimized for analytics

**Characteristics:**
- Schema-on-write (enforced schemas)
- Optimized for SQL queries
- Columnar storage
- Fast analytical queries

**Use Cases:**
- Business intelligence
- Analytics and reporting
- Structured data analysis
- Aggregated data

**Technologies:**
- Snowflake, BigQuery, Redshift
- Databricks SQL
- ClickHouse

**Pros:**
- Fast queries
- SQL support
- ACID guarantees
- Optimized for analytics

**Cons:**
- Less flexible
- More expensive
- Requires structured data
- Schema changes are costly

---

### 3. Feature Store

**Purpose:** Store and serve features for ML models

**Characteristics:**
- Online and offline storage
- Low-latency serving
- Feature versioning
- Point-in-time correctness

**Use Cases:**
- Feature serving for inference
- Feature storage for training
- Feature discovery
- Feature monitoring

**Technologies:**
- Feast, Tecton, Hopsworks
- SageMaker Feature Store
- Custom implementations

**Pros:**
- Optimized for ML
- Consistent features
- Low latency
- Feature versioning

**Cons:**
- ML-specific
- Additional infrastructure
- Learning curve

---

### 4. Object Storage

**Purpose:** Store files and blobs

**Characteristics:**
- Key-value access
- REST APIs
- Highly scalable
- Cost-effective

**Use Cases:**
- Model artifacts
- Large files
- Images, videos
- Backup storage

**Technologies:**
- AWS S3, Azure Blob Storage, Google Cloud Storage
- MinIO

**Pros:**
- Simple API
- Highly scalable
- Cost-effective
- Durable

**Cons:**
- Not for structured queries
- Eventual consistency
- No transactions

---

## 📊 Storage Selection Guide

### Data Lake vs Data Warehouse

| Aspect | Data Lake | Data Warehouse |
|--------|-----------|----------------|
| **Data Type** | All types | Structured |
| **Schema** | Schema-on-read | Schema-on-write |
| **Cost** | Lower | Higher |
| **Query Speed** | Slower | Faster |
| **Flexibility** | High | Low |
| **Use Case** | Raw data, exploration | Analytics, reporting |

**When to Use Data Lake:**
- Raw data storage
- Multiple data formats
- Exploratory analysis
- Cost-sensitive

**When to Use Data Warehouse:**
- Structured analytics
- Fast SQL queries
- Business reporting
- Performance-critical

---

## 🏗️ Storage Architecture Patterns

### 1. Lambda Architecture

**Pattern:** Separate batch and streaming layers

```
┌─────────────────────────────────────────┐
│         Streaming Layer                  │
│  (Real-time processing, Kappa)          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Batch Layer                      │
│  (Historical processing, Lambda)         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Serving Layer                    │
│  (Unified view, queries)                 │
└─────────────────────────────────────────┘
```

**Use Cases:**
- Real-time and batch requirements
- Different latency needs
- Historical and live data

---

### 2. Medallion Architecture

**Pattern:** Bronze → Silver → Gold layers

```
┌─────────────────────────────────────────┐
│         Bronze Layer                     │
│  (Raw data, data lake)                   │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Silver Layer                     │
│  (Cleaned, validated data)               │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Gold Layer                       │
│  (Aggregated, business-ready data)       │
└─────────────────────────────────────────┘
```

**Use Cases:**
- Data quality progression
- Incremental processing
- Business-ready datasets

---

### 3. Data Mesh

**Pattern:** Decentralized data ownership

**Principles:**
- Domain-oriented ownership
- Data as a product
- Self-serve infrastructure
- Federated governance

**Use Cases:**
- Large organizations
- Multiple domains
- Distributed teams

---

## 💾 Storage Optimization

### 1. Partitioning

**Purpose:** Improve query performance

**Strategies:**
- Time-based (date, hour)
- Category-based (region, product)
- Hash-based (user_id)

**Example:**
```
s3://data-lake/events/
  year=2024/
    month=01/
      day=15/
        events.parquet
```

---

### 2. Compression

**Purpose:** Reduce storage costs and I/O

**Formats:**
- Parquet (columnar, compressed)
- ORC (optimized row columnar)
- Gzip, Snappy (general compression)

**Trade-offs:**
- Compression ratio vs speed
- Read vs write performance

---

### 3. Caching

**Purpose:** Speed up frequent queries

**Strategies:**
- Query result caching
- Frequently accessed data
- In-memory caches

**Tools:**
- Redis, Memcached
- Query result caches

---

### 4. Lifecycle Management

**Purpose:** Optimize costs

**Strategies:**
- Hot storage (frequent access)
- Warm storage (occasional access)
- Cold storage (rare access)
- Archive storage (long-term)

**Example:**
- Hot: Last 30 days → Standard storage
- Warm: 30-90 days → Infrequent access
- Cold: 90-365 days → Archive storage
- Archive: >365 days → Glacier

---

## ✅ Best Practices

### 1. Choose the Right Storage
- Data Lake for raw data
- Data Warehouse for analytics
- Feature Store for ML features
- Object Storage for files

### 2. Optimize for Access Patterns
- Partition by query patterns
- Compress appropriately
- Cache frequently accessed data

### 3. Implement Lifecycle Policies
- Move data to cheaper storage over time
- Delete unnecessary data
- Archive old data

### 4. Monitor Costs
- Track storage usage
- Optimize based on access patterns
- Use cost alerts

### 5. Ensure Data Quality
- Validate at ingestion
- Monitor data quality
- Clean and transform appropriately

---

## 🎯 Storage by ML Use Case

### Training Data
- **Storage**: Data Lake or Data Warehouse
- **Format**: Parquet, CSV
- **Partitioning**: By date, dataset
- **Versioning**: Critical

### Feature Storage
- **Storage**: Feature Store
- **Format**: Optimized for serving
- **Partitioning**: By feature group
- **Versioning**: Critical

### Model Artifacts
- **Storage**: Object Storage
- **Format**: Model format (pickle, ONNX, etc.)
- **Versioning**: Critical
- **Metadata**: Model registry

---

## 🛠️ Advanced Storage Patterns

### Lakehouse Architecture

Combines the best of data lakes and warehouses.

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

class LakehouseStorage:
    """Lakehouse implementation with Delta Lake."""
    
    def __init__(self, spark: SparkSession, base_path: str):
        self.spark = spark
        self.base_path = base_path
    
    def write_bronze(self, df, table_name: str):
        """Write raw data to bronze layer."""
        path = f"{self.base_path}/bronze/{table_name}"
        (df.write
           .format("delta")
           .mode("append")
           .option("mergeSchema", "true")
           .save(path))
    
    def write_silver(self, df, table_name: str, partition_cols: List[str] = None):
        """Write cleaned data to silver layer."""
        path = f"{self.base_path}/silver/{table_name}"
        writer = (df.write
                    .format("delta")
                    .mode("overwrite")
                    .option("overwriteSchema", "true"))
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        writer.save(path)
    
    def write_gold(self, df, table_name: str):
        """Write aggregated/feature data to gold layer."""
        path = f"{self.base_path}/gold/{table_name}"
        (df.write
           .format("delta")
           .mode("overwrite")
           .save(path))
    
    def upsert(self, source_df, table_path: str, merge_keys: List[str]):
        """Upsert data using Delta Lake merge."""
        delta_table = DeltaTable.forPath(self.spark, table_path)
        
        merge_condition = " AND ".join(
            [f"target.{key} = source.{key}" for key in merge_keys]
        )
        
        (delta_table.alias("target")
         .merge(source_df.alias("source"), merge_condition)
         .whenMatchedUpdateAll()
         .whenNotMatchedInsertAll()
         .execute())
    
    def time_travel_query(self, table_path: str, as_of: str):
        """Query data as of a specific time or version."""
        if as_of.isdigit():
            # Version number
            return self.spark.read.format("delta").option(
                "versionAsOf", as_of
            ).load(table_path)
        else:
            # Timestamp
            return self.spark.read.format("delta").option(
                "timestampAsOf", as_of
            ).load(table_path)
    
    def vacuum(self, table_path: str, retention_hours: int = 168):
        """Remove old files to save storage."""
        delta_table = DeltaTable.forPath(self.spark, table_path)
        delta_table.vacuum(retention_hours)
```

### Multi-Modal Data Storage

Handle different data types for ML models.

```python
class MultiModalStorage:
    """Storage for multi-modal ML data (text, images, structured)."""
    
    def __init__(self, config: Dict):
        self.structured_store = PostgresStore(config['postgres'])
        self.blob_store = S3Store(config['s3'])
        self.vector_store = PineconeStore(config['pinecone'])
        self.metadata_store = MongoStore(config['mongo'])
    
    def store_training_example(self, example: Dict):
        """Store a multi-modal training example."""
        example_id = str(uuid.uuid4())
        
        # Store structured features
        if 'structured_features' in example:
            self.structured_store.insert(
                table='training_features',
                data={'id': example_id, **example['structured_features']}
            )
        
        # Store images/files
        if 'image' in example:
            image_path = f"training_data/{example_id}/image.jpg"
            self.blob_store.upload(example['image'], image_path)
        
        # Store text embeddings
        if 'text_embedding' in example:
            self.vector_store.upsert(
                id=example_id,
                vector=example['text_embedding'],
                metadata={'label': example.get('label')}
            )
        
        # Store metadata
        self.metadata_store.insert({
            '_id': example_id,
            'created_at': datetime.utcnow(),
            'data_types': list(example.keys()),
            'label': example.get('label')
        })
        
        return example_id
    
    def get_training_batch(self, batch_size: int) -> List[Dict]:
        """Retrieve a batch of training examples."""
        # Get metadata
        examples = self.metadata_store.find({}).limit(batch_size)
        
        batch = []
        for ex in examples:
            example = {'id': ex['_id']}
            
            # Load structured features
            if 'structured_features' in ex['data_types']:
                example['features'] = self.structured_store.get(ex['_id'])
            
            # Load image path (lazy loading)
            if 'image' in ex['data_types']:
                example['image_path'] = f"training_data/{ex['_id']}/image.jpg"
            
            batch.append(example)
        
        return batch
```

---

## 📊 Storage Cost Optimization

### Intelligent Tiering Strategy

```python
class StorageOptimizer:
    """Automatically optimize storage costs based on access patterns."""
    
    def __init__(self, s3_client, cloudwatch_client):
        self.s3 = s3_client
        self.cloudwatch = cloudwatch_client
        
    def analyze_access_patterns(self, bucket: str, prefix: str) -> Dict:
        """Analyze data access patterns."""
        # Get access metrics
        metrics = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/S3',
            MetricName='GetRequests',
            Dimensions=[
                {'Name': 'BucketName', 'Value': bucket},
                {'Name': 'FilterId', 'Value': prefix}
            ],
            Period=86400,  # Daily
            Statistics=['Sum'],
            StartTime=datetime.now() - timedelta(days=90),
            EndTime=datetime.now()
        )
        
        return self._categorize_access(metrics)
    
    def recommend_storage_class(self, access_pattern: Dict) -> str:
        """Recommend storage class based on access patterns."""
        avg_daily_access = access_pattern['avg_daily_requests']
        last_access_days = access_pattern['days_since_last_access']
        
        if avg_daily_access > 10:
            return 'STANDARD'
        elif avg_daily_access > 1:
            return 'STANDARD_IA'
        elif last_access_days < 30:
            return 'INTELLIGENT_TIERING'
        elif last_access_days < 90:
            return 'GLACIER_IR'
        else:
            return 'GLACIER_DEEP_ARCHIVE'
    
    def apply_lifecycle_rules(self, bucket: str):
        """Apply lifecycle rules for automatic tiering."""
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'TrainingDataTiering',
                    'Filter': {'Prefix': 'training_data/'},
                    'Status': 'Enabled',
                    'Transitions': [
                        {'Days': 30, 'StorageClass': 'STANDARD_IA'},
                        {'Days': 90, 'StorageClass': 'GLACIER_IR'},
                        {'Days': 365, 'StorageClass': 'GLACIER_DEEP_ARCHIVE'}
                    ]
                },
                {
                    'ID': 'ModelArtifactRetention',
                    'Filter': {'Prefix': 'models/'},
                    'Status': 'Enabled',
                    'Transitions': [
                        {'Days': 90, 'StorageClass': 'STANDARD_IA'}
                    ],
                    'NoncurrentVersionTransitions': [
                        {'NoncurrentDays': 30, 'StorageClass': 'GLACIER'}
                    ]
                }
            ]
        }
        
        self.s3.put_bucket_lifecycle_configuration(
            Bucket=bucket,
            LifecycleConfiguration=lifecycle_config
        )
```

---

## 🎯 Interview Questions

**Q1: How would you design storage for a system that needs both real-time predictions and batch model training?**

**Answer:**
```
Dual-storage architecture:

Online Storage (Real-time):
├── Redis Cluster (features, < 10ms latency)
├── DynamoDB (user profiles, < 20ms)
└── Feature Store (Feast/Tecton)

Offline Storage (Training):
├── Data Lake (S3 + Delta Lake)
├── Data Warehouse (Snowflake/BigQuery)
└── Feature Store (offline part)

Sync Strategy:
1. Write-through: Updates go to both stores
2. Batch sync: Daily ETL from online to offline
3. CDC: Stream changes to keep in sync
```

**Q2: What partitioning strategy would you use for ML training data?**

**Answer:**
```
Depends on access patterns:

1. Time-partitioned (most common):
   s3://bucket/data/year=2024/month=01/day=15/
   - Good for: Time-series data, incremental training
   - Query: Filter by date range

2. Label-partitioned:
   s3://bucket/data/label=fraud/
   s3://bucket/data/label=legitimate/
   - Good for: Imbalanced datasets, stratified sampling
   - Query: Sample from each class

3. Hash-partitioned:
   s3://bucket/data/user_hash=0/ ... /user_hash=99/
   - Good for: Distributed training, user-level splits
   - Query: Parallel reads

4. Hybrid:
   s3://bucket/data/date=2024-01/label=fraud/
   - Good for: Complex access patterns
```

**Q3: How do you handle storage for a petabyte-scale ML dataset?**

**Answer:**
- Use **columnar formats** (Parquet, ORC) for compression
- Implement **data lifecycle** policies
- Use **partition pruning** for efficient queries
- Consider **data sampling** for development
- Use **distributed storage** (HDFS, S3)
- Implement **incremental processing**

---

## 🔑 Key Takeaways

1. **Use the right storage** - different types for different purposes
2. **Optimize for access patterns** - partition and cache appropriately
3. **Manage lifecycle** - move data to cheaper storage over time
4. **Monitor costs** - track and optimize storage usage
5. **Version everything** - track data changes over time
6. **Consider lakehouse** - best of lakes and warehouses
7. **Multi-modal support** - different storage for different data types

---

## 📚 Further Reading

- [Data Lake vs Data Warehouse](https://aws.amazon.com/analytics/data-lake-vs-data-warehouse/)
- [Feature Store Guide](https://www.featurestore.org/)
- [Delta Lake Documentation](https://delta.io/learn/getting-started)
- [The Lakehouse Architecture](https://databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html)

---

## 🔗 Related Topics

- [Data Collection](./01-data-collection.md)
- [Data Versioning](./03-data-versioning.md)
- [Data Quality](./04-data-quality.md)
