# Vector Databases

## Overview

A **vector database** stores high-dimensional vectors and provides similarity search at scale. Unlike in-memory ANN libraries (FAISS), vector databases offer persistence, filtering, replication, and managed infrastructure—making them suitable for production workloads.

Use a vector database when you need: persistence across restarts, metadata filtering (hybrid search), multi-tenant isolation, or managed scalability. Use in-memory FAISS when you have a single-tenant, in-process workload where you control the full stack.

---

## When to Use a Vector DB vs In-Memory ANN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Vector DB vs In-Memory ANN                                                   │
│                                                                               │
│  Use Vector DB when:                    Use FAISS/In-Memory when:             │
│  • Need persistence                     • Single application, single tenant   │
│  • Need metadata filtering              • Latency-critical (< 5ms)            │
│  • Multi-tenant (many apps)             • Full control over infra             │
│  • Managed service / ops simplicity     • Cost-sensitive at scale             │
│  • Scale across many nodes              • Embedding service co-located        │
│  • Real-time updates                    • Predictable memory footprint        │
│                                                                               │
│  Hybrid: Pre-filter in DB, then ANN; or ANN in app, DB for metadata only      │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Criterion | Vector DB | In-Memory ANN |
|-----------|-----------|---------------|
| **Latency** | 5-20ms (network + search) | 1-5ms (in-process) |
| **Persistence** | Yes | No (rebuild from source) |
| **Filtering** | Native (metadata + vector) | Requires post-filter or custom |
| **Ops** | Managed or self-hosted | Self-managed |
| **Scale** | Horizontal (sharding) | Vertical (bigger machine) |

---

## Vector Database Comparison

| System | Type | Strengths | Limitations | Best For |
|--------|------|-----------|-------------|----------|
| **Pinecone** | Managed SaaS | Easy setup, serverless, good docs | Cost at scale, vendor lock-in | Startups, prototypes |
| **Milvus** | Open-source | Scalable, rich features, cloud-native | Operational complexity | Large-scale self-hosted |
| **Weaviate** | Open-source | GraphQL, modules, hybrid search | Smaller community | Semantic search + filters |
| **Qdrant** | Open-source | Rust, payload filtering, good perf | Newer | Production self-hosted |
| **pgvector** | PostgreSQL extension | SQL, ACID, familiar | Not optimized for 100M+ | Existing Postgres users |
| **Chroma** | Open-source | Simple, embeddings-first | Less battle-tested | Local dev, small prod |
| **Elasticsearch** | Search engine | Full-text + kNN, mature | kNN is secondary | Hybrid text + vector |

---

## Architecture of a Vector Database

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Vector Database Internal Architecture                                       │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  API Layer (gRPC/REST)                                                │    │
│  │  • Upsert, Delete, Query                                              │    │
│  │  • Filter expressions (metadata)                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Query Planner                                                        │    │
│  │  • Parse filter → predicate                                            │    │
│  │  • Choose: pre-filter, post-filter, or hybrid                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                          │
│          ┌─────────────────────────┼─────────────────────────┐               │
│          ▼                         ▼                         ▼               │
│  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐       │
│  │  Vector Index │         │  Metadata     │         │  Storage      │       │
│  │  (HNSW/IVF)   │         │  (B-tree/etc) │         │  (WAL, SST)   │       │
│  └───────────────┘         └───────────────┘         └───────────────┘       │
│          │                         │                         │               │
│          └─────────────────────────┼─────────────────────────┘               │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Sharding & Replication                                               │    │
│  │  • Shard by collection/partition                                      │    │
│  │  • Replicas for HA and read scaling                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hybrid Search (Vector + Metadata Filtering)

Vector search alone often isn't enough. Production systems need **filtering by metadata** (category, price, date, user_id) before or after vector search.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Hybrid Search Strategies                                                    │
│                                                                               │
│  Pre-filter:  Filter first → Vector search on subset                         │
│  • Pro: Correct semantics                                                    │
│  • Con: May over-filter; empty results                                      │
│                                                                               │
│  Post-filter: Vector search first → Filter top-K×N → Take top-K              │
│  • Pro: Always get results                                                   │
│  • Con: Need K×N to ensure K after filter; more compute                      │
│                                                                               │
│  Hybrid (BM25 + Vector): Combine keyword score + vector score                │
│  • score = α × BM25(query, doc) + (1-α) × sim(query_emb, doc_emb)           │
│  • Used in: Elasticsearch, Weaviate                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Python: Milvus Hybrid Search Example

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def setup_milvus_collection(dim: int = 128, collection_name: str = "items"):
    """Create Milvus collection with vector + scalar fields."""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="item_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="price", dtype=DataType.FLOAT),
    ]
    schema = CollectionSchema(fields=fields, description="Item embeddings with metadata")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create HNSW index
    index_params = {
        "metric_type": "IP",  # Inner product for normalized vectors
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def query_with_filter(collection, query_vector, filter_expr, top_k=10):
    """Query with metadata filter. Milvus supports pre-filter or post-filter."""
    search_params = {"metric_type": "IP", "params": {"ef": 128}}
    
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=filter_expr  # e.g., 'category == "electronics" and price < 100'
    )
    return results
```

### Python: pgvector Example

```python
# pgvector: Vector search inside PostgreSQL
# CREATE EXTENSION vector;
# CREATE TABLE items (id SERIAL PRIMARY KEY, item_id VARCHAR(64), embedding vector(128), category VARCHAR(64));

import psycopg2
from psycopg2.extras import execute_values
import numpy as np

def pgvector_query(conn, query_embedding: np.ndarray, category: str = None, top_k: int = 10):
    """Query pgvector with optional filter."""
    vec = query_embedding.astype(np.float32).tobytes()
    
    if category:
        sql = """
            SELECT item_id, embedding <=> %s::vector as distance
            FROM items
            WHERE category = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur = conn.cursor()
        cur.execute(sql, (vec, category, vec, top_k))
    else:
        sql = """
            SELECT item_id, embedding <=> %s::vector as distance
            FROM items
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur = conn.cursor()
        cur.execute(sql, (vec, vec, top_k))
    
    return cur.fetchall()
```

---

## Sharding and Replication Strategies

### Sharding

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **By collection** | Each collection on different shards | Simple | Uneven if one collection is huge |
| **By partition key** | Hash(category) or hash(id) | Balanced load | Need to query all for global search |
| **By vector** | Cluster vectors; shard by cluster | Locality | Complex, repartition on growth |

### Replication

- **Sync replication**: Write to N replicas before ack; strong consistency, higher latency
- **Async replication**: Replicate in background; eventual consistency, lower write latency
- **Read replicas**: Scale reads; typically 2-3 replicas per shard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Example: 1B Vectors, 10 Shards, 2 Replicas                                  │
│                                                                               │
│  Shard 0 (100M) ──┬── Replica 0a  (50GB RAM each)                            │
│                   └── Replica 0b                                              │
│  Shard 1 (100M) ──┬── Replica 1a                                            │
│                   └── Replica 1b                                              │
│  ...                                                                          │
│  Total: 10 × 100M = 1B vectors; 20 nodes (10 × 2 replicas)                    │
│  Query: Fan-out to 10 shards; merge top-K                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Production Deployment Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Managed SaaS** | Pinecone, Zilliz Cloud | Fastest time-to-production |
| **Self-hosted K8s** | Milvus, Qdrant on Kubernetes | Full control, cost at scale |
| **Embedded** | FAISS in app process | Lowest latency, single-tenant |
| **Hybrid** | Vector DB for storage + FAISS cache | Hot path in-memory, cold in DB |

### Cost Considerations

| Factor | Impact |
|--------|--------|
| **Vector dimension** | 128 vs 512 = 4× storage & index size |
| **Quantization** | PQ reduces storage 10-50× |
| **Replication** | 2× replicas = 2× cost |
| **Managed vs self-host** | SaaS: $0.09-0.25/百万 vectors/month; self-host: infra cost |

---

## Python: FAISS + Metadata (Lightweight Vector Store Pattern)

When a full vector DB is overkill, combine FAISS with a key-value store for metadata:

```python
import faiss
import numpy as np
import redis  # or any KV store

class SimpleVectorStore:
    """FAISS for vectors + Redis for metadata. Good for < 100M vectors."""
    
    def __init__(self, dim: int, redis_client=None):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.redis = redis_client
        self.id_to_idx = {}  # External ID -> FAISS index
        self.idx_to_id = []
    
    def add(self, ids: list, vectors: np.ndarray, metadata: list = None):
        """Add vectors with optional metadata."""
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        
        start = self.index.ntotal
        self.index.add(vectors)
        
        for i, id in enumerate(ids):
            idx = start + i
            self.id_to_idx[id] = idx
            self.idx_to_id.append(id)
            if metadata and self.redis:
                self.redis.hset(f"meta:{id}", mapping=metadata[i])
    
    def search(self, query: np.ndarray, k: int = 10, filter_fn=None):
        """Search with optional metadata filter."""
        query = query.astype(np.float32)
        faiss.normalize_L2(np.expand_dims(query, 0))
        
        fetch_k = k * 10 if filter_fn else k  # Over-fetch for post-filter
        scores, indices = self.index.search(query.reshape(1, -1), fetch_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                break
            id = self.idx_to_id[idx]
            if filter_fn and not filter_fn(id):
                continue
            meta = self.redis.hgetall(f"meta:{id}") if self.redis else {}
            results.append((id, float(score), meta))
            if len(results) >= k:
                break
        return results
```

---

## Index Updates and Consistency

| Update Type | Strategy | Latency | Consistency |
|-------------|----------|---------|-------------|
| **Bulk upsert** | Rebuild index in background; swap | Minutes-hours | Eventually |
| **Incremental add** | HNSW supports insert | Milliseconds | Immediate |
| **Delete** | Soft delete + periodic compact | Varies | Eventually |
| **Update** | Delete + insert (for vectors) | 2× write | Immediate |

**Best practice**: Batch updates (e.g., nightly) for item embeddings; avoid per-request index mutation in hot path.

---

## Query Optimization Tips

1. **Reduce dimensions**: 128 often sufficient; 256-512 for text. Lower = faster.
2. **Use appropriate metric**: IP for normalized (cosine); L2 for raw. Avoid conversion overhead.
3. **Tune top_k**: Request only what you need; over-fetching costs latency.
4. **Pre-filter when selective**: If `category = X` reduces to 1% of data, pre-filter saves work.
5. **Connection pooling**: Reuse connections; vector DBs are network-bound.

---

## Production Checklist

- [ ] Chosen index type (HNSW/IVF) and tuned parameters
- [ ] Sharding strategy for scale (if > 100M vectors)
- [ ] Replication for HA (min 2 replicas)
- [ ] Monitoring: latency p99, recall@k, error rate
- [ ] Index rebuild pipeline (batch refresh schedule)
- [ ] Backup/restore for vector + metadata
- [ ] Cost projection at 1M, 10M, 100M vectors

---

## Detailed System Comparison

### Pinecone
- **Pricing**: ~$0.096 per 1M vectors/month (serverless); dedicated from ~$70/mo
- **Scale**: Billions of vectors; auto-scaling
- **Features**: Namespaces, metadata filter, sparse+dense hybrid
- **Use case**: Fast prototyping; teams without infra

### Milvus
- **Deployment**: K8s, Docker; Zilliz Cloud managed
- **Scale**: 1B+ vectors per cluster
- **Features**: Multiple index types, GPU support, attu UI
- **Use case**: Self-hosted at scale; full control

### Qdrant
- **Stack**: Rust; good single-node performance
- **Features**: Payload filtering, sparse vectors, groups
- **Use case**: Production self-hosted; moderate scale

### pgvector
- **Integration**: PostgreSQL extension; use existing SQL skills
- **Scale**: ~10M vectors on single node; not for 100M+
- **Features**: HNSW, IVFFlat indexes; ACID
- **Use case**: Already on Postgres; smaller vector workloads

---

## Migration Path: FAISS to Vector DB

When moving from in-memory FAISS to a vector database:

1. **Export**: Dump vectors + IDs from FAISS index
2. **Schema**: Define collection with vector + metadata fields
3. **Load**: Bulk insert; create index
4. **Switch**: Update client to use vector DB SDK; keep FAISS as fallback
5. **Validate**: Compare recall and latency; tune parameters
6. **Deprecate**: Remove FAISS once vector DB is stable

---

## Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| **Vector DB vs FAISS** | Managed DB | In-memory FAISS | DB: persistence, filtering, ops. FAISS: lowest latency |
| **Pre vs post-filter** | Pre-filter | Post-filter | Pre: exact, may empty. Post: always results, over-fetch |
| **Sharding** | Few big shards | Many small shards | Few: less merge overhead. Many: more parallelism |
| **Replication** | 1 replica | 2-3 replicas | 1: lower cost. 2-3: HA, read scaling |
| **Quantization** | Full precision | PQ | Full: exact. PQ: 10× less memory, ~5% recall loss |

---

## Interview Tips

1. **"When would you use a vector DB vs FAISS?"** → Vector DB: multi-tenant, persistence, filtering, managed. FAISS: single-app, latency-critical, full control.
2. **"How do you do hybrid search?"** → Pre-filter (if filter is selective), post-filter (if not), or fused BM25+vector score.
3. **"How do you shard a vector DB?"** → By partition key (e.g., category) or random; fan-out query and merge.
4. **"How do you handle 10B vectors?"** → Shard (e.g., 100 × 100M), PQ for memory, disk-backed if needed.
5. **"Pinecone vs Milvus?"** → Pinecone: easy, managed. Milvus: open-source, scalable, self-hosted.

---

## Related Topics

- [Approximate Nearest Neighbors](./02-approximate-nearest-neighbors.md) – Algorithms inside vector DBs
- [Two-Tower Architecture](./04-two-tower-architecture.md) – Produces embeddings for the DB
- [Recommendation Systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) – Primary consumer
- [Search Systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/02-search-systems.md) – Hybrid text + vector
