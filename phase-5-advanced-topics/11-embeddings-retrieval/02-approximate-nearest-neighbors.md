# Approximate Nearest Neighbors (ANN)

## Overview

Given a query vector and a corpus of N vectors, **nearest neighbor search** finds the K vectors most similar to the query. **Exact** K-NN requires computing similarity to every vector—O(N) per query—which becomes impractical at scale: 1B items × 1ms per distance ≈ 11 days per query.

**Approximate Nearest Neighbor (ANN)** algorithms trade a small recall loss for orders-of-magnitude speedup. At production scale (100M–1B items), ANN is the only viable approach for real-time retrieval with sub-100ms latency.

---

## Why Exact K-NN Doesn't Scale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exact K-NN: Brute Force Complexity                                            │
│                                                                               │
│  For each query:                                                              │
│  1. Compute distance to all N vectors: O(N × D)                               │
│  2. Partial sort for top-K: O(N log K)                                        │
│                                                                               │
│  Scale Analysis (D=128, float32):                                             │
│  • N=1M:    ~0.5GB memory, ~50ms query (single-threaded)                      │
│  • N=100M:  ~50GB memory, ~5s query                                          │
│  • N=1B:    ~500GB memory, ~50s query                                         │
│                                                                               │
│  Target: 100K QPS, 10ms p99 → Exact K-NN impossible at 1B scale               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key ANN Algorithms

### HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer graph where each layer is a subset of the previous. Search starts at the top (sparse) layer and "navigates" down to the dense bottom layer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HNSW Graph Structure                                                         │
│                                                                               │
│  Layer 2 (sparse):     ●────●         ●                                       │
│                           \   \     /                                        │
│  Layer 1:              ●───●───●───●───●───●                                 │
│                            \   \ /   \   /                                   │
│  Layer 0 (dense):       ●─●─●─●─●─●─●─●─●─●─●─●  (all points)               │
│                                                                               │
│  Query flow: Start at top layer → greedy nearest → drop to next layer → ...  │
│  Search complexity: O(log N) with high probability                           │
│  Build: O(N log N)                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Parameters**: `efConstruction` (build quality), `efSearch` (query-time recall/speed trade-off), `M` (max neighbors per node).

**HNSW Search Algorithm**:
1. Start at top layer; greedily move to nearest neighbor
2. Repeat until no closer neighbor; then "drop down" to next layer
3. Continue until layer 0; return best candidates

**Why it works**: Sparse top layers provide "express lanes" for long-range jumps; dense bottom layer ensures local accuracy.

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| M | Degree of graph; higher = better recall, more memory | 16-64 |
| efConstruction | Build-time search breadth | 100-500 |
| efSearch | Query-time search breadth | 50-500 |

### IVF (Inverted File Index)

Partition the space into clusters (e.g., via k-means). At query time, search only the nearest cluster(s).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  IVF Structure                                                                │
│                                                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                          │
│  │Cluster 1│  │Cluster 2│  │Cluster 3│  │Cluster 4│  ... (nlist=1000)        │
│  │ ● ● ●   │  │   ● ●   │  │ ● ● ● ● │  │     ●   │                          │
│  │   ●     │  │ ●   ●   │  │   ●     │  │   ●   ● │                          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                          │
│                                                                               │
│  Query: 1. Find nearest nprobe clusters (e.g., nprobe=32)                      │
│         2. Search only vectors in those clusters                             │
│                                                                               │
│  Trade-off: nprobe ↑ → recall ↑, latency ↑                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Product Quantization (PQ)

Compress vectors by splitting into sub-vectors and quantizing each to a small codebook. Enables in-memory distance estimation without full reconstruction.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Product Quantization                                                         │
│                                                                               │
│  Original: [d₁, d₂, ..., d₅₁₂]  (512 floats)                                │
│               │         │         │         │                                 │
│  Split:    [sub1]   [sub2]   [sub3]   [sub4]  (4 × 128 dims)                │
│               │         │         │         │                                 │
│  Quantize: c1      c2      c3      c4      (4 × 8 bits = 32 bits total)      │
│                                                                               │
│  Distance: Sum of chunk distances (precomputed lookup table)                  │
│  Compression: 512 × 4 bytes → 4 bytes (128×)                                 │
│  1B × 128d: 512GB → 4GB with m=4, nbits=8                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ScaNN (Scalable Nearest Neighbors)

Google's ScaNN combines partitioning, quantization, and reordering. Often used as IVF+PQ with anharmonic scoring for better recall.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ScaNN Architecture (Simplified)                                              │
│                                                                               │
│  1. Partition: Tree or IVF-style clustering                                   │
│  2. Quantize: Residual PQ (quantize residual from partition center)           │
│  3. Reorder: Anharmonic scoring for asymmetric distance                      │
│                                                                               │
│  Key innovation: Better distance estimation than standard PQ                  │
│  Used at scale: Google Search, YouTube recommendations                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Disk-Based ANN (When Memory Doesn't Fit)

For 10B+ vectors that don't fit in RAM:
- **DiskANN**: SSD-optimized graph index (Microsoft)
- **HNSW on disk**: Memory-map index; SSD bandwidth limits QPS
- **Partitioned search**: Load shards on demand

---

## Algorithm Comparison

| Algorithm | Build Time | Query Time | Memory | Recall@10 | Best For |
|-----------|------------|------------|--------|-----------|----------|
| **Brute force** | O(1) | O(N) | O(N×D) | 1.0 | N < 100K |
| **HNSW** | O(N log N) | O(log N) | O(N×M) | 0.95-0.99 | High recall, low latency |
| **IVF** | O(N×nlist) | O(nprobe×N/nlist) | O(N×D) | 0.85-0.95 | Large scale, tunable |
| **IVF+PQ** | O(N×nlist) | O(nprobe×m) | O(N×m/8) | 0.80-0.92 | Memory-constrained |
| **ScaNN** | Medium | Low | Medium | 0.90-0.98 | Production (Google) |

---

## Index Build vs Query Time Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Build Time vs Query Time (1B vectors, 128 dim)                               │
│                                                                               │
│  Algorithm      Build Time      Query (1 thread)   Recall@10                  │
│  ─────────────────────────────────────────────────────────────               │
│  HNSW           ~2-4 hours      ~1-3 ms           0.97                       │
│  IVF-4096       ~1-2 hours      ~2-5 ms           0.90                       │
│  IVF-PQ         ~30 min         ~0.5-1 ms          0.85                       │
│  ScaNN          ~1 hour         ~1-2 ms            0.94                       │
│                                                                               │
│  Rule: Higher build quality (efConstruction, nlist) → better recall           │
│        Higher query params (efSearch, nprobe) → better recall, slower         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Python: FAISS Index Creation and Querying

FAISS (Facebook AI Similarity Search) is the most widely used ANN library.

```python
import numpy as np
import faiss
import time

def create_hnsw_index(dimension: int, n_vectors: int, M: int = 32, ef_construction: int = 200):
    """Create HNSW index for approximate nearest neighbor search."""
    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    
    # For cosine similarity: L2-normalize vectors, use inner product
    # index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)
    
    return index


def create_ivf_index(dimension: int, n_vectors: int, nlist: int = 1024, nprobe: int = 32):
    """Create IVF index. Requires training on representative vectors."""
    quantizer = faiss.IndexFlatIP(dimension)  # or IndexFlatL2
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe
    return index


def create_ivf_pq_index(dimension: int, n_vectors: int, nlist: int = 4096, 
                        m: int = 8, n_bits: int = 8, nprobe: int = 64):
    """IVF + Product Quantization for memory-efficient search."""
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, n_bits)
    index.nprobe = nprobe
    return index


def benchmark_index(index: faiss.Index, vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark: build time, query latency, recall."""
    dim = vectors.shape[1]
    n = vectors.shape[0]
    
    # L2 normalize for cosine via inner product
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(queries)
    
    # Train if needed (IVF)
    if index.is_trained == False:
        train_size = min(100000, n)
        print(f"Training on {train_size} vectors...")
        index.train(vectors[:train_size].astype(np.float32))
    
    # Add vectors
    t0 = time.time()
    index.add(vectors.astype(np.float32))
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s for {n} vectors")
    
    # Query
    n_queries = queries.shape[0]
    t0 = time.time()
    _, I = index.search(queries.astype(np.float32), k)
    query_time = (time.time() - t0) / n_queries * 1000
    print(f"Query latency: {query_time:.3f} ms per query")
    print(f"QPS (single thread): {1000/query_time:.0f}")
    
    # Recall: compare to brute force
    brute = faiss.IndexFlatIP(dim)
    brute.add(vectors.astype(np.float32))
    _, I_gt = brute.search(queries.astype(np.float32), k)
    
    recall = (I[:, :] == I_gt[:, :]).any(axis=1).mean()
    print(f"Recall@{k}: {recall:.4f}")
    
    return {"build_time": build_time, "query_ms": query_time, "recall": recall}


def faiss_gpu_example():
    """FAISS GPU for 10-100× speedup at scale."""
    # res = faiss.StandardGpuResources()
    # index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    # index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    # index_gpu.add(vectors)
    # D, I = index_gpu.search(queries, k)
    pass


# Full pipeline example
def main():
    np.random.seed(42)
    n, dim, n_queries = 1_000_000, 128, 1000
    vectors = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # HNSW
    index_hnsw = create_hnsw_index(dim, n)
    benchmark_index(index_hnsw, vectors.copy(), queries, k=10)
    
    # IVF
    index_ivf = create_ivf_index(dim, n)
    benchmark_index(index_ivf, vectors.copy(), queries, k=10)
```

---

## Benchmarking: Recall@K vs QPS, Memory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Typical Benchmark Results (1M vectors, 128 dim, 1 thread)                   │
│                                                                               │
│  Index Type    Recall@10   QPS      Memory   Build Time                       │
│  ─────────────────────────────────────────────────────────────               │
│  Flat (exact)  1.00        ~2K     ~500MB    instant                          │
│  HNSW M=32     0.98        ~50K    ~600MB    ~30s                             │
│  HNSW M=64     0.99        ~30K    ~700MB    ~60s                             │
│  IVF 1024      0.92        ~80K    ~500MB    ~10s                             │
│  IVF 4096      0.95        ~40K    ~500MB    ~20s                             │
│  IVF-PQ m=8    0.85        ~100K   ~50MB     ~15s                             │
│                                                                               │
│  Scaling to 1B: Use sharding (e.g., 10 shards of 100M each)                   │
│  Query: Fan-out to shards, merge top-K from each                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Production Considerations

### Index Updates

| Update Type | HNSW | IVF | IVF-PQ |
|-------------|------|-----|--------|
| **Add** | Yes (online insert) | Rebuild or incremental | Rebuild |
| **Delete** | Mark deleted (or rebuild) | Rebuild | Rebuild |
| **Full refresh** | Rebuild | Rebuild | Rebuild |

**Pattern**: Build new index in background; atomically swap when ready. Dual-index for zero-downtime.

### Sharding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Sharded ANN for 1B+ Vectors                                                 │
│                                                                               │
│  Query ──▶ Router ──┬──▶ Shard 1 (100M) ──▶ Top-100                           │
│                     ├──▶ Shard 2 (100M) ──▶ Top-100                           │
│                     ├──▶ Shard 3 (100M) ──▶ Top-100                           │
│                     └──▶ ...                                                   │
│                              │                                                │
│                              ▼                                                │
│                     Merge & Re-rank ──▶ Top-K                                 │
│                                                                               │
│  Sharding key: User ID hash, Item category, or random                        │
│  Trade-off: More shards = less memory per node, higher fan-out latency       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Replication

- **Read replicas**: Each shard replicated 2-3× for availability and load spreading
- **Query routing**: Round-robin or least-loaded

---

## Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| **Algorithm** | HNSW | IVF-PQ | HNSW: better recall, more memory. IVF-PQ: 10× less memory, lower recall |
| **efSearch / nprobe** | Low | High | Low: fast, lower recall. High: slower, better recall |
| **Sharding** | Single node | Multi-shard | Single: simpler. Multi: scales to 1B+, adds merge overhead |
| **Metric** | Inner product | L2 | IP for normalized vectors (cosine). L2 for raw embeddings |
| **Quantization** | None | PQ | None: exact distances. PQ: 10-100× compression, approximate |

---

## Interview Tips

1. **"Why ANN instead of exact?"** → 1B vectors × 128 dim = O(10^11) ops per query; need sub-10ms → ANN essential.
2. **"HNSW vs IVF?"** → HNSW: better recall, graph structure. IVF: simpler, easier to shard, good for very large scale.
3. **"How do you choose efSearch?"** → Trade recall vs latency; tune on held-out set. Typical: 100-500.
4. **"How do you update the index?"** → Batch rebuild + atomic swap. Or incremental for HNSW with delete markers.
5. **"How do you scale to 10B vectors?"** → Shard by key (e.g., category), 100 shards × 100M, merge results.

---

## Parameter Tuning Guide

| Parameter | Algorithm | Low Value | High Value | Recommendation |
|-----------|-----------|-----------|------------|----------------|
| M | HNSW | 16 | 64 | 32 for balanced; 64 for max recall |
| efConstruction | HNSW | 100 | 500 | 200-400; higher = better index quality |
| efSearch | HNSW | 50 | 500 | Start 100; increase until recall saturates |
| nlist | IVF | 256 | 65536 | sqrt(N) to 4×sqrt(N) |
| nprobe | IVF | 8 | 256 | 32-128; tune for recall/latency |
| m | PQ | 4 | 32 | 8-16; must divide dimension |
| n_bits | PQ | 4 | 8 | 8 typically |

---

## Monitoring in Production

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Recall@K** | Sampled vs ground truth | < 0.90 |
| **Query latency p99** | End-to-end ANN query | > 20ms |
| **Index size** | Memory/disk usage | Growth anomaly |
| **Build duration** | Time to rebuild index | > 2× baseline |
| **OOM** | Out of memory during build | Any |

```python
def monitor_ann_recall(index, sample_queries, ground_truth, k=10):
    """Periodic recall check for production ANN."""
    _, retrieved = index.search(sample_queries, k)
    recalls = []
    for i, gt in enumerate(ground_truth):
        hits = len(set(retrieved[i]) & set(gt))
        recalls.append(hits / len(gt) if gt else 0)
    return np.mean(recalls)
```

---

## Distance Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Inner Product (IP)** | u·v | L2-normalized vectors (equivalent to cosine) |
| **L2 (Euclidean)** | ‖u-v‖² | Raw embeddings, when magnitude matters |
| **Cosine** | u·v/(‖u‖‖v‖) | Normalize vectors, then use IP |

**FAISS tip**: For cosine, L2-normalize vectors and use `METRIC_INNER_PRODUCT`. For max similarity, negate L2 distances.

---

## Related Topics

- [Vector Databases](./03-vector-databases.md) – Managed ANN (Pinecone, Milvus)
- [Two-Tower Architecture](./04-two-tower-architecture.md) – Produces embeddings for ANN
- [Embedding Fundamentals](./01-embedding-fundamentals.md) – Creating the vectors
- [Recommendation Systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) – Primary use case
