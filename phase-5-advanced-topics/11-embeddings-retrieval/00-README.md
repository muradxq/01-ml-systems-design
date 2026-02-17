# Embeddings & Retrieval

## Overview

Embeddings and retrieval form the **foundational infrastructure** for modern ML systems. Whether you're building recommendation engines (Netflix, Spotify), search systems (Google, Amazon), ad targeting (Meta, Google Ads), People You May Know (LinkedIn, Facebook), or RAG-powered LLM applications—the ability to efficiently map entities to dense vectors and retrieve similar items at scale is the common denominator.

This section covers the complete pipeline: from creating high-quality embeddings through Word2Vec, contrastive learning, and embedding tables; scaling retrieval with approximate nearest neighbor (ANN) algorithms; choosing and operating vector databases; and architecting production two-tower systems. Mastery of these concepts is essential for ML Systems Design interviews and for building production retrieval systems.

---

## Why Embeddings Matter

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Embeddings: The Universal Representation Layer             │
│                                                                               │
│  Raw Entities          Embedding Model             Dense Vectors              │
│  ┌─────────────┐      ┌──────────────────┐       ┌─────────────────────┐    │
│  │ "user_123"  │─────▶│ User Tower /     │──────▶│ [0.2, -0.5, 0.8,...]│    │
│  │ "item_456"  │      │ Item Tower       │       │ dim=128              │    │
│  │ "query:..." │      │ BERT / ResNet    │       │                      │    │
│  │ Image bytes │      │ Two-Tower        │       │ Similarity = dot()   │    │
│  └─────────────┘      └──────────────────┘       └─────────────────────┘    │
│                                                                               │
│  Key Properties:                                                              │
│  • Semantically similar entities → similar vectors (cosine/dot)              │
│  • Enables: recommendations, search, clustering, classification               │
│  • Scale: 1B items × 128 dim = 512GB uncompressed                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Embedding Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Embedding Retrieval System: End-to-End Flow                   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         OFFLINE / BATCH                                   │   │
│  │                                                                           │   │
│  │   Training Data ──▶ Embedding Model ──▶ Item Embeddings ──▶ ANN Index    │   │
│  │   (clicks, co-views)   (Two-Tower)      (1B items × 128d)   (HNSW/IVF)   │   │
│  │                                                                           │   │
│  │   Index Build: 2-4 hours for 1B items    │  Stored in Vector DB          │   │
│  └──────────────────────────────────────────┼───────────────────────────────┘   │
│                                             │                                    │
│                                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         ONLINE / REAL-TIME                                │   │
│  │                                                                           │   │
│  │   Request ──▶ User Embedding ──▶ ANN Query ──▶ Top-K Items ──▶ Response  │   │
│  │   (user_id)     (10-50ms)         (1-5ms)      (100-1000)                 │   │
│  │                    │                  │                                    │   │
│  │                    │                  │  Latency SLA: < 100ms p99          │   │
│  │                    │                  │  Throughput: 100K QPS              │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Key Metrics                                                              │   │
│  │  • Recall@10: 0.85-0.95 (ANN vs exact)                                   │   │
│  │  • Latency: 10ms user emb + 5ms ANN + 50ms ranker = 65ms                  │   │
│  │  • Memory: ~500GB for 1B × 128d (float32); ~125GB with PQ                  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Embedding Fundamentals](./01-embedding-fundamentals.md) | Word2Vec, Item2Vec, contrastive learning, embedding tables, quality evaluation |
| 2 | [Approximate Nearest Neighbors](./02-approximate-nearest-neighbors.md) | HNSW, IVF, Product Quantization, ScaNN; indexing and query trade-offs |
| 3 | [Vector Databases](./03-vector-databases.md) | Pinecone, Milvus, Weaviate, pgvector; hybrid search; production patterns |
| 4 | [Two-Tower Architecture](./04-two-tower-architecture.md) | Dual encoders, negative sampling, serving, cold start, extensions |

---

## Key Concepts Summary

### Embedding Creation
- **Word2Vec**: CBOW/Skip-gram for word embeddings; transferable to items (Item2Vec)
- **Contrastive learning**: SimCLR, triplet loss, InfoNCE for learning similarity
- **Embedding tables**: Categorical features → learned vectors; hashing trick for huge vocabularies

### Retrieval at Scale
- **Exact K-NN**: O(n) per query—impractical for 1B+ items
- **ANN**: HNSW (high recall, low latency), IVF (sharding), PQ (compression)
- **Recall@k vs QPS trade-off**: Higher recall → more compute → lower QPS

### Production Patterns
- **Offline precompute**: Item embeddings + ANN index built in batch
- **Online compute**: User embedding per request; ANN lookup
- **Two-stage**: ANN retrieval (1M→1K) → ranking model (1K→10)

### Scale Reference
| Scale | Items | Latency Target | Typical Approach |
|-------|-------|----------------|------------------|
| Small | < 1M | < 10ms | Exact search, in-memory |
| Medium | 1M-100M | < 50ms | FAISS, single-node ANN |
| Large | 100M-1B | < 100ms | Sharded vector DB, HNSW |
| XLarge | 1B+ | < 100ms | Distributed ANN, PQ compression |

---

## How This Section Connects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Embeddings & Retrieval in the ML Stack                     │
│                                                                               │
│  Upstream Dependencies:                                                       │
│  • Feature Engineering: Embedding tables use categorical features            │
│  • Data Management: Training data (clicks, co-views) for contrastive loss   │
│  • Model Training: Two-tower training, negative sampling                     │
│                                                                               │
│  Downstream Consumers:                                                        │
│  • Recommendation Systems: Candidate generation via ANN                       │
│  • Search Systems: Semantic search over embeddings                            │
│  • NLP/LLM Systems: RAG retrieval, dense passage retrieval                   │
│  • Ads / PYMK: User-item matching at scale                                   │
│                                                                               │
│  This section = the retrieval layer shared by all of the above                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Preparation Focus

When discussing embeddings and retrieval in ML Systems Design interviews:

1. **Start with the problem**: "We need to find similar items from 1B candidates in < 50ms"
2. **Explain the pipeline**: Embedding model → ANN index → query flow
3. **Quantify trade-offs**: Recall vs latency, memory vs accuracy
4. **Know production details**: Index updates, sharding, monitoring
5. **Connect to use cases**: Recs, search, ads, RAG—all use this stack

Continue to:
1. [Embedding Fundamentals](./01-embedding-fundamentals.md)
2. [Approximate Nearest Neighbors](./02-approximate-nearest-neighbors.md)
3. [Vector Databases](./03-vector-databases.md)
4. [Two-Tower Architecture](./04-two-tower-architecture.md)
