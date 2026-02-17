# Embedding Fundamentals

## Overview

Embeddings are dense, low-dimensional vector representations of entities (words, items, users, images) that capture semantic similarity. Items that are "similar" in meaning or behavior map to nearby vectors in embedding space. This property enables recommendations (similar items), search (query-document matching), and many downstream ML tasks.

At production scale—Netflix with 17K titles, Amazon with 350M+ products, Pinterest with 100B+ pins—embedding-based retrieval is the only practical approach. This chapter covers the foundational algorithms and production patterns for creating and evaluating embeddings.

---

## Why Embeddings Matter

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    From Sparse to Dense Representations                       │
│                                                                               │
│  Sparse (One-Hot):                    Dense (Embedding):                      │
│  "cat"  → [0,0,1,0,0,...,0]          "cat"  → [0.2, -0.5, 0.8, 0.1, ...]    │
│  10M dim (vocab size)                 128 dim (embedding dim)                 │
│                                                                               │
│  Problems with sparse:                 Benefits of dense:                     │
│  • No notion of similarity            • Similar items → close vectors        │
│  • Curse of dimensionality            • Fixed size regardless of vocab       │
│  • Memory: 10M floats per item        • Enables fast similarity (dot product) │
│  • O(vocab) for similarity            • O(dim) for similarity                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Embeddings learn a geometry where "similar" entities cluster. Cosine similarity or dot product between vectors quantifies this.

---

## Word2Vec

Word2Vec (Mikolov et al., 2013) learns word embeddings from co-occurrence patterns. Two architectures:

### CBOW (Continuous Bag of Words)

Predict the center word from surrounding context words.

```
┌─────────────────────────────────────────────────────────────────┐
│  CBOW Architecture                                                │
│                                                                   │
│  Context: [w₁, w₂, ___, w₄, w₅]  →  Predict: w₃ (center)         │
│              │         │         │                                │
│              ▼         ▼         ▼                                │
│         [emb] [emb] [emb] [emb]                                   │
│              │         │         │                                │
│              └────┬────┴────┬────┘                                │
│                   ▼         ▼                                     │
│              Average → Projection → Softmax → w₃                  │
│                                                                   │
│  Window size: typically 5 (2 before, 2 after)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Skip-gram

Predict context words from the center word. Often performs better for rare words.

```
┌─────────────────────────────────────────────────────────────────┐
│  Skip-gram Architecture                                           │
│                                                                   │
│  Center: w₃  →  Predict: [w₁, w₂, w₄, w₅] (context)              │
│      │                                                             │
│      ▼                                                             │
│  [embedding]  →  Projection  →  Softmax  →  w₁, w₂, w₄, w₅       │
│                                                                   │
│  Negative sampling: Sample K negative words instead of full       │
│  softmax (e.g., K=5) → O(1) per training example vs O(V)          │
└─────────────────────────────────────────────────────────────────┘
```

### Python: Word2Vec Training Example

```python
import numpy as np
from collections import defaultdict

class SimpleWord2Vec:
    """Minimal Skip-gram with negative sampling."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, neg_samples: int = 5):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neg_samples = neg_samples
        # Input (center) and output (context) embeddings
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01
    
    def train_pair(self, center: int, context: int, neg_indices: list, lr: float = 0.01):
        """Train on one (center, context) pair with negative samples."""
        h = self.W_in[center]  # (embed_dim,)
        pos_score = np.dot(h, self.W_out[context])
        neg_scores = np.dot(h, self.W_out[neg_indices].T)
        
        # Sigmoid gradients
        pos_grad = 1 - 1 / (1 + np.exp(-pos_score))
        neg_grads = 1 / (1 + np.exp(neg_scores))
        
        # Update
        self.W_in[center] += lr * (pos_grad * self.W_out[context] - neg_grads @ self.W_out[neg_indices])
        self.W_out[context] += lr * pos_grad * h
        for i, neg in enumerate(neg_indices):
            self.W_out[neg] -= lr * neg_grads[i] * h
    
    def get_embedding(self, word_id: int) -> np.ndarray:
        return self.W_in[word_id]
    
    def similarity(self, id1: int, id2: int) -> float:
        """Cosine similarity between two word embeddings."""
        v1, v2 = self.W_in[id1], self.W_in[id2]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def train_word2vec(corpus: list, vocab: dict, embed_dim: int = 128, epochs: int = 5):
    """Train Word2Vec on tokenized corpus."""
    vocab_size = len(vocab)
    model = SimpleWord2Vec(vocab_size, embed_dim)
    
    # Unigram distribution for negative sampling (raise to 3/4 power)
    word_freq = defaultdict(int)
    for sent in corpus:
        for w in sent:
            word_freq[w] += 1
    total = sum(word_freq.values())
    neg_probs = np.array([(word_freq.get(i, 0) / total) ** 0.75 for i in range(vocab_size)])
    neg_probs /= neg_probs.sum()
    
    for epoch in range(epochs):
        for sent in corpus:
            for i, center in enumerate(sent):
                window = 2
                for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                    if j != i:
                        context = sent[j]
                        neg = np.random.choice(vocab_size, size=5, p=neg_probs, replace=False)
                        neg = [n for n in neg if n != context]
                        model.train_pair(center, context, neg[:5])
    
    return model
```

---

## Item2Vec for Recommendation Systems

Item2Vec applies the Skip-gram idea to item sequences (e.g., purchase history, session clicks). Items that frequently co-occur get similar embeddings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Item2Vec: Items as "Words"                                                   │
│                                                                               │
│  User session: [item_A, item_B, item_C, item_D, item_E]                       │
│                        │                                                       │
│                        ▼                                                       │
│  Training pairs: (B, A), (B, C), (C, B), (C, D), (D, C), (D, E), ...         │
│                                                                               │
│  "item_B and item_C are similar" because they co-occur in user sessions      │
│                                                                               │
│  Use case: "Users who viewed X also viewed Y" → use item embedding similarity │
│  Scale: Spotify (70M songs), Amazon (350M products) use variants            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Python: Item2Vec Training

```python
from collections import defaultdict
import numpy as np

class Item2Vec:
    """Item2Vec: Skip-gram over item sequences (sessions)."""
    
    def __init__(self, num_items: int, embed_dim: int = 64, window: int = 5, neg_samples: int = 5):
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.window = window
        self.neg_samples = neg_samples
        self.W = np.random.randn(num_items, embed_dim).astype(np.float32) * 0.01
    
    def train_sequence(self, seq: list, lr: float = 0.025):
        """Train on one item sequence (e.g., user session)."""
        for i, center in enumerate(seq):
            start = max(0, i - self.window)
            end = min(len(seq), i + self.window + 1)
            for j in range(start, end):
                if j != i:
                    context = seq[j]
                    neg = np.random.randint(0, self.num_items, size=self.neg_samples)
                    neg = [n for n in neg if n != context]
                    self._update(center, context, neg, lr)
    
    def _update(self, center: int, context: int, neg: list, lr: float):
        h = self.W[center]
        pos_score = np.dot(h, self.W[context])
        pos_grad = 1 - 1 / (1 + np.exp(-np.clip(pos_score, -10, 10)))
        self.W[center] += lr * pos_grad * self.W[context]
        self.W[context] += lr * pos_grad * h
        
        for n in neg:
            neg_score = np.dot(h, self.W[n])
            neg_grad = 1 / (1 + np.exp(np.clip(neg_score, -10, 10)))
            self.W[center] -= lr * neg_grad * self.W[n]
            self.W[n] -= lr * neg_grad * h
    
    def get_similar_items(self, item_id: int, k: int = 10, exclude: set = None) -> list:
        """Get top-k similar items by cosine similarity."""
        exclude = exclude or set()
        v = self.W[item_id]
        norms = np.linalg.norm(self.W, axis=1)
        sims = self.W @ v / (norms * np.linalg.norm(v) + 1e-8)
        top = np.argsort(sims)[::-1]
        result = []
        for i in top:
            if i not in exclude and i != item_id:
                result.append((i, float(sims[i])))
                if len(result) >= k:
                    break
        return result
```

---

## Contrastive Learning

Contrastive learning learns embeddings by contrasting positive pairs (similar) against negative pairs (dissimilar).

### Triplet Loss

```
┌─────────────────────────────────────────────────────────────────┐
│  Triplet Loss: (anchor, positive, negative)                       │
│                                                                   │
│  L = max(0, d(a,p) - d(a,n) + margin)                            │
│                                                                   │
│  Goal: Pull anchor closer to positive than to negative            │
│  Margin: typically 0.2 - 0.5                                     │
│                                                                   │
│  Used in: Face recognition, image retrieval                      │
└─────────────────────────────────────────────────────────────────┘
```

### InfoNCE / NT-Xent (SimCLR-style)

```
┌─────────────────────────────────────────────────────────────────┐
│  InfoNCE: Multi-negative contrastive loss                         │
│                                                                   │
│  For batch of N pairs (x_i, y_i):                                 │
│  L_i = -log( exp(sim(x_i,y_i)/τ) / Σ_j exp(sim(x_i,y_j)/τ) )    │
│                                                                   │
│  • In-batch negatives: All other items in batch are negatives     │
│  • Temperature τ: 0.05-0.1 typical; lower = sharper              │
│  • Used by: SimCLR, MoCo, two-tower models                       │
└─────────────────────────────────────────────────────────────────┘
```

### Python: Contrastive Loss Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss_triplet(anchor: torch.Tensor, positive: torch.Tensor, 
                             negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Triplet loss."""
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()


def contrastive_loss_infonce(queries: torch.Tensor, keys: torch.Tensor, 
                              temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE / NT-Xent: queries (N, D), keys (N, D). 
    Assumes (query[i], key[i]) is the positive pair.
    """
    # L2 normalize
    queries = F.normalize(queries, dim=1)
    keys = F.normalize(keys, dim=1)
    
    # Logits: (N, N) - query i vs all keys
    logits = (queries @ keys.T) / temperature
    
    # Labels: positive is diagonal
    labels = torch.arange(queries.size(0), device=queries.device)
    return F.cross_entropy(logits, labels)


def contrastive_loss_with_hard_negatives(queries: torch.Tensor, positives: torch.Tensor,
                                         negatives: torch.Tensor, temperature: float = 0.1):
    """
    Contrastive with explicit hard negatives.
    queries: (B, D), positives: (B, D), negatives: (B, K, D)
    """
    q = F.normalize(queries, dim=1)
    p = F.normalize(positives, dim=1)
    n = F.normalize(negatives, dim=2)
    
    pos_logits = (q * p).sum(dim=1, keepdim=True) / temperature
    neg_logits = torch.bmm(q.unsqueeze(1), n.transpose(1, 2)).squeeze(1) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(queries.size(0), dtype=torch.long, device=queries.device)
    return F.cross_entropy(logits, labels)
```

---

## Embedding Tables for Categorical Features

For categorical features (user_id, item_id, category), we use **embedding tables**: a lookup from ID → learned vector.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Embedding Table                                                              │
│                                                                               │
│  categorical_id  ──▶  Embedding Layer  ──▶  [d₁, d₂, ..., d_k]               │
│  (user_12345)         (num_ids × dim)       (dim = 64, 128, 256)             │
│                                                                               │
│  Vocabulary sizes:                                                            │
│  • User IDs: 100M - 1B+                                                       │
│  • Item IDs: 10M - 1B+                                                        │
│  • Categories: 1K - 100K                                                      │
│                                                                               │
│  Memory: 1B users × 128 dim × 4 bytes = 512 GB (float32)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hashing Trick

When vocabulary exceeds memory, use hashing: map ID to bucket, share embeddings across collisions.

```python
def hashed_embedding(id: int, num_buckets: int, embed_dim: int) -> int:
    """Map ID to bucket for embedding lookup. Collisions share embedding."""
    return hash(id) % num_buckets

# Example: 1B users → 10M buckets (100:1 compression)
# Trade-off: collision reduces model capacity
```

### Multi-Hot Encoding

For multi-valued categoricals (user interests, item tags):

```python
import torch
import torch.nn as nn

class MultiHotEmbedding(nn.Module):
    """Embed multiple IDs and pool (mean/sum)."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, pooling: str = "mean"):
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, mode=pooling)
        self.pooling = pooling
    
    def forward(self, ids: torch.Tensor, offsets: torch.Tensor = None) -> torch.Tensor:
        # ids: flat list of IDs; offsets: start index of each sample
        if offsets is None:
            return self.embedding(ids)
        return self.embedding(ids, offsets)
```

---

## Embedding Quality Evaluation

### Intrinsic Evaluation

- **Analogy tasks**: "king - man + woman ≈ queen" (word embeddings)
- **Clustering metrics**: Silhouette, NMI on known clusters
- **Retrieval metrics**: Precision@k, Recall@k on held-out similar pairs

### Extrinsic Evaluation

- **Downstream task**: Use embeddings as features; measure AUC, NDCG
- **A/B test**: Deploy; measure CTR, engagement, revenue

### Python: Similarity and Evaluation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """(N, D) → (N, N) similarity matrix."""
    return cosine_similarity(embeddings)

def recall_at_k(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    ground_truth: list,  # list of sets: ground_truth[i] = relevant item IDs for query i
    k: int = 10
) -> float:
    """Recall@k: fraction of relevant items in top-k."""
    sims = query_embeddings @ item_embeddings.T
    top_k = np.argsort(-sims, axis=1)[:, :k]
    
    recalls = []
    for i, gt in enumerate(ground_truth):
        if not gt:
            continue
        retrieved = set(top_k[i])
        hit = len(retrieved & gt)
        recalls.append(hit / len(gt))
    return np.mean(recalls) if recalls else 0.0
```

---

## Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| **Embedding dimension** | 64 | 256 | Lower dim: faster, less memory, may underfit. Higher: more expressive, slower |
| **Pre-trained vs fine-tuned** | Frozen pre-trained (e.g., BERT) | Fine-tuned on task | Pre-trained: less data, faster. Fine-tuned: better task fit |
| **Training objective** | Cross-entropy (classification) | Contrastive (similarity) | CE: simple, needs labels. Contrastive: self-supervised, needs good negatives |
| **Negative sampling** | Random | Hard negatives | Random: fast. Hard: better discrimination, needs mining |
| **Hashing trick** | No hashing (full vocab) | Hash to buckets | Full: no collision. Hash: memory-efficient, collision loss |
| **Pooling (multi-value)** | Mean | Sum | Mean: length-invariant. Sum: emphasizes multiplicity |

---

## Production Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Dimension** | 64-256 for items; 128-512 for text (BERT) |
| **Normalization** | L2-normalize for cosine; use dot product (equivalent for unit vectors) |
| **Quantization** | FP16 or int8 for serving to reduce memory and latency |
| **Update frequency** | Item embeddings: daily/hourly. User: online or cached |
| **Monitoring** | Embedding drift, retrieval recall, downstream metrics |

---

## Interview Tips

1. **"How do you create item embeddings?"** → Item2Vec (co-occurrence), two-tower (supervised), content-based (metadata).
2. **"Why contrastive over cross-entropy?"** → Contrastive learns similarity structure; good for retrieval. CE needs explicit labels.
3. **"How do you handle 1B-item vocabulary?"** → Hashing trick, partitioned embedding tables, or hash embeddings.
4. **"How do you evaluate embedding quality?"** → Intrinsic: analogy, clustering. Extrinsic: recall@k, downstream AUC.
5. **"Pre-trained vs train-from-scratch?"** → Pre-trained for text/images with limited data. Train-from-scratch when you have large domain-specific interaction data.

---

## Related Topics

- [Two-Tower Architecture](./04-two-tower-architecture.md) – Production embedding models
- [Approximate Nearest Neighbors](./02-approximate-nearest-neighbors.md) – Scaling retrieval
- [Recommendation Systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) – Primary use case
- [Feature Engineering](../../phase-2-core-components/03-feature-engineering/01-feature-stores.md) – Embedding tables in feature stores
