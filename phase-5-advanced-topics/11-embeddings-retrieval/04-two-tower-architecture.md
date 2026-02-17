# Two-Tower Architecture

## Overview

The **two-tower** (dual encoder) model consists of separate networks for queries and items. Each tower produces an embedding; similarity is computed via dot product or cosine. This design enables **offline precomputation** of item embeddings and **online** computation of only the query (user) embedding—essential for scaling to 1B+ items with sub-100ms latency.

Two-tower models power recommendation, search, and retrieval at Pinterest, YouTube, LinkedIn, and most major platforms. Understanding this architecture is critical for ML Systems Design interviews.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Two-Tower Model                                                              │
│                                                                               │
│  User/Query Tower                    Item Tower                               │
│  ┌─────────────────────┐            ┌─────────────────────┐                  │
│  │ User ID embedding   │            │ Item ID embedding   │                  │
│  │ User features       │            │ Item features       │                  │
│  │ Context features    │            │ Content features    │                  │
│  │        │            │            │        │            │                  │
│  │        ▼            │            │        ▼            │                  │
│  │   Dense layers      │            │   Dense layers      │                  │
│  │        │            │            │        │            │                  │
│  │        ▼            │            │        ▼            │                  │
│  │   u ∈ R^d           │            │   v ∈ R^d           │                  │
│  └────────┬────────────┘            └────────┬────────────┘                  │
│           │                                   │                               │
│           └───────────────┬───────────────────┘                               │
│                           ▼                                                   │
│                    score = u · v  (dot product)                               │
│                                                                               │
│  Key: Item tower runs OFFLINE; User tower runs ONLINE per request             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Two Towers?

| Single tower (cross-attention) | Two towers (dual encoder) |
|-------------------------------|---------------------------|
| Query and item interact during inference | No interaction; separate encoding |
| Must compute for every (query, item) pair | Query once; item embeddings precomputed |
| O(N) per query for N items | O(1) for user; O(log N) ANN for items |
| Higher accuracy potential | Slightly lower accuracy; 1000× faster |

**Production reality**: For 1B items, single-tower would require 1B forward passes per query. Two-tower: 1 user forward + 1 ANN lookup (~1-5ms).

---

## Training: Negative Sampling Strategies

The model learns to score positive pairs (user clicked item) higher than negative pairs. The choice of negatives dominates quality.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Negative Sampling Strategies                                                 │
│                                                                               │
│  1. Random: Sample uniformly from corpus                                     │
│     • Easy, fast; often too easy (model doesn't learn subtlety)              │
│                                                                               │
│  2. In-batch: Use other items in batch as negatives                          │
│     • Free; effective when batch size large (4096-8192)                      │
│     • Used by: YouTube, Google                                               │
│                                                                               │
│  3. Hard negatives: Items that are similar but not clicked                   │
│     • e.g., ANN retrieved but not in positive set                            │
│     • Improves discrimination; needs mining pipeline                         │
│                                                                               │
│  4. Popularity-biased: Sample proportional to popularity                     │
│     • Avoids overfitting to rare items; matches exposure                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Strategy | Implementation | Quality | Cost |
|----------|----------------|---------|------|
| Random | Uniform sample | Low | Low |
| In-batch | No extra sampling | Medium-High | None |
| Hard (ANN) | Retrieve, exclude positives | High | Medium |
| Hard (cross-batch) | Stale cache of negatives | High | Medium |

---

## Loss Functions

### Binary Cross-Entropy (Pointwise)

For each (user, item) pair: positive or negative. Binary label.

```python
loss = -[y * log(σ(u·v)) + (1-y) * log(1 - σ(u·v))]
```

### Sampled Softmax

Treat as multiclass: positive item vs N sampled negatives.

```python
# logits[i] = u · v_i for positive and negatives
loss = CrossEntropy(logits, label=0)  # positive is index 0
```

### InfoNCE / NT-Xent (Listwise)

In-batch negatives: for each query, all other items in batch are negatives.

```python
# For query i, positive is key i. All other keys in batch are negatives.
logits = (queries @ keys.T) / temperature
labels = range(batch_size)
loss = CrossEntropy(logits, labels)
```

**Temperature**: Lower τ (e.g., 0.05) = sharper distribution, harder training. Typical: 0.05-0.1.

---

## Serving Pattern: Offline + Online

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Two-Tower Serving at Scale                                                  │
│                                                                               │
│  OFFLINE (Batch, e.g., hourly):                                              │
│  Items ──▶ Item Tower ──▶ Item Embeddings ──▶ ANN Index ──▶ Vector DB       │
│  (1B items)    (GPU batch)     (1B × 128)      (HNSW)                        │
│                                                                               │
│  ONLINE (Per request, ~10-50ms):                                             │
│  User + Context ──▶ User Tower ──▶ User Embedding ──▶ ANN Query ──▶ Top-K   │
│  (features)           (GPU/CPU)       (1 × 128)         (1-5ms)               │
│                                                                               │
│  Total latency: User emb (10-30ms) + ANN (1-5ms) + Ranker (20-50ms) = 50ms   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Python: PyTorch Two-Tower Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """Two-tower model for recommendation retrieval."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dims: list = [256, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding tables
        self.user_embedding = nn.Embedding(num_users, 64)
        self.item_embedding = nn.Embedding(num_items, 64)
        
        # User tower
        user_input_dim = 64 + user_feature_dim  # emb + other features
        user_layers = []
        prev = user_input_dim
        for h in hidden_dims:
            user_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev = h
        user_layers.append(nn.Linear(prev, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item tower
        item_input_dim = 64 + item_feature_dim
        item_layers = []
        prev = item_input_dim
        for h in hidden_dims:
            item_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev = h
        item_layers.append(nn.Linear(prev, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)
    
    def forward_user(self, user_ids, user_features):
        """Encode user to embedding. Used online."""
        u_emb = self.user_embedding(user_ids)
        x = torch.cat([u_emb, user_features], dim=1)
        return F.normalize(self.user_tower(x), dim=1)
    
    def forward_item(self, item_ids, item_features):
        """Encode item to embedding. Used offline for indexing."""
        i_emb = self.item_embedding(item_ids)
        x = torch.cat([i_emb, item_features], dim=1)
        return F.normalize(self.item_tower(x), dim=1)
    
    def forward(self, user_ids, user_features, item_ids, item_features):
        """Full forward for training."""
        u = self.forward_user(user_ids, user_features)
        v = self.forward_item(item_ids, item_features)
        return (u * v).sum(dim=1)  # dot product
```

### Training Loop with In-Batch Negatives

```python
def train_step(model, batch, optimizer, temperature=0.1):
    """Training step with in-batch negatives (InfoNCE)."""
    user_ids, user_feat, item_ids, item_feat = batch
    
    u = model.forward_user(user_ids, user_feat)
    v = model.forward_item(item_ids, item_feat)
    
    # In-batch: positive pair is (i, i); negatives are (i, j) for j != i
    logits = (u @ v.T) / temperature
    labels = torch.arange(u.size(0), device=u.device)
    
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Training with Explicit Negatives

```python
def train_step_with_negatives(model, batch, optimizer, temperature=0.1):
    """Training with explicit positive + negative items."""
    user_ids, user_feat, pos_items, pos_item_feat, neg_items, neg_item_feat = batch
    
    u = model.forward_user(user_ids, user_feat)
    v_pos = model.forward_item(pos_items, pos_item_feat)
    v_neg = model.forward_item(neg_items, neg_item_feat)  # (B, K, D)
    
    pos_logits = (u * v_pos).sum(dim=1, keepdim=True) / temperature
    neg_logits = torch.bmm(u.unsqueeze(1), v_neg.transpose(1, 2)).squeeze(1) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
    
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## Handling Cold Start

New users and items have no or sparse interaction history.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Cold Start Strategies                                                        │
│                                                                               │
│  New User:                                                                   │
│  • Use content features only (demographics, device, location)                 │
│  • Default embedding (mean of population)                                    │
│  • Explore: random/most popular until enough signals                         │
│                                                                               │
│  New Item:                                                                   │
│  • Content features: title, image, category embedding                        │
│  • Zero out ID embedding; rely on content tower                              │
│  • Boost in retrieval (exploration) until enough interactions                │
│                                                                               │
│  Architecture: Ensure towers can run with only content features               │
│  (e.g., train with dropout on ID embedding sometimes)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Content-Based Tower for Cold Start

```python
class TwoTowerWithContent(nn.Module):
    """Two-tower with content features for cold start."""
    
    def __init__(self, ..., title_vocab_size=50000, title_embed_dim=64):
        # ... standard tower init ...
        
        # Content pathway (for cold start)
        self.title_embedding = nn.Embedding(title_vocab_size, title_embed_dim)
        self.content_proj = nn.Linear(title_embed_dim * max_title_len, item_feature_dim)
    
    def forward_item(self, item_ids, item_features, title_ids=None):
        i_emb = self.item_embedding(item_ids)
        
        # For new items: mask ID emb, use content
        if title_ids is not None:
            content_emb = self.title_embedding(title_ids).flatten(1)
            content_feat = self.content_proj(content_emb)
            # Blend: 0.5 * id_features + 0.5 * content for new items
            x = torch.cat([i_emb, item_features + 0.5 * content_feat], dim=1)
        else:
            x = torch.cat([i_emb, item_features], dim=1)
        
        return F.normalize(self.item_tower(x), dim=1)
```

---

## Extensions: Multi-Task and Cross-Attention

### Multi-Task Towers

Shared bottom, task-specific tops (e.g., click vs like vs purchase):

```python
# Shared user tower → multiple heads
u = self.user_tower(user_features)
click_logits = self.click_head(u)
like_logits = self.like_head(u)
loss = click_loss + 0.5 * like_loss
```

### Cross-Attention (Late Interaction)

Models like ColBERT use late interaction: encode separately but match at token level for ranking. Not used for first-stage retrieval (too slow).

---

## Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| **Negative sampling** | Random | Hard negatives | Random: fast. Hard: better, needs pipeline |
| **Batch size** | 256 | 8192 | Larger: more in-batch negatives, better; memory bound |
| **Temperature** | 0.1 | 0.05 | Lower: sharper, harder training |
| **Embedding dim** | 64 | 256 | Higher: more expressive, more memory & compute |
| **Tower depth** | 2 layers | 4 layers | Deeper: more capacity, slower inference |
| **ID embedding** | Yes | Content-only | ID: better for warm items. Content: cold start |

---

## Production Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Recall@10** | 0.90+ | ANN vs exact |
| **User emb latency** | < 20ms p99 | Critical path |
| **Index freshness** | < 24h | New items |
| **Training throughput** | Millions samples/hour | GPU utilization |
| **AUC / NDCG** | Baseline + 2% | Vs previous model |

---

## Full Training Pipeline Example

```python
# Pseudo-code for production two-tower training
def train_two_tower(click_log, num_epochs=3):
    model = TwoTowerModel(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for batch in DataLoader(click_log, batch_size=4096, shuffle=True):
            # batch: (user_id, user_feat, item_id, item_feat)
            loss = train_step(model, batch, optimizer, temperature=0.1)
            log("loss", loss)
        
        # Periodic evaluation
        recall = evaluate_recall(model, val_data, ann_index)
        log("recall@10", recall)
        
        # Save for serving
        if recall > best_recall:
            save_item_embeddings(model, all_items)
            rebuild_ann_index()
```

---

## Interview Tips

1. **"Why two towers?"** → Precompute item embeddings; only compute user embedding online. Enables ANN at 1B scale.
2. **"How do you train it?"** → Positive = clicked; negatives = random, in-batch, or hard. Loss: BCE, InfoNCE, or sampled softmax.
3. **"What are hard negatives?"** → Items similar to positive (e.g., ANN-retrieved) but not clicked. Improve discrimination.
4. **"How do you handle cold start?"** → Content features in item tower; default user embedding; exploration.
5. **"How do you serve at scale?"** → Offline: item tower → ANN index. Online: user tower → ANN query → ranker.

---

## Related Topics

- [Embedding Fundamentals](./01-embedding-fundamentals.md) – Contrastive learning, embedding tables
- [Approximate Nearest Neighbors](./02-approximate-nearest-neighbors.md) – ANN for retrieval
- [Vector Databases](./03-vector-databases.md) – Storing item embeddings
- [Recommendation Systems](../../phase-4-end-to-end-systems/10-end-to-end-systems/01-recommendation-systems.md) – End-to-end system
