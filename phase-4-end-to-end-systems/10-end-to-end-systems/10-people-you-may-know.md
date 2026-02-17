# People You May Know (PYMK)

## Overview

People You May Know (PYMK) is a social graph recommendation system that suggests potential connections to users—friends, colleagues, or acquaintances they might want to add. These systems drive growth and engagement on social platforms by helping users expand their networks. The challenge is personalizing suggestions for billions of users while respecting privacy, handling cold start, and scaling graph feature computation. **Meta favorite, graph-based ML system.**

---

## 🎯 Problem Definition

### Business Goals
- **Grow social graph:** Increase total connections, strengthen platform network effects
- **Increase engagement:** More connections → more content to consume, more reasons to return
- **Improve acceptance rate:** Suggest people users actually want to connect with
- **Respect privacy:** Never suggest based on sensitive attributes or reveal problematic patterns

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Scale** | 3B+ users, hundreds of billions of edges |
| **Personalization** | Per-user rankings |
| **Latency** | < 200ms for suggestion response |
| **Freshness** | Reflect new connections within hours |
| **Privacy** | No sensitive-attribute-based suggestions |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      People You May Know (PYMK) Architecture                          │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                          USER PROFILE / REQUEST                                │   │
│  │  user_id, context (where shown: feed, profile, add friend), exclude_list      │   │
│  └─────────────────────────────────────────┬────────────────────────────────────┘   │
│                                            │                                          │
│                                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    CANDIDATE GENERATION (Recall Stage)                           │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  │ │
│  │  │  Graph-Based         │  │  Embedding-Based     │  │  Content-Based       │  │ │
│  │  │  - Friends-of-friends │  │  - Node2Vec         │  │  - School, workplace │  │ │
│  │  │  - Shared communities │  │  - GraphSAGE        │  │  - Location, contacts │  │ │
│  │  │  - Co-engagement      │  │  - GAT               │  │  (cold start)         │  │ │
│  │  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘  │ │
│  │             └─────────────────────────┼─────────────────────────┘              │ │
│  │                                       │                                          │ │
│  │                         Merged candidate set: ~1K-10K candidates                 │ │
│  └───────────────────────────────────────┬─────────────────────────────────────────┘ │
│                                          │                                            │
│                                          ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    FEATURE ENRICHMENT                                             │ │
│  │  Graph features (common friends, Jaccard, Adamic-Adar) | Profile | Behavioral    │ │
│  └───────────────────────────────────────┬─────────────────────────────────────────┘ │
│                                          │                                            │
│                                          ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    RANKING MODEL                                                  │ │
│  │  GBDT / Neural ranker → P(accept | suggestion shown)                            │ │
│  └───────────────────────────────────────┬─────────────────────────────────────────┘ │
│                                          │                                            │
│                                          ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    FILTERING                                                     │ │
│  │  Privacy, blocks, already suggested, already friends, deactivated accounts       │ │
│  └───────────────────────────────────────┬─────────────────────────────────────────┘ │
│                                          │                                            │
│                                          ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    NOTIFICATION / DISPLAY                                         │ │
│  │  When to send, how many, fatigue control, placement (feed vs sidebar vs add tab)  │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### Graph-Based Features

| Feature Type | Formula/Description | Use Case |
|--------------|---------------------|----------|
| **Common friends** | \|N(u) ∩ N(v)\| | Core signal: more overlap → more likely to know |
| **Jaccard similarity** | \|N(u) ∩ N(v)\| / \|N(u) ∪ N(v)\| | Normalize by total friends |
| **Adamic-Adar index** | Σ 1/log(\|N(z)\|) for z ∈ N(u) ∩ N(v) | Weight rare common friends higher |
| **Preferential attachment** | \|N(u)\| × \|N(v)\| | Popularity bias (use carefully) |
| **Shortest path** | BFS distance | 2-hop ideal; 3-hop weaker |
| **Shared communities** | Groups, pages, events | Strong signal for affinity |
| **Shared workplace/school** | Profile attributes | Explicit affinity |
| **Temporal** | Recently added friends, recent profile views | Recency signal |

### Embedding-Based Approaches

| Method | Idea | Pros | Cons |
|--------|------|------|------|
| **Node2Vec** | Random walks → Skip-gram → node embeddings | Flexible (BFS/DFS), scalable | Offline only |
| **GraphSAGE** | Sample neighborhood → aggregate → embed | Inductive (new nodes), scalable | Needs training |
| **GAT** | Graph Attention Networks | Learn importance of neighbors | Heavier compute |
| **LINE** | 1st + 2nd order proximity | Fast | Less expressive |

### Cold Start Strategies

- **Profile-based:** School, workplace, location → match with others sharing same
- **Contact import:** Phone contacts (with permission) → suggest if on platform
- **Content-based:** Interests, liked pages → similar users
- **Popular suggestions:** Top connected users (with diversity)

### Privacy Constraints

- **No sensitive attributes:** Don't suggest based on protected attributes (race, religion, etc.)
- **No pattern revelation:** Avoid suggesting someone because both viewed same sensitive content
- **Block list:** Never suggest blocked users or users who blocked you
- **Data minimization:** Use only necessary signals

### Python Code: GraphFeatureComputer

```python
from typing import Dict, List, Set
from collections import defaultdict
import math

class GraphFeatureComputer:
    """Compute graph-based features for PYMK candidate pairs."""
    
    def __init__(self, graph_store):
        self.graph = graph_store  # Adjacency list or graph DB interface
    
    def compute_features(
        self,
        user_id: str,
        candidate_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute features for (user, candidate) pairs."""
        user_friends = set(self.graph.get_neighbors(user_id))
        user_degree = len(user_friends)
        
        result = {}
        for cid in candidate_ids:
            candidate_friends = set(self.graph.get_neighbors(cid))
            common = user_friends & candidate_friends
            
            # Jaccard similarity
            union = user_friends | candidate_friends
            jaccard = len(common) / len(union) if union else 0
            
            # Adamic-Adar index
            adamic_adar = sum(
                1.0 / math.log(max(self.graph.degree(z), 2))
                for z in common
            )
            
            # Preferential attachment (log to avoid huge values)
            pref_attach = math.log1p(user_degree * len(candidate_friends))
            
            # Shared communities
            user_communities = set(self.graph.get_communities(user_id))
            candidate_communities = set(self.graph.get_communities(cid))
            shared_communities = len(user_communities & candidate_communities)
            
            result[cid] = {
                "common_friends": len(common),
                "common_friends_log": math.log1p(len(common)),
                "jaccard_similarity": jaccard,
                "adamic_adar": adamic_adar,
                "preferential_attachment_log": pref_attach,
                "shared_communities": shared_communities,
                "user_degree_log": math.log1p(user_degree),
                "candidate_degree_log": math.log1p(len(candidate_friends)),
            }
        
        return result
```

### Python Code: Node2VecGenerator

```python
import random
from typing import List, Dict
import numpy as np

class Node2VecGenerator:
    """Generate PYMK candidates using Node2Vec embeddings."""
    
    def __init__(
        self,
        walk_length: int = 10,
        num_walks: int = 80,
        p: float = 1.0,  # Return parameter (BFS)
        q: float = 1.0,  # In-out parameter (DFS)
        embedding_dim: int = 64
    ):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.embedding_dim = embedding_dim
        self.embeddings = {}  # node_id -> np.array
    
    def random_walk(self, start: str, graph: Dict[str, List[str]]) -> List[str]:
        """Biased random walk for Node2Vec."""
        walk = [start]
        
        for _ in range(self.walk_length - 1):
            curr = walk[-1]
            neighbors = graph.get(curr, [])
            if not neighbors:
                break
            
            if len(walk) == 1:
                next_node = random.choice(neighbors)
            else:
                prev = walk[-2]
                weights = []
                for n in neighbors:
                    if n == prev:
                        weights.append(1.0 / self.p)
                    elif n in graph.get(prev, []):
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / self.q)
                probs = np.array(weights) / sum(weights)
                next_node = np.random.choice(neighbors, p=probs)
            
            walk.append(next_node)
        
        return walk
    
    def generate_walks(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Generate corpus of random walks."""
        nodes = list(graph.keys())
        walks = []
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(node, graph)
                walks.append(walk)
        return walks
    
    def fit(self, graph: Dict[str, List[str]]):
        """Fit embeddings using Node2Vec (simplified; production uses gensim/Word2Vec)."""
        walks = self.generate_walks(graph)
        
        # Build vocab and train skip-gram (simplified)
        from collections import Counter
        vocab = {}
        for walk in walks:
            for node in walk:
                vocab.setdefault(node, len(vocab))
        
        # Placeholder: real implementation uses Word2Vec on walk strings
        for node in vocab:
            self.embeddings[node] = np.random.randn(self.embedding_dim) * 0.1
        
        return self
    
    def get_similar(self, node_id: str, k: int = 100) -> List[str]:
        """Get top-k most similar nodes by embedding (for candidate generation)."""
        if node_id not in self.embeddings:
            return []
        
        vec = self.embeddings[node_id]
        scores = []
        for nid, emb in self.embeddings.items():
            if nid == node_id:
                continue
            sim = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb) + 1e-9)
            scores.append((nid, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scores[:k]]
```

### Python Code: CandidateGenerator

```python
from typing import List, Set
from abc import ABC, abstractmethod

class CandidateGenerator(ABC):
    @abstractmethod
    def generate(self, user_id: str, k: int) -> List[str]:
        pass

class FriendsOfFriendsGenerator(CandidateGenerator):
    """Classic FoF candidate generation."""
    
    def __init__(self, graph_store):
        self.graph = graph_store
    
    def generate(self, user_id: str, k: int = 500) -> List[str]:
        friends = set(self.graph.get_neighbors(user_id))
        candidates = {}
        
        for f in friends:
            for ff in self.graph.get_neighbors(f):
                if ff != user_id and ff not in friends:
                    candidates[ff] = candidates.get(ff, 0) + 1
        
        # Sort by number of mutual friends
        sorted_cands = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [c for c, _ in sorted_cands[:k]]

class SharedCommunityGenerator(CandidateGenerator):
    """Generate from shared groups/communities."""
    
    def __init__(self, graph_store):
        self.graph = graph_store
    
    def generate(self, user_id: str, k: int = 200) -> List[str]:
        user_communities = self.graph.get_communities(user_id)
        candidates = {}
        
        for comm in user_communities:
            members = self.graph.get_community_members(comm)
            for m in members:
                if m != user_id:
                    candidates[m] = candidates.get(m, 0) + 1
        
        sorted_cands = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [c for c, _ in sorted_cands[:k]]

class EmbeddingCandidateGenerator(CandidateGenerator):
    """Generate from graph embeddings (Node2Vec, etc.)."""
    
    def __init__(self, embedding_model, graph_store):
        self.model = embedding_model
        self.graph = graph_store
    
    def generate(self, user_id: str, k: int = 300) -> List[str]:
        return self.model.get_similar(user_id, k=k)

class HybridCandidateGenerator:
    """Combine multiple sources with deduplication and scoring."""
    
    def __init__(
        self,
        generators: List[tuple]  # (generator, weight)
    ):
        self.generators = generators
    
    def generate(self, user_id: str, k: int = 1000) -> List[str]:
        scores = {}
        
        for gen, weight in self.generators:
            candidates = gen.generate(user_id, k=k)
            for i, cid in enumerate(candidates):
                # Reciprocal rank scoring
                score = weight / (i + 1)
                scores[cid] = scores.get(cid, 0) + score
        
        sorted_cands = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [c for c, _ in sorted_cands[:k]]
```

### Python Code: PYMKRanker

```python
import numpy as np
from typing import List, Dict
from sklearn.ensemble import GradientBoostingClassifier

class PYMKRanker:
    """Rank PYMK candidates with GBDT or similar."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None
    ):
        """Train on (features, label) where label = 1 if accepted, 0 if shown but not."""
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(accept | shown)."""
        return self.model.predict_proba(X)[:, 1]
    
    def rank(
        self,
        candidates: List[str],
        features: Dict[str, Dict[str, float]]
    ) -> List[tuple]:
        """Rank candidates by predicted acceptance probability."""
        if not self.is_fitted:
            return [(c, 0.0) for c in candidates]
        
        X = np.array([
            [features.get(c, {}).get(f, 0) for f in self.feature_names]
            for c in candidates
        ])
        probs = self.predict_proba(X)
        
        ranked = sorted(
            zip(candidates, probs),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked

class PYMKService:
    """End-to-end PYMK suggestion service."""
    
    def __init__(
        self,
        candidate_generator: HybridCandidateGenerator,
        feature_computer: GraphFeatureComputer,
        ranker: PYMKRanker,
        filter_service
    ):
        self.candidate_gen = candidate_generator
        self.feature_computer = feature_computer
        self.ranker = ranker
        self.filter = filter_service
    
    def get_suggestions(
        self,
        user_id: str,
        k: int = 10,
        exclude: Set[str] = None
    ) -> List[str]:
        """Get top-k PYMK suggestions for a user."""
        exclude = exclude or set()
        
        # 1. Generate candidates
        candidates = self.candidate_gen.generate(user_id, k=1000)
        
        # 2. Filter (blocks, already friends, already suggested)
        candidates = self.filter.apply(user_id, candidates, exclude)
        
        if not candidates:
            return []
        
        # 3. Compute features
        features = self.feature_computer.compute_features(user_id, candidates)
        
        # 4. Rank
        ranked = self.ranker.rank(candidates, features)
        
        return [cid for cid, _ in ranked[:k]]
```

### Notification Optimization

- **When to send:** After meaningful engagement (e.g., added 2+ friends recently)
- **How many:** 5-10 per surface; avoid flooding
- **Fatigue:** Cap suggestions per day; diversify surfaces (feed vs sidebar vs add tab)
- **Placement:** Higher CTR for "Add Friend" tab vs feed widget

### Scale Considerations

| Challenge | Solution |
|-----------|----------|
| **Billions of nodes** | Sharding by user; distributed graph DB (TAO, etc.) |
| **Hundreds of billions of edges** | Batch precompute FoF; streaming updates |
| **Latency** | Precompute and cache candidates; online ranking only |
| **Freshness** | Incremental graph updates; periodic full recompute |
| **Cold start** | Profile + contacts + content-based fallback |

---

## 📈 Metrics & Evaluation

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Acceptance Rate** | % of suggestions that become connections | 10-20% |
| **Friend Request Rate** | Requests sent per suggestion shown | 30-50% |
| **New Connections/Day** | Total new edges attributed to PYMK | Growth KPI |
| **Long-term Retention Impact** | Do PYMK-acquired friends increase retention? | Positive lift |
| **Diversity** | Variety of suggested sources (FoF vs community vs embedding) | Balanced |
| **Latency** | p99 response time | < 200ms |

### Offline Evaluation

```python
def evaluate_pymk_offline(
    ranked_suggestions: List[List[str]],
    ground_truth: List[Set[str]],  # Accepted connections per user
    k: int = 10
) -> Dict[str, float]:
    """Compute precision@k, recall@k, NDCG@k."""
    precisions, recalls, ndcgs = [], [], []
    
    for pred, truth in zip(ranked_suggestions, ground_truth):
        pred_k = pred[:k]
        hits = len(set(pred_k) & truth)
        
        precisions.append(hits / k)
        recalls.append(hits / len(truth) if truth else 0)
        
        dcg = sum(1.0 / np.log2(i + 2) for i, u in enumerate(pred_k) if u in truth)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(truth))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return {
        "precision_at_k": np.mean(precisions),
        "recall_at_k": np.mean(recalls),
        "ndcg_at_k": np.mean(ndcgs)
    }
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Candidate generation** | Graph-based only (interpretable) | Embedding-based (capture latent) |
| **Cold start** | Profile matching (privacy-safe) | Contact import (higher accuracy, consent) |
| **Ranking model** | GBDT (fast, interpretable) | Neural ranker (higher accuracy) |
| **Precomputation** | Batch nightly (simple) | Real-time (fresh, complex) |
| **Scale** | Precompute all pairs (storage) | Compute on-demand (latency) |
| **Privacy** | Minimal features (safe) | Rich features (better ranking) |
| **Notification** | Aggressive (growth) | Conservative (UX, fatigue) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How would you design PYMK for 3B users?
2. How do you handle cold start for new users?
3. What graph features would you use and why?
4. How do you balance growth (more suggestions) vs. quality (acceptance rate)?
5. How do you respect privacy while still personalizing?
6. How would you scale FoF computation?
7. When would you use Node2Vec vs. GraphSAGE?

**Key Points to Mention:**
- Two-stage: candidate generation (recall) + ranking (precision)
- Graph features: Jaccard, Adamic-Adar, shared communities
- Embeddings: Node2Vec for scalability, GraphSAGE for inductive
- Cold start: profile, contacts (with permission), content
- Privacy: no sensitive attributes, block handling, data minimization
- Scale: sharding, precomputation, incremental updates

---

## 🔗 Related Topics

- [Recommendation Systems](./01-recommendation-systems.md)
- [Embeddings & Retrieval](../../phase-5-advanced-topics/11-embeddings-retrieval/00-README.md)
- [Feature Engineering](../../phase-2-core-components/03-feature-engineering/02-online-vs-offline-features.md)
- [Fairness & Privacy](../../phase-5-advanced-topics/13-fairness-responsible-ai/00-README.md)
- [Caching Strategies](../../phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md)
