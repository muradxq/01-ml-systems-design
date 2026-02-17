# Video Recommendation System

## Overview

Video recommendation systems personalize content feeds for billions of users, balancing watch time, user satisfaction, and creator ecosystem health. They use multi-stage pipelines (candidate generation → ranking → reranking) with watch-time optimization to avoid click-bait and maximize long-term engagement. **Classic Google/YouTube-style interview question.**

---

## 🎯 Problem Definition

### Business Goals

- **Increase watch time:** Primary engagement metric; longer sessions = higher value
- **User satisfaction:** Beyond raw watch time—likes, shares, survey responses
- **Creator ecosystem health:** Fair exposure for new creators; sustainable monetization
- **Discovery:** Help users find diverse, serendipitous content
- **Monetization:** Effective ad insertion without degrading experience

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Personalization** | 2B+ users; unique feeds per user |
| **Catalog size** | Billions of videos |
| **Serving latency** | < 200ms p99 end-to-end |
| **Freshness** | New uploads discoverable within hours |
| **Diversity** | Avoid filter bubbles; inject exploration |
| **Fairness** | Balanced exposure across creators |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Video Recommendation System Architecture                               │
│                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         USER OPENS APP / REFRESH FEED                              │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    CANDIDATE GENERATION (Parallel, Multiple Sources)               │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐  │    │
│  │  │ Collaborative│ │ Content-based │ │ Subscription │ │   Trending   │ │Explore │  │    │
│  │  │   Filtering  │ │  (embeddings) │ │ (subs feed)  │ │ (regional)   │ │ (new)  │  │    │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └───┬────┘  │    │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────────┼──────┘    │
│            └────────────────┴────────────────┴────────────────┴─────────────┘            │
│                                        │                                                     │
│                                        ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    MERGE & DEDUPLICATE → ~1000 candidates                          │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    LIGHT RANKER (1000 → 200)                                       │    │
│  │  Fast model: Logistic regression / small NN | Latency: ~10ms                       │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    HEAVY RANKER (200 → 50)                                         │    │
│  │  Complex model: Deep & Cross, DCN, Transformer | Latency: ~50ms                    │    │
│  │  Target: Expected Watch Time (EWT)                                                 │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    RERANKING                                                       │    │
│  │  Diversity (topic, creator), Creator fairness, Ads insertion, Recency             │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    FEED DISPLAY                                                    │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    USER ACTIONS → LOGGING → TRAINING PIPELINE                      │    │
│  │  Watch, skip, like, share, subscribe, impression (no click)                        │    │
│  └──────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Multi-Stage Pipeline

**Why multi-stage?**
- Cannot run heavy model on billions of videos
- Stage 1: Broad retrieval (fast, ~1000 candidates)
- Stage 2: Light ranker (prune to ~200)
- Stage 3: Heavy ranker (precise ranking, ~50)
- Stage 4: Reranking (business rules, diversity, ads)

**Candidate generation sources (parallel):**
- **Collaborative filtering:** Users who watched X also watched Y (matrix factorization, two-tower)
- **Content-based:** Video embeddings from title, thumbnail, audio, frames
- **Subscription:** Recent uploads from subscribed channels
- **Trending:** Globally/regionally popular
- **Exploration:** New, underexposed content (MAB, exploration slot)

### 2. Video Embeddings

**Multi-modal:**
- **Title:** Text embedding (BERT, sentence-transformers)
- **Thumbnail:** Image embedding (ResNet, ViT)
- **Audio:** Audio embedding (trained or pretrained)
- **Video frames:** Frame-level embeddings → aggregate (mean, attention)

**Two-tower model for retrieval:**
- **User tower:** User history (video IDs → embeddings), aggregated
- **Item tower:** Video embedding (multi-modal fusion)
- Train with contrastive loss (in-batch negatives or sampled negatives)
- Inference: Precompute video embeddings; user embedding on-the-fly; ANN search

### 3. Watch-Time Prediction

**Why watch time > CTR?**
- CTR rewards click-bait (attractive thumbnails, misleading titles)
- Watch time rewards satisfying content
- YouTube paper: Weighted logistic regression with watch time as weight
- Or: Direct EWT regression (expected watch time in seconds)

**Weighted logistic regression:**
- Label: `y = 1` if watch_time > 30s (or similar threshold)
- Weight: `w = min(watch_time, max_cap)` (e.g., cap at 300s)
- Loss: `-w * (y * log(p) + (1-y) * log(1-p))`
- Inference: `p * E[watch_time | clicked]` ≈ expected watch time

**Expected Watch Time (EWT):**
- EWT = P(click) × E[watch_time | click]
- Or train separate models and multiply

### 4. User Satisfaction Modeling

**Beyond watch time:**
- Explicit: likes, shares, survey responses ("Was this helpful?")
- Implicit: early abandonment (< 10s), replay, speed changes (1.5x, 2x)

**Long-term vs short-term:**
- Short-term: Maximize next-session watch time
- Long-term: Retention, return visits, satisfaction surveys
- Risk: Over-optimizing short-term (addictive content) hurts long-term

### 5. Creator-Side Considerations

**Fair exposure:**
- New creators: Exploration slots, "rising" feeds
- Avoid winner-take-all; cap % of feed from single creator

**Content quality signals:**
- Completion rate, like ratio, comment sentiment
- Penalize click-bait (high CTR, low watch time)

**Creator monetization:**
- Ad-friendly content ranked fairly
- Balance user experience with creator revenue

### 6. Explore/Exploit and Filter Bubbles

**Avoid echo chambers:**
- Serendipity injection: 10–20% from exploration sources
- Topic diversity: Ensure variety in reranking
- Multi-stakeholder: User + creator + platform objectives

**Exploration strategies:**
- Thompson sampling, UCB for new videos
- ε-greedy: Small % random recommendations

---

## 💻 Python Code

### VideoRecommender

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class VideoCandidate:
    video_id: str
    source: str  # cf, content, subscription, trending, explore
    score: float
    features: Dict[str, Any]

class VideoRecommender:
    """Orchestrates the full video recommendation pipeline."""
    
    def __init__(
        self,
        candidate_gen: "MultiSourceCandidateGenerator",
        light_ranker: "LightRanker",
        heavy_ranker: "HeavyRanker",
        reranker: "DiversityReranker",
    ):
        self.candidate_gen = candidate_gen
        self.light_ranker = light_ranker
        self.heavy_ranker = heavy_ranker
        self.reranker = reranker
    
    def recommend(
        self,
        user_id: str,
        context: Dict[str, Any],
        n: int = 50,
    ) -> List[str]:
        # 1. Candidate generation (~1000)
        candidates = self.candidate_gen.get_candidates(user_id, context, top_k=1000)
        
        if not candidates:
            return []
        
        # 2. Light ranker (1000 -> 200)
        light_scored = self.light_ranker.score(user_id, candidates, top_k=200)
        
        if not light_scored:
            return []
        
        # 3. Heavy ranker (200 -> 50)
        heavy_scored = self.heavy_ranker.score(user_id, light_scored, top_k=50)
        
        if not heavy_scored:
            return []
        
        # 4. Rerank (diversity, fairness, ads)
        final = self.reranker.rerank(user_id, heavy_scored, n=n)
        
        return [c.video_id for c in final]
```

### MultiStagePipeline

```python
class MultiSourceCandidateGenerator:
    """Generates candidates from multiple sources in parallel."""
    
    def __init__(
        self,
        cf_retriever: Any,
        content_retriever: Any,
        subscription_retriever: Any,
        trending_retriever: Any,
        explore_retriever: Any,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        self.retrievers = {
            "cf": cf_retriever,
            "content": content_retriever,
            "subscription": subscription_retriever,
            "trending": trending_retriever,
            "explore": explore_retriever,
        }
        self.source_weights = source_weights or {s: 1.0 for s in self.retrievers}
    
    def get_candidates(
        self,
        user_id: str,
        context: Dict[str, Any],
        top_k: int = 1000,
    ) -> List[VideoCandidate]:
        all_candidates: Dict[str, VideoCandidate] = {}
        
        for source, retriever in self.retrievers.items():
            try:
                candidates = retriever.retrieve(user_id, context, top_k=top_k // 5)
                for c in candidates:
                    if c.video_id not in all_candidates:
                        all_candidates[c.video_id] = c
                    else:
                        # Merge scores (e.g., max or weighted sum)
                        prev = all_candidates[c.video_id]
                        w = self.source_weights.get(source, 1.0)
                        all_candidates[c.video_id] = VideoCandidate(
                            video_id=c.video_id,
                            source=f"{prev.source}+{source}",
                            score=prev.score + w * c.score,
                            features={**prev.features, **c.features},
                        )
            except Exception:
                continue
        
        sorted_cands = sorted(
            all_candidates.values(),
            key=lambda x: -x.score,
        )
        return sorted_cands[:top_k]
```

### WatchTimePredictor

```python
class WatchTimePredictor:
    """Predicts expected watch time. Uses weighted logistic regression concept."""
    
    def __init__(self, model: Any, watch_threshold_sec: float = 30.0):
        self.model = model
        self.watch_threshold = watch_threshold_sec
    
    def predict_expected_watch_time(
        self,
        user_features: np.ndarray,
        video_features: np.ndarray,
        context_features: np.ndarray,
    ) -> float:
        """Returns expected watch time in seconds."""
        # Option 1: Single EWT model
        # ewt = self.model.predict(np.concatenate([user_features, video_features, context_features]))
        
        # Option 2: P(click) * E[watch | click]
        features = np.concatenate([user_features, video_features, context_features])
        p_click = self.model.predict_proba(features)[0][1]
        ewt_given_click = self._predict_ewt_given_click(features)
        return p_click * ewt_given_click
    
    def _predict_ewt_given_click(self, features: np.ndarray) -> float:
        # Separate model or heuristic
        return 120.0  # placeholder
```

### DiversityReranker

```python
class DiversityReranker:
    """Reranks for diversity, creator fairness, and ad insertion."""
    
    def __init__(
        self,
        max_videos_per_creator: int = 3,
        diversity_weight: float = 0.2,
        topic_embedding_dim: int = 64,
    ):
        self.max_videos_per_creator = max_videos_per_creator
        self.diversity_weight = diversity_weight
        self.topic_embedding_dim = topic_embedding_dim
    
    def rerank(
        self,
        user_id: str,
        candidates: List[VideoCandidate],
        n: int = 50,
    ) -> List[VideoCandidate]:
        result: List[VideoCandidate] = []
        creator_counts: Dict[str, int] = {}
        
        for c in candidates:
            creator_id = c.features.get("creator_id", "unknown")
            
            # Creator cap
            if creator_counts.get(creator_id, 0) >= self.max_videos_per_creator:
                continue
            
            # Diversity: penalize if too similar to already selected
            if result:
                topic_emb = c.features.get("topic_embedding", np.zeros(self.topic_embedding_dim))
                min_sim = min(
                    np.dot(topic_emb, r.features.get("topic_embedding", np.zeros(self.topic_embedding_dim)))
                    for r in result[-10:]
                )
                # Penalize high similarity
                adjusted_score = c.score * (1 - self.diversity_weight * max(0, min_sim))
            else:
                adjusted_score = c.score
            
            result.append(VideoCandidate(
                video_id=c.video_id,
                source=c.source,
                score=adjusted_score,
                features=c.features,
            ))
            creator_counts[creator_id] = creator_counts.get(creator_id, 0) + 1
            
            if len(result) >= n:
                break
        
        return result
```

### Two-Tower Retrieval (Conceptual)

```python
class TwoTowerRetriever:
    """Two-tower model for collaborative filtering retrieval."""
    
    def __init__(self, user_tower: Any, item_tower: Any, item_index: Any):
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.item_index = item_index  # FAISS, Annoy, etc.
    
    def retrieve(
        self,
        user_id: str,
        context: Dict[str, Any],
        top_k: int = 100,
    ) -> List[VideoCandidate]:
        # User embedding from history
        user_history = context.get("watched_video_ids", [])[-50:]
        user_emb = self.user_tower.encode(user_id, user_history)
        
        # ANN search in item index (precomputed video embeddings)
        ids, scores = self.item_index.search(user_emb, top_k)
        
        return [
            VideoCandidate(
                video_id=vid,
                source="cf",
                score=float(scr),
                features={"embedding": self.item_index.get_embedding(vid)},
            )
            for vid, scr in zip(ids[0], scores[0])
        ]
```

---

## 📈 Metrics & Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Watch time** | Total / avg watch time per session | Maximize; A/B test |
| **User satisfaction** | Surveys ("How relevant?"), likes ratio | > 4.0/5 |
| **Creator upload rate** | New content from creators | Maintain/grow |
| **Content diversity** | Topic entropy, creator distribution | Avoid collapse |
| **CTR** | Click-through rate | Secondary to watch time |
| **Retention** | Return visits, D1/D7/D30 | Long-term health |
| **Serving latency** | p99 end-to-end | < 200ms |

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Objective** | CTR (simple) | Watch time (anti-clickbait) |
| **Candidate sources** | CF only (personalized) | Multi-source (diversity) |
| **Heavy ranker** | Logistic regression (fast) | Deep model (accuracy) |
| **Exploration** | Greedy (short-term) | MAB/exploration (long-term) |
| **Creator fairness** | Pure relevance | Cap per creator |
| **Freshness** | Batch update (cheap) | Real-time (expensive) |

---

## 🎤 Interview Tips

**Common Questions:**
1. Why optimize for watch time instead of CTR?
2. How do you avoid filter bubbles?
3. How do you serve 2B users with <200ms latency?
4. How do you give new creators exposure?
5. How do you handle the cold start problem for new videos?

**Key Points to Mention:**
- Multi-stage: candidate → light ranker → heavy ranker → rerank
- Multiple candidate sources (CF, content, subscription, trending, exploration)
- Watch-time weighted objective
- Two-tower for retrieval; deep model for ranking
- Diversity and creator fairness in reranking
- Explore/exploit trade-off

---

## 📐 Scale Considerations

### Serving 2B Users

- **Precomputation:** Batch score user × candidate pairs; cache top-K per user (refreshed hourly)
- **Partitioning:** Shard by user_id; each shard serves ~10M users
- **Caching:** LRU cache for hot users; pre-warm popular segments
- **Async:** Return cached feed; refresh in background for next request

### Billions of Videos

- **Item index:** Precomputed video embeddings in ANN (FAISS, ScaNN); O(log N) lookup
- **Incremental indexing:** New videos embedded and indexed within minutes of upload
- **Tiered catalog:** Hot videos (recent, trending) in fast tier; long-tail in cheaper storage

### Latency Budget (<200ms)

| Stage | Budget | Strategy |
|-------|--------|----------|
| Candidate gen | 50ms | Parallel fetches; ANN in-memory |
| Light ranker | 20ms | Small model; batched inference |
| Heavy ranker | 80ms | GPU inference; feature prefetch |
| Reranking | 30ms | Rule-based + fast heuristics |
| Overhead | 20ms | Network, serialization |

---

## 🔗 Related Topics

- [Recommendation Systems](./01-recommendation-systems.md)
- [Two-Tower Architecture](../../phase-5-advanced-topics/11-embeddings-retrieval/04-two-tower-architecture.md)
- [Approximate Nearest Neighbors](../../phase-5-advanced-topics/11-embeddings-retrieval/02-approximate-nearest-neighbors.md)
- [Model Serving](../../phase-2-core-components/05-model-serving/02-model-deployment.md)
- [A/B Testing](../../phase-2-core-components/05-model-serving/03-ab-testing.md)
