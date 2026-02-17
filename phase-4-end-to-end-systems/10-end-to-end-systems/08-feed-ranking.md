# Feed Ranking System

## Overview

Feed ranking powers the core product experience at Meta—**News Feed** and **Instagram Feed**. The system personalizes content for billions of users, balancing engagement signals (likes, comments, shares) with quality, integrity, and long-term satisfaction. Unlike simple recommendation, feed ranking is **multi-objective**, must incorporate **real-time signals**, and must balance **exploration vs exploitation** to prevent filter bubbles. Latency targets are strict (<200ms) as users scroll in real-time.

---

## 🎯 Problem Definition

### Business Goals

- **Maximize long-term engagement:** Time spent, sessions, return visits
- **User satisfaction:** Content relevance, quality, reduced negative experiences
- **Platform health:** Integrity (no misinformation, hate speech), diversity
- **Revenue integration:** Seamlessly blend organic content with ads
- **Creator value:** Surface content from creators users care about

### Requirements

| Requirement | Specification | Scale Context |
|-------------|---------------|---------------|
| **Latency** | < 200ms p99 | End-to-end feed request |
| **Throughput** | Billions of feed requests/day | ~500K-2M QPS peak |
| **Personalization** | Per-user, per-session | 2B+ users |
| **Freshness** | Real-time signal incorporation | New posts within seconds |
| **Diversity** | Avoid homogeneity | Category, source, recency mix |
| **Integrity** | Down-rank harmful content | Policy enforcement |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Feed Ranking System (News Feed / Instagram)                 │
│                                                                                   │
│  ┌────────────────┐                                                               │
│  │  User Opens    │  App launch, pull-to-refresh, infinite scroll                 │
│  │  Feed          │  Context: device, connectivity, time, session state            │
│  └───────┬────────┘                                                               │
│          │                                                                        │
│          ▼                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  CANDIDATE SELECTION (20-40ms)                                    │            │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │            │
│  │  │ Connections │ │ Followed    │ │ Groups      │ │ Ads         │  │            │
│  │  │ (friends)   │ │ Pages       │ │ Joined      │ │ (from ad    │  │            │
│  │  │             │ │             │ │             │ │  system)    │  │            │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘  │            │
│  │         └───────────────┼───────────────┼───────────────┘         │            │
│  │                         ▼                                          │            │
│  │            Hundreds of thousands → ~500-2000 candidates            │            │
│  └──────────────────────────────────────────────────────────────────┘            │
│                                  │                                               │
│                                  ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  FEATURE ENRICHMENT (10-20ms)                                     │            │
│  │  ┌──────────────────────────────────────────────────────────┐    │            │
│  │  │  User features | Author features | Content features       │    │            │
│  │  │  User-content affinity | Social proof | Real-time         │    │            │
│  │  └──────────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────────────────────────────────────────────┘            │
│                                  │                                               │
│                                  ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  MULTI-STAGE RANKING                                              │            │
│  │                                                                    │            │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐│            │
│  │  │ Lightweight     │──▶│ Heavy Ranker    │──▶│ Integrity       ││            │
│  │  │ Ranker          │   │ (MMoE, multi-   │   │ Filter          ││            │
│  │  │ (simple model,  │   │  objective)     │   │ (policy,        ││            │
│  │  │  500→100)       │   │ 100→50          │   │  P(misinfo))    ││            │
│  │  └─────────────────┘   └─────────────────┘   └────────┬────────┘│            │
│  │                                                       │          │            │
│  │                                                       ▼          │            │
│  │                                            ┌─────────────────┐   │            │
│  │                                            │ Diversity       │   │            │
│  │                                            │ Injector        │   │            │
│  │                                            │ (MMR, category  │   │            │
│  │                                            │  spread)        │   │            │
│  │                                            └────────┬────────┘   │            │
│  └─────────────────────────────────────────────────────┼────────────┘            │
│                                                        │                          │
│                                                        ▼                          │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  FEED COMPOSITION                                                 │            │
│  │  Mix: Organic posts | Ads | Suggested (Explore) | Sponsored       │            │
│  │  Slots, frequency capping, "see more" injections                  │            │
│  └──────────────────────────────────────────────────────────────────┘            │
│                                  │                                               │
│                                  ▼                                               │
│  ┌────────────────┐     ┌────────────────┐                                        │
│  │  Feed Response │────▶│  LOGGING       │  Impressions, engagement, dwell         │
│  │  (ordered list)│     │  (async)       │  for model retraining & analytics       │
│  └────────────────┘     └────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Multi-Objective Ranking

The feed optimizes multiple objectives simultaneously:

| Signal Type | Examples | Weight (Learned) |
|-------------|----------|------------------|
| **Engagement** | P(like), P(comment), P(share), P(click), P(dwell > 15s) | High |
| **Quality** | Content quality score, source credibility | Medium |
| **Integrity** | P(misinformation), P(hate_speech), P(clickbait) | Negative (down-rank) |
| **Value** | Weighted combination → single score | Final |

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class FeedItem:
    """A candidate item for the feed."""
    item_id: str
    author_id: str
    content_type: str  # post, story, ad, suggested
    content_embedding: Optional[np.ndarray] = None
    created_at: float = 0.0

class MultiObjectiveRanker:
    """
    Combines multiple objectives into a single ranking score.
    Uses a value model: score = Σ w_i * f_i(predictions)
    """
    
    def __init__(
        self,
        objective_weights: Optional[Dict[str, float]] = None
    ):
        self.weights = objective_weights or {
            "like": 1.0,
            "comment": 2.0,  # Comments are stronger engagement
            "share": 3.0,
            "click": 0.5,
            "dwell_15s": 1.5,
            "quality": 0.8,
            "integrity_penalty": -10.0,  # Heavy penalty for bad content
        }
    
    def compute_value_score(
        self,
        predictions: Dict[str, float]
    ) -> float:
        """
        Combine task predictions into single value score.
        predictions: {"like": 0.1, "comment": 0.02, "share": 0.01, ...}
        """
        score = 0.0
        for task, pred in predictions.items():
            weight = self.weights.get(task, 0.0)
            if "integrity" in task or "misinformation" in task:
                score += weight * pred  # Penalty: higher pred = lower score
            else:
                score += weight * pred
        return max(0.0, score)
    
    def rank(
        self,
        items: List[FeedItem],
        predictions: List[Dict[str, float]]
    ) -> List[FeedItem]:
        """Rank items by value score."""
        scored = [
            (item, self.compute_value_score(pred))
            for item, pred in zip(items, predictions)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored]
```

### 2. Model Architecture: MMoE (Multi-gate Mixture of Experts)

```python
import torch
import torch.nn as nn
from typing import List, Dict

class MMoEFeedRanker(nn.Module):
    """
    Multi-gate Mixture of Experts for multi-task feed ranking.
    Shared bottom + expert layers + task-specific gates and towers.
    """
    
    def __init__(
        self,
        input_dim: int,
        expert_dim: int = 64,
        num_experts: int = 4,
        num_tasks: int = 5,
        task_hidden_dims: List[int] = [64, 32],
        tasks: List[str] = ["like", "comment", "share", "click", "dwell"]
    ):
        super().__init__()
        self.tasks = tasks
        self.num_tasks = num_tasks
        
        # Shared bottom
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, expert_dim),
                nn.ReLU()
            )
            for _ in range(num_experts)
        ])
        
        # Task-specific gates (one per task)
        self.gates = nn.ModuleList([
            nn.Linear(64, num_experts)  # Gate weights over experts
            for _ in range(num_tasks)
        ])
        
        # Task-specific towers
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            layers = []
            prev = expert_dim
            for h in task_hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.towers.append(nn.Sequential(*layers))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared bottom
        shared = self.shared_bottom(x)
        
        # Expert outputs
        expert_outputs = torch.stack([e(shared) for e in self.experts], dim=1)
        # expert_outputs: [batch, num_experts, expert_dim]
        
        outputs = {}
        for i, task in enumerate(self.tasks):
            # Gate: softmax over experts
            gate_weights = torch.softmax(self.gates[i](./shared), dim=1)
            # gate_weights: [batch, num_experts]
            
            # Mixture of experts
            task_input = torch.einsum("be,bed->bd", gate_weights, expert_outputs)
            
            # Task tower
            outputs[task] = torch.sigmoid(self.towers[i](./task_input)).squeeze(-1)
        
        return outputs
```

### 3. Feature Engineering

| Category | Examples | Source |
|----------|----------|--------|
| **Author** | Follower count, post frequency, engagement rate, verification | Feature store |
| **Content** | Text embeddings, image quality score, video watch patterns | Precomputed |
| **User-Content** | Affinity score, interaction history (liked similar?) | Real-time join |
| **Social** | Common friends who engaged, social proof count | Graph service |
| **Real-time** | Trending score, recency decay, session context | Streaming |

```python
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

class FeedFeatureEngineer:
    """Feature engineering for feed ranking."""
    
    def get_author_features(self, author_id: str) -> Dict[str, float]:
        return {
            "author_follower_count_log": 12.0,  # log scale
            "author_engagement_rate": 0.05,
            "author_post_frequency_7d": 3.0,
            "author_verified": 1.0,
            "author_avg_dwell_time": 8.5,
        }
    
    def get_content_features(self, item_id: str) -> Dict[str, float]:
        return {
            "content_type": 0,  # 0=text, 1=image, 2=video
            "text_embedding_0": 0.1,
            "text_embedding_1": -0.2,
            "image_quality_score": 0.8,
            "video_duration": 60.0,
            "has_link": 0.0,
            "num_hashtags": 3.0,
        }
    
    def get_user_content_features(
        self,
        user_id: str,
        item_id: str,
        author_id: str
    ) -> Dict[str, float]:
        """User-item and user-author affinity."""
        return {
            "user_author_affinity": 0.7,  # Past interactions
            "user_content_affinity": 0.6,  # Similar content engagement
            "user_liked_author_before": 1.0,
            "user_commented_author_before": 0.0,
            "embedding_similarity": 0.65,
        }
    
    def get_social_proof_features(
        self,
        user_id: str,
        item_id: str
    ) -> Dict[str, float]:
        """Social proof: what friends did."""
        return {
            "friends_who_liked": 3.0,
            "friends_who_commented": 1.0,
            "friends_who_shared": 0.0,
            "social_proof_score": 0.5,  # Aggregate
        }
    
    def get_realtime_features(
        self,
        item_id: str,
        created_at: float,
        context: Dict
    ) -> Dict[str, float]:
        """Real-time and recency signals."""
        now = datetime.utcnow().timestamp()
        age_hours = (now - created_at) / 3600.0
        recency_decay = np.exp(-age_hours / 24.0)  # Decay over ~24h
        
        return {
            "recency_score": recency_decay,
            "age_hours": age_hours,
            "trending_score": context.get("trending_score", 0.5),
            "hour_of_day": datetime.utcnow().hour / 24.0,
        }
```

### 4. Diversity and Exploration

```python
from typing import List, Callable
import numpy as np

class DiversityReranker:
    """
    Inject diversity into ranked feed.
    MMR (Maximal Marginal Relevance) or category diversification.
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.3,
        strategy: str = "mmr"  # "mmr" or "category"
    ):
        self.diversity_weight = diversity_weight
        self.strategy = strategy
    
    def rerank_mmr(
        self,
        items: List[Dict],
        scores: List[float],
        similarity_fn: Callable,
        k: int = 20
    ) -> List[Dict]:
        """
        MMR: balance relevance and diversity.
        score_new = λ * rel(i) - (1-λ) * max_j∈S sim(i, j)
        """
        selected = []
        remaining = list(zip(items, scores))
        
        while len(selected) < k and remaining:
            best_score = float("-inf")
            best_idx = 0
            
            for idx, (item, rel_score) in enumerate(remaining):
                if len(selected) == 0:
                    div_penalty = 0.0
                else:
                    max_sim = max(
                        similarity_fn(item, s) for s in selected
                    )
                    div_penalty = max_sim
                
                mmr_score = (
                    self.diversity_weight * rel_score -
                    (1 - self.diversity_weight) * div_penalty
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx)[0])
        
        return selected
    
    def rerank_category_diversity(
        self,
        items: List[Dict],
        scores: List[float],
        category_fn: Callable,
        slots_per_category: Dict[str, int]
    ) -> List[Dict]:
        """Ensure minimum representation from each category."""
        by_category = {}
        for item, score in zip(items, scores):
            cat = category_fn(item)
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((item, score))
        
        result = []
        for cat, slot_count in slots_per_category.items():
            cat_items = sorted(by_category.get(cat, []), key=lambda x: x[1], reverse=True)
            result.extend([item for item, _ in cat_items[:slot_count]])
        
        return result
```

### 5. Explore/Exploit

```python
import numpy as np
from typing import List, Optional

class ThompsonSamplingExplorer:
    """
    Thompson sampling for exploring new content.
    Balance showing familiar content vs discovering new creators.
    """
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.stats = {}  # item_id -> (alpha, beta)
    
    def update(self, item_id: str, clicked: bool):
        """Update Beta distribution with feedback."""
        if item_id not in self.stats:
            self.stats[item_id] = (self.alpha_prior, self.beta_prior)
        alpha, beta = self.stats[item_id]
        if clicked:
            self.stats[item_id] = (alpha + 1, beta)
        else:
            self.stats[item_id] = (alpha, beta + 1)
    
    def sample_ctr(self, item_id: str) -> float:
        """Sample from posterior - used for exploration."""
        alpha, beta = self.stats.get(item_id, (self.alpha_prior, self.beta_prior))
        return np.random.beta(alpha, beta)
    
    def inject_exploration(
        self,
        ranked_items: List[Dict],
        exploration_fraction: float = 0.1,
        num_inject: int = 5
    ) -> List[Dict]:
        """Inject exploratory items (e.g., from new creators) into feed."""
        # In practice: pull from "explore" candidate pool
        # Score by sampled CTR to give new items a chance
        return ranked_items  # Simplified; full impl would merge explore pool
```

### 6. Feed Composition

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ContentSlot(Enum):
    ORGANIC = "organic"
    AD = "ad"
    SUGGESTED = "suggested"
    SPONSORED = "sponsored"

@dataclass
class FeedCompositionRule:
    """Rules for blending content types."""
    slot_type: ContentSlot
    min_position: int
    max_position: int
    frequency: int  # Every N slots
    max_per_feed: int

class FeedComposer:
    """
    Compose final feed from ranked organic + ads + suggested.
    """
    
    def __init__(
        self,
        rules: Optional[List[FeedCompositionRule]] = None
    ):
        self.rules = rules or [
            FeedCompositionRule(ContentSlot.AD, 3, 10, 5, 3),
            FeedCompositionRule(ContentSlot.SUGGESTED, 5, 15, 7, 2),
        ]
    
    def compose(
        self,
        organic_ranked: List[Dict],
        ads_ranked: List[Dict],
        suggested: List[Dict],
        num_slots: int = 20
    ) -> List[Dict]:
        """Interleave content following composition rules."""
        feed = []
        ad_idx = 0
        suggested_idx = 0
        
        for i in range(num_slots):
            # Check rules for this position
            slot_type = ContentSlot.ORGANIC
            for rule in self.rules:
                if (rule.min_position <= i <= rule.max_position and
                    (i + 1) % rule.frequency == 0 and
                    len([x for x in feed if x.get("type") == rule.slot_type.value]) < rule.max_per_feed):
                    slot_type = rule.slot_type
                    break
            
            if slot_type == ContentSlot.AD and ad_idx < len(ads_ranked):
                feed.append({**ads_ranked[ad_idx], "type": "ad"})
                ad_idx += 1
            elif slot_type == ContentSlot.SUGGESTED and suggested_idx < len(suggested):
                feed.append({**suggested[suggested_idx], "type": "suggested"})
                suggested_idx += 1
            else:
                # Organic
                org_idx = len([x for x in feed if x.get("type") == "organic"])
                if org_idx < len(organic_ranked):
                    feed.append({**organic_ranked[org_idx], "type": "organic"})
        
        return feed
```

### 7. Pointwise vs Pairwise vs Listwise

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Pointwise** | Predict score per item independently | Simple, scalable | Ignores relative order |
| **Pairwise** | Learn to compare pairs (A > B?) | Better ranking | O(n²) pairs |
| **Listwise** | Optimize list-level metric (NDCG) | Theoretically best | Complex, slow |

Meta typically uses **pointwise** at scale (predict engagement prob per item, then rank by value model) with listwise fine-tuning in some cases.

---

## 📈 Metrics & Evaluation

### Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Session Time** | Avg time per session | Primary engagement |
| **DAU/MAU** | Daily/monthly active users | Retention |
| **Engagement Rate** | Likes, comments, shares per view | 5-10% |
| **Negative Feedback** | Hide, "see fewer", unfollow | Minimize |
| **Ad Revenue** | Revenue from feed ads | Revenue |
| **Creator Satisfaction** | Creator-side distribution | Balance |

### Offline Metrics

| Metric | Description |
|--------|-------------|
| **AUC (per task)** | Like, comment, share, etc. |
| **NDCG** | Listwise ranking quality |
| **MRR** | Mean reciprocal rank |
| **Calibration** | Per-task probability calibration |

```python
def compute_feed_metrics(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray]
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    return {
        task: roc_auc_score(labels[task], preds)
        for task, preds in predictions.items()
        if task in labels
    }
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Pointwise vs Listwise** | Pointwise (fast) | Listwise (better) | Pointwise at scale; listwise for heavy ranker |
| **Diversity** | Pure relevance | Strong diversity | Balance; MMR or category slots |
| **Exploration** | Exploit only | High exploration | 5-10% exploration for new content |
| **Model** | Single model | MMoE multi-task | MMoE for multiple objectives |
| **Real-time features** | Batch only | Full real-time | Hybrid; hottest signals real-time |
| **Integrity** | Reactive | Proactive down-rank | Proactive; integrity in value model |
| **Ad load** | More ads | Fewer ads | A/B test; balance revenue vs UX |

---

## 🎤 Interview Tips

### What to Emphasize

1. **Multi-objective**—engagement + quality + integrity; value model to combine.
2. **Multi-stage**—lightweight ranker → heavy ranker → integrity → diversity.
3. **Diversity**—MMR, category slots, filter bubble prevention.
4. **Explore/exploit**—Thompson sampling, epsilon-greedy for new creators.
5. **Scale**—billions of feed requests, <200ms, real-time signals.

### Common Follow-ups

1. **How do you prevent filter bubbles?** — Diversity injection, exploration fraction, down-rank over-exposed sources.
2. **How do you balance engagement vs long-term satisfaction?** — Include quality signals, reduce clickbait, measure retention.
3. **How do you incorporate real-time signals?** — Streaming feature pipeline, separate real-time feature store.
4. **How do you A/B test a new ranking model?** — Holdback, compare engagement + satisfaction + revenue.
5. **How do you handle integrity (misinformation)?** — Integrity classifier in value model with heavy negative weight; policy layer.

### Red Flags to Avoid

- Ignoring multi-objective trade-offs
- No discussion of diversity
- No exploration strategy
- Underestimating latency budget (split across stages)

---

## 🔗 Related Topics

- [Recommendation Systems](./01-recommendation-systems.md) — Similar retrieval + ranking
- [Ad Click Prediction](./07-ad-click-prediction.md) — Ads integrated into feed
- [Feature Stores](../../phase-2-core-components/03-feature-engineering/01-feature-stores.md) — Feature enrichment
- [A/B Testing](../../phase-2-core-components/05-model-serving/03-ab-testing.md) — Ranking experiments
