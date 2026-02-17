# Recommendation Systems

## Overview

Recommendation systems suggest relevant items to users based on their preferences, behavior, and context. They power personalization across e-commerce, streaming, social media, and content platforms. A well-designed recommendation system can significantly improve user engagement and business metrics like revenue, retention, and time spent.

---

## 🎯 Problem Definition

### Business Goals
- Increase user engagement (clicks, time spent)
- Improve conversion rates (purchases, subscriptions)
- Enhance user satisfaction
- Increase content/product discovery
- Reduce churn

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Latency** | < 100ms p99 |
| **Throughput** | 10K-100K QPS |
| **Scale** | Millions of users, millions of items |
| **Freshness** | New items discoverable within hours |
| **Personalization** | User-specific recommendations |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Recommendation System Architecture                     │
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  User Action   │                                                          │
│  │  (View, Click, │                                                          │
│  │   Purchase)    │                                                          │
│  └───────┬────────┘                                                          │
│          │                                                                    │
│          ▼                                                                    │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │  Event Stream  │───▶│  Feature       │───▶│  Training      │            │
│  │  (Kafka)       │    │  Pipeline      │    │  Pipeline      │            │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘            │
│          │                     │                     │                       │
│          │                     ▼                     ▼                       │
│          │             ┌────────────────┐    ┌────────────────┐            │
│          │             │  Feature Store │    │  Model         │            │
│          │             │  (Online/      │    │  Registry      │            │
│          │             │   Offline)     │    │                │            │
│          │             └───────┬────────┘    └───────┬────────┘            │
│          │                     │                     │                       │
│          ▼                     ▼                     ▼                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Recommendation Service                         │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │       │
│  │  │  Candidate   │──│   Ranking    │──│  Re-ranking  │──▶ Items  │       │
│  │  │  Generation  │  │   Model      │  │  (Diversity, │           │       │
│  │  │              │  │              │  │   Business)  │           │       │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Monitoring    │◀─── Metrics, Predictions, Feedback                      │
│  │  & Analytics   │                                                          │
│  └────────────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Data Collection

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import json
from kafka import KafkaProducer

@dataclass
class UserEvent:
    """User interaction event."""
    event_id: str
    user_id: str
    item_id: str
    event_type: str  # view, click, add_to_cart, purchase, rating
    timestamp: datetime
    
    # Context
    device_type: str
    platform: str
    session_id: str
    page_type: str  # home, search, category, item_detail
    
    # Optional attributes
    rating: Optional[float] = None
    dwell_time_seconds: Optional[float] = None
    position: Optional[int] = None  # Position in list
    
    # Attribution
    recommendation_id: Optional[str] = None
    experiment_id: Optional[str] = None

class EventCollector:
    """Collect and publish user events."""
    
    def __init__(self, kafka_servers: str, topic: str):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic
    
    def track_event(self, event: UserEvent):
        """Track user interaction event."""
        self.producer.send(
            self.topic,
            key=event.user_id.encode(),
            value=event.__dict__
        )
    
    def track_impression(
        self,
        user_id: str,
        items: List[str],
        recommendation_id: str,
        page_type: str
    ):
        """Track recommendation impression."""
        for position, item_id in enumerate(items):
            event = UserEvent(
                event_id=f"{recommendation_id}_{position}",
                user_id=user_id,
                item_id=item_id,
                event_type="impression",
                timestamp=datetime.utcnow(),
                position=position,
                recommendation_id=recommendation_id,
                page_type=page_type
            )
            self.track_event(event)
```

### 2. Feature Engineering

```python
from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta

class RecommendationFeatures:
    """Feature engineering for recommendations."""
    
    def __init__(self, feature_store, item_catalog):
        self.feature_store = feature_store
        self.catalog = item_catalog
    
    def get_user_features(self, user_id: str) -> Dict[str, Any]:
        """Get user features from feature store."""
        features = self.feature_store.get_online_features(
            entity_rows=[{"user_id": user_id}],
            feature_refs=[
                # Demographics
                "user_features:age_bucket",
                "user_features:gender",
                "user_features:location_cluster",
                
                # Engagement
                "user_features:total_views_7d",
                "user_features:total_purchases_30d",
                "user_features:avg_session_duration",
                "user_features:days_since_last_visit",
                
                # Preferences
                "user_features:preferred_categories",
                "user_features:preferred_price_range",
                "user_features:preferred_brands",
                
                # Embeddings
                "user_features:user_embedding"
            ]
        )
        return features.to_dict()
    
    def get_item_features(self, item_ids: List[str]) -> Dict[str, Dict]:
        """Get item features for multiple items."""
        features = self.feature_store.get_online_features(
            entity_rows=[{"item_id": item_id} for item_id in item_ids],
            feature_refs=[
                # Static attributes
                "item_features:category",
                "item_features:brand",
                "item_features:price",
                
                # Popularity
                "item_features:view_count_7d",
                "item_features:purchase_count_7d",
                "item_features:avg_rating",
                "item_features:num_ratings",
                
                # Content
                "item_features:title_embedding",
                "item_features:image_embedding",
                
                # Temporal
                "item_features:days_since_added",
                "item_features:trending_score"
            ]
        )
        return {row["item_id"]: row for row in features.to_dicts()}
    
    def compute_user_item_features(
        self,
        user_id: str,
        item_id: str,
        context: Dict
    ) -> Dict[str, float]:
        """Compute features for user-item pair."""
        user = self.get_user_features(user_id)
        item = self.get_item_features([item_id])[item_id]
        
        features = {
            # Interaction features
            "user_viewed_category": int(item["category"] in user.get("preferred_categories", [])),
            "user_bought_brand": int(item["brand"] in user.get("preferred_brands", [])),
            "price_in_range": self._price_in_range(item["price"], user.get("preferred_price_range")),
            
            # Embedding similarity
            "embedding_similarity": self._cosine_similarity(
                user.get("user_embedding"),
                item.get("title_embedding")
            ),
            
            # Popularity features
            "item_popularity_score": np.log1p(item.get("view_count_7d", 0)),
            "item_conversion_rate": item.get("purchase_count_7d", 0) / max(item.get("view_count_7d", 1), 1),
            
            # Context features
            "is_weekend": int(datetime.now().weekday() >= 5),
            "hour_of_day": datetime.now().hour,
            "device_mobile": int(context.get("device_type") == "mobile")
        }
        
        return features
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _price_in_range(self, price: float, price_range: tuple) -> float:
        if price_range is None:
            return 0.5
        low, high = price_range
        if price < low:
            return max(0, 1 - (low - price) / low)
        elif price > high:
            return max(0, 1 - (price - high) / high)
        return 1.0
```

### 3. Candidate Generation

```python
from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

class CandidateGenerator(ABC):
    """Base class for candidate generation."""
    
    @abstractmethod
    def generate(self, user_id: str, n: int) -> List[str]:
        pass

class CollaborativeFilteringCandidates(CandidateGenerator):
    """Item-based collaborative filtering."""
    
    def __init__(self, item_similarity_index, user_history_store):
        self.similarity_index = item_similarity_index
        self.history_store = user_history_store
    
    def generate(self, user_id: str, n: int = 100) -> List[str]:
        # Get user's recent interactions
        recent_items = self.history_store.get_recent_items(user_id, limit=50)
        
        # Get similar items
        candidates = set()
        for item_id in recent_items:
            similar = self.similarity_index.get_similar(item_id, k=20)
            candidates.update(similar)
        
        # Remove already interacted items
        candidates = candidates - set(recent_items)
        
        return list(candidates)[:n]

class EmbeddingCandidates(CandidateGenerator):
    """ANN-based candidate generation using embeddings."""
    
    def __init__(self, user_embedding_model, item_index):
        self.user_model = user_embedding_model
        self.item_index = item_index  # Approximate nearest neighbor index
    
    def generate(self, user_id: str, n: int = 100) -> List[str]:
        # Get user embedding
        user_embedding = self.user_model.get_embedding(user_id)
        
        # Query ANN index
        item_ids, distances = self.item_index.search(
            user_embedding,
            k=n
        )
        
        return item_ids

class PopularityCandidates(CandidateGenerator):
    """Popularity-based candidates (fallback)."""
    
    def __init__(self, popularity_store):
        self.popularity_store = popularity_store
    
    def generate(self, user_id: str, n: int = 100) -> List[str]:
        # Get popular items by category
        return self.popularity_store.get_top_items(n=n)

class HybridCandidateGenerator:
    """Combine multiple candidate generation strategies."""
    
    def __init__(self, generators: Dict[str, CandidateGenerator], weights: Dict[str, float]):
        self.generators = generators
        self.weights = weights
    
    def generate(self, user_id: str, n: int = 500) -> List[str]:
        all_candidates = {}
        
        for name, generator in self.generators.items():
            weight = self.weights.get(name, 1.0)
            items_per_source = int(n * weight / sum(self.weights.values()))
            
            candidates = generator.generate(user_id, items_per_source)
            
            for item_id in candidates:
                if item_id not in all_candidates:
                    all_candidates[item_id] = []
                all_candidates[item_id].append(name)
        
        # Score by number of sources that nominated the item
        scored = [(item, len(sources)) for item, sources in all_candidates.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in scored[:n]]

# Usage
generator = HybridCandidateGenerator(
    generators={
        "collaborative": CollaborativeFilteringCandidates(...),
        "embedding": EmbeddingCandidates(...),
        "popularity": PopularityCandidates(...)
    },
    weights={
        "collaborative": 0.4,
        "embedding": 0.4,
        "popularity": 0.2
    }
)
```

### 4. Ranking Model

```python
import torch
import torch.nn as nn
from typing import List, Dict

class TwoTowerModel(nn.Module):
    """Two-tower model for recommendation ranking."""
    
    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()
        
        # User tower
        user_layers = []
        prev_dim = user_feature_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item tower
        item_layers = []
        prev_dim = item_feature_dim
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        item_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)
    
    def forward(self, user_features, item_features):
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        
        # Dot product similarity
        scores = torch.sum(user_embedding * item_embedding, dim=1)
        return scores
    
    def get_user_embedding(self, user_features):
        return self.user_tower(user_features)
    
    def get_item_embedding(self, item_features):
        return self.item_tower(item_features)

class RankingService:
    """Service for ranking candidates."""
    
    def __init__(self, model, feature_service):
        self.model = model
        self.model.eval()
        self.feature_service = feature_service
    
    @torch.no_grad()
    def rank(
        self,
        user_id: str,
        candidate_items: List[str],
        context: Dict,
        n: int = 20
    ) -> List[Dict]:
        """Rank candidate items for a user."""
        
        # Get features
        user_features = self.feature_service.get_user_features(user_id)
        item_features = self.feature_service.get_item_features(candidate_items)
        
        # Prepare tensors
        user_tensor = torch.tensor([self._encode_user(user_features)] * len(candidate_items))
        item_tensor = torch.tensor([self._encode_item(item_features[item]) for item in candidate_items])
        
        # Score
        scores = self.model(user_tensor, item_tensor).numpy()
        
        # Rank
        ranked_indices = np.argsort(scores)[::-1][:n]
        
        results = []
        for idx in ranked_indices:
            results.append({
                "item_id": candidate_items[idx],
                "score": float(scores[idx]),
                "features": item_features[candidate_items[idx]]
            })
        
        return results
    
    def _encode_user(self, features: Dict) -> List[float]:
        """Encode user features to vector."""
        # Implementation depends on feature schema
        pass
    
    def _encode_item(self, features: Dict) -> List[float]:
        """Encode item features to vector."""
        # Implementation depends on feature schema
        pass
```

### 5. Re-ranking (Business Rules & Diversity)

```python
from typing import List, Dict
from collections import defaultdict

class ReRanker:
    """Apply business rules and diversity to ranked items."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def rerank(
        self,
        ranked_items: List[Dict],
        user_context: Dict
    ) -> List[Dict]:
        """Apply re-ranking rules."""
        
        items = ranked_items.copy()
        
        # Apply business rules
        items = self._apply_business_rules(items, user_context)
        
        # Apply diversity
        items = self._apply_diversity(items)
        
        # Apply position bias correction
        items = self._apply_position_bias(items)
        
        return items
    
    def _apply_business_rules(
        self,
        items: List[Dict],
        context: Dict
    ) -> List[Dict]:
        """Apply business rules (promotions, filtering)."""
        
        result = []
        for item in items:
            # Filter out of stock
            if not item.get("features", {}).get("in_stock", True):
                continue
            
            # Boost promoted items
            if item.get("features", {}).get("is_promoted"):
                item["score"] *= self.config.get("promotion_boost", 1.5)
            
            # Boost new arrivals for returning users
            if (item.get("features", {}).get("days_since_added", 100) < 7 and
                context.get("is_returning_user")):
                item["score"] *= self.config.get("new_item_boost", 1.2)
            
            result.append(item)
        
        return result
    
    def _apply_diversity(self, items: List[Dict]) -> List[Dict]:
        """Apply MMR-style diversity."""
        
        if not items:
            return items
        
        selected = [items[0]]
        remaining = items[1:]
        
        max_per_category = self.config.get("max_per_category", 5)
        category_counts = defaultdict(int)
        category_counts[items[0]["features"].get("category")] = 1
        
        diversity_weight = self.config.get("diversity_weight", 0.3)
        
        while remaining and len(selected) < len(items):
            best_score = -float('inf')
            best_idx = 0
            
            for i, item in enumerate(remaining):
                category = item["features"].get("category")
                
                # Skip if category is saturated
                if category_counts[category] >= max_per_category:
                    continue
                
                # Calculate diversity score (average distance to selected)
                similarity_to_selected = self._avg_similarity(item, selected)
                
                # Combined score: relevance - diversity_weight * similarity
                combined_score = item["score"] - diversity_weight * similarity_to_selected
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i
            
            if best_score > -float('inf'):
                selected_item = remaining.pop(best_idx)
                selected.append(selected_item)
                category_counts[selected_item["features"].get("category")] += 1
            else:
                break
        
        return selected
    
    def _avg_similarity(self, item: Dict, selected: List[Dict]) -> float:
        """Calculate average similarity to selected items."""
        if not selected:
            return 0
        
        similarities = []
        for s in selected:
            # Same category = high similarity
            if item["features"].get("category") == s["features"].get("category"):
                similarities.append(1.0)
            elif item["features"].get("brand") == s["features"].get("brand"):
                similarities.append(0.5)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities)
    
    def _apply_position_bias(self, items: List[Dict]) -> List[Dict]:
        """Correct for position bias in training data."""
        # Implementation of inverse propensity weighting or similar
        return items
```

### 6. Complete Recommendation Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: str
    page_type: str  # home, item_detail, cart
    context_item_id: Optional[str] = None
    num_items: int = 20
    exclude_items: List[str] = []

class RecommendationResponse(BaseModel):
    recommendation_id: str
    items: List[Dict]
    latency_ms: float

class RecommendationService:
    """Complete recommendation service."""
    
    def __init__(
        self,
        candidate_generator: HybridCandidateGenerator,
        ranking_service: RankingService,
        reranker: ReRanker,
        event_logger
    ):
        self.candidate_gen = candidate_generator
        self.ranking = ranking_service
        self.reranker = reranker
        self.logger = event_logger
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """Get personalized recommendations."""
        
        start_time = time.time()
        recommendation_id = str(uuid.uuid4())
        
        try:
            # Stage 1: Candidate Generation
            candidates = self.candidate_gen.generate(
                user_id=request.user_id,
                n=500
            )
            
            # Filter excluded items
            candidates = [c for c in candidates if c not in request.exclude_items]
            
            # Stage 2: Ranking
            context = {
                "page_type": request.page_type,
                "context_item_id": request.context_item_id
            }
            
            ranked_items = self.ranking.rank(
                user_id=request.user_id,
                candidate_items=candidates,
                context=context,
                n=request.num_items * 2  # Get more for re-ranking
            )
            
            # Stage 3: Re-ranking
            user_context = await self._get_user_context(request.user_id)
            final_items = self.reranker.rerank(
                ranked_items=ranked_items,
                user_context=user_context
            )[:request.num_items]
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Log for analytics
            await self.logger.log_recommendation(
                recommendation_id=recommendation_id,
                user_id=request.user_id,
                items=[item["item_id"] for item in final_items],
                latency_ms=latency_ms
            )
            
            return RecommendationResponse(
                recommendation_id=recommendation_id,
                items=final_items,
                latency_ms=latency_ms
            )
        
        except Exception as e:
            # Fallback to popular items
            return await self._fallback_recommendations(request)
    
    async def _get_user_context(self, user_id: str) -> Dict:
        """Get user context for re-ranking."""
        # Implementation
        pass
    
    async def _fallback_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """Fallback to popularity-based recommendations."""
        # Implementation
        pass

# API Endpoint
service = RecommendationService(...)

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    return await service.get_recommendations(request)
```

---

## 📈 Metrics & Evaluation

### Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CTR** | Click-through rate | > 5% |
| **Conversion Rate** | Purchase rate | > 2% |
| **Coverage** | % of catalog recommended | > 30% |
| **Diversity** | Category diversity in results | > 0.7 |
| **Novelty** | Avg popularity rank of recs | Medium |

### Offline Metrics

```python
def evaluate_recommendations(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int = 10
) -> Dict[str, float]:
    """Evaluate recommendation quality."""
    
    metrics = {
        "precision@k": [],
        "recall@k": [],
        "ndcg@k": [],
        "map@k": [],
        "hit_rate@k": []
    }
    
    for pred, truth in zip(predictions, ground_truth):
        pred_k = pred[:k]
        truth_set = set(truth)
        
        # Precision@K
        hits = len(set(pred_k) & truth_set)
        metrics["precision@k"].append(hits / k)
        
        # Recall@K
        metrics["recall@k"].append(hits / len(truth_set) if truth_set else 0)
        
        # Hit Rate@K
        metrics["hit_rate@k"].append(1 if hits > 0 else 0)
        
        # NDCG@K
        dcg = sum((1 / np.log2(i + 2)) for i, item in enumerate(pred_k) if item in truth_set)
        idcg = sum((1 / np.log2(i + 2)) for i in range(min(k, len(truth_set))))
        metrics["ndcg@k"].append(dcg / idcg if idcg > 0 else 0)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Latency vs Quality** | More candidates + complex ranking (higher quality) | Fewer candidates + simple ranking (lower latency) |
| **Personalization vs Exploration** | Heavily personalized (engagement) | More diverse (discovery) |
| **Freshness vs Stability** | Frequent updates (fresh) | Less frequent (stable) |
| **Simple vs Complex** | Rule-based/CF (interpretable) | Deep learning (higher accuracy) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How would you handle cold start for new users/items?
2. How do you ensure diversity in recommendations?
3. How would you A/B test a new recommendation algorithm?
4. How do you handle the feedback loop problem?
5. How would you scale to 100M users?

**Key Points to Mention:**
- Two-stage architecture (candidate generation + ranking)
- Feature engineering importance
- Online vs offline evaluation
- Business constraints and rules
- Monitoring and feedback loops

---

## 🔗 Related Topics

- [Feature Engineering](../../phase-2-core-components/03-feature-engineering/00-README.md)
- [A/B Testing](../../phase-2-core-components/05-model-serving/03-ab-testing.md)
- [Caching Strategies](../../phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md)
