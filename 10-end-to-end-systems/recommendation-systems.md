# Recommendation Systems

## Overview

Recommendation systems suggest items to users based on their preferences and behavior. They are used in e-commerce, content platforms, and many other applications.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              User Interactions                           │
│  (Clicks, Views, Purchases, Ratings)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Collection                             │
│  (Event Streaming, Batch Ingestion)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  User       │  │  Item       │  │  Context     │ │
│  │  Features   │  │  Features   │  │  Features    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Model Training                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Collaborative│ │  Content-    │  │  Hybrid     │ │
│  │  Filtering   │  │  Based      │  │  Models     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Model Serving                               │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  Real-time   │  │  Batch      │                     │
│  │  Ranking     │  │  Precompute │                     │
│  └──────────────┘  └──────────────┘                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Recommendations                             │
│  (Personalized, Ranked, Filtered)                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Components

### 1. Data Collection
- User interactions (clicks, views, purchases)
- User profiles (demographics, preferences)
- Item metadata (categories, attributes)
- Context (time, location, device)

### 2. Feature Engineering
- User features (history, preferences, demographics)
- Item features (popularity, categories, attributes)
- Interaction features (co-occurrence, sequences)
- Context features (time, location)

### 3. Models
- **Collaborative Filtering**: User-item interactions
- **Content-Based**: Item attributes
- **Deep Learning**: Neural collaborative filtering
- **Hybrid**: Combine multiple approaches

### 4. Serving
- **Real-time**: On-demand ranking
- **Batch**: Precomputed recommendations
- **Hybrid**: Precompute + real-time reranking

---

## 📝 Implementation Example

### Feature Engineering

```python
def compute_user_features(user_id):
    # User history
    interactions = get_user_interactions(user_id)
    
    features = {
        'total_interactions': len(interactions),
        'avg_rating': interactions['rating'].mean(),
        'preferred_categories': get_top_categories(interactions),
        'recency_score': compute_recency(interactions)
    }
    
    return features
```

### Model Training

```python
from surprise import SVD, Dataset, Reader

# Load data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)

# Train model
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Save model
save_model(algo, 'recommendation_model')
```

### Serving

```python
def get_recommendations(user_id, n=10):
    # Get candidate items
    candidates = get_candidate_items(user_id)
    
    # Score items
    scores = []
    for item_id in candidates:
        score = model.predict(user_id, item_id).est
        scores.append((item_id, score))
    
    # Rank and return top N
    top_items = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    return [item_id for item_id, _ in top_items]
```

---

## ✅ Best Practices

1. **Cold start handling** - new users/items
2. **Diversity** - avoid filter bubbles
3. **Freshness** - update recommendations regularly
4. **A/B testing** - test different approaches
5. **Monitoring** - track CTR, engagement

---

## 🔗 Related Topics

- [Feature Engineering](../03-feature-engineering/README.md)
- [Model Serving](../05-model-serving/README.md)
- [Monitoring](../06-monitoring-observability/README.md)
