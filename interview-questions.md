# ML System Design Interview Questions

A comprehensive collection of ML system design interview questions with frameworks for approaching them.

---

## 📋 Interview Framework

### Step 1: Clarify Requirements (5 min)
- What is the business goal?
- Who are the users?
- What are the scale requirements (QPS, data volume)?
- What are the latency requirements?
- What are the accuracy requirements?
- What constraints exist (budget, time, team)?

### Step 2: Define Metrics (5 min)
- Offline metrics (accuracy, precision, recall, AUC)
- Online metrics (CTR, conversion, revenue)
- System metrics (latency, throughput, availability)

### Step 3: High-Level Design (10 min)
- Data pipeline
- Feature engineering
- Model training
- Model serving
- Monitoring

### Step 4: Deep Dive (15 min)
- Pick 1-2 components to detail
- Discuss trade-offs
- Address interviewer questions

### Step 5: Extensions (5 min)
- Scaling considerations
- Edge cases
- Future improvements

---

## 🎯 Common Interview Questions

### 1. Design a Recommendation System

**Clarifying Questions:**
- What type? (Products, content, friends)
- How many users/items?
- Real-time or batch?
- Cold start handling?

**Key Components:**
```
Data Collection → Feature Engineering → Candidate Generation → Ranking → Serving
```

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  User Request                                                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Candidate Generation (1000s → 100s)                            │
│  - Collaborative filtering                                       │
│  - Content-based filtering                                       │
│  - Popular items (cold start)                                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ranking Model (100s → 10s)                                     │
│  - User features + Item features + Context                      │
│  - Deep learning model (Two-tower, DCN)                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Post-processing                                                 │
│  - Diversity                                                     │
│  - Business rules                                                │
│  - Filtering (already seen, inappropriate)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Trade-offs:**
| Aspect | Option A | Option B |
|--------|----------|----------|
| **Latency** | Pre-computed (fast) | Real-time ranking (fresh) |
| **Cold start** | Popular items | Content-based |
| **Model** | Matrix factorization (simple) | Deep learning (accurate) |

---

### 2. Design a Fraud Detection System

**Clarifying Questions:**
- What type of fraud? (Payment, account takeover, fake accounts)
- What's the fraud rate? (Affects evaluation)
- Latency requirements? (Real-time vs batch)
- False positive tolerance?

**Key Components:**
```
Transaction → Feature Engineering → Real-time Scoring → Decision → Feedback Loop
```

**Feature Categories:**
- **Transaction features:** Amount, merchant, location, time
- **User features:** History, behavior patterns, device
- **Aggregated features:** Transaction velocity, amount patterns
- **Graph features:** Connection to known fraudsters

**Real-time Pipeline:**
```python
async def score_transaction(transaction):
    # 1. Feature retrieval (parallel)
    user_features, merchant_features = await asyncio.gather(
        feature_store.get(f"user:{transaction.user_id}"),
        feature_store.get(f"merchant:{transaction.merchant_id}")
    )
    
    # 2. Real-time features
    realtime_features = compute_realtime_features(transaction)
    
    # 3. Scoring
    features = combine_features(transaction, user_features, 
                               merchant_features, realtime_features)
    score = model.predict(features)
    
    # 4. Decision
    if score > HIGH_RISK_THRESHOLD:
        return Decision.BLOCK
    elif score > MEDIUM_RISK_THRESHOLD:
        return Decision.REVIEW
    else:
        return Decision.ALLOW
```

**Challenges:**
- **Class imbalance:** 0.1% fraud rate → use precision@recall, AUC-PR
- **Concept drift:** Fraud patterns evolve → continuous monitoring
- **Latency:** <100ms for real-time decisions
- **Explainability:** Why was this flagged?

---

### 3. Design a Search Ranking System

**Clarifying Questions:**
- What are we searching? (Products, documents, users)
- Query volume?
- Latency requirements?
- Personalization needed?

**Key Components:**
```
Query → Query Understanding → Retrieval → Ranking → Results
```

**Two-Stage Architecture:**
1. **Retrieval (1M → 1000):** Fast, approximate
   - Inverted index (BM25)
   - Vector similarity (embeddings)
   
2. **Ranking (1000 → 10):** Slow, precise
   - Learning to rank
   - Deep models (BERT-based)

**Features:**
- **Query features:** Length, intent, entities
- **Document features:** Quality, freshness, popularity
- **Query-document features:** BM25, semantic similarity
- **User features:** History, preferences

**Evaluation:**
- **Offline:** NDCG, MRR, MAP
- **Online:** CTR, time to click, search success rate

---

### 4. Design an Ad Click Prediction System

**Clarifying Questions:**
- Ad format? (Search, display, video)
- Prediction target? (Click, conversion, engagement)
- Latency budget?
- Training data volume?

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Ad Request (User + Context)                                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│  Ad Candidate Pool   │        │  User Features       │
│  (Targeting rules)   │        │  (From feature store)│
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Click Prediction Model                                          │
│  - User embeddings + Ad embeddings + Context                    │
│  - Output: P(click)                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Auction (eCPM = bid × P(click))                                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Challenges:**
- **Scale:** Billions of predictions/day
- **Latency:** <50ms end-to-end
- **Feature freshness:** Real-time user behavior
- **Calibration:** P(click) must be well-calibrated for bidding

---

### 5. Design a Content Moderation System

**Clarifying Questions:**
- Content type? (Text, image, video)
- Moderation policies?
- Accuracy vs coverage trade-off?
- Human review workflow?

**Multi-Model Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Content Upload                                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Text            │ │  Image           │ │  Video           │
│  Classifier      │ │  Classifier      │ │  Classifier      │
│  - Hate speech   │ │  - NSFW          │ │  - Violence      │
│  - Spam          │ │  - Violence      │ │  - Audio check   │
│  - Harassment    │ │  - Illegal       │ │  - Frame sample  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Decision Engine                                                 │
│  - Auto-remove (high confidence violations)                     │
│  - Human review (medium confidence)                             │
│  - Allow (low risk)                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Human-in-the-Loop:**
- Use human review for edge cases
- Human labels improve model over time
- Handle appeals

---

### 6. Design an ML Platform / Feature Store

**Clarifying Questions:**
- Team size?
- Model types to support?
- Online vs offline features?
- Latency requirements for feature serving?

**Feature Store Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Definition Layer                      │
│  - Feature schemas                                               │
│  - Transformation logic                                          │
│  - Dependencies                                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    Offline Store         │    │    Online Store          │
│  - Historical features   │    │  - Low latency (<10ms)   │
│  - Training data         │    │  - Current values        │
│  - Point-in-time correct │    │  - Key-value access      │
│  (Data warehouse)        │    │  (Redis/DynamoDB)        │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    Model Training        │    │    Model Serving         │
│  (Offline features)      │    │  (Online features)       │
└──────────────────────────┘    └──────────────────────────┘
```

---

## 💡 Common Follow-up Questions

### Scaling Questions
- "How would you handle 10x traffic?"
- "What if you have 1B users?"
- "How do you reduce latency?"

### Reliability Questions
- "What happens if the model fails?"
- "How do you handle data pipeline failures?"
- "How do you ensure 99.9% availability?"

### ML-Specific Questions
- "How do you handle cold start?"
- "How do you detect model degradation?"
- "How do you handle class imbalance?"
- "How do you ensure fairness?"

### Trade-off Questions
- "Latency vs accuracy trade-off?"
- "Batch vs real-time trade-off?"
- "Simple model vs complex model?"

---

## 📊 Quick Reference: System Design Numbers

| Metric | Typical Range |
|--------|---------------|
| **API latency (p99)** | 100-500ms |
| **Feature store latency** | 5-50ms |
| **Model inference** | 10-100ms |
| **Batch job frequency** | Daily/hourly |
| **Training data size** | GB-TB |
| **Model size** | MB-GB |
| **QPS per instance** | 100-1000 |
| **Cache hit rate** | 80-95% |

---

## ✅ Interview Checklist

Before finishing, make sure you've covered:

- [ ] Clarified requirements and constraints
- [ ] Defined success metrics (offline + online)
- [ ] Drew high-level architecture
- [ ] Discussed data collection and storage
- [ ] Explained feature engineering approach
- [ ] Described model choice and training
- [ ] Explained serving strategy
- [ ] Discussed monitoring and iteration
- [ ] Addressed scaling and reliability
- [ ] Discussed trade-offs made

---

## 🔗 Related Topics

- [End-to-End Systems](./10-end-to-end-systems/README.md)
- [Model Serving](./05-model-serving/README.md)
- [Monitoring](./06-monitoring-observability/README.md)
