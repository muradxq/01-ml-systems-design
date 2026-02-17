# Interview Framework: The CLEAR Method

The CLEAR Method is a structured approach for answering ML System Design questions in 45 minutes. It ensures you cover the right topics, use time effectively, and demonstrate systematic thinking.

---

## Overview: The Five Phases

| Phase | Time | Focus |
|-------|------|-------|
| **C** - Clarify | 5–8 min | Understand the problem, constraints, and scope |
| **L** - List Requirements | 3–5 min | Functional, non-functional, and ML-specific requirements |
| **E** - Estimate Scale | 3–5 min | Users, QPS, storage, bandwidth |
| **A** - Architect | 15–20 min | End-to-end design with 2–3 deep dives |
| **R** - Refine | 5–10 min | Trade-offs, failure modes, improvements |

---

## C — Clarify (5–8 minutes)

Never jump into design. Spend 5–8 minutes ensuring you understand the problem.

### What to Ask

**Business Context**
- What business problem are we solving?
- Who are the users? (e.g., consumers, advertisers, internal)
- What does success look like? (engagement, revenue, CTR, satisfaction)
- Is this a new system or improving an existing one?

**Product and Scope**
- What are we predicting/recommending/classifying?
- What is the input? (user context, query, history, item catalog)
- What is the output? (ranking, score, category, list of items)
- Are there any product constraints? (e.g., diversity, fairness, safety)

**Performance and Scale**
- What are the latency requirements? (p50, p99)
- What is the expected throughput (QPS)?
- How many users? How many items in the catalog (if applicable)?
- What is the expected growth in 1–2 years?

**Constraints**
- Budget (compute, storage, team size)?
- Timeline (MVP in 3 months vs production-ready in 6)?
- Existing infrastructure? (feature store, model serving, data warehouse)
- Any compliance or fairness requirements?

**Edge Cases**
- How do we handle new users (cold start)?
- How do we handle new items (cold start)?
- Offline vs real-time use cases?

### Sample Clarification Dialog

**Interviewer:** *"Design a recommendation system for a news app."*

**Candidate:** *"Thanks. Before I dive in, I’d like to clarify a few things. First, what business goal are we optimizing for—engagement, time on app, or subscriptions?"*

**Interviewer:** *"Engagement. We want users to read more articles."*

**Candidate:** *"Got it. Who are the primary users—new users, power users, or both? And roughly how many daily active users are we targeting?"*

**Interviewer:** *"Both. Assume around 10M DAU."*

**Candidate:** *"What are the latency requirements? Is this for a feed that loads on app open, or something more real-time?"*

**Interviewer:** *"Feed on app open. We need results within 200ms p99."*

**Candidate:** *"Is there an existing system we’re replacing, or is this greenfield?"*

**Interviewer:** *"We have a rule-based system. We want to add ML ranking."*

**Candidate:** *"Last question: any diversity or fairness constraints—e.g., balancing topics, sources, or recency?"*

**Interviewer:** *"Yes, we want some diversity. No single publisher should dominate the feed."*

*[Candidate has enough context to proceed.]*

---

## L — List Requirements (3–5 minutes)

Write down requirements in a structured format. This shows rigor and gives you a checklist for the design.

### Functional Requirements

| Requirement | Description |
|-------------|-------------|
| FR1 | Generate personalized ranking of N items (e.g., 20 articles) for each user |
| FR2 | Support new user cold start (anonymous or first-time) |
| FR3 | Support new item cold start |
| FR4 | Apply diversity and fairness constraints |
| FR5 | Log interactions for feedback and evaluation |

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| NFR1 | Latency: p50 < 50ms, p99 < 200ms |
| NFR2 | Throughput: 10K QPS peak |
| NFR3 | Availability: 99.9% |
| NFR4 | Freshness: model retrained daily; features updated in near real-time |

### ML-Specific Requirements

| Requirement | Target |
|-------------|--------|
| ML1 | Offline metric: NDCG@20 or similar |
| ML2 | Online: A/B test against baseline (e.g., +2% engagement) |
| ML3 | Training-serving consistency: features match across training and serving |
| ML4 | Monitoring: data drift, concept drift, model performance |

### Out of Scope

| Item | Reason |
|------|--------|
| Content creation/ingestion | Assume upstream pipeline exists |
| User authentication | Assume identity service exists |
| Ad serving | Separate system |
| Push notifications | Separate system |

---

## E — Estimate Scale (3–5 minutes)

Show you can reason about numbers. Use round numbers and simple math.

### Users, Items, Events

| Metric | Estimate | Notes |
|--------|----------|-------|
| DAU | 10M | From clarification |
| Items (articles) | 100K new/day, 10M catalog | Scale with publishers |
| Events (clicks, views) | 500M/day | ~50 interactions per DAU |

### QPS

| Type | Calculation | Result |
|------|-------------|--------|
| Read (recommendation requests) | 10M DAU × 5 sessions × 2 requests / 86,400 s | ~1,200 QPS average |
| Peak | 3× average | ~3,600 QPS |
| Write (events) | 500M / 86,400 | ~5,800 write QPS |

### Storage (Annual)

| Data | Calculation | Result |
|------|-------------|--------|
| Events (1KB each) | 500M × 365 × 1KB | ~180 TB/year |
| Model artifacts | 1GB per model × 10 versions | ~10 GB |
| Features | 10M users × 500 floats × 4 bytes | ~20 GB for user embeddings |

### Bandwidth

| Flow | Calculation | Result |
|------|-------------|--------|
| Request (20 items × 1KB) | 3,600 × 20KB | ~70 MB/s out |
| Event ingestion | 5,800 × 0.5KB | ~3 MB/s in |

Use these estimates to justify choices (e.g., “At 3.6K QPS we can serve from a single region with replication”).

---

## A — Architect (15–20 minutes)

Design the end-to-end system. Start high-level, then go deep on 2–3 components.

### High-Level Flow

```
[Data Sources] → [Data Pipeline] → [Feature Store]
                                        ↓
[Training Pipeline] ← [Offline Features] 
        ↓
[Model Registry] → [Serving Layer] → [API / Client]
        ↑
[Monitoring] ← [Feedback Logs]
```

### 1. Data Pipeline Design

- **Sources**: Clicks, views, dwell time, user attributes, item metadata
- **Batch**: Daily/hourly aggregation (Spark/Flink)
- **Streaming**: Real-time events (Kafka) for online features
- **Storage**: Data lake (S3/GCS) for raw events; feature store for derived features

### 2. Feature Engineering

- **User features**: Demographics, past engagement, session context
- **Item features**: Content, recency, popularity
- **Interaction features**: Historical CTR, co-engagement
- **Online vs offline**: User embedding (offline, refreshed daily) vs real-time context (online)
- **Feature store**: Central store for offline training and online serving; ensures consistency

### 3. Model Selection and Training

- **MVP**: Two-tower (user tower, item tower) or logistic regression on handcrafted features
- **Evolution**: Matrix factorization, then deep learning (two-tower, DNN) if needed
- **Training**: Batch on historical data; daily retraining; experiment tracking (MLflow)
- **Data**: Sample negative examples (e.g., impressed but not clicked); handle position bias

### 4. Serving Infrastructure

- **Retrieval**: Candidate generation (e.g., top 500 from ANN/FAISS) based on user embedding
- **Ranking**: Lightweight model (e.g., DNN) reranks top 500 to top 20
- **Infra**: Model server (TensorFlow Serving, TorchServe); replica for load; caching for hot items
- **A/B testing**: Traffic split by experiment config

### 5. Monitoring and Feedback Loop

- **Input monitoring**: Feature distributions, missing values, drift
- **Output monitoring**: Score distribution, latency
- **Business metrics**: CTR, engagement (via logging pipeline)
- **Feedback**: Log model scores and outcomes; use for retraining and evaluation

### Whiteboard Sketch

Draw from left to right:

1. **Data layer**: Sources (DB, logs, Kafka) → ETL → Feature Store  
2. **Training layer**: Offline features → Training job → Model registry  
3. **Serving layer**: API → Retrieval (ANN) → Ranker → Response  
4. **Monitoring**: Logs → Metrics dashboards; Alerts  

### Deep Dives (Pick 2–3 Based on Interviewer)

- **Feature store**: Schema, offline vs online, freshness, consistency
- **Retrieval**: ANN (FAISS/HNSW), two-tower, retrieval vs ranking split
- **Training**: Negative sampling, position bias, evaluation metrics
- **Serving**: Latency budget, caching, fallbacks
- **Monitoring**: Drift detection, alerting, retraining triggers

---

## R — Refine (5–10 minutes)

Discuss trade-offs, failure modes, and future improvements.

### Trade-Offs

| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Two-tower vs single model | Single model captures interaction; two-tower is fast for retrieval | Latency vs expressiveness |
| Batch vs streaming features | Streaming is fresher; batch is simpler and cheaper | Freshness vs complexity |
| ANN vs exact search | ANN is faster; exact is more accurate | Speed vs recall |
| Daily vs hourly retraining | Hourly is fresher; daily is easier to operate | Freshness vs ops cost |

### Failure Modes and Fallbacks

- **Model server down**: Fallback to cached popular items or rule-based ranking
- **Feature store slow**: Use stale features or default values with monitoring
- **Data pipeline delay**: Serve with previous day’s model; alert on staleness
- **Spike in traffic**: Autoscaling, circuit breakers, graceful degradation

### Scaling Considerations

- **10× traffic**: Horizontal scaling, more replicas, CDN for static data
- **10× catalog**: Better ANN (e.g., HNSW), sharding, approximate retrieval
- **10× features**: Feature selection, embedding, compressed representations

### Future Improvements

- Multi-objective optimization (engagement + diversity)
- Real-time learning (e.g., bandits)
- Cross-domain recommendations
- Better冷启动 (metadata-based, exploration)
- Fairness and interpretability improvements

### Fairness and Ethics

- Represent diverse sources and topics
- Monitor metrics across user segments
- Avoid feedback loops that amplify bias
- Consider regulatory and policy constraints

---

## What Interviewers Are Actually Scoring

| Criterion | Weight | What they look for |
|-----------|--------|---------------------|
| **Clarification** | 15% | Good questions, correct scope |
| **Requirements** | 10% | Structured, complete requirements |
| **Estimation** | 10% | Reasonable numbers, clear reasoning |
| **Architecture** | 35% | End-to-end design, sensible components |
| **Depth** | 20% | Ability to go deep when asked |
| **Trade-offs** | 10% | Alternatives, justification |

---

## Time Management Tips

- **0–8 min**: Clarify. If the interviewer keeps answers short, move on after 5 min.
- **8–13 min**: Requirements + Estimates. Keep concise; use tables.
- **13–33 min**: Architecture. First 5 min: high-level diagram. Next 15 min: components + 2 deep dives.
- **33–45 min**: Refine. Trade-offs, failures, scaling, future work.
- **Buffer**: Leave 2–3 min for “what would you do differently?” and wrap-up.

---

## Handling “What Would You Do Differently?”

Good structure:

1. **Acknowledge**: *“There are several things I’d reconsider.”*
2. **Prioritize**: *“First, I’d strengthen the cold-start approach…”*
3. **Be specific**: *“Instead of metadata-only, I’d add a light exploration bandit for new items.”*
4. **Link to trade-offs**: *“The trade-off is added complexity vs better new-item discovery.”*
5. **Mention 2–3 more**: Monitoring, fairness, latency, cost.

Avoid: *“I think I did everything right.”* or vague answers like *“Maybe use a better model.”*

---

## Sample Opening: “Design a Recommendation System”

**First 60 seconds:**

*“I’ll use a structured approach. First, I’ll clarify the problem and constraints. Then I’ll list requirements and do quick scale estimates. After that, I’ll design the end-to-end architecture—data, training, serving, monitoring—and go deeper on a couple of components. I’ll finish with trade-offs and failure modes.*

*To start: What are we recommending—products, content, connections? Who are the users, and what’s the scale we’re designing for? What are the latency and freshness requirements?*

*[Listen, then ask 2–3 more targeted questions before moving to requirements.]*

---

## Architecture Diagram: Recommended Whiteboard Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  [Events] → Kafka → [Stream Processing] → [Feature Store]                    │
│  [Batch DB] → Spark/Flink → [Aggregations] ─────────────────→ [Feature Store]│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LAYER                                       │
│  [Feature Store] → [Training Job] → [Model Registry] → [Versioned Models]   │
│  (Offline features)   (Daily/Weekly)    (MLflow/SageMaker)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVING LAYER                                        │
│  [API Gateway] → [Retrieval: ANN/FAISS] → [Ranker: DNN] → [Response]        │
│       │                  │                      │                           │
│       └── [Cache] ───────┴──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING                                           │
│  [Logs] → [Metrics] → [Alerts]  |  [A/B Results] → [Retraining Trigger]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

Draw this left-to-right, top-to-bottom. Label each box. Add arrows for data flow. Keep it readable from 6 feet away.

---

## When to Go Deep: Reading Interviewer Signals

| Signal | Interpretation | Action |
|--------|----------------|--------|
| "Tell me more about retrieval" | They want depth on ANN, two-tower, recall | Go technical: HNSW vs IVF, recall@K trade-offs |
| "How would you handle X?" | They're probing edge cases | Give 2–3 approaches with trade-offs |
| "What about cost?" | They care about operational efficiency | Discuss compute, storage, caching |
| "How does this work at scale?" | They want scaling strategy | Sharding, batching, approximate methods |
| Silent, taking notes | They may be satisfied or waiting | Briefly summarize; ask "Any area you'd like me to go deeper?" |
| "We only have 10 minutes" | Time to wrap up | Prioritize refine phase; skip secondary components |

---

## Estimation Cheat Sheet

| Scale | Users | QPS (reads) | Storage (events/year) |
|-------|-------|-------------|------------------------|
| Small | 100K DAU | ~100 | ~50 GB |
| Medium | 10M DAU | ~1–5K | ~50–200 TB |
| Large | 100M DAU | ~10–50K | ~500 TB – 2 PB |
| Hyperscale | 1B+ DAU | 100K+ | 10+ PB |

Assume: ~5–10 requests per user per day for feeds; ~50 events per user per day for logging.

---

## Requirements Template (Copy-Paste)

```
FUNCTIONAL:
- FR1: [Primary function]
- FR2: [Cold start - users]
- FR3: [Cold start - items]
- FR4: [Constraints: diversity, fairness]
- FR5: [Logging/feedback]

NON-FUNCTIONAL:
- NFR1: Latency: p50 __ ms, p99 __ ms
- NFR2: Throughput: __ QPS
- NFR3: Availability: __%
- NFR4: Freshness: model __, features __

ML-SPECIFIC:
- ML1: Offline metric: __
- ML2: Online: A/B test, metric __
- ML3: Training-serving consistency: __
- ML4: Monitoring: __

OUT OF SCOPE:
- [Explicit exclusions]
```

---

## Second Sample: "Design Ad Click Prediction"

**Clarify (abbreviated):** "What's the scale—impressions per day? Latency budget for ad auction? Do we have historical CTR data? Are we predicting for display or search ads? Any calibration requirements for the auction?"

**Key architecture decisions:**
- **Features**: User (demographics, history), ad (creative, placement), context (device, time)
- **Model**: Logistic regression or small DNN; calibration matters for auction
- **Training**: Per-impression labels (click/no-click); handle position and selection bias
- **Serving**: Low latency (often <10ms); batch predictions for auction
- **Monitoring**: CTR by segment, calibration curves, latency

**Trade-off to mention:** "Calibrated probabilities matter for second-price auctions. We might use isotonic regression or temperature scaling post-training to improve calibration."

---

## Handling Curveball Questions

**"Design for 10x the scale"** — Discuss sharding (user/item), approximate retrieval, caching, read replicas. Be specific about bottleneck.

**"What if you had 1/10 the data?"** — Simpler models (logistic regression, fewer features), transfer learning, rules, or semi-supervised approaches.

**"Make it real-time"** — Streaming features, online learning vs frequent batch retraining, lower latency serving (maybe lighter model).

**"Add multi-objective optimization"** — Pareto approaches, weighted combo, or contextual bandits. Discuss how to tune weights.

**"How would you improve this in 6 months?"** — Better features, model upgrades, real-time learning, fairness, diversity, cost optimization.

---

## Verbose vs Concise: Finding the Balance

**Too verbose:** Explaining every detail of Spark for 5 minutes when "batch ETL" would suffice.

**Too concise:** "We use a model." (No architecture, no trade-offs.)

**Good balance:** "We'd use batch ETL—Spark or similar—to aggregate daily events into features. The key is consistency with online serving; we'd use a feature store. Any area you'd like me to go deeper?" (Covers the idea, signals depth, invites follow-up.)

---

*Practice this framework until it feels natural. The structure frees you to focus on the actual design.*
