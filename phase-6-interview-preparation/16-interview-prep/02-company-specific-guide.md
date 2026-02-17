# Company-Specific Interview Guide

Different companies emphasize different aspects of ML system design. This guide covers Meta and Google in detail, with patterns you can adapt for other large tech companies.

---

## Meta ML System Design Interviews

### Interview Format

- **Duration**: 45 minutes
- **Format**: One interviewer, whiteboard or virtual whiteboard (e.g., CoderPad, Miro)
- **Structure**: Open-ended prompt; you drive the conversation; interviewer asks follow-ups to probe depth
- **Evaluation**: Product sense, technical design, scale, communication

### What Meta Emphasizes

1. **Product thinking** — Connect the system to user value (engagement, discovery, safety)
2. **Real-time systems** — Many products need low latency and fresh data
3. **Scale** — 3B+ users; design for billions, not millions
4. **Integrity** — Safety, moderation, and policy alignment matter
5. **Social graph** — Connections, engagement, and viral mechanics are central

### Common Meta Questions

| Question | What They Probe |
|----------|-----------------|
| **1. Design News Feed ranking** | Real-time ranking, multi-objective optimization, integrity |
| **2. Design Instagram Reels recommendations** | Video content, engagement signals, cold start, discovery |
| **3. Design ad click prediction** | CTR prediction, auction integration, calibration |
| **4. Design content moderation** | Classification, scale, human-in-the-loop, policy |
| **5. Design People You May Know** | Graph algorithms, ranking, privacy, growth |
| **6. Design notification ranking** | Multi-channel, urgency, user preferences, fatigue |
| **7. Design integrity/harmful content detection** | Real-time detection, escalation, feedback loops |

### Meta: Deep Dive on News Feed Ranking

For "Design News Feed ranking," expect to cover:

- **Multi-objective optimization**: Engagement (likes, comments, shares) vs diversity vs integrity (downranking harmful content)
- **Real-time signals**: Recent interactions, session context, time of day
- **Scale**: Billions of posts, milliseconds latency
- **Integrity**: How to demote or remove violating content; human review pipeline
- **Personalization vs diversity**: Avoid filter bubbles; surface new connections and content types

### Meta: Deep Dive on People You May Know (PYMK)

- **Graph-based**: Friend-of-friend, mutual connections, engagement on same content
- **Ranking**: Beyond connectivity—engagement likelihood, relevance
- **Privacy**: What data can we use? Cross-app considerations
- **Growth levers**: Optimizing for new connections vs engagement
- **Cold start**: New users with no graph—use demographics, interests, device signals

### Meta-Specific Tech Context

Knowing their stack helps you use relevant terminology and avoid mismatches:

| Component | Meta Tech |
|-----------|-----------|
| ML framework | PyTorch |
| Feature store | Internal feature store (similar to Feathr/F feast) |
| Social graph | TAO (Their Own Graph) |
| Vector search | FAISS (Facebook AI Similarity Search) |
| Training | FBLearner, internal pipelines |
| Serving | Real-time inference, low-latency requirements |
| Scale | Billions of users, trillions of events |

### Meta Levels and Expectations

| Level | Expectation |
|-------|-------------|
| **E4 (IC4)** | Solid design with some guidance; can deep-dive on one area; follows the framework |
| **E5 (IC5)** | Independent end-to-end design; proactively discusses trade-offs; handles follow-ups well; shows product sense |
| **E6 (IC6)** | Anticipates edge cases; drives discussion; thinks across systems and company impact |

---

## Google ML System Design Interviews

### Interview Format

- **Duration**: 45 minutes
- **Format**: May have separate ML-focused and system-focused interviewers across the loop
- **Structure**: Open-ended prompt; more emphasis on infrastructure and scalability
- **Evaluation**: Algorithmic depth, infrastructure thinking, research awareness

### What Google Emphasizes

1. **Algorithmic depth** — Understanding of models, metrics, and optimization
2. **Infrastructure thinking** — Data pipelines, storage, serving at scale
3. **TFX familiarity** — Training, validation, deployment pipelines
4. **Research awareness** — Knowledge of recent papers and techniques

### Common Google Questions

| Question | What They Probe |
|----------|-----------------|
| **1. Design YouTube recommendations** | Large-scale retrieval, engagement, diversity, cold start |
| **2. Design Google Search ranking** | Query understanding, relevance, freshness, scale |
| **3. Design autocomplete/query suggestions** | Prefix matching, personalization, latency |
| **4. Design spam/abuse detection for Gmail** | Classification at scale, false positives, user feedback |
| **5. Design Google Maps ETA prediction** | Time-series, traffic, routing, real-time updates |
| **6. Design image search** | Embeddings, similarity search, multi-modal |
| **7. Design ad click prediction for Search Ads** | CTR, calibration, auction, latency |

### Google: Deep Dive on YouTube Recommendations

- **Retrieval at scale**: Millions of videos; need candidate generation (ANN) before ranking
- **Engagement optimization**: Watch time, completion rate, next-video clicks
- **Diversity**: Avoid repetition; surface different creators and topics
- **Cold start**: New videos—use metadata, early engagement; new users—trending, categories
- **Two-tower + ranker**: Retrieval with embeddings; lightweight ranker for final ordering

### Google: Deep Dive on Search Ranking

- **Query understanding**: Intent, entities, spelling
- **Relevance signals**: BM25, embeddings, neural ranker
- **Freshness**: News, real-time events—recency in ranking
- **Infrastructure**: Distributed serving, caching, low latency
- **Evaluation**: Human relevance judgments, A/B testing, offline metrics

### Google-Specific Tech Context

| Component | Google Tech |
|-----------|-------------|
| ML framework | TensorFlow, JAX |
| Training infra | TPUs, distributed training |
| ML pipelines | TFX (TensorFlow Extended) |
| Storage | Bigtable, Spanner |
| Batch processing | MapReduce, Flume |
| Serving | Low-latency inference, batching |
| Scale | Billions of queries, petabytes of data |

### Google Levels and Expectations

| Level | Expectation |
|-------|-------------|
| **L4** | Solid fundamentals; reasonable design; can explain trade-offs; may need hints |
| **L5** | Independent design; deep understanding; proactive about reliability and scale |
| **L6** | Cross-system thinking; organizational impact; novel approaches; mentors others |

---

## Common Patterns Across Both Companies

### Both Want

- **Clarification first** — Don’t design in a vacuum; ask questions
- **Structured approach** — Clear phases (clarify → design → refine)
- **Trade-offs** — Alternatives and justification for choices
- **End-to-end thinking** — Data → training → serving → monitoring
- **Scale awareness** — Reasonable estimates and scaling strategies

### Both Care About

- **Data quality** — Garbage in, garbage out
- **Monitoring** — How you detect and respond to issues
- **Fairness** — Bias, representation, safety
- **Reliability** — Fallbacks, degradation, incident response

### Key Differences (Summary Table)

| Dimension | Meta | Google |
|-----------|------|--------|
| **Product focus** | Social, engagement, virality | Search, discovery, infrastructure |
| **Framework** | PyTorch | TensorFlow, JAX |
| **Feature store** | Internal | TFX, internal |
| **Scale emphasis** | Users, social graph | Queries, data volume |
| **Integrity/safety** | Very prominent | Important for many products |
| **Interview style** | Product + systems | Infrastructure + algorithms |

---

## Preparation by Company

### If Interviewing at Meta

1. **Practice these prompts**: News Feed, Reels, PYMK, ad CTR, content moderation
2. **Know**: Real-time ranking, social graph, multi-objective optimization
3. **Mention**: PyTorch, FAISS, feature store, TAO (if relevant)
4. **Discuss**: Integrity, diversity, user experience

### If Interviewing at Google

1. **Practice these prompts**: YouTube, Search ranking, autocomplete, Maps ETA, image search
2. **Know**: TFX, large-scale retrieval, embedding-based search
3. **Mention**: TensorFlow, TPUs, Bigtable (if relevant)
4. **Discuss**: Infrastructure, data pipelines, research-driven improvements

---

## Adapting for Other Companies

### Amazon

- **Focus**: E-commerce, personalization, logistics
- **Common prompts**: Product recommendations, search ranking, demand forecasting, fraud detection
- **Tech**: SageMaker, internal tools, scale for retail

### Netflix

- **Focus**: Content recommendation, diversity, engagement
- **Common prompts**: Homepage ranking, similarity, A/B testing
- **Tech**: Personalization at scale, experimentation

### Uber/Lyft

- **Focus**: ETA, pricing, matching, maps
- **Common prompts**: ETA prediction, surge pricing, driver-rider matching
- **Tech**: Real-time, geospatial, time-series

### LinkedIn

- **Focus**: Professional graph, job recommendations, feed
- **Common prompts**: Feed ranking, People You May Know, job recommendations
- **Tech**: Graph algorithms, professional context

---

## Interview Day: What to Bring

- **Whiteboard markers** (if in-person): Test them; have backups
- **Water**: Stay hydrated; pause briefly if you need to think
- **Notepad**: Jot down key numbers from clarification (scale, latency)
- **Mental checklist**: CLEAR phases; 15 mistakes to avoid

---

## Sample Meta vs Google: Same Prompt, Different Emphasis

**Prompt:** "Design a recommendation system."

**Meta-style follow-ups:**
- "How does this optimize for engagement?"
- "What about integrity—how do you handle harmful content?"
- "How would you ensure diversity in the feed?"
- "What's the cold start strategy for new users in a social context?"

**Google-style follow-ups:**
- "Walk me through the retrieval architecture at scale"
- "How does your feature store integrate with the training pipeline?"
- "What's your strategy for approximate nearest neighbor search?"
- "How would you handle 10x growth in the catalog?"

Prepare for both types. Tailor your initial design to signal you understand their focus.

---

## Company-Specific Vocabulary

Using the right terms can signal familiarity:

| Company | Say |
|---------|-----|
| Meta | Engagement, integrity, feed, PYMK, TAO, FAISS, PyTorch |
| Google | TFX, retrieval, TPU, Bigtable, relevance, query understanding |
| Amazon | Personalization, conversion, SageMaker, catalog |
| Netflix | Discovery, diversity, experimentation, homepage |

Don't force it. Use naturally when it fits the design.

---

## Question-Specific Preparation Matrix

| If you get... | Emphasize |
|---------------|-----------|
| News Feed / Feed ranking | Multi-objective, real-time, integrity, diversity |
| Recommendations (YouTube, Netflix, products) | Retrieval + ranking, cold start, engagement metrics |
| Search ranking | Query understanding, relevance, freshness, scale |
| Ad CTR prediction | Calibration, auction integration, feature freshness |
| Content moderation | Classification at scale, human-in-loop, policy, escalation |
| PYMK / Social graph | Graph algorithms, ranking, privacy |
| ETA / Time-series | Traffic, routing, real-time updates, uncertainty |
| Autocomplete | Prefix matching, personalization, latency |

---

## Interview Loop Structure (Typical)

A full ML interview loop often includes:

- **1–2 ML System Design** (45 min each): This guide focuses here
- **1 Coding**: Algorithm/data structure problems; sometimes ML-adjacent (e.g., implement AUC)
- **1 Behavioral**: Leadership principles, past projects, conflict resolution
- **Possibly**: Research/technical deep dive (for research-focused roles)

The system design interview is usually the highest-weight technical interview for ML roles.

---

## What to Research Before Your Interview

- **Company blog**: Meta AI, Google AI, Netflix Tech Blog—recent ML posts
- **Papers**: Search "[Company] [problem] paper" (e.g., "Meta news feed ranking")
- **Tech stack**: PyTorch vs TensorFlow, internal tools, open-source contributions
- **Recent product launches**: New features often show up in interviews

---

*Tailor your examples and terminology to the company. A Meta interviewer cares more about engagement loops; a Google interviewer may care more about retrieval architecture.*
