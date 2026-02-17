# Question Bank: 25 ML System Design Questions

This question bank organizes 25 common ML System Design interview questions by category. For each question, use the CLEAR framework and prepare the elements below. Practice answering under time pressure (45 minutes per question).

---

## How to Use This Bank

- **Clarifying questions:** Ask 2–3 before designing; demonstrates thoroughness
- **Architecture summary:** 3-sentence high-level design
- **Deep-dive talking points:** Areas to go deep when interviewer probes
- **Trade-offs:** Be ready to discuss alternatives
- **Follow-ups:** Anticipate what the interviewer might ask next

---

## Category 1: Recommendation & Ranking (8 questions)

---

### 1. Design a recommendation system for an e-commerce platform

**Clarifying questions:**
- What are we recommending? (products, categories, bundles?)
- What's the primary goal—conversion, revenue, or discovery?
- What's the scale? (users, catalog size, QPS?)

**Architecture summary:**
Multi-stage pipeline: candidate generation (collaborative filtering, trending, user history) retrieves 500–1000 items; a light ranker narrows to 100; a heavy ranker (GBDT or DNN) produces final top-20. Feature store provides user/item features; training uses click/ purchase signals with negative sampling. A/B test for online validation.

**Deep-dive talking points:**
- Two-tower model for retrieval vs single model for ranking; when to use each
- Handling cold start for new users (metadata, popularity) and new products (content-based, exploration)
- Session-based vs long-term personalization; feature freshness
- Conversion vs CTR optimization; multi-objective ranking

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Two-tower retrieval | Single-model ranking only | Latency vs recall |
| Batch features | Real-time features | Freshness vs complexity |
| Click-only signal | Add add-to-cart, purchase | Signal quality vs sparsity |

**Follow-ups:** How do you handle seasonality? How do you avoid filter bubbles? How would you add "similar products"?

---

### 2. Design Instagram/TikTok feed ranking

**Clarifying questions:**
- What metrics are we optimizing? (engagement, watch time, retention?)
- What's the content mix—UGC only, or ads too?
- Latency budget for feed load?

**Architecture summary:**
Candidate generation from multiple sources (following feed, explore, trending, saved) merged and deduplicated. Light ranker (LR or small NN) filters to ~200; heavy ranker predicts engagement (likes, shares, watch time) and diversity. Reranker applies creator fairness, recency, and diversity constraints. Real-time features from recent interactions; batch features from embeddings and aggregates.

**Deep-dive talking points:**
- Multi-objective optimization: engagement + diversity + creator exposure
- Integrity signals: reduce reach of misinformation, harmful content
- Position bias and how to correct in training
- Explore vs exploit: injection of new creators/content

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Engagement objective | Watch time | Simpler vs long-term value |
| Single heavy ranker | Multi-stage cascade | Latency vs model capacity |
| Daily retraining | Real-time learning | Freshness vs stability |

**Follow-ups:** How do you detect and reduce filter bubbles? How do you balance new vs established creators?

---

### 3. Design a music recommendation system (Spotify)

**Clarifying questions:**
- Context: radio, playlist, discovery, or "for you"?
- What signals do we have? (skips, saves, playlist adds, listening duration)
- Cold start for new users and new tracks?

**Architecture summary:**
Content-based (audio embeddings, metadata) and collaborative (listen history) candidate sources. Two-tower or matrix factorization for retrieval; ranking model predicts skip probability and listening duration. Sequence models (RNN/Transformer) for playlist continuation. Audio features from upstream pipeline; user listening history in feature store.

**Deep-dive talking points:**
- Audio embeddings (Spectrogram, CNN) vs collaborative filtering
- Sequential recommendation: next-track prediction
- Handling sparse feedback (many skips, few explicit likes)
- Cold start: metadata, popularity, audio similarity for new tracks

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Audio embeddings | Collaborative only | Cold start vs data need |
| Skip prediction | Listen duration | Easier signal vs more valuable |
| Playlist context | User-level only | Richer vs simpler |

**Follow-ups:** How do you handle "radio" vs "discover" differently? How do you balance discovery vs familiarity?

---

### 4. Design a job recommendation system (LinkedIn)

**Clarifying questions:**
- What are we recommending—jobs, people, content?
- Two-sided: job seekers and recruiters—which perspective?
- Latency and freshness requirements?

**Architecture summary:**
Candidate generation from job similarity, member skills, applied history, and recruiter activity. Ranking model predicts apply probability and job fit. Two-sided marketplace: also recommend candidates to recruiters. Skills graph and job taxonomy for semantic matching. Feature store with member profile, job features, and interaction history.

**Deep-dive talking points:**
- Skill and experience matching; ontology and embeddings
- Two-sided optimization: relevance for both seeker and recruiter
- Handling class imbalance (few applies per impression)
- Cold start: new graduates, career changers, new job postings

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Apply prediction | Click prediction | Better signal vs sparser |
| Skill embedding | Rule-based matching | Flexibility vs interpretability |
| Global model | Per-industry models | Simplicity vs specialization |

**Follow-ups:** How do you handle geographic constraints? How do you avoid over-recommending the same jobs?

---

### 5. Design a restaurant recommendation system (Yelp/Google Maps)

**Clarifying questions:**
- Context: search, map view, or "nearby"?
- Signals: ratings, reviews, check-ins, photos?
- Personalization vs general quality?

**Architecture summary:**
Geographic filtering first; then candidate set from category, popularity, and user preferences. Ranking by predicted rating, relevance to query, and diversity (cuisine, price). Feature store with POI attributes, user history, and aggregate ratings. Consider real-time availability and wait times.

**Deep-dive talking points:**
- Geographic constraints and spatial indexing
- Incorporating reviews (sentiment, keywords) into ranking
- Handling sparse and biased ratings (selection bias)
- Context: time of day, party size, occasion

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Collaborative | Content-based | Personalization vs cold start |
| Rating prediction | Engagement prediction | Interpretable vs actionable |
| Static features | Real-time (wait time) | Freshness vs complexity |

**Follow-ups:** How do you surface new restaurants? How do you balance popularity vs quality?

---

### 6. Design a news feed for a social network

**Clarifying questions:**
- Mix of content: posts, ads, stories?
- Primary metric: engagement, time on site, retention?
- Real-time vs batch?

**Architecture summary:**
Candidate generation from social graph (friends, follow), trending, and interest-based. Ranker optimizes engagement with diversity (source, topic, recency). Real-time ingestion for breaking news; batch for historical features. Moderation pipeline for safety. Feature store with user interests, post features, and recency.

**Deep-dive talking points:**
- Recency vs relevance; time decay
- Diversity: avoid echo chambers, balance sources
- Ads integration: native ads in feed ranking
- Breaking news: real-time feature pipelines

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Engagement objective | Relevance (predicted rating) | Easier to optimize vs user intent |
| Real-time ranking | Batch precomputation | Freshness vs latency/cost |
| Single feed | Segmented (following vs discover) | Simplicity vs control |

**Follow-ups:** How do you handle misinformation? How do you balance viral content vs quality?

---

### 7. Design a "Related Products" feature

**Clarifying questions:**
- Placement: PDP, cart, checkout?
- Goal: cross-sell, discovery, or both?
- Catalog size and latency?

**Architecture summary:**
Candidate generation from co-purchase, co-view, same category, and content similarity. Light ranker for business rules (availability, margin); heavy ranker for predicted conversion. Often session-based: "users who viewed this also viewed." Feature store with product attributes and aggregate co-engagement.

**Deep-dive talking points:**
- Co-purchase vs co-view: complementary vs substitute
- Session context: what user is currently looking at
- Catalog coverage: avoid always recommending same items
- Real-time: update as user browses

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Collaborative | Content similarity | Serendipity vs cold start |
| Conversion objective | Revenue per impression | Long-term value vs simplicity |
| Precomputed | Real-time ranking | Latency vs freshness |

**Follow-ups:** How do you avoid recommending out-of-stock items? How do you handle very new products?

---

### 8. Design a personalized homepage

**Clarifying questions:**
- Content types: articles, videos, products?
- New vs returning users?
- Latency for above-the-fold?

**Architecture summary:**
Modular layout (hero, carousels, grid) with per-slot personalization. Candidate generation per slot type; ranker predicts engagement per slot. Layout optimization: which modules to show and in what order. A/B test layout and ranking. Feature store with user interests, history, and content metadata.

**Deep-dive talking points:**
- Layout optimization: contextual bandits or slate optimization
- Above-the-fold vs scroll: different latency budgets
- Modular ranking: different models per content type
- Cold start: default to popular or exploratory

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Fixed layout | Dynamic layout | Simplicity vs personalization |
| Per-slot ranking | Joint slate optimization | Tractability vs quality |
| Engagement | Click-through | Easier vs more nuanced |

**Follow-ups:** How do you balance different content types? How do you avoid redundant recommendations across slots?

---

## Category 2: Ads & Monetization (4 questions)

---

### 9. Design an ad click prediction system

**Clarifying questions:**
- Auction type: first-price, second-price?
- Prediction: pCTR only, or pCVR too?
- Latency budget for the auction?

**Architecture summary:**
Ad retrieval (targeting + ANN) narrows to hundreds of candidates. Feature enrichment from user, ad, and context; pCTR model (DCN, Wide&Deep) with calibration (Platt/isotonic). Auction ranks by eCPM = bid × pCTR; second-price or bid shading. Multi-task learning for pCTR + pCVR. Real-time feature pipeline for recent behavior.

**Deep-dive talking points:**
- Calibration: why raw probabilities aren't enough for auctions
- Multi-task: click and conversion; shared representations
- Feature freshness: last-click, last-view in real time
- Ad fatigue and frequency capping in features

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Calibrated pCTR | Raw scores | Auction fairness vs model simplicity |
| Real-time features | Batch only | Relevance vs latency |
| Multi-task | Separate models | Efficiency vs task conflict |

**Follow-ups:** How do you handle new advertisers? How do you prevent click fraud from affecting the model?

---

### 10. Design a search ads ranking system

**Clarifying questions:**
- Integration with organic results?
- Auction: per-query or session?
- Quality signals beyond CTR?

**Architecture summary:**
Query理解 and ad retrieval by keyword matching and semantic similarity. Ranking by expected relevance and revenue: bid × pCTR × quality score. Quality score from landing page, CTR history, relevance. Feature store with query features, ad features, and historical performance. Latency budget shared with organic ranking.

**Deep-dive talking points:**
- Quality score: prevent low-quality ads from winning
- Query-ad relevance: exact match, phrase, broad
- Position bias: ads in different positions get different CTR
- Session-level optimization vs per-query

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Quality score | Pure bid × pCTR | User experience vs revenue |
| Per-query auction | Session auction | Simplicity vs optimization |
| Separate ranking | Joint with organic | Independence vs coherence |

**Follow-ups:** How do you handle rare queries? How do you balance revenue vs user experience?

---

### 11. Design an ad targeting system

**Clarifying questions:**
- Who defines segments—advertisers or system?
- Signals: demographic, interest, behavior?
- Privacy constraints?

**Architecture summary:**
Audience definition: rule-based (demo, interest) or lookalike (similar to seed). Segment size estimation from user graph and features. Delivery optimization: which users see which campaigns. Feature store with user attributes (aggregated, privacy-safe); lookalike via embeddings or RF. Overlap handling across campaigns.

**Deep-dive talking points:**
- Lookalike modeling: similarity to converters
- Privacy: aggregated segments, differential privacy
- Reach vs relevance: broadening for scale
- Frequency capping and budget pacing

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Lookalike | Rule-based only | Performance vs control |
| Real-time targeting | Batch segments | Freshness vs cost |
| Broad targeting | Narrow | Reach vs relevance |

**Follow-ups:** How do you handle overlap between campaigns? How do you respect privacy regulations?

---

### 12. Design a conversion prediction system

**Clarifying questions:**
- Conversion: purchase, sign-up, install?
- Attribution: last-click, multi-touch?
- Use case: bidding (oCPM) or reporting?

**Architecture summary:**
pCVR model trained on conversions with delayed feedback (purchase can be days later). Attribution: last-click or data-driven (Shapley). Features from user, ad, and journey. Used for oCPM bidding: bid per conversion. Handle conversion delay with delayed feedback modeling (importance weighting, two-step model). Calibration for bidding fairness.

**Deep-dive talking points:**
- Delayed feedback: conversions occur hours/days later
- Attribution: which touchpoints get credit
- Class imbalance: conversions are rare
- Calibration for budget pacing

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Last-click attribution | Multi-touch | Simplicity vs accuracy |
| Separate pCVR | Multi-task with pCTR | Specialization vs efficiency |
| Implicit feedback | Conversion only | More data vs cleaner signal |

**Follow-ups:** How do you handle view-through conversions? How do you prevent fraud in conversion reporting?

---

## Category 3: Search & Discovery (4 questions)

---

### 13. Design a search ranking system

**Clarifying questions:**
- Content type: web, products, documents?
- Metrics: relevance, engagement, success?
- Scale: queries/day, index size?

**Architecture summary:**
Retrieval: inverted index + optionally vector search for semantic. Two-stage: retrieval returns hundreds; ranker (BERT, GBDT) produces final order. Features: query-doc match, document quality, user context. Learning-to-rank loss (e.g., LambdaRank). Feature store for document features; query understanding pipeline.

**Deep-dive talking points:**
- Sparse vs dense retrieval; hybrid approaches
- Learning-to-rank: pointwise, pairwise, listwise
- Query understanding: rewriting, intent, entities
- Zero-result and long-tail query handling

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Dense retrieval | Sparse (BM25) | Semantic vs exact match |
| BERT ranker | GBDT | Quality vs latency |
| Single stage | Multi-stage | Simplicity vs recall |

**Follow-ups:** How do you handle query reformulation? How do you evaluate relevance at scale?

---

### 14. Design autocomplete/typeahead

**Clarifying questions:**
- Scope: query suggestions, entity search, or both?
- Latency per keystroke?
- Personalization?

**Architecture summary:**
Trie or finite-state automaton for prefix matching. Suggestions from query log (popular, trending) and optionally personalized (user history). Rank by frequency, recency, and context. Sub-10 ms latency. Trie in memory; prefix lookup. Cache hot queries. Update trie from query log (hourly or real-time stream).

**Deep-dive talking points:**
- Trie vs other structures; memory vs speed
- Personalized vs global: balance freshness and relevance
- Prefix handling: typos, normalization
- Real-time updates: new trending queries

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Global suggestions | Personalized | Simplicity vs relevance |
| Trie | Elasticsearch/similar | Latency vs flexibility |
| Batch update | Streaming | Consistency vs freshness |

**Follow-ups:** How do you handle malicious or abusive queries? How do you surface new, trending queries quickly?

---

### 15. Design an image search system

**Clarifying questions:**
- Query: text, image, or both?
- Catalog size?
- Use case: e-commerce, general web, internal?

**Architecture summary:**
Text query: dual-encoder (text → embedding, image → embedding); ANN search in image embedding space. Image query: CNN embeddings; nearest neighbors. Hybrid: combine text and image embeddings. Index billions of images with HNSW or IVF. Two-stage: retrieval then reranking. Feature store for image metadata.

**Deep-dive talking points:**
- CLIP-style dual encoders for text-to-image
- Image embeddings: ResNet, ViT, or domain-specific
- Scalability: indexing billions of vectors
- Reranking: cross-attention for query-doc matching

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Dual-encoder | Single model | Flexibility vs training complexity |
| Vector only | Hybrid with metadata | Semantic vs structured |
| Batch indexing | Real-time | Freshness vs cost |

**Follow-ups:** How do you handle multimodal (text + image) queries? How do you reduce bias in image results?

---

### 16. Design a question answering system

**Clarifying questions:**
- Domain: open-domain or closed (e.g., support docs)?
- Source: retrieval-augmented or model-only?
- Latency and accuracy expectations?

**Architecture summary:**
Retrieval: embed question, search document/chunk index (ANN). Reader: extract or generate answer from retrieved chunks. RAG: retrieve → concatenate → LLM generate. For closed domain: smaller model or fine-tuned. Evaluation: exact match, F1, human judgment. Pipeline: retriever → reranker → generator.

**Deep-dive talking points:**
- Retriever: sparse (BM25) vs dense (embeddings) vs hybrid
- Chunking: size, overlap, semantic boundaries
- Generator: extractive vs generative
- Evaluation and hallucination mitigation

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| RAG | Fine-tuned LLM | Factuality vs flexibility |
| Dense retriever | Sparse | Semantic vs exact match |
| Extractive | Generative | Precision vs naturalness |

**Follow-ups:** How do you handle "no answer" cases? How do you keep the knowledge base up to date?

---

## Category 4: Trust & Safety / Integrity (4 questions)

---

### 17. Design a content moderation system

**Clarifying questions:**
- Content types: text, image, video?
- Scale: uploads per day?
- Human-in-the-loop: when?

**Architecture summary:**
Multi-modal pipeline: text classifier (BERT), image classifier (CNN), video (frame sampling + classifier). First-pass ML filter; high-confidence auto-action; medium confidence to human review. Feedback loop: human labels → retrain. Feature store for content and user history. Queue for review with prioritization (virality, severity).

**Deep-dive talking points:**
- Multi-modal: combining signals
- Calibration: when to escalate to humans
- Adversarial robustness: evolved abuse
- Fairness: error rates across demographics

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Auto-remove at threshold | Human review always | Scale vs accuracy |
| Single model | Per-category models | Efficiency vs specialization |
| Reactive | Proactive (before viral) | Simplicity vs impact |

**Follow-ups:** How do you handle novel abuse? How do you minimize false positives?

---

### 18. Design a spam detection system

**Clarifying questions:**
- Spam type: email, comments, messages?
- Real-time or batch?
- False positive tolerance?

**Architecture summary:**
Feature extraction: content (text, links), sender/receiver graph, behavior. Classifier (LR, GBDT, or NN) predicts spam score. Real-time: sub-100 ms for messages; batch for email. Feedback: user reports, manual review. Handle adversarial evolution with periodic retraining and ensemble. Graph features for coordinated inauthentic behavior.

**Deep-dive talking points:**
- Content features: URLs, patterns, language
- Graph features: sender reputation, recipient patterns
- Adversarial: attackers adapt to rules
- Class imbalance: spam is rare in some channels

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Rule-based | ML only | Interpretability vs adaptability |
| Real-time | Batch | User experience vs throughput |
| High precision | High recall | User friction vs spam leakage |

**Follow-ups:** How do you handle legitimate bulk senders? How do you detect coordinated campaigns?

---

### 19. Design a fake account detection system

**Clarifying questions:**
- Platform: social, marketplace, financial?
- Signals: registration, behavior, graph?
- Action: block, limit, or flag?

**Architecture summary:**
Registration signals: device, IP, email/phone patterns. Behavioral: activity velocity, content, engagement patterns. Graph: clustering, sybil detection, bot networks. ML model combines signals; rule overlay for known patterns. Feedback from manual review and appeals. Handle concept drift as attackers adapt.

**Deep-dive talking points:**
- Sybil detection: graph structure of fake clusters
- Registration fraud: stolen identities, disposable emails
- Behavioral signals: automation, unnatural patterns
- Cold start: new account has little history

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Strict at signup | Monitor over time | Early prevention vs false positives |
| Graph-based | Content-only | Network detection vs simplicity |
| Auto-block | Manual review | Scale vs accuracy |

**Follow-ups:** How do you handle sophisticated bots? How do you minimize false positives for real users?

---

### 20. Design a misinformation detection system

**Clarifying questions:**
- Content: text, image, video?
- Definition: fact-check, virality risk, or both?
- Integration: label, downrank, or remove?

**Architecture summary:**
Claim extraction and matching to fact-check database. Virality prediction: will this go viral? Credibility scoring for sources. Pipeline: ingest → extract claim → fact-check lookup → credibility → score. Optional: image/video verification (deepfake detection). Downrank or label in feed; avoid hard removal for borderline cases.

**Deep-dive talking points:**
- Fact-checking: claim extraction, database matching
- Virality prediction: stem spread before fact-check
- Source credibility: historical accuracy
- Multimodal: deepfakes, out-of-context images

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Fact-check integration | Model-only | Accuracy vs coverage |
| Downrank | Remove | Free speech vs harm reduction |
| Reactive | Proactive (before viral) | Feasibility vs impact |

**Follow-ups:** How do you handle evolving narratives? How do you avoid censorship concerns?

---

## Category 5: Social & Graph (3 questions)

---

### 21. Design People You May Know

**Clarifying questions:**
- Goal: connection growth, engagement?
- Signals: profile, graph, behavior?
- Diversity and safety constraints?

**Architecture summary:**
Candidate generation: friends-of-friends, same school/company, mutual connections, profile similarity. Ranker predicts connection probability and value. Negative filtering: already connected, blocked, inappropriate. Graph features from social graph; embeddings for similarity. Feature store with user profile and graph aggregates. Diversity: don't over-recommend from one cluster.

**Deep-dive talking points:**
- Friends-of-friends: scalability at billions of edges
- Graph embeddings: Node2Vec, GNN for scoring
- Diversity: avoid recommending 10 people from same company
- Safety: block lists, spam/abuse signals

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| FoF only | Add content similarity | Graph structure vs richness |
| Single ranker | Multi-objective | Simplicity vs balance |
| Batch computation | Real-time | Freshness vs cost |

**Follow-ups:** How do you handle cold start (new users)? How do you avoid recommending strangers inappropriately?

---

### 22. Design a social influence/trending detection system

**Clarifying questions:**
- Trend type: hashtags, topics, memes?
- Time scale: real-time or daily?
- Use case: explore tab, trending topics list?

**Architecture summary:**
Event stream (posts, shares, likes) aggregated by entity (hashtag, topic). Velocity and acceleration detection; novelty (burst vs sustained). Anomaly detection for emerging trends. Ranking by engagement velocity, diversity, and safety. Graph: propagation paths. Near real-time with streaming (Kafka, Flink); batch for historical.

**Deep-dive talking points:**
- Velocity vs volume: sudden spike vs gradual growth
- Novelty: new vs recurring trends
- Geo and demographic breakdown
- Spam and manipulation resistance

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Real-time | Batch (hourly) | Freshness vs cost |
| Velocity-based | Volume-based | Early detection vs stability |
| Global trending | Personalized | Simplicity vs relevance |

**Follow-ups:** How do you prevent gaming? How do you surface niche vs mass trends?

---

### 23. Design a community recommendation system

**Clarifying questions:**
- Community type: subreddits, groups, forums?
- Goal: growth, engagement, retention?
- Cold start for new communities?

**Architecture summary:**
Candidate generation: similar communities (content, member overlap), user interest match, trending. Ranker predicts join probability and engagement. Features: community description, members, activity, topic embeddings. Graph: overlap, co-membership. Cold start: metadata, invite from similar communities. Diversity: topic, size.

**Deep-dive talking points:**
- Community embeddings: content + member overlap
- Cold start: new communities with few members
- Balance: not only largest communities
- Moderation and safety signals

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Content similarity | Member overlap | Cold start vs accuracy |
| Join prediction | Engagement prediction | Acquisition vs retention |
| Global ranking | Personalized | Simplicity vs relevance |

**Follow-ups:** How do you surface small but relevant communities? How do you handle community lifecycle (dying vs growing)?

---

## Category 6: Infrastructure & Platform (2 questions)

---

### 24. Design a feature store

**Clarifying questions:**
- Users: how many teams, use cases?
- Latency: online serving vs offline training?
- Feature types: batch, streaming, or both?

**Architecture summary:**
Dual storage: online store (Redis, DynamoDB) for low-latency serving; offline store (Parquet, data lake) for training. Unified schema and access API. Ingestion: batch (Spark) and streaming (Flink, Kafka). Point-in-time correctness for training. Discovery and lineage. Support both entity-centric (user features) and event-centric features.

**Deep-dive talking points:**
- Online vs offline: consistency, freshness, schema
- Point-in-time correctness: avoid data leakage
- Backfill: recomputing historical features
- Governance: access control, PII, lineage

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Dual store | Single store | Latency vs simplicity |
| Batch + streaming | Batch only | Freshness vs complexity |
| Centralized | Per-team | Consistency vs autonomy |

**Follow-ups:** How do you handle feature versioning? How do you manage backfill at scale?

---

### 25. Design an ML experimentation platform

**Clarifying questions:**
- Users: data scientists, engineers, or both?
- Scope: training, deployment, or full lifecycle?
- Integration with existing infra?

**Architecture summary:**
Experiment tracking: params, metrics, artifacts (MLflow, custom). Orchestration: job scheduling, dependencies, resource management. Model registry: versions, stages, approval. A/B testing: traffic split, metric aggregation. Reproducibility: code, data, env versioning. Integrated with feature store and model serving.

**Deep-dive talking points:**
- Reproducibility: code, data, environment
- A/B testing: statistical rigor, multiple comparisons
- Model registry: promotion, rollback
- Resource management: GPU allocation, quotas

**Trade-offs:**
| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Integrated platform | Best-of-breed tools | Consistency vs flexibility |
| Heavy governance | Lightweight | Compliance vs velocity |
| Centralized compute | Distributed | Cost vs complexity |

**Follow-ups:** How do you handle experiment explosion? How do you do offline evaluation before A/B test?

---

*Practice 2–3 questions per category. Focus on questions most likely at your target companies (e.g., Meta: ads, feed ranking; Google: search, YouTube; Amazon: recommendations).*
