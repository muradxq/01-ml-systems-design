# Mock Interview Walkthroughs

This document contains **3 full 45-minute mock interview transcripts** demonstrating exemplary performance. Study how the candidate structures their response, asks clarifying questions, designs the system, and handles follow-ups. [Interviewer notes] in brackets show what impresses evaluators.

---

## How to Use These Walkthroughs

1. **Read each walkthrough once** to absorb the flow—clarification → requirements → architecture → deep dive → trade-offs → follow-ups.
2. **Note the timing**—each section has an approximate duration. In a real interview, you have ~45 minutes total.
3. **Study the [Interviewer notes]**—they reveal what evaluators care about: structure, depth, trade-offs, failure modes.
4. **Practice aloud**—pick one prompt, set a timer, and run through it. Compare your performance to the transcript.
5. **Vary the prompts**—use the same structure for different questions from the question bank.

---

## Walkthrough 1: Design Instagram Feed Ranking (Meta-style)

*Approximate timing: Clarify 2 min | Requirements + Estimate 3 min | Architecture 8 min | Deep dive 10 min | Features 5 min | Integrity 5 min | Trade-offs 7 min | Follow-ups 5 min = ~45 min*

**Interviewer:** *"I'd like you to design the feed ranking system for Instagram—you know, the main feed when users open the app. Take your time, ask questions, and walk me through your approach."*

**Candidate:** *"Thanks! I'll use a structured approach: clarify the problem and constraints, list requirements and do scale estimates, then design the end-to-end architecture and go deeper on key components. I'll finish with trade-offs and failure modes.*

*To start—a few clarifying questions. First, what are we optimizing for? Is it engagement—likes, comments, shares—or something like time spent, or a combination?"*

**I:** *"Good question. Primarily engagement, but we care about long-term satisfaction too—we don't want to optimize for clickbait."*

**C:** *"Got it. Second: what's the scale we're designing for? DAU, orders of magnitude for QPS?"*

**I:** *"Assume Instagram scale—hundreds of millions of DAU, and the feed loads many times per user per day."*

**C:** *"And latency—is there a specific budget? Users are scrolling, so I'm guessing sub-200ms?"*

**I:** *"Yes, we need p99 under 200ms for the initial feed load."*

**C:** *"One more: does the feed include ads, or are we focusing on organic content only?"*

**I:** *"Include ads—they're part of the feed and need to be ranked alongside organic content."*

**C:** *"Perfect. And any diversity or integrity constraints? Like, we don't want 10 posts in a row from the same person, or content that violates policy?"*

**I:** *"Yes to both. We want diversity, and we have integrity signals that should down-rank problematic content."*

*[Interviewer notes: Strong clarification—covered business goal, scale, latency, scope (ads), and constraints. Didn't jump into design. Five focused questions in ~2 min.]*

**C:** *"Great, I have enough context. Let me summarize requirements and then estimate scale."*

---

### Requirements & Scale Estimation (3 min)

**C:** *"I'll jot these down.*

*Functional: FR1—Generate personalized ranking of 20–30 posts per request. FR2—Support cold start for new users and new creators. FR3—Apply diversity constraints—don't cluster by source. FR4—Incorporate integrity signals to down-rank policy-violating content. FR5—Log impressions and engagement for feedback and retraining.*

*Non-functional: NFR1—Latency p99 < 200ms. NFR2—High availability—this is core product. NFR3—Freshness: new posts discoverable within seconds; model retraining daily or more.*

*For scale—assuming 500M DAU, 10 feed opens per user per day, that's 5B feed requests per day. 5B / 86,400 ≈ 60K QPS average. With a 3× peak factor, peak QPS ≈ 180K. Each request might need to rank hundreds of candidates, so we're looking at millions of ranking inferences per second—that informs our architecture. Feature store: 500M users × 200 features × 4 bytes ≈ 400 GB for user features alone. We'd need to support 180K × 50 features = 9M feature reads per second before caching."*

*[Interviewer notes: Clear structure. Estimation shows reasoning. Mentions feature store scale—shows systems thinking.]*

---

### High-Level Architecture (8 min)

**C:** *"Let me draw the flow."*

```
[User Request] → [Candidate Selection: following + explore + ads]
       → [Merge: ~500-2000 candidates]
       → [Light Ranker: 500→100]
       → [Heavy Ranker: 100→50]
       → [Integrity Filter + Diversity]
       → [Feed Composition]
       → [Response + Logging]
```

**C:** *"Candidate selection pulls from multiple sources in parallel: posts from people you follow, explore/recommended content, ads. We merge and deduplicate to maybe 500–2000 candidates.*

*Then a light ranker—logistic regression or small NN—narrows to ~100. This keeps latency low while pruning obviously irrelevant content. The light ranker uses a subset of features, maybe 50, and runs in 10–20ms.*

*The heavy ranker is where we invest compute. It's a larger model—DNN, maybe multi-task—that predicts engagement. We rank down to ~50. This might take 50–80ms.*

*After that, we apply integrity filters: content that scores high on misinformation, hate speech, clickbait gets down-ranked or removed. Then a diversity pass—MMR or category spread—so we don't have 10 posts from the same creator in a row.*

*Finally, feed composition: we slot in ads at certain positions, apply frequency capping per user and per advertiser, and return the ordered list. We log impressions and engagement asynchronously for retraining. The logging pipeline writes to Kafka or similar, and a batch job consumes for training."*

**I:** *"Can you go deeper into the multi-objective ranking model? How do you combine engagement, quality, and integrity?"*

*[Interviewer notes: Good probe—wants to see depth on the core ML component.]*

---

### Deep Dive: Multi-Objective Ranking (10 min)

**C:** *"The heavy ranker predicts several things: P(like), P(comment), P(share), P(dwell > 15s), and optionally P(click)—we want meaningful engagement, not just clicks. We also predict negative signals: P(misinformation), P(clickbait), P(low_quality).*

*There are a few ways to combine these. One is weighted sum: we learn weights for each head and combine into a single score. The weights might come from a separate optimization—what combination correlates with long-term retention?*

*Another is Multi-gate Mixture-of-Experts (MMoE) or similar—separate expert networks per objective, with a shared gating mechanism. That way we can model task relationships and avoid negative transfer.*

*For the final score, we'd do something like: value = w1×P(like) + w2×P(comment) + w3×P(share) + w4×P(dwell) - w5×P(misinfo) - w6×P(clickbait). The negative terms ensure we down-rank problematic content. The weights could be tuned via offline metrics or online A/B tests.*

*We train with multi-task loss—each head has a loss; the total is a weighted sum. We need to handle label sparsity—likes are rarer than impressions—so we might use different sampling or loss weighting."*

**I:** *"How do you avoid position bias when training?"*

**C:** *"Position bias is a big deal. Items higher in the feed get more engagement partly because they're seen first, not because they're better. We need to correct for that.*

*Options: (1) Use position as a feature and let the model learn it—but then at inference we need to account for it. (2) Inverse propensity weighting—weight losses by 1/P(seen | position). (3) Click models that explicitly model P(click | seen) × P(seen | position).*

*In practice, we'd add position as a feature during training. At serving, we either use a fixed position (e.g., assume position 1) or do a pass where we iteratively predict scores and update assumed positions. Or we use a simpler correction like inverse propensity.*

*The key is we don't want the model to learn 'stuff at the top gets more clicks' as a signal of quality."*

*[Interviewer notes: Strong ML depth. Multi-task, MMoE, position bias correction. Candidate knows the literature.]*

**I:** *"What features would you use?"*

**C:** *"I'd group them into user, content, author, and interaction features.*

*User: demographics, past engagement patterns, interests inferred from behavior, session context—time of day, device, connectivity.*

*Content: post type (photo, video, carousel), media metadata, text embeddings if we have caption, recency, virality (likes, comments in last N hours).*

*Author: follower count, verified status, historical engagement rate, relationship to user—close friend vs casual follow.*

*Interaction: user-author affinity—have they liked this creator's posts before? Similar content engagement. Social proof—what have friends engaged with?*

*For real-time: 'has user seen this post before?'—we don't want to reshow. 'Time since user last engaged with this creator'—recency.*

*We'd store batch features—user embeddings, author stats—in a feature store. Real-time features—current session, recent impressions—from a fast store or computed on the fly. Critical that training and serving use the same feature definitions and logic."*

**I:** *"How does the integrity system work?"*

**C:** *"Integrity is multi-layered. First, we have classifiers that run on upload—text, image, video—that flag policy violations. Those scores can be stored with the content.*

*At ranking time, we fetch those scores and use them in the ranker—either as features that the model learns to down-rank, or as a separate filter. High P(misinfo) or P(hate_speech) gets demoted or removed.*

*We might also have a lightweight real-time check—e.g., if a post went viral in the last hour, we re-score it for integrity before boosting it. That catches things that slip through at upload.*

*For quality—clickbait, low-quality—we can train the ranker to predict those and down-rank. Or have a separate quality model. The idea is we don't want engagement at the cost of user trust."*

---

### Trade-offs, Monitoring, Fairness (7 min)

**I:** *"What trade-offs would you call out?"*

**C:** *"A few. First, engagement vs diversity. If we purely optimize engagement, we might create filter bubbles—same topics, same creators. The diversity injector addresses that, but it's a balance. We might sacrifice some short-term engagement for diversity that improves long-term satisfaction.*

*Second, model complexity vs latency. A heavier model might predict better but blow our 200ms budget. That's why we have a light ranker first—we only run the heavy model on 100 candidates.*

*Third, exploration vs exploitation. New creators need exposure. We might inject a fraction of recommendations from newer or smaller creators even if their predicted engagement is lower. That's an explore/exploit trade-off.*

*For monitoring: we'd track input drift—feature distributions—and output drift—score distributions. Model performance—engagement metrics by cohort. And integrity metrics—what fraction of served content gets reported or down-ranked later. We'd alert on significant shifts.*

*Fairness: we want to avoid systematically under-exposing certain creator segments—e.g., creators from particular regions or languages. We could segment metrics by creator attributes and ensure we're not under-serving unfairly."*

**I:** *"What if the heavy ranker goes down?"*

**C:** *"We'd need a fallback. Options: serve from the light ranker output only—maybe top 20 by that score. Or a cached 'emergency' ranking—e.g., recency-based or popularity-based. We'd have a circuit breaker so we fail fast and serve something reasonable rather than timing out or erroring. We'd also run multiple replicas across regions for redundancy."*

*[Interviewer notes: Good failure mode thinking. Fallback, circuit breaker, multi-region.]*

---

### Follow-ups (5 min)

**I:** *"How would you handle a new user with no history?"*

**C:** *"Cold start. We'd have limited signals—device, location, maybe interests from onboarding. We could default to a mix: trending in their region, popular accounts in categories they expressed interest in, and some exploration. As they engage, we'd update their embedding and preferences. We might have a separate 'cold start' model or a default path in the main pipeline that uses fewer personalization features."*

**I:** *"How do you balance ads with organic content?"*

**C:** *"Ads are part of the candidate set. We'd rank them jointly—ads get a score from the ad system, organic from our ranker. We blend them based on business rules: ad density (e.g., 1 ad per 5 organic), frequency capping per advertiser, and relevance. We might use a slate optimization formulation—which positions for ads maximize overall value—but in practice it's often simpler rules with A/B-tuned parameters."*

**I:** *"That's time. Good work. Anything you'd do differently?"*

**C:** *"A few things. I'd strengthen the real-time learning angle—incorporating engagement from the current session before the next request, if feasible. I'd also dig deeper into the diversity formulation—MMR vs learned diversity—and do more on offline evaluation setup before A/B testing. But I'm happy with the overall structure."*

*[Interviewer notes: Strong close. Acknowledges limitations without over-apologizing. Would hire.]*

---

## Walkthrough 2: Design YouTube Recommendations (Google-style)

*Approximate timing: Clarify 2 min | Architecture 10 min | Watch time deep dive 10 min | Features & training 5 min | Trade-offs 5 min | Follow-ups 3 min*

**I:** *"Design the video recommendation system for YouTube—the recommendations you see on the home page and in the 'Up Next' sidebar."*

**C:** *"Thanks. I'll clarify first, then design.*

*First: are we designing both home page and Up Next, or one? They have different contexts—Up Next has the currently playing video as a strong signal."*

**I:** *"Focus on the home page. Up Next can be similar in spirit."*

**C:** *"Second: what's the primary metric—watch time, clicks, or something else?"*

**I:** *"Watch time is the main one. We want users to stay and watch."*

**C:** *"Scale?"*

**I:** *"Assume YouTube scale—billions of users, billions of videos."*

**C:** *"And latency?"*

**I:** *"Sub-200ms for the initial load."*

**C:** *"Any constraints on diversity—avoiding filter bubbles, surfacing new creators?"*

**I:** *"Yes, we care about both."*

*[Interviewer notes: Concise clarification. Covers product scope, metric, scale, latency, constraints. Gets to design quickly.]*

---

### Architecture (10 min)

**C:** *"I'll outline a multi-stage pipeline and sketch the flow."*

```
[User Opens Home] → [Candidate Gen: collaborative | subscriptions | trending | topic | exploration]
       → [Merge & dedupe: ~1000-2500 candidates]
       → [Light Ranker: 1000→200, ~10-20ms]
       → [Heavy Ranker: 200→50, ~50ms, EWT model]
       → [Rerank: diversity, creator fairness, ads]
       → [Home Page Response]
       → [Log impressions, watch time]
```

**C:** *"Stage 1—Candidate generation. We pull from multiple sources in parallel: (1) Collaborative—videos similar to what the user watched, via two-tower or ANN; (2) Subscriptions—new uploads from subscribed channels; (3) Trending—regional and global; (4) Topic/interest—based on watch history; (5) Exploration—new creators, diverse topics. Each source returns hundreds of candidates. We merge and deduplicate to ~1000–2500 candidates.*

*Stage 2—Light ranker. Fast model—LR or small NN—on a subset of features. Reduces 1000 → 200. Latency budget maybe 10–20ms. This prunes the long tail so we don't run the heavy model on everything.*

*Stage 3—Heavy ranker. This is where we predict expected watch time (EWT). DNN—Deep & Cross, DCN, or similar. We rank 200 → 50. Latency ~50ms.*

*Stage 4—Reranking. We apply business logic: diversity by topic and creator, avoid too many from one channel, recency, and ad slot insertion. Final ordered list.*

*Data flow: user and video embeddings from a feature store; real-time features—current session, last watched—from a fast store. Training pipeline consumes watch logs—we predict watch time from impressions, with appropriate negative sampling."*

**I:** *"Deep dive on the watch time prediction model."*

---

### Deep Dive: Watch Time Model (10 min)

**C:** *"The target is expected watch time—how long will the user watch this video? Not just click. That matters because a 10-minute video that gets 30 seconds of watch is different from a 2-minute video that gets fully watched.*

*We could train on: (1) regression—predict minutes; (2) classification—binned watch time; (3) learning-to-rank—pairwise or listwise, where we want to rank videos by actual watch time. Listwise is often best—we want the ordering to match what would maximize total session watch time.*

*Features: user embedding—from history; video embedding—from metadata, content, engagement; context—time of day, device, session length so far; cross features—user-video affinity, similar videos watched. We need to handle the currently playing video if we're doing Up Next—that's a huge signal.*

*For training: we sample negatives from impressed-but-not-clicked or impressed-but-short-watch. We need to account for position bias—videos higher in the list get more watch time partly due to position. Inverse propensity weighting or position as a feature with careful serving.*

*The model outputs a score proportional to expected watch time. We use that for ranking. Calibration isn't as critical as in ads—we care about relative order."*

**I:** *"How do you handle cold start for new creators?"*

**C:** *"New creators are hard. We have little history. Options: (1) Content-based—video metadata, title, thumbnail, early engagement; (2) Creator embedding from similar creators; (3) Exploration—we inject a fraction of recommendations from new/small creators, maybe via a bandit—explore when uncertainty is high; (4) Bootstrap with a small amount of traffic and quick feedback. We'd want a dedicated 'new creator' candidate source that gets a fixed fraction of slots, and we'd monitor their exposure and watch time to tune that fraction."*

**I:** *"Explore vs exploit?"*

**C:** *"We need to balance showing what we know works vs discovering new content. Techniques: (1) Thompson sampling or UCB—treat candidate sources or items as arms; (2) Epsilon-greedy—small fraction random; (3) Diversity injector—explicitly add diverse items; (4) Uncertainty-based—boost items where our prediction variance is high. In practice, a mix: we have an exploration candidate source, and the diversity pass in reranking ensures we're not only showing tail-exploited content."*

*[Interviewer notes: Solid ML fundamentals. Watch time as target, cold start, exploration. Good.]*

**I:** *"What features would you use for the heavy ranker?"*

**C:** *"User-side: embedding from watch history, demographics if we have them, session context—how long they've been on, time of day, device. Video-side: embedding from metadata and engagement, length, upload date, channel subscriber count, engagement rate. Cross: user-video affinity—have they watched this channel before? Similar videos watched. Context: session length so far—if they've watched 5 long videos, maybe they want something shorter. We'd get embeddings from a training job that runs daily; real-time context from the request. Feature store holds precomputed user and video features; we'd need point-in-time correctness for training—no data leakage from future."*

**I:** *"How does the training pipeline work?"*

**C:** *"We have watch logs: user, video, timestamp, watch duration, position. We sample positives—impressions that led to watch—and negatives—impressed but not clicked, or clicked but very short watch. We need to handle position bias in the sampling or loss. We train on a window—e.g., last 30 days—and retrain daily. We'd use something like TensorFlow or PyTorch with a learning-to-rank objective—ListNet, LambdaRank, or similar. Model versioning and A/B testing for rollout."*

---

### Trade-offs and Extensions (5 min)

**I:** *"Key trade-offs?"*

**C:** *"Latency vs model capacity—we use multi-stage to keep latency; the heavy ranker only sees 200 candidates. Freshness vs cost—new uploads affect candidate gen and features; real-time pipelines cost more. Click vs watch time—if we optimized clicks we might favor clickbait; watch time is better but noisier. And exploration vs exploitation—we sacrifice some short-term watch time for long-term discovery."*

**I:** *"How would you scale this to 10× traffic?"*

**C:** *"Horizontal scaling—more replicas for each stage. Candidate gen is often the bottleneck—ANN search at 100K+ QPS—we'd shard the embedding index. Feature store read capacity—caching hot users. Heavy ranker—batch requests where possible; more GPU replicas. We'd also consider regional deployment to reduce latency and egress."*

**I:** *"What if the heavy ranker is slow or down?"*

**C:** *"Fallback: serve from light ranker output. Or use a cached 'popular in region' ranking. Circuit breaker to fail fast. We'd have multiple replicas; if one region is down, route to another."*

**I:** *"Time. Good work."*

*[Interviewer notes: Candidate handles scaling and failure modes well. Strong overall.]*

---

## Walkthrough 3: Design a Real-time Fraud Detection System

*Approximate timing: Clarify 2 min | Architecture 10 min | Features & model 8 min | Concept drift 5 min | Monitoring 3 min | Trade-offs 5 min | Follow-ups 7 min*

**I:** *"Design a real-time fraud detection system for a payment processor. When a transaction comes in, we need to decide: approve, decline, or send to manual review."*

**C:** *"Thanks. A few clarifications.*

*What types of fraud—card-not-present, account takeover, identity theft?"*

**I:** *"Assume card-not-present primarily—e-commerce transactions. But the system should be flexible."*

**C:** *"Latency budget?"*

**I:** *"We need a decision within 100ms—the user is waiting at checkout."*

**C:** *"How do we feel about false positives—declining good transactions?"*

**I:** *"We want to minimize them. Declining a real customer is costly—lost sale, bad experience. But we also can't let fraud through. So we have a manual review queue for borderline cases."*

**C:** *"Transaction volume?"*

**I:** *"Assume millions of transactions per day—tens of thousands of QPS at peak."*

**C:** *"Do we have labeled data—chargebacks, disputes—for training?"*

**I:** *"Yes, but it's delayed. Chargebacks come in 30–90 days later."*

*[Interviewer notes: Good—fraud type, latency, FP tolerance, scale, data. Covers the key constraints. Candidate asked about labels—shows ML thinking.]*

---

### Architecture (10 min)

**C:** *"Let me sketch the flow."*

```
[Transaction] → [Validate] → [Enrich: user, device, merchant]
       → [Feature computation: real-time + store lookup]
       → [GBDT model: score 0-1]
       → [Threshold: approve | review | decline]
       → [Log decision]
       ↓
[Feedback: chargeback, dispute] (delayed, days/weeks)
       → [Retraining pipeline] → [Model update]
```

**C:** *"Real-time path: transaction hits our API. We validate—format, basic rules (e.g., amount limits). We enrich—look up user, device, merchant from our stores. We compute features—transaction amount, velocity, user history, device fingerprint, graph features if we have them. The model scores 0–1. We apply thresholds: below 0.1 approve, 0.1–0.7 review, above 0.7 decline. We log the decision and eventually the outcome—chargeback, dispute—for retraining.*

*We also have a batch path: overnight we analyze patterns, build graph features (e.g., device clusters, IP clusters, sybil detection), retrain the model, and push updates. The batch job feeds features into the online store. Graph features are expensive to compute—we can't do full graph traversal in 100ms—so we precompute and refresh hourly or daily."*

**I:** *"What features?"*

**C:** *"Transaction: amount, merchant category, time, country, currency mismatch. User: account age, past chargebacks, velocity—transactions in last hour, last day. Device: fingerprint, new device? IP: geo consistency, VPN/proxy. Graph: has this device been seen with other fraudulent accounts? Is this merchant high-risk? Some features are real-time—computed from the current transaction and recent history. Others are precomputed—user aggregates, device reputation—and stored in a feature store for fast lookup."*

**I:** *"What model? And when would you use rules vs ML?"*

**C:** *"We'd use both. Rules first for obvious cases: amount over a threshold, known bad merchant, blocked card. That's fast and interpretable. For the rest, we'd score with ML.*

*Model: For 100ms we need something fast. Options: (1) GBDT—XGBoost, LightGBM—very fast, handles tabular well, interpretable; (2) Small NN—if we have many features; (3) Hybrid. I'd start with GBDT—sub-millisecond inference, good for structured data. We can add a neural component later if it helps.*

*Class imbalance is huge—fraud is maybe 0.1–1%. We'd use class weights, oversample fraud, or adjust the decision threshold. We'd optimize for precision at high recall—we can't decline too many good users. Or use a cost-sensitive formulation: cost of FP (lost sale, customer friction) vs cost of FN (fraud loss). We'd tune the threshold to minimize expected cost.*

*Rules are good for: known patterns, regulatory requirements, fast reaction to new attack vectors. ML is better for: complex patterns, adapting to novel fraud, combining many weak signals. We'd run rules before the model—if a rule fires we might skip the model or override."*

**I:** *"Concept drift—fraudsters adapt. How do you handle that?"*

**C:** *"Concept drift is critical. Fraudsters change tactics. We need: (1) Frequent retraining—daily or more; (2) Monitor performance—precision, recall, chargeback rate over time; alert on drops; (3) Feedback loop—chargebacks and disputes are delayed (days, weeks), so we use partial feedback or delayed feedback techniques; (4) Ensemble or multiple models—some tuned for known fraud, some for novelty; (5) Human review queue—analysts label and we retrain on that. We might also have an online learning component for fast adaptation, but that's complex—batch retraining is usually the first line."*

*[Interviewer notes: Strong on drift, feedback, class imbalance. Practical model choice.]*

**I:** *"How do you monitor this system?"*

**C:** *"We'd track: (1) Model performance—precision, recall, AUC by segment; if precision drops we're declining more good users. (2) Chargeback rate—if it spikes, we're missing fraud. (3) Feature drift—distributions of key features; if transaction patterns change, we might need retraining. (4) Latency—p50, p99 for the scoring path; if we're slow we risk checkout abandonment. (5) Review queue size—if it grows we might need more analysts or threshold tuning. We'd set alerts on these and have a dashboard. We'd also do periodic backtests—replay historical transactions with the current model and see what we would have decided."*

---

### Trade-offs and Scaling (5 min)

**I:** *"Trade-offs?"*

**C:** *"Latency vs feature richness—graph features might require multiple lookups; we might precompute and cache. False positive vs false negative—we tune the threshold; more declines reduce fraud but hurt good customers. Rules vs ML—rules are interpretable and fast to update; ML catches complex patterns. We'd use both. Batch vs streaming features—streaming is fresher but more complex; we'd prioritize the most important features for real-time."*

**I:** *"Scale to 10× traffic?"*

**C:** *"Horizontal scaling—more scoring service replicas. Feature store—ensure it can handle the read QPS; we might add caching for hot users/merchants. The model is small—GBDT—so each replica can handle high QPS. We'd also partition—e.g., by region or merchant segment—if we hit limits."*

**I:** *"How do you handle the delayed feedback from chargebacks?"*

**C:** *"Chargebacks come in 30–90 days later. Options: (1) Delayed feedback modeling—train a model to predict 'will this eventually be fraudulent?' from early signals; (2) Importance weighting—upweight recent negatives that might become positives; (3) Two-stage: immediate model for real-time, separate model trained on confirmed labels for periodic calibration; (4) Use partial labels—e.g., if a transaction goes to review and is approved, we have a negative sooner. In practice, we'd use a combination—train on what we have, with appropriate handling of the delay."*

**I:** *"What would you do differently?"*

**C:** *"I'd explore online learning for faster adaptation to new fraud patterns—though it's complex with delayed labels. I'd also invest more in the graph features pipeline—device clustering, merchant risk—since those catch coordinated fraud. And I'd strengthen the human review queue—better tools for analysts, active learning to prioritize the most valuable reviews."*

*[Interviewer notes: Strong on delayed feedback, practical. Good close.]*

---

## What Strong Candidates Do Differently

Across all three walkthroughs, the candidate demonstrates behaviors that distinguish strong from average performance:

| Behavior | Strong Candidate | Average Candidate |
|----------|------------------|-------------------|
| **Clarification** | Asks 5–6 focused questions before designing | Jumps into design with assumptions |
| **Structure** | States framework upfront; follows CLEAR phases | Meanders; covers topics reactively |
| **Estimation** | Shows formula and reasoning; states assumptions | Skips or does hand-wavy numbers |
| **Architecture** | Draws flow; explains each component's role | Describes vaguely without diagram |
| **Depth** | Goes deep when probed; cites techniques (MMoE, position bias) | Stays shallow; "we'd use a model" |
| **Trade-offs** | Names alternatives and justification | Single solution; no alternatives |
| **Failure modes** | Fallback, circuit breaker, scaling | Doesn't address |
| **Close** | Acknowledges limitations; specific improvements | "I think I did well" or vague |

**Practice tip:** Record yourself doing a 45-minute mock. Compare your transcript to these walkthroughs. Identify gaps—did you clarify? Did you estimate? Did you go deep when probed?—and drill those areas.

---

## Walkthrough 4 (Condensed): Design a Feature Store

*This is a 20-minute example for an infrastructure-focused question. The candidate moves quickly through clarification and goes deep on design.*

**I:** *"Design a feature store for an ML platform."*

**C:** *"Quick clarifications: Who are the users—how many teams? Online serving, offline training, or both? Feature types—batch, streaming, or both?"*

**I:** *"Multiple teams. Both online and offline. Batch and streaming."*

**C:** *"I'll design for dual storage. Online store for low-latency serving—Redis or DynamoDB—p99 under 10ms. Offline store for training—Parquet in a data lake or warehouse. Unified schema: feature groups keyed by entity (user_id, item_id).*

*Ingestion: Batch pipeline (Spark, daily/hourly) writes to offline store and backfills online for hot entities. Streaming pipeline (Flink, Kafka) writes real-time features to online store; we optionally append to offline for point-in-time correct training.*

*Key design: point-in-time correctness. When training on date T, we must not use features from after T. We store feature values with timestamps; at training we use "as of" semantics—for each (entity, timestamp) we get the latest feature value before that timestamp. That requires careful backfill logic.*

*API: get_features(entity_ids, feature_names, timestamp) for offline; get_features(entity_ids, feature_names) for online—returns latest.*

*Governance: access control, PII handling, lineage—which jobs produce/consume each feature. Discovery: catalog of feature groups, descriptions, owners."*

**I:** *"How do you handle backfill when a feature definition changes?"*

**C:** *"We version features. When logic changes, we run a backfill job over historical data to recompute. That can be expensive—we might do it incrementally or only for recent dates. We'd track feature version in the schema so training jobs can request a specific version."*

**I:** *"Good. Time."*

*[Interviewer notes: Candidate covered dual store, point-in-time correctness, ingestion paths, API, governance. Good infrastructure depth.]*

---

## Walkthrough 5 (Condensed): Design Ad Click Prediction

*Another common Meta-style question. Focus on calibration, latency, and scale.*

**I:** *"Design an ad click prediction system."*

**C:** *"Clarifying: auction type—first or second price? Latency budget? Scale?"*

**I:** *"Second price. 50ms. Billions of predictions per day."*

**C:** *"Architecture: Ad retrieval (targeting + candidate selection) → Feature enrichment → pCTR model with calibration → Auction (eCPM = bid × pCTR).*

*Key point: calibration. Raw model scores aren't probabilities. For a second-price auction we need well-calibrated pCTR—overestimate and we hurt advertisers; underestimate and we leave money on the table. We'd use Platt scaling or isotonic regression as a post-processing step.*

*Model: DCN or Wide & Deep, multi-task with pCVR (conversion) if we have that. Features: user (demographics, history), ad (creative, advertiser), cross (user-ad affinity). Real-time features for last-click, last-view.*

*Scale: 1B DAU × 40 ad impressions/day ≈ 40B/day ≈ 500K QPS avg; peak 1.5M. With 80% cache hit on popular ads, we'd need to score ~300K QPS. At 10ms per inference, that's thousands of GPU-seconds of compute—we'd need thousands of GPUs with batching. In practice, aggressive caching and optimization reduce that."*

**I:** *"Why calibration matters for second-price auction?"*

**C:** *"In second-price, the winner pays the second-highest bid. If our pCTR is miscalibrated—say we systematically overestimate—we might rank a low true CTR ad above a high true CTR ad. That hurts both user experience and advertiser ROI. Advertisers optimize their bids assuming our pCTR is correct. Calibration ensures our predictions are trustworthy for bidding."*

**I:** *"Good."*

*[Interviewer notes: Hit calibration—critical for ads. Scale reasoning. Would probe deeper on feature freshness if time.]*

---

## Appendix: Common Interviewer Probes

Interviewers often drill into specific areas. Prepare 2–3 sentence answers for these:

| Probe | Strong Answer Direction |
|-------|-------------------------|
| *"How do you handle cold start?"* | Metadata/popularity for new items; exploration for new users; separate cold-start path or default features |
| *"How do you prevent feedback loops?"* | Diversify recommendations; monitor metrics by segment; avoid over-exploiting high-scoring items |
| *"What if the model is wrong?"* | Fallback (cached ranking, rules); monitor and alert; A/B test; human oversight for critical decisions |
| *"How do you ensure training-serving consistency?"* | Shared feature definitions; feature store; same preprocessing library; integration tests |
| *"How would you scale 10×?"* | Horizontal scaling; sharding; caching; regional deployment; optimize bottlenecks (often feature store or retrieval) |
| *"What's your latency budget breakdown?"* | LB 1–2ms, auth 1–2ms, feature fetch 5–20ms, model 10–50ms, post-process 1–5ms; total &lt; 100ms for ads, &lt; 200ms for feed |
| *"How do you handle class imbalance?"* | Class weights, oversampling, threshold tuning, cost-sensitive loss, or two-stage (retrieval then classification) |
| *"How do you detect drift?"* | Feature distribution monitoring; model performance by segment; statistical tests (PSI, KS); alert on significant shifts |
| *"How do you do offline evaluation?"* | Holdout set; metrics (NDCG, AUC, precision@k); counterfactual evaluation if possible; align with online metrics |
| *"What would you do differently?"* | Acknowledge 2–3 specific improvements; link to trade-offs; don't over-apologize |

---

## Appendix: Red Flags Interviewers Watch For

Avoid these behaviors—they cost points even when the rest of your answer is strong:

| Red Flag | Why It Hurts | Better Approach |
|----------|--------------|-----------------|
| **No clarifying questions** | Suggests you'll build the wrong thing | Ask 5–6 questions in first 2 min |
| **"We'll use ML"** (no specifics) | Vague; doesn't show depth | "We'd use a two-tower model for retrieval, then a DNN ranker" |
| **Ignoring scale** | Real systems have constraints | Estimate QPS, storage; mention caching, sharding |
| **No trade-offs** | Suggests you don't think critically | "The trade-off is X vs Y; we chose X because…" |
| **No failure modes** | Production systems fail | "If the ranker goes down, we'd fall back to…" |
| **Over-complicating early** | Wastes time; seems inexperienced | Start simple (LR, rule-based); iterate |
| **Blaming "the team"** for gaps | Take ownership | "I would ensure we…" |
| **Rigid memorization** | Can't adapt to the question | Use framework flexibly; listen to signals |
| **Running over time** | Poor time management | Practice with a timer; leave buffer |
| **Defensive when probed** | Suggests you can't handle feedback | "Good point—I'd add…" or "I'd reconsider…" |

---

## Appendix: Time Management in a 45-Minute Interview

| Phase | Time | What to Do | What to Avoid |
|-------|------|------------|---------------|
| **Clarify** | 5–8 min | 5–6 questions; scope the problem | Jumping to design; asking too many questions |
| **Requirements + Estimate** | 3–5 min | Write down FR/NFR; show QPS/storage math | Spending 10 min on requirements; skipping estimation |
| **High-level architecture** | 5–8 min | Draw flow; name components; explain data flow | Over-detailing; drawing for 15 min |
| **Deep dive (1–2 areas)** | 12–18 min | Go deep when probed; cite techniques | Staying shallow; ignoring interviewer signals |
| **Trade-offs + Failure modes** | 5–8 min | Name alternatives; fallbacks; scaling | "It just works"; no alternatives |
| **Follow-ups** | 5 min | Answer concisely; "What would you do differently?" | Rambling; running out of time |

**Buffer:** Leave 2–3 min for wrap-up. If you're ahead, offer: "I can go deeper on X or discuss Y—what would be most useful?"

---

## Appendix: Sample Opening and Transition Phrases

Strong candidates use clear signposting. Practice these:

**Opening (first 30 seconds):**
- *"I'll use a structured approach: clarify the problem, list requirements and estimate scale, design the architecture, go deep on key components, then discuss trade-offs and failure modes."*
- *"I'd like to ask a few clarifying questions before we dive in."*

**Transitions:**
- *"Now that I have context, let me summarize requirements and do a quick scale estimate."*
- *"Here's the high-level architecture—I'll draw the flow and then we can go deeper where you'd like."*
- *"The most critical part is X—can I go deeper there?"*
- *"A key trade-off here is…"*
- *"If that component failed, we'd…"*
- *"To scale 10×, we'd need to…"*

**When you need to buy time:**
- *"Let me think through that for a moment."* (5–10 sec pause is fine)
- *"There are a few approaches—let me walk through the one I'd recommend first."*

**When you're unsure:**
- *"I'm not 100% on the latest approach for X, but my understanding is…"*
- *"I'd want to validate that with the team, but my initial thought is…"*

---

## Appendix: Full Deep-Dive Example (Training Pipeline)

When an interviewer says *"Go deeper on the training pipeline,"* you have 5–8 minutes. Here's a full expansion:

**I:** *"Walk me through how you'd train the recommendation model."*

**C:** *"I'll cover data, objective, features, and infrastructure.*

*Data: we have impression logs—user, item, timestamp, position, and labels: click, like, share, dwell time. We need to construct training examples. For each positive—an impression that led to engagement—we need negatives. We'd sample from: (1) impressed but not clicked—items we showed and the user didn't interact with; (2) random items from the catalog—to improve recall on the long tail. We might use in-batch negatives for efficiency—within a training batch, other users' positives serve as negatives for a given user.*

*Objective: we're doing multi-task—predict P(like), P(comment), P(share), P(dwell>15s). We'd use binary cross-entropy per head. The total loss is a weighted sum. We might add a listwise ranking loss—LambdaRank or similar—so we optimize order, not just pointwise accuracy.*

*Position bias: items at the top get more engagement. We add position as a feature. At serving we'd use a fixed position or inverse propensity weighting so we don't learn that top=better.*

*Features: we'd pull from the feature store—user embedding, item embedding, context. Critical that training and serving use the same feature logic. We'd use a library like Feathr or a custom pipeline that generates training data with point-in-time correctness—for each (user, item, timestamp) we get features as they existed at that time, no leakage from the future.*

*Infrastructure: Spark or Flink for batch; we'd run daily. Training on GPU with something like TensorFlow or PyTorch. Experiment tracking with MLflow. We'd validate on holdout—NDCG, AUC—and A/B test before full rollout."*

**I:** *"How long would training take?"*

**C:** *"Depends on data size and model. For 500M examples, a DNN might take 2–4 hours on 8–16 GPUs. We'd optimize with distributed training, mixed precision. If it's too long we'd sample—e.g., 10% of data—or use a smaller model for the light ranker."*

**I:** *"How do you handle the delay between training and deployment?"*

**C:** *"We'd run training on a schedule—e.g., 2am—and deploy by 6am. For that 4-hour gap, we serve the previous model. If we need fresher models we could do more frequent retraining or look at streaming/online learning—but that's more complex. For most recommendation systems, daily retraining is sufficient."*

*[This level of detail—data, objective, bias, features, infra, timing—is what "deep dive" means. Practice 2–3 such full expansions for each major component: training, retrieval, feature store, monitoring.]*

---

## Appendix: Handling Curveballs

Interviewers sometimes shift direction or add constraints mid-design. Stay calm and adapt.

| Curveball | Strong Response |
|-----------|-----------------|
| *"Actually, latency is 50ms, not 200ms"* | "That changes things. We'd need a lighter model—maybe skip the heavy ranker or use a much smaller one. We'd rely more on candidate generation quality and caching." |
| *"We have no historical data—cold start for the whole system"* | "We'd start with content-based—metadata, embeddings from thumbnails/titles. Use popularity and trending. Add collaborative signals as we get engagement. Bootstrap with exploration." |
| *"How would you design this for a startup with 10 engineers?"* | "Simplify: single-stage ranking, maybe LR or GBDT. Batch features only. Use managed services—BigQuery, a hosted feature store. Defer real-time and multi-stage until we have traction." |
| *"The business wants to optimize for revenue, not engagement"* | "We'd incorporate revenue signals—ad revenue per impression, subscription conversion. Multi-objective: engagement + revenue. Or a separate revenue model that re-ranks." |
| *"How do you prevent the system from reinforcing harmful stereotypes?"* | "Careful feature selection—avoid protected attributes as direct inputs. Monitor metrics by demographic segment. Diversity and fairness constraints in ranking. Regular audits." |
| *"What if you had to design this in a week?"* | "MVP: rule-based or simple model (LR) on top 10 features. Batch training daily. Single region. No multi-stage. Get something live, then iterate." |

**Principle:** Don't argue with the new constraint. Acknowledge it, state the impact, and adjust your design. *"Given that, I would…"*

---

## Appendix: How the Walkthroughs Map to CLEAR

Use this to self-check: did you hit each phase?

| Walkthrough | Clarify | List & Estimate | Architect | Refine |
|-------------|---------|-----------------|-----------|--------|
| **1. Instagram Feed** | 5 Qs: goal, scale, latency, ads, diversity/integrity | FR/NFR; 60K QPS, 180K peak, feature store 400GB | Multi-stage: candidates → light → heavy → integrity → diversity | Engagement vs diversity; fallback; fairness |
| **2. YouTube** | 5 Qs: scope, metric, scale, latency, diversity | Implicit | Candidate gen → light → heavy (EWT) → rerank | Latency vs capacity; cold start; exploration |
| **3. Fraud** | 6 Qs: fraud type, latency, FP tolerance, scale, labels | Implicit | Validate → enrich → features → model → decide → log | Rule vs ML; drift; delayed feedback |
| **4. Feature Store** | 3 Qs: users, online/offline, batch/streaming | Implicit | Dual store; ingestion; API; point-in-time | Backfill; versioning |
| **5. Ad Click** | 3 Qs: auction, latency, scale | 500K QPS, 1.5M peak | Retrieve → features → pCTR + calibration → auction | Calibration for second-price |

**Common gaps:** Candidates often skip Estimate or Refine. Both matter. Practice including them explicitly.

---

## Appendix: What to Practice

| Before the interview | Practice |
|---------------------|----------|
| **Full mocks** | 3–5 complete 45-min sessions with a partner or recorder |
| **Estimation drills** | 5 min each: estimate QPS/storage for 3 different systems |
| **Deep-dive prep** | Prepare 3-min answers for: training pipeline, retrieval, feature store, monitoring |
| **Trade-off bank** | For each major decision, write down 2 alternatives and the trade-off |
| **Company tuning** | Meta: ads, feed; Google: search, YouTube; Amazon: recommendations |
| **"What would you do differently?"** | Practice 3 specific answers per design you've done |

---

*Each walkthrough demonstrates: structured clarification, clear requirements and estimation, end-to-end architecture, deep dives on ML components, trade-offs, and graceful handling of follow-ups. Practice these patterns until they feel natural.*
