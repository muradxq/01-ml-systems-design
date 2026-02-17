# Common Mistakes and Anti-Patterns

These 15 mistakes are frequently made in ML system design interviews. Each section includes a bad example, why it’s wrong, a good example, and how interviewers typically react.

---

## 1. Jumping to Model Choice

**What it looks like (bad):**

*Interviewer:* "Design a recommendation system for our e-commerce platform."

*Candidate:* "We’ll use a transformer-based model with cross-attention between user and item embeddings. BERT4Rec has shown great results for sequential recommendations..."

**Why it’s wrong:** The model is chosen before the problem is scoped. Transformers may be unnecessary; you don’t yet know scale, latency, or data.

**How to fix it (good):**

*Candidate:* "Before picking a model, I’d like to clarify a few things. What’s the scale—users and catalog size? What are the latency requirements? Do we have historical interactions? ... [After clarification] For an MVP with 1M users and 100ms latency, I’d start with a two-tower model or even logistic regression. We can evolve to more complex models if needed."

**How interviewers react:** Interviewers notice when you skip clarification. They may redirect with “What if we have different constraints?” or note lack of problem-scoping.

---

## 2. Ignoring Data

**What it looks like (bad):**

*Candidate:* [Spends 25 minutes on model architecture, training loop, serving] "So we’ll serve the model with 5 replicas. Any questions?"

*Interviewer:* "Where does your training data come from?"

*Candidate:* "Oh, we’d use click data. I guess we’d get it from somewhere."

**Why it’s wrong:** ML systems are driven by data. Ignoring sources, quality, freshness, and biases leads to weak or harmful models.

**How to fix it (good):**

*Candidate:* "Data is foundational. We’d collect implicit signals—clicks, views, add-to-cart—and explicit feedback if available. We need to handle position bias—items higher in the list get more clicks regardless of relevance. We’d sample negatives from impressed-but-not-clicked. For cold start, we’d use item metadata. I’d also plan for data quality checks—missing values, distribution shift—before training."

**How interviewers react:** Interviewers expect data to be a first-class topic. Skipping it suggests shallow ML experience.

---

## 3. No Clarifying Questions

**What it looks like (bad):**

*Interviewer:* "Design a search ranking system."

*Candidate:* "Great. So we’ll have a retrieval stage with ANN, then a ranking stage with a BERT model. The retrieval stage will..."

**Why it’s wrong:** “Search” could mean web search, product search, or internal document search. Scale, latency, and product goals differ widely.

**How to fix it (good):**

*Candidate:* "A few clarifying questions. Is this web search, product search, or something else? What’s the expected QPS and latency? What’s the document corpus size? Are we optimizing for relevance, engagement, or something else? Is there an existing system we’re improving?"

**How interviewers react:** Interviewers expect 5–8 minutes of clarification. Jumping straight to design is a red flag for problem framing.

---

## 4. Over-Engineering

**What it looks like (bad):**

*Interviewer:* "We have a simple spam detector. 10K emails per day. How would you improve it?"

*Candidate:* "We’d build a full ML pipeline with Kafka for streaming, a feature store, distributed training with 8 GPUs, and a real-time serving tier with A/B testing infrastructure."

**Why it’s wrong:** 10K/day is tiny. A rule-based system or a single logistic regression model is enough. Complexity adds cost and failure modes.

**How to fix it (good):**

*Candidate:* "At 10K emails per day, I’d start simple: a logistic regression or small gradient-boosted tree on a few features—sender reputation, keywords, links. Run it in a batch job daily. Add monitoring for precision/recall. Only add streaming, feature stores, and distributed training if we scale to millions or need real-time."

**How interviewers react:** Interviewers want appropriate solutions for the problem. Over-engineering suggests poor judgment.

---

## 5. Forgetting Monitoring

**What it looks like (bad):**

*Candidate:* "So we train the model, deploy to the serving tier, and we’re done. The system will recommend items to users."

*Interviewer:* "How do you know if it’s working?"

*Candidate:* "We’d... look at metrics? Maybe set up some dashboards."

**Why it’s wrong:** Models degrade over time. Without monitoring, you won’t detect drift, bugs, or failures.

**How to fix it (good):**

*Candidate:* "We need monitoring at several levels. Input: feature distributions, missing rates, data freshness. Output: score distribution, latency percentiles. Business: CTR, engagement—via a logging pipeline into our metrics system. We’d set alerts for drift, latency spikes, and metric drops. If we see significant drift, we trigger retraining or investigation."

**How interviewers react:** Interviewers often ask “How do you know it’s working?” Missing monitoring is a common reason for down-level feedback.

---

## 6. Ignoring Fairness

**What it looks like (bad):**

*Interviewer:* "Design a system to rank job candidates."

*Candidate:* "We’ll use a model trained on historical hire data. Features: resume, experience, skills. We’ll rank by predicted success."

**Why it’s wrong:** Historical data often encodes bias. A model trained on biased labels can worsen disparities.

**How to fix it (good):**

*Candidate:* "I’d be careful about bias. Historical hire data may reflect past discrimination. I’d avoid proxy features for protected attributes, monitor performance across demographics, consider fairness constraints (e.g., demographic parity), and maybe use a human-in-the-loop for sensitive decisions. We’d also document limitations and get legal/review input."

**How interviewers react:** For user-facing or high-stakes systems, interviewers expect some discussion of fairness and bias.

---

## 7. No Fallback Plan

**What it looks like (bad):**

*Interviewer:* "What if your model server goes down?"

*Candidate:* "We’d fix it. Maybe add more replicas. Our infra team handles that."

**Why it’s wrong:** You need a concrete fallback so the product keeps working. “Fix it” is not a design.

**How to fix it (good):**

*Candidate:* "We’d have multiple fallbacks. First, we’d rely on replicas and load balancing. If the model server is unhealthy, we’d fail over to a cached ranking—e.g., popular items or last-known-good ranking. We could also fall back to a lighter rule-based ranker. We’d circuit-break to prevent cascade failures and alert on-call."

**How interviewers react:** “What if X fails?” is a standard follow-up. Lack of fallbacks suggests weak reliability thinking.

---

## 8. Not Discussing Trade-Offs

**What it looks like (bad):**

*Candidate:* "We’ll use a two-tower model for retrieval."

*Interviewer:* "Why two-tower vs a single cross-attention model?"

*Candidate:* "Two-tower is standard. It’s what everyone uses."

**Why it’s wrong:** You haven’t explained the trade-off. Interviewers want to see structured reasoning.

**How to fix it (good):**

*Candidate:* "Two main options: two-tower vs single model with cross-attention. Two-tower precomputes item embeddings, so retrieval is just vector search—very fast, scales to millions. A single cross-attention model can capture richer interactions but doesn’t precompute; we’d need to score each user–item pair, which is too slow at scale. So we use two-tower for retrieval and can add a cross-model ranker on top of a smaller candidate set."

**How interviewers react:** Explicit trade-offs show depth and judgment. Vague answers often lead to more probing.

---

## 9. Ignoring Training-Serving Skew

**What it looks like (bad):**

*Interviewer:* "How do you ensure features match between training and serving?"

*Candidate:* "We use the same code. Should be fine."

**Why it’s wrong:** Feature computation often differs across environments. Skew causes silent degradation.

**How to fix it (good):**

*Candidate:* "We’d use a feature store that serves both training and online. Same feature definitions and logic in both. We’d log features at serving time and compare distributions to training. We’d also version features and models together. If we can’t use a shared store, we’d at least run the same feature pipeline in both and add validation checks."

**How interviewers react:** Interviewers familiar with ML in production care about this. It’s a quick way to show maturity.

---

## 10. Missing Online Evaluation

**What it looks like (bad):**

*Candidate:* "We’ll evaluate with AUC and NDCG on a holdout set. When those look good, we deploy."

*Interviewer:* "How do you validate in production?"

*Candidate:* "We’d... hope it works? Maybe watch the metrics."

**Why it’s wrong:** Offline metrics don’t always predict online performance. You need a clear A/B testing strategy.

**How to fix it (good):**

*Candidate:* "Offline metrics like NDCG help us iterate quickly, but they don’t capture position bias or user behavior changes. We’d run A/B tests: control gets the current model, treatment gets the new one. Primary metric: engagement or CTR. We’d power the experiment for a meaningful lift, run it for at least a week to capture weekly patterns, and check for segment-level regressions before full rollout."

**How interviewers react:** “Hope it works” is a red flag. A/B testing and gradual rollout are expected.

---

## 11. Poor Time Management

**What it looks like (bad):**

*Candidate:* [Spends 30 minutes on the data pipeline, 5 minutes on model and serving, 2 minutes on monitoring]

*Interviewer:* "We have 5 minutes left. Can you cover failure modes and scaling?"

*Candidate:* "Uh, we’d add more servers. And maybe retrain if something goes wrong."

**Why it’s wrong:** You’ve spent too long on one area and left critical topics for the end.

**How to fix it (good):**

*Candidate:* [Uses the CLEAR framework: 7 min clarify, 4 min requirements+estimates, 20 min architecture with balanced coverage, 8 min refine] "I’ve covered the main components. Let me spend the last few minutes on failure modes and scaling..."

**How interviewers react:** Interviewers notice coverage. Use a timer and a mental checklist to balance time.

---

## 12. Not Connecting to Business Goals

**What it looks like (bad):**

*Candidate:* "We’ll use NDCG@20 as our metric. We’ll optimize for that."

*Interviewer:* "How does that relate to our business?"

*Candidate:* "NDCG measures ranking quality. Better rankings mean better recommendations."

**Why it’s wrong:** You haven’t tied NDCG to revenue, engagement, or other business outcomes.

**How to fix it (good):**

*Candidate:* "Our business goal is increased engagement—more sessions, more items viewed. NDCG proxies that—better relevance leads to more clicks and engagement. We’d validate the proxy with A/B tests. We’d also track downstream metrics like add-to-cart and purchase to ensure we’re not optimizing for clicks at the expense of conversion."

**How interviewers react:** Strong candidates connect metrics to outcomes. Disconnected metrics suggest weak product sense.

---

## 13. Ignoring Cold Start

**What it looks like (bad):**

*Interviewer:* "How do you handle new users who have no history?"

*Candidate:* "We’d recommend popular items. Or random. Something like that."

**Why it’s wrong:** Cold start is a core challenge. A vague answer suggests it wasn’t considered in the design.

**How to fix it (good):**

*Candidate:* "For new users we have several options: (1) Popular/trending items as a baseline. (2) Ask for preferences up front—categories, interests—and use that as initial features. (3) Use context—device, location, referral—if available. (4) Use a bandit or exploration strategy to learn quickly. We’d A/B test these and likely combine (2) and (4) for a personalized cold-start experience."

**How interviewers react:** Cold start is a common follow-up. A clear strategy shows you’ve thought about it.

---

## 14. Feature Engineering as Afterthought

**What it looks like (bad):**

*Candidate:* "We’ll feed user IDs and item IDs into a neural network. The model will learn the embeddings."

*Interviewer:* "What about user demographics, item attributes, recency?"

*Candidate:* "We could add those. The model might use them."

**Why it’s wrong:** Features often matter more than model choice. Treating them as optional undervalues their impact.

**How to fix it (good):**

*Candidate:* "Features are critical. We’d have user features—demographics, past engagement, session context. Item features—category, recency, popularity. Interaction features—historical CTR for this user–category, co-engagement. We’d also add recency—when the user last engaged with similar items. We’d use a feature store for consistency and iterate on features based on importance analysis and ablation."

**How interviewers react:** Strong candidates discuss features early and in detail. Treating them as an afterthought suggests inexperience.

---

## 15. Treating ML as a Black Box

**What it looks like (bad):**

*Candidate:* "We’ll use a deep learning model. It’s state of the art. It will figure things out."

*Interviewer:* "Why is that a good fit for this problem?"

*Candidate:* "Deep learning works well for most things. It’s flexible."

**Why it’s wrong:** You haven’t explained why the model fits the problem, data, and constraints.

**How to fix it (good):**

*Candidate:* "For this recommendation problem we have categorical IDs (user, item) and continuous features. A two-tower model fits well: we need to embed users and items for fast retrieval, and we have enough data for learning embeddings. A simple DNN would underfit; a giant transformer would be overkill and too slow for our latency budget. Two-tower gives us fast retrieval and reasonable expressiveness. We could add a lightweight DNN ranker on top for the final 20 results."

**How interviewers react:** Interviewers want to see that you understand model choice. “It’s state of the art” without reasoning is weak.

---

## Red Flags Interviewers Notice

| Red Flag | What It Suggests |
|----------|------------------|
| No clarifying questions in first 5 minutes | Weak problem framing |
| Model choice in first 2 minutes | Jumping to solutions |
| "We'd use Kafka" without explaining why | Buzzword dropping |
| No mention of data or features | Inexperience with ML systems |
| No monitoring or evaluation plan | Not production-minded |
| Can't explain trade-offs when asked | Surface-level understanding |
| 30+ minutes on one component | Poor time management |
| "I'd Google it" or "I'd ask someone" | Avoid; show your reasoning |
| Dismissing fairness for user-facing systems | Lack of responsibility |

---

## Recovery Strategies

**If you realize you missed clarification:** "Actually, before I continue—can I confirm the scale and latency requirements? That would affect the architecture."

**If you went too deep too early:** "Let me step back. I've been focusing on retrieval—I should also cover training, serving, and monitoring. Here's the high-level flow..."

**If you don't know something:** "I'm not deeply familiar with X. My understanding is [reasonable guess]. I'd validate with the team and docs. The key trade-off I'd consider is..."

**If you're running out of time:** "We have 5 minutes. Let me prioritize: I'll cover fallbacks and one scaling scenario, then we can discuss what you'd like to explore."

---

## Checklist: Before Your Next Practice

- [ ] Did I ask clarifying questions first?
- [ ] Did I discuss data sources and quality?
- [ ] Did I start simple (MVP) before complex?
- [ ] Did I cover monitoring and feedback?
- [ ] Did I mention fairness for user-facing systems?
- [ ] Did I have a fallback plan?
- [ ] Did I explain trade-offs for key decisions?
- [ ] Did I address training-serving consistency?
- [ ] Did I describe online evaluation (A/B testing)?
- [ ] Did I manage time across all phases?
- [ ] Did I connect metrics to business goals?
- [ ] Did I address cold start?
- [ ] Did I treat feature engineering seriously?
- [ ] Did I justify my model choice?
- [ ] Did I avoid over-engineering for the stated scale?

---

*Use this checklist after each practice session. Fixing these mistakes will significantly improve your interview performance.*

---

## Practice Exercise: Spot the Mistakes

Read this fictional exchange and identify which of the 15 mistakes the candidate makes:

**Interviewer:** "Design a job recommendation system for LinkedIn."

**Candidate:** "We'll use a GNN to model the professional graph, then a transformer for sequence encoding. We'll train on historical apply data. Serving: we'll deploy with 100 GPUs for real-time inference. We'll use NDCG for evaluation. That should work."

**Mistakes present:**
1. No clarifying questions (scale, latency, cold start, fairness)
2. Jumping to model choice (GNN, transformer) before scoping
3. Over-engineering for unknown scale
4. No monitoring, fallbacks, or online evaluation
5. Ignoring fairness (job recommendations are high-stakes)
6. No discussion of training-serving consistency or data
7. Treating ML as black box (why GNN? why transformer?)

**Better opening:** "Before I design, a few questions: What's the scale—users and job postings? Latency requirements? Do we have explicit feedback (applies, saves) or only implicit? Any fairness requirements? ... [After answers] I'd start with a two-tower or simpler approach for MVP, then iterate based on scale and constraints."

---

## Summary: The Five-Minute Refresher

Before each practice or interview, remind yourself:

1. **Clarify first** — 5–8 minutes of questions
2. **Data matters** — Sources, quality, bias, freshness
3. **Start simple** — MVP before complex models
4. **End-to-end** — Data → Training → Serving → Monitoring
5. **Trade-offs** — For every choice, know the alternative
6. **Fallbacks** — What happens when things break?
7. **Connect to business** — Why does this matter?

---

## Mistake Severity: What Hurts Most

Not all mistakes are equal. Based on typical feedback:

| High impact (often disqualifying) | Medium impact (lowers score) | Lower impact (recoverable) |
|-----------------------------------|------------------------------|----------------------------|
| No clarifying questions | Over-engineering | Poor time management |
| Ignoring data | No trade-off discussion | Missing cold start |
| No monitoring | Black-box model choice | Feature engineering light |
| No fallback plan | Training-serving skew ignored | Fairness not mentioned |
| Ignoring fairness (for sensitive domains) | No online evaluation | |

Focus on fixing the high-impact mistakes first. The medium ones add up.

---

## Role-Play: Turning a Bad Answer into a Good One

**Bad:** "We'll use a neural network for the recommendation system."

**Interviewer:** "Why?"

**Bad:** "Because they work well for this kind of problem."

**Good recovery:** "Let me be more specific. We need to rank items for users. The options are: (1) collaborative filtering—doesn't scale well for cold start; (2) content-based—limited by metadata; (3) two-tower or DNN—learns from interactions, handles scale. Given we have enough click data and need low-latency retrieval, I'd choose a two-tower model. The user tower and item tower give us precomputed embeddings for fast ANN retrieval, with a light ranker on top for the final ordering. The neural network fits because we have rich features and need to capture non-linear interactions."

Practice this kind of recovery. Interviewers often give you a chance to expand.

---

## Appendix: Quick Reference Card

**CLEAR phases:** Clarify (5–8) → List (3–5) → Estimate (3–5) → Architect (15–20) → Refine (5–10)

**Must-cover topics:** Data, features, model, training, serving, monitoring, fallbacks, trade-offs

**Never skip:** Clarification, data discussion, monitoring, at least one trade-off

**When stuck:** "Let me think about that for a moment" — then structure your answer (e.g., "There are two approaches: A and B. A gives us X but costs Y. B...")

---

## Building Good Habits Through Deliberate Practice

1. **Week 1**: Do one design per day. After each, review against the 15 mistakes. Note which you made.
2. **Week 2**: Focus on your top 3 recurring mistakes. Consciously avoid them in each practice.
3. **Week 3+**: Practice with a partner. Have them score you on the rubric (clarification, architecture, trade-offs, depth, communication).
4. **Before interview**: Re-read this file. Run through the checklist. Do one cold practice.

The goal is not to memorize answers but to internalize the habits—clarifying first, discussing data, covering monitoring, explaining trade-offs—so they become automatic under pressure.
