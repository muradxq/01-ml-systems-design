# ML System Design Interview Preparation

This section provides comprehensive material to prepare for ML System Design interviews at companies like Meta, Google, Amazon, and similar tech firms. The content is structured around a proven framework and real interview patterns.

---

## Overview

ML System Design interviews test your ability to design end-to-end machine learning systems—from data pipelines through model training to production serving and monitoring. Unlike traditional system design, these interviews require you to think about **data quality**, **feature engineering**, **model selection**, **training-serving consistency**, and **continuous improvement**—all while handling scale and reliability.

This material covers:
- A structured interview framework (the CLEAR Method)
- Company-specific expectations and common questions
- Common mistakes and how to avoid them
- Scale estimation methodology with worked examples
- A question bank of 25 ML system design questions
- Mock interview walkthroughs with exemplary performance

---

## How to Use This Material

### Study Plan

1. **Read the framework first** — Internalize the CLEAR Method before practicing. Time yourself on each phase.
2. **Practice with constraints** — Do full 45-minute mocks. Use a timer. Practice on whiteboard or digital equivalent.
3. **Review company guides** — Before interviewing at a specific company, study their common questions and tech stack.
4. **Learn from mistakes** — Read the anti-patterns file and consciously avoid each mistake during practice.

### Practice Approach

- **Solo practice**: Pick a prompt, set a 45-minute timer, run through the full framework. Record yourself and review.
- **Mock interviews**: Pair with a friend or use platforms like Pramp. Have your partner play interviewer and ask follow-ups.
- **Trade-off drills**: For each design decision, practice explaining 2–3 alternatives and why you chose one.
- **Depth preparation**: For your top 3 target companies, deep-dive into 2–3 components (e.g., feature stores, real-time serving, monitoring) so you can go deeper when signaled.

---

## Table of Contents

| File | Topic |
|------|-------|
| [01-interview-framework.md](./01-interview-framework.md) | Interview Framework: The CLEAR Method |
| [02-company-specific-guide.md](./02-company-specific-guide.md) | Company-Specific Interview Guide |
| [03-common-mistakes.md](./03-common-mistakes.md) | Common Mistakes and Anti-Patterns |
| [04-scale-estimation-guide.md](./04-scale-estimation-guide.md) | Scale Estimation for ML Systems |
| [05-question-bank.md](./05-question-bank.md) | Question Bank: 25 ML System Design Questions |
| [06-mock-interview-walkthroughs.md](./06-mock-interview-walkthroughs.md) | Mock Interview Walkthroughs |

---

## Key Principles for ML System Design Interviews

1. **Clarify before designing** — The worst designs come from assumptions. Spend 5–8 minutes scoping the problem.
2. **Data first** — ML systems are only as good as their data. Always discuss data sources, quality, and freshness.
3. **Start simple, then refine** — Begin with an MVP (e.g., rule-based or logistic regression), then iterate toward more complex models.
4. **End-to-end thinking** — Cover the full loop: data → features → training → serving → monitoring → feedback.
5. **Trade-offs matter** — For every major decision, mention alternatives and why you chose your approach.
6. **Connect to business** — Link technical choices to business outcomes (engagement, revenue, latency SLOs).
7. **Know your audience** — Adapt depth based on interviewer signals. Some want breadth; others want deep dives.

---

## Suggested 2-Week Intensive Study Plan

### Week 1: Foundations

| Day | Focus |
|-----|-------|
| 1 | Embeddings and retrieval (vector search, FAISS, two-tower models) |
| 2 | Feature stores, offline vs online features, feature freshness |
| 3 | Training pipelines (batch vs streaming, data versioning, experiment tracking) |
| 4 | Serving patterns (batch vs real-time, model versioning, A/B testing) |
| 5 | Monitoring, logging, and feedback loops (data drift, concept drift) |
| 6 | Back-of-the-envelope estimation (QPS, storage, latency budgets) |
| 7 | Review and practice one full design (pick from practice prompts) |

### Week 2: Practice

| Day | Focus |
|-----|-------|
| 1 | 2–3 full system designs (45 min each). Review with the common-mistakes checklist. |
| 2 | 2–3 full designs. Focus on trade-off explanations. |
| 3 | Mock interview with partner. Target company: Meta. |
| 4 | 2–3 full designs. Focus on data and monitoring. |
| 5 | Mock interview with partner. Target company: Google. |
| 6 | 2–3 full designs. Practice handling “what would you do differently?” |
| 7 | Light review. Do one final mock. Rest before interview day. |

---

## How Interviews Are Scored

Interviewers typically evaluate across these dimensions:

| Dimension | What they look for |
|-----------|-------------------|
| **Problem framing** | Did you ask clarifying questions? Did you scope correctly? |
| **System design** | End-to-end coverage, reasonable architecture, component choices |
| **Trade-offs** | Did you consider alternatives? Can you justify your choices? |
| **Depth** | Can you go deep when prompted on a specific area? |
| **Communication** | Clear structure, logical flow, good use of time |

You don’t need to be perfect in every area. Strong performance in 3–4 dimensions with reasonable coverage in the rest is usually sufficient. Prioritize **clarity** and **structure** over trying to cover every detail.

---

## Additional Resources

- **Question bank**: Use `05-question-bank.md` for 25 practice questions by category
- **Mock walkthroughs**: Use `06-mock-interview-walkthroughs.md` for full 45-minute exemplars
- **Technical deep dives**: Refer to other sections (embeddings, feature stores, training, serving) for component-level detail
- **Mock interview platforms**: Pramp, Interviewing.io, or peers for timed practice

---

## Common Question Categories

Most ML system design prompts fall into:

- **Recommendation/ranking**: Feed, products, content, connections
- **Search**: Query understanding, retrieval, ranking
- **Prediction**: CTR, ETA, demand forecasting
- **Classification**: Moderation, spam, abuse detection
- **Embedding/similarity**: Image search, semantic search, PYMK

Prepare at least one full design per category.

---

## Final Tips

- **Don’t memorize** — Internalize the framework and adapt it to each problem
- **Listen to signals** — If the interviewer probes an area, go deeper there
- **Think aloud** — Share your reasoning; interviewers score your process, not just the final answer
- **It’s OK to correct yourself** — Say “Actually, I’d reconsider…” and adjust
- **Stay calm** — 45 minutes is tight; focus on structure and coverage over perfection

---

*Good luck with your preparation. The framework and practice matter more than memorization.*
