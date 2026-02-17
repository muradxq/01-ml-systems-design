# Phase 4: End-to-End Systems

This is the heart of ML system design interview preparation. Each system design demonstrates how data management, feature engineering, model training, serving, and monitoring come together in production. These 15 designs cover the most commonly asked questions at Meta, Google, and other top tech companies.

---

## Systems

| # | System | Key ML Task | Commonly Asked At |
|---|--------|-------------|-------------------|
| 1 | [Recommendation Systems](./10-end-to-end-systems/01-recommendation-systems.md) | Ranking, collaborative filtering | Everywhere |
| 2 | [Search Systems](./10-end-to-end-systems/02-search-systems.md) | Information retrieval, ranking | Google, Amazon |
| 3 | [Fraud Detection](./10-end-to-end-systems/03-fraud-detection.md) | Real-time classification | Both |
| 4 | [Computer Vision Systems](./10-end-to-end-systems/04-computer-vision-systems.md) | Image classification, detection | Both |
| 5 | [NLP Systems](./10-end-to-end-systems/05-nlp-systems.md) | Text classification, generation | Both |
| 6 | [Time Series Forecasting](./10-end-to-end-systems/06-time-series-forecasting.md) | Regression, sequence modeling | Both |
| 7 | [Ad Click Prediction](./10-end-to-end-systems/07-ad-click-prediction.md) | Calibrated classification | **Meta #1** |
| 8 | [Feed Ranking](./10-end-to-end-systems/08-feed-ranking.md) | Multi-objective ranking | **Meta** |
| 9 | [Content Moderation](./10-end-to-end-systems/09-content-moderation.md) | Multi-modal classification | **Meta** |
| 10 | [People You May Know](./10-end-to-end-systems/10-people-you-may-know.md) | Graph-based ranking | **Meta** |
| 11 | [Autocomplete & Typeahead](./10-end-to-end-systems/11-autocomplete-typeahead.md) | Language model ranking | **Google** |
| 12 | [Notification Ranking](./10-end-to-end-systems/12-notification-ranking.md) | Multi-channel ranking | Both |
| 13 | [Chatbot / LLM System](./10-end-to-end-systems/13-chatbot-llm-system.md) | RAG, dialog management | Both |
| 14 | [Video Recommendation](./10-end-to-end-systems/14-video-recommendation.md) | Watch-time prediction | **Google** |
| 15 | [Entity Resolution](./10-end-to-end-systems/15-entity-resolution.md) | Deduplication at scale | Both |

---

## How to Study These

1. **Read the problem definition** and try to design the system yourself in 45 minutes before reading the solution
2. **Focus on the architecture** -- understand the multi-stage pipeline pattern (candidate generation -> ranking -> re-ranking)
3. **Study the trade-offs** -- interviewers care more about your reasoning than the specific choice
4. **Practice the interview tips** at the end of each design

---

## Next Phase

Continue to [Phase 5: Advanced Topics](../phase-5-advanced-topics/00-README.md) for embeddings, LLMs, fairness, experimentation, and capacity planning.
