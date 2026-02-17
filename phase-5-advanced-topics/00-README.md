# Phase 5: Advanced Topics

These topics differentiate strong candidates from average ones. Embeddings and retrieval underpin ~80% of modern ML systems. LLM systems are increasingly asked since 2024. Fairness is a known pass/fail differentiator. Experimentation and capacity planning are expected at senior levels.

---

## Chapters

11. [Embeddings & Retrieval](./11-embeddings-retrieval/00-README.md)
    - [Embedding Fundamentals](./11-embeddings-retrieval/01-embedding-fundamentals.md) - Word2Vec, Item2Vec, contrastive learning
    - [Approximate Nearest Neighbors](./11-embeddings-retrieval/02-approximate-nearest-neighbors.md) - HNSW, IVF, PQ, ScaNN, FAISS
    - [Vector Databases](./11-embeddings-retrieval/03-vector-databases.md) - Pinecone, Milvus, Weaviate, pgvector
    - [Two-Tower Architecture](./11-embeddings-retrieval/04-two-tower-architecture.md) - User/item towers, negative sampling, serving patterns

12. [LLM & GenAI Systems](./12-llm-genai-systems/00-README.md)
    - [LLM Serving Infrastructure](./12-llm-genai-systems/01-llm-serving-infrastructure.md) - KV cache, continuous batching, vLLM, GPU management
    - [Retrieval-Augmented Generation](./12-llm-genai-systems/02-retrieval-augmented-generation.md) - RAG architecture, chunking, hallucination mitigation
    - [Fine-Tuning & Alignment](./12-llm-genai-systems/03-fine-tuning-alignment.md) - LoRA, RLHF, DPO, evaluation
    - [Cost & Latency Optimization](./12-llm-genai-systems/04-cost-latency-optimization.md) - Quantization, distillation, model routing, caching

13. [Fairness & Responsible AI](./13-fairness-responsible-ai/00-README.md)
    - [Bias Detection](./13-fairness-responsible-ai/01-bias-detection.md) - Bias types, fairness metrics, sliced evaluation
    - [Fairness Constraints](./13-fairness-responsible-ai/02-fairness-constraints.md) - Pre/in/post-processing interventions
    - [Model Auditing](./13-fairness-responsible-ai/03-model-auditing.md) - Model cards, SHAP, LIME, regulatory requirements
    - [Responsible Deployment](./13-fairness-responsible-ai/04-responsible-deployment.md) - Feedback loops, human-in-the-loop, red teaming

14. [Online Experimentation](./14-online-experimentation/00-README.md)
    - [Experiment Design](./14-online-experimentation/01-experiment-design.md) - Hypothesis, randomization, sample size, guardrails
    - [Advanced Testing](./14-online-experimentation/02-advanced-testing.md) - Multi-armed bandits, interleaving, switchback experiments
    - [Metric Design](./14-online-experimentation/03-metric-design.md) - North star, proxy, counter metrics, CUPED, OEC
    - [Analysis Pitfalls](./14-online-experimentation/04-analysis-pitfalls.md) - Simpson's paradox, peeking, multiple testing

15. [Capacity & Cost Planning](./15-capacity-cost-planning/00-README.md)
    - [Back-of-Envelope Estimation](./15-capacity-cost-planning/01-back-of-envelope-estimation.md) - QPS, storage, bandwidth worked examples
    - [Cost Modeling](./15-capacity-cost-planning/02-cost-modeling.md) - GPU costs, TCO, build vs buy
    - [Capacity Planning](./15-capacity-cost-planning/03-capacity-planning.md) - Peak traffic, autoscaling, multi-region

---

## What You'll Learn

- How embeddings and ANN search power recommendations, search, and ads
- How to design, serve, and optimize LLM-based systems
- How to detect bias, enforce fairness, and audit models
- Advanced A/B testing: bandits, interleaving, metric design
- Back-of-envelope estimation for ML system interviews

---

## Next Phase

Continue to [Phase 6: Interview Preparation](../phase-6-interview-preparation/00-README.md) for frameworks, company guides, question banks, and mock walkthroughs.
