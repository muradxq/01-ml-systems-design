# Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation grounds LLM outputs in retrieved documents, reducing hallucination, enabling use of private or stale knowledge, and supporting domain-specific applications. This document covers RAG architecture, document processing, embedding models, retrieval strategies, generation patterns, evaluation, hallucination mitigation, and advanced techniques like multi-hop RAG.

---

## 1. Why RAG?

### Problems RAG Addresses

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM Limitations → RAG Solutions                                │
│                                                                  │
│  Hallucination     → Ground in retrieved documents              │
│  Stale knowledge   → Index updated docs, retrieve at query time  │
│  No private data   → Index internal docs, retrieve privately     │
│  Domain-specific   → Index domain corpus, inject at inference    │
│  Citations needed  → Return source chunks with generated answer  │
└─────────────────────────────────────────────────────────────────┘
```

| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| "What's our Q4 revenue?" | Hallucination or generic answer | Retrieve from internal docs, answer with citation |
| "Latest Python 3.12 features" | Cutoff at training date | Index recent docs, retrieve |
| Medical/legal domain | Generic, possibly wrong | Domain index + retrieval |

### Decision: RAG vs Fine-Tuning vs Prompt Engineering

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **RAG** | Need external/private data, citations | No retrain, updatable, citable | Retrieval quality critical |
| **Fine-tuning** | Model behavior/style change | Better task fit | Expensive, can forget, needs data |
| **Prompt engineering** | Simple tasks, public knowledge | Cheap, fast | Limited by context, no private data |

---

## 2. RAG Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  RAG SYSTEM ARCHITECTURE                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  OFFLINE (Indexing Pipeline)                    ONLINE (Query-Time)
  ──────────────────────────                    ─────────────────

  Documents ──▶ Chunking ──▶ Embedding ──▶ Vector Store (Index)
       │           │              │                    │
       │           │              │                    │
       │           │              │         Query ──▶  │
       │           │              │              │     │
       │           │              │              │     ▼
       │           │              │         Embed query ──▶ Retrieve top-K
       │           │              │                    │
       │           │              │                    ▼
       │           │              │              Rerank (optional)
       │           │              │                    │
       │           │              │                    ▼
       │           │              │         Build prompt: System + Context + Query
       │           │              │                    │
       │           │              │                    ▼
       │           │              └──────────▶  LLM Generation
       │           │                                    │
       │           │                                    ▼
       │           │                            Post-process: Parse, cite
       │           │                                    │
       │           │                                    ▼
       │           └───────────────────────────▶  Response + Citations
```

### Components

1. **Indexing pipeline:** Chunk documents → embed → store in vector DB
2. **Retrieval:** Embed query → search index → optional rerank
3. **Generation:** Inject context into prompt → LLM generates → extract citations

---

## 3. Document Processing

### Chunking Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Fixed-size** | Split by char/token count (e.g., 512 tokens, 50% overlap) | Simple, predictable | Can cut sentences, lose structure |
| **Sentence** | Split on sentence boundaries | Clean boundaries | Variable chunk sizes |
| **Paragraph** | Split on paragraphs | Preserves topical coherence | May be too large/small |
| **Semantic** | Group by embedding similarity | Thematically coherent | Complex, needs tuning |
| **Recursive** | Hierarchical: try paragraph, then sentence, then char | Good default | More logic |

### Chunk Size Trade-offs

```
┌─────────────────────────────────────────────────────────────────┐
│  Chunk Size Trade-offs                                            │
│                                                                  │
│  Small (128-256 tokens):                                         │
│  + Precise retrieval, less noise                                 │
│  - May miss context, more chunks to retrieve                      │
│                                                                  │
│  Medium (512 tokens):  ← Common default                          │
│  + Balance of context and precision                              │
│  - May include irrelevant parts                                  │
│                                                                  │
│  Large (1024+ tokens):                                           │
│  + Richer context per chunk                                      │
│  - Noisier, hits context limit faster                            │
└─────────────────────────────────────────────────────────────────┘
```

### Overlap

- **Overlap (e.g., 50-100 tokens):** Reduces boundary effects; adjacent chunks share content
- **Trade-off:** More storage, some redundancy; usually worth it for quality

---

## 4. Embedding Models for Retrieval

### Popular Models (2024–2025)

| Model | Dimensions | Best For | Notes |
|-------|------------|----------|-------|
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | General, fast | Lightweight |
| **BAAI/bge-large-en-v1.5** | 1024 | General, quality | Strong retrieval |
| **OpenAI text-embedding-3** | 1536 | API-based | Good quality, pay per use |
| **Cohere embed-v3** | 1024 | Multilingual, hybrid | Good for hybrid search |
| **Voyage-2** | 1024 | Domain-tuned | Strong for niche tasks |

### Domain-Specific Embeddings

- Fine-tune on domain pairs (query, relevant doc) for better retrieval
- Use contrastive loss: (q, d+) close, (q, d-) far
- Improves recall 10–30% on domain data

---

## 5. Retrieval

### Vector Search (Dense)

- Embed query and chunks; retrieve by cosine similarity or inner product
- Approximate Nearest Neighbor (ANN): FAISS, HNSW, IVF for scale
- Typical: Top-K (5–20) chunks

### BM25 (Sparse, Lexical)

- TF-IDF–style; matches keywords, good for exact terms, entities
- Fails on paraphrasing, synonyms
- Hybrid: Combine dense + BM25 scores (e.g., 0.7 × dense + 0.3 × BM25)

### Hybrid: Vector + BM25

```
┌─────────────────────────────────────────────────────────────────┐
│  Hybrid Retrieval                                                 │
│                                                                  │
│  Query: "Python async programming"                               │
│                                                                  │
│  Dense: Finds "asyncio", "await", "coroutines" (semantic)        │
│  BM25: Finds exact "Python", "async" (lexical)                   │
│                                                                  │
│  Reciprocal Rank Fusion (RRF):                                   │
│  score(d) = Σ 1/(k + rank_i(d))  for each retrieval i           │
│  k=60 typical                                                    │
│                                                                  │
│  → Best of both: semantic + keyword                              │
└─────────────────────────────────────────────────────────────────┘
```

### Reranking (Cross-Encoder)

- First stage: Retrieve top-50 or top-100 with cheap vector/BM25
- Second stage: Rerank with cross-encoder (query + chunk) for top-5 to top-10
- Models: BAAI/bge-reranker, Cohere rerank
- Improves precision; adds ~50–200ms per query

---

## 6. Generation: Context Injection & Prompt Construction

### Prompt Template

```
System: You are a helpful assistant. Answer based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{retrieved_chunk_1}

{retrieved_chunk_2}

{retrieved_chunk_3}

Question: {user_query}

Answer:
```

### Citation / Grounding

- Instruct model to cite: "Answer with [Source N] references."
- Parse output for `[1]`, `[2]` and map to chunks
- Optional: Verify that cited chunks support each claim

### Truncation

- Context window limited (e.g., 4K–128K tokens)
- Prioritize highest-ranked chunks; truncate if needed
- Consider sliding window over long documents for long-context models

---

## 7. Evaluation

### Retrieval Quality

| Metric | Definition |
|--------|------------|
| **Recall@K** | % of relevant docs in top-K |
| **MRR** | Mean Reciprocal Rank: 1/rank of first relevant |
| **NDCG** | Normalized Discounted Cumulative Gain (ranking quality) |
| **Hit rate** | % queries with ≥1 relevant in top-K |

### Generation Quality

| Metric | Definition |
|--------|------------|
| **Faithfulness** | Does the answer stay within the context? (hallucination check) |
| **Relevance** | Does the answer address the question? |
| **Citation precision** | Are citations correct? |
| **LLM-as-judge** | Use an LLM to score faithfulness/relevance (cheaper than human) |

### Synthetic Eval

- Generate (question, answer) from docs; use question to retrieve; check if answer is in retrieved set
- Scales evaluation without human labels

---

## 8. Hallucination Mitigation Strategies

| Strategy | How |
|----------|-----|
| **Strong retrieval** | More relevant chunks → less model invention |
| **Explicit instructions** | "Answer only from context"; "Say I don't know if unsure" |
| **Citation** | Force citations; verify support |
| **Reranking** | Fewer irrelevant chunks in context |
| **Verification** | NLI or LLM check: does context entail the claim? |
| **Multiple retrievals** | Retrieve with different strategies; compare consistency |
| **Reduce generation freedom** | Lower temperature, top_p; max_tokens |

---

## 9. Python Code: End-to-End RAG Pipeline

```python
# rag_pipeline_example.py
"""
End-to-end RAG pipeline (LangChain-style structure)
Uses: pip install sentence-transformers faiss-cpu langchain openai
"""

from typing import List, Optional
from dataclasses import dataclass
import hashlib

# --- Embedding ---
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

# --- Chunking ---
def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]:
    """Recursive chunking: try separator, then fall back to char-level."""
    if not text:
        return []
    
    parts = text.split(separator)
    chunks = []
    current = ""
    
    for part in parts:
        if len(current) + len(part) < chunk_size:
            current += (separator if current else "") + part
        else:
            if current:
                chunks.append(current.strip())
            # Handle long part
            if len(part) > chunk_size:
                for i in range(0, len(part), chunk_size - overlap):
                    chunks.append(part[i:i + chunk_size])
                current = ""
            else:
                current = part
    if current:
        chunks.append(current.strip())
    return chunks

# --- Index ---
import faiss
import numpy as np
import json
from pathlib import Path

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vecs
        self.chunks: List[Chunk] = []
    
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        # Normalize for cosine sim via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        query_embedding = query_embedding.reshape(1, -1)
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / norms
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i < len(self.chunks)]

# --- RAG Pipeline ---
class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        llm_client=None,  # OpenAI, Anthropic, or local
    ):
        self.embedder = embedder
        self.store = vector_store
        self.llm = llm_client
    
    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        q_emb = np.array(self.embedder.embed_single(query), dtype=np.float32)
        results = self.store.search(q_emb, k=k)
        return [r[0] for r in results]
    
    def build_prompt(self, query: str, chunks: List[Chunk]) -> str:
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(chunks)])
        return f"""Answer based only on the context below. Cite sources with [N].
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
    
    def generate(self, prompt: str) -> str:
        if self.llm is None:
            return "[LLM not configured - install openai and set API key]"
        # Example OpenAI call
        resp = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content
    
    def run(self, query: str, k: int = 5) -> str:
        chunks = self.retrieve(query, k=k)
        prompt = self.build_prompt(query, chunks)
        return self.generate(prompt)

# --- Indexing ---
def index_documents(docs: List[str], embedder: Embedder, store: VectorStore):
    all_chunks = []
    for i, doc in enumerate(docs):
        for j, chunk_text in enumerate(chunk_text(doc)):
            chunk_id = hashlib.sha256(f"{i}_{j}_{chunk_text[:100]}".encode()).hexdigest()[:16]
            all_chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata={"doc_id": i}))
    
    texts = [c.text for c in all_chunks]
    embeddings = np.array(embedder.embed(texts), dtype=np.float32)
    store.add(embeddings, all_chunks)

# Usage
if __name__ == "__main__":
    embedder = Embedder()
    store = VectorStore(dimension=384)
    docs = [
        "Python asyncio allows concurrent I/O. Use async/await for coroutines.",
        "Machine learning pipelines include data, training, and serving stages.",
    ]
    index_documents(docs, embedder, store)
    pipeline = RAGPipeline(embedder, store)
    answer = pipeline.run("How does Python handle async?")
    print(answer)
```

---

## 10. Advanced: Multi-Hop RAG, Iterative Retrieval, Query Decomposition

### Multi-Hop RAG

- Answer requires multiple retrieval steps (e.g., "Compare X in doc A and doc B")
- Flow: Retrieve → generate partial answer or sub-query → retrieve again → final answer
- Implement via agent loop or predefined sub-queries

### Iterative Retrieval

- Retrieve initial set → generate → check if sufficient → if not, retrieve more (e.g., with refined query)
- Improves recall for complex questions
- Cost: More retrieval + LLM calls

### Query Decomposition

- Split complex query into sub-queries
- Example: "What's the revenue of X and how does it compare to Y?"
  - Sub1: "Revenue of X"
  - Sub2: "Revenue of Y"
- Retrieve for each → combine context → generate
- Improves retrieval precision for each sub-question

```
┌─────────────────────────────────────────────────────────────────┐
│  Query Decomposition Pipeline                                    │
│                                                                  │
│  "Compare Q4 revenue of Product A and Product B"                  │
│           │                                                       │
│           ▼                                                       │
│  Decomposer LLM:                                                 │
│  - "Q4 revenue Product A"                                        │
│  - "Q4 revenue Product B"                                        │
│           │                                                       │
│           ▼                                                       │
│  Parallel retrieval for each                                     │
│           │                                                       │
│           ▼                                                       │
│  Merge chunks, deduplicate                                        │
│           │                                                       │
│           ▼                                                       │
│  Generate with full context                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Trade-offs Table

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| Chunk size | 256 | 512 | Smaller: precise, more retrievals. Larger: more context, noisier |
| Retrieval | Dense only | Hybrid (dense+BM25) | Hybrid better for exact terms; slightly more compute |
| Reranking | None | Cross-encoder | Rerank: better precision, +50–200ms latency |
| Embedding | General | Domain-tuned | Domain: better recall, needs labeled data |
| Multi-hop | Single | Iterative | Iterative: better for complex Q, higher cost |

---

## 12. Interview Tips

1. **Why RAG?** "Reduces hallucination, uses private/stale data, enables citations."
2. **Chunking:** "512 tokens with 50 overlap is a common default. Smaller for precision, larger for context."
3. **Hybrid retrieval:** "BM25 for keywords and entities, dense for semantics. Combine with RRF."
4. **Reranking:** "Retrieve 50 with cheap model, rerank to top-5 with cross-encoder."
5. **Evaluation:** "Recall@K for retrieval; faithfulness and relevance for generation. Use LLM-as-judge for scale."
6. **Hallucination:** "Strong retrieval, explicit 'answer only from context', citations, verification."

---

## 13. Related Topics

- [01 - LLM Serving Infrastructure](./01-llm-serving-infrastructure.md) – Serving the generator
- [03 - Fine-Tuning & Alignment](./03-fine-tuning-alignment.md) – When to use RAG vs fine-tuning
- [04 - Cost & Latency Optimization](./04-cost-latency-optimization.md) – Caching, routing
- [10-end-to-end-systems/05-nlp-systems.md](../../phase-4-end-to-end-systems/10-end-to-end-systems/05-nlp-systems.md) – NLP system design
