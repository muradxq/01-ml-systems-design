# Search Systems

## Overview

ML-powered search systems combine traditional information retrieval with machine learning to provide relevant, personalized search results. Modern search goes beyond keyword matching to understand user intent, semantic meaning, and context. Search is a critical component of e-commerce, content platforms, and enterprise applications.

---

## 🎯 Problem Definition

### Business Goals
- Help users find what they're looking for quickly
- Increase conversion (purchase, engagement)
- Reduce search abandonment
- Surface relevant content/products
- Handle diverse queries (navigational, informational, transactional)

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Latency** | < 200ms p99 |
| **Throughput** | 10K-100K QPS |
| **Index Size** | Millions to billions of documents |
| **Freshness** | New documents indexed within minutes |
| **Relevance** | High precision in top results |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Search System Architecture                          │
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Search Query  │                                                          │
│  └───────┬────────┘                                                          │
│          │                                                                    │
│          ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    Query Processing Layer                        │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │         │
│  │  │ Spelling │──│ Query    │──│ Intent   │──│ Query    │       │         │
│  │  │ Correct  │  │ Expansion│  │ Class.   │  │ Rewrite  │       │         │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │         │
│  └───────────────────────────┬────────────────────────────────────┘         │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    Retrieval Layer                               │         │
│  │  ┌──────────────────┐      ┌──────────────────┐                │         │
│  │  │  Lexical Search  │      │  Semantic Search │                │         │
│  │  │  (Inverted Index)│      │  (Vector Search) │                │         │
│  │  │  BM25, TF-IDF    │      │  Dense Retrieval │                │         │
│  │  └────────┬─────────┘      └────────┬─────────┘                │         │
│  │           └──────────┬──────────────┘                          │         │
│  │                      ▼                                          │         │
│  │              Candidate Fusion                                   │         │
│  └───────────────────────────┬────────────────────────────────────┘         │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    Ranking Layer                                 │         │
│  │  ┌──────────────────┐      ┌──────────────────┐                │         │
│  │  │  L1: Lightweight │ ───▶ │  L2: Full Model  │                │         │
│  │  │  Scoring         │      │  (LTR/BERT)      │                │         │
│  │  │  1000 → 100      │      │  100 → 20        │                │         │
│  │  └──────────────────┘      └──────────────────┘                │         │
│  └───────────────────────────┬────────────────────────────────────┘         │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    Post-Processing Layer                         │         │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                │         │
│  │  │Personalize │──│ Diversity  │──│ Business   │──▶ Results     │         │
│  │  │            │  │            │  │ Rules      │                │         │
│  │  └────────────┘  └────────────┘  └────────────┘                │         │
│  └────────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Query Processing

```python
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class ProcessedQuery:
    """Processed search query with all enhancements."""
    original_query: str
    corrected_query: str
    expanded_terms: List[str]
    intent: str
    entities: Dict[str, str]
    filters: Dict[str, any]

class QueryProcessor:
    """Process and enhance search queries."""
    
    def __init__(
        self,
        spell_checker,
        query_expander,
        intent_classifier,
        entity_extractor
    ):
        self.spell_checker = spell_checker
        self.expander = query_expander
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
    
    def process(self, query: str, user_context: Dict = None) -> ProcessedQuery:
        """Process query through all stages."""
        
        # Clean query
        cleaned = self._clean_query(query)
        
        # Spell correction
        corrected = self.spell_checker.correct(cleaned)
        
        # Intent classification
        intent = self.intent_classifier.classify(corrected)
        
        # Entity extraction
        entities = self.entity_extractor.extract(corrected)
        
        # Query expansion
        expanded = self.expander.expand(corrected, intent)
        
        # Extract filters from query
        filters = self._extract_filters(corrected, entities)
        
        return ProcessedQuery(
            original_query=query,
            corrected_query=corrected,
            expanded_terms=expanded,
            intent=intent,
            entities=entities,
            filters=filters
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Lowercase
        query = query.lower()
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        # Remove special characters (keep some)
        query = re.sub(r'[^\w\s\-\+\"]', '', query)
        return query
    
    def _extract_filters(
        self,
        query: str,
        entities: Dict
    ) -> Dict:
        """Extract filters from query entities."""
        filters = {}
        
        # Price filter
        if "price" in entities:
            filters["price_range"] = self._parse_price(entities["price"])
        
        # Category filter
        if "category" in entities:
            filters["category"] = entities["category"]
        
        # Brand filter
        if "brand" in entities:
            filters["brand"] = entities["brand"]
        
        return filters
    
    def _parse_price(self, price_text: str) -> Tuple[float, float]:
        """Parse price range from text."""
        # "under $50" -> (0, 50)
        # "$50-$100" -> (50, 100)
        # Implementation
        pass

class SpellChecker:
    """Spell correction for search queries."""
    
    def __init__(self, vocabulary: set, edit_distance_model):
        self.vocabulary = vocabulary
        self.model = edit_distance_model
    
    def correct(self, query: str) -> str:
        """Correct spelling errors in query."""
        words = query.split()
        corrected = []
        
        for word in words:
            if word in self.vocabulary:
                corrected.append(word)
            else:
                # Find closest word in vocabulary
                candidates = self._get_candidates(word)
                if candidates:
                    corrected.append(candidates[0])
                else:
                    corrected.append(word)
        
        return ' '.join(corrected)
    
    def _get_candidates(self, word: str, max_distance: int = 2) -> List[str]:
        """Get correction candidates."""
        candidates = []
        for vocab_word in self.vocabulary:
            distance = self._edit_distance(word, vocab_word)
            if distance <= max_distance:
                candidates.append((vocab_word, distance))
        
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates[:5]]

class IntentClassifier:
    """Classify search intent."""
    
    INTENTS = [
        "navigational",  # Looking for specific page/brand
        "informational", # Looking for information
        "transactional", # Ready to purchase
        "comparison",    # Comparing options
        "local"          # Location-based
    ]
    
    def __init__(self, model):
        self.model = model
    
    def classify(self, query: str) -> str:
        """Classify query intent."""
        # Features
        features = self._extract_features(query)
        
        # Predict
        intent = self.model.predict([features])[0]
        return intent
    
    def _extract_features(self, query: str) -> List[float]:
        """Extract features for intent classification."""
        features = []
        
        # Query length
        features.append(len(query.split()))
        
        # Contains brand name
        features.append(int(self._contains_brand(query)))
        
        # Contains price terms
        features.append(int(any(w in query for w in ['cheap', 'price', '$', 'under', 'budget'])))
        
        # Contains comparison terms
        features.append(int(any(w in query for w in ['vs', 'compare', 'best', 'top'])))
        
        # Contains action terms
        features.append(int(any(w in query for w in ['buy', 'purchase', 'order', 'get'])))
        
        return features
```

### 2. Retrieval (Candidate Generation)

```python
from typing import List, Tuple
import numpy as np
from elasticsearch import Elasticsearch

class HybridRetriever:
    """Hybrid retrieval combining lexical and semantic search."""
    
    def __init__(
        self,
        es_client: Elasticsearch,
        vector_index,
        embedding_model,
        lexical_weight: float = 0.5
    ):
        self.es = es_client
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.lexical_weight = lexical_weight
    
    def retrieve(
        self,
        query: ProcessedQuery,
        n: int = 1000,
        filters: Dict = None
    ) -> List[Tuple[str, float]]:
        """Retrieve candidates using hybrid approach."""
        
        # Lexical retrieval (BM25)
        lexical_results = self._lexical_search(
            query=query.corrected_query,
            expanded_terms=query.expanded_terms,
            filters=filters,
            n=n
        )
        
        # Semantic retrieval (Dense vectors)
        semantic_results = self._semantic_search(
            query=query.corrected_query,
            filters=filters,
            n=n
        )
        
        # Fusion
        fused = self._reciprocal_rank_fusion(
            lexical_results,
            semantic_results,
            weights=[self.lexical_weight, 1 - self.lexical_weight]
        )
        
        return fused[:n]
    
    def _lexical_search(
        self,
        query: str,
        expanded_terms: List[str],
        filters: Dict,
        n: int
    ) -> List[Tuple[str, float]]:
        """BM25-based lexical search."""
        
        # Build query
        should_clauses = [
            {"match": {"title": {"query": query, "boost": 2.0}}},
            {"match": {"description": {"query": query}}}
        ]
        
        # Add expanded terms
        for term in expanded_terms:
            should_clauses.append({
                "match": {"title": {"query": term, "boost": 0.5}}
            })
        
        # Build filter clauses
        filter_clauses = []
        if filters:
            if "category" in filters:
                filter_clauses.append({"term": {"category": filters["category"]}})
            if "brand" in filters:
                filter_clauses.append({"term": {"brand": filters["brand"]}})
            if "price_range" in filters:
                filter_clauses.append({
                    "range": {
                        "price": {
                            "gte": filters["price_range"][0],
                            "lte": filters["price_range"][1]
                        }
                    }
                })
        
        # Execute search
        body = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "filter": filter_clauses,
                    "minimum_should_match": 1
                }
            },
            "size": n
        }
        
        response = self.es.search(index="products", body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append((hit["_id"], hit["_score"]))
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        filters: Dict,
        n: int
    ) -> List[Tuple[str, float]]:
        """Dense vector search."""
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Build filter for vector search
        filter_expr = None
        if filters:
            filter_conditions = []
            if "category" in filters:
                filter_conditions.append(f"category == '{filters['category']}'")
            if "price_range" in filters:
                filter_conditions.append(
                    f"price >= {filters['price_range'][0]} && price <= {filters['price_range'][1]}"
                )
            if filter_conditions:
                filter_expr = " && ".join(filter_conditions)
        
        # Search vector index
        results = self.vector_index.search(
            query_embedding,
            top_k=n,
            filter=filter_expr
        )
        
        return [(r.id, r.score) for r in results]
    
    def _reciprocal_rank_fusion(
        self,
        *result_lists: List[List[Tuple[str, float]]],
        weights: List[float] = None,
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """Fuse multiple result lists using RRF."""
        
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        scores = {}
        
        for results, weight in zip(result_lists, weights):
            for rank, (doc_id, _) in enumerate(results):
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += weight / (k + rank + 1)
        
        # Sort by fused score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
```

### 3. Ranking Model

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CrossEncoderRanker(nn.Module):
    """BERT-based cross-encoder for re-ranking."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, query: str, documents: List[str]) -> torch.Tensor:
        """Score query-document pairs."""
        
        # Prepare inputs
        pairs = [[query, doc] for doc in documents]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Forward pass
        outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        scores = self.classifier(cls_embeddings).squeeze(-1)
        
        return scores

class LightweightRanker:
    """Fast lightweight ranker for initial scoring."""
    
    def __init__(self, model):
        self.model = model  # Gradient boosted tree or similar
    
    def score(
        self,
        query_features: Dict,
        doc_features: List[Dict]
    ) -> List[float]:
        """Score documents with lightweight features."""
        
        features = []
        for doc in doc_features:
            f = self._combine_features(query_features, doc)
            features.append(f)
        
        scores = self.model.predict(features)
        return scores.tolist()
    
    def _combine_features(self, query_features: Dict, doc_features: Dict) -> List[float]:
        """Create feature vector for query-doc pair."""
        features = [
            # Query-document overlap
            doc_features.get("title_match_ratio", 0),
            doc_features.get("description_match_ratio", 0),
            
            # BM25 scores
            doc_features.get("bm25_title", 0),
            doc_features.get("bm25_description", 0),
            
            # Document quality signals
            doc_features.get("click_through_rate", 0),
            doc_features.get("conversion_rate", 0),
            doc_features.get("avg_rating", 0),
            doc_features.get("num_reviews", 0),
            
            # Freshness
            doc_features.get("days_since_update", 0),
            
            # Popularity
            doc_features.get("popularity_score", 0),
        ]
        return features

class TwoStageRanker:
    """Two-stage ranking pipeline."""
    
    def __init__(
        self,
        lightweight_ranker: LightweightRanker,
        cross_encoder: CrossEncoderRanker,
        l1_cutoff: int = 100,
        l2_cutoff: int = 20
    ):
        self.l1_ranker = lightweight_ranker
        self.l2_ranker = cross_encoder
        self.l1_cutoff = l1_cutoff
        self.l2_cutoff = l2_cutoff
    
    def rank(
        self,
        query: str,
        candidates: List[Dict],
        query_features: Dict
    ) -> List[Dict]:
        """Two-stage ranking."""
        
        # Stage 1: Lightweight ranking
        doc_features = [self._extract_doc_features(c) for c in candidates]
        l1_scores = self.l1_ranker.score(query_features, doc_features)
        
        # Sort and cut
        scored = list(zip(candidates, l1_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        l1_results = [c for c, _ in scored[:self.l1_cutoff]]
        
        # Stage 2: Cross-encoder ranking
        documents = [c.get("title", "") + " " + c.get("description", "") for c in l1_results]
        
        with torch.no_grad():
            l2_scores = self.l2_ranker(query, documents)
        
        # Sort by L2 scores
        scored = list(zip(l1_results, l2_scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored[:self.l2_cutoff]]
```

### 4. Complete Search Service

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    page: int = 0
    page_size: int = 20
    filters: Optional[Dict] = None

class SearchResult(BaseModel):
    doc_id: str
    title: str
    description: str
    price: Optional[float]
    score: float
    highlights: Dict[str, List[str]]

class SearchResponse(BaseModel):
    query: str
    corrected_query: Optional[str]
    total_results: int
    results: List[SearchResult]
    facets: Dict[str, List[Dict]]
    latency_ms: float

class SearchService:
    """Complete search service."""
    
    def __init__(
        self,
        query_processor: QueryProcessor,
        retriever: HybridRetriever,
        ranker: TwoStageRanker,
        document_store,
        personalization_service
    ):
        self.query_processor = query_processor
        self.retriever = retriever
        self.ranker = ranker
        self.doc_store = document_store
        self.personalization = personalization_service
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search pipeline."""
        
        start_time = time.time()
        
        # 1. Query Processing
        processed_query = self.query_processor.process(
            query=request.query,
            user_context={"user_id": request.user_id}
        )
        
        # 2. Retrieval
        candidates = self.retriever.retrieve(
            query=processed_query,
            n=1000,
            filters=request.filters or processed_query.filters
        )
        
        # 3. Fetch documents
        doc_ids = [doc_id for doc_id, _ in candidates]
        documents = await self.doc_store.get_many(doc_ids)
        
        # 4. Ranking
        query_features = self._extract_query_features(processed_query)
        ranked = self.ranker.rank(
            query=processed_query.corrected_query,
            candidates=documents,
            query_features=query_features
        )
        
        # 5. Personalization (if user_id provided)
        if request.user_id:
            ranked = await self.personalization.rerank(
                user_id=request.user_id,
                results=ranked
            )
        
        # 6. Pagination
        start_idx = request.page * request.page_size
        end_idx = start_idx + request.page_size
        page_results = ranked[start_idx:end_idx]
        
        # 7. Facets
        facets = self._compute_facets(documents)
        
        # 8. Highlights
        results = []
        for doc in page_results:
            highlights = self._generate_highlights(
                processed_query.corrected_query,
                doc
            )
            results.append(SearchResult(
                doc_id=doc["id"],
                title=doc["title"],
                description=doc["description"],
                price=doc.get("price"),
                score=doc.get("_score", 0),
                highlights=highlights
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            corrected_query=processed_query.corrected_query if processed_query.corrected_query != request.query else None,
            total_results=len(ranked),
            results=results,
            facets=facets,
            latency_ms=latency_ms
        )
    
    def _compute_facets(self, documents: List[Dict]) -> Dict:
        """Compute facets for filtering."""
        facets = {
            "category": {},
            "brand": {},
            "price_range": {}
        }
        
        for doc in documents:
            # Category facet
            cat = doc.get("category")
            if cat:
                facets["category"][cat] = facets["category"].get(cat, 0) + 1
            
            # Brand facet
            brand = doc.get("brand")
            if brand:
                facets["brand"][brand] = facets["brand"].get(brand, 0) + 1
        
        # Convert to list format
        return {
            k: [{"value": key, "count": count} for key, count in sorted(v.items(), key=lambda x: x[1], reverse=True)[:20]]
            for k, v in facets.items()
        }
    
    def _generate_highlights(self, query: str, doc: Dict) -> Dict[str, List[str]]:
        """Generate highlighted snippets."""
        highlights = {}
        query_terms = set(query.lower().split())
        
        for field in ["title", "description"]:
            text = doc.get(field, "")
            if text:
                snippets = []
                sentences = text.split(". ")
                for sentence in sentences:
                    if any(term in sentence.lower() for term in query_terms):
                        # Highlight matching terms
                        for term in query_terms:
                            sentence = sentence.replace(term, f"<em>{term}</em>")
                        snippets.append(sentence)
                        if len(snippets) >= 2:
                            break
                if snippets:
                    highlights[field] = snippets
        
        return highlights

# API Endpoint
search_service = SearchService(...)

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    return await search_service.search(request)
```

---

## 📈 Metrics & Evaluation

### Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CTR** | Clicks / Impressions | > 10% |
| **Zero Results Rate** | Queries with no results | < 5% |
| **Mean Reciprocal Rank** | 1/rank of first click | > 0.5 |
| **Time to First Click** | Time to first result click | < 10s |
| **Reformulation Rate** | Users who modify query | < 20% |

### Offline Metrics

```python
def evaluate_search(
    queries: List[str],
    relevance_labels: Dict[str, Dict[str, int]],  # query -> {doc_id: relevance}
    system_rankings: Dict[str, List[str]],  # query -> [doc_ids]
    k: int = 10
) -> Dict[str, float]:
    """Evaluate search system."""
    
    metrics = {
        "ndcg@k": [],
        "map@k": [],
        "mrr": [],
        "precision@k": []
    }
    
    for query in queries:
        relevance = relevance_labels.get(query, {})
        ranking = system_rankings.get(query, [])[:k]
        
        # NDCG@K
        dcg = sum(
            relevance.get(doc, 0) / np.log2(i + 2)
            for i, doc in enumerate(ranking)
        )
        ideal_ranking = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_ranking)
        )
        metrics["ndcg@k"].append(dcg / idcg if idcg > 0 else 0)
        
        # MRR
        for i, doc in enumerate(ranking):
            if relevance.get(doc, 0) > 0:
                metrics["mrr"].append(1 / (i + 1))
                break
        else:
            metrics["mrr"].append(0)
        
        # Precision@K
        relevant_in_top_k = sum(1 for doc in ranking if relevance.get(doc, 0) > 0)
        metrics["precision@k"].append(relevant_in_top_k / k)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Lexical vs Semantic** | BM25 (fast, interpretable) | Neural (better understanding) |
| **Index Freshness** | Real-time (complex) | Batch (simpler, delayed) |
| **Ranker Complexity** | Simple LTR (fast) | BERT cross-encoder (accurate) |
| **Personalization** | User history (better) | Anonymous (privacy) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How would you handle typos and synonyms?
2. How do you balance relevance and freshness?
3. How would you personalize search results?
4. How do you evaluate search quality?
5. How would you handle 10M QPS?

**Key Points:**
- Query understanding is critical
- Hybrid retrieval (lexical + semantic) works best
- Two-stage ranking for efficiency
- Relevance feedback for improvement
- A/B testing for optimization

---

## 🔗 Related Topics

- [NLP Systems](./05-nlp-systems.md)
- [Recommendation Systems](./01-recommendation-systems.md)
- [Caching Strategies](../../phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md)
