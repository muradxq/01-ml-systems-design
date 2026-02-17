# Autocomplete & Typeahead System

## Overview

Autocomplete (also called typeahead, search-as-you-type, or query suggestions) is one of the **most frequently asked Google ML system design questions**. The system predicts and suggests query completions as users type, reducing typing effort, guiding users to successful queries, and improving search experience. At Google scale, autocomplete serves **billions of queries per day** with strict latency budgets (<50ms) across hundreds of languages. A well-designed autocomplete system balances relevance, personalization, novelty, and safety while maintaining sub-50ms response times.

---

## 🎯 Problem Definition

### Business Goals

- **Reduce typing effort:** Minimize keystrokes users need to reach their intended query (target: 40-50% keystroke reduction)
- **Guide users to successful queries:** Surface completions that lead to clicks and satisfied search sessions
- **Reduce search latency:** Faster query entry translates to faster overall search experience
- **Increase query diversity:** Help users discover popular or trending queries they might not have typed
- **Monetization (where applicable):** Surface sponsored suggestions in some contexts

### Requirements

| Requirement | Specification | Scale Context |
|-------------|---------------|---------------|
| **Latency** | < 50ms p99 | Per-keystroke; must feel instantaneous |
| **Throughput** | Billions of queries/day | 100K-1M+ QPS at peak |
| **Personalization** | User-specific suggestions | 2B+ users, privacy-preserving |
| **Multi-language** | 100+ languages | Tokenization, transliteration, code-switching |
| **Freshness** | Trending queries within minutes | Real-time popularity signals |
| **Safety** | No offensive/prohibited content | Policy violations, regional differences |
| **Scale** | Millions of distinct queries | Query corpus, trie size |
| **Availability** | 99.99% uptime | Critical for search funnel |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        Autocomplete & Typeahead System                                    │
│                                                                                          │
│  ┌────────────────┐                                                                      │
│  │  User Types    │  Each keystroke triggers request                                     │
│  │  "how to coo"  │  Debouncing: typically 50-100ms delay                                │
│  └───────┬────────┘                                                                      │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  API GATEWAY (2-5ms)                                                          │       │
│  │  Rate limiting | Authentication | Request validation | Load balancing         │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  QUERY PROCESSING (5-10ms)                                                    │       │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │       │
│  │  │ Normalization   │  │ Spell           │  │ Tokenization    │              │       │
│  │  │ (lowercase,     │  │ Correction      │  │ (lang-specific, │              │       │
│  │  │  trim, unicode) │  │ (did you mean)  │  │  compound words)│              │       │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘              │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  CANDIDATE GENERATION (10-20ms)                                               │       │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │       │
│  │  │ Trie/Prefix │ │ User        │ │ Trending/   │ │ Collaborative│            │       │
│  │  │ Lookup      │ │ History     │ │ Rising      │ │ (similar     │            │       │
│  │  │             │ │ (personal)  │ │ Queries     │ │  users)      │            │       │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘            │       │
│  │         └───────────────┼───────────────┼───────────────┘                    │       │
│  │                         ▼                                                   │       │
│  │              Candidate Pool (100s-1000s)                                    │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  RANKING (5-15ms)                                                             │       │
│  │  Features: popularity, recency, personalization, context, query quality        │       │
│  │  Lightweight model: linear / small GBDT (must be <10ms)                      │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  FILTERING (2-5ms)                                                            │       │
│  │  Bloom filter (known bad) | ML classifier (novel offensive) | Policy rules   │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌────────────────┐                                                                     │
│  │  Response      │  Top-k suggestions (typically 5-10)                                  │
│  │  [suggestions] │                                                                      │
│  └────────────────┘                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Data Structures

#### Trie (Prefix Tree)

The trie enables O(m) prefix lookup where m = prefix length. Each node stores children and optionally stores query metadata (popularity, last_seen) at leaf nodes.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

@dataclass
class TrieNode:
    """Node in prefix tree for autocomplete."""
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_end: bool = False
    # Metadata for ranking (stored at leaf or along path)
    query: Optional[str] = None
    popularity: float = 0.0
    last_seen_ts: float = 0.0
    successful_search_rate: float = 0.0

class TrieIndex:
    """
    Trie for fast prefix-based autocomplete candidate retrieval.
    Supports millions of queries with O(prefix_length) lookup.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self._size = 0
    
    def insert(self, query: str, popularity: float = 0.0, 
               last_seen: float = 0.0, success_rate: float = 0.0) -> None:
        """Insert a query into the trie with metadata."""
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end:
            self._size += 1
        node.is_end = True
        node.query = query
        node.popularity = popularity
        node.last_seen_ts = last_seen
        node.successful_search_rate = success_rate
    
    def search_prefix(self, prefix: str, max_candidates: int = 100) -> List[TrieNode]:
        """
        Find all queries matching prefix. Returns nodes with metadata for ranking.
        Uses DFS to collect leaf nodes under prefix.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        self._collect_queries(node, results, max_candidates)
        return results
    
    def _collect_queries(
        self, 
        node: TrieNode, 
        results: List[TrieNode], 
        max_results: int,
        visited: Optional[set] = None
    ) -> None:
        """DFS to collect query nodes. Can add early stopping by popularity."""
        if visited is None:
            visited = set()
        if len(results) >= max_results:
            return
        if node.is_end and node.query and node.query not in visited:
            results.append(node)
            visited.add(node.query)
        for child in sorted(node.children.values(), 
                           key=lambda n: -n.popularity)[:20]:  # Prune low-pop
            self._collect_queries(child, results, max_results, visited)
    
    def __len__(self) -> int:
        return self._size
```

#### Inverted Index for Substring Matching

For "fuzzy" or mid-query matching (e.g., "coo" matching "how to cook"), use an inverted index mapping tokens to query IDs.

```python
from collections import defaultdict
from typing import Dict, List

class InvertedIndex:
    """
    Token -> query mapping for substring/ token-based candidate generation.
    Useful when prefix doesn't match (e.g., typo, mid-query).
    """
    
    def __init__(self):
        self.index: Dict[str, set] = defaultdict(set)  # token -> {query_ids}
        self.queries: Dict[str, dict] = {}  # query_id -> metadata
    
    def add(self, query_id: str, query: str, tokens: List[str], metadata: dict) -> None:
        self.queries[query_id] = {"query": query, **metadata}
        for token in tokens:
            if len(token) >= 2:  # Skip very short tokens
                self.index[token].add(query_id)
    
    def search_tokens(self, tokens: List[str], max_results: int = 50) -> List[str]:
        """Find queries containing any of the tokens (OR logic)."""
        if not tokens:
            return []
        candidates = set()
        for token in tokens:
            candidates.update(self.index.get(token, set()))
        return list(candidates)[:max_results]
```

#### Bloom Filter for Offensive Query Detection

```python
class OffensiveQueryFilter:
    """
    Fast rejection of known offensive/prohibited queries.
    In production: use Redis Bloom or pybloom_live for memory efficiency.
    Here we use a set for simplicity (O(1) lookup, no false positives).
    """
    
    def __init__(self, expected_items: int = 1_000_000, fp_rate: float = 0.001):
        # Simplified: set for exact match. Production = Bloom filter for scale
        self._known_bad: set = set()
    
    def add_bad_query(self, query: str) -> None:
        normalized = query.lower().strip()
        self._known_bad.add(normalized)
    
    def is_offensive(self, query: str) -> bool:
        """Returns True if query is known to be offensive."""
        normalized = query.lower().strip()
        return normalized in self._known_bad
```

### 2. Candidate Generation

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math

@dataclass
class AutocompleteCandidate:
    query: str
    source: str  # "trie", "history", "trending", "collaborative"
    popularity: float
    recency_score: float
    personalization_score: float
    metadata: dict

class CandidateGenerator:
    """
    Multi-source candidate generation for autocomplete.
    Fetches from trie, user history, trending queries, collaborative signals.
    """
    
    def __init__(
        self,
        trie: TrieIndex,
        user_history_store: Optional[dict] = None,
        trending_store: Optional[dict] = None,
        max_candidates: int = 500
    ):
        self.trie = trie
        self.user_history_store = user_history_store or {}
        self.trending_store = trending_store or {}
        self.max_candidates = max_candidates
    
    def generate(
        self,
        prefix: str,
        user_id: Optional[str] = None,
        locale: str = "en",
        limit: int = 500
    ) -> List[AutocompleteCandidate]:
        """Generate candidates from all sources."""
        candidates_dict: Dict[str, AutocompleteCandidate] = {}
        
        # 1. Trie prefix matching (primary source)
        trie_nodes = self.trie.search_prefix(prefix, max_candidates=limit // 2)
        for node in trie_nodes:
            if node.query and node.query not in candidates_dict:
                candidates_dict[node.query] = AutocompleteCandidate(
                    query=node.query,
                    source="trie",
                    popularity=node.popularity,
                    recency_score=self._time_decay(node.last_seen_ts),
                    personalization_score=0.0,
                    metadata={"success_rate": node.successful_search_rate}
                )
        
        # 2. User's personal search history (personalization)
        if user_id and len(candidates_dict) < limit:
            history = self.user_history_store.get(user_id, [])
            for h in history:
                q, ts, clicked = h.get("query"), h.get("timestamp", 0), h.get("clicked", False)
                if q and q.startswith(prefix) and q not in candidates_dict:
                    recency = self._time_decay(ts)
                    candidates_dict[q] = AutocompleteCandidate(
                        query=q,
                        source="history",
                        popularity=0.0,
                        recency_score=recency,
                        personalization_score=1.0 if clicked else 0.5,
                        metadata={}
                    )
        
        # 3. Trending/rising queries (time-decayed popularity)
        trending = self.trending_store.get(locale, [])
        for t in trending:
            q, pop, delta = t.get("query"), t.get("popularity", 0), t.get("growth", 0)
            if q and q.startswith(prefix) and q not in candidates_dict:
                # Boost for rising queries
                trending_score = pop * (1 + math.log1p(delta))
                candidates_dict[q] = AutocompleteCandidate(
                    query=q,
                    source="trending",
                    popularity=trending_score,
                    recency_score=1.0,
                    personalization_score=0.0,
                    metadata={"growth": delta}
                )
        
        return list(candidates_dict.values())[:limit]
    
    def _time_decay(self, timestamp: float, half_life_hours: float = 24.0) -> float:
        """Exponential decay for recency. Recent = higher score."""
        if timestamp <= 0:
            return 0.0
        age_hours = (datetime.utcnow().timestamp() - timestamp) / 3600
        return math.exp(-0.693 * age_hours / half_life_hours)
```

### 3. Ranking Model

```python
import math
import numpy as np
from typing import List

class QueryRanker:
    """
    Lightweight ranking model for autocomplete. Must run in <10ms.
    Uses linear model or small GBDT. Features: popularity, recency, personalization.
    """
    
    def __init__(self, feature_weights: Optional[dict] = None):
        # Learned weights (from offline training). Example defaults:
        self.weights = feature_weights or {
            "popularity": 0.3,
            "recency": 0.2,
            "personalization": 0.3,
            "success_rate": 0.15,
            "query_length_penalty": -0.05,
            "source_trie": 0.1,
            "source_history": 0.3,
            "source_trending": 0.05,
        }
    
    def rank(
        self,
        candidates: List[AutocompleteCandidate],
        user_id: Optional[str] = None,
        context: Optional[dict] = None,
        top_k: int = 10
    ) -> List[str]:
        """Rank candidates and return top-k query strings."""
        if not candidates:
            return []
        
        scored = []
        for c in candidates:
            features = self._extract_features(c, context)
            score = sum(self.weights.get(k, 0) * v for k, v in features.items())
            scored.append((c.query, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in scored[:top_k]]
    
    def _extract_features(self, c: AutocompleteCandidate, context: dict) -> dict:
        """Extract features for scoring."""
        return {
            "popularity": min(1.0, math.log1p(c.popularity) / 10),
            "recency": c.recency_score,
            "personalization": c.personalization_score,
            "success_rate": c.metadata.get("success_rate", 0.5),
            "query_length_penalty": -len(c.query) / 100,
            f"source_{c.source}": 1.0,
        }
```

### 4. Prefix Caching

```python
from functools import lru_cache
import time

class PrefixCache:
    """
    Cache top-k results for common prefixes. Reduces load on trie/ranking.
    Invalidation: TTL + invalidation on trending query updates.
    """
    
    def __init__(
        self,
        backend: dict,  # Redis in production
        ttl_seconds: int = 300,
        max_entries: int = 1_000_000
    ):
        self.backend = backend
        self.ttl = ttl_seconds
        self.max_entries = max_entries
    
    def get(self, prefix: str, user_id: Optional[str] = None) -> Optional[List[str]]:
        """Get cached suggestions. User-specific cache for personalized."""
        key = f"ac:{prefix}" if not user_id else f"ac:{user_id}:{prefix}"
        entry = self.backend.get(key)
        if entry is None:
            return None
        results, ts = entry
        if time.time() - ts > self.ttl:
            del self.backend[key]
            return None
        return results
    
    def set(
        self,
        prefix: str,
        results: List[str],
        user_id: Optional[str] = None
    ) -> None:
        key = f"ac:{prefix}" if not user_id else f"ac:{user_id}:{prefix}"
        self.backend[key] = (results, time.time())
    
    def invalidate_prefix(self, prefix: str) -> None:
        """Invalidate cache when trending data changes (e.g., new viral query)."""
        keys_to_del = [k for k in self.backend if k.endswith(prefix) or f":{prefix}" in k]
        for k in keys_to_del:
            del self.backend[k]
```

### 5. Autocomplete Service (Orchestrator)

```python
class AutocompleteService:
    """
    End-to-end autocomplete service. Orchestrates processing, candidate gen,
    ranking, filtering, caching.
    """
    
    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        ranker: QueryRanker,
        offensive_filter: OffensiveQueryFilter,
        cache: PrefixCache,
        top_k: int = 10
    ):
        self.candidate_generator = candidate_generator
        self.ranker = ranker
        self.offensive_filter = offensive_filter
        self.cache = cache
        self.top_k = top_k
    
    def suggest(
        self,
        prefix: str,
        user_id: Optional[str] = None,
        locale: str = "en",
        use_cache: bool = True
    ) -> List[str]:
        """Main entry point. Returns top-k safe, ranked suggestions."""
        # Normalize
        prefix = prefix.lower().strip()
        if len(prefix) < 2:
            return []
        
        # Cache lookup (for non-personalized or when user cache hit)
        if use_cache:
            cached = self.cache.get(prefix, user_id)
            if cached is not None:
                return cached[:self.top_k]
        
        # Generate candidates
        candidates = self.candidate_generator.generate(
            prefix, user_id=user_id, locale=locale
        )
        
        # Filter offensive
        safe = [c for c in candidates if not self.offensive_filter.is_offensive(c.query)]
        
        # Rank
        results = self.ranker.rank(safe, user_id=user_id, top_k=self.top_k)
        
        # Cache
        if use_cache and results:
            self.cache.set(prefix, results, user_id)
        
        return results
```

### 6. Language Model Integration

**When to use LM vs popularity-based:**

| Scenario | Approach | Rationale |
|----------|----------|-----------|
| Short prefix (1-3 chars) | Popularity + trie | Too many completions; LM overfits |
| Medium prefix (4-8 chars) | Hybrid: LM score + popularity | LM captures syntax/semantics |
| Long prefix (>8 chars) | LM-weighted | Strong context for next-token prediction |
| Rare/long-tail queries | LM | Popularity fails; LM generalizes |
| Multi-word queries | N-gram or neural LM | Word order matters |

```python
from collections import defaultdict
from typing import Dict, List

class NGramCompletionModel:
    """
    N-gram language model for completion probability.
    P(word_n | word_{n-1}, ..., word_{n-k})
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def add_sequence(self, words: List[str]) -> None:
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i:i+self.n-1])
            next_word = words[i+self.n-1]
            self.ngrams[context][next_word] += 1.0
    
    def get_completion_score(self, context: List[str], completion: str) -> float:
        """Score for completing with 'completion' given context."""
        ctx = tuple(context[-(self.n-1):])
        counts = self.ngrams.get(ctx, {})
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return counts.get(completion, 0) / total
```

### 7. Multi-Language Support

| Challenge | Solution |
|-----------|----------|
| **Tokenization** | Language-specific tokenizers (jieba, mecab, whitespace) |
| **Transliteration** | Romanized input → native script (e.g., "namaste" → "नमस्ते") |
| **Code-switching** | Mixed-language queries; tokenize per segment |
| **RTL languages** | Reverse display order; cursor handling |
| **Compound words** | German, Finnish: split or use subword tokenization |

---

## 📈 Metrics & Evaluation

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Completion Acceptance Rate (CAR)** | % of sessions where user selected a suggestion | 20-40% |
| **Keystrokes Saved** | (Chars typed without AC - with AC) / chars without AC | 40-50% |
| **Search Success Rate (SSR)** | % of suggested queries that led to successful search (click, dwell) | > 60% |
| **Latency p50/p99** | Response time per request | p50 < 20ms, p99 < 50ms |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Suggestion diversity** | Unique queries in top-k across sessions |
| **Trending coverage** | % of trending queries surfaced within 15 min |
| **Offensive escape rate** | % of bad queries that slip through (should be ~0) |
| **Personalization lift** | CAR improvement with personalization vs without |

```python
def compute_autocomplete_metrics(
    sessions: List[dict]  # Each: {queries, suggestions_shown, selected, search_success}
) -> dict:
    total_sessions = len(sessions)
    with_selection = sum(1 for s in sessions if s.get("selected"))
    keystrokes_without = sum(s.get("keystrokes_without_ac", 0) for s in sessions)
    keystrokes_with = sum(s.get("keystrokes_with_ac", 0) for s in sessions)
    success_after = sum(1 for s in sessions if s.get("selected") and s.get("search_success"))
    
    return {
        "completion_acceptance_rate": with_selection / max(1, total_sessions),
        "keystrokes_saved": 1 - (keystrokes_with / max(1, keystrokes_without)),
        "search_success_rate": success_after / max(1, with_selection),
    }
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Trie vs suffix array** | Trie (simple, prefix-only) | Suffix array (substring) | Trie for prefix; add inverted index for fuzzy |
| **Personalization** | No (faster, simpler) | Yes (better CAR) | Yes for logged-in users; cache per-user |
| **LM vs popularity** | Popularity only | LM-weighted | Hybrid: LM for long prefixes |
| **Cache strategy** | No cache | Prefix + CDN cache | Cache aggressively; invalidate on trending |
| **Offensive filter** | Bloom only | Bloom + ML classifier | Both: Bloom for known, ML for novel |
| **Ranking model** | Heuristics | ML model | Start heuristics; ML when data allows |
| **Debouncing** | 50ms | 100ms | 50-100ms; tune for latency vs request volume |

---

## 🎤 Interview Tips

### What to Emphasize

1. **Latency budget**—<50ms end-to-end; every component must be optimized
2. **Multi-source candidates**—trie + history + trending + collaborative
3. **Lightweight ranking**—linear or small GBDT; no heavy neural nets in hot path
4. **Caching**—prefix caching and CDN for popular prefixes
5. **Safety**—Bloom filter + ML classifier for offensive content
6. **Personalization**—user history and embeddings, with privacy awareness

### Common Follow-ups

1. **How do you handle typos?** — Spell correction before trie lookup; Levenshtein/edit distance for fuzzy prefix
2. **How do you add a new trending query in real-time?** — Stream processing (e.g., Flink) for query logs; update trending store; invalidate relevant prefix caches
3. **How do you support 100+ languages?** — Per-language tries, tokenizers, transliteration; sharding by language
4. **How do you reduce latency further?** — More caching, precomputation of top-k for common prefixes, model distillation
5. **How do you avoid filter bubbles?** — Diversity in ranking; occasional random/exploratory suggestions; don't over-weight history

### Red Flags to Avoid

- Ignoring latency constraints
- Proposing heavy neural models in the hot path
- Not addressing offensive content filtering
- Forgetting multi-language and internationalization

---

## 🔗 Related Topics

- [Search Systems](./02-search-systems.md) — Autocomplete feeds into search; shared ranking concepts
- [NLP Systems](./05-nlp-systems.md) — Tokenization, LMs, multi-language
- [Caching Strategies](../../phase-3-operations-and-reliability/07-scalability-performance/02-caching-strategies.md) — Prefix caching, CDN
- [Feature Stores](../../phase-2-core-components/03-feature-engineering/01-feature-stores.md) — User embeddings, query features
