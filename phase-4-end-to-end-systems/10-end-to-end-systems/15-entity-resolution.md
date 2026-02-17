# Entity Resolution System

## Overview

Entity resolution (also known as record linkage, deduplication, or identity resolution) identifies and merges records that refer to the same real-world entity across different data sources. It is critical for data quality, master data management, and linking entities (users, products, businesses) across systems. **Important for data quality across many systems—common at data-heavy companies.**

---

## 🎯 Problem Definition

### Business Goals

- **Deduplicate entities:** Merge duplicate user accounts, product listings, business records
- **Maintain data quality:** Single source of truth; clean CRM, inventory, analytics
- **Enable linking:** Join data across sources (e.g., user from App A = user from App B)
- **Compliance:** Accurate identity for KYC, fraud prevention, GDPR
- **Analytics:** Correct counts (e.g., unique users) and attribution

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Scale** | Process billions of entity pairs |
| **Precision** | High; avoid false merges (hard to undo) |
| **Recall** | Catch true duplicates; balance with precision |
| **Scalability** | Distributed; incremental updates |
| **Incremental** | Handle new records without full recompute |
| **Latency** | Batch: hours; real-time: <100ms for lookup |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Entity Resolution System Architecture                                  │
│                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA SOURCES                                              │    │
│  │  [CRM] [Transactional DB] [Third-party] [User signups] [Product catalog]          │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    ENTITY EXTRACTION & NORMALIZATION                               │    │
│  │  Parse, standardize, clean (address, name, phone, email)                           │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    BLOCKING (Reduce O(n²) to manageable)                           │    │
│  │  Standard | Sorted Neighborhood | LSH | Canopy | Token                            │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    PAIRWISE COMPARISON                                            │    │
│  │  String similarity, phonetic, numeric, ML feature extraction                       │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    CLASSIFICATION (Match / No-Match / Maybe)                       │    │
│  │  Rules | Probabilistic (Fellegi-Sunter) | ML (RF, GB, DL)                         │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    CLUSTERING (Transitive closure)                                │    │
│  │  Connected components | Correlation clustering | Handle transitivity violations   │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    MERGE / LINK                                                    │    │
│  │  Create canonical entity; store linkages                                           │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    ENTITY STORE + MONITORING                                        │    │
│  │  Golden records; linkage table; quality metrics                                    │    │
│  └──────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Blocking Strategies

**Problem:** Naive pairwise comparison is O(n²)—infeasible for millions of records.

**Standard blocking:**
- Partition by shared attribute value (e.g., first 3 chars of last name, zip code)
- Only compare pairs within same block
- Risk: Miss pairs with different block keys (e.g., "Smith" vs "Smyth")

**Sorted neighborhood:**
- Sort records by a key (e.g., phonetic encoding of name)
- Compare only records within a sliding window (e.g., ±10)
- Reduces comparisons; sensitive to sort order

**LSH (Locality-Sensitive Hashing):**
- MinHash, SimHash for approximate similarity
- Hash similar records to same buckets with high probability
- Tunable: more bands = higher recall, more comparisons

**Canopy clustering:**
- Fast, approximate clustering (e.g., TF-IDF cosine)
- Loose threshold forms "canopies"; exact comparison only within canopy
- Two tiers: cheap + expensive

**Token blocking:**
- Block on tokens (e.g., all tokens in name)
- Records sharing any token are candidates
- Good for names: "John Smith" and "Smith, John" share "Smith"

### 2. Pairwise Matching

**String similarity:**
- **Jaro-Winkler:** Good for names; rewards matching prefix
- **Levenshtein:** Edit distance; O(m×n)
- **TF-IDF cosine:** For longer text (addresses, descriptions)

**Phonetic matching:**
- **Soundex:** Encode by sound; "Smith" and "Smyth" → same code
- **Metaphone / Double Metaphone:** More accurate for English

**Numeric/date similarity:**
- Normalized difference for ages, amounts
- Date parsing (MM/DD/YYYY vs YYYY-MM-DD)

**Address parsing and normalization:**
- Split street, city, state, zip
- Abbreviate (St → Street); geocode for lat/lon similarity

**ML-based:**
- Feature vector from similarity scores (Jaro-Winkler, cosine, etc.)
- Train classifier: match / no-match
- Or deep learning: Siamese network on raw text

### 3. Classification

**Rule-based:**
- Deterministic: if Jaro-Winkler(name) > 0.9 and zip match → match
- Interpretable; brittle for edge cases

**Probabilistic (Fellegi-Sunter):**
- m-probability: P(agree on attr | match)
- u-probability: P(agree on attr | non-match)
- Likelihood ratio; threshold for match
- EM to estimate m, u from data

**ML-based:**
- Random forest, gradient boosting on similarity features
- Deep learning: BERT-based siamese for text pairs
- Output: probability; use threshold + "maybe" for human review

**Active learning:**
- Uncertainty sampling: Label pairs where model is unsure
- Query-by-committee: Disagreement across ensemble
- Reduces labeling cost; improves model

### 4. Clustering and Transitivity

**Connected components:**
- If A=B and B=C, then A, B, C in same cluster
- Union-Find to compute components
- Risk: One bad link merges many entities (transitivity error)

**Correlation clustering:**
- Optimize: maximize agreements (match edges inside cluster, non-match outside)
- Handles inconsistent pairwise labels

**Transitivity violations:**
- A=B, B=C, but A≠C (rare but possible with noise)
- Resolve: Remove lowest-confidence link; or use more sophisticated clustering (e.g., correlation clustering)

### 5. Active Learning for Labels

**Uncertainty sampling:**
- Label pairs with P(match) ≈ 0.5
- Maximal information gain

**Query-by-committee:**
- Train ensemble; label where models disagree

**Reducing labeling costs:**
- Use rules for clear match/non-match; ML for borderline
- Human-in-loop for "maybe" only

### 6. Scale Considerations

**Distributed blocking (Spark/MapReduce):**
- Partition by block key; process blocks in parallel
- Reduce stage to aggregate pair scores
- Handle blocks that span partitions

**Incremental entity resolution:**
- New records: Block with existing; compare only new × existing (not existing × existing)
- Update clusters incrementally
- Periodic full run to fix drift

**Real-time vs batch:**
- **Batch:** Nightly job; full resolution
- **Real-time:** Lookup index (block key → entity cluster); add new record to cluster or create new

---

## 💻 Python Code

### BlockingStrategy

```python
from abc import ABC, abstractmethod
from typing import List, Set, Tuple
from collections import defaultdict

class BlockingStrategy(ABC):
    """Base class for blocking strategies."""
    
    @abstractmethod
    def get_blocks(self, records: List[dict]) -> dict:
        """Return block_key -> set of record_ids."""
        pass
    
    def get_candidate_pairs(self, records: List[dict]) -> List[Tuple[str, str]]:
        """Return list of (id_a, id_b) candidate pairs."""
        blocks = self.get_blocks(records)
        pairs = set()
        for block_key, ids in blocks.items():
            id_list = list(ids)
            for i in range(len(id_list)):
                for j in range(i + 1, len(id_list)):
                    pairs.add((id_list[i], id_list[j]))
        return list(pairs)


class StandardBlocking(BlockingStrategy):
    """Block by shared attribute value."""
    
    def __init__(self, attribute: str, tokenize: bool = False):
        self.attribute = attribute
        self.tokenize = tokenize
    
    def get_blocks(self, records: List[dict]) -> dict:
        blocks = defaultdict(set)
        for r in records:
            val = r.get(self.attribute, "")
            if self.tokenize:
                for token in str(val).lower().split():
                    blocks[token].add(r["id"])
            else:
                blocks[str(val)].add(r["id"])
        return dict(blocks)


class SortedNeighborhoodBlocking(BlockingStrategy):
    """Sliding window over sorted records."""
    
    def __init__(self, sort_key: str, window_size: int = 10):
        self.sort_key = sort_key
        self.window_size = window_size
    
    def get_candidate_pairs(self, records: List[dict]) -> List[Tuple[str, str]]:
        sorted_recs = sorted(records, key=lambda r: str(r.get(self.sort_key, "")))
        pairs = []
        for i in range(len(sorted_recs)):
            for j in range(i + 1, min(i + self.window_size + 1, len(sorted_recs))):
                pairs.append((sorted_recs[i]["id"], sorted_recs[j]["id"]))
        return pairs
    
    def get_blocks(self, records: List[dict]) -> dict:
        raise NotImplementedError("Use get_candidate_pairs for sorted neighborhood")


class MinHashLSHBlocking(BlockingStrategy):
    """LSH with MinHash for approximate Jaccard similarity."""
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.5):
        self.num_perm = num_perm
        self.threshold = threshold
    
    def _minhash_signature(self, tokens: Set[str], permutations: List[Tuple[int, int]]) -> List[int]:
        import hashlib
        sig = []
        for a, b in permutations:
            min_h = float('inf')
            for t in tokens:
                h = (a * hash(t) + b) % (2**32)
                min_h = min(min_h, h)
            sig.append(min_h)
        return sig
    
    def get_blocks(self, records: List[dict]) -> dict:
        import random
        blocks = defaultdict(set)
        perms = [(random.randint(1, 2**32), random.randint(1, 2**32)) for _ in range(self.num_perm)]
        
        for r in records:
            # Use name + address as tokens
            text = f"{r.get('name','')} {r.get('address','')}".lower()
            tokens = set(text.split())
            sig = self._minhash_signature(tokens, perms)
            # Banding: group consecutive hash values
            band_size = 4
            for i in range(0, len(sig), band_size):
                band = tuple(sig[i:i+band_size])
                blocks[band].add(r["id"])
        
        return dict(blocks)
```

### PairwiseMatcher

```python
from typing import Dict, List, Tuple
import numpy as np

def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity. 1.0 = identical."""
    if not s1 or not s2:
        return 0.0
    # Simplified implementation; use jellyfish or rapidfuzz in production
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    match_window = max(len(s1), len(s2)) // 2 - 1
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    for i, c in enumerate(s1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len(s2))
        for j in range(start, end):
            if not s2_matches[j] and c == s2[j]:
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
    
    if matches == 0:
        return 0.0
    
    transpositions = 0
    k = 0
    for i, c in enumerate(s1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if c != s2[k]:
                transpositions += 1
            k += 1
    
    jaro = (matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches) / 3
    
    # Winkler: bonus for common prefix
    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


class PairwiseMatcher:
    """Computes similarity features for a record pair."""
    
    def __init__(self, attributes: List[str], weights: Dict[str, float] = None):
        self.attributes = attributes
        self.weights = weights or {a: 1.0 for a in attributes}
    
    def compute_features(self, r1: dict, r2: dict) -> Dict[str, float]:
        features = {}
        for attr in self.attributes:
            v1 = str(r1.get(attr, "") or "")
            v2 = str(r2.get(attr, "") or "")
            features[f"{attr}_jaro"] = jaro_winkler(v1, v2)
            features[f"{attr}_exact"] = 1.0 if v1 == v2 else 0.0
        return features
    
    def weighted_score(self, features: Dict[str, float]) -> float:
        score = 0.0
        total_weight = 0.0
        for k, v in features.items():
            attr = k.replace("_jaro", "").replace("_exact", "")
            w = self.weights.get(attr, 1.0)
            score += v * w
            total_weight += w
        return score / total_weight if total_weight > 0 else 0.0
```

### EntityResolver

```python
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class ResolutionResult:
    entity_id: str
    canonical_record: dict
    linked_record_ids: Set[str]
    confidence: float

class EntityResolver:
    """Full entity resolution pipeline."""
    
    def __init__(
        self,
        blocking: BlockingStrategy,
        matcher: PairwiseMatcher,
        classifier: Any,
        match_threshold: float = 0.9,
        maybe_threshold: float = 0.6,
    ):
        self.blocking = blocking
        self.matcher = matcher
        self.classifier = classifier
        self.match_threshold = match_threshold
        self.maybe_threshold = maybe_threshold
    
    def resolve(self, records: List[dict]) -> List[ResolutionResult]:
        # 1. Blocking
        pairs = self.blocking.get_candidate_pairs(records)
        rec_map = {r["id"]: r for r in records}
        
        # 2. Pairwise comparison & classification
        match_edges: List[Tuple[str, str, float]] = []
        for id_a, id_b in pairs:
            r1, r2 = rec_map[id_a], rec_map[id_b]
            features = self.matcher.compute_features(r1, r2)
            score = self._classify(features)
            if score >= self.match_threshold:
                match_edges.append((id_a, id_b, score))
        
        # 3. Clustering (connected components)
        clusters = self._connected_components(match_edges)
        
        # 4. Create resolution results
        results = []
        for cluster in clusters:
            ids = list(cluster)
            canonical = rec_map[ids[0]]  # Or merge/choose best
            results.append(ResolutionResult(
                entity_id=canonical["id"],
                canonical_record=canonical,
                linked_record_ids=set(ids),
                confidence=0.95,
            ))
        
        return results
    
    def _classify(self, features: Dict[str, float]) -> float:
        # Use classifier or weighted score
        if self.classifier:
            return self.classifier.predict_proba([list(features.values())])[0][1]
        return self.matcher.weighted_score(features)
    
    def _connected_components(self, edges: List[Tuple[str, str, float]]) -> List[Set[str]]:
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb
        
        for a, b, _ in edges:
            union(a, b)
        
        components = defaultdict(set)
        for x in parent:
            components[find(x)].add(x)
        
        return list(components.values())
```

### ActiveLearner

```python
class ActiveLearner:
    """Uncertainty sampling for label acquisition."""
    
    def __init__(self, model: Any, uncertainty_metric: str = "entropy"):
        self.model = model
        self.uncertainty_metric = uncertainty_metric
    
    def get_next_batch(
        self,
        unlabeled_pairs: List[Tuple[dict, dict]],
        batch_size: int = 10,
    ) -> List[int]:
        """Return indices of pairs to label (highest uncertainty)."""
        if not unlabeled_pairs:
            return []
        
        probs = self.model.predict_proba(unlabeled_pairs)
        
        if self.uncertainty_metric == "entropy":
            # -p*log(p) - (1-p)*log(1-p); max at p=0.5
            import math
            uncertainties = []
            for p in probs:
                prob_match = p[1] if len(p) > 1 else p[0]
                prob_match = max(1e-10, min(1 - 1e-10, prob_match))
                ent = -prob_match * math.log(prob_match) - (1 - prob_match) * math.log(1 - prob_match)
                uncertainties.append(ent)
        else:
            # Distance from 0.5
            uncertainties = [abs(p[1] - 0.5) if len(p) > 1 else abs(p[0] - 0.5) for p in probs]
            uncertainties = [1 - u for u in uncertainties]  # Invert: prefer near 0.5
        
        indices = np.argsort(uncertainties)[::-1][:batch_size]
        return indices.tolist()
```

---

## 📈 Metrics & Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision** | Of predicted matches, % true matches | > 99% (avoid false merges) |
| **Recall** | Of true matches, % found | > 95% |
| **F1** | Harmonic mean of P and R | Balance |
| **Pair Completeness** | % of true pairs in blocks | Measure blocking recall |
| **Reduction Ratio** | 1 - (pairs after blocking) / (all pairs) | Measure blocking efficiency |
| **Cluster Quality** | Purity, NMI of clusters | Internal coherence |

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Blocking** | Standard (simple, fast) | LSH (flexible, tunable) |
| **Matching** | Rule-based (interpretable) | ML (accuracy) |
| **Precision vs Recall** | High precision (few false merges) | High recall (catch more duplicates) |
| **Batch vs Incremental** | Full batch (accurate) | Incremental (fresh) |
| **Human review** | Auto-merge all matches | Review "maybe" only |
| **Transitivity** | Simple connected components | Correlation clustering (robust) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you reduce O(n²) comparisons?
2. How do you handle transitivity (A=B, B=C, A≠C)?
3. Why is precision often more important than recall for entity resolution?
4. How do you do incremental resolution when new records arrive?
5. How do you scale to billions of records?

**Key Points to Mention:**
- Blocking is critical: standard, sorted neighborhood, LSH, token blocking
- Pairwise: string similarity (Jaro-Winkler), phonetic, TF-IDF
- Classification: rules, Fellegi-Sunter, ML
- Clustering: connected components; handle transitivity
- Active learning to reduce labeling cost
- Precision > recall (false merges hard to undo)

---

## 🔗 Related Topics

- [Data Quality](../../phase-2-core-components/02-data-management/04-data-quality.md)
- [Data Storage](../../phase-2-core-components/02-data-management/02-data-storage.md)
- [Embedding Fundamentals](../../phase-5-advanced-topics/11-embeddings-retrieval/01-embedding-fundamentals.md)
- [Distributed Training](../../phase-2-core-components/04-model-training/05-distributed-training.md)
- [Batch vs Real-time](../../phase-3-operations-and-reliability/07-scalability-performance/03-batch-vs-realtime.md)
