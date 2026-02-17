# Content Moderation System

## Overview

Content moderation systems identify and act on content that violates platform policies across text, images, video, and audio. These systems are critical for social media platforms, marketplaces, and user-generated content platforms. The challenge is balancing platform safety with minimizing over-enforcement that harms legitimate users—all while processing billions of content pieces daily across multiple modalities. **Frequently asked at Meta.**

---

## 🎯 Problem Definition

### Business Goals
- **Keep platform safe:** Protect users from harmful content while complying with regulations
- **Minimize over-enforcement:** Avoid wrongly removing legitimate content (false positives hurt trust)
- **Scale efficiently:** Process billions of posts per day within budget
- **Support human review:** Route borderline cases to trained reviewers effectively
- **Adapt to evolving threats:** Continuously improve as adversarial actors adapt

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Throughput** | Billions of content pieces/day |
| **Latency (high-severity)** | < 1s for critical content (violence, self-harm, terrorism) |
| **Latency (standard)** | < 30s for general content |
| **Modalities** | Text, image, video, audio |
| **Languages** | 100+ languages supported |
| **Accuracy** | High precision for borderline; high recall for severe categories |

### Policy Taxonomy

| Policy Category | Examples | Severity | Typical Action |
|----------------|----------|----------|----------------|
| **Hate speech** | Dehumanization, slurs, incitement | High | Remove, warn |
| **Violence/Graphic** | Gore, graphic violence | High | Remove, age-gate |
| **Nudity/Sexual** | NSFW, sexual exploitation | High | Remove, blur |
| **Harassment/Bullying** | Targeted abuse, doxxing | Medium | Remove, restrict |
| **Misinformation** | False claims, manipulated media | Medium | Label, downrank |
| **Spam** | Fake accounts, clickbait | Low | Remove, limit reach |
| **Self-harm** | Suicide, eating disorders | Critical | Remove, resources |
| **Terrorism** | Violent extremism, recruitment | Critical | Remove, report |
| **Copyright** | DMCA violations | Legal | Remove on request |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      Content Moderation System Architecture                            │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                          CONTENT UPLOAD                                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │   │
│  │  │   Text      │  │   Image     │  │   Video     │  │   Audio     │           │   │
│  │  │   Post      │  │   Upload    │  │   Upload    │  │   Clip      │           │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │   │
│  └─────────┼────────────────┼────────────────┼────────────────┼───────────────────┘   │
│            │                │                │                │                         │
│            └────────────────┴────────────────┴────────────────┘                         │
│                                     │                                                    │
│                                     ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    PRE-SCREENING (Fast Path)                                     │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐         │   │
│  │  │  Hash Matching     │  │  Known Bad Content │  │  User/Account      │         │   │
│  │  │  (PhotoDNA, pHash) │  │  Database (NCMEC)  │  │  Reputation       │         │   │
│  │  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘         │   │
│  └────────────┼───────────────────────┼───────────────────────┼─────────────────────┘   │
│               │                       │                       │                          │
│               └───────────────────────┼───────────────────────┘                          │
│                                       │                                                   │
│                           Known bad? ─┼── YES ──▶ IMMEDIATE REMOVE + REPORT              │
│                                       │                                                   │
│                                       NO                                                  │
│                                       ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │              MULTI-MODAL CLASSIFICATION PIPELINE (Parallel)                       │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │   │
│  │  │    Text      │ │    Image     │ │    Video     │ │    Audio     │           │   │
│  │  │  Classifier  │ │  Classifier  │ │  Classifier  │ │  Classifier  │           │   │
│  │  │ (Transformer)│ │(CNN/ViT+OCR) │ │(Frame+Audio) │ │  (Speech)    │           │   │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘           │   │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────────────────┘   │
│            └────────────────┴────────────────┴────────────────┴───────────────────────  │
│                                        │                                                  │
│                                        ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         MULTIMODAL FUSION                                         │   │
│  │            (Early Fusion / Late Fusion / Score Aggregation)                        │   │
│  └─────────────────────────────────────┬─────────────────────────────────────────────┘   │
│                                        │                                                  │
│                                        ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         POLICY ENGINE                                             │   │
│  │  ┌────────────────────┐  ┌────────────────────┐                                  │   │
│  │  │  Rule Evaluation   │  │  ML Score Mapping   │  → Policy-specific scores       │   │
│  │  │  (Country, Age)   │  │  (Per-policy)       │                                  │   │
│  │  └────────────────────┘  └────────────────────┘                                  │   │
│  └─────────────────────────────────────┬─────────────────────────────────────────────┘   │
│                                        │                                                  │
│                                        ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         DECISION ENGINE                                           │   │
│  │     ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │   │
│  │     │  AUTO-REMOVE  │  │ HUMAN REVIEW  │  │     ALLOW     │                    │   │
│  │     │ (High conf)   │  │  (Borderline) │  │  (Low risk)   │                    │   │
│  │     └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                    │   │
│  └─────────────┼──────────────────┼──────────────────┼──────────────────────────────┘   │
│                │                  │                  │                                   │
│                ▼                  ▼                  ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    HUMAN REVIEW QUEUE                                             │   │
│  │  Priority = f(severity, confidence, user impact) | Routing by expertise/lang      │   │
│  └─────────────────────────────────────┬─────────────────────────────────────────────┘   │
│                                        │                                                  │
│                                        ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    APPEALS & FEEDBACK LOOP                                        │   │
│  │  Appeals → Overturn Rate → Model Retraining → Policy Updates                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### Multi-Modal Pipeline Details

#### Text Modality
- **Model:** Transformer-based (BERT, XLM-R) with multi-label heads per policy
- **Multi-language:** Cross-lingual embeddings, language-specific fine-tuning
- **Challenges:** Sarcasm, context, code-switching, adversarial phrasing

#### Image Modality
- **Model:** CNN/ViT classifier for NSFW, objectionable content
- **OCR:** Extract text-in-image for hate speech, misinformation
- **Challenges:** Memes, context-dependent images, subtle violations

#### Video Modality
- **Frame sampling:** 1-5 fps → image classifier per frame
- **Temporal models:** 3D CNN or transformer over frame sequence
- **Audio track:** Separate speech/audio classifier (hate speech, music copyright)

#### Multimodal Fusion
- **Early fusion:** Concatenate embeddings before classification
- **Late fusion:** Separate classifiers → weighted score aggregation
- **Hybrid:** Use late fusion with cross-modal attention for complex cases

### Hash-Based Detection (PhotoDNA, Perceptual Hashing)

```python
from dataclasses import dataclass
from typing import List, Optional
import hashlib

@dataclass
class ContentHash:
    """Perceptual hash for known bad content."""
    content_id: str
    hash_value: str
    hash_type: str  # "photo_dna", "p_hash", "md5"
    policy_category: str

class HashBasedDetector:
    """Detect known bad content via hash matching."""
    
    def __init__(self, hash_index, threshold: float = 0.95):
        self.hash_index = hash_index  # LSH or exact match index
        self.threshold = threshold
    
    def check_image(self, image_bytes: bytes) -> Optional[ContentHash]:
        """Check image against known bad content database."""
        phash = self._compute_perceptual_hash(image_bytes)
        
        # Query hash index for near-duplicates
        matches = self.hash_index.search(phash, threshold=self.threshold)
        
        if matches:
            return ContentHash(
                content_id=matches[0].content_id,
                hash_value=phash,
                hash_type="p_hash",
                policy_category=matches[0].policy
            )
        return None
    
    def _compute_perceptual_hash(self, image_bytes: bytes) -> str:
        """Compute perceptual hash (simplified; production uses PhotoDNA/pHash)."""
        # In production: use imagehash, PhotoDNA API, or similar
        return hashlib.sha256(image_bytes).hexdigest()[:32]
```

### ContentModerationPipeline

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class Decision(Enum):
    ALLOW = "allow"
    HUMAN_REVIEW = "human_review"
    AUTO_REMOVE = "auto_remove"

@dataclass
class ModalityScore:
    modality: Modality
    policy_scores: Dict[str, float]
    confidence: float

@dataclass
class ModerationResult:
    content_id: str
    decision: Decision
    policy_violations: List[str]
    scores: Dict[str, float]
    requires_human_review: bool

class ContentModerationPipeline:
    """End-to-end content moderation pipeline."""
    
    def __init__(
        self,
        hash_detector,
        text_classifier,
        image_classifier,
        video_classifier,
        audio_classifier,
        policy_engine,
        fusion_strategy: str = "late"
    ):
        self.hash_detector = hash_detector
        self.text_classifier = text_classifier
        self.image_classifier = image_classifier
        self.video_classifier = video_classifier
        self.audio_classifier = audio_classifier
        self.policy_engine = policy_engine
        self.fusion_strategy = fusion_strategy
        
        # Per-policy precision/recall thresholds
        self.auto_remove_thresholds = {
            "terrorism": 0.7,
            "self_harm": 0.7,
            "child_safety": 0.6,
            "violence": 0.8,
            "hate_speech": 0.85,
            "nudity": 0.9,
        }
        self.human_review_thresholds = {k: v - 0.15 for k, v in self.auto_remove_thresholds.items()}
    
    async def moderate(self, content: Dict[str, Any]) -> ModerationResult:
        """Moderate content across all modalities."""
        
        # 1. Pre-screening: Hash check for images/video frames
        if content.get("image_bytes"):
            hash_match = self.hash_detector.check_image(content["image_bytes"])
            if hash_match:
                return ModerationResult(
                    content_id=content["content_id"],
                    decision=Decision.AUTO_REMOVE,
                    policy_violations=[hash_match.policy_category],
                    scores={hash_match.policy_category: 1.0},
                    requires_human_review=False
                )
        
        # 2. Run modality classifiers in parallel
        modality_scores = await self._run_classifiers(content)
        
        # 3. Fuse scores
        fused_scores = self._fuse_scores(modality_scores)
        
        # 4. Policy engine (rules + score mapping)
        final_scores = self.policy_engine.evaluate(content, fused_scores)
        
        # 5. Decision
        decision, violations = self._make_decision(final_scores)
        
        return ModerationResult(
            content_id=content["content_id"],
            decision=decision,
            policy_violations=violations,
            scores=final_scores,
            requires_human_review=(decision == Decision.HUMAN_REVIEW)
        )
    
    async def _run_classifiers(self, content: Dict) -> List[ModalityScore]:
        """Run all applicable classifiers in parallel."""
        tasks = []
        
        if content.get("text"):
            tasks.append(self._get_text_scores(content["text"]))
        if content.get("image_bytes"):
            tasks.append(self._get_image_scores(content["image_bytes"]))
        if content.get("video_url"):
            tasks.append(self._get_video_scores(content["video_url"]))
        if content.get("audio_url"):
            tasks.append(self._get_audio_scores(content["audio_url"]))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, ModalityScore)]
    
    def _fuse_scores(self, modality_scores: List[ModalityScore]) -> Dict[str, float]:
        """Late fusion: take max per policy across modalities."""
        fused = {}
        for ms in modality_scores:
            for policy, score in ms.policy_scores.items():
                fused[policy] = max(fused.get(policy, 0), score)
        return fused
    
    def _make_decision(self, scores: Dict[str, float]) -> tuple:
        """Apply thresholds to make allow/review/remove decision."""
        violations = []
        for policy, score in scores.items():
            if score >= self.auto_remove_thresholds.get(policy, 0.8):
                violations.append(policy)
            elif score >= self.human_review_thresholds.get(policy, 0.6):
                violations.append(f"{policy}_review")
        
        if any("_review" in v for v in violations):
            return Decision.HUMAN_REVIEW, violations
        if violations:
            return Decision.AUTO_REMOVE, [v for v in violations if "_review" not in v]
        
        return Decision.ALLOW, []
```

### MultiModalClassifier (Text + Image Example)

```python
import torch
import torch.nn as nn
from typing import Dict, List

class TextClassifier(nn.Module):
    """Transformer-based multi-label text classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_policies: int, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True),
            num_layers=4
        )
        self.classifier = nn.Linear(embedding_dim, num_policies)
        self.policy_names = []  # Populate from config
    
    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.classifier(pooled))

class ImageClassifier(nn.Module):
    """CNN/ViT-based image classifier for NSFW and policy violations."""
    
    def __init__(self, num_policies: int, backbone: str = "resnet50"):
        super().__init__()
        # Use pretrained backbone
        if backbone == "resnet50":
            self.backbone = nn.Sequential(
                *list(torch.hub.load('pytorch/vision', 'resnet50', pretrained=True).children())[:-1]
            )
            feat_dim = 2048
        self.fc = nn.Linear(feat_dim, num_policies)
    
    def forward(self, images):
        features = self.backbone(images).flatten(1)
        return torch.sigmoid(self.fc(features))

class MultiModalClassifier:
    """Orchestrates text and image classifiers."""
    
    def __init__(self, text_model, image_model, policy_names: List[str]):
        self.text_model = text_model
        self.image_model = image_model
        self.policy_names = policy_names
    
    def predict(self, text_tokens=None, image_tensor=None) -> Dict[str, float]:
        scores = {p: 0.0 for p in self.policy_names}
        
        if text_tokens is not None:
            with torch.no_grad():
                text_scores = self.text_model(text_tokens)
            for i, p in enumerate(self.policy_names):
                scores[p] = max(scores[p], text_scores[0, i].item())
        
        if image_tensor is not None:
            with torch.no_grad():
                img_scores = self.image_model(image_tensor)
            for i, p in enumerate(self.policy_names):
                scores[p] = max(scores[p], img_scores[0, i].item())
        
        return scores
```

### HumanReviewQueue

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum
import heapq

class ReviewPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class ReviewTask:
    content_id: str
    content_preview: str
    predicted_policies: List[str]
    scores: Dict[str, float]
    created_at: datetime
    priority_score: float
    required_expertise: List[str]
    required_languages: List[str]
    
    def __lt__(self, other):
        return self.priority_score < other.priority_score

class HumanReviewQueue:
    """Prioritized human review queue with reviewer routing."""
    
    def __init__(
        self,
        severity_weights: Dict[str, float],
        capacity_per_reviewer: int = 100
    ):
        self.severity_weights = severity_weights
        self.capacity = capacity_per_reviewer
        self.queue = []  # Min-heap by priority (lower = higher priority)
        self.reviewer_loads = {}  # reviewer_id -> current load
        self.reviewer_expertise = {}  # reviewer_id -> {policies, languages}
    
    def add_task(self, result: ModerationResult, content_meta: Dict) -> None:
        """Add content to review queue with computed priority."""
        priority = self._compute_priority(result)
        
        task = ReviewTask(
            content_id=result.content_id,
            content_preview=content_meta.get("preview", "")[:200],
            predicted_policies=result.policy_violations,
            scores=result.scores,
            created_at=datetime.utcnow(),
            priority_score=priority,
            required_expertise=self._infer_expertise(result),
            required_languages=content_meta.get("languages", ["en"])
        )
        heapq.heappush(self.queue, task)
    
    def _compute_priority(self, result: ModerationResult) -> float:
        """Priority = severity × confidence. Lower = more urgent."""
        severity_sum = 0
        for policy in result.policy_violations:
            severity_sum += self.severity_weights.get(policy.replace("_review", ""), 0.5)
        
        avg_confidence = sum(result.scores.values()) / max(len(result.scores), 1)
        
        # Lower value = higher priority
        return severity_sum * (1 - avg_confidence)
    
    def _infer_expertise(self, result: ModerationResult) -> List[str]:
        """Infer required reviewer expertise from predicted policies."""
        return [p.replace("_review", "") for p in result.policy_violations]
    
    def get_next_task(self, reviewer_id: str) -> Optional[ReviewTask]:
        """Get next task for reviewer based on expertise and capacity."""
        if self.reviewer_loads.get(reviewer_id, 0) >= self.capacity:
            return None
        
        # Find matching task
        temp_removed = []
        task = None
        while self.queue:
            t = heapq.heappop(self.queue)
            reviewer_exp = self.reviewer_expertise.get(reviewer_id, {})
            if (set(t.required_expertise) <= set(reviewer_exp.get("policies", [])) and
                any(lang in reviewer_exp.get("languages", []) for lang in t.required_languages)):
                task = t
                break
            temp_removed.append(t)
        
        for t in temp_removed:
            heapq.heappush(self.queue, t)
        
        if task:
            self.reviewer_loads[reviewer_id] = self.reviewer_loads.get(reviewer_id, 0) + 1
        
        return task
    
    def record_review(
        self,
        content_id: str,
        reviewer_id: str,
        decision: str,
        correct_policies: List[str]
    ) -> None:
        """Record review outcome for QA and model feedback."""
        self.reviewer_loads[reviewer_id] = max(0, self.reviewer_loads.get(reviewer_id, 0) - 1)
        # Log to feedback pipeline for model improvement
```

### Adversarial Robustness Considerations

| Attack Type | Example | Mitigation |
|-------------|---------|------------|
| **Unicode tricks** | Zero-width chars, homoglyphs | Unicode normalization, character filtering |
| **Image perturbation** | Slight crop, color shift | Perceptual hashing, robust augmentations in training |
| **Rephrasing** | "K1ll" instead of "Kill" | Character n-grams, phonetic matching, ensemble |
| **Context stripping** | Screenshot of article out of context | Context-aware models, source attribution |
| **Multi-modal evasion** | image OK, caption violates | Multimodal fusion, cross-modal attention |

### Proactive vs Reactive Detection

- **Proactive:** Scan content at upload; block before publication
- **Reactive:** User reports → queue for review; retroactive scanning of viral content
- **Hybrid:** Proactive for high-risk (new accounts, sensitive topics); reactive for reports + periodic batch scans

---

## 📈 Metrics & Evaluation

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision (by policy)** | Of flagged content, % truly violating | > 95% for borderline; > 99% for auto-remove |
| **Recall (by policy)** | Of true violations, % caught | > 90% for severe (violence, self-harm) |
| **False Positive Rate** | Legitimate content wrongly removed | < 0.1% |
| **Time-to-Action** | Time from upload to decision | < 1s critical; < 30s standard |
| **Appeal Overturn Rate** | % of appeals that overturn decision | < 5% (indicates over-enforcement if high) |
| **Human Review SLA** | % of queue reviewed within 24h | > 95% |
| **Reviewer Agreement** | Inter-rater reliability (Cohen's kappa) | > 0.7 |

### Precision/Recall Trade-offs by Policy

| Policy | Preferred | Rationale |
|--------|-----------|-----------|
| Child safety, Terrorism | High recall | Miss = severe harm |
| Borderline hate, Misinformation | High precision | False positive = censorship concern |
| Spam | Balance | Volume allows some FP |

### Offline Evaluation Script

```python
def evaluate_moderation_model(
    predictions: List[Dict[str, float]],
    labels: List[Dict[str, bool]],
    policies: List[str]
) -> Dict[str, Dict[str, float]]:
    """Evaluate per-policy precision, recall, F1."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    results = {}
    for policy in policies:
        y_true = [l.get(policy, False) for l in labels]
        y_pred = [p.get(policy, 0) >= 0.5 for p in predictions]
        
        results[policy] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
    return results
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Precision vs Recall** | High precision (less over-removal) | High recall (catch more violations) |
| **Proactive vs Reactive** | Scan at upload (block before publish) | React to reports (less compute) |
| **Model complexity** | Simple rules + heuristics (interpretable) | Deep learning (higher accuracy) |
| **Human-in-loop** | More human review (quality) | More automation (scale, cost) |
| **Latency vs coverage** | Fast path for known bad only | Full ML for all (thorough) |
| **Multimodal fusion** | Early fusion (joint reasoning) | Late fusion (modular, easier to update) |
| **Hash vs ML** | Hash for exact known bad (zero FP) | ML for variants (broader catch) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you balance precision and recall for different policy types?
2. How would you handle adversarial attacks (Unicode, image perturbation)?
3. How do you scale to billions of pieces of content per day?
4. How do you support 100+ languages?
5. How do you prevent reviewer burnout from viewing harmful content?
6. How would you detect new policy violations without labeled data?
7. How do you handle the "borderline" zone between allow and remove?

**Key Points to Mention:**
- Multi-stage pipeline: hash → ML → policy engine → decision
- Per-policy thresholds (high recall for severe, high precision for borderline)
- Human-in-the-loop with prioritization and reviewer well-being (exposure limits, content warnings)
- Multimodal fusion (text + image + video + audio)
- Feedback loop from appeals and human review to retraining
- Adversarial robustness: normalization, augmentation, ensemble
- Proactive vs reactive detection trade-offs

---

## 🔗 Related Topics

- [NLP Systems](./05-nlp-systems.md)
- [Computer Vision Systems](./04-computer-vision-systems.md)
- [Fairness & Responsible AI](../../phase-5-advanced-topics/13-fairness-responsible-ai/00-README.md)
- [Model Monitoring](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md)
- [A/B Testing](../../phase-2-core-components/05-model-serving/03-ab-testing.md)
