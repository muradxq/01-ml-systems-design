# Ad Click Prediction

## Overview

Ad click prediction (pCTR) is the **#1 most-asked Meta ML system design question**. The system predicts the probability that a user will click on an ad, enabling optimal ad allocation through real-time auctions. Meta's ad system serves **billions of predictions per day** and generates the majority of company revenue. A well-calibrated pCTR model is critical—overestimating click probability hurts user experience and advertiser ROI; underestimating loses revenue.

---

## 🎯 Problem Definition

### Business Goals

- **Maximize ad revenue:** Show ads with highest expected value (bid × pCTR)
- **Maintain user experience:** Don't overload with irrelevant ads; relevance drives engagement
- **Advertiser satisfaction:** Accurate predictions enable better budget allocation and ROI
- **Platform value:** Balance short-term revenue with long-term user retention

### Requirements

| Requirement | Specification | Scale Context |
|-------------|---------------|---------------|
| **Latency** | < 50ms p99 | Per-ad request; millions concurrent |
| **Throughput** | Billions of predictions/day | ~100K-1M QPS peak |
| **Calibration** | Well-calibrated probabilities | Critical for auction fairness |
| **Freshness** | Model retrained every 4-24 hours | Data freshness < 1 hour |
| **Availability** | 99.99% uptime | Revenue-critical system |
| **Scale** | 2B+ users, millions of advertisers | ~1M ad requests/second |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Ad Click Prediction System (Meta-scale)                     │
│                                                                                   │
│  ┌────────────────┐                                                               │
│  │  Ad Request    │  User opens feed, scrolls, or visits page                     │
│  │  (User ID,     │  Request context: device, placement, session                  │
│  │   Context)     │                                                               │
│  └───────┬────────┘                                                               │
│          │                                                                        │
│          ▼                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  AD RETRIEVAL (10-20ms)                                           │            │
│  │  ┌─────────────────┐  ┌─────────────────┐                        │            │
│  │  │ Targeting Rules │  │ Candidate       │                        │            │
│  │  │ (audience,      │  │ Selection       │  Hundreds→Thousands    │            │
│  │  │  placement)     │  │ (ANN, rules)    │  of ad candidates       │            │
│  │  └────────┬────────┘  └────────┬────────┘                        │            │
│  └───────────┼───────────────────┼──────────────────────────────────┘            │
│              └───────────────────┼                                               │
│                                  ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  FEATURE ENRICHMENT (5-10ms)                                      │            │
│  │  ┌─────────────────────────────────────────────────────────┐    │            │
│  │  │  Feature Store Lookup                                     │    │            │
│  │  │  User features | Ad features | Cross features | Context   │    │            │
│  │  └─────────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────────────────────────────────────────────┘            │
│                                  │                                               │
│                                  ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  CLICK PREDICTION MODEL (15-25ms)                                 │            │
│  │  ┌─────────────────┐  ┌─────────────────┐                        │            │
│  │  │ pCTR Model      │  │ pCVR Model      │  Multi-task: click +   │            │
│  │  │ (DCN, Wide&Deep)│  │ (conversion)    │  conversion prediction │            │
│  │  └────────┬────────┘  └────────┬────────┘                        │            │
│  │           └──────────┬─────────┘                                  │            │
│  │                      ▼                                            │            │
│  │  ┌─────────────────────────────────────────┐                     │            │
│  │  │  Calibration Module (Platt/Isotonic)     │  Critical for auc-  │            │
│  │  └─────────────────────────────────────────┘  tion fairness      │            │
│  └──────────────────────────────────────────────────────────────────┘            │
│                                  │                                               │
│                                  ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐            │
│  │  AUCTION (5-10ms)                                                 │            │
│  │  eCPM = bid × pCTR (or bid × pCVR for oCPM)                       │            │
│  │  Second-price auction | Bid modification | Reserve price          │            │
│  └──────────────────────────────────────────────────────────────────┘            │
│                                  │                                               │
│                                  ▼                                               │
│  ┌────────────────┐     ┌────────────────┐                                        │
│  │  Ad Selection  │────▶│  LOGGING       │  Impression, click, conversion         │
│  │  Top-ranked    │     │  (async)       │  logs for model retraining             │
│  └────────────────┘     └────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Data Collection

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Data Streams (Scale: Petabytes/day)                                     │
│                                                                         │
│  Impression Logs ──▶ (user_id, ad_id, timestamp, position, context)     │
│  Click Logs     ──▶ (impression_id, click_timestamp, dwell_time)         │
│  Conversion Logs ─▶ (click_id, conversion_timestamp, value) ← DELAYED   │
│                     Attribution window: 1d view, 7d click typical       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Delayed Feedback Handling:** Conversions can arrive 1-30 days after click. Solutions:
- **Waiting:** Train only on "final" labels (high latency)
- **Positive-Unlabeled (PU):** Treat no-conversion-within-window as negative
- **Delayed Feedback Model (DFM):** Model delay distribution, correct during training
- **Real-time + Delayed Model:** Two-model approach for immediate vs delayed signals

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import json

@dataclass
class ImpressionLog:
    """Ad impression event—base for training data."""
    impression_id: str
    user_id: str
    ad_id: str
    advertiser_id: str
    timestamp: datetime
    position: int  # 0-indexed; critical for position bias
    placement: str  # feed, story, right_column
    device: str
    session_id: str
    experiment_id: Optional[str] = None

@dataclass 
class ClickLog:
    """Click event—positive signal."""
    impression_id: str
    click_timestamp: datetime
    dwell_time_ms: Optional[float] = None

@dataclass
class ConversionLog:
    """Conversion event—delayed by 1-30 days."""
    click_id: str
    conversion_timestamp: datetime
    conversion_value: float
    conversion_type: str  # purchase, signup, etc.

class DelayedFeedbackProcessor:
    """Handle delayed conversions for training data."""
    
    def __init__(self, attribution_window_days: int = 7):
        self.attribution_window = timedelta(days=attribution_window_days)
    
    def create_training_example(
        self,
        impression: ImpressionLog,
        clicked: bool,
        converted: Optional[bool] = None,
        observation_timestamp: Optional[datetime] = None
    ) -> dict:
        """Create training example with proper label handling."""
        obs_time = observation_timestamp or datetime.utcnow()
        
        # Positive if clicked (and optionally converted)
        # Use PU learning: if no conversion within window, treat as unknown negative
        if clicked:
            if converted is None:
                # Still in attribution window—use as positive for click
                label = 1.0
            else:
                label = 1.0 if converted else 0.0
        else:
            label = 0.0
            
        return {
            "impression_id": impression.impression_id,
            "user_id": impression.user_id,
            "ad_id": impression.ad_id,
            "position": impression.position,
            "label": label,
            "observation_time": obs_time.isoformat()
        }
```

### 2. Feature Engineering

| Category | Examples | Freshness |
|----------|----------|-----------|
| **User** | Demographics, browsing history, past ad interactions, interest embeddings | Real-time where possible |
| **Ad** | Creative type, text embeddings, advertiser quality, historical CTR | Cached (hourly refresh) |
| **Context** | Time, device, page placement, session depth, connection quality | Real-time |
| **Cross** | User-ad affinity, user-advertiser interaction history | Precomputed + real-time |

```python
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class AdClickFeatureEngineer:
    """Feature engineering for ad click prediction."""
    
    def get_user_features(self, user_id: str) -> Dict[str, float]:
        """User-level features from feature store."""
        return {
            "user_ctr_7d": 0.02,  # Historical CTR
            "user_ctr_30d": 0.018,
            "user_impressions_7d": 150.0,
            "user_clicks_7d": 3.0,
            "user_age_bucket": 2,  # 25-34
            "user_gender": 1,
            "user_interest_embedding_0": 0.1,  # Sparse/dense embedding dims
            "user_interest_embedding_1": -0.2,
            "user_engagement_score": 0.65,
            "user_ad_frequency_cap_remaining": 0.8,  # How many more ads user can see
        }
    
    def get_ad_features(self, ad_id: str) -> Dict[str, float]:
        """Ad-level features."""
        return {
            "ad_ctr_7d": 0.025,
            "ad_ctr_30d": 0.022,
            "ad_impressions_7d": 10000.0,
            "advertiser_quality_score": 0.8,
            "ad_creative_type": 1,  # Image=0, Video=1, Carousel=2
            "ad_text_embedding_0": 0.05,
            "landing_page_quality": 0.9,
            "ad_engagement_rate": 0.03,
        }
    
    def get_context_features(self, context: Dict) -> Dict[str, float]:
        """Real-time context features."""
        now = datetime.utcnow()
        return {
            "hour_of_day": now.hour / 24.0,
            "day_of_week": now.weekday() / 7.0,
            "is_weekend": int(now.weekday() >= 5),
            "device_type": context.get("device_type", "mobile") == "mobile",
            "placement": self._placement_encoding(context.get("placement", "feed")),
            "session_depth": min(context.get("session_depth", 0) / 20.0, 1.0),
            "connection_wifi": int(context.get("connection", "wifi") == "wifi"),
        }
    
    def get_cross_features(
        self,
        user_id: str,
        ad_id: str,
        advertiser_id: str
    ) -> Dict[str, float]:
        """User-ad and user-advertiser affinity."""
        return {
            "user_advertiser_click_rate": 0.03,  # This user + this advertiser
            "user_advertiser_impressions": 5.0,
            "user_category_affinity": 0.7,  # Ad category vs user interests
            "embedding_similarity": 0.65,  # User embedding · ad embedding
        }
    
    def _placement_encoding(self, placement: str) -> float:
        encodings = {"feed": 0.0, "story": 0.5, "right_column": 1.0}
        return encodings.get(placement, 0.0)
    
    def build_feature_vector(
        self,
        user_id: str,
        ad_id: str,
        advertiser_id: str,
        context: Dict
    ) -> np.ndarray:
        """Build complete feature vector for model."""
        user_f = self.get_user_features(user_id)
        ad_f = self.get_ad_features(ad_id)
        ctx_f = self.get_context_features(context)
        cross_f = self.get_cross_features(user_id, ad_id, advertiser_id)
        
        all_features = {**user_f, **ad_f, **ctx_f, **cross_f}
        return np.array(list(all_features.values()), dtype=np.float32)
```

### 3. Model Architecture: DCN (Deep & Cross Network)

```python
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np

class ClickPredictionModel(nn.Module):
    """
    DCN-style model for ad click prediction.
    Combines explicit feature crosses (Cross Network) with DNN for implicit patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        cross_layer_num: int = 3,
        dnn_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Cross network: explicitly models feature interactions
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(cross_layer_num)
        ])
        
        # DNN for implicit patterns
        dnn_layers = []
        prev_dim = input_dim
        for hidden_dim in dnn_hidden_dims:
            dnn_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.dnn = nn.Sequential(*dnn_layers)
        
        # Output: combine cross + dnn
        self.output_layer = nn.Linear(input_dim + prev_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cross network: x_{l+1} = x_0 * x_l^T w_l + b_l + x_l
        x_cross = x
        for layer in self.cross_layers:
            x_cross = x * layer(x_cross) + x_cross
        
        # DNN path
        x_dnn = self.dnn(x)
        
        # Concatenate and predict
        combined = torch.cat([x_cross, x_dnn], dim=1)
        logits = self.output_layer(combined)
        return torch.sigmoid(logits)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Inference: returns calibrated probabilities."""
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            return self.forward(x_t).numpy().flatten()
```

### 4. Calibration Module

**Why critical:** Raw model outputs are often poorly calibrated. Auction uses pCTR directly in eCPM; miscalibration biases the auction.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Literal

class CalibrationModule:
    """
    Post-hoc calibration for click probabilities.
    Platt scaling (logistic) or Isotonic regression.
    """
    
    def __init__(self, method: Literal["platt", "isotonic"] = "isotonic"):
        self.method = method
        self.calibrator = (
            LogisticRegression() if method == "platt" 
            else IsotonicRegression(out_of_bounds="clip")
        )
        
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Fit calibrator on validation set."""
        if self.method == "platt":
            # Platt needs 2D input for sklearn
            self.calibrator.fit(y_pred.reshape(-1, 1), y_true)
        else:
            self.calibrator.fit(y_pred, y_true)
        return self
    
    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply calibration to predicted probabilities."""
        if self.method == "platt":
            return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return self.calibrator.predict(y_pred)
    
    def evaluate_calibration(self, y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_pred)
        
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_pred = y_pred[mask].mean()
                avg_true = y_true[mask].mean()
                ece += mask.sum() * np.abs(avg_pred - avg_true)
        
        return ece / total
```

### 5. Auction Ranker

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class AdCandidate:
    ad_id: str
    advertiser_id: str
    bid_cpm: float  # Bid in cost-per-mille (thousand impressions)
    pctr: float
    pcvr: Optional[float] = None  # For oCPM (optimize for conversions)
    ad_creative: dict = None

class AuctionRanker:
    """
    Rank ads by eCPM for second-price auction.
    eCPM = bid × pCTR (for CPC) or bid × pCVR (for oCPM)
    """
    
    def __init__(
        self,
        objective: str = "click",  # "click" or "conversion"
        reserve_price: float = 0.01,
        bid_modification: Optional[float] = None  # e.g., 1.2 for 20% boost
    ):
        self.objective = objective
        self.reserve_price = reserve_price
        self.bid_modification = bid_modification or 1.0
    
    def compute_ecpm(self, ad: AdCandidate) -> float:
        """Expected cost per mille."""
        bid = ad.bid_cpm * self.bid_modification
        if self.objective == "click":
            return bid * ad.pctr
        else:  # conversion
            return bid * (ad.pcvr or ad.pctr)  # Fallback to pCTR
        
    def run_auction(
        self,
        candidates: List[AdCandidate],
        num_slots: int = 3
    ) -> List[tuple]:
        """
        Second-price auction: winner pays next-highest bid.
        Returns list of (ad, price_to_pay) for selected ads.
        """
        # Rank by eCPM
        scored = [(ad, self.compute_ecpm(ad)) for ad in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (ad, ecpm) in enumerate(scored[:num_slots]):
            # Second price: pay next highest eCPM (or reserve)
            if i + 1 < len(scored):
                next_ecpm = scored[i + 1][1]
                # Convert eCPM back to price (simplified)
                price = max(self.reserve_price, next_ecpm / ad.pctr) if ad.pctr > 0 else 0
            else:
                price = max(self.reserve_price, self.reserve_price)
            results.append((ad, price))
        
        return results
```

### 6. Training Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Delayed conversions** | DFM, PU learning, or two-model approach |
| **Position bias** | Inverse propensity scoring (IPS): weight = 1/P(show \| position) |
| **Class imbalance** | Negative downsampling (e.g., 1:10) + calibration correction (scale up probs) |
| **Training on logged data** | Counterfactual: importance weighting, doubly robust estimation |
| **Calibration** | Platt scaling or isotonic regression on held-out set |

```python
def inverse_propensity_weight(position: int, propensity_scores: dict) -> float:
    """
    Position bias: items at top get more clicks regardless of relevance.
    Weight training examples by 1/P(shown at position) to debias.
    """
    # Propensity = probability ad was shown at this position (from logging)
    propensity = propensity_scores.get(position, 0.1)  # Default if unknown
    return 1.0 / max(propensity, 1e-6)

def correct_downsampling_probability(
    raw_prob: float,
    downsampling_ratio: float  # e.g., 0.1 for 1:10 neg sampling
) -> float:
    """Correct probability after negative downsampling during training."""
    # p_corrected = p_raw / (p_raw + (1 - p_raw) / ratio)
    return raw_prob / (raw_prob + (1 - raw_prob) / downsampling_ratio)
```

---

## 📈 Metrics & Evaluation

### Online Metrics

| Metric | Description | Target (Typical) |
|--------|-------------|-----------------|
| **Revenue** | Total ad revenue | Primary business KPI |
| **CTR** | Clicks / Impressions | 1-3% (varies by placement) |
| **User Satisfaction** | Surveys, hide/ad relevance reports | Minimize negative feedback |
| **Ad Load** | Ads per content unit | Balance revenue vs experience |
| **Advertiser ROI** | Conversions / spend | Retention metric |

### Offline Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **AUC** | Ranking quality | > 0.75 (0.8+ for mature systems) |
| **Log Loss** | Probability quality | Lower is better |
| **Calibration Error (ECE)** | Probability calibration | < 0.01 |
| **Calibration Plot** | Predicted vs actual by bin | Diagonal line |

```python
def compute_offline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    from sklearn.metrics import roc_auc_score, log_loss
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred),
        "ece": CalibrationModule(method="isotonic").evaluate_calibration(y_pred, y_true)
    }
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Model complexity** | Logistic regression (fast, interpretable) | DCN/Transformer (higher AUC) | Start simple, scale to DCN at Meta-scale |
| **Calibration** | No calibration | Platt/Isotonic | Always calibrate for auction systems |
| **Negative sampling** | 1:1 (slow) | 1:10 or 1:100 (fast) | Use downsampling + correction |
| **Feature freshness** | Batch daily | Real-time streaming | Hybrid: hot user features real-time |
| **Delayed feedback** | Wait full window | DFM/PU learning | DFM for faster iteration |
| **Multi-task** | pCTR only | pCTR + pCVR | Multi-task for oCPM advertisers |
| **Latency vs candidates** | 100 candidates, 50ms | 1000 candidates, 50ms | Tune cascade; more candidates = more revenue |

---

## 🎤 Interview Tips

### What to Emphasize

1. **Calibration is critical**—raw model outputs can't drive the auction fairly.
2. **Position bias**—always mention IPS or similar debiasing.
3. **Delayed feedback**—conversions arrive late; DFM or two-model approach.
4. **Scale numbers**—billions of predictions/day, <50ms latency, petabytes of logs.
5. **Multi-objective**—pCTR for CPC, pCVR for oCPM, sometimes both.

### Common Follow-ups

1. **How would you handle a sudden drop in CTR?** — Check calibration drift, feature pipeline, model staleness, position bias changes.
2. **How do you A/B test a new pCTR model?** — Shadow mode first; then small % traffic; compare revenue, CTR, calibration.
3. **How would you reduce latency?** — Feature precomputation, model distillation, candidate reduction, caching.
4. **How do you prevent feedback loops?** — Diversify exploration; avoid only showing high-pCTR ads (explore/exploit).
5. **What if advertisers game the system?** — Click fraud detection, quality signals, landing page quality score.

### Red Flags to Avoid

- Ignoring calibration
- Not addressing position bias
- Forgetting delayed conversions
- Underestimating scale (say "billions" not "millions")

---

## 🔗 Related Topics

- [Recommendation Systems](./01-recommendation-systems.md) — Similar two-stage retrieval + ranking
- [Feature Stores](../../phase-2-core-components/03-feature-engineering/01-feature-stores.md) — Feature enrichment at scale
- [A/B Testing](../../phase-2-core-components/05-model-serving/03-ab-testing.md) — Model experimentation
- [Model Monitoring](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md) — Drift, calibration drift
