# Notification Ranking System

## Overview

Notification ranking is a critical ML system at **Meta, Google, and other engagement-driven platforms**. The system decides which notifications to send, when to send them, and through which channel (push, email, in-app)—balancing re-engagement with avoiding notification fatigue. A poorly designed system either spams users (leading to uninstalls and opt-outs) or under-notifies (missing re-engagement opportunities). The goal is to maximize long-term user value: app opens, retention, and satisfaction, while respecting user preferences and volume constraints. This system is inherently **multi-objective** and **real-time**.

---

## 🎯 Problem Definition

### Business Goals

- **Re-engage users:** Bring back churned or dormant users through timely, relevant notifications
- **Drive app opens:** Increase DAU (Daily Active Users) and session frequency
- **Avoid notification fatigue:** Prevent users from muting, disabling, or uninstalling due to excessive notifications
- **Maximize long-term retention:** Not just immediate opens, but sustained engagement
- **Respect user preferences:** Honor explicit settings (quiet hours, channel preferences) and learn implicit preferences

### Requirements

| Requirement | Specification | Scale Context |
|-------------|---------------|---------------|
| **Personalization** | Per-user, per-notification-type | Billions of users |
| **Multi-channel** | Push, email, in-app, SMS | Channel-specific logic |
| **Real-time** | Event-to-notification < 1 min (urgent) | Event streams, millions/min |
| **Volume control** | Per-user daily/weekly budget | Fatigue modeling |
| **Send-time optimization** | When user likely to engage | Time zones, usage patterns |
| **Freshness** | Notify while content/event is relevant | Stale notification = bad UX |
| **Availability** | 99.9%+ | Critical for engagement |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        Notification Ranking System                                         │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  EVENT TRIGGERS                                                                   │  │
│  │  Friend activity | Content update | Milestone | Security | Direct message          │  │
│  └───────┬──────────────────────────────────────────────────────────────────────────┘  │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  NOTIFICATION GENERATION                                                      │       │
│  │  Event → Template selection | Content assembly | Recipient resolution         │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  CANDIDATE POOL                                                              │       │
│  │  Pending notifications for user (from multiple event streams)                │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  VOLUME CONTROLLER (Per-user budget)                                         │       │
│  │  Daily/weekly caps | Fatigue modeling | User preference learning             │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  RANKING MODEL                                                               │       │
│  │  P(open | sent), value, urgency, relevance                                    │       │
│  │  Multi-objective: open rate vs satisfaction vs long-term retention           │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  CHANNEL SELECTION                                                           │       │
│  │  Push (highest urgency) | Email (digest) | In-app (next visit) | SMS (rare)  │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  SEND-TIME OPTIMIZATION                                                      │       │
│  │  When is user most likely to engage? Time zone, usage patterns, survival model│       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  DELIVERY                                                                     │       │
│  │  Push service | Email service | In-app feed                                   │       │
│  └───────┬──────────────────────────────────────────────────────────────────────┘       │
│          │                                                                               │
│          ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  FEEDBACK LOOP                                                               │       │
│  │  Open, click, dismiss, mute, unsubscribe → Retraining                         │       │
│  └──────────────────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Notification Types and Priorities

| Priority | Type | Examples | Channel | Typical Budget |
|----------|------|----------|---------|----------------|
| **High** | Direct messages, security alerts, time-sensitive | "Your friend sent a message", "Login from new device" | Push, SMS | Always allow |
| **Medium** | Friend activities, content recommendations | "X liked your post", "New video from Y" | Push, In-app | 3-5/day |
| **Low** | Digests, feature announcements | "Your weekly summary", "New feature" | Email, In-app | 1-2/week |

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

class NotificationType(Enum):
    DIRECT_MESSAGE = "direct_message"
    FRIEND_ACTIVITY = "friend_activity"
    CONTENT_UPDATE = "content_update"
    MILESTONE = "milestone"
    SECURITY_ALERT = "security_alert"
    DIGEST = "digest"
    PROMOTIONAL = "promotional"

class NotificationPriority(Enum):
    HIGH = 3    # Bypass budget for critical
    MEDIUM = 2
    LOW = 1

@dataclass
class Notification:
    notification_id: str
    user_id: str
    type: NotificationType
    priority: NotificationPriority
    title: str
    body: str
    created_at: datetime
    # Context for ranking
    sender_id: Optional[str] = None
    content_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    expires_at: Optional[datetime] = None
```

### 2. Ranking Model

```python
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class NotificationFeatures:
    """Features for notification ranking model."""
    notification_type: int
    hours_since_created: float
    sender_relationship_strength: float  # 0-1, how strong is connection
    content_relevance: float  # User-content affinity
    user_engagement_7d: float  # How active user has been
    similar_notification_open_rate: float
    time_since_last_notification_hours: float
    is_quiet_hours: int
    user_preference_score: float  # Learned from explicit + implicit signals

class NotificationRanker:
    """
    Ranks notifications by expected value. Primary objective: P(open | sent).
    Secondary: long-term retention impact, satisfaction (avoid fatigue).
    """
    
    def __init__(self, weights: Optional[dict] = None):
        # Learned weights from offline training (e.g., GBDT or linear)
        self.weights = weights or {
            "notification_type": 0.1,
            "recency": 0.15,
            "sender_strength": 0.25,
            "content_relevance": 0.2,
            "engagement_7d": -0.05,  # Less active users get higher score
            "similar_open_rate": 0.2,
            "time_since_last": 0.1,  # Spread out notifications
            "quiet_hours": -0.5,     # Penalize during quiet hours
            "user_preference": 0.2,
        }
    
    def score(self, features: NotificationFeatures) -> float:
        """Compute relevance score for a notification."""
        vec = [
            features.notification_type / 10.0,
            min(1.0, features.hours_since_created / 24),
            features.sender_relationship_strength,
            features.content_relevance,
            features.user_engagement_7d,
            features.similar_notification_open_rate,
            min(1.0, features.time_since_last_notification_hours / 24),
            float(features.is_quiet_hours),
            features.user_preference_score,
        ]
        return sum(w * v for w, v in zip(self.weights.values(), vec))
    
    def rank(
        self,
        notifications: List[tuple],  # (Notification, NotificationFeatures)
        top_k: int = 5
    ) -> List[tuple]:
        """Rank and return top-k (notification, features) by score."""
        scored = [(n, f, self.score(f)) for n, f in notifications]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(n, f) for n, f, _ in scored[:top_k]]
```

### 3. Volume Controller

```python
from datetime import datetime, timedelta
from typing import Dict, Optional

class VolumeController:
    """
    Per-user notification budget. Prevents fatigue.
    Uses diminishing returns: more notifications → lower marginal value.
    """
    
    def __init__(
        self,
        daily_push_budget: int = 5,
        weekly_push_budget: int = 20,
        daily_email_budget: int = 1,
        fatigue_decay: float = 0.8  # Each additional notif worth 80% of previous
    ):
        self.daily_push = daily_push_budget
        self.weekly_push = weekly_push_budget
        self.daily_email = daily_email_budget
        self.fatigue_decay = fatigue_decay
    
    def get_budget_remaining(
        self,
        user_id: str,
        sent_counts: Dict[str, Dict[str, int]],  # user_id -> {date: count}
        channel: str = "push"
    ) -> int:
        """How many more notifications can we send today?"""
        counts = sent_counts.get(user_id, {})
        today = datetime.utcnow().date().isoformat()
        daily = counts.get(today, 0)
        budget = self.daily_push if channel == "push" else self.daily_email
        return max(0, budget - daily)
    
    def apply_fatigue_penalty(
        self,
        base_score: float,
        num_already_sent_today: int
    ) -> float:
        """
        Diminishing returns: first notification = 1.0, second = 0.8, third = 0.64...
        """
        return base_score * (self.fatigue_decay ** num_already_sent_today)
    
    def should_send(
        self,
        user_id: str,
        notification_priority: NotificationPriority,
        sent_counts: Dict[str, Dict[str, int]],
        channel: str
    ) -> bool:
        """Check if we have budget and should send."""
        if notification_priority == NotificationPriority.HIGH:
            return True  # Security, DMs always go through
        remaining = self.get_budget_remaining(user_id, sent_counts, channel)
        return remaining > 0
```

### 4. Channel Selection

```python
class ChannelSelector:
    """
    Chooses channel: Push, Email, In-app, SMS.
    Push: highest urgency, limited budget, immediate
    Email: digest-friendly, longer content, lower urgency
    In-app: seen on next visit, no fatigue
    SMS: very high urgency only (cost, friction)
    """
    
    def __init__(self):
        self.channel_priority = {
            NotificationType.DIRECT_MESSAGE: ["push"],
            NotificationType.SECURITY_ALERT: ["push", "sms"],
            NotificationType.FRIEND_ACTIVITY: ["push", "in_app"],
            NotificationType.CONTENT_UPDATE: ["push", "in_app"],
            NotificationType.MILESTONE: ["push", "in_app"],
            NotificationType.DIGEST: ["email", "in_app"],
            NotificationType.PROMOTIONAL: ["email", "in_app"],
        }
    
    def select_channel(
        self,
        notification: Notification,
        user_preferences: Dict[str, str],  # user's channel prefs
        channel_budgets_remaining: Dict[str, int]
    ) -> str:
        """Select best available channel."""
        candidates = self.channel_priority.get(
            notification.type, 
            ["in_app"]
        )
        for ch in candidates:
            if user_preferences.get(ch, "on") != "off":
                if channel_budgets_remaining.get(ch, 0) > 0:
                    return ch
        return "in_app"  # Fallback: always available
```

### 5. Send-Time Optimization

```python
import math
from typing import List, Tuple

class SendTimeOptimizer:
    """
    Predicts when user is most likely to engage.
    Uses: time zone, historical open times, device usage patterns.
    Survival model: P(open in next hour | not yet opened).
    """
    
    def __init__(self):
        # Per-user model: distribution of open times (hour of day, day of week)
        # In production: trained survival model or simple histogram
        self.default_peak_hours = [9, 12, 18, 21]  # Common engagement peaks
    
    def get_best_send_time(
        self,
        user_id: str,
        user_timezone: str,
        user_open_histogram: Optional[Dict[int, float]] = None,
        max_delay_hours: float = 24.0
    ) -> datetime:
        """
        Returns optimal send time. For urgent notifications, send immediately.
        For non-urgent, delay until predicted peak.
        """
        now = datetime.utcnow()
        
        if user_open_histogram:
            best_hour = max(
                user_open_histogram.keys(),
                key=lambda h: user_open_histogram[h]
            )
            # Schedule for next occurrence of best hour
            target = now.replace(hour=best_hour, minute=0, second=0)
            if target <= now:
                target += timedelta(days=1)
            return target
        else:
            # Default: send within next 2 hours during typical peak
            for hour in self.default_peak_hours:
                target = now.replace(hour=hour, minute=0, second=0)
                if target > now and (target - now).total_seconds() / 3600 <= max_delay_hours:
                    return target
            return now  # Send now if no good slot
    
    def should_delay(
        self,
        notification_priority: NotificationPriority,
        hours_until_peak: float
    ) -> bool:
        """For low-priority, delay to peak. For high, send now."""
        if notification_priority == NotificationPriority.HIGH:
            return False
        if notification_priority == NotificationPriority.LOW and hours_until_peak < 4:
            return True
        return False
```

### 6. End-to-End Notification Pipeline

```python
class NotificationPipeline:
    """
    Orchestrates the full notification flow: volume control → rank → channel → send-time.
    """
    
    def __init__(
        self,
        ranker: NotificationRanker,
        volume_controller: VolumeController,
        channel_selector: ChannelSelector,
        send_time_optimizer: SendTimeOptimizer
    ):
        self.ranker = ranker
        self.volume_controller = volume_controller
        self.channel_selector = channel_selector
        self.send_time_optimizer = send_time_optimizer
    
    def process(
        self,
        user_id: str,
        candidates: List[tuple],  # (Notification, NotificationFeatures)
        sent_counts: Dict[str, Dict[str, int]],
        user_preferences: Dict,
        user_timezone: str = "UTC"
    ) -> List[dict]:
        """
        Process candidate notifications. Returns list of {notification, channel, send_time}.
        """
        results = []
        
        # Filter by volume budget
        for notif, feats in candidates:
            channel = self.channel_selector.select_channel(
                notif, user_preferences, 
                {"push": 5, "email": 1, "in_app": 999}
            )
            if not self.volume_controller.should_send(
                user_id, notif.priority, sent_counts, channel
            ):
                continue
            
            # Apply fatigue penalty to score
            daily_count = sent_counts.get(user_id, {}).get(
                datetime.utcnow().date().isoformat(), 0
            )
            base_score = self.ranker.score(feats)
            adjusted_score = self.volume_controller.apply_fatigue_penalty(
                base_score, daily_count
            )
            results.append((notif, feats, channel, adjusted_score))
        
        # Rank by adjusted score
        results.sort(key=lambda x: x[3], reverse=True)
        
        # Select top, compute send time
        output = []
        for notif, feats, channel, _ in results[:5]:
            send_time = self.send_time_optimizer.get_best_send_time(
                user_id, user_timezone
            )
            output.append({
                "notification": notif,
                "channel": channel,
                "send_time": send_time
            })
        
        return output
```

### 7. Multi-Objective Optimization

**Objectives:**
- **Open rate:** P(open | sent) — short-term engagement
- **Satisfaction:** Minimize mute/unsubscribe — long-term trust
- **Retention impact:** Does this notification improve 7d/30d retention?

**Approaches:**
1. **Weighted sum:** `score = w1*open_prob + w2*satisfaction - w3*fatigue_cost`
2. **Pareto optimization:** Maintain frontier of models; choose by product preference
3. **Constraint optimization:** Maximize opens s.t. unsubscribe rate < threshold

```python
def multi_objective_score(
    open_prob: float,
    retention_lift: float,
    fatigue_penalty: float,
    weights: tuple = (0.5, 0.3, 0.2)
) -> float:
    """
    Combine objectives. retention_lift: expected DAU delta.
    fatigue_penalty: increasing with recent send rate.
    """
    w1, w2, w3 = weights
    return w1 * open_prob + w2 * retention_lift - w3 * fatigue_penalty
```

---

## 📈 Metrics & Evaluation

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Open Rate** | Opens / Notifications Sent | 5-15% (varies by type) |
| **Click-Through Rate (CTR)** | Clicks / Opens | 20-40% |
| **Unsubscribe/Mute Rate** | Users disabling / muting | < 1% monthly |
| **App DAU Impact** | DAU lift from notification cohort | Positive incremental |
| **Long-term Retention** | 7d/30d retention of notified vs control | Net positive |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Time to open** | Distribution of open latency |
| **Send-time accuracy** | Opens within 1h of send / total opens |
| **Channel mix** | % push vs email vs in-app |
| **Fatigue curves** | Open rate vs notifications/day (diminishing returns) |

### Offline Evaluation

```python
def evaluate_notification_model(
    predictions: List[float],  # Predicted open probability
    labels: List[int],         # Actual opened (1) or not (0)
    sent_timestamps: List[datetime],
    open_timestamps: List[Optional[datetime]]
) -> dict:
    from sklearn.metrics import roc_auc_score, log_loss
    
    return {
        "auc": roc_auc_score(labels, predictions),
        "log_loss": log_loss(labels, predictions),
        "open_rate": sum(labels) / len(labels),
        # Send-time: % of opens within 1h of send
        "quick_open_rate": _compute_quick_open(sent_timestamps, open_timestamps),
    }

def _compute_quick_open(sent: List, opened: List) -> float:
    quick = sum(1 for s, o in zip(sent, opened) 
                if o and (o - s).total_seconds() < 3600)
    total_opens = sum(1 for o in opened if o)
    return quick / max(1, total_opens)
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Volume control** | Fixed daily cap | Fatigue-modeled (diminishing returns) | Fatigue model for better UX |
| **Send-time** | Send immediately | Optimize for peak | Optimize for non-urgent; immediate for urgent |
| **Channel** | Push for everything | Strict channel rules | Type-based + user pref |
| **Ranking objective** | Open rate only | Multi-objective (open + retention + satisfaction) | Multi-objective |
| **Personalization** | Global model | Per-user or per-segment | Per-segment minimum; per-user when data allows |
| **Real-time vs batch** | Real-time for all | Batch for digests | Real-time for urgent; batch for digests |
| **Exploration** | Greedy (highest score) | Epsilon-greedy / Thompson | Some exploration for learning |

---

## 🎤 Interview Tips

### What to Emphasize

1. **Multi-objective**—open rate vs fatigue vs long-term retention
2. **Volume control**—per-user budget and fatigue modeling
3. **Channel selection**—push for urgent, email for digest
4. **Send-time optimization**—when will user engage?
5. **Feedback loop**—mute/unsubscribe as negative signal for retraining

### Common Follow-ups

1. **How do you prevent notification fatigue?** — Per-user budget, diminishing returns, respect quiet hours, learn from mute/unsubscribe
2. **How do you handle time zones?** — Store user timezone; send during their local peak hours; avoid night sends
3. **How do you A/B test a new ranking model?** — Holdout on open rate, but also watch unsubscribe and long-term retention
4. **How do you balance open rate vs user satisfaction?** — Multi-objective scoring; constraint: keep unsubscribe below threshold
5. **What if a notification becomes stale?** — Expiration; don't send "X liked your post" if it's 3 days old

### Red Flags to Avoid

- Only optimizing for open rate (ignoring fatigue)
- No volume control
- Ignoring user preferences and quiet hours
- Not considering channel appropriateness

---

## 🔗 Related Topics

- [Recommendation Systems](./01-recommendation-systems.md) — Similar ranking and personalization
- [Ad Click Prediction](./07-ad-click-prediction.md) — Engagement prediction, calibration
- [Experiment Design](../../phase-5-advanced-topics/14-online-experimentation/01-experiment-design.md) — A/B testing notifications
- [Model Monitoring](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md) — Drift in open rate, fatigue
