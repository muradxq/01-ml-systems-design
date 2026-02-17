# Scale Estimation for ML Systems

This guide provides a **step-by-step methodology** for back-of-envelope estimation during ML Systems Design interviews. Estimation questions ("How many servers? How much storage? What's the QPS?") test your ability to reason quantitatively about real-world scale. Master this process and use it in the **Estimate** phase of the CLEAR framework.

---

## Methodology: The 6-Step Process

Follow this order when asked to estimate scale. State assumptions clearly and show your work.

### Step 1: Start with User Base (DAU, MAU)

- **DAU** (Daily Active Users): Primary driver for throughput
- **MAU** (Monthly Active Users): For storage (feature store, embeddings)
- Typical ratio: DAU/MAU ≈ 0.2–0.5 for consumer apps
- If not given: Assume a scale tier (startup: 1M DAU; mid: 10M; big tech: 100M–1B)

### Step 2: Estimate Actions per User per Day

What triggers an ML inference?

| System Type | Action | Typical Rate |
|-------------|--------|--------------|
| Recommendation feed | Feed opens / refreshes | 5–15 per user/day |
| Ad prediction | Ad impressions seen | 20–50 per user/day |
| Search ranking | Searches | 2–5 per user/day |
| Fraud detection | Transactions | 1–10 per user/day |

### Step 3: Calculate QPS (Average and Peak)

$$
\text{Avg QPS} = \frac{\text{DAU} \times \text{Actions per user per day}}{86{,}400}
$$

- **86,400** = seconds per day (or round to 100,000 for mental math)
- **Peak QPS** = Avg QPS × 2–3 (typical); × 5–10 for events (launch, Black Friday)

### Step 4: Estimate Storage (Features, Models, Logs)

| Data Type | Formula | Notes |
|-----------|---------|-------|
| Feature store | entities × features × bytes_per_feature | 4 bytes for float32 |
| Embeddings | entities × dim × 4 | User/item embeddings |
| Event logs | events/day × bytes × retention_days | 100–500 bytes/event |
| Training data | events × features × bytes × retention | Historical for training |
| Models | params × 4 (FP32) or × 2 (FP16) | Per checkpoint |

### Step 5: Estimate Bandwidth

| Direction | Formula |
|-----------|---------|
| Inbound (events) | QPS × bytes_per_event |
| Outbound (responses) | QPS × response_size |
| Feature reads | QPS × features_per_request × bytes (before cache) |

### Step 6: Calculate GPU Requirements for Inference

$$
\text{GPUs needed} = \frac{\text{Peak QPS}}{\text{Inferences per GPU per second}}
$$

- Small model (LR, small NN): 500–2,000 inferences/GPU/s
- Medium (GBDT, DNN): 100–500
- Large (BERT, heavy ranker): 50–200
- **Caching**: With 80% hit rate, backend sees 20% of QPS → 5× fewer GPUs

---

## Worked Example 1: Recommendation System (Instagram-scale)

### Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| MAU | 2B | Monthly active users |
| DAU | 500M | ~25% of MAU |
| Sessions/user/day | 10 | Feed opens |
| Items per session | 30 | Recommendations per view |
| Peak factor | 3 | Peak ≈ 3× average |

### QPS Calculation

**Read QPS (recommendation requests):**

$$
\text{Daily actions} = 500\text{M} \times 10 \times 30 = 150\text{B} \text{ item views}
$$

$$
\text{Avg QPS} = \frac{150\text{B}}{86{,}400} \approx \frac{150\text{B}}{100\text{K}} \approx 1.7\text{M QPS}
$$

$$
\text{Peak QPS} \approx 1.7\text{M} \times 3 \approx 5\text{M QPS}
$$

### Feature Store Sizing

**Online store** (low-latency, hot features):

- 2B users × 200 features × 4 bytes = **1.6 TB**
- Must support 5M read QPS at p99 < 10 ms → Redis/DynamoDB with sharding

### Model Inference (GPU Estimation)

- Peak: 5M QPS
- Assume 1,000 inferences/GPU/s (small-to-medium ranker)
- **Without cache:** 5M / 1,000 = **5,000 GPUs**
- **With 80% cache hit:** 1M QPS hits model → **1,000 GPUs**

### Training Data (Annual)

- 500M users × 300 interactions/day × 365 days × 100 bytes ≈ 5.5 PB/year
- Stored in data lake; training samples from last 30–90 days

### Bandwidth

- Response: 5M QPS × 30 items × 1 KB ≈ 150 GB/s outbound (requires CDN/edge)
- Events: ~150B/day × 200 B ≈ 30 TB/day ingest

---

## Worked Example 2: Ad Click Prediction (Meta-scale)

### Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| DAU | 1B | Feed users |
| Ad impressions/user/day | 40 | ~4 ads per feed view × 10 sessions |
| Peak factor | 3 | |
| Latency budget | 50 ms | Auction constraint |
| Calibration required | Yes | For fair bidding |

### Bidding QPS

$$
\text{Avg QPS} = \frac{1\text{B} \times 40}{86{,}400} \approx \frac{40\text{B}}{100\text{K}} \approx 400\text{K QPS}
$$

$$
\text{Peak QPS} \approx 1.2\text{M QPS}
$$

### Feature Freshness

- **Real-time features:** Last click, last view (past 30 min) → streaming pipeline
- **Batch features:** Demographics, historical CTR → daily refresh
- Feature store: 1B users × 300 features × 4 bytes = **1.2 TB** online

### Calibration Overhead

- Post-processing (Platt scaling, isotonic regression) adds ~1–2 ms
- Calibrated predictions critical for second-price auction fairness

### Compute

- pCTR model: ~10 ms inference, 500 inferences/GPU/s
- 1.2M QPS / 500 ≈ 2,400 GPUs (before cache)
- With 70% cache (popular ads): ~700 GPUs

---

## Worked Example 3: Search Ranking (Google-scale)

### Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Queries/day | 8.5B | Global search |
| Peak factor | 2.5 | |
| Two-stage: retrieval + ranking | | |

### QPS

$$
\text{Avg QPS} = \frac{8.5\text{B}}{86{,}400} \approx 100\text{K QPS}
$$

$$
\text{Peak QPS} \approx 250\text{K QPS}
$$

### Two-Stage Estimation

**Stage 1 – Retrieval (ANN):**

- 250K QPS × 1,000 candidates per query = 250M ANN queries/s conceptually, but retrieval is batched
- ANN: 1–10 ms, ~1K queries/GPU/s → ~250 GPUs for retrieval
- Alternatively: inverted index + vector search; many queries hit cache

**Stage 2 – Ranking:**

- 250K QPS × 100 candidates to rank = 25M ranking inferences/s
- Heavy ranker: 50 ms, 20 inferences/GPU/s → 25M/20 = **1.25M GPUs** (naive)
- **Reality:** Batching (64–256), model optimization, early termination → ~5K–20K GPUs in practice
- Light ranker first reduces candidates: 250K × 1,000 → light ranker → 200 → heavy ranker on 200

**Refined ranking QPS:** 250K requests × 200 candidates = 50M inferences/s; at 500 inferences/GPU/s → **100K GPUs** → with batching and optimization: **5K–15K GPUs**

---

## Worked Example 4: Feature Store Sizing

### Requirements

| Store | Use Case | Latency | Freshness |
|-------|----------|---------|-----------|
| **Online** | Real-time serving | p99 < 10 ms | Near real-time for hot features |
| **Offline** | Training, batch jobs | Hours OK | Daily or hourly |

### Online Store

- **Entities:** 500M users + 100M items = 600M entity records
- **Features:** 200 per entity × 4 bytes = 800 bytes/entity
- **Storage:** 600M × 800 = **480 GB**
- **QPS:** 100K prediction QPS × 50 features/request = 5M feature reads/s
- **With 95% cache hit:** 250K backend reads/s → need Redis cluster or DynamoDB with high throughput

### Offline Store

- Same schema; stored in Parquet/ORC
- 600M × 200 × 4 = 480 GB (raw)
- With history (30 days): 600M × 200 × 4 × 30 ≈ **14 TB** (if storing daily snapshots)
- Usually incremental; 1–2 TB typical for point-in-time correct training

---

## Worked Example 5: Video Recommendation (YouTube-scale)

### Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| DAU | 500M | |
| Sessions/user/day | 8 | |
| Videos per home request | 20 | Initial load |
| Candidate generation | 5 sources × 500 = 2,500 | Before merge |
| Catalog | 1B videos | Embeddings needed |

### Candidate Generation QPS

- 500M × 8 = 4B requests/day → ~46K QPS avg; peak ~140K QPS
- ANN search: 140K × 5 sources = 700K ANN queries/s (or parallel per request)
- ANN: 5 ms, 200 queries/GPU/s → 3,500 GPUs for retrieval (with batching: ~1K)

### Ranking

- 140K QPS × 200 candidates (after light ranker) = 28M heavy ranker inferences/s
- 50 ms model, 20 inferences/GPU/s → 1.4M GPUs (naive)
- **With batching (256):** 140K batches/s × 256 = 36M inferences; at 500/GPU/s → 72K GPUs
- **With optimization:** ~5K–15K GPUs for heavy ranker

### Embedding Storage

- **User embeddings:** 500M × 256 × 4 = **512 GB**
- **Video embeddings:** 1B × 256 × 4 = **1 TB**
- Total: **~1.5 TB** for embedding tables

---

## Worked Example 6: Fraud Detection (Real-time)

### Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Transactions/day | 100M | Payment processor |
| Peak factor | 3 | |
| Latency budget | 100 ms | Must decide before settlement |
| Model | Light GBDT or small NN | 1–2 ms inference |

### Transaction Volume

$$
\text{Avg QPS} = \frac{100\text{M}}{86{,}400} \approx 1{,}200 \text{ QPS}
$$

$$
\text{Peak QPS} \approx 3{,}600 \text{ QPS}
$$

### Latency Budget Breakdown

| Component | Budget | Notes |
|-----------|--------|-------|
| Event validation | 5 ms | |
| Feature computation | 30 ms | User history, graph features |
| Feature store lookup | 20 ms | |
| Model inference | 10 ms | |
| Rule engine | 5 ms | |
| Decision + logging | 10 ms | |
| **Total** | **80 ms** | Within 100 ms |

### Feature Computation

- **Transaction features:** Amount, merchant, time, geolocation (real-time)
- **User features:** History, velocity (from feature store; 10–20 features)
- **Graph features:** Device cluster, IP cluster (precomputed, refreshed hourly)

### Compute

- 3,600 QPS × 2 ms = 7.2 concurrent seconds of compute → trivial for 1 server
- 1 GPU at 1,000 inferences/s: 3,600 / 1,000 = **4 GPUs** (with headroom)
- Often CPU sufficient for small models: **5–10 CPU instances**

---

## Quick Reference Formulas

| Estimate | Formula |
|----------|---------|
| **QPS** | DAU × actions_per_user / 86,400 |
| **Peak QPS** | Average QPS × 3 (typical) |
| **Backend QPS (cached)** | QPS × (1 − cache_hit_rate) |
| **Storage (events)** | events_per_day × bytes × retention_days |
| **Feature store** | entities × features × bytes_per_feature |
| **Embeddings** | entities × dim × 4 bytes |
| **GPUs needed** | Peak QPS / inferences_per_GPU |
| **GPUs (with cache)** | (Peak QPS × (1 − hit_rate)) / inferences_per_GPU |
| **Bandwidth (out)** | QPS × response_size |

### Key Numbers to Memorize

- Seconds per day: **86,400** (≈ 10⁵)
- 1 million = 10⁶; 1 billion = 10⁹
- Float32: 4 bytes; Float16: 2 bytes
- Peak factor: 2–3× typical; 5–10× for events

---

## Common Mistakes in Estimation

### 1. Forgetting Peak vs Average

- Always multiply by peak factor (2–3) for capacity planning
- "We need servers for peak, not average"

### 2. Not Accounting for Caching

- 80% cache hit → 5× reduction in backend load
- Mention: "With aggressive caching, we'd need 20% of the naive GPU count"

### 3. Confusing Model Size with Serving Memory

- Model size (e.g., 440 MB for BERT) ≠ GPU memory for inference
- Inference needs: model + activations + batch buffer (often 2–4× model size)

### 4. Ignoring Feature Store Latency

- Feature fetch can be 5–20 ms of a 50 ms budget
- At scale, feature store is often the bottleneck, not the model

### 5. Overlooking Replication and Headroom

- Production: 1.5–2× headroom for spikes
- Multi-region: 2–3× for redundancy

### 6. Wrong Entity Count for Feature Store

- User features: DAU or MAU? Use MAU for storage (need all users who might return)
- Item features: Full catalog size

### 7. Naive GPU Math

- Don't forget: batching increases throughput 5–10×
- Don't forget: model optimization (TensorRT, quantization) can 2–4× throughput
- Real systems use many small replicas, not one huge cluster

---

## Interview Tips

1. **State assumptions first:** "Let's assume 500M DAU and 10 sessions per user..."
2. **Round aggressively:** 86,400 → 100K; 150B/100K → 1.5M
3. **Show the formula:** Write `QPS = DAU × actions / 86400` before plugging in
4. **Sanity check:** 1B users, 5 actions/day → 5B/86K ≈ 60K QPS (ballpark)
5. **Mention caveats:** "This assumes no caching; with 80% hit rate we'd need 20% of the compute"
6. **Connect to design:** "At 5M QPS we need multi-region deployment and aggressive caching"

---

*Use this guide to practice estimation for each system you design. Time yourself: you should complete a full estimation (Steps 1–6) in 3–5 minutes.*
