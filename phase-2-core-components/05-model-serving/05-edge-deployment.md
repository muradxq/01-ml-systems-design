# Edge & Mobile Deployment

## Overview

Edge deployment runs ML models directly on end-user devices (phones, tablets, IoT devices) or near the source of data (edge servers, gateways) rather than in centralized cloud data centers. It has become essential for applications requiring low latency, privacy preservation, offline capability, and bandwidth efficiency. As mobile devices gain dedicated ML accelerators (NPUs, Neural Processing Units) and frameworks mature, on-device inference has moved from research to mainstream production.

**Key drivers:** Sub-10ms latency for real-time experiences, GDPR/privacy (data never leaves device), offline functionality in poor connectivity, reduced bandwidth costs for audio/video processing, and improved user experience through instant responses.

---

## 🎯 Why Edge Deployment Matters

```
┌─────────────────────────────────────────────────────────────────┐
│  Cloud vs Edge: Latency & Privacy                                 │
│                                                                  │
│  Cloud:   Device ──[50-200ms]──▶ Server ──[inference]──▶ Device  │
│           Round-trip: 100-500ms typical                          │
│                                                                  │
│  Edge:    Device ──[local inference 5-20ms]──▶ Response           │
│           Total: 5-20ms (10-50x faster)                          │
│                                                                  │
│  Privacy: Cloud sends raw data; Edge keeps data on device        │
└─────────────────────────────────────────────────────────────────┘
```

| Benefit | Cloud | Edge |
|---------|-------|------|
| **Latency** | 50-500ms RTT | 5-50ms local |
| **Privacy** | Data sent to server | Data stays on device |
| **Offline** | Requires connectivity | Works offline |
| **Bandwidth** | Sends audio/video | Processes locally |
| **Cost** | Server inference cost | Device compute (free to provider) |
| **Scalability** | Server capacity limit | Scales with devices |

---

## 📱 On-Device Constraints

### Resource Limits

| Resource | Typical Range | Impact |
|----------|---------------|--------|
| **Memory** | 256MB - 4GB (mobile) | Model size, batch size |
| **Compute** | Mobile CPU 2-8 cores | Inference speed |
| **GPU/NPU** | Limited cores, shared | Use delegates for acceleration |
| **Battery** | Must minimize mAh | Inference frequency, model size |
| **Storage** | 50-200MB for models | Quantization, pruning |
| **Thermal** | Throttling under load | Burst inference only |

### Constraint Trade-Offs

```
┌─────────────────────────────────────────────────────────────────┐
│  The Edge Triangle                                                │
│                                                                  │
│                    Accuracy                                      │
│                        *                                         │
│                       / \                                        │
│                      /   \                                       │
│                     /     \                                      │
│                    /       \                                     │
│                   /         \                                    │
│         Latency *-------------* Model Size                       │
│                                                                  │
│  Improve one corner → typically sacrifice another                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Model Formats

### Format Comparison

| Format | Platform | Strengths | Limitations |
|--------|----------|-----------|-------------|
| **TensorFlow Lite (TFLite)** | Android, iOS, embedded | Mature, wide adoption | TF ecosystem only |
| **Core ML** | Apple only | Native Apple optimization | iOS/macOS only |
| **ONNX Runtime Mobile** | Cross-platform | Portable, many backends | Less Apple optimization |
| **PyTorch Mobile / ExecuTorch** | Cross-platform | PyTorch native | Newer, evolving |
| **TensorRT** | NVIDIA edge GPUs | High perf on Jetson | NVIDIA hardware only |

### Conversion Pipelines

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Format Conversion Flow                                    │
│                                                                  │
│  PyTorch/TF ──▶ ONNX ──▶ TFLite / CoreML / ExecuTorch           │
│       │              │                                          │
│       │              └── Optional: Quantize, prune first         │
│       │                                                          │
│       └── Some frameworks: Direct export (e.g., TF→TFLite)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗜️ Model Compression Pipeline

### Pipeline Order

```
Full Model → Distill → Prune → Quantize → Profile on Device
```

**Rationale:** Distillation first (smaller architecture), then prune (remove weights), then quantize (reduce precision). Each step builds on the previous.

### 1. Quantization

| Type | Description | Size Reduction | Accuracy Impact |
|------|-------------|----------------|-----------------|
| **Post-Training (PTQ)** | Quantize after training | 4x (FP32→INT8) | Low-Medium |
| **Quantization-Aware (QAT)** | Simulate quant during training | 4x | Low |
| **INT8** | 8-bit weights/activations | 4x | Usually <1% |
| **INT4** | 4-bit (experimental) | 8x | Variable |

```python
# Post-Training Quantization (TFLite example)
import tensorflow as tf

def convert_to_tflite_quantized(model_path: str, representative_dataset) -> bytes:
    """Convert model to TFLite with PTQ."""
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # INT8 full quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    return tflite_model
```

### 2. Pruning

| Type | Description | Use Case |
|------|-------------|----------|
| **Unstructured** | Remove individual weights | Max compression, needs sparse kernels |
| **Structured** | Remove channels/filters | Simpler deployment, good speedup |
| **Magnitude-based** | Prune smallest weights | Simple, effective |
| **Iterative** | Prune → retrain → repeat | Better accuracy retention |

### 3. Knowledge Distillation

Train a small "student" model to mimic a large "teacher" model's outputs.

```
Teacher (large) ──soft labels──▶ Student (small)
         │                            │
         └── distillation loss ───────┘
```

### 4. Efficient Architectures

| Architecture | Params | Use Case |
|---------------|--------|----------|
| **MobileNetV2/V3** | 3-6M | Image classification |
| **EfficientNet-Lite** | 4-10M | Mobile vision |
| **BERT-tiny / DistilBERT** | 4-66M | Mobile NLP |
| **ConvNeXt-Tiny** | 28M | Balanced accuracy/size |

---

## 🔧 On-Device Inference

### Hardware Delegates

```
┌─────────────────────────────────────────────────────────────────┐
│  Inference Path with Delegates                                    │
│                                                                  │
│  TFLite Interpreter                                              │
│       │                                                          │
│       ├── GPU Delegate ──▶ Mobile GPU (faster for conv layers)   │
│       ├── NNAPI Delegate ──▶ Android Neural Networks API        │
│       ├── CoreML Delegate ──▶ Apple Neural Engine (iOS)          │
│       └── XNNPACK ──▶ Optimized CPU fallback                    │
└─────────────────────────────────────────────────────────────────┘
```

### Memory-Mapped Models

Load model from storage without full RAM copy—critical for large models on memory-constrained devices.

```python
# TFLite memory-mapped model loading
import tflite_runtime.interpreter as tflite

# Memory-map: model stays on disk, pages loaded on demand
interpreter = tflite.Interpreter(
    model_path="model.tflite",
    num_threads=4
)
interpreter.allocate_tensors()
```

### Streaming Inference

For audio/video: process chunks as they arrive rather than buffering entire clip.

```
Audio stream ──▶ [Chunk 1] ──▶ [Chunk 2] ──▶ [Chunk 3] ──▶ ...
                     │              │              │
                     ▼              ▼              ▼
                 Inference      Inference      Inference
                     │              │              │
                     └──────────────┴──────────────┘
                                    │
                                    ▼
                            Aggregated result
```

---

## 📡 OTA (Over-The-Air) Model Updates

### Update Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OTA Model Update Flow                                           │
│                                                                  │
│  Model Registry ──▶ CDN ──▶ Device (background download)        │
│        │                           │                             │
│        │                           ▼                             │
│        │                    Version check                        │
│        │                    Hash verification                    │
│        │                           │                             │
│        │                           ▼                             │
│        │                    Hot-swap on next launch               │
│        │                    (or scheduled restart)               │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation: OTA Update Manager

```python
from dataclasses import dataclass
from typing import Optional, Callable
import hashlib
import json
import os
import tempfile
import urllib.request

@dataclass
class ModelMetadata:
    version: str
    url: str
    sha256: str
    size_bytes: int
    min_app_version: str

class OTAUpdateManager:
    """
    Manages over-the-air model updates for edge deployment.
    Supports version check, delta updates (concept), rollback.
    """
    
    def __init__(
        self,
        model_dir: str,
        metadata_url: str,
        current_app_version: str,
        on_update_callback: Optional[Callable[[str], None]] = None
    ):
        self.model_dir = model_dir
        self.metadata_url = metadata_url
        self.current_app_version = current_app_version
        self.on_update = on_update_callback
        self.metadata_path = os.path.join(model_dir, "metadata.json")
    
    def get_installed_version(self) -> Optional[str]:
        """Get currently installed model version."""
        if not os.path.exists(self.metadata_path):
            return None
        with open(self.metadata_path) as f:
            meta = json.load(f)
        return meta.get("version")
    
    def fetch_metadata(self) -> Optional[ModelMetadata]:
        """Fetch latest model metadata from server."""
        try:
            with urllib.request.urlopen(self.metadata_url) as resp:
                data = json.loads(resp.read())
            
            return ModelMetadata(
                version=data["version"],
                url=data["url"],
                sha256=data["sha256"],
                size_bytes=data["size_bytes"],
                min_app_version=data.get("min_app_version", "0.0.0")
            )
        except Exception as e:
            print(f"Failed to fetch metadata: {e}")
            return None
    
    def needs_update(self) -> bool:
        """Check if a newer model is available."""
        meta = self.fetch_metadata()
        if not meta:
            return False
        
        if meta.min_app_version > self.current_app_version:
            return False  # App too old for this model
        
        current = self.get_installed_version()
        return current != meta.version
    
    def download_and_install(self, meta: ModelMetadata) -> bool:
        """Download model and install atomically."""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                urllib.request.urlretrieve(meta.url, tmp.name)
                
                with open(tmp.name, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                
                if digest != meta.sha256:
                    os.unlink(tmp.name)
                    return False
                
                # Atomic install: write to temp, then rename
                target = os.path.join(self.model_dir, "model.tflite")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.replace(tmp.name, target)
            
            # Save metadata
            with open(self.metadata_path, "w") as f:
                json.dump({
                    "version": meta.version,
                    "sha256": meta.sha256,
                    "size_bytes": meta.size_bytes
                }, f)
            
            if self.on_update:
                self.on_update(meta.version)
            
            return True
        except Exception as e:
            print(f"Install failed: {e}")
            return False
    
    def rollback(self, previous_version_path: str) -> bool:
        """Rollback to previous model version."""
        current = os.path.join(self.model_dir, "model.tflite")
        if os.path.exists(previous_version_path):
            os.replace(previous_version_path, current)
            return True
        return False
```

### Delta Updates

Download only changed weights (diffs) instead of full model—saves bandwidth.

```
Full model: 25MB
Delta (weights change ~5%): ~2MB
```

---

## 🤝 Federated Learning

Train on device, send only gradients/updates to server—never raw data.

```
┌─────────────────────────────────────────────────────────────────┐
│  Federated Learning Round                                        │
│                                                                  │
│  Server: Broadcast model M to devices                            │
│     │                                                            │
│     ├──▶ Device 1: Train on local data → gradients g1            │
│     ├──▶ Device 2: Train on local data → gradients g2           │
│     └──▶ Device N: Train on local data → gradients gN            │
│                                                                  │
│  Server: Aggregate gradients → Update M → Repeat                │
│                                                                  │
│  Privacy: Raw data never leaves devices                          │
└─────────────────────────────────────────────────────────────────┘
```

**Use cases:** Keyboard suggestions, health data, on-device personalization.

---

## 📊 Model Conversion & Edge Profiler

```python
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import time

@dataclass
class EdgeProfileResult:
    model_size_mb: float
    inference_latency_ms: float
    memory_peak_mb: float
    format: str

class EdgeModelPipeline:
    """
    Pipeline: convert model → optimize → profile on target.
    """
    
    def __init__(self, source_format: str = "pytorch"):
        self.source_format = source_format
    
    def convert_to_tflite(
        self,
        input_path: str,
        output_path: str,
        quantize: bool = True
    ) -> bool:
        """Convert model to TFLite format."""
        # Pseudocode - actual impl would use tf.lite.TFLiteConverter
        # or onnx -> tflite conversion
        try:
            # converter = ...
            # tflite_model = converter.convert()
            Path(output_path).write_bytes(b"")  # placeholder
            return True
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False
    
    def get_model_size_mb(self, path: str) -> float:
        """Get model file size in MB."""
        return os.path.getsize(path) / (1024 * 1024)
    
    def profile_inference(
        self,
        model_path: str,
        num_runs: int = 100,
        warmup: int = 10
    ) -> EdgeProfileResult:
        """
        Profile model on device (or simulator).
        In production: run benchmark binary on target device.
        """
        # Placeholder - real impl would invoke TFLite benchmark tool
        # or on-device profiling
        size_mb = self.get_model_size_mb(model_path)
        
        # Simulated latency (would be actual measurement)
        latency_ms = 8.5 + (size_mb * 0.5)  # Heuristic
        memory_mb = size_mb * 2 + 50  # Model + inference overhead
        
        return EdgeProfileResult(
            model_size_mb=size_mb,
            inference_latency_ms=latency_ms,
            memory_peak_mb=memory_mb,
            format="tflite"
        )
    
    def full_pipeline(
        self,
        source_model: str,
        output_path: str
    ) -> EdgeProfileResult:
        """Run full: convert → optimize → profile."""
        if not self.convert_to_tflite(source_model, output_path):
            raise RuntimeError("Conversion failed")
        return self.profile_inference(output_path)
```

---

## 📈 Metrics

| Metric | Target (Mobile) | Notes |
|--------|-----------------|-------|
| **Model size** | <20MB for apps | App store limits, storage |
| **Inference latency** | <50ms (real-time) | 16ms for 60fps video |
| **Memory** | <100MB peak | Avoid OOM, multi-tasking |
| **Battery impact** | <5% per session | User tolerance |
| **Accuracy retention** | >98% of original | After compression |

---

## 📊 Trade-Offs Table

| Dimension | Option A | Option B | Trade-Off |
|-----------|----------|----------|-----------|
| **Accuracy vs Size** | Full FP32 | INT8 quantized | ~1-2% accuracy for 4x size |
| **Latency vs Power** | GPU delegate | CPU only | GPU faster but more power |
| **Offline vs Freshness** | On-device | Cloud hybrid | Edge = offline, cloud = fresher |
| **Development vs Performance** | Cross-platform ONNX | Native CoreML/TFLite | Native = better perf |
| **Compression vs Iteration** | Heavy compress | Light compress | Heavy = smaller, more tuning |
| **Privacy vs Centralized training** | Federated | Centralized | Federated = private, slower |

---

## 💡 Interview Tips

### When to Discuss Edge

- **Question mentions:** mobile app, offline, latency, privacy, IoT, on-device
- **System design:** "Recommendation system for mobile app" → consider edge for latency
- **Scale:** "Millions of devices" → edge reduces server load

### Key Points to Hit

1. **Constraints first** — "We're limited by memory, battery, and thermal on device"
2. **Compression pipeline** — "We use distillation → pruning → quantization, in that order"
3. **Format choice** — "TFLite for Android, CoreML for iOS; ONNX if we need portability"
4. **OTA updates** — "We version models, verify hashes, hot-swap on next app launch"
5. **Metrics** — "We track model size, inference latency on real devices, and accuracy retention"

### Sample Answer

**Q: "How would you deploy an ML model to millions of mobile devices?"**

**A:** "I'd focus on:

1. **Model compression** — Distill to a smaller architecture, then prune and quantize to INT8. Target <15MB to stay under typical app budgets.

2. **Format** — TFLite for Android with GPU/NNAPI delegates; CoreML for iOS to use the Neural Engine. Use a shared training pipeline, then export to each format.

3. **OTA updates** — Host models on CDN with version metadata. Clients check for updates in the background, download with hash verification, and hot-swap on next session. Support rollback if new model underperforms.

4. **Monitoring** — Log inference latency and error rates from a sample of devices. A/B test new models before full rollout.

5. **Offline** — Design for no connectivity: bundle a default model in the app, allow OTA updates when online."

---

## 🔗 Related Topics

- [Model Deployment](./02-model-deployment.md) - General deployment strategies
- [Model Updates](./04-model-updates.md) - OTA and versioning
- [Batch vs Real-time](../../phase-3-operations-and-reliability/07-scalability-performance/03-batch-vs-realtime.md) - Inference patterns
- [Optimization Techniques](../../phase-3-operations-and-reliability/07-scalability-performance/04-optimization-techniques.md) - Model optimization
- [Data Privacy](../../phase-3-operations-and-reliability/09-security-privacy/01-data-privacy.md) - Privacy-preserving ML
