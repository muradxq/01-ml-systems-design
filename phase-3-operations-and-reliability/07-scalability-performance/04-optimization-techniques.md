# Optimization Techniques

## Overview

Model optimization reduces latency and resource usage while maintaining acceptable accuracy. Optimization is essential for production ML systems where every millisecond matters and infrastructure costs scale with usage. A well-optimized model can be 2-10x faster with minimal accuracy loss.

---

## 🎯 Optimization Methods

### 1. Model Quantization

Reduce numerical precision from FP32 to INT8 or lower.

| Precision | Size | Speed | Accuracy Loss |
|-----------|------|-------|---------------|
| **FP32** | 100% | 1x | Baseline |
| **FP16** | 50% | 1.5-2x | <0.1% |
| **INT8** | 25% | 2-4x | 0.5-1% |
| **INT4** | 12.5% | 3-6x | 1-3% |

```python
import torch
import torch.quantization as quant

# Dynamic Quantization (easiest)
def quantize_dynamic(model):
    """
    Quantize weights to INT8, activations computed dynamically.
    Best for: LSTM, Transformer, models with large weight matrices.
    """
    quantized = quant.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=torch.qint8
    )
    return quantized

# Static Quantization (better performance)
def quantize_static(model, calibration_data):
    """
    Quantize both weights and activations.
    Requires calibration data to determine activation ranges.
    """
    # Prepare model
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)
    
    # Calibrate with representative data
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    # Convert to quantized model
    quant.convert(model, inplace=True)
    return model

# Quantization-Aware Training (best accuracy)
def quantize_aware_training(model, train_loader, epochs=5):
    """
    Train with quantization simulation for best accuracy.
    """
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    
    # Fine-tune with quantization simulation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        for batch, labels in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
    # Convert to quantized model
    model.eval()
    quant.convert(model, inplace=True)
    return model
```

### 2. Model Pruning

Remove unnecessary weights/neurons to reduce model size and computation.

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Prune 30% of weights with smallest magnitude.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Prune 30% of weights
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
        elif isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    
    return model

def structured_pruning(model, amount=0.3):
    """
    Remove entire neurons/filters (more hardware-friendly).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module, 
                name='weight', 
                amount=amount, 
                n=2,  # L2 norm
                dim=0  # Prune output channels
            )
    return model
```

### 3. Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model.

```python
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, 
                 temperature=3.0, alpha=0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha  # Balance between hard and soft labels
        
        self.teacher.eval()  # Teacher in eval mode
        
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Combined loss: soft labels from teacher + hard labels.
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence for soft labels
        soft_loss = F.kl_div(
            soft_student, 
            soft_targets, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Cross entropy for hard labels
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    
    def train_step(self, batch, labels):
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_logits = self.teacher(batch)
        
        # Get student predictions
        student_logits = self.student(batch)
        
        # Calculate loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        return loss

# Example: Distill BERT to smaller model
teacher = BertModel.from_pretrained('bert-large')
student = DistilBertModel.from_pretrained('distilbert-base')

trainer = DistillationTrainer(teacher, student, temperature=4.0)
```

### 4. ONNX Optimization

Export to ONNX for optimized inference across platforms.

```python
import torch
import onnx
import onnxruntime as ort

def export_to_onnx(model, example_input, output_path):
    """Export PyTorch model to optimized ONNX."""
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        output_path,
        opset_version=14,
        do_constant_folding=True,  # Optimize constants
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Optimize ONNX graph
    from onnxruntime.transformers import optimizer
    optimized_model = optimizer.optimize_model(
        output_path,
        model_type='bert',  # or 'gpt2', etc.
        num_heads=12,
        hidden_size=768
    )
    optimized_model.save_model_to_file(output_path.replace('.onnx', '_optimized.onnx'))

def create_onnx_session(model_path):
    """Create optimized ONNX Runtime session."""
    # Session options for optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    
    # Use GPU if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    return session

# Usage
session = create_onnx_session('model_optimized.onnx')
result = session.run(None, {'input': input_data})
```

### 5. TensorRT Optimization (NVIDIA GPUs)

```python
import tensorrt as trt
import pycuda.driver as cuda

def optimize_with_tensorrt(onnx_path, output_path, precision='fp16'):
    """
    Optimize model with TensorRT for NVIDIA GPUs.
    Can achieve 2-5x speedup over PyTorch.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # Need calibration data for INT8
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

### 6. Batching Optimization

```python
import asyncio
from typing import List, Dict, Any
import time

class DynamicBatcher:
    """
    Batch requests dynamically for better GPU utilization.
    """
    
    def __init__(self, model, max_batch_size=32, 
                 max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds
        
        self.pending_requests: List[Dict] = []
        self.lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Add request to batch and wait for result."""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_requests.append({
                'features': features,
                'future': future
            })
            
            # Check if batch is full
            if len(self.pending_requests) >= self.max_batch_size:
                self.batch_event.set()
        
        # Wait for result
        return await future
    
    async def batch_processor(self):
        """Background task that processes batches."""
        while True:
            # Wait for batch to fill or timeout
            try:
                await asyncio.wait_for(
                    self.batch_event.wait(),
                    timeout=self.max_wait_ms
                )
            except asyncio.TimeoutError:
                pass
            
            self.batch_event.clear()
            
            # Get pending requests
            async with self.lock:
                if not self.pending_requests:
                    continue
                
                batch = self.pending_requests[:self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size:]
            
            # Process batch
            features_batch = [r['features'] for r in batch]
            predictions = self.model.predict_batch(features_batch)
            
            # Return results
            for request, prediction in zip(batch, predictions):
                request['future'].set_result(prediction)

# Usage
batcher = DynamicBatcher(model, max_batch_size=32, max_wait_ms=10)
asyncio.create_task(batcher.batch_processor())

# In request handler
prediction = await batcher.predict(features)
```

---

## 📊 Optimization Comparison

| Technique | Latency Reduction | Size Reduction | Accuracy Impact | Complexity |
|-----------|------------------|----------------|-----------------|------------|
| **Quantization (INT8)** | 2-4x | 4x | <1% | Low |
| **Pruning (30%)** | 1.5-2x | 1.3x | <1% | Medium |
| **Distillation** | 2-10x | Variable | 1-5% | High |
| **ONNX Runtime** | 1.5-2x | Same | None | Low |
| **TensorRT** | 2-5x | Same | <0.5% | Medium |
| **Batching** | N/A (throughput) | N/A | None | Low |

---

## 🔧 Optimization Pipeline

```
Original Model
      │
      ▼
┌─────────────┐
│  Profiling  │  ← Identify bottlenecks
└─────────────┘
      │
      ▼
┌─────────────┐
│ Distillation│  ← If model too large (optional)
└─────────────┘
      │
      ▼
┌─────────────┐
│  Pruning    │  ← Remove unnecessary weights
└─────────────┘
      │
      ▼
┌─────────────┐
│Quantization │  ← Reduce precision
└─────────────┘
      │
      ▼
┌─────────────┐
│ONNX/TensorRT│  ← Hardware optimization
└─────────────┘
      │
      ▼
┌─────────────┐
│  Batching   │  ← Runtime optimization
└─────────────┘
      │
      ▼
Optimized Model
```

---

## ✅ Best Practices

1. **Profile first** - measure before optimizing
2. **Optimize incrementally** - one technique at a time
3. **Measure accuracy** - track quality degradation
4. **Test thoroughly** - edge cases, distributions
5. **Benchmark realistically** - production-like conditions
6. **Consider hardware** - optimizations are hardware-specific
7. **Document changes** - track what was done

---

## ⚠️ Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Over-optimization** | Accuracy drops too much | Set acceptable threshold |
| **Wrong metric** | Optimizing CPU when GPU-bound | Profile first |
| **Ignoring batching** | Low GPU utilization | Dynamic batching |
| **Not testing** | Works in lab, fails in prod | Production testing |

---

## 🔗 Related Topics

- [Horizontal Scaling](./01-horizontal-scaling.md) - Scale after optimization
- [Caching Strategies](./02-caching-strategies.md) - Cache optimized predictions
- [Model Serving](../../phase-2-core-components/05-model-serving/00-README.md) - Serve optimized models
