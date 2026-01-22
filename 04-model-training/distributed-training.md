# Distributed Training

## Overview

Distributed training scales model training across multiple devices/machines to reduce training time.

---

## 🎯 Distributed Strategies

### 1. Data Parallelism
- Split data across devices
- Each device has full model
- Synchronize gradients
- Most common approach

### 2. Model Parallelism
- Split model across devices
- Each device processes part of model
- For very large models

### 3. Pipeline Parallelism
- Split model into stages
- Process batches in pipeline
- Overlap computation

---

## 🛠️ Frameworks

### PyTorch Distributed

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize
dist.init_process_group("nccl")

# Model
model = DistributedDataParallel(model)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### TensorFlow Distributed

```python
import tensorflow as tf

# Strategy
strategy = tf.distribute.MirroredStrategy()

# Model within strategy
with strategy.scope():
    model = create_model()
    model.compile(optimizer, loss)

# Train
model.fit(dataset, epochs=epochs)
```

---

## ✅ Best Practices

1. **Choose right strategy** - data parallelism for most cases
2. **Optimize communication** - reduce sync overhead
3. **Handle failures** - checkpoint and resume
4. **Monitor performance** - track scaling efficiency
5. **Use managed services** - SageMaker, Vertex AI

---

## 🔗 Related Topics

- [Training Infrastructure](./training-infrastructure.md)
- [Model Versioning](./model-versioning.md)
