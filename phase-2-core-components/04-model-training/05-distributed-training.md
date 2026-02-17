# Distributed Training

## Overview

Distributed training scales model training across multiple devices and machines to handle large models and datasets. As model sizes grow (GPT-4: ~1.7T parameters, modern vision models: billions of parameters), distributed training has become essential. It reduces training time from weeks to hours and enables training models that don't fit on single devices.

---

## 🎯 When to Use Distributed Training

### Indicators You Need It
- Model doesn't fit in single GPU memory
- Training takes too long (days/weeks)
- Dataset too large to process efficiently
- Need to experiment faster

### Scale Reference

| Model Size | Single GPU | Distributed |
|------------|-----------|-------------|
| < 1B params | ✅ Feasible | Optional |
| 1-10B params | ⚠️ Difficult | Recommended |
| 10-100B params | ❌ Impossible | Required |
| > 100B params | ❌ Impossible | Required (large clusters) |

---

## 📊 Distributed Training Strategies

### 1. Data Parallelism

**How it works:** Same model on each device, different data batches

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Parallelism                                                │
│                                                                  │
│  Dataset: [Batch 1, Batch 2, Batch 3, Batch 4, ...]             │
│                │         │         │         │                   │
│                ▼         ▼         ▼         ▼                   │
│           ┌────────┐┌────────┐┌────────┐┌────────┐              │
│           │ GPU 0  ││ GPU 1  ││ GPU 2  ││ GPU 3  │              │
│           │ Model  ││ Model  ││ Model  ││ Model  │              │
│           │ Copy   ││ Copy   ││ Copy   ││ Copy   │              │
│           └───┬────┘└───┬────┘└───┬────┘└───┬────┘              │
│               │         │         │         │                    │
│               └─────────┴─────────┴─────────┘                   │
│                         │                                        │
│                         ▼                                        │
│               ┌──────────────────┐                              │
│               │  Gradient Sync   │                              │
│               │  (All-Reduce)    │                              │
│               └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

**When to use:**
- Model fits on single GPU
- Large datasets
- Want linear speedup

**Implementation (PyTorch DDP):**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup(rank, world_size):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # NCCL for GPU, gloo for CPU
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()

class DistributedTrainer:
    """Data parallel distributed training with PyTorch DDP."""
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        train_dataset,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Move model to GPU and wrap with DDP
        self.model = model.to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
        
        # Distributed sampler ensures each GPU gets different data
        self.sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # DataLoader with distributed sampler
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Ensures even batch sizes
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate * world_size  # Scale LR with world size
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        self.sampler.set_epoch(epoch)  # Ensure proper shuffling
        
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress (only on rank 0)
            if self.rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Synchronize and average loss across all processes
        avg_loss = total_loss / len(self.train_loader)
        loss_tensor = torch.tensor([avg_loss]).to(self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        return loss_tensor.item()
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save checkpoint (only on rank 0)."""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(checkpoint, path)

def train_distributed(rank, world_size, model_class, dataset, epochs):
    """Main training function for each process."""
    setup(rank, world_size)
    
    # Create model and trainer
    model = model_class()
    trainer = DistributedTrainer(model, rank, world_size, dataset)
    
    # Training loop
    for epoch in range(epochs):
        loss = trainer.train_epoch(epoch)
        
        if rank == 0:
            print(f"Epoch {epoch} completed. Average loss: {loss:.4f}")
            trainer.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')
    
    cleanup()

# Launch distributed training
if __name__ == '__main__':
    import torch.multiprocessing as mp
    
    world_size = torch.cuda.device_count()
    model_class = MyModel
    dataset = MyDataset()
    
    mp.spawn(
        train_distributed,
        args=(world_size, model_class, dataset, 100),
        nprocs=world_size,
        join=True
    )
```

---

### 2. Model Parallelism

**How it works:** Split model across devices, each device handles part of the model

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Parallelism (Pipeline)                                    │
│                                                                  │
│  Input ──▶ ┌─────────┐──▶┌─────────┐──▶┌─────────┐──▶ Output    │
│            │ Layers  │   │ Layers  │   │ Layers  │              │
│            │  1-4    │   │  5-8    │   │  9-12   │              │
│            │ (GPU 0) │   │ (GPU 1) │   │ (GPU 2) │              │
│            └─────────┘   └─────────┘   └─────────┘              │
│                                                                  │
│  Forward:  ──────────────────────────────────────▶              │
│  Backward: ◀──────────────────────────────────────              │
└─────────────────────────────────────────────────────────────────┘
```

**When to use:**
- Model too large for single GPU
- Deep models with sequential layers

```python
import torch
import torch.nn as nn

class ModelParallelNet(nn.Module):
    """Model split across multiple GPUs."""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # First part on GPU 0
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).to('cuda:0')
        
        # Second part on GPU 1
        self.seq2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).to('cuda:1')
        
        # Third part on GPU 2
        self.seq3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        ).to('cuda:2')
    
    def forward(self, x):
        # Transfer between GPUs
        x = self.seq1(x.to('cuda:0'))
        x = self.seq2(x.to('cuda:1'))
        x = self.seq3(x.to('cuda:2'))
        return x

# Training
model = ModelParallelNet()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target.to('cuda:2'))  # Target on same device as output
    loss.backward()
    optimizer.step()
```

---

### 3. Pipeline Parallelism

**How it works:** Combine model parallelism with micro-batching to reduce idle time

```
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Parallelism with Micro-batches                         │
│                                                                  │
│  Time ──▶                                                        │
│                                                                  │
│  GPU 0  │ F1 │ F2 │ F3 │ F4 │    │ B4 │ B3 │ B2 │ B1 │         │
│  GPU 1  │    │ F1 │ F2 │ F3 │ F4 │    │ B4 │ B3 │ B2 │ B1 │    │
│  GPU 2  │    │    │ F1 │ F2 │ F3 │ F4 │    │ B4 │ B3 │ B2 │ B1 ││
│                                                                  │
│  F = Forward pass, B = Backward pass                            │
│  Numbers = Micro-batch index                                    │
│                                                                  │
│  Bubble (idle time) reduced by overlapping computation          │
└─────────────────────────────────────────────────────────────────┘
```

```python
from torch.distributed.pipeline.sync import Pipe
import torch.nn as nn

# Create a model that can be split
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create pipeline
encoder = Encoder(1024, 512).to('cuda:0')
decoder = Decoder(512, 10).to('cuda:1')

model = nn.Sequential(encoder, decoder)

# Wrap in Pipe for pipeline parallelism
pipeline_model = Pipe(
    model,
    chunks=8,  # Number of micro-batches
    checkpoint='never'  # or 'always' for memory savings
)

# Training
optimizer = torch.optim.Adam(pipeline_model.parameters())

for data, target in train_loader:
    optimizer.zero_grad()
    
    # Pipe handles micro-batch splitting
    output = pipeline_model(data.to('cuda:0'))
    
    # RRef for distributed output
    loss = nn.functional.cross_entropy(output.local_value(), target.to('cuda:1'))
    loss.backward()
    optimizer.step()
```

---

### 4. Tensor Parallelism

**How it works:** Split individual tensors (weights) across devices

```
┌─────────────────────────────────────────────────────────────────┐
│  Tensor Parallelism - Split Weight Matrix                        │
│                                                                  │
│  Single GPU:                                                     │
│  ┌─────────────────────────────────┐                            │
│  │         Weight Matrix           │                            │
│  │          [4096 x 4096]          │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  4 GPUs (Column Parallel):                                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                   │
│  │ W[:, 0]│ │ W[:, 1]│ │ W[:, 2]│ │ W[:, 3]│                   │
│  │  1024  │ │  1024  │ │  1024  │ │  1024  │                   │
│  │ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │                   │
│  └────────┘ └────────┘ └────────┘ └────────┘                   │
│                                                                  │
│  Each GPU computes partial result, then reduce                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer."""
    
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        
        # Each GPU has a portion of the output dimension
        self.out_features_per_partition = out_features // world_size
        
        self.weight = nn.Parameter(
            torch.randn(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(self.out_features_per_partition)
        )
    
    def forward(self, x):
        # Local computation
        local_output = nn.functional.linear(x, self.weight, self.bias)
        
        # All-gather to combine results
        output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output)
        
        return torch.cat(output_list, dim=-1)

class RowParallelLinear(nn.Module):
    """Row-parallel linear layer."""
    
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        
        # Each GPU has a portion of the input dimension
        self.in_features_per_partition = in_features // world_size
        
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_partition)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        ) if rank == 0 else None
    
    def forward(self, x):
        # Split input
        x_partition = x[..., self.rank * self.in_features_per_partition:
                         (self.rank + 1) * self.in_features_per_partition]
        
        # Local computation
        local_output = nn.functional.linear(x_partition, self.weight)
        
        # All-reduce to sum results
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        
        # Add bias on rank 0
        if self.bias is not None:
            local_output = local_output + self.bias
        
        return local_output
```

---

### 5. Fully Sharded Data Parallelism (FSDP)

**How it works:** Shard model parameters, gradients, and optimizer states across devices

```
┌─────────────────────────────────────────────────────────────────┐
│  FSDP - Shard Everything                                         │
│                                                                  │
│  Traditional DDP:                                                │
│  GPU 0: Full Model + Full Gradients + Full Optimizer            │
│  GPU 1: Full Model + Full Gradients + Full Optimizer            │
│  (High memory per GPU)                                          │
│                                                                  │
│  FSDP:                                                          │
│  GPU 0: Shard 0 Params + Shard 0 Grads + Shard 0 Opt            │
│  GPU 1: Shard 1 Params + Shard 1 Grads + Shard 1 Opt            │
│  (Low memory per GPU, gather when needed)                       │
│                                                                  │
│  Forward: All-gather params → compute → discard non-local       │
│  Backward: All-gather params → compute grads → reduce-scatter   │
└─────────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)
import functools

def setup_fsdp(
    model: nn.Module,
    rank: int,
    world_size: int,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False
) -> FSDP:
    """Setup FSDP wrapped model."""
    
    # Mixed precision configuration
    if use_mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    else:
        mixed_precision_policy = None
    
    # CPU offload for very large models
    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Auto wrap policy - wrap layers above certain size
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1e6  # Wrap modules with >1M parameters
    )
    
    # Wrap model in FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
    )
    
    return fsdp_model

class FSDPTrainer:
    """Trainer using FSDP for large model training."""
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        train_dataset,
        batch_size: int = 8,
        learning_rate: float = 1e-4
    ):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        
        # Wrap model with FSDP
        self.model = setup_fsdp(model, rank, world_size)
        
        # Distributed sampler
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, epoch: int):
        """Train one epoch with FSDP."""
        self.model.train()
        self.sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
            
            # Scaled backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    def save_checkpoint(self, path: str):
        """Save FSDP checkpoint."""
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        
        # Get full state dict (gathered on rank 0)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model, 
            StateDictType.FULL_STATE_DICT, 
            save_policy
        ):
            state_dict = self.model.state_dict()
            
            if self.rank == 0:
                torch.save(state_dict, path)
```

---

## 🛠️ Distributed Training Frameworks

### Comparison

| Framework | Ease of Use | Scalability | Features |
|-----------|-------------|-------------|----------|
| **PyTorch DDP** | Medium | High | Data parallelism |
| **PyTorch FSDP** | Medium | Very High | Full sharding |
| **DeepSpeed** | Medium | Very High | ZeRO, offloading |
| **Horovod** | High | High | Multi-framework |
| **Ray Train** | High | High | Orchestration |
| **Megatron-LM** | Low | Very High | LLM training |

### DeepSpeed Integration

```python
import deepspeed
import torch
import torch.nn as nn

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,  # 0, 1, 2, or 3
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 10000
        }
    }
}

class DeepSpeedTrainer:
    """Trainer using DeepSpeed ZeRO optimization."""
    
    def __init__(self, model: nn.Module, train_dataset):
        # Initialize DeepSpeed
        self.model, self.optimizer, self.train_loader, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            training_data=train_dataset
        )
    
    def train_epoch(self, epoch: int):
        """Train one epoch with DeepSpeed."""
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.model.local_rank)
            target = target.to(self.model.local_rank)
            
            # Forward
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            
            # Backward (DeepSpeed handles gradient accumulation)
            self.model.backward(loss)
            
            # Step (DeepSpeed handles all-reduce)
            self.model.step()
            
            if self.model.local_rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    def save_checkpoint(self, path: str, tag: str):
        """Save DeepSpeed checkpoint."""
        self.model.save_checkpoint(path, tag)
    
    def load_checkpoint(self, path: str, tag: str):
        """Load DeepSpeed checkpoint."""
        self.model.load_checkpoint(path, tag)

# Launch: deepspeed --num_gpus=4 train.py
```

---

## 📊 Communication Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│  All-Reduce (Used in DDP)                                        │
│                                                                  │
│  Before:        After:                                          │
│  GPU 0: [1,2]   GPU 0: [10,20]  (sum of all)                   │
│  GPU 1: [3,4]   GPU 1: [10,20]                                 │
│  GPU 2: [6,14]  GPU 2: [10,20]                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  All-Gather (Used in FSDP forward)                               │
│                                                                  │
│  Before:        After:                                          │
│  GPU 0: [1,2]   GPU 0: [1,2,3,4,5,6]                           │
│  GPU 1: [3,4]   GPU 1: [1,2,3,4,5,6]                           │
│  GPU 2: [5,6]   GPU 2: [1,2,3,4,5,6]                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Reduce-Scatter (Used in FSDP backward)                          │
│                                                                  │
│  Before:              After:                                    │
│  GPU 0: [1,2,3,4,5,6] GPU 0: [6,12] (sum of positions 0-1)     │
│  GPU 1: [1,2,3,4,5,6] GPU 1: [18,24] (sum of positions 2-3)    │
│  GPU 2: [1,2,3,4,5,6] GPU 2: [30,36] (sum of positions 4-5)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Best Practices

### 1. Start Simple
- Begin with DDP
- Move to FSDP/DeepSpeed for larger models
- Profile before optimizing

### 2. Tune Batch Size
- Scale effective batch size with world size
- Adjust learning rate accordingly
- Use gradient accumulation if needed

### 3. Handle Checkpoints Properly
- Save from single rank (usually rank 0)
- Support checkpoint resumption
- Consider sharded saving for large models

### 4. Monitor Efficiently
- Log only from rank 0
- Use distributed metrics aggregation
- Monitor GPU utilization

### 5. Optimize Communication
- Use NCCL for GPU communication
- Consider gradient compression
- Overlap compute and communication

---

## ⚠️ Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Gradient explosion** | NaN loss | Gradient clipping, lower LR |
| **OOM errors** | CUDA out of memory | FSDP, gradient checkpointing |
| **Slow training** | Low GPU utilization | More workers, larger batches |
| **Hanging processes** | Training freezes | Check network, synchronization |
| **Inconsistent results** | Different metrics | Set seeds, check data splitting |

---

## 🔗 Related Topics

- [Training Infrastructure](./01-training-infrastructure.md) - Setup training clusters
- [Hyperparameter Tuning](./04-hyperparameter-tuning.md) - Tune distributed training
- [Scalability & Performance](../../phase-3-operations-and-reliability/07-scalability-performance/00-README.md) - Optimize performance
