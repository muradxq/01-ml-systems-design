# Computer Vision Systems

## Overview

Computer vision systems process images and videos to extract information, classify content, detect objects, and generate visual understanding. These systems power applications like autonomous vehicles, medical imaging, content moderation, visual search, and augmented reality. CV systems present unique challenges around GPU compute, large data sizes, and real-time processing requirements.

---

## 🎯 Problem Definition

### Common Use Cases

| Application | Task | Latency Requirement |
|-------------|------|---------------------|
| **Autonomous Vehicles** | Object detection, segmentation | < 50ms |
| **Medical Imaging** | Classification, segmentation | < 5s |
| **Content Moderation** | Image classification | < 500ms |
| **Visual Search** | Similarity search | < 200ms |
| **Quality Inspection** | Defect detection | < 100ms |

### Requirements (Object Detection Example)

| Requirement | Specification |
|-------------|---------------|
| **Latency** | < 100ms p99 |
| **Throughput** | 100-1000 images/second |
| **Accuracy** | > 90% mAP |
| **Availability** | 99.9% uptime |
| **Image Size** | Up to 4K resolution |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Computer Vision System Architecture                      │
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Image/Video   │                                                          │
│  │  Input         │                                                          │
│  └───────┬────────┘                                                          │
│          │                                                                    │
│          ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Preprocessing Layer                            │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │   Decode     │──│   Resize     │──│  Normalize   │           │        │
│  │  │   (JPEG/PNG) │  │   & Pad      │  │  & Transform │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Inference Layer (GPU)                          │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │  Model Server (TensorRT / ONNX / TorchServe)             │   │        │
│  │  │  - Dynamic batching                                       │   │        │
│  │  │  - Model optimization (FP16, INT8)                       │   │        │
│  │  │  - Multi-model serving                                   │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Post-processing Layer                          │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │   NMS        │──│   Threshold  │──│   Format     │─▶ Output  │        │
│  │  │              │  │              │  │   Response   │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Image Preprocessing

```python
import numpy as np
import cv2
from typing import Tuple, List
from dataclasses import dataclass
import albumentations as A
from PIL import Image
import io

@dataclass
class ImageConfig:
    """Image preprocessing configuration."""
    target_size: Tuple[int, int] = (640, 640)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    keep_aspect_ratio: bool = True
    pad_value: int = 114

class ImagePreprocessor:
    """Preprocess images for inference."""
    
    def __init__(self, config: ImageConfig):
        self.config = config
        self.transform = self._create_transform()
    
    def _create_transform(self) -> A.Compose:
        """Create albumentations transform pipeline."""
        transforms = []
        
        if self.config.keep_aspect_ratio:
            transforms.append(
                A.LongestMaxSize(max_size=max(self.config.target_size))
            )
            transforms.append(
                A.PadIfNeeded(
                    min_height=self.config.target_size[0],
                    min_width=self.config.target_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=self.config.pad_value
                )
            )
        else:
            transforms.append(
                A.Resize(
                    height=self.config.target_size[0],
                    width=self.config.target_size[1]
                )
            )
        
        transforms.append(A.Normalize(
            mean=self.config.normalize_mean,
            std=self.config.normalize_std
        ))
        
        return A.Compose(transforms)
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Preprocess single image."""
        
        original_shape = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        processed = transformed["image"]
        
        # Convert to CHW format
        processed = processed.transpose(2, 0, 1)
        
        # Track scaling for post-processing
        scale_x = self.config.target_size[1] / original_shape[1]
        scale_y = self.config.target_size[0] / original_shape[0]
        
        metadata = {
            "original_shape": original_shape,
            "scale": (scale_x, scale_y),
            "pad_offset": self._compute_pad_offset(original_shape)
        }
        
        return processed, metadata
    
    def preprocess_batch(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[dict]]:
        """Preprocess batch of images."""
        processed = []
        metadata = []
        
        for image in images:
            p, m = self.preprocess(image)
            processed.append(p)
            metadata.append(m)
        
        return np.stack(processed), metadata
    
    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode image from bytes."""
        # Try different formats
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image.convert("RGB"))
            return image
        except:
            # Fallback to OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _compute_pad_offset(self, original_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Compute padding offset for letterboxing."""
        if not self.config.keep_aspect_ratio:
            return (0, 0)
        
        scale = min(
            self.config.target_size[0] / original_shape[0],
            self.config.target_size[1] / original_shape[1]
        )
        
        new_h = int(original_shape[0] * scale)
        new_w = int(original_shape[1] * scale)
        
        pad_h = (self.config.target_size[0] - new_h) // 2
        pad_w = (self.config.target_size[1] - new_w) // 2
        
        return (pad_w, pad_h)
```

### 2. Model Inference with TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import List, Dict

class TensorRTInference:
    """TensorRT-optimized inference engine."""
    
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        self.stream = cuda.Stream()
    
    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs."""
        inputs = []
        outputs = []
        bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})
        
        return inputs, outputs, bindings
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on input data."""
        
        # Copy input to host buffer
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        
        # Transfer to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )
        
        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer outputs back
        results = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
            results.append(output["host"].copy())
        
        self.stream.synchronize()
        
        return results

class ONNXInference:
    """ONNX Runtime inference."""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        import onnxruntime as ort
        
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference."""
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )
        return outputs
```

### 3. Object Detection Post-Processing

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float

class DetectionPostProcessor:
    """Post-process object detection outputs."""
    
    def __init__(
        self,
        class_names: List[str],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45
    ):
        self.class_names = class_names
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
    
    def process(
        self,
        raw_output: np.ndarray,
        metadata: dict
    ) -> List[Detection]:
        """Process raw model output to detections."""
        
        # Parse output (format depends on model)
        # Assuming YOLO-style output: [batch, num_boxes, 5 + num_classes]
        boxes = raw_output[..., :4]
        objectness = raw_output[..., 4]
        class_probs = raw_output[..., 5:]
        
        # Compute class scores
        class_scores = objectness[..., np.newaxis] * class_probs
        class_ids = np.argmax(class_scores, axis=-1)
        confidences = np.max(class_scores, axis=-1)
        
        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        # Apply NMS
        keep_indices = self._nms(boxes, confidences)
        
        boxes = boxes[keep_indices]
        class_ids = class_ids[keep_indices]
        confidences = confidences[keep_indices]
        
        # Scale back to original image coordinates
        boxes = self._scale_boxes(boxes, metadata)
        
        # Create detection objects
        detections = []
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            detections.append(Detection(
                bbox=tuple(box),
                class_id=int(class_id),
                class_name=self.class_names[class_id],
                confidence=float(conf)
            ))
        
        return detections
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> List[int]:
        """Non-maximum suppression."""
        
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU below threshold
            mask = iou <= self.nms_threshold
            order = order[1:][mask]
        
        return keep
    
    def _scale_boxes(
        self,
        boxes: np.ndarray,
        metadata: dict
    ) -> np.ndarray:
        """Scale boxes back to original image coordinates."""
        
        scale_x, scale_y = metadata["scale"]
        pad_x, pad_y = metadata.get("pad_offset", (0, 0))
        
        boxes = boxes.copy()
        
        # Remove padding offset
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        
        # Scale back
        boxes[:, [0, 2]] /= scale_x
        boxes[:, [1, 3]] /= scale_y
        
        return boxes
```

### 4. Complete Vision Service

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import time

app = FastAPI()

class DetectionResult(BaseModel):
    bbox: List[float]
    class_name: str
    confidence: float

class InferenceResponse(BaseModel):
    detections: List[DetectionResult]
    latency_ms: float
    image_size: List[int]

class VisionService:
    """Complete computer vision inference service."""
    
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        inference_engine,  # TensorRT or ONNX
        postprocessor: DetectionPostProcessor,
        batch_size: int = 8,
        max_batch_wait_ms: int = 10
    ):
        self.preprocessor = preprocessor
        self.engine = inference_engine
        self.postprocessor = postprocessor
        self.batch_size = batch_size
        self.max_batch_wait = max_batch_wait_ms / 1000
        
        # Request queue for dynamic batching
        self.request_queue = asyncio.Queue()
        self.batch_processor_task = None
    
    async def start(self):
        """Start batch processor."""
        self.batch_processor_task = asyncio.create_task(
            self._batch_processor()
        )
    
    async def detect(self, image_bytes: bytes) -> InferenceResponse:
        """Detect objects in image."""
        
        start_time = time.time()
        
        # Decode and preprocess
        image = self.preprocessor.decode_image(image_bytes)
        processed, metadata = self.preprocessor.preprocess(image)
        
        # Submit to batch processor
        future = asyncio.get_event_loop().create_future()
        await self.request_queue.put((processed, metadata, future))
        
        # Wait for result
        detections = await future
        
        latency_ms = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            detections=[
                DetectionResult(
                    bbox=list(d.bbox),
                    class_name=d.class_name,
                    confidence=d.confidence
                )
                for d in detections
            ],
            latency_ms=latency_ms,
            image_size=list(metadata["original_shape"])
        )
    
    async def _batch_processor(self):
        """Process requests in batches."""
        
        while True:
            batch = []
            metadata_list = []
            futures = []
            
            # Collect batch
            try:
                while len(batch) < self.batch_size:
                    timeout = self.max_batch_wait if batch else None
                    item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=timeout
                    )
                    processed, metadata, future = item
                    batch.append(processed)
                    metadata_list.append(metadata)
                    futures.append(future)
            except asyncio.TimeoutError:
                pass
            
            if not batch:
                continue
            
            # Run inference
            try:
                batch_input = np.stack(batch)
                outputs = self.engine.infer(batch_input)
                
                # Post-process each result
                for i, (output, metadata, future) in enumerate(
                    zip(outputs[0], metadata_list, futures)
                ):
                    detections = self.postprocessor.process(output, metadata)
                    future.set_result(detections)
            
            except Exception as e:
                # Set exception on all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

# Service initialization
vision_service = VisionService(
    preprocessor=ImagePreprocessor(ImageConfig()),
    inference_engine=ONNXInference("model.onnx"),
    postprocessor=DetectionPostProcessor(CLASS_NAMES),
    batch_size=8
)

@app.on_event("startup")
async def startup():
    await vision_service.start()

@app.post("/detect", response_model=InferenceResponse)
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return await vision_service.detect(image_bytes)
```

### 5. Video Processing Pipeline

```python
import cv2
from typing import Generator, List
import time
from collections import deque

class VideoProcessor:
    """Process video streams for real-time inference."""
    
    def __init__(
        self,
        vision_service: VisionService,
        skip_frames: int = 1,
        tracker = None
    ):
        self.service = vision_service
        self.skip_frames = skip_frames
        self.tracker = tracker
        self.frame_count = 0
    
    def process_video(
        self,
        video_source: str,
        output_path: str = None
    ) -> Generator[List[Detection], None, None]:
        """Process video and yield detections per frame."""
        
        cap = cv2.VideoCapture(video_source)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Tracking state
        last_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames for efficiency
            if self.frame_count % (self.skip_frames + 1) == 0:
                # Run detection
                _, buffer = cv2.imencode('.jpg', frame)
                result = asyncio.run(self.service.detect(buffer.tobytes()))
                last_detections = result.detections
                
                # Update tracker
                if self.tracker:
                    self.tracker.update(last_detections)
            else:
                # Use tracker predictions
                if self.tracker:
                    last_detections = self.tracker.predict()
            
            # Draw detections
            if output_path:
                annotated = self._draw_detections(frame, last_detections)
                writer.write(annotated)
            
            yield last_detections
        
        cap.release()
        if output_path:
            writer.release()
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult]
    ) -> np.ndarray:
        """Draw bounding boxes on frame."""
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return frame
```

---

## 📈 Metrics & Evaluation

### Model Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **mAP** | Mean Average Precision | > 50% |
| **mAP@50** | mAP at IoU 0.5 | > 70% |
| **FPS** | Frames per second | > 30 |
| **Latency** | Inference time | < 50ms |

### System Metrics

```python
from prometheus_client import Histogram, Counter, Gauge

inference_latency = Histogram(
    'cv_inference_latency_ms',
    'Inference latency in milliseconds',
    buckets=[10, 20, 50, 100, 200, 500]
)

detections_count = Counter(
    'cv_detections_total',
    'Total detections',
    ['class_name']
)

gpu_utilization = Gauge(
    'cv_gpu_utilization',
    'GPU utilization percentage'
)

batch_size_histogram = Histogram(
    'cv_batch_size',
    'Dynamic batch sizes',
    buckets=[1, 2, 4, 8, 16, 32]
)
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Accuracy vs Speed** | Large model (accurate) | Small model (fast) |
| **Precision** | FP32 (accurate) | FP16/INT8 (fast) |
| **Batching** | Large batches (throughput) | Small batches (latency) |
| **Resolution** | High res (accurate) | Low res (fast) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you optimize inference latency?
2. How do you handle varying image sizes?
3. How would you scale to 10K images/second?
4. How do you handle video vs image processing?
5. How do you deploy models to edge devices?

**Key Points:**
- TensorRT/ONNX for optimization
- Dynamic batching for throughput
- Quantization (FP16, INT8) for speed
- GPU memory management
- Pre/post-processing optimization

---

## 🔗 Related Topics

- [Model Optimization](../../phase-3-operations-and-reliability/07-scalability-performance/04-optimization-techniques.md)
- [Horizontal Scaling](../../phase-3-operations-and-reliability/07-scalability-performance/01-horizontal-scaling.md)
- [Model Serving](../../phase-2-core-components/05-model-serving/01-serving-patterns.md)
