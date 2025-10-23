# Optimization Techniques

## Table of Contents

1. [GPU Optimization](#gpu-optimization)
2. [CPU Optimization](#cpu-optimization)
3. [Memory Optimization](#memory-optimization)
4. [Pipeline Optimization](#pipeline-optimization)
5. [Algorithm Optimization](#algorithm-optimization)
6. [Profiling and Analysis](#profiling-and-analysis)

## GPU Optimization

### 1. Mixed Precision Training (FP16)

**Theory**: Use 16-bit floating point instead of 32-bit to reduce memory and increase speed.

**Implementation**:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Enable automatic mixed precision
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

# Scale gradients
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2x faster inference on Ampere GPUs (RTX 30xx)
- 50% memory reduction
- Minimal accuracy loss

**Configuration**:
```yaml
gpu:
  precision: fp16
  allow_tf32: true  # Additional speedup on Ampere
```

### 2. Kernel Fusion

**Theory**: Combine multiple operations into single GPU kernel to reduce memory bandwidth.

**Implementation**:

```python
# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Enable JIT fusion
torch.jit.set_fusion_strategy([('STATIC', 3)])
```

**Benefits**:
- 10-20% speedup
- Reduced memory bandwidth
- Fewer kernel launches

### 3. Batch Size Optimization

**Theory**: Larger batches improve GPU utilization but increase memory usage.

**Finding Optimal Batch Size**:

```python
def find_optimal_batch_size(model, input_shape, max_memory_percent=0.9):
    """Binary search for optimal batch size."""
    device = torch.device('cuda')
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = total_memory * max_memory_percent

    low, high = 1, 128
    optimal = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            # Test batch size
            torch.cuda.empty_cache()
            dummy_input = torch.randn(mid, *input_shape, device=device)
            _ = model(dummy_input)

            used_memory = torch.cuda.memory_allocated()
            if used_memory <= target_memory:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError:  # OOM
            high = mid - 1

    return optimal
```

**Auto-tune**:
```bash
python scripts/optimize.py --auto-optimize
```

### 4. CUDA Graphs

**Theory**: Capture and replay GPU operations to reduce CPU overhead.

**Implementation**:

```python
# Warmup
for _ in range(10):
    output = model(input)

# Capture graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(input)

# Replay
graph.replay()
```

**Benefits**:
- 10-30% speedup for small models
- Reduced CPU-GPU synchronization
- Lower latency

### 5. Asynchronous Data Transfer

**Theory**: Overlap CPU-GPU data transfer with computation.

**Implementation**:

```python
# Use pinned memory
dataloader = DataLoader(dataset, pin_memory=True)

# Non-blocking transfer
input = input.to(device, non_blocking=True)

# Overlap with computation
with torch.cuda.stream(stream):
    output = model(input)
```

### 6. TensorRT Optimization

**Theory**: NVIDIA TensorRT optimizes models for inference.

**Implementation**:

```python
import torch_tensorrt

# Compile model
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((batch_size, 3, height, width))],
    enabled_precisions={torch.float16},
)

# Inference
output = trt_model(input)
```

**Benefits**:
- 2-5x speedup
- Optimized for specific hardware
- Layer fusion and quantization

## CPU Optimization

### 1. Multi-Threading

**Theory**: Use multiple CPU threads for parallel preprocessing.

**Implementation**:

```python
from concurrent.futures import ThreadPoolExecutor
import threading

class PreprocessingPipeline:
    def __init__(self, num_threads=32):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    def preprocess_batch(self, frames):
        """Preprocess frames in parallel."""
        futures = [
            self.executor.submit(self.preprocess_frame, frame)
            for frame in frames
        ]
        return [f.result() for f in futures]
```

**Threadripper Optimization**:
```yaml
processing:
  num_threads: 64  # Use 50% of 128 threads
  use_multiprocessing: true
```

### 2. SIMD Vectorization

**Theory**: Use CPU SIMD instructions for parallel operations.

**Implementation**:

```python
import numpy as np

# Vectorized operations (use NumPy)
def normalize_batch(images):
    # Instead of loop
    return (images - mean) / std  # Vectorized
```

**Compilation Flags**:
```bash
# Enable AVX2/AVX512
export CFLAGS="-mavx2 -mfma"
pip install --no-binary :all: numpy
```

### 3. Memory-Mapped Files

**Theory**: Load large files without loading entire content into memory.

**Implementation**:

```python
import numpy as np

# Memory-mapped array
data = np.memmap('large_file.npy', dtype='float32', mode='r', shape=(n, h, w, c))

# Access without full load
batch = data[start:end]
```

### 4. Process Pool for CPU-Heavy Tasks

**Theory**: Use separate processes to avoid GIL limitations.

**Implementation**:

```python
from multiprocessing import Pool

def process_chunk(chunk):
    """Process chunk of frames."""
    return [preprocess(frame) for frame in chunk]

# Parallel processing
with Pool(processes=32) as pool:
    results = pool.map(process_chunk, chunks)
```

## Memory Optimization

### 1. Gradient Checkpointing

**Theory**: Trade computation for memory by recomputing intermediate activations.

**Implementation**:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # Checkpoint expensive operations
        x = checkpoint(self.expensive_layer, x)
        return x
```

**Benefits**:
- 50-80% memory reduction
- 20-30% slower (recomputation)
- Enables larger batch sizes

### 2. In-Place Operations

**Theory**: Modify tensors in-place to avoid allocations.

**Implementation**:

```python
# In-place operations (end with _)
x.relu_()  # Instead of: x = x.relu()
x.add_(y)  # Instead of: x = x + y
```

**Caution**: Don't use in-place ops if tensor is needed for gradients.

### 3. Memory Pooling

**Theory**: Reuse memory allocations to reduce fragmentation.

**Implementation**:

```python
class MemoryPool:
    def __init__(self):
        self.pool = {}

    def get_tensor(self, shape, dtype):
        key = (shape, dtype)
        if key not in self.pool:
            self.pool[key] = torch.empty(shape, dtype=dtype)
        return self.pool[key]

    def clear(self):
        self.pool.clear()
        torch.cuda.empty_cache()
```

### 4. Disk Caching

**Theory**: Cache preprocessed data to disk to avoid recomputation.

**Implementation**:

```python
from src.utils.cache_manager import CacheManager

cache = CacheManager(
    disk_cache_gb=10.0,
    enable_disk_cache=True
)

# Cache preprocessed frames
def preprocess_with_cache(frame, frame_id):
    cached = cache.get(f"frame_{frame_id}", use_disk=True)
    if cached is not None:
        return cached

    preprocessed = preprocess(frame)
    cache.put(f"frame_{frame_id}", preprocessed, to_disk=True)
    return preprocessed
```

### 5. Dynamic Memory Management

**Theory**: Adjust batch size dynamically based on available memory.

**Implementation**:

```python
def adaptive_batch_process(frames):
    """Process with adaptive batch size."""
    batch_size = initial_batch_size

    while frames:
        try:
            batch = frames[:batch_size]
            result = model(batch)
            frames = frames[batch_size:]

            # Success - try increasing batch size
            batch_size = min(batch_size * 2, max_batch_size)

        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM - reduce batch size
                torch.cuda.empty_cache()
                batch_size = max(batch_size // 2, 1)
            else:
                raise
```

## Pipeline Optimization

### 1. Prefetching

**Theory**: Load next batch while processing current batch.

**Implementation**:

```python
class PrefetchDataLoader:
    def __init__(self, dataloader, device):
        self.loader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        # Prefetch first batch
        first_batch = next(iter(self.loader))
        with torch.cuda.stream(self.stream):
            next_batch = first_batch.to(self.device, non_blocking=True)

        for batch in self.loader:
            # Wait for prefetch
            torch.cuda.current_stream().wait_stream(self.stream)
            current_batch = next_batch

            # Prefetch next
            with torch.cuda.stream(self.stream):
                next_batch = batch.to(self.device, non_blocking=True)

            yield current_batch
```

**Configuration**:
```yaml
model:
  prefetch_factor: 4  # Prefetch 4 batches ahead
```

### 2. Pipeline Parallelism

**Theory**: Overlap different pipeline stages (preprocess, inference, postprocess).

**Implementation**:

```python
from queue import Queue
from threading import Thread

class PipelineProcessor:
    def __init__(self):
        self.preprocess_queue = Queue(maxsize=100)
        self.inference_queue = Queue(maxsize=100)
        self.postprocess_queue = Queue(maxsize=100)

    def start(self):
        # Start pipeline threads
        Thread(target=self._preprocess_worker).start()
        Thread(target=self._inference_worker).start()
        Thread(target=self._postprocess_worker).start()
```

### 3. Batch Processing

**Theory**: Process multiple frames together for better GPU utilization.

**Implementation**:

```python
def batch_processor(frames, batch_size=8):
    """Process frames in batches."""
    results = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]

        # Pad if necessary
        if len(batch) < batch_size:
            batch = batch + [batch[-1]] * (batch_size - len(batch))

        # Process batch
        batch_results = model(torch.stack(batch))
        results.extend(batch_results[:len(frames[i:i+batch_size])])

    return results
```

### 4. Queue Management

**Theory**: Optimal queue sizes balance memory usage and throughput.

**Implementation**:

```yaml
processing:
  max_queue_size: 100  # Balance memory and smoothness

  # Rule of thumb:
  # queue_size = target_fps * latency_tolerance_seconds
  # Example: 90 FPS * 1s = 90 frames
```

## Algorithm Optimization

### 1. Early Exit

**Theory**: Skip unnecessary computation when confidence is high.

**Implementation**:

```python
def early_exit_inference(model, input, confidence_threshold=0.95):
    """Exit early if confidence is high."""
    # First stage (fast)
    fast_output = model.fast_branch(input)
    confidence = fast_output.max()

    if confidence > confidence_threshold:
        return fast_output

    # Full inference (slow but accurate)
    return model.full_branch(input)
```

### 2. Spatial Pyramid Processing

**Theory**: Process different resolutions in parallel.

**Implementation**:

```python
def multi_scale_inference(model, image):
    """Process multiple scales for better accuracy."""
    scales = [0.5, 1.0, 2.0]
    results = []

    for scale in scales:
        scaled = F.interpolate(image, scale_factor=scale)
        result = model(scaled)
        results.append(F.interpolate(result, size=image.shape[-2:]))

    return torch.mean(torch.stack(results), dim=0)
```

### 3. Temporal Coherence

**Theory**: Exploit temporal coherence in video for faster processing.

**Implementation**:

```python
class TemporalOptimizer:
    def __init__(self, skip_threshold=0.01):
        self.prev_frame = None
        self.prev_result = None
        self.skip_threshold = skip_threshold

    def process(self, frame):
        if self.prev_frame is not None:
            # Check if frame changed significantly
            diff = torch.abs(frame - self.prev_frame).mean()

            if diff < self.skip_threshold:
                # Frame barely changed, reuse previous result
                return self.prev_result

        # Process new frame
        result = model(frame)

        self.prev_frame = frame
        self.prev_result = result
        return result
```

### 4. Region of Interest (ROI) Processing

**Theory**: Only process relevant regions of the image.

**Implementation**:

```python
def roi_inference(model, image, roi_detector):
    """Process only regions of interest."""
    # Detect ROIs (fast)
    rois = roi_detector(image)

    results = torch.zeros_like(image)

    for roi in rois:
        x1, y1, x2, y2 = roi
        region = image[:, :, y1:y2, x1:x2]

        # Process ROI only
        result = model(region)
        results[:, :, y1:y2, x1:x2] = result

    return results
```

## Profiling and Analysis

### 1. PyTorch Profiler

**Implementation**:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(100):
        output = model(input)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
```

### 2. NVIDIA Nsight Systems

**Usage**:

```bash
# Profile application
nsys profile -o profile.qdrep python cli.py input.mp4 -o output.mp4

# View with Nsight Systems GUI
nsight-sys profile.qdrep
```

### 3. Custom Profiling

**Implementation**:

```python
from src.utils.logger import get_logger

logger = get_logger()

# Profile operation
with logger.timer("model_inference"):
    output = model(input)

# Get statistics
stats = logger.get_stats()
print(f"Average inference time: {stats['timers']['model_inference']['mean']*1000:.2f}ms")
```

### 4. Bottleneck Detection

**Usage**:

```bash
# Detect bottlenecks automatically
python scripts/optimize.py --detect-bottlenecks
```

## Best Practices

### Do's
- ✓ Always benchmark before and after optimization
- ✓ Profile to find actual bottlenecks
- ✓ Use FP16 on modern GPUs
- ✓ Enable cuDNN auto-tuner
- ✓ Maximize batch size within memory limits
- ✓ Use asynchronous data transfer
- ✓ Cache expensive operations
- ✓ Monitor GPU utilization (target >80%)

### Don'ts
- ✗ Don't optimize without profiling first
- ✗ Don't use in-place ops where gradients needed
- ✗ Don't over-optimize CPU at expense of GPU
- ✗ Don't ignore memory fragmentation
- ✗ Don't use synchronous operations unnecessarily
- ✗ Don't process frames individually (use batches)

## Performance Targets

For RTX 3090 + Threadripper 3990X:

| Resolution | Batch Size | Precision | Expected FPS |
|------------|------------|-----------|--------------|
| 1080p      | 16-32      | FP16      | 150+ FPS     |
| 4K         | 8-16       | FP16      | 100+ FPS     |
| 6K         | 4-8        | FP16      | 60-80 FPS    |
| 8K         | 2-4        | FP16      | 30-50 FPS    |

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
