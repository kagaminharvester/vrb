# GPU Optimization Guide for VR Body Segmentation

## Hardware Configuration

**Target System:**
- **GPU:** NVIDIA RTX 3090 (24GB VRAM, 10496 CUDA cores, Ampere architecture)
- **CPU:** AMD Ryzen Threadripper 3990X (64 cores, 128 threads)
- **RAM:** 48GB DDR4

This guide provides comprehensive documentation for maximizing performance on this hardware configuration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [CUDA Kernel Optimization](#cuda-kernel-optimization)
3. [TensorRT Integration](#tensorrt-integration)
4. [Memory Management](#memory-management)
5. [Batch Processing](#batch-processing)
6. [Async Pipeline](#async-pipeline)
7. [Performance Profiling](#performance-profiling)
8. [Optimization Strategies](#optimization-strategies)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT VIDEO STREAM                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: VIDEO DECODE (CPU - 8 threads)                        │
│  - Multi-threaded H.264/H.265 decoding                          │
│  - Hardware acceleration if available                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: PREPROCESS (GPU - 4 CUDA streams)                     │
│  - Custom CUDA kernels for fused operations                      │
│  - Resize + Color Conversion + Normalization                     │
│  - Async execution with pinned memory                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: INFERENCE (GPU - TensorRT FP16)                       │
│  - Dynamic batching (1-8 frames)                                │
│  - Tensor Core acceleration                                      │
│  - Concurrent execution with streams                             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: POSTPROCESS (CPU - 16 threads)                        │
│  - Argmax + smoothing                                           │
│  - Mask refinement                                               │
│  - Parallel processing                                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: ENCODE (CPU - 8 threads)                              │
│  - Video encoding / output formatting                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT STREAM                             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Performance Principles

1. **Overlap CPU and GPU work** - Use async execution and streams
2. **Minimize data transfers** - Keep data on GPU as long as possible
3. **Batch operations** - Group small operations for better throughput
4. **Fuse kernels** - Combine operations to reduce memory bandwidth
5. **Use Tensor Cores** - FP16 operations leverage specialized hardware

---

## CUDA Kernel Optimization

### Custom CUDA Kernels

Located in: `src/gpu/cuda_kernels.py`

#### 1. Fused Preprocessing Kernel

Combines three operations in a single kernel:
- Bilinear resize
- RGB normalization (0-255 → 0-1)
- Channel-wise standardization (mean/std)

**Benefits:**
- 3x reduction in memory bandwidth
- 2-3x speedup vs. separate operations
- Optimal for Ampere architecture

**Usage:**

```python
from src.gpu.cuda_kernels import CUDAKernelProcessor

processor = CUDAKernelProcessor(device_id=0)

# Process single frame
preprocessed = processor.fused_preprocess(
    image=frame,                    # [H, W, 3] uint8
    target_size=(512, 512),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    async_exec=True
)

# Synchronize when needed
processor.synchronize()
```

#### 2. Pinned Memory Pool

Fast CPU-GPU transfers using page-locked memory:

```python
from src.gpu.cuda_kernels import PinnedMemoryPool

pool = PinnedMemoryPool(max_size_gb=4.0)
pinned_array = pool.allocate(shape=(1080, 1920, 3), dtype=np.uint8)

# This array can be transferred to GPU 2-3x faster
```

#### 3. Block/Grid Configuration

Optimal settings for RTX 3090:
- **Block size:** 16x16 (256 threads) for 2D operations
- **Block size:** 256 or 512 threads for 1D operations
- **Grid size:** Calculated to cover entire data

**Example:**

```python
# For 1920x1080 image with 16x16 blocks
grid_x = (1920 + 15) // 16  # = 120 blocks
grid_y = (1080 + 15) // 16  # = 68 blocks
grid = (grid_x, grid_y, 1)
block = (16, 16, 1)
```

### Performance Tips

1. **Minimize divergence:** Avoid if/else in tight loops
2. **Coalesce memory:** Access memory in sequential patterns
3. **Use shared memory:** Cache frequently accessed data
4. **Occupancy:** Aim for 50-100% occupancy
5. **Register pressure:** Keep local variables minimal

---

## TensorRT Integration

### Model Conversion Pipeline

Located in: `src/gpu/tensorrt_engine.py`

#### Step 1: PyTorch → ONNX

```python
from src.gpu.tensorrt_engine import TensorRTEngine

# Define dynamic shapes
input_shapes = {
    'input': {
        'min': (1, 3, 512, 512),   # Minimum batch size
        'opt': (4, 3, 512, 512),   # Optimal batch size
        'max': (8, 3, 512, 512)    # Maximum batch size
    }
}

# Initialize engine builder
engine = TensorRTEngine(
    max_workspace_size=8,  # GB
    fp16_mode=True,
    int8_mode=False,
    device_id=0
)

# Convert model
engine.build_engine_from_pytorch(
    model=your_model,
    input_shapes=input_shapes,
    output_path='model_fp16.trt',
    opset_version=13
)
```

#### Step 2: Load and Run Inference

```python
# Load pre-built engine
engine.load_engine('model_fp16.trt')

# Warmup
engine.warmup(num_iterations=10)

# Run inference
output = engine.infer(input_data)  # Synchronous

# Or async
engine.infer_async(input_data)
result = engine.get_async_results()
```

### Precision Modes

#### FP16 (Recommended)

**Benefits:**
- 2x speedup vs. FP32
- 2x memory reduction
- Tensor Core acceleration
- Minimal accuracy loss (<1%)

**When to use:** Always, unless accuracy issues

#### INT8

**Benefits:**
- 4x speedup vs. FP32
- 4x memory reduction
- Best throughput

**Requirements:**
- Calibration dataset (100-1000 samples)
- May have accuracy loss (1-5%)

**When to use:** Maximum throughput needed, can tolerate slight accuracy loss

```python
# Enable INT8 with calibration
calibration_data = load_calibration_data()

engine = TensorRTEngine(
    fp16_mode=False,
    int8_mode=True
)

engine.build_engine_from_pytorch(
    model=model,
    input_shapes=input_shapes,
    output_path='model_int8.trt',
    calibration_data=calibration_data
)
```

### Dynamic Shape Optimization

TensorRT creates optimization profiles for different batch sizes:

```python
# Engine is optimized for these shapes
min_shape: [1, 3, 512, 512]   # Low latency
opt_shape: [4, 3, 512, 512]   # Balanced
max_shape: [8, 3, 512, 512]   # High throughput
```

At runtime, TensorRT interpolates between profiles for best performance.

### Performance Benchmarks

Expected performance on RTX 3090:

| Model | Precision | Batch Size | Latency (ms) | Throughput (FPS) |
|-------|-----------|------------|--------------|------------------|
| DeepLabV3+ | FP32 | 1 | 45 | 22 |
| DeepLabV3+ | FP16 | 1 | 22 | 45 |
| DeepLabV3+ | FP16 | 4 | 50 | 80 |
| DeepLabV3+ | FP16 | 8 | 85 | 94 |
| DeepLabV3+ | INT8 | 8 | 45 | 178 |

---

## Memory Management

### VRAM Monitoring

Located in: `src/gpu/memory_manager.py`

```python
from src.gpu.memory_manager import VRAMMonitor

monitor = VRAMMonitor(device_id=0, alert_threshold=0.9)

# Get current stats
stats = monitor.get_stats()
print(f"VRAM: {stats.allocated_vram_gb:.2f}/{stats.total_vram_gb:.2f} GB")

# Check for alerts
if monitor.check_alert():
    monitor.clear_cache()
```

### Memory Pooling

Pre-allocate memory for common tensor sizes:

```python
from src.gpu.memory_manager import MemoryPool

pool = MemoryPool(
    device_id=0,
    pool_size_gb=4.0,
    chunk_sizes=[
        1 * 3 * 512 * 512,    # Single frame
        4 * 3 * 512 * 512,    # Batch of 4
        8 * 3 * 512 * 512,    # Batch of 8
    ]
)

# Allocate from pool (fast)
tensor = pool.allocate(size=1*3*512*512, dtype=torch.float32)

# Use tensor...

# Return to pool (no deallocation)
pool.free(tensor)
```

### Model Quantization

#### FP32 → FP16

```python
from src.gpu.memory_manager import ModelQuantizer

model_fp16 = ModelQuantizer.quantize_fp16(model)

# Measure size reduction
size_fp32 = ModelQuantizer.measure_model_size(model)
size_fp16 = ModelQuantizer.measure_model_size(model_fp16)
print(f"Size reduction: {size_fp32:.2f} MB → {size_fp16:.2f} MB")
```

#### FP32 → INT8

```python
# Dynamic quantization (no calibration)
model_int8 = ModelQuantizer.quantize_int8_dynamic(model)

# Static quantization (with calibration)
model_int8 = ModelQuantizer.quantize_int8_static(
    model=model,
    calibration_loader=calib_loader,
    device=torch.device('cuda:0')
)
```

### Adaptive Batch Sizing

Automatically adjust batch size based on VRAM:

```python
from src.gpu.memory_manager import AdaptiveBatchSizer

batch_sizer = AdaptiveBatchSizer(
    initial_batch_size=4,
    min_batch_size=1,
    max_batch_size=16,
    target_utilization=0.85
)

# During training/inference
for epoch in range(num_epochs):
    batch_size = batch_sizer.adjust_batch_size()

    # Use batch_size...

    try:
        # Run inference
        output = model(batch)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            batch_size = batch_sizer.handle_oom()
```

---

## Batch Processing

### Dynamic Batching

Located in: `src/gpu/batch_processor.py`

Automatically accumulate requests into batches:

```python
from src.gpu.batch_processor import DynamicBatcher, BatchConfig

# Configure batcher
config = BatchConfig(
    min_batch_size=1,
    max_batch_size=8,
    timeout_ms=50.0,
    dynamic_batching=True,
    priority_mode=False
)

# Define inference function
def inference_fn(batch):
    return model(torch.from_numpy(batch).cuda())

# Create batcher
batcher = DynamicBatcher(config, inference_fn, device_id=0)
batcher.start()

# Submit requests
from src.gpu.batch_processor import InferenceRequest
import time

request = InferenceRequest(
    data=frame,
    request_id="frame_001",
    timestamp=time.perf_counter(),
    priority=0
)

response = batcher.submit(request)  # Blocking

print(f"Latency: {response.latency_ms:.2f} ms")
print(f"Batch size: {response.batch_size}")

batcher.stop()
```

### Stream-based Batching

Use multiple CUDA streams for parallel execution:

```python
from src.gpu.batch_processor import StreamBatcher

batcher = StreamBatcher(
    batch_size=4,
    num_streams=4,
    device_id=0
)

# Allocate buffers
batcher.allocate_buffers(
    input_shape=(3, 512, 512),
    output_shape=(2, 512, 512)
)

# Process with pipelining
results = batcher.process_pipeline(model, data_iterator)
```

### Finding Optimal Batch Size

```python
from src.gpu.batch_processor import BatchSizeOptimizer

optimizer = BatchSizeOptimizer(
    min_batch_size=1,
    max_batch_size=32,
    device_id=0
)

optimal_batch = optimizer.find_optimal_batch_size(
    model=model,
    input_shape=(3, 512, 512),
    num_warmup=5,
    num_test=10
)

print(f"Optimal batch size: {optimal_batch}")
```

---

## Async Pipeline

### Complete Pipeline Setup

Located in: `src/gpu/async_pipeline.py`

```python
from src.gpu.async_pipeline import AsyncPipeline

# Configuration
config = {
    'decode_threads': 8,          # CPU threads for video decode
    'preprocess_streams': 4,      # CUDA streams for preprocessing
    'batch_size': 4,              # Inference batch size
    'postprocess_threads': 16,    # CPU threads for post-processing
    'queue_size': 100,            # Max queue depth
    'target_size': (512, 512)     # Output resolution
}

# Create pipeline
pipeline = AsyncPipeline(
    model=segmentation_model,
    config=config,
    device_id=0
)

# Start pipeline
pipeline.start()

# Submit frames
for frame_id, frame_data in enumerate(video_frames):
    pipeline.submit_frame(frame_data, frame_id)

# Get results
result = pipeline.get_result(timeout=5.0)
if result:
    mask = result.data
    latency = result.metadata['total_latency']
    print(f"Frame {result.frame_id}: {latency*1000:.2f} ms")

# Monitor performance
stats = pipeline.get_stats()
print(f"Pipeline FPS: {stats.fps:.2f}")
print(f"Bottleneck: {stats.bottleneck_stage}")

# Stop pipeline
pipeline.stop()
```

### Pipeline Optimization

#### 1. Thread Allocation

Use CPU threads efficiently:

```python
# Decode: 8 threads (I/O bound)
# Preprocess: 4 CUDA streams (GPU bound)
# Inference: 1-2 CUDA streams (GPU bound)
# Postprocess: 16 threads (CPU bound, parallel work)
# Encode: 8 threads (I/O bound)
```

#### 2. Queue Sizing

Balance latency vs. buffering:

```python
# Small queues (10-20): Low latency, risk of stalls
# Medium queues (50-100): Balanced
# Large queues (200+): High throughput, higher latency
```

#### 3. Backpressure Handling

Prevent queue overflow:

```python
# Option 1: Block submission
pipeline.submit_frame(frame, frame_id)  # Blocks if queue full

# Option 2: Drop frames
if not pipeline.decode_queue.full():
    pipeline.submit_frame(frame, frame_id)
else:
    print("Queue full, dropping frame")
```

---

## Performance Profiling

### Basic Profiling

Located in: `src/gpu/profiler.py`

```python
from src.gpu.profiler import Profiler

profiler = Profiler(enabled=True)

# Profile code blocks
with profiler.profile('preprocessing'):
    preprocessed = preprocess(frame)

with profiler.profile('inference'):
    output = model(preprocessed)

with profiler.profile('postprocessing'):
    mask = postprocess(output)

# Print summary
profiler.print_summary()

# Export data
profiler.export_json('profile_results.json')

# Plot visualizations
profiler.plot_distribution('latency_distribution.png')
profiler.plot_timeline('timeline.png')
```

### CUDA Profiling

Detailed GPU kernel profiling:

```python
from src.gpu.profiler import CUDAProfiler

cuda_profiler = CUDAProfiler()

with cuda_profiler.profile():
    for _ in range(100):
        output = model(input_batch)
        torch.cuda.synchronize()

# Print summary
cuda_profiler.print_summary()

# Export Chrome trace (view in chrome://tracing)
cuda_profiler.export_chrome_trace('cuda_trace.json')
```

### Memory Profiling

Track memory allocations:

```python
from src.gpu.profiler import MemoryProfiler

mem_profiler = MemoryProfiler(device_id=0)

mem_profiler.snapshot('start')

# Run operations
model.load_state_dict(checkpoint)
mem_profiler.snapshot('after_load')

output = model(input_batch)
mem_profiler.snapshot('after_inference')

# Compare snapshots
mem_profiler.compare_snapshots('start', 'after_load')
mem_profiler.compare_snapshots('after_load', 'after_inference')
```

---

## Optimization Strategies

### Latency Optimization (VR Priority)

For VR applications, minimize end-to-end latency:

1. **Use small batch sizes** (1-2 frames)
2. **Disable dynamic batching** (no accumulation timeout)
3. **FP16 precision** (good balance)
4. **Minimize queue depths** (10-20)
5. **Pin threads to cores** (reduce context switching)

```python
config = {
    'batch_size': 1,
    'timeout_ms': 0,  # No waiting
    'queue_size': 10,
    'priority': 'latency'
}
```

**Expected latency:** 20-30ms end-to-end

### Throughput Optimization

For offline processing, maximize FPS:

1. **Use large batch sizes** (8-16 frames)
2. **Enable dynamic batching** (50-100ms timeout)
3. **INT8 quantization** (if accuracy acceptable)
4. **Large queue depths** (100-200)
5. **Multi-GPU if available**

```python
config = {
    'batch_size': 16,
    'timeout_ms': 100,
    'queue_size': 200,
    'priority': 'throughput'
}
```

**Expected throughput:** 150-200 FPS

### Balanced Configuration

Good starting point:

```yaml
# From gpu_config.yaml
batching:
  dynamic:
    min_batch_size: 1
    max_batch_size: 8
    timeout_ms: 50

pipeline:
  stages:
    decode:
      threads: 8
    preprocess:
      cuda_streams: 4
    inference:
      batch_size: 4
    postprocess:
      threads: 16
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** RuntimeError: CUDA out of memory

**Solutions:**

1. **Reduce batch size:**
   ```python
   config['batch_size'] = 2  # Try smaller batches
   ```

2. **Enable gradient checkpointing** (training):
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Clear cache:**
   ```python
   torch.cuda.empty_cache()
   ```

4. **Use adaptive batch sizing:**
   ```python
   batch_sizer = AdaptiveBatchSizer(max_batch_size=4)
   ```

### Low GPU Utilization

**Symptoms:** GPU usage < 80%

**Causes & Solutions:**

1. **CPU bottleneck:**
   - Increase decode/postprocess threads
   - Use faster data loading (pinned memory)

2. **Small batch size:**
   - Increase batch size
   - Enable dynamic batching

3. **Memory transfers:**
   - Keep data on GPU longer
   - Use CUDA streams
   - Pinned memory for transfers

4. **Insufficient parallelism:**
   - Use multiple CUDA streams
   - Overlap CPU and GPU work

### High Latency

**Symptoms:** End-to-end latency > 50ms

**Causes & Solutions:**

1. **Large batch accumulation:**
   - Reduce timeout_ms
   - Use smaller batch sizes

2. **Queue buildup:**
   - Reduce queue sizes
   - Check for bottleneck stage

3. **Synchronization:**
   - Use async execution
   - Minimize synchronize() calls

4. **Data transfers:**
   - Use pinned memory
   - Keep tensors on GPU

### Accuracy Issues

**Symptoms:** Model predictions incorrect after optimization

**Causes & Solutions:**

1. **INT8 quantization too aggressive:**
   - Use FP16 instead
   - Recalibrate with more data

2. **Numerical precision:**
   - Check for overflow/underflow
   - Use higher precision for critical ops

3. **Incorrect preprocessing:**
   - Verify normalization values
   - Check resize interpolation

---

## Performance Checklist

Before deploying:

- [ ] Model converted to TensorRT FP16
- [ ] Batch size optimized (run BatchSizeOptimizer)
- [ ] CUDA kernels enabled for preprocessing
- [ ] Memory pooling configured
- [ ] Async pipeline with appropriate thread counts
- [ ] Queue sizes tuned for use case
- [ ] Profiling enabled and bottlenecks identified
- [ ] VRAM monitoring active
- [ ] Warmup iterations run before benchmark
- [ ] Achieved target FPS (60+ for VR)
- [ ] Latency within target (<50ms for VR)

---

## Configuration Files

### Load Configuration

```python
import yaml

with open('configs/gpu_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access settings
batch_size = config['batching']['dynamic']['max_batch_size']
num_streams = config['cuda']['streams']['num_preprocess_streams']
```

### Environment Variables

```bash
# Set device
export CUDA_VISIBLE_DEVICES=0

# TensorRT cache
export TENSORRT_CACHE_DIR=./tensorrt_cache

# Logging
export LOG_LEVEL=INFO
```

---

## Additional Resources

### Documentation
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

### Tools
- NVIDIA Nsight Systems - System-wide profiling
- NVIDIA Nsight Compute - Kernel-level profiling
- nvtop - Real-time GPU monitoring

### Benchmarking

```bash
# Install dependencies
pip install torch torchvision tensorrt pycuda cupy

# Run benchmark
python -m src.gpu.profiler --benchmark --config configs/gpu_config.yaml

# Generate report
python -m src.gpu.profiler --report --output report.html
```

---

## Contact & Support

For issues or questions:
1. Check profiling output for bottlenecks
2. Review this guide's troubleshooting section
3. Enable debug logging in gpu_config.yaml
4. Collect performance metrics with profiler

---

**Last Updated:** 2025-10-23
**Hardware Target:** RTX 3090 + Threadripper 3990X
**Software:** PyTorch 2.0+, TensorRT 8.6+, CUDA 12.0+
