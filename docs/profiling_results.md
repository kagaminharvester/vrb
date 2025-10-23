# VR Body Segmentation - GPU Profiling Methodology

## Overview

This document describes the GPU profiling methodology for the VR body segmentation application, including tools, metrics, analysis procedures, and optimization strategies.

## Profiling Tools

### 1. PyTorch Profiler

**Purpose:** Analyze PyTorch operations and CUDA kernels

**Usage:**
```bash
./scripts/profile_gpu.sh -t pytorch
```

**Metrics Collected:**
- CUDA kernel execution times
- CPU operation times
- Memory allocations
- Data transfer overhead
- Op-level breakdown

**Output Format:** JSON trace file for Chrome tracing

**Analysis:**
- Identify slowest operations
- Find memory bottlenecks
- Detect unnecessary synchronization
- Optimize data transfers

### 2. NVIDIA Nsight Systems

**Purpose:** System-level GPU profiling

**Usage:**
```bash
./scripts/profile_gpu.sh -t nsight
```

**Metrics Collected:**
- GPU timeline
- CPU/GPU synchronization
- CUDA API calls
- Memory transfers
- Kernel launches

**Visualization:** Timeline view showing GPU utilization

**Key Insights:**
- GPU idle time
- Kernel launch overhead
- Memory transfer patterns
- CPU/GPU dependencies

### 3. NVIDIA Nsight Compute

**Purpose:** Detailed kernel-level analysis

**Usage:**
```bash
ncu --set full python your_script.py
```

**Metrics Collected:**
- Kernel performance counters
- Memory bandwidth utilization
- Occupancy metrics
- Warp efficiency
- Cache hit rates

**Optimization Recommendations:**
- Occupancy improvements
- Memory coalescing
- Shared memory usage
- Register pressure reduction

### 4. Custom Memory Profiler

**Purpose:** Track memory usage over time

**Usage:**
```bash
./scripts/profile_gpu.sh -t pytorch
```

**Metrics Collected:**
- Allocated memory (MB)
- Reserved memory (MB)
- Peak memory usage
- Memory timeline

**Analysis:**
- Memory leak detection
- Peak usage identification
- Allocation patterns
- Cache behavior

## Key Performance Metrics

### Throughput Metrics

#### Frames Per Second (FPS)
```
FPS = Total Frames Processed / Total Time
```

**Target Values:**
- 1080p @ 60 FPS (batch=4)
- 4K @ 30 FPS (batch=2)
- VR Stereo @ 60 FPS (batch=4 per eye)

#### Throughput (Frames/Second)
```
Throughput = (Iterations × Batch Size) / Time
```

### Latency Metrics

#### Per-Frame Latency
```
Latency = Time per Iteration / Batch Size
```

**Target Values:**
- P50 < 16.67ms (60 FPS)
- P95 < 20ms
- P99 < 25ms

#### End-to-End Latency
```
E2E Latency = Decode + Preprocess + Inference + Postprocess + Encode
```

**Breakdown:**
- Decode: ~2ms
- Preprocess: ~1ms
- Inference: ~10ms
- Postprocess: ~1ms
- Encode: ~2ms

### Memory Metrics

#### GPU Memory Usage
```
Usage % = (Allocated Memory / Total Memory) × 100
```

**Targets:**
- Peak usage < 80% of available memory
- Efficient memory reuse
- No memory leaks

#### Memory Bandwidth Utilization
```
Bandwidth % = (Actual Bandwidth / Peak Bandwidth) × 100
```

### Compute Metrics

#### GPU Utilization
```
GPU Util % = (Active Time / Total Time) × 100
```

**Target:** > 85% during processing

#### Kernel Efficiency
```
Efficiency = (Theoretical Peak / Actual Performance) × 100
```

## Profiling Methodology

### 1. Baseline Profiling

**Objective:** Establish performance baseline

**Steps:**
1. Run standard benchmark:
   ```bash
   ./scripts/run_tests.sh -t performance
   ```

2. Profile with PyTorch profiler:
   ```bash
   ./scripts/profile_gpu.sh -t pytorch -d 30
   ```

3. Collect metrics:
   - Average FPS
   - P50/P95/P99 latency
   - Peak GPU memory
   - GPU utilization

4. Document results in this file

### 2. Bottleneck Identification

**Process:**

1. **Check GPU Utilization**
   ```bash
   nvidia-smi dmon -s u
   ```
   - If < 80%: Investigate CPU bottlenecks or memory transfers

2. **Analyze Kernel Times**
   - Review PyTorch profiler output
   - Identify slowest kernels
   - Check for inefficient operations

3. **Memory Transfer Analysis**
   - Measure CPU→GPU transfer time
   - Identify unnecessary transfers
   - Check for synchronization overhead

4. **CPU Profiling**
   - Profile preprocessing/postprocessing
   - Identify CPU-bound operations
   - Optimize data loading

### 3. Optimization Cycle

**Iterative Process:**

1. **Identify Bottleneck**
   - Profile current implementation
   - Find slowest component

2. **Implement Optimization**
   - Apply targeted optimization
   - Document changes

3. **Measure Impact**
   - Re-profile with same workload
   - Compare metrics

4. **Validate Results**
   - Ensure correctness maintained
   - Check for regressions

5. **Repeat**
   - Move to next bottleneck
   - Continue until targets met

## Optimization Strategies

### 1. Model Optimization

#### Mixed Precision (FP16)
```python
with torch.cuda.amp.autocast():
    output = model(input)
```

**Expected Speedup:** 1.5-2x

**Trade-offs:**
- Reduced numerical precision
- Potential accuracy loss
- Not all ops support FP16

#### TensorRT Optimization
```python
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 640, 640))],
    enabled_precisions={torch.float16}
)
```

**Expected Speedup:** 2-4x

**Benefits:**
- Kernel fusion
- Layer optimization
- Precision calibration

#### Torch Compile
```python
compiled_model = torch.compile(model, mode="max-autotune")
```

**Expected Speedup:** 1.2-1.5x (PyTorch 2.0+)

### 2. Batch Processing

#### Optimal Batch Size
```python
# Find optimal batch size
for batch_size in [1, 2, 4, 8, 16, 32]:
    measure_throughput(batch_size)
```

**Guidelines:**
- Larger batches → Higher throughput
- Limited by GPU memory
- Consider latency requirements

#### Dynamic Batching
```python
# Accumulate frames until batch full or timeout
batch = []
timeout = 100ms

while True:
    frame = get_frame()
    batch.append(frame)

    if len(batch) == batch_size or timeout_exceeded():
        process_batch(batch)
        batch = []
```

### 3. Memory Optimization

#### Memory Pooling
```python
# Pre-allocate buffers
input_pool = [torch.empty(1, 3, 640, 640, device='cuda')
              for _ in range(pool_size)]

# Reuse buffers
for i, frame in enumerate(frames):
    buffer = input_pool[i % pool_size]
    buffer.copy_(frame)
    output = model(buffer)
```

#### Gradient Checkpointing
```python
# Trade compute for memory
from torch.utils.checkpoint import checkpoint

output = checkpoint(model_segment, input)
```

**Memory Savings:** 30-50%

**Cost:** 20-30% slower inference

### 4. Pipeline Optimization

#### Overlapped Execution
```python
# Overlap preprocessing and inference
preprocess_stream = torch.cuda.Stream()
inference_stream = torch.cuda.Stream()

with torch.cuda.stream(preprocess_stream):
    preprocessed = preprocess(frame)

with torch.cuda.stream(inference_stream):
    output = model(preprocessed)
```

#### Pinned Memory
```python
# Use pinned memory for faster transfers
input_cpu = torch.rand(1, 3, 640, 640, pin_memory=True)
input_gpu = input_cpu.to('cuda', non_blocking=True)
```

## Profiling Results Template

### Configuration

**Hardware:**
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- CPU: AMD Ryzen Threadripper 3990X (128 threads)
- RAM: 48GB DDR4

**Software:**
- PyTorch: 2.1.0
- CUDA: 12.1
- cuDNN: 8.9

**Model:**
- Architecture: YOLOv8n-seg
- Input Size: 640×640
- Precision: FP32

### Baseline Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| FPS (1080p, batch=1) | X.XX | 60 | ⚠️/✅ |
| FPS (1080p, batch=4) | X.XX | 120 | ⚠️/✅ |
| FPS (4K, batch=1) | X.XX | 30 | ⚠️/✅ |
| Latency P50 (ms) | X.XX | <16.67 | ⚠️/✅ |
| Latency P95 (ms) | X.XX | <20 | ⚠️/✅ |
| Latency P99 (ms) | X.XX | <25 | ⚠️/✅ |
| Peak GPU Memory (MB) | X.XX | <19200 | ⚠️/✅ |
| GPU Utilization (%) | X.XX | >85 | ⚠️/✅ |

### Operation Breakdown

**Inference Pipeline (1 frame):**

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Decode | X.XX | XX% |
| Preprocessing | X.XX | XX% |
| Host→Device Transfer | X.XX | XX% |
| Model Inference | X.XX | XX% |
| Device→Host Transfer | X.XX | XX% |
| Postprocessing | X.XX | XX% |
| Encode | X.XX | XX% |
| **Total** | **X.XX** | **100%** |

### Top GPU Operations

**By Execution Time:**

| Operation | Time (ms) | % Total | Calls |
|-----------|-----------|---------|-------|
| Conv2d | X.XX | XX% | XX |
| BatchNorm | X.XX | XX% | XX |
| ReLU | X.XX | XX% | XX |
| MaxPool2d | X.XX | XX% | XX |
| Sigmoid | X.XX | XX% | XX |

**By Memory Usage:**

| Operation | Memory (MB) | % Total |
|-----------|-------------|---------|
| Conv2d weights | X.XX | XX% |
| Activation maps | X.XX | XX% |
| Input buffers | X.XX | XX% |
| Output buffers | X.XX | XX% |

### Optimization History

#### Optimization 1: Enable Mixed Precision

**Date:** YYYY-MM-DD

**Changes:**
- Enabled AMP autocast
- Used FP16 for inference

**Results:**
- FPS: X.XX → Y.YY (+Z.Z%)
- Latency: X.XX → Y.YY ms (-Z.Z%)
- Memory: X.XX → Y.YY MB (-Z.Z%)

**Trade-offs:**
- Minimal accuracy impact (<0.1% mAP)

#### Optimization 2: Batch Processing

**Date:** YYYY-MM-DD

**Changes:**
- Implemented dynamic batching
- Batch size: 1 → 4

**Results:**
- Throughput: X.XX → Y.YY FPS (+Z.Z%)
- Latency: X.XX → Y.YY ms (+Z.Z%)

**Trade-offs:**
- Increased latency due to batching
- Higher memory usage

## Continuous Monitoring

### Performance Regression Testing

**Process:**
1. Establish baseline before changes
2. Run profiling after changes
3. Compare metrics
4. Investigate regressions

**Automation:**
```bash
# Run nightly performance tests
./scripts/run_tests.sh -t performance > results_$(date +%Y%m%d).log

# Compare with baseline
python scripts/compare_results.py baseline.log results_latest.log
```

### Metrics Dashboard

**Tools:**
- TensorBoard for training metrics
- Grafana for production monitoring
- Custom scripts for profiling data

**Key Metrics to Track:**
- Average FPS over time
- P95/P99 latency trends
- Memory usage patterns
- GPU utilization

## Troubleshooting

### Low GPU Utilization

**Symptoms:**
- GPU util < 70%
- Low throughput

**Possible Causes:**
1. CPU bottleneck
   - Profile CPU operations
   - Optimize preprocessing
   - Use multiprocessing

2. Small batch size
   - Increase batch size
   - Use dynamic batching

3. Memory transfers
   - Use pinned memory
   - Minimize CPU↔GPU transfers
   - Overlap transfers with compute

### High Memory Usage

**Symptoms:**
- OOM errors
- Memory usage > 90%

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing
3. Enable mixed precision
4. Clear cache regularly
5. Optimize model architecture

### Poor Latency

**Symptoms:**
- High P95/P99 latency
- Variable processing time

**Investigation:**
1. Check for GC pauses
2. Identify synchronization points
3. Measure transfer times
4. Profile CPU preprocessing

**Solutions:**
- Pre-allocate buffers
- Use memory pools
- Optimize synchronization
- Pipeline operations

## Best Practices

### Profiling Best Practices

1. **Warmup:** Run several iterations before profiling
2. **Consistency:** Use same hardware/software for comparisons
3. **Isolation:** Profile in controlled environment
4. **Repetition:** Run multiple times, report average
5. **Documentation:** Record all configuration details

### Optimization Best Practices

1. **Measure First:** Profile before optimizing
2. **One Change at a Time:** Isolate impact of each optimization
3. **Validate Correctness:** Ensure accuracy maintained
4. **Document Changes:** Record all modifications
5. **Version Control:** Tag optimized versions

## References

### Tools Documentation

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [TensorRT](https://developer.nvidia.com/tensorrt)

### Performance Guides

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

---

**Last Updated:** October 2025
**Version:** 1.0
**Maintainer:** VR Body Segmentation Team
