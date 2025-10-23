# Performance Tuning Guide

## Overview

This guide covers performance optimization for the VR Body Segmentation application running on:
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: AMD Ryzen Threadripper 3990X (128 threads)
- **RAM**: 48GB
- **Target**: 90+ FPS for real-time VR processing

## Quick Start

### 1. Auto-Optimization

The fastest way to get optimal performance:

```bash
# Auto-detect and optimize configuration
python scripts/optimize.py --auto-optimize --save-config configs/my_optimized.yaml

# Use optimized configuration
python cli.py input.mp4 -c configs/my_optimized.yaml -o output.mp4
```

### 2. Use Preset Profiles

Choose a profile based on your needs:

```bash
# Fast profile - Maximum speed (120+ FPS target)
python cli.py input.mp4 --profile fast -o output.mp4

# Balanced profile - Good speed and quality (90 FPS target)
python cli.py input.mp4 --profile balanced -o output.mp4

# Quality profile - Best quality (60 FPS minimum)
python cli.py input.mp4 --profile quality -o output.mp4
```

## Performance Parameters

### GPU Configuration

#### 1. Precision Mode

**Impact**: 30-50% performance difference

```yaml
gpu:
  precision: fp16  # fp32, fp16, int8, mixed
```

**Recommendations**:
- `fp16`: Best for RTX 3090 (2x faster than FP32)
- `fp32`: Use only if precision is critical
- `int8`: Experimental, highest speed but quality loss
- `mixed`: Automatic mixed precision

**Benchmark**:
```bash
python benchmarks/benchmark_suite.py --precision-only
```

#### 2. VRAM Utilization

```yaml
gpu:
  max_vram_usage: 0.9  # Use 90% of VRAM
  memory_fraction: null  # Alternative: set exact fraction
```

**Recommendations**:
- Set `max_vram_usage` to 0.85-0.95 for stability
- Leave headroom for system processes
- Monitor with: `nvidia-smi`

#### 3. cuDNN Settings

```yaml
gpu:
  cudnn_benchmark: true  # Auto-tune convolution algorithms
  cudnn_deterministic: false  # Faster but non-deterministic
  allow_tf32: true  # Enable TF32 on Ampere GPUs
```

**Impact**: 10-20% performance gain

### Model Configuration

#### 1. Batch Size

**Impact**: 50-100% throughput difference

```yaml
model:
  batch_size: 8  # Process N frames in parallel
```

**Finding Optimal Batch Size**:

```bash
# Run batch size benchmark
python benchmarks/benchmark_suite.py --batch-only

# Auto-optimize
python scripts/optimize.py --auto-optimize
```

**Guidelines**:
- **1080p**: batch_size = 16-32
- **4K**: batch_size = 8-16
- **6K**: batch_size = 4-8
- **8K**: batch_size = 2-4

**Rule of Thumb**:
- Increase until GPU memory is 80-90% utilized
- Monitor with: `nvidia-smi -l 1`

#### 2. Input Resolution

```yaml
model:
  input_size: [1024, 1024]  # Model input resolution
```

**Impact**: Linear to quadratic performance scaling

**Recommendations**:
- Don't exceed native video resolution
- Common sizes: [512, 512], [1024, 1024], [2048, 2048]
- Lower resolution = higher FPS

#### 3. Data Loading

```yaml
model:
  num_workers: 4  # CPU workers for data loading
  prefetch_factor: 2  # Frames to prefetch per worker
```

**Optimal Workers**:
- **General**: num_workers = CPU_cores / 4
- **Threadripper 3990X**: 16-32 workers
- **Balance**: Don't starve GPU, don't overload CPU

**Prefetch Factor**:
- Increase if GPU is waiting for data
- Decrease if CPU/memory is bottlenecked

### Processing Pipeline

#### 1. CPU Threading

```yaml
processing:
  num_threads: 32  # CPU threads for preprocessing
  use_multiprocessing: true  # Use process pool
```

**Threadripper Optimization**:
- Use 25-50% of cores (32-64 threads)
- Enable `use_multiprocessing` for CPU-heavy tasks
- Monitor CPU with: `htop`

#### 2. Queue Management

```yaml
processing:
  max_queue_size: 100  # Frame buffer size
  chunk_size: 100  # Process in chunks
```

**Tuning**:
- Large queue = smoother processing, more memory
- Small queue = less memory, more variance
- **Rule**: queue_size ≈ FPS × 1-2 seconds

#### 3. Caching

```yaml
processing:
  enable_caching: true
  cache_size_gb: 4.0
```

**Impact**: 2-10x speedup for repeated operations

**Recommendations**:
- Enable for development/testing
- Set to 4-8GB for RTX 3090 system
- Disable for production if memory limited

### Performance Targets

```yaml
performance:
  target_fps: 90  # VR target
  max_latency_ms: 50.0  # Maximum per-frame latency
  auto_tune: true  # Automatic optimization
```

## Benchmarking

### Comprehensive Benchmark

Run all benchmarks to find optimal settings:

```bash
python benchmarks/benchmark_suite.py --output-dir ./results
```

This tests:
- Resolutions: 1080p, 4K, 6K, 8K
- Batch sizes: 1, 4, 8, 16, 32
- Precision modes: FP32, FP16

### Analyze Results

```bash
python benchmarks/performance_analyzer.py results/benchmark_results.json
```

Generates:
- Performance reports
- Visualization graphs
- Optimization recommendations

### Resolution-Specific Benchmarks

```bash
# Test only resolutions
python benchmarks/benchmark_suite.py --resolution-only

# Test only batch sizes
python benchmarks/benchmark_suite.py --batch-only

# Test only precision modes
python benchmarks/benchmark_suite.py --precision-only
```

## Bottleneck Detection

### Automatic Detection

```bash
python scripts/optimize.py --detect-bottlenecks
```

### Common Bottlenecks

#### 1. Low GPU Utilization (<50%)

**Symptoms**:
- GPU at 30-40% usage
- Low FPS despite available VRAM

**Solutions**:
- Increase batch size
- Reduce CPU preprocessing overhead
- Check CPU-GPU transfer bottleneck

```yaml
model:
  batch_size: 16  # Increase from 8
  num_workers: 8  # Increase workers
```

#### 2. High GPU Memory Usage (>95%)

**Symptoms**:
- OOM errors
- Crashes during processing

**Solutions**:
- Reduce batch size
- Lower input resolution
- Enable gradient checkpointing

```yaml
model:
  batch_size: 4  # Reduce from 8
gpu:
  max_vram_usage: 0.85  # More conservative
```

#### 3. High CPU Usage (>90%)

**Symptoms**:
- CPU at 100%
- Slow frame loading

**Solutions**:
- Reduce workers
- Optimize preprocessing
- Move more computation to GPU

```yaml
model:
  num_workers: 4  # Reduce from 8
processing:
  num_threads: 16  # Reduce from 32
```

#### 4. High Latency

**Symptoms**:
- Per-frame latency >50ms
- Stuttering playback

**Solutions**:
- Reduce batch size (lower latency)
- Use faster precision mode
- Optimize model

```yaml
gpu:
  precision: fp16  # Switch from fp32
model:
  batch_size: 4  # Reduce for lower latency
```

## Resolution-Specific Tuning

### 1080p (1920x1080)

**Recommended Settings**:
```yaml
model:
  batch_size: 16-32
  input_size: [1024, 1024]
gpu:
  precision: fp16
performance:
  target_fps: 120
```

**Expected Performance**: 150+ FPS

### 4K (3840x2160)

**Recommended Settings**:
```yaml
model:
  batch_size: 8-16
  input_size: [1024, 1024]
gpu:
  precision: fp16
performance:
  target_fps: 90
```

**Expected Performance**: 100+ FPS

### 6K (5760x3240)

**Recommended Settings**:
```yaml
model:
  batch_size: 4-8
  input_size: [1024, 1024]
gpu:
  precision: fp16
performance:
  target_fps: 60
```

**Expected Performance**: 60-80 FPS

### 8K (7680x4320)

**Recommended Settings**:
```yaml
model:
  batch_size: 2-4
  input_size: [1024, 1024]
gpu:
  precision: fp16
performance:
  target_fps: 30
```

**Expected Performance**: 30-50 FPS

**Note**: 8K at 90 FPS may require:
- Model optimization
- Lower input resolution
- Multi-GPU setup

## Real-Time Monitoring

### CLI Monitoring

```bash
# Interactive mode with live stats
python cli.py --interactive

# Commands in interactive mode:
stats      # Show current performance
hardware   # Show hardware info
config     # Show configuration
```

### GPU Monitoring

```bash
# Terminal 1: Run application
python cli.py input.mp4 -o output.mp4

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

### CPU Monitoring

```bash
htop
```

## Advanced Optimization

### 1. CUDA Streams

Use multiple CUDA streams for overlapped execution:

```python
# In model code
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    output1 = model(batch1)

with torch.cuda.stream(stream2):
    output2 = model(batch2)
```

### 2. Pin Memory

Enable pinned memory for faster CPU-GPU transfers:

```yaml
# Custom dataloader settings
dataloader:
  pin_memory: true
  persistent_workers: true
```

### 3. JIT Compilation

Use TorchScript for optimized execution:

```python
model = torch.jit.script(model)
```

### 4. Operator Fusion

Enable operator fusion for reduced kernel launches:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## Troubleshooting Performance Issues

### Issue: FPS Below Target

**Checklist**:
1. ✓ Check GPU utilization (should be >80%)
2. ✓ Verify precision mode (use FP16)
3. ✓ Increase batch size
4. ✓ Run benchmark to identify bottleneck

### Issue: Out of Memory

**Checklist**:
1. ✓ Reduce batch size
2. ✓ Lower input resolution
3. ✓ Clear cache: `torch.cuda.empty_cache()`
4. ✓ Check for memory leaks

### Issue: High Latency

**Checklist**:
1. ✓ Reduce batch size
2. ✓ Optimize preprocessing
3. ✓ Check queue sizes
4. ✓ Profile with: `torch.profiler`

## Performance Checklist

Before deploying to production:

- [ ] Run comprehensive benchmark
- [ ] Verify 90+ FPS on target resolution
- [ ] Check GPU utilization >80%
- [ ] Confirm latency <50ms
- [ ] Test with real VR content
- [ ] Monitor for memory leaks (24h test)
- [ ] Verify thermal throttling not occurring
- [ ] Document optimal configuration

## References

- [NVIDIA GPU Optimization Guide](https://docs.nvidia.com/deeplearning/performance/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
