# Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command Line Interface](#command-line-interface)
4. [Configuration](#configuration)
5. [Benchmarking](#benchmarking)
6. [Optimization](#optimization)
7. [Advanced Usage](#advanced-usage)
8. [Examples](#examples)

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **NVIDIA Driver**: 520.xx or higher
- **FFmpeg**: For video processing

### Install Dependencies

```bash
# Clone repository
cd /home/pi/vr-body-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pyyaml rich matplotlib seaborn psutil

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Verify Installation

```bash
# Check hardware detection
python cli.py --show-hardware

# Should display:
# - GPU: NVIDIA RTX 3090
# - GPU Memory: 24.0 GB
# - CPU: 128 cores
```

## Quick Start

### 1. Process a Video (Default Settings)

```bash
python cli.py input_video.mp4 -o output_video.mp4
```

### 2. Use a Profile

```bash
# Fast profile (120+ FPS)
python cli.py input.mp4 --profile fast -o output.mp4

# Balanced profile (90 FPS)
python cli.py input.mp4 --profile balanced -o output.mp4

# Quality profile (60+ FPS)
python cli.py input.mp4 --profile quality -o output.mp4
```

### 3. Auto-Optimize and Process

```bash
# Optimize configuration for your hardware
python scripts/optimize.py --auto-optimize --save-config optimized.yaml

# Use optimized configuration
python cli.py input.mp4 -c optimized.yaml -o output.mp4
```

## Command Line Interface

### Main CLI (`cli.py`)

```bash
python cli.py [OPTIONS] INPUT_VIDEO
```

**Options**:

| Option | Description | Example |
|--------|-------------|---------|
| `-o, --output` | Output video path | `-o output.mp4` |
| `-c, --config` | Configuration file | `-c configs/my_config.yaml` |
| `-p, --profile` | Use preset profile | `--profile fast` |
| `--batch-size` | Override batch size | `--batch-size 16` |
| `--precision` | Override precision | `--precision fp16` |
| `--workers` | Override num workers | `--workers 8` |
| `--interactive` | Interactive mode | `--interactive` |
| `--benchmark` | Run benchmark | `--benchmark` |
| `--optimize` | Auto-optimize | `--optimize` |
| `--show-hardware` | Show hardware info | `--show-hardware` |
| `--show-config` | Show configuration | `--show-config` |
| `--log-level` | Logging level | `--log-level DEBUG` |
| `--log-dir` | Log directory | `--log-dir ./logs` |

### Interactive Mode

```bash
python cli.py --interactive
```

**Available Commands**:
- `help` - Show available commands
- `hardware` - Display hardware information
- `config` - Show current configuration
- `stats` - Show performance statistics
- `profile <name>` - Load a profile (fast/balanced/quality)
- `clear` - Clear screen
- `exit` - Exit interactive mode

### Benchmarking Suite

```bash
# Run all benchmarks
python benchmarks/benchmark_suite.py

# Run specific benchmarks
python benchmarks/benchmark_suite.py --resolution-only
python benchmarks/benchmark_suite.py --batch-only
python benchmarks/benchmark_suite.py --precision-only

# Specify output directory
python benchmarks/benchmark_suite.py --output-dir ./my_results
```

### Performance Analysis

```bash
# Analyze benchmark results
python benchmarks/performance_analyzer.py results/benchmark_results.json

# Generate report only (no plots)
python benchmarks/performance_analyzer.py results/benchmark_results.json --no-plots

# Custom output directory
python benchmarks/performance_analyzer.py results/benchmark_results.json --output-dir ./analysis
```

### Optimization Tool

```bash
# Detect bottlenecks
python scripts/optimize.py --detect-bottlenecks

# Auto-optimize configuration
python scripts/optimize.py --auto-optimize

# Save optimized configuration
python scripts/optimize.py --auto-optimize --save-config my_optimized.yaml

# Benchmark configuration
python scripts/optimize.py --benchmark --config configs/my_config.yaml

# Use specific profile
python scripts/optimize.py --profile fast --benchmark
```

## Configuration

### Configuration Files

Configuration files are in YAML format and located in `configs/`.

**Default Configuration**:
```bash
configs/default_config.yaml
```

**Profiles**:
- `configs/profiles/fast.yaml` - Maximum speed
- `configs/profiles/balanced.yaml` - Balance of speed and quality
- `configs/profiles/quality.yaml` - Maximum quality

### Creating Custom Configuration

1. **Copy Default Configuration**:
```bash
cp configs/default_config.yaml configs/my_config.yaml
```

2. **Edit Configuration**:
```yaml
# configs/my_config.yaml
profile: custom

gpu:
  precision: fp16
  max_vram_usage: 0.9

model:
  batch_size: 16
  input_size: [1024, 1024]
  num_workers: 8

performance:
  target_fps: 90
  max_latency_ms: 50.0
```

3. **Validate Configuration**:
```bash
python cli.py --config configs/my_config.yaml --show-config
```

4. **Use Configuration**:
```bash
python cli.py input.mp4 -c configs/my_config.yaml -o output.mp4
```

### Configuration Override

Override specific parameters via command line:

```bash
python cli.py input.mp4 \
  --profile balanced \
  --batch-size 16 \
  --precision fp16 \
  --workers 8 \
  -o output.mp4
```

## Benchmarking

### Running Benchmarks

```bash
# Full benchmark suite (recommended first time)
python benchmarks/benchmark_suite.py --output-dir ./results

# This will test:
# - Resolutions: 1080p, 4K, 6K, 8K
# - Batch sizes: 1, 4, 8, 16, 32
# - Precision modes: FP32, FP16
```

### Analyzing Results

```bash
# Generate analysis report and plots
python benchmarks/performance_analyzer.py results/benchmark_results.json

# Output:
# - results/analysis/performance_report.txt
# - results/analysis/resolution_impact.png
# - results/analysis/batch_size_impact.png
# - results/analysis/precision_impact.png
# - results/analysis/performance_summary.png
```

### Understanding Results

**Key Metrics**:
- **FPS**: Frames processed per second
- **Latency**: Time to process single frame
- **GPU Utilization**: Percentage of GPU used
- **GPU Memory**: VRAM usage

**Target Values** (RTX 3090):
- FPS: 90+ for VR
- Latency: <50ms per frame
- GPU Utilization: >80%
- GPU Memory: 70-90%

### Benchmark Examples

**Find Optimal Batch Size**:
```bash
# Test different batch sizes
python benchmarks/benchmark_suite.py --batch-only

# Analyze results
python benchmarks/performance_analyzer.py results/benchmark_results.json

# Look for "Optimal Batch Size" in report
```

**Compare Precision Modes**:
```bash
# Test FP32 vs FP16
python benchmarks/benchmark_suite.py --precision-only

# Check speedup in report
# Expected: ~2x speedup with FP16 on RTX 3090
```

## Optimization

### Automatic Optimization

```bash
# Detect hardware and optimize
python scripts/optimize.py --auto-optimize --save-config optimized.yaml

# This will:
# 1. Detect GPU/CPU specs
# 2. Find optimal batch size
# 3. Set optimal num_workers
# 4. Choose best precision mode
# 5. Save to optimized.yaml
```

### Manual Optimization

#### Step 1: Detect Bottlenecks

```bash
python scripts/optimize.py --detect-bottlenecks
```

**Common Bottlenecks**:
- Low GPU utilization → Increase batch size
- High GPU memory → Reduce batch size
- High CPU usage → Reduce workers
- High latency → Reduce batch size or use FP16

#### Step 2: Adjust Configuration

Based on bottleneck detection, edit configuration:

```yaml
# If low GPU utilization
model:
  batch_size: 16  # Increase

# If high GPU memory
model:
  batch_size: 4  # Decrease

# If high CPU usage
model:
  num_workers: 4  # Decrease

# If high latency
gpu:
  precision: fp16  # Use FP16
```

#### Step 3: Benchmark

```bash
python scripts/optimize.py --benchmark --config configs/my_config.yaml
```

#### Step 4: Iterate

Repeat steps 1-3 until performance targets are met.

## Advanced Usage

### Processing VR Stereoscopic Video

```yaml
vr:
  stereoscopic: true
  ipd: 63.0  # Interpupillary distance (mm)
  process_both_eyes: true
  eye_separation_method: split  # or 'metadata'
```

```bash
python cli.py vr_video.mp4 -c configs/vr_config.yaml -o vr_output.mp4
```

### Batch Processing Multiple Videos

```bash
#!/bin/bash
# process_batch.sh

for video in videos/*.mp4; do
  echo "Processing $video"
  python cli.py "$video" -o "output/$(basename $video)" --profile fast
done
```

### Using Custom Models

```yaml
model:
  name: custom_model
  checkpoint_path: /path/to/model.pth
  config_path: /path/to/config.yaml
```

### Caching for Faster Re-processing

```yaml
processing:
  enable_caching: true
  cache_size_gb: 8.0  # Increase for more caching
```

**Clear Cache**:
```bash
rm -rf ~/.cache/vr_body_segmentation
```

### Logging

**Configure Logging**:
```yaml
logging:
  level: DEBUG  # DEBUG, INFO, WARNING, ERROR
  log_dir: ./logs
  json_logging: true  # Structured JSON logs
```

**View Logs**:
```bash
# Real-time log viewing
tail -f logs/vr_body_segmentation.log

# Performance logs (JSON)
cat logs/vr_body_segmentation_performance.json | jq
```

### Multi-GPU Processing

```yaml
gpu:
  device_id: 0  # Primary GPU

# For multi-GPU, use process pool
processing:
  use_multiprocessing: true
```

## Examples

### Example 1: Quick Processing

Process a video with default settings:

```bash
python cli.py my_video.mp4 -o output.mp4
```

### Example 2: High-Performance VR Processing

Process VR content at maximum speed:

```bash
python cli.py vr_video.mp4 \
  --profile fast \
  --batch-size 32 \
  --precision fp16 \
  -o vr_output.mp4
```

### Example 3: Quality-First Processing

Process with highest quality (slower):

```bash
python cli.py input.mp4 \
  --profile quality \
  --batch-size 4 \
  --precision fp32 \
  -o high_quality_output.mp4
```

### Example 4: Custom Configuration

```yaml
# configs/my_vr_config.yaml
profile: custom

gpu:
  precision: fp16
  max_vram_usage: 0.9

model:
  batch_size: 16
  input_size: [1024, 1024]
  num_workers: 8

vr:
  stereoscopic: true
  process_both_eyes: true

output:
  format: mp4
  codec: h264_nvenc
  bitrate: 100M
```

```bash
python cli.py vr_input.mp4 -c configs/my_vr_config.yaml -o vr_output.mp4
```

### Example 5: Benchmarking and Optimization Workflow

Complete workflow from benchmarking to optimized processing:

```bash
# Step 1: Run comprehensive benchmark
python benchmarks/benchmark_suite.py --output-dir ./benchmark_results

# Step 2: Analyze results
python benchmarks/performance_analyzer.py \
  benchmark_results/benchmark_results.json

# Step 3: Auto-optimize configuration
python scripts/optimize.py \
  --auto-optimize \
  --save-config configs/optimized.yaml

# Step 4: Process with optimized config
python cli.py input.mp4 \
  -c configs/optimized.yaml \
  -o output.mp4

# Step 5: Monitor performance
python cli.py --interactive
# In interactive mode, type: stats
```

### Example 6: 8K Video Processing

Process 8K video with appropriate settings:

```bash
# Create 8K configuration
cat > configs/8k_config.yaml << EOF
profile: 8k

model:
  batch_size: 2
  input_size: [1024, 1024]
  num_workers: 4

gpu:
  precision: fp16
  max_vram_usage: 0.95

performance:
  target_fps: 30
  max_latency_ms: 100.0
EOF

# Process 8K video
python cli.py 8k_video.mp4 -c configs/8k_config.yaml -o 8k_output.mp4
```

### Example 7: Real-time Monitoring

Monitor performance in real-time:

```bash
# Terminal 1: Run processing with live stats
python cli.py input.mp4 -o output.mp4 --log-level INFO

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor CPU/Memory
htop

# Terminal 4: Monitor logs
tail -f logs/vr_body_segmentation.log
```

## Tips and Best Practices

### Performance Tips

1. **Always Benchmark First**:
   ```bash
   python benchmarks/benchmark_suite.py
   ```

2. **Use FP16 on RTX 3090**:
   - 2x faster than FP32
   - Minimal quality loss
   - More VRAM available

3. **Optimize Batch Size**:
   - Start with 8, adjust based on GPU utilization
   - Target 80-90% VRAM usage
   - Monitor with `nvidia-smi`

4. **Enable Caching for Development**:
   ```yaml
   processing:
     enable_caching: true
     cache_size_gb: 8.0
   ```

5. **Use Appropriate Profile**:
   - Development/Testing: `balanced`
   - Production VR: `fast`
   - Archival Quality: `quality`

### Workflow Tips

1. **Test with Short Clips First**:
   ```bash
   # Extract first 100 frames
   ffmpeg -i input.mp4 -frames:v 100 test.mp4
   python cli.py test.mp4 -o test_output.mp4
   ```

2. **Verify Output Quality**:
   ```bash
   # Compare input and output
   ffplay -i output.mp4
   ```

3. **Monitor System Resources**:
   - GPU: `nvidia-smi -l 1`
   - CPU: `htop`
   - Disk: `iostat -x 1`

4. **Keep Logs**:
   ```bash
   # Archive logs after processing
   tar -czf logs_$(date +%Y%m%d).tar.gz logs/
   ```

### Troubleshooting Tips

1. **Start Simple**:
   ```bash
   # Minimal configuration
   python cli.py input.mp4 -o output.mp4 --batch-size 1
   ```

2. **Check Hardware**:
   ```bash
   python cli.py --show-hardware
   ```

3. **Enable Debug Logging**:
   ```bash
   python cli.py input.mp4 -o output.mp4 --log-level DEBUG
   ```

4. **Run Diagnostics**:
   ```bash
   python scripts/optimize.py --detect-bottlenecks
   ```

## Getting Help

### Documentation

- **Performance Guide**: `docs/performance_guide.md`
- **Optimization Techniques**: `docs/optimization_techniques.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **This Guide**: `docs/usage_guide.md`

### Command Help

```bash
# CLI help
python cli.py --help

# Benchmark help
python benchmarks/benchmark_suite.py --help

# Optimization help
python scripts/optimize.py --help
```

### Interactive Help

```bash
# Start interactive mode
python cli.py --interactive

# Type 'help' for commands
> help
```

## Next Steps

After reading this guide:

1. ✓ Run hardware detection: `python cli.py --show-hardware`
2. ✓ Run benchmarks: `python benchmarks/benchmark_suite.py`
3. ✓ Auto-optimize: `python scripts/optimize.py --auto-optimize`
4. ✓ Test with sample video
5. ✓ Read performance guide for advanced tuning
6. ✓ Process your VR content!
