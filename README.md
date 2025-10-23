# VR Body Segmentation

High-performance real-time body segmentation for VR content, optimized for NVIDIA RTX 3090 and AMD Ryzen Threadripper 3990X.

## Features

- **Real-time Performance**: 90+ FPS for VR content
- **Multi-Resolution Support**: 1080p to 8K video processing
- **GPU Acceleration**: Optimized for NVIDIA RTX 3090 (24GB VRAM)
- **CPU Optimization**: Efficient use of 128 CPU threads
- **Automatic Optimization**: Auto-tune for optimal performance
- **Comprehensive Benchmarking**: Test all configurations
- **VR Stereoscopic Support**: Process both eyes separately
- **Flexible Configuration**: YAML-based config with profiles
- **Rich CLI**: Interactive mode with real-time stats
- **Production-Ready**: Error handling, logging, caching

## Quick Start

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml rich matplotlib seaborn psutil

# Process a video
python cli.py input.mp4 -o output.mp4 --profile fast

# Interactive mode
python cli.py --interactive

# Run benchmarks
python benchmarks/benchmark_suite.py
```

## Performance Targets

Target hardware: NVIDIA RTX 3090, AMD Ryzen Threadripper 3990X (128 threads), 48GB RAM

| Resolution | Batch Size | Precision | Target FPS | Latency |
|------------|------------|-----------|------------|---------|
| 1080p      | 16-32      | FP16      | 120+ FPS   | <20ms   |
| 4K         | 8-16       | FP16      | 90+ FPS    | <30ms   |
| 6K         | 4-8        | FP16      | 60+ FPS    | <40ms   |
| 8K         | 2-4        | FP16      | 30+ FPS    | <50ms   |

## Project Structure

```
vr-body-segmentation/
├── cli.py                          # Main CLI application
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── configs/                        # Configuration files
│   ├── default_config.yaml         # Default configuration
│   └── profiles/                   # Performance profiles
│       ├── fast.yaml               # Maximum speed (120+ FPS)
│       ├── balanced.yaml           # Balanced (90 FPS)
│       └── quality.yaml            # Maximum quality (60 FPS)
│
├── src/                            # Source code
│   ├── utils/                      # Utility modules
│   │   ├── logger.py               # Logging framework
│   │   ├── config_manager.py       # Configuration management
│   │   └── cache_manager.py        # Caching system
│   ├── core/                       # Core processing
│   ├── models/                     # Model implementations
│   ├── preprocessing/              # Data preprocessing
│   ├── postprocessing/             # Result postprocessing
│   ├── gpu/                        # GPU optimizations
│   └── vr/                         # VR-specific processing
│
├── benchmarks/                     # Benchmarking suite
│   ├── benchmark_suite.py          # Comprehensive benchmarks
│   └── performance_analyzer.py     # Analysis and visualization
│
├── scripts/                        # Utility scripts
│   └── optimize.py                 # Optimization tool
│
├── docs/                           # Documentation
│   ├── usage_guide.md              # Usage instructions
│   ├── performance_guide.md        # Performance tuning
│   ├── optimization_techniques.md  # Optimization details
│   └── troubleshooting.md          # Common issues
│
└── tests/                          # Unit tests
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+
- NVIDIA Driver 520.xx+
- FFmpeg

### Install

```bash
# Clone repository
cd /home/pi/vr-body-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python cli.py --show-hardware
```

## Usage

### Basic Usage

```bash
# Process video with default settings
python cli.py input.mp4 -o output.mp4

# Use fast profile
python cli.py input.mp4 --profile fast -o output.mp4

# Custom configuration
python cli.py input.mp4 -c configs/my_config.yaml -o output.mp4
```

### Optimization

```bash
# Auto-optimize for your hardware
python scripts/optimize.py --auto-optimize --save-config optimized.yaml

# Detect bottlenecks
python scripts/optimize.py --detect-bottlenecks

# Use optimized configuration
python cli.py input.mp4 -c optimized.yaml -o output.mp4
```

### Benchmarking

```bash
# Run all benchmarks
python benchmarks/benchmark_suite.py

# Analyze results
python benchmarks/performance_analyzer.py results/benchmark_results.json

# Specific benchmarks
python benchmarks/benchmark_suite.py --resolution-only
python benchmarks/benchmark_suite.py --batch-only
python benchmarks/benchmark_suite.py --precision-only
```

### Interactive Mode

```bash
python cli.py --interactive

# Available commands:
# - hardware: Show hardware info
# - config: Show current configuration
# - stats: Show performance statistics
# - profile <name>: Load a profile
# - exit: Exit interactive mode
```

## Configuration

### Profiles

Three built-in profiles for different use cases:

1. **Fast** (`--profile fast`):
   - Target: 120+ FPS
   - Batch size: 16-32
   - Precision: FP16
   - Use case: Real-time VR processing

2. **Balanced** (`--profile balanced`):
   - Target: 90 FPS
   - Batch size: 8-16
   - Precision: FP16
   - Use case: Production VR content

3. **Quality** (`--profile quality`):
   - Target: 60 FPS
   - Batch size: 4
   - Precision: FP32
   - Use case: High-quality archival

### Custom Configuration

Create custom YAML configuration:

```yaml
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

output:
  codec: h264_nvenc
  bitrate: 50M
```

## Performance Optimization

### Key Parameters

1. **Batch Size**: Most impactful for throughput
   - Increase until GPU memory is 80-90% utilized
   - Monitor with: `nvidia-smi`

2. **Precision Mode**: 2x speedup with FP16
   - Use FP16 on RTX 3090
   - Minimal quality loss

3. **Number of Workers**: CPU preprocessing
   - Rule: `num_workers = CPU_cores / 4`
   - Threadripper: 16-32 workers

4. **Input Resolution**: Linear to quadratic impact
   - Match to video resolution
   - Common: 512x512, 1024x1024

### Optimization Workflow

```bash
# 1. Benchmark to understand performance
python benchmarks/benchmark_suite.py

# 2. Analyze results
python benchmarks/performance_analyzer.py results/benchmark_results.json

# 3. Auto-optimize
python scripts/optimize.py --auto-optimize --save-config optimized.yaml

# 4. Test optimized configuration
python cli.py test_video.mp4 -c optimized.yaml -o test_output.mp4

# 5. Production processing
python cli.py production_video.mp4 -c optimized.yaml -o final_output.mp4
```

## Monitoring

### Real-time Monitoring

```bash
# Terminal 1: Run application
python cli.py input.mp4 -o output.mp4

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor CPU
htop
```

### Performance Metrics

Monitor in interactive mode:
```bash
python cli.py --interactive
> stats
```

Shows:
- Current FPS
- GPU utilization
- GPU memory usage
- CPU utilization
- Frames processed

## Documentation

- **[Usage Guide](docs/usage_guide.md)**: Complete usage instructions
- **[Performance Guide](docs/performance_guide.md)**: Performance tuning
- **[Optimization Techniques](docs/optimization_techniques.md)**: Advanced optimization
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

## Examples

### Example 1: Quick VR Processing

```bash
python cli.py vr_video.mp4 --profile fast -o vr_output.mp4
```

### Example 2: High-Quality 8K Processing

```bash
python cli.py 8k_video.mp4 \
  --profile quality \
  --batch-size 2 \
  --precision fp16 \
  -o 8k_output.mp4
```

### Example 3: Batch Processing

```bash
#!/bin/bash
for video in videos/*.mp4; do
  python cli.py "$video" -o "output/$(basename $video)" --profile fast
done
```

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Reduce batch size
python cli.py input.mp4 -o output.mp4 --batch-size 4
```

**Low FPS**:
```bash
# Auto-optimize
python scripts/optimize.py --auto-optimize

# Or increase batch size
python cli.py input.mp4 -o output.mp4 --batch-size 16
```

**High Latency**:
```bash
# Use FP16 and reduce batch size
python cli.py input.mp4 -o output.mp4 --precision fp16 --batch-size 4
```

See [Troubleshooting Guide](docs/troubleshooting.md) for more solutions.

## Performance Tips

1. Always run benchmarks first
2. Use FP16 on RTX 3090 (2x faster)
3. Optimize batch size for 80-90% GPU utilization
4. Enable caching for development
5. Monitor with `nvidia-smi` and `htop`
6. Use appropriate profile for use case
7. Test with short clips first

## System Requirements

### Minimum
- GPU: NVIDIA RTX 2060 (6GB VRAM)
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB free

### Recommended (Target)
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- CPU: AMD Ryzen Threadripper 3990X (128 threads)
- RAM: 48GB
- Storage: 500GB NVMe SSD

### Optimal
- GPU: NVIDIA RTX 4090 or A100
- CPU: AMD Ryzen Threadripper PRO 5995WX
- RAM: 128GB
- Storage: 2TB NVMe SSD

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
1. Check [Troubleshooting Guide](docs/troubleshooting.md)
2. Run diagnostics: `python scripts/optimize.py --detect-bottlenecks`
3. Enable debug logging: `--log-level DEBUG`
4. Check logs: `tail -f logs/vr_body_segmentation.log`

## Acknowledgments

- SAM2 model by Meta AI
- PyTorch team
- NVIDIA CUDA team
