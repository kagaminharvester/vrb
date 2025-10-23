# Quick Start Guide

Get started with VR Body Segmentation in 5 minutes!

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or similar
- **CPU**: Multi-core processor (Threadripper 3990X recommended)
- **RAM**: 48GB recommended, 16GB minimum
- **Storage**: 50GB+ free space

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.7 or higher (for GPU acceleration)
- **FFmpeg**: Latest version

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/vr-body-segmentation.git
cd vr-body-segmentation
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 5. Verify Installation
```bash
python src/main.py --help
```

## Quick Usage

### Basic Segmentation

Process a VR video with default settings:

```bash
python src/main.py \
    -i input_vr_video.mp4 \
    -o output_segmented.mp4
```

### With Custom Settings

```bash
python src/main.py \
    -i input_vr_video.mp4 \
    -o output_segmented.mp4 \
    --model deeplabv3 \
    --device cuda \
    --batch-size 4 \
    --quality high \
    --codec h264 \
    --save-masks \
    --visualization-mode overlay
```

### Batch Processing

Process multiple videos at once:

```bash
python src/main.py \
    -i /path/to/video/directory/ \
    -o /path/to/output/directory/ \
    --batch-mode \
    --model deeplabv3
```

## Python API Usage

### Simple Example

```python
from core.video_pipeline import VRVideoSegmentationPipeline
from main import create_default_config

# Create configuration
config = create_default_config(
    input_path="input.mp4",
    output_path="output.mp4",
    model_name="deeplabv3",
    device="cuda"
)

# Create and run pipeline
pipeline = VRVideoSegmentationPipeline(config)
summary = pipeline.process()

print(f"Processed {summary['processed_frames']} frames in {summary['elapsed_time']:.2f}s")
```

### With Progress Callback

```python
def progress_callback(stats):
    print(f"Progress: {stats['progress_percent']:.1f}%")

pipeline = VRVideoSegmentationPipeline(config)
summary = pipeline.process(progress_callback=progress_callback)
```

## Supported Models

| Model | Speed | Accuracy | VRAM | Best For |
|-------|-------|----------|------|----------|
| **DeepLabV3** | Medium | High | 6GB | Balanced (Default) |
| **BiSeNet** | Fast | Medium | 4GB | Real-time |
| **SAM** | Slow | Very High | 8GB | Quality |

### Selecting a Model

```bash
# Fast (real-time)
python src/main.py -i input.mp4 -o output.mp4 --model bisenet

# Balanced (default)
python src/main.py -i input.mp4 -o output.mp4 --model deeplabv3

# High quality
python src/main.py -i input.mp4 -o output.mp4 --model sam
```

## VR Video Formats

The application automatically detects and handles:

- **Side-by-Side (SBS)**: Left and right views side-by-side
- **Over-Under (OU)**: Left view over right view
- **Equirectangular**: 360° video format
- **Mono**: Standard video (non-VR)

## Output Options

### Visualization Modes

```bash
# Overlay mask on original (default)
--visualization-mode overlay

# Side-by-side comparison
--visualization-mode side_by_side

# Mask only
--visualization-mode mask_only
```

### Quality Settings

```bash
# Low quality (fast encoding)
--quality low

# Medium quality
--quality medium

# High quality (default)
--quality high

# Lossless (large file size)
--quality lossless
```

### Codec Options

```bash
# H.264 (best compatibility)
--codec h264

# H.265 (better compression)
--codec h265

# VP9 (WebM format)
--codec vp9

# AV1 (best compression, slower)
--codec av1
```

## Performance Tips

### 1. Use GPU Acceleration
Always use `--device cuda` if you have a compatible GPU.

### 2. Optimize Batch Size
```bash
# For RTX 3090 (24GB)
--batch-size 8

# For RTX 3080 (10GB)
--batch-size 4

# For RTX 3060 (12GB)
--batch-size 6
```

### 3. Use Half Precision
Automatically enabled on CUDA devices for 2x speedup.

### 4. Hardware Encoding
Use hardware-accelerated encoding when available:
```bash
--codec h264  # Will use NVENC on NVIDIA GPUs
```

## Common Issues

### Out of Memory (OOM)

**Solution**: Reduce batch size or resolution
```bash
--batch-size 1  # Process one frame at a time
```

### Slow Processing

**Solution**:
1. Check GPU utilization: `nvidia-smi`
2. Use faster model: `--model bisenet`
3. Reduce quality: `--quality medium`

### FFmpeg Not Found

**Solution**: Install FFmpeg and ensure it's in PATH
```bash
which ffmpeg  # Should show path to ffmpeg
```

### CUDA Out of Memory

**Solution**:
1. Reduce batch size
2. Use smaller model
3. Process at lower resolution

## Examples

See the `examples/` directory for more detailed examples:

- `basic_usage.py` - Simple usage example
- `batch_processing.py` - Process multiple videos
- `advanced_features.py` - Advanced features and optimization

Run examples:
```bash
python examples/basic_usage.py
python examples/batch_processing.py
python examples/advanced_features.py
```

## Configuration Files

Use JSON configuration for complex setups:

```bash
python src/main.py --config config_example.json
```

See `src/config_example.json` for a complete configuration template.

## Next Steps

1. **Read the Documentation**: See `src/README.md` for detailed documentation
2. **Explore Examples**: Check the `examples/` directory
3. **Optimize Performance**: See performance tuning guide
4. **Contribute**: Submit issues or pull requests

## Getting Help

- **Documentation**: `src/README.md`
- **Issues**: GitHub Issues
- **Examples**: `examples/` directory
- **Logs**: Check `vr_segmentation.log`

## Performance Benchmarks

On NVIDIA RTX 3090:

| Model | Resolution | FPS | VRAM Usage |
|-------|-----------|-----|------------|
| BiSeNet | 1080p | 80-120 | 4GB |
| DeepLabV3 | 1080p | 40-60 | 6GB |
| SAM | 1080p | 15-25 | 8GB |
| DeepLabV3 | 4K | 15-25 | 12GB |
| DeepLabV3 | 8K | 4-8 | 20GB |

*Note: FPS varies based on video complexity and settings*

## License

See LICENSE file for details.

---

**Ready to go?** Start with:
```bash
python src/main.py -i your_vr_video.mp4 -o output.mp4
```
