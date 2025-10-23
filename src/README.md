# VR Body Segmentation - Source Code

This directory contains the core implementation of the VR video body segmentation application.

## Directory Structure

```
src/
├── core/                          # Core application logic
│   ├── video_pipeline.py         # Main video processing pipeline
│   └── segmentation_engine.py    # Body segmentation inference engine
│
├── models/                        # Model loading and management
│   └── model_loader.py           # Model loader with support for multiple architectures
│
├── preprocessing/                 # Input processing
│   ├── video_decoder.py          # VR video decoder with stereo support
│   └── frame_preprocessor.py     # Frame preprocessing and augmentation
│
├── postprocessing/               # Output processing
│   ├── mask_processor.py         # Mask refinement and filtering
│   └── video_encoder.py          # Video encoding with multiple codecs
│
├── vr/                           # VR-specific features
│   ├── stereo_processor.py       # Stereo consistency enforcement
│   └── depth_estimator.py        # Optional depth estimation
│
├── utils/                        # Utility functions
│   └── video_utils.py            # Video processing utilities
│
├── main.py                       # Command-line entry point
└── config_example.json           # Example configuration file
```

## Key Components

### 1. Video Pipeline (`core/video_pipeline.py`)

Main orchestrator that integrates all components:
- `VRVideoSegmentationPipeline`: Complete processing pipeline
- `BatchVideoPipeline`: Process multiple videos
- `ProgressTracker`: Track and report progress

### 2. Segmentation Engine (`core/segmentation_engine.py`)

Core inference engine:
- `SegmentationEngine`: Main inference engine
- `BatchSegmentationEngine`: Optimized batch processing
- `StreamingSegmentationEngine`: Real-time processing

### 3. Model Loader (`models/model_loader.py`)

Model management:
- Support for DeepLabV3, BiSeNet, SAM
- Model optimization (FP16, torch.compile)
- Model warm-up and ensemble predictions

### 4. Video Decoder (`preprocessing/video_decoder.py`)

VR video input handling:
- `VRVideoDecoder`: Main decoder with stereo support
- Format detection (side-by-side, over-under, equirectangular)
- Threaded decoding with frame buffering
- Support for H.264, H.265, VP9, AV1

### 5. Frame Preprocessor (`preprocessing/frame_preprocessor.py`)

Frame preparation:
- `FramePreprocessor`: Single frame preprocessing
- `StereoPreprocessor`: Stereo pair preprocessing
- `AugmentationPipeline`: Data augmentation
- Normalization and color conversion

### 6. Mask Processor (`postprocessing/mask_processor.py`)

Mask refinement:
- Morphological operations
- Edge smoothing
- Temporal filtering
- Small object removal

### 7. Video Encoder (`postprocessing/video_encoder.py`)

Video output:
- `VideoEncoder`: Main encoder with FFmpeg support
- `MaskVideoEncoder`: Specialized mask encoder
- Multiple codec support
- Hardware acceleration

### 8. Stereo Processor (`vr/stereo_processor.py`)

Stereo consistency:
- Multiple consistency methods (average, weighted, cross-check, optical flow)
- Temporal smoothing
- Disparity estimation
- Geometric correction

### 9. Depth Estimator (`vr/depth_estimator.py`)

Optional depth estimation:
- MiDaS-based monocular depth
- Stereo matching depth
- Depth-guided processing
- Depth effects (blur, fog, bokeh)

## Usage

### Command Line Interface

Basic usage:
```bash
python src/main.py -i input.mp4 -o output.mp4
```

With options:
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

Batch processing:
```bash
python src/main.py \
    -i /path/to/videos/ \
    -o /path/to/output/ \
    --batch-mode \
    --model deeplabv3
```

### Python API

```python
from core.video_pipeline import VRVideoSegmentationPipeline, PipelineConfig
from models.model_loader import ModelConfig
from preprocessing.frame_preprocessor import PreprocessConfig
from postprocessing.mask_processor import MaskProcessorConfig
from core.segmentation_engine import SegmentationEngineConfig

# Create configurations
model_config = ModelConfig(
    model_name='deeplabv3',
    device='cuda',
    use_fp16=True,
    batch_size=4
)

preprocess_config = PreprocessConfig(
    target_size=(1080, 1920),
    normalize=True
)

postprocess_config = MaskProcessorConfig(
    enable_refinement=True,
    enable_temporal_filtering=True
)

segmentation_config = SegmentationEngineConfig(
    model_config=model_config,
    preprocess_config=preprocess_config,
    postprocess_config=postprocess_config
)

# Create pipeline config
pipeline_config = PipelineConfig(
    input_video_path='input.mp4',
    output_video_path='output.mp4',
    segmentation_config=segmentation_config,
    # ... other configs
)

# Run pipeline
pipeline = VRVideoSegmentationPipeline(pipeline_config)
summary = pipeline.process()

print(f"Processed {summary['processed_frames']} frames")
print(f"Average FPS: {summary['avg_fps']:.2f}")
```

## Supported Models

1. **DeepLabV3+** (Default)
   - Good balance of speed and accuracy
   - Pretrained on COCO dataset
   - ~30-60 FPS on RTX 3090

2. **BiSeNet**
   - Fast real-time segmentation
   - Lower accuracy than DeepLabV3
   - ~60-120 FPS on RTX 3090

3. **Segment Anything (SAM)**
   - Highest accuracy
   - Slower inference
   - ~10-20 FPS on RTX 3090

## Supported Video Formats

### Input Formats
- Side-by-side (SBS)
- Over-under (OU)
- Equirectangular (360°)
- Mono (standard video)

### Codecs
- H.264 (AVC)
- H.265 (HEVC)
- VP9
- AV1

### Resolutions
- 1080p (1920x1080)
- 2K (2560x1440)
- 4K (3840x2160)
- 6K (5760x2880)
- 8K (7680x4320)

## Performance Optimization

### GPU Acceleration
- FP16 inference for 2x speedup
- Torch.compile for optimized kernels
- Batch processing for higher throughput
- Hardware-accelerated encoding (NVENC)

### CPU Optimization
- Multi-threaded frame decoding
- Efficient frame buffering
- Parallel preprocessing

### Memory Management
- Streaming processing for large videos
- Dynamic batch sizing
- Frame queue management

## Configuration

See `config_example.json` for a complete configuration example.

Key configuration sections:
- `model`: Model selection and optimization
- `preprocessing`: Frame preparation settings
- `postprocessing`: Mask refinement options
- `stereo`: Stereo consistency settings
- `encoder`: Output video settings

## Error Handling

The application includes comprehensive error handling:
- Video validation before processing
- Graceful degradation on errors
- Detailed error logging
- Resource cleanup on failure

## Logging

Logs are written to:
- Console (stdout)
- `vr_segmentation.log` file

Log levels:
- INFO: Progress and status updates
- WARNING: Non-critical issues
- ERROR: Processing failures

## Requirements

See `requirements.txt` for Python dependencies.

Key dependencies:
- PyTorch 2.0+ with CUDA support
- OpenCV 4.x
- NumPy
- SciPy
- FFmpeg (for advanced encoding)

## Hardware Requirements

### Minimum
- GPU: NVIDIA GTX 1060 (6GB VRAM)
- CPU: 4 cores
- RAM: 8GB

### Recommended (as specified)
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- CPU: AMD Ryzen Threadripper 3990X (128 threads)
- RAM: 48GB

## License

See LICENSE file in repository root.

## Support

For issues and questions, please open an issue on the project repository.
