#!/usr/bin/env python3
"""
Basic Usage Example for VR Body Segmentation

Demonstrates simple usage of the VR body segmentation pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.video_pipeline import VRVideoSegmentationPipeline, PipelineConfig
from core.segmentation_engine import SegmentationEngineConfig
from models.model_loader import ModelConfig
from preprocessing.video_decoder import VideoDecoderConfig
from preprocessing.frame_preprocessor import PreprocessConfig
from postprocessing.mask_processor import MaskProcessorConfig
from postprocessing.video_encoder import VideoEncoderConfig
from vr.stereo_processor import StereoConsistencyConfig


def progress_callback(stats):
    """Print progress updates."""
    print(f"Progress: {stats['progress_percent']:.1f}% | "
          f"FPS: {stats['fps']:.2f} | "
          f"Frames: {stats['processed_frames']}/{stats['total_frames']}")


def main():
    """Run basic segmentation example."""

    # Input and output paths
    input_video = "input_vr_video.mp4"
    output_video = "output_segmented.mp4"

    print("=" * 60)
    print("VR Body Segmentation - Basic Usage Example")
    print("=" * 60)

    # Step 1: Configure the model
    print("\n1. Configuring model...")
    model_config = ModelConfig(
        model_name='deeplabv3',  # Use DeepLabV3 (good balance)
        device='cuda',            # Use GPU
        use_fp16=True,           # Use half precision for speed
        batch_size=1,            # Process 1 frame at a time
        input_size=(1080, 1920), # Target resolution
    )

    # Step 2: Configure preprocessing
    print("2. Configuring preprocessing...")
    preprocess_config = PreprocessConfig(
        target_size=(1080, 1920),
        normalize=True,
        device='cuda',
    )

    # Step 3: Configure postprocessing
    print("3. Configuring postprocessing...")
    postprocess_config = MaskProcessorConfig(
        enable_refinement=True,
        enable_edge_smoothing=True,
        enable_temporal_filtering=True,
    )

    # Step 4: Configure segmentation engine
    print("4. Configuring segmentation engine...")
    segmentation_config = SegmentationEngineConfig(
        model_config=model_config,
        preprocess_config=preprocess_config,
        postprocess_config=postprocess_config,
    )

    # Step 5: Configure video decoder
    print("5. Configuring video decoder...")
    decoder_config = VideoDecoderConfig(
        video_path=input_video,
        buffer_size=30,
        use_threading=True,
        decode_mode='stereo',  # Auto-detect and handle stereo
    )

    # Step 6: Configure video encoder
    print("6. Configuring video encoder...")
    encoder_config = VideoEncoderConfig(
        output_path=output_video,
        codec='h264',
        quality='high',
        hardware_acceleration=True,
    )

    # Step 7: Configure stereo processing (optional)
    print("7. Configuring stereo processing...")
    stereo_config = StereoConsistencyConfig(
        enable=True,
        method='weighted',
        temporal_weight=0.3,
    )

    # Step 8: Create pipeline configuration
    print("8. Creating pipeline configuration...")
    pipeline_config = PipelineConfig(
        input_video_path=input_video,
        output_video_path=output_video,
        decoder_config=decoder_config,
        segmentation_config=segmentation_config,
        encoder_config=encoder_config,
        stereo_config=stereo_config,
        enable_stereo_processing=True,
        create_visualization=True,
        visualization_mode='overlay',
    )

    # Step 9: Create and run pipeline
    print("9. Creating pipeline...")
    pipeline = VRVideoSegmentationPipeline(pipeline_config)

    print("10. Processing video...")
    print("-" * 60)

    # Process with progress callback
    summary = pipeline.process(progress_callback=progress_callback)

    # Step 10: Print results
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Status: {summary['status']}")
    print(f"Total Frames: {summary['total_frames']}")
    print(f"Processed Frames: {summary['processed_frames']}")
    print(f"Total Time: {summary['elapsed_time']:.2f}s")
    print(f"Average FPS: {summary['avg_fps']:.2f}")
    print(f"Output saved to: {output_video}")
    print("=" * 60)


if __name__ == '__main__':
    main()
