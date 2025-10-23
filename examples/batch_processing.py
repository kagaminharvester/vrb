#!/usr/bin/env python3
"""
Batch Processing Example

Demonstrates processing multiple VR videos in batch.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.video_pipeline import BatchVideoPipeline, PipelineConfig
from core.segmentation_engine import SegmentationEngineConfig
from models.model_loader import ModelConfig
from preprocessing.video_decoder import VideoDecoderConfig
from preprocessing.frame_preprocessor import PreprocessConfig
from postprocessing.mask_processor import MaskProcessorConfig
from postprocessing.video_encoder import VideoEncoderConfig


def batch_progress_callback(video_idx, total_videos, stats):
    """Print batch progress updates."""
    print(f"[Video {video_idx+1}/{total_videos}] "
          f"Progress: {stats['progress_percent']:.1f}% | "
          f"FPS: {stats['fps']:.2f}")


def main():
    """Run batch processing example."""

    # Input directory with VR videos
    input_videos = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4",
    ]
    output_dir = "output_batch/"

    print("=" * 60)
    print("VR Body Segmentation - Batch Processing Example")
    print("=" * 60)
    print(f"Processing {len(input_videos)} videos...")

    # Create base configuration
    model_config = ModelConfig(
        model_name='deeplabv3',
        device='cuda',
        use_fp16=True,
        batch_size=4,  # Larger batch for efficiency
    )

    preprocess_config = PreprocessConfig(
        target_size=(1080, 1920),
        normalize=True,
        device='cuda',
    )

    postprocess_config = MaskProcessorConfig(
        enable_refinement=True,
        enable_temporal_filtering=True,
    )

    segmentation_config = SegmentationEngineConfig(
        model_config=model_config,
        preprocess_config=preprocess_config,
        postprocess_config=postprocess_config,
    )

    decoder_config = VideoDecoderConfig(
        video_path="",  # Will be set per video
        use_threading=True,
        decode_mode='stereo',
    )

    encoder_config = VideoEncoderConfig(
        output_path="",  # Will be set per video
        codec='h264',
        quality='high',
        hardware_acceleration=True,
    )

    # Create base pipeline config
    base_config = PipelineConfig(
        input_video_path="",  # Will be set per video
        output_video_path="",  # Will be set per video
        decoder_config=decoder_config,
        segmentation_config=segmentation_config,
        encoder_config=encoder_config,
        enable_stereo_processing=True,
        visualization_mode='overlay',
    )

    # Create batch pipeline
    batch_pipeline = BatchVideoPipeline(
        input_videos=input_videos,
        output_dir=output_dir,
        base_config=base_config
    )

    # Process all videos
    print("\nProcessing videos...")
    results = batch_pipeline.process_all(progress_callback=batch_progress_callback)

    # Print summary
    print("\n" + "=" * 60)
    print("Batch Processing Complete!")
    print("=" * 60)

    for i, result in enumerate(results):
        print(f"\nVideo {i+1}: {result.get('input_path', 'unknown')}")
        if result.get('status') == 'completed':
            print(f"  Status: ✓ Success")
            print(f"  Frames: {result.get('processed_frames', 0)}")
            print(f"  Time: {result.get('total_time', 0):.2f}s")
            print(f"  FPS: {result.get('avg_fps', 0):.2f}")
            print(f"  Output: {result.get('output_path', 'unknown')}")
        else:
            print(f"  Status: ✗ Failed")
            print(f"  Error: {result.get('error', 'unknown')}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
