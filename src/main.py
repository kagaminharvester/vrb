#!/usr/bin/env python3
"""
Main Entry Point for VR Body Segmentation Application

Command-line interface for running the VR video body segmentation pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.video_pipeline import VRVideoSegmentationPipeline, PipelineConfig, BatchVideoPipeline
from core.segmentation_engine import SegmentationEngineConfig
from models.model_loader import ModelConfig
from preprocessing.video_decoder import VideoDecoderConfig
from preprocessing.frame_preprocessor import PreprocessConfig
from postprocessing.mask_processor import MaskProcessorConfig
from postprocessing.video_encoder import VideoEncoderConfig
from vr.stereo_processor import StereoConsistencyConfig
from utils.video_utils import VRFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vr_segmentation.log')
    ]
)

logger = logging.getLogger(__name__)


def create_default_config(
    input_path: str,
    output_path: str,
    model_name: str = 'deeplabv3',
    model_path: str = None,
    device: str = 'cuda',
    batch_size: int = 1,
    quality: str = 'high',
    codec: str = 'h264'
) -> PipelineConfig:
    """
    Create default pipeline configuration.

    Args:
        input_path: Input video path
        output_path: Output video path
        model_name: Model name ('deeplabv3', 'bisenet', 'sam')
        model_path: Optional path to model weights
        device: Device for inference ('cuda' or 'cpu')
        batch_size: Batch size for inference
        quality: Output quality ('low', 'medium', 'high', 'lossless')
        codec: Output codec ('h264', 'h265', 'vp9', 'av1')

    Returns:
        PipelineConfig object
    """
    # Model configuration
    model_config = ModelConfig(
        model_name=model_name,
        model_path=model_path,
        device=device,
        use_fp16=True,
        use_tensorrt=False,
        use_torch_compile=True,
        batch_size=batch_size,
        input_size=(1080, 1920),
        num_classes=2,
        warmup_iterations=10
    )

    # Preprocessing configuration
    preprocess_config = PreprocessConfig(
        target_size=(1080, 1920),
        maintain_aspect=True,
        input_color_space='BGR',
        model_color_space='RGB',
        normalize=True,
        to_tensor=True,
        device=device,
        dtype='float16' if device == 'cuda' else 'float32'
    )

    # Postprocessing configuration
    postprocess_config = MaskProcessorConfig(
        enable_refinement=True,
        enable_edge_smoothing=True,
        enable_temporal_filtering=True,
        edge_smoothing_kernel=5,
        morphology_kernel_size=5,
        temporal_window_size=5,
        temporal_alpha=0.3,
        min_object_size=1000,
        hole_filling=True
    )

    # Segmentation engine configuration
    segmentation_config = SegmentationEngineConfig(
        model_config=model_config,
        preprocess_config=preprocess_config,
        postprocess_config=postprocess_config,
        batch_size=batch_size,
        enable_timing=True,
        use_half_precision=True,
        device=device
    )

    # Decoder configuration
    decoder_config = VideoDecoderConfig(
        video_path=input_path,
        vr_format=None,  # Auto-detect
        buffer_size=30,
        use_threading=True,
        start_frame=0,
        end_frame=None,
        frame_skip=0,
        decode_mode='stereo'
    )

    # Encoder configuration
    encoder_config = VideoEncoderConfig(
        output_path=output_path,
        fps=30.0,  # Will be overridden by input video fps
        codec=codec,
        quality=quality,
        preset='medium',
        hardware_acceleration=True if device == 'cuda' else False,
        vr_format=None,  # Will be set by pipeline
        preserve_metadata=True
    )

    # Stereo consistency configuration
    stereo_config = StereoConsistencyConfig(
        enable=True,
        method='weighted',
        weight_left=0.5,
        weight_right=0.5,
        consistency_threshold=0.8,
        smoothing_kernel=5,
        temporal_weight=0.3
    )

    # Pipeline configuration
    pipeline_config = PipelineConfig(
        input_video_path=input_path,
        output_video_path=output_path,
        decoder_config=decoder_config,
        segmentation_config=segmentation_config,
        encoder_config=encoder_config,
        stereo_config=stereo_config,
        enable_stereo_processing=True,
        enable_progress_callback=True,
        save_masks_separately=False,
        mask_output_path=None,
        num_worker_threads=2,
        frame_queue_size=30,
        create_visualization=True,
        visualization_mode='overlay'
    )

    return pipeline_config


def progress_callback(stats: Dict[str, Any]) -> None:
    """Print progress to console."""
    progress = stats['progress_percent']
    fps = stats['fps']
    eta = stats['eta']

    logger.info(
        f"Progress: {progress:.1f}% | "
        f"FPS: {fps:.2f} | "
        f"ETA: {eta:.1f}s | "
        f"Frames: {stats['processed_frames']}/{stats['total_frames']}"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='VR Video Body Segmentation Application'
    )

    # Input/Output
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input VR video path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output video path'
    )

    # Model settings
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='deeplabv3',
        choices=['deeplabv3', 'bisenet', 'sam'],
        help='Segmentation model to use'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model weights (optional)'
    )

    # Processing settings
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    # Output settings
    parser.add_argument(
        '-q', '--quality',
        type=str,
        default='high',
        choices=['low', 'medium', 'high', 'lossless'],
        help='Output video quality'
    )
    parser.add_argument(
        '-c', '--codec',
        type=str,
        default='h264',
        choices=['h264', 'h265', 'vp9', 'av1'],
        help='Output video codec'
    )

    # Additional options
    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save segmentation masks separately'
    )
    parser.add_argument(
        '--mask-output',
        type=str,
        default=None,
        help='Output path for masks (if --save-masks is set)'
    )
    parser.add_argument(
        '--visualization-mode',
        type=str,
        default='overlay',
        choices=['overlay', 'side_by_side', 'mask_only'],
        help='Visualization mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Load configuration from JSON file'
    )
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Process multiple videos (input should be directory)'
    )

    args = parser.parse_args()

    try:
        # Load config from file if provided
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            # TODO: Parse config dict to PipelineConfig
            logger.warning("Config file parsing not fully implemented, using defaults")

        # Create configuration
        config = create_default_config(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            quality=args.quality,
            codec=args.codec
        )

        # Update additional settings
        config.save_masks_separately = args.save_masks
        if args.save_masks and args.mask_output:
            config.mask_output_path = args.mask_output
        elif args.save_masks:
            # Auto-generate mask output path
            output_path = Path(args.output)
            config.mask_output_path = str(
                output_path.parent / f"{output_path.stem}_masks{output_path.suffix}"
            )

        config.visualization_mode = args.visualization_mode

        # Process video(s)
        if args.batch_mode:
            logger.info("Running in batch mode")
            input_dir = Path(args.input)
            output_dir = Path(args.output)

            # Find all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            input_videos = [
                str(p) for p in input_dir.iterdir()
                if p.suffix.lower() in video_extensions
            ]

            logger.info(f"Found {len(input_videos)} videos to process")

            # Create batch pipeline
            batch_pipeline = BatchVideoPipeline(
                input_videos=input_videos,
                output_dir=str(output_dir),
                base_config=config
            )

            # Process all videos
            def batch_progress(video_idx, total_videos, stats):
                logger.info(
                    f"Video {video_idx+1}/{total_videos} | "
                    f"Progress: {stats['progress_percent']:.1f}% | "
                    f"FPS: {stats['fps']:.2f}"
                )

            results = batch_pipeline.process_all(progress_callback=batch_progress)

            # Print summary
            logger.info("=" * 60)
            logger.info("BATCH PROCESSING SUMMARY")
            logger.info("=" * 60)
            for i, result in enumerate(results):
                logger.info(f"Video {i+1}: {result.get('status', 'unknown')}")
                if result.get('status') == 'completed':
                    logger.info(f"  Frames: {result.get('processed_frames', 0)}")
                    logger.info(f"  FPS: {result.get('avg_fps', 0):.2f}")
                    logger.info(f"  Time: {result.get('total_time', 0):.2f}s")

        else:
            logger.info("Processing single video")

            # Create pipeline
            pipeline = VRVideoSegmentationPipeline(config)

            # Process video
            logger.info(f"Input: {args.input}")
            logger.info(f"Output: {args.output}")
            logger.info(f"Model: {args.model}")
            logger.info(f"Device: {args.device}")

            summary = pipeline.process(progress_callback=progress_callback)

            # Print summary
            logger.info("=" * 60)
            logger.info("PROCESSING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Status: {summary['status']}")
            logger.info(f"Total Frames: {summary['total_frames']}")
            logger.info(f"Processed Frames: {summary['processed_frames']}")
            logger.info(f"Total Time: {summary['elapsed_time']:.2f}s")
            logger.info(f"Average FPS: {summary['avg_fps']:.2f}")
            logger.info(f"Output: {args.output}")

        logger.info("Processing complete!")
        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
