"""
Main Video Processing Pipeline

Orchestrates the complete VR video body segmentation pipeline.
Integrates video decoding, segmentation, postprocessing, and encoding.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from threading import Thread, Event
from queue import Queue
import numpy as np

from preprocessing.video_decoder import VRVideoDecoder, VideoDecoderConfig
from core.segmentation_engine import SegmentationEngine, SegmentationEngineConfig
from postprocessing.video_encoder import VideoEncoder, VideoEncoderConfig, MaskVideoEncoder
from vr.stereo_processor import StereoProcessor, StereoConsistencyConfig
from utils.video_utils import VRFormat, get_video_info, validate_video_file

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for video processing pipeline."""
    # Input/Output
    input_video_path: str
    output_video_path: str

    # Component configs
    decoder_config: VideoDecoderConfig
    segmentation_config: SegmentationEngineConfig
    encoder_config: VideoEncoderConfig
    stereo_config: Optional[StereoConsistencyConfig] = None

    # Pipeline settings
    enable_stereo_processing: bool = True
    enable_progress_callback: bool = True
    save_masks_separately: bool = False
    mask_output_path: Optional[str] = None

    # Performance
    num_worker_threads: int = 2
    frame_queue_size: int = 30

    # Visualization
    create_visualization: bool = True
    visualization_mode: str = 'overlay'  # 'overlay', 'side_by_side', 'mask_only'


class ProgressTracker:
    """Tracks and reports pipeline progress."""

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.processed_frames = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def update(self, frames_processed: int = 1) -> Dict[str, Any]:
        """
        Update progress.

        Args:
            frames_processed: Number of frames processed in this update

        Returns:
            Progress statistics dictionary
        """
        self.processed_frames += frames_processed
        current_time = time.time()

        elapsed = current_time - self.start_time
        progress = self.processed_frames / self.total_frames if self.total_frames > 0 else 0
        fps = self.processed_frames / elapsed if elapsed > 0 else 0

        remaining_frames = self.total_frames - self.processed_frames
        eta = remaining_frames / fps if fps > 0 else 0

        self.last_update_time = current_time

        return {
            'processed_frames': self.processed_frames,
            'total_frames': self.total_frames,
            'progress': progress,
            'progress_percent': progress * 100,
            'elapsed_time': elapsed,
            'fps': fps,
            'eta': eta,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get final summary statistics."""
        total_time = time.time() - self.start_time
        avg_fps = self.processed_frames / total_time if total_time > 0 else 0

        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
        }


class VRVideoSegmentationPipeline:
    """
    Main pipeline for VR video body segmentation.
    Handles complete processing from input video to segmented output.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Components
        self.decoder: Optional[VRVideoDecoder] = None
        self.segmentation_engine: Optional[SegmentationEngine] = None
        self.encoder: Optional[VideoEncoder] = None
        self.mask_encoder: Optional[MaskVideoEncoder] = None
        self.stereo_processor: Optional[StereoProcessor] = None

        # Progress tracking
        self.progress_tracker: Optional[ProgressTracker] = None
        self.progress_callback: Optional[Callable] = None

        # State
        self.is_initialized = False
        self.is_running = False
        self.should_stop = Event()

        logger.info("VR video segmentation pipeline created")

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self.is_initialized:
            logger.warning("Pipeline already initialized")
            return

        logger.info("Initializing pipeline components")

        # Validate input video
        if not validate_video_file(self.config.input_video_path):
            raise ValueError(f"Invalid input video: {self.config.input_video_path}")

        # Get video info
        video_info = get_video_info(self.config.input_video_path)
        logger.info(f"Input video: {video_info['width']}x{video_info['height']} "
                   f"@ {video_info['fps']:.2f}fps, {video_info['frame_count']} frames")

        # Initialize decoder
        logger.info("Initializing video decoder")
        self.decoder = VRVideoDecoder(self.config.decoder_config)

        # Initialize segmentation engine
        logger.info("Initializing segmentation engine")
        self.segmentation_engine = SegmentationEngine(self.config.segmentation_config)

        # Warm up segmentation engine
        logger.info("Warming up segmentation engine")
        self.segmentation_engine.warmup(iterations=10)

        # Initialize encoder
        logger.info("Initializing video encoder")
        vr_format = VRFormat(video_info['vr_format'])

        if self.config.enable_stereo_processing and vr_format != VRFormat.MONO:
            # For stereo, output size matches input
            frame_size = (video_info['width'], video_info['height'])
        else:
            # For mono, use single frame size
            frame_size = (video_info['width'], video_info['height'])

        self.config.encoder_config.fps = video_info['fps']
        self.config.encoder_config.vr_format = vr_format

        self.encoder = VideoEncoder(self.config.encoder_config, frame_size)

        # Initialize mask encoder if requested
        if self.config.save_masks_separately and self.config.mask_output_path:
            logger.info("Initializing mask encoder")
            mask_encoder_config = VideoEncoderConfig(
                output_path=self.config.mask_output_path,
                fps=video_info['fps'],
                codec='h264',
                quality='high',
                vr_format=vr_format
            )
            self.mask_encoder = MaskVideoEncoder(
                mask_encoder_config,
                frame_size,
                visualization_mode='binary'
            )

        # Initialize stereo processor if needed
        if self.config.enable_stereo_processing and self.config.stereo_config:
            logger.info("Initializing stereo processor")
            self.stereo_processor = StereoProcessor(self.config.stereo_config)

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(video_info['frame_count'])

        self.is_initialized = True
        logger.info("Pipeline initialization complete")

    def process(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete processing pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Processing summary statistics
        """
        if not self.is_initialized:
            self.initialize()

        self.progress_callback = progress_callback
        self.is_running = True
        self.should_stop.clear()

        logger.info("Starting video processing pipeline")
        start_time = time.time()

        try:
            # Start decoder
            if self.config.decoder_config.use_threading:
                self.decoder.start_threaded_decode()

            # Process frames
            if self.config.decoder_config.decode_mode == 'stereo':
                self._process_stereo()
            else:
                self._process_mono()

            # Finalize
            self._finalize()

            # Get statistics
            elapsed_time = time.time() - start_time
            summary = self.progress_tracker.get_summary()
            summary['elapsed_time'] = elapsed_time
            summary['status'] = 'completed'

            logger.info(f"Pipeline completed: {summary['processed_frames']} frames "
                       f"in {elapsed_time:.2f}s ({summary['avg_fps']:.2f} fps)")

            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._cleanup()
            raise

        finally:
            self.is_running = False

    def _process_stereo(self) -> None:
        """Process stereo video frames."""
        logger.info("Processing stereo video")

        frame_count = 0
        for frame_data in self.decoder.frames():
            if self.should_stop.is_set():
                logger.info("Processing stopped by user")
                break

            # Extract left and right frames
            left_frame = frame_data['left_eye']
            right_frame = frame_data['right_eye']

            # Segment both eyes
            left_result, right_result = self.segmentation_engine.segment_stereo_frame(
                left_frame,
                right_frame,
                apply_postprocess=True,
                enforce_consistency=self.config.enable_stereo_processing
            )

            # Create output frames
            if self.config.create_visualization:
                # Apply masks to original frames
                left_output = self._create_visualization(left_frame, left_result.mask)
                right_output = self._create_visualization(right_frame, right_result.mask)
            else:
                left_output = left_frame
                right_output = right_frame

            # Write to encoder
            self.encoder.write_stereo_frame(left_output, right_output)

            # Write masks if requested
            if self.mask_encoder:
                self.mask_encoder.write_stereo_frame(
                    left_result.mask,
                    right_result.mask
                )

            # Update progress
            frame_count += 1
            if frame_count % 10 == 0:
                self._report_progress()

        logger.info(f"Processed {frame_count} stereo frames")

    def _process_mono(self) -> None:
        """Process mono video frames."""
        logger.info("Processing mono video")

        frame_count = 0
        for frame_data in self.decoder.frames():
            if self.should_stop.is_set():
                logger.info("Processing stopped by user")
                break

            # Extract frame
            frame = frame_data['frame']

            # Segment frame
            result = self.segmentation_engine.segment_frame(
                frame,
                apply_postprocess=True
            )

            # Create output frame
            if self.config.create_visualization:
                output_frame = self._create_visualization(frame, result.mask)
            else:
                output_frame = frame

            # Write to encoder
            self.encoder.write_frame(output_frame)

            # Write mask if requested
            if self.mask_encoder:
                from postprocessing.mask_processor import MaskVisualizer
                mask_frame = result.mask
                if len(mask_frame.shape) == 2:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
                self.mask_encoder.encoder.write_frame(mask_frame)

            # Update progress
            frame_count += 1
            if frame_count % 10 == 0:
                self._report_progress()

        logger.info(f"Processed {frame_count} mono frames")

    def _create_visualization(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Create visualization based on mode."""
        import cv2
        from postprocessing.mask_processor import MaskVisualizer

        mode = self.config.visualization_mode

        if mode == 'overlay':
            return MaskVisualizer.overlay_mask(frame, mask, alpha=0.5)
        elif mode == 'side_by_side':
            return MaskVisualizer.create_side_by_side(frame, mask)
        elif mode == 'mask_only':
            if len(mask.shape) == 2:
                return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return mask
        else:
            return frame

    def _report_progress(self) -> None:
        """Report progress to callback."""
        if self.progress_callback and self.progress_tracker:
            stats = self.progress_tracker.update()

            # Add segmentation engine stats
            seg_stats = self.segmentation_engine.get_performance_stats()
            stats.update({
                'segmentation_fps': seg_stats['recent_fps'],
                'device': seg_stats['device']
            })

            self.progress_callback(stats)

    def _finalize(self) -> None:
        """Finalize all components."""
        logger.info("Finalizing pipeline")

        if self.encoder:
            self.encoder.finalize()

        if self.mask_encoder:
            self.mask_encoder.finalize()

        if self.decoder:
            self.decoder.stop()

    def _cleanup(self) -> None:
        """Cleanup resources on error."""
        logger.info("Cleaning up pipeline resources")

        try:
            if self.encoder:
                self.encoder.finalize()
        except:
            pass

        try:
            if self.mask_encoder:
                self.mask_encoder.finalize()
        except:
            pass

        try:
            if self.decoder:
                self.decoder.stop()
        except:
            pass

    def stop(self) -> None:
        """Stop processing."""
        logger.info("Stopping pipeline")
        self.should_stop.set()

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_running:
            self.stop()
        self._cleanup()


class BatchVideoPipeline:
    """
    Pipeline for processing multiple videos in batch.
    Handles multiple input videos with shared configuration.
    """

    def __init__(
        self,
        input_videos: List[str],
        output_dir: str,
        base_config: PipelineConfig
    ):
        self.input_videos = input_videos
        self.output_dir = Path(output_dir)
        self.base_config = base_config

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch pipeline initialized: {len(input_videos)} videos")

    def process_all(
        self,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all videos.

        Args:
            progress_callback: Optional callback(video_index, total_videos, stats)

        Returns:
            List of processing summaries
        """
        results = []

        for i, input_path in enumerate(self.input_videos):
            logger.info(f"Processing video {i+1}/{len(self.input_videos)}: {input_path}")

            # Create output path
            input_name = Path(input_path).stem
            output_path = str(self.output_dir / f"{input_name}_segmented.mp4")
            mask_path = str(self.output_dir / f"{input_name}_masks.mp4")

            # Update config
            config = self.base_config
            config.input_video_path = input_path
            config.output_video_path = output_path
            config.mask_output_path = mask_path

            # Process video
            try:
                pipeline = VRVideoSegmentationPipeline(config)

                # Create per-video progress callback
                def video_progress(stats):
                    if progress_callback:
                        progress_callback(i, len(self.input_videos), stats)

                summary = pipeline.process(progress_callback=video_progress)
                summary['input_path'] = input_path
                summary['output_path'] = output_path
                results.append(summary)

                logger.info(f"Video {i+1} completed successfully")

            except Exception as e:
                logger.error(f"Failed to process video {i+1}: {e}")
                results.append({
                    'input_path': input_path,
                    'status': 'failed',
                    'error': str(e)
                })

        logger.info(f"Batch processing complete: {len(results)} videos processed")
        return results


# Import cv2 for visualization
import cv2
