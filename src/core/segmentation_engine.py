"""
Segmentation Engine

Core body segmentation inference engine with batch processing and error handling.
Integrates model loading, preprocessing, inference, and postprocessing.
"""

import logging
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque

from models.model_loader import ModelConfig, ModelLoader, ModelManager, BaseSegmentationModel
from preprocessing.frame_preprocessor import PreprocessConfig, FramePreprocessor, StereoPreprocessor
from postprocessing.mask_processor import MaskProcessorConfig, MaskProcessor

logger = logging.getLogger(__name__)


@dataclass
class SegmentationEngineConfig:
    """Configuration for segmentation engine."""
    model_config: ModelConfig
    preprocess_config: PreprocessConfig
    postprocess_config: MaskProcessorConfig
    batch_size: int = 1
    enable_timing: bool = True
    cache_size: int = 10
    use_half_precision: bool = True
    device: str = 'cuda'


@dataclass
class SegmentationResult:
    """Result from segmentation inference."""
    mask: np.ndarray
    confidence: Optional[np.ndarray] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SegmentationEngine:
    """
    Core segmentation engine that handles the complete inference pipeline.
    Integrates model loading, preprocessing, inference, and postprocessing.
    """

    def __init__(self, config: SegmentationEngineConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.model: Optional[BaseSegmentationModel] = None
        self.preprocessor: Optional[FramePreprocessor] = None
        self.postprocessor: Optional[MaskProcessor] = None

        # Performance tracking
        self.total_frames = 0
        self.total_time = 0.0
        self.inference_times = deque(maxlen=100)

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing segmentation engine")

        # Load model
        logger.info("Loading segmentation model")
        self.model = ModelLoader.load_model(self.config.model_config)

        # Initialize preprocessor
        logger.info("Initializing preprocessor")
        self.preprocessor = FramePreprocessor(self.config.preprocess_config)

        # Initialize postprocessor
        logger.info("Initializing postprocessor")
        self.postprocessor = MaskProcessor(self.config.postprocess_config)

        logger.info("Segmentation engine initialized successfully")

    def segment_frame(
        self,
        frame: np.ndarray,
        apply_postprocess: bool = True,
        return_confidence: bool = False
    ) -> SegmentationResult:
        """
        Segment a single frame.

        Args:
            frame: Input frame (HxWxC numpy array)
            apply_postprocess: Whether to apply postprocessing
            return_confidence: Whether to return confidence scores

        Returns:
            SegmentationResult object
        """
        start_time = time.time()

        try:
            # Preprocess
            preprocessed, preprocess_info = self.preprocessor.preprocess(
                frame, return_info=True
            )

            # Inference
            with torch.no_grad():
                prediction = self.model.predict(preprocessed)

            # Convert to numpy
            if isinstance(prediction, torch.Tensor):
                mask = prediction.cpu().numpy()
            else:
                mask = prediction

            # Remove batch dimension if present
            if mask.ndim == 4:
                mask = mask[0]

            # Handle multi-class output (take foreground class)
            if mask.ndim == 3 and mask.shape[0] > 1:
                # Assume class 1 is foreground (person)
                confidence = mask[1] if return_confidence else None
                mask = mask[1]
            else:
                confidence = mask.copy() if return_confidence else None
                if mask.ndim == 3:
                    mask = mask[0]

            # Resize to original size
            original_shape = preprocess_info['original_shape']
            mask = self.preprocessor.postprocess(
                torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
                original_shape
            )

            # Convert to uint8
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

            # Postprocess
            if apply_postprocess:
                mask = self.postprocessor.process(mask)

            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_frames += 1
            self.total_time += processing_time
            self.inference_times.append(processing_time)

            result = SegmentationResult(
                mask=mask,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    'frame_shape': frame.shape,
                    'mask_shape': mask.shape,
                    'avg_fps': 1.0 / (sum(self.inference_times) / len(self.inference_times))
                }
            )

            return result

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise

    def segment_batch(
        self,
        frames: List[np.ndarray],
        apply_postprocess: bool = True
    ) -> List[SegmentationResult]:
        """
        Segment a batch of frames.

        Args:
            frames: List of input frames
            apply_postprocess: Whether to apply postprocessing

        Returns:
            List of SegmentationResult objects
        """
        start_time = time.time()

        try:
            # Preprocess batch
            preprocessed_batch = self.preprocessor.preprocess_batch(frames)

            # Inference
            with torch.no_grad():
                predictions = self.model.predict(preprocessed_batch)

            # Convert to numpy
            if isinstance(predictions, torch.Tensor):
                masks = predictions.cpu().numpy()
            else:
                masks = predictions

            # Process each mask
            results = []
            for i, (frame, mask) in enumerate(zip(frames, masks)):
                # Handle multi-class output
                if mask.ndim == 3 and mask.shape[0] > 1:
                    mask = mask[1]  # Foreground class
                elif mask.ndim == 3:
                    mask = mask[0]

                # Resize to original size
                mask_resized = self.preprocessor.postprocess(
                    torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
                    frame.shape
                )

                # Convert to uint8
                if mask_resized.max() <= 1.0:
                    mask_resized = (mask_resized * 255).astype(np.uint8)
                else:
                    mask_resized = mask_resized.astype(np.uint8)

                # Postprocess
                if apply_postprocess:
                    mask_resized = self.postprocessor.process(mask_resized, apply_temporal=False)

                results.append(SegmentationResult(
                    mask=mask_resized,
                    processing_time=0.0,  # Will be updated below
                ))

            # Update timing
            batch_time = time.time() - start_time
            time_per_frame = batch_time / len(frames)
            for result in results:
                result.processing_time = time_per_frame
                self.inference_times.append(time_per_frame)

            self.total_frames += len(frames)
            self.total_time += batch_time

            return results

        except Exception as e:
            logger.error(f"Batch segmentation failed: {e}")
            raise

    def segment_stereo_frame(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        apply_postprocess: bool = True,
        enforce_consistency: bool = True
    ) -> Tuple[SegmentationResult, SegmentationResult]:
        """
        Segment stereo frame pair.

        Args:
            left_frame: Left eye frame
            right_frame: Right eye frame
            apply_postprocess: Whether to apply postprocessing
            enforce_consistency: Whether to enforce stereo consistency

        Returns:
            (left_result, right_result) tuple
        """
        # Segment both frames
        left_result = self.segment_frame(left_frame, apply_postprocess=False)
        right_result = self.segment_frame(right_frame, apply_postprocess=False)

        # Enforce stereo consistency
        if enforce_consistency:
            from vr.stereo_processor import StereoProcessor, StereoConsistencyConfig

            stereo_config = StereoConsistencyConfig(
                enable=True,
                method='weighted',
                temporal_weight=0.3
            )
            stereo_processor = StereoProcessor(stereo_config)

            left_mask, right_mask = stereo_processor.process_stereo_pair(
                left_result.mask,
                right_result.mask,
                apply_temporal=True
            )

            left_result.mask = left_mask
            right_result.mask = right_mask

        # Apply postprocessing
        if apply_postprocess:
            left_result.mask = self.postprocessor.process(left_result.mask)
            right_result.mask = self.postprocessor.process(right_result.mask)

        return left_result, right_result

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary of performance metrics
        """
        avg_time = self.total_time / self.total_frames if self.total_frames > 0 else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        recent_times = list(self.inference_times)
        recent_avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
        recent_fps = 1.0 / recent_avg_time if recent_avg_time > 0 else 0

        return {
            'total_frames': self.total_frames,
            'total_time': self.total_time,
            'avg_time_per_frame': avg_time,
            'avg_fps': avg_fps,
            'recent_avg_time': recent_avg_time,
            'recent_fps': recent_fps,
            'device': str(self.device),
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_frames = 0
        self.total_time = 0.0
        self.inference_times.clear()
        logger.info("Performance statistics reset")

    def warmup(self, iterations: int = 10) -> None:
        """
        Warm up the engine with dummy frames.

        Args:
            iterations: Number of warmup iterations
        """
        logger.info(f"Warming up segmentation engine ({iterations} iterations)")

        h, w = self.config.preprocess_config.target_size
        dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        for i in range(iterations):
            self.segment_frame(dummy_frame, apply_postprocess=False)
            if (i + 1) % 5 == 0:
                logger.info(f"Warmup iteration {i+1}/{iterations}")

        # Reset stats after warmup
        self.reset_stats()
        logger.info("Warmup complete")


class BatchSegmentationEngine:
    """
    Optimized engine for batch processing.
    Handles large batches efficiently with GPU memory management.
    """

    def __init__(
        self,
        config: SegmentationEngineConfig,
        max_batch_size: int = 8
    ):
        self.config = config
        self.max_batch_size = max_batch_size
        self.engine = SegmentationEngine(config)

        logger.info(f"Batch segmentation engine initialized (max_batch_size={max_batch_size})")

    def segment_frames(
        self,
        frames: List[np.ndarray],
        apply_postprocess: bool = True,
        show_progress: bool = True
    ) -> List[SegmentationResult]:
        """
        Segment list of frames in batches.

        Args:
            frames: List of input frames
            apply_postprocess: Whether to apply postprocessing
            show_progress: Whether to show progress

        Returns:
            List of SegmentationResult objects
        """
        all_results = []
        total_batches = (len(frames) + self.max_batch_size - 1) // self.max_batch_size

        for i in range(0, len(frames), self.max_batch_size):
            batch = frames[i:i + self.max_batch_size]
            batch_num = i // self.max_batch_size + 1

            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches} "
                           f"({len(batch)} frames)")

            results = self.engine.segment_batch(batch, apply_postprocess)
            all_results.extend(results)

        return all_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.engine.get_performance_stats()


class StreamingSegmentationEngine:
    """
    Engine optimized for streaming/real-time processing.
    Minimizes latency and handles frame drops gracefully.
    """

    def __init__(
        self,
        config: SegmentationEngineConfig,
        target_fps: float = 30.0,
        allow_frame_skip: bool = True
    ):
        self.config = config
        self.target_fps = target_fps
        self.allow_frame_skip = allow_frame_skip
        self.target_frame_time = 1.0 / target_fps

        self.engine = SegmentationEngine(config)
        self.frames_processed = 0
        self.frames_skipped = 0

        logger.info(f"Streaming segmentation engine initialized (target_fps={target_fps})")

    def segment_frame(
        self,
        frame: np.ndarray,
        current_time: float,
        last_process_time: float
    ) -> Optional[SegmentationResult]:
        """
        Segment frame with timing constraints.

        Args:
            frame: Input frame
            current_time: Current timestamp
            last_process_time: Last processing timestamp

        Returns:
            SegmentationResult or None if frame skipped
        """
        # Check if we should skip this frame
        if self.allow_frame_skip:
            time_since_last = current_time - last_process_time
            if time_since_last < self.target_frame_time:
                self.frames_skipped += 1
                return None

        # Process frame
        result = self.engine.segment_frame(frame)
        self.frames_processed += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        stats = self.engine.get_performance_stats()
        stats.update({
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'skip_rate': self.frames_skipped / (self.frames_processed + self.frames_skipped)
                        if (self.frames_processed + self.frames_skipped) > 0 else 0
        })
        return stats
