"""
Frame Preprocessor

Handles frame preprocessing for segmentation model input.
Includes resizing, normalization, color conversion, and augmentation.
"""

import logging
import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from utils.video_utils import (
    resize_frame, convert_color_space, normalize_frame, denormalize_frame
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for frame preprocessing."""
    target_size: Tuple[int, int] = (1080, 1920)  # (height, width)
    maintain_aspect: bool = True
    input_color_space: str = 'BGR'
    model_color_space: str = 'RGB'
    normalize: bool = True
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    to_tensor: bool = True
    device: str = 'cuda'
    dtype: str = 'float32'  # 'float32' or 'float16'


class FramePreprocessor:
    """
    Preprocessor for frames before segmentation inference.
    Handles resizing, normalization, and conversion to tensor format.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        logger.info(f"Frame preprocessor initialized: "
                   f"target_size={config.target_size}, "
                   f"normalize={config.normalize}, "
                   f"device={self.device}")

    def preprocess(
        self,
        frame: np.ndarray,
        return_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Preprocess a single frame.

        Args:
            frame: Input frame (HxWxC numpy array)
            return_info: Whether to return preprocessing info

        Returns:
            Preprocessed tensor or (tensor, info_dict)
        """
        original_shape = frame.shape
        info = {'original_shape': original_shape}

        # Resize frame
        if self.config.target_size:
            frame = resize_frame(
                frame,
                self.config.target_size,
                maintain_aspect=self.config.maintain_aspect,
                interpolation=cv2.INTER_LINEAR
            )
            info['resized_shape'] = frame.shape

        # Convert color space
        if self.config.input_color_space != self.config.model_color_space:
            frame = convert_color_space(
                frame,
                self.config.input_color_space,
                self.config.model_color_space
            )

        # Normalize
        if self.config.normalize:
            frame = normalize_frame(
                frame,
                self.config.normalization_mean,
                self.config.normalization_std
            )
        else:
            # Just scale to [0, 1]
            frame = frame.astype(np.float32) / 255.0

        # Convert to tensor
        if self.config.to_tensor:
            # Convert HWC to CHW
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.from_numpy(frame).to(self.device)

            # Convert dtype
            if self.config.dtype == 'float16':
                frame = frame.half()
            else:
                frame = frame.float()

            # Add batch dimension
            frame = frame.unsqueeze(0)

            info['tensor_shape'] = frame.shape

        if return_info:
            return frame, info
        return frame

    def preprocess_batch(
        self,
        frames: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            Batched tensor (BxCxHxW)
        """
        processed_frames = []

        for frame in frames:
            processed = self.preprocess(frame, return_info=False)
            # Remove batch dimension added by preprocess
            if processed.dim() == 4:
                processed = processed.squeeze(0)
            processed_frames.append(processed)

        # Stack into batch
        batch = torch.stack(processed_frames, dim=0)
        return batch

    def postprocess(
        self,
        tensor: torch.Tensor,
        original_shape: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Postprocess tensor back to frame format.

        Args:
            tensor: Preprocessed tensor
            original_shape: Original frame shape for resizing back

        Returns:
            Postprocessed frame
        """
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            frame = tensor.cpu().numpy()
        else:
            frame = tensor

        # Convert CHW to HWC
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = np.transpose(frame, (1, 2, 0))

        # Denormalize if needed
        if self.config.normalize:
            frame = denormalize_frame(
                frame,
                self.config.normalization_mean,
                self.config.normalization_std
            )
        else:
            # Scale back to [0, 255]
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)

        # Resize back to original shape
        if original_shape is not None:
            h, w = original_shape[:2]
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert color space back
        if self.config.input_color_space != self.config.model_color_space:
            frame = convert_color_space(
                frame,
                self.config.model_color_space,
                self.config.input_color_space
            )

        return frame


class StereoPreprocessor:
    """
    Preprocessor for stereo VR frames.
    Handles left and right eye preprocessing with consistency.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.preprocessor = FramePreprocessor(config)

        logger.info("Stereo preprocessor initialized")

    def preprocess_stereo(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        return_info: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict]
    ]:
        """
        Preprocess stereo pair.

        Args:
            left_eye: Left eye frame
            right_eye: Right eye frame
            return_info: Whether to return preprocessing info

        Returns:
            (left_tensor, right_tensor) or (left_tensor, right_tensor, info_dict)
        """
        # Preprocess both eyes
        if return_info:
            left_processed, left_info = self.preprocessor.preprocess(
                left_eye, return_info=True
            )
            right_processed, right_info = self.preprocessor.preprocess(
                right_eye, return_info=True
            )

            info = {
                'left': left_info,
                'right': right_info
            }
            return left_processed, right_processed, info
        else:
            left_processed = self.preprocessor.preprocess(left_eye)
            right_processed = self.preprocessor.preprocess(right_eye)
            return left_processed, right_processed

    def preprocess_stereo_batch(
        self,
        left_eyes: List[np.ndarray],
        right_eyes: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess batch of stereo pairs.

        Args:
            left_eyes: List of left eye frames
            right_eyes: List of right eye frames

        Returns:
            (left_batch, right_batch) tensors
        """
        left_batch = self.preprocessor.preprocess_batch(left_eyes)
        right_batch = self.preprocessor.preprocess_batch(right_eyes)
        return left_batch, right_batch


class AugmentationPipeline:
    """
    Data augmentation pipeline for training/fine-tuning.
    Includes various augmentations suitable for VR video.
    """

    def __init__(
        self,
        enable_flip: bool = True,
        enable_rotation: bool = True,
        enable_brightness: bool = True,
        enable_contrast: bool = True,
        enable_noise: bool = True,
        max_rotation: float = 10.0,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.01,
    ):
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_brightness = enable_brightness
        self.enable_contrast = enable_contrast
        self.enable_noise = enable_noise
        self.max_rotation = max_rotation
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

        logger.info("Augmentation pipeline initialized")

    def augment(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply augmentations to frame and optional mask.

        Args:
            frame: Input frame
            mask: Optional segmentation mask

        Returns:
            Augmented frame or (frame, mask)
        """
        # Random horizontal flip
        if self.enable_flip and np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        # Random rotation
        if self.enable_rotation and np.random.rand() > 0.5:
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            frame = cv2.warpAffine(frame, M, (w, h))
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h))

        # Random brightness
        if self.enable_brightness and np.random.rand() > 0.5:
            factor = np.random.uniform(*self.brightness_range)
            frame = np.clip(frame * factor, 0, 255).astype(np.uint8)

        # Random contrast
        if self.enable_contrast and np.random.rand() > 0.5:
            factor = np.random.uniform(*self.contrast_range)
            mean = frame.mean()
            frame = np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Random noise
        if self.enable_noise and np.random.rand() > 0.5:
            noise = np.random.normal(0, self.noise_std * 255, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        if mask is not None:
            return frame, mask
        return frame

    def augment_stereo(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        left_mask: Optional[np.ndarray] = None,
        right_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply consistent augmentations to stereo pair.

        Args:
            left_eye: Left eye frame
            right_eye: Right eye frame
            left_mask: Optional left eye mask
            right_mask: Optional right eye mask

        Returns:
            (aug_left, aug_right, aug_left_mask, aug_right_mask)
        """
        # Use same random seed for consistent augmentation
        state = np.random.get_state()

        # Augment left eye
        if left_mask is not None:
            left_eye, left_mask = self.augment(left_eye, left_mask)
        else:
            left_eye = self.augment(left_eye)

        # Reset state for right eye
        np.random.set_state(state)

        # Augment right eye with same parameters
        if right_mask is not None:
            right_eye, right_mask = self.augment(right_eye, right_mask)
        else:
            right_eye = self.augment(right_eye)

        return left_eye, right_eye, left_mask, right_mask


class BatchPreprocessor:
    """
    Efficient batch preprocessing with GPU acceleration.
    Handles large batches for maximum throughput.
    """

    def __init__(
        self,
        config: PreprocessConfig,
        batch_size: int = 8,
        num_workers: int = 4
    ):
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessor = FramePreprocessor(config)

        logger.info(f"Batch preprocessor initialized: batch_size={batch_size}")

    def preprocess_stream(
        self,
        frames_iterator: Iterator[np.ndarray]
    ) -> Iterator[torch.Tensor]:
        """
        Preprocess stream of frames in batches.

        Args:
            frames_iterator: Iterator yielding frames

        Yields:
            Batched tensors
        """
        batch = []

        for frame in frames_iterator:
            batch.append(frame)

            if len(batch) >= self.batch_size:
                # Process batch
                batch_tensor = self.preprocessor.preprocess_batch(batch)
                yield batch_tensor
                batch = []

        # Process remaining frames
        if batch:
            batch_tensor = self.preprocessor.preprocess_batch(batch)
            yield batch_tensor


# Type alias for Union return type
from typing import Union, Iterator
