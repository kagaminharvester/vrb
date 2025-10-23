"""
Unit tests for preprocessing module.

Tests frame preprocessing including resizing, normalization,
and format conversion for model input.
"""

import pytest
import numpy as np
import torch
import cv2
from unittest.mock import Mock, patch


class TestFrameResizing:
    """Test frame resizing functionality."""

    @pytest.mark.parametrize("input_size,target_size", [
        ((1920, 1080), (640, 640)),
        ((3840, 2160), (640, 640)),
        ((1280, 720), (512, 512)),
        ((640, 480), (640, 640)),
    ])
    def test_resize_frame(self, sample_frame, input_size, target_size):
        """Test resizing frames to target dimensions."""
        # Create frame with specific size
        frame = cv2.resize(sample_frame, input_size)

        # Resize to target
        resized = cv2.resize(frame, target_size)

        assert resized.shape[:2] == target_size[::-1]  # OpenCV uses (height, width)

    def test_resize_maintains_aspect_ratio(self, sample_frame):
        """Test aspect ratio preserving resize."""
        height, width = sample_frame.shape[:2]
        target_size = 640

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        resized = cv2.resize(sample_frame, (new_width, new_height))

        assert resized.shape[0] == new_height
        assert resized.shape[1] == new_width

    def test_resize_with_padding(self, sample_frame):
        """Test resize with padding to maintain aspect ratio."""
        target_size = 640
        height, width = sample_frame.shape[:2]

        # Resize maintaining aspect ratio
        scale = target_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        resized = cv2.resize(sample_frame, (new_width, new_height))

        # Add padding
        top = (target_size - new_height) // 2
        bottom = target_size - new_height - top
        left = (target_size - new_width) // 2
        right = target_size - new_width - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        assert padded.shape == (target_size, target_size, 3)

    @pytest.mark.parametrize("interpolation", [
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ])
    def test_resize_interpolation_methods(self, sample_frame, interpolation):
        """Test different interpolation methods."""
        resized = cv2.resize(sample_frame, (640, 640), interpolation=interpolation)
        assert resized.shape == (640, 640, 3)


class TestNormalization:
    """Test frame normalization."""

    def test_normalize_to_unit_range(self, sample_frame):
        """Test normalizing pixel values to [0, 1]."""
        normalized = sample_frame.astype(np.float32) / 255.0

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32

    def test_normalize_with_mean_std(self, sample_frame):
        """Test normalization with mean and std."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Normalize to [0, 1]
        frame_float = sample_frame.astype(np.float32) / 255.0

        # Apply mean/std normalization
        normalized = (frame_float - mean) / std

        assert normalized.dtype == np.float32
        # Values can be outside [0, 1] after standardization

    def test_denormalize_frame(self):
        """Test denormalizing frame back to [0, 255]."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Create normalized frame
        normalized = np.random.randn(480, 640, 3).astype(np.float32)

        # Denormalize
        denormalized = (normalized * std + mean) * 255.0
        denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)

        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255

    def test_batch_normalization(self, sample_frame_batch):
        """Test normalizing batch of frames."""
        batch_float = sample_frame_batch.astype(np.float32) / 255.0

        assert batch_float.shape == sample_frame_batch.shape
        assert batch_float.dtype == np.float32
        assert 0 <= batch_float.min() <= batch_float.max() <= 1


class TestColorConversion:
    """Test color space conversions."""

    def test_bgr_to_rgb(self, sample_frame):
        """Test BGR to RGB conversion."""
        rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)

        assert rgb.shape == sample_frame.shape
        # Channels should be swapped
        np.testing.assert_array_equal(rgb[:, :, 0], sample_frame[:, :, 2])
        np.testing.assert_array_equal(rgb[:, :, 2], sample_frame[:, :, 0])

    def test_rgb_to_grayscale(self, sample_frame):
        """Test RGB to grayscale conversion."""
        rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        assert gray.shape == sample_frame.shape[:2]
        assert gray.dtype == np.uint8

    def test_hsv_conversion(self, sample_frame):
        """Test RGB to HSV conversion."""
        hsv = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2HSV)

        assert hsv.shape == sample_frame.shape
        assert hsv[:, :, 0].max() <= 180  # Hue range in OpenCV


class TestTensorConversion:
    """Test NumPy to PyTorch tensor conversion."""

    def test_numpy_to_tensor(self, sample_frame, device):
        """Test converting numpy array to tensor."""
        tensor = torch.from_numpy(sample_frame)
        tensor = tensor.to(device)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == sample_frame.shape
        assert tensor.device.type == device

    def test_channel_first_conversion(self, sample_frame, device):
        """Test converting HWC to CHW format."""
        # OpenCV format: HWC
        assert sample_frame.shape == (1080, 1920, 3)

        # Convert to CHW
        tensor = torch.from_numpy(sample_frame).permute(2, 0, 1)
        tensor = tensor.to(device)

        assert tensor.shape == (3, 1080, 1920)

    def test_batch_tensor_conversion(self, sample_frame_batch, device):
        """Test converting batch of frames to tensor."""
        # BHWC to BCHW
        tensor = torch.from_numpy(sample_frame_batch).permute(0, 3, 1, 2)
        tensor = tensor.to(device)

        assert tensor.shape == (4, 3, 1080, 1920)

    def test_normalize_tensor(self, sample_frame, device):
        """Test normalizing tensor."""
        tensor = torch.from_numpy(sample_frame).permute(2, 0, 1).float()
        tensor = tensor.to(device)

        # Normalize to [0, 1]
        normalized = tensor / 255.0

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_standardize_tensor(self, sample_frame, device):
        """Test standardizing tensor with mean/std."""
        tensor = torch.from_numpy(sample_frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

        standardized = (tensor - mean) / std

        assert standardized.shape == tensor.shape


class TestBatchPreprocessing:
    """Test batch preprocessing operations."""

    def test_preprocess_batch(self, sample_frame_batch, device):
        """Test preprocessing a batch of frames."""
        # Resize all frames
        target_size = (640, 640)
        resized_batch = np.array([
            cv2.resize(frame, target_size) for frame in sample_frame_batch
        ])

        # Convert to tensor
        tensor = torch.from_numpy(resized_batch).permute(0, 3, 1, 2).float()
        tensor = tensor.to(device)

        # Normalize
        normalized = tensor / 255.0

        assert normalized.shape == (4, 3, 640, 640)
        assert normalized.device.type == device

    def test_variable_size_batch_handling(self, temp_dir):
        """Test handling frames of different sizes in a batch."""
        from test_utils import TestDataGenerator

        # Create frames of different sizes
        sizes = [(640, 480), (1280, 720), (1920, 1080)]
        frames = []

        for size in sizes:
            frame = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
            frames.append(frame)

        # Resize all to same size
        target_size = (640, 640)
        resized_frames = [cv2.resize(f, target_size) for f in frames]

        # Now can stack
        batch = np.stack(resized_frames)
        assert batch.shape == (3, 640, 640, 3)


class TestDataAugmentation:
    """Test data augmentation techniques."""

    def test_random_flip(self, sample_frame):
        """Test random horizontal flip."""
        flipped = cv2.flip(sample_frame, 1)  # Horizontal flip

        assert flipped.shape == sample_frame.shape
        np.testing.assert_array_equal(flipped[:, ::-1, :], sample_frame)

    def test_random_brightness(self, sample_frame):
        """Test random brightness adjustment."""
        factor = 1.2
        brightened = np.clip(sample_frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        assert brightened.shape == sample_frame.shape
        assert brightened.dtype == np.uint8

    def test_random_crop(self, sample_frame):
        """Test random cropping."""
        height, width = sample_frame.shape[:2]
        crop_size = (512, 512)

        y = np.random.randint(0, height - crop_size[0])
        x = np.random.randint(0, width - crop_size[1])

        cropped = sample_frame[y:y+crop_size[0], x:x+crop_size[1]]

        assert cropped.shape == (*crop_size, 3)

    def test_random_rotation(self, sample_frame):
        """Test random rotation."""
        angle = 15
        height, width = sample_frame.shape[:2]
        center = (width // 2, height // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(sample_frame, matrix, (width, height))

        assert rotated.shape == sample_frame.shape


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self, sample_frame, device):
        """Test complete preprocessing pipeline."""
        # 1. Resize
        resized = cv2.resize(sample_frame, (640, 640))

        # 2. Color conversion (BGR to RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Convert to tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = tensor.to(device)

        # 4. Normalize
        normalized = tensor / 255.0

        # 5. Standardize
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        standardized = (normalized - mean) / std

        assert standardized.shape == (3, 640, 640)
        assert standardized.device.type == device

    def test_pipeline_preserves_batch_dim(self, sample_frame_batch, device):
        """Test pipeline preserves batch dimension."""
        batch_size = len(sample_frame_batch)

        # Process batch
        resized = np.array([cv2.resize(f, (640, 640)) for f in sample_frame_batch])
        rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized])
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.to(device)

        assert tensor.shape[0] == batch_size


class TestMemoryEfficiency:
    """Test memory efficiency of preprocessing."""

    def test_in_place_normalization(self, sample_frame):
        """Test in-place operations for memory efficiency."""
        # Convert to float32
        frame_float = sample_frame.astype(np.float32)

        # In-place division
        frame_float /= 255.0

        assert 0 <= frame_float.min() <= frame_float.max() <= 1

    @pytest.mark.gpu
    def test_gpu_preprocessing(self, sample_frame, device):
        """Test preprocessing on GPU."""
        if device != 'cuda':
            pytest.skip("GPU not available")

        # Upload to GPU
        tensor = torch.from_numpy(sample_frame).to(device)

        # Preprocess on GPU
        tensor = tensor.permute(2, 0, 1).float() / 255.0

        assert tensor.device.type == 'cuda'


class TestEdgeCases:
    """Test edge cases in preprocessing."""

    def test_empty_frame_handling(self):
        """Test handling of empty frames."""
        empty = np.array([])

        with pytest.raises((ValueError, cv2.error)):
            cv2.resize(empty, (640, 640))

    def test_single_pixel_frame(self):
        """Test handling of 1x1 frame."""
        single_pixel = np.array([[[255, 128, 64]]], dtype=np.uint8)
        resized = cv2.resize(single_pixel, (640, 640))

        assert resized.shape == (640, 640, 3)

    def test_very_large_frame(self):
        """Test handling of very large frames."""
        # 8K resolution
        large_frame = np.random.randint(0, 255, (4320, 7680, 3), dtype=np.uint8)
        resized = cv2.resize(large_frame, (640, 640))

        assert resized.shape == (640, 640, 3)

    def test_grayscale_frame_preprocessing(self):
        """Test preprocessing grayscale frames."""
        gray = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)

        # Convert to 3-channel
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        assert rgb.shape == (1080, 1920, 3)
