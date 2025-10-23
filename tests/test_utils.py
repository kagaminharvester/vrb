"""
Test utilities and helper functions for VR body segmentation testing.

This module provides common utilities for creating test data, fixtures,
and helper functions used across unit and integration tests.
"""

import os
import tempfile
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import json


class TestDataGenerator:
    """Generate synthetic test data for VR video processing."""

    @staticmethod
    def create_test_video(
        output_path: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        duration_seconds: float = 1.0,
        codec: str = 'mp4v',
        has_body: bool = True
    ) -> str:
        """
        Create a synthetic test video with optional body silhouette.

        Args:
            output_path: Path to save the video
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            duration_seconds: Duration of video
            codec: Video codec to use
            has_body: Whether to render a body silhouette

        Returns:
            Path to the created video file
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        num_frames = int(fps * duration_seconds)

        for i in range(num_frames):
            # Create a gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
            frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)

            if has_body:
                # Draw a simple body silhouette (ellipse)
                center_x = int(width / 2 + 100 * np.sin(i * 0.1))
                center_y = int(height / 2)
                cv2.ellipse(frame, (center_x, center_y),
                           (int(width * 0.15), int(height * 0.4)),
                           0, 0, 360, (255, 255, 255), -1)

            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def create_vr_stereo_video(
        output_path: str,
        width: int = 3840,
        height: int = 1080,
        fps: int = 30,
        duration_seconds: float = 1.0,
        layout: str = 'side_by_side'
    ) -> str:
        """
        Create a synthetic VR stereo video (side-by-side or top-bottom).

        Args:
            output_path: Path to save the video
            width: Total video width
            height: Total video height
            fps: Frames per second
            duration_seconds: Duration of video
            layout: 'side_by_side' or 'top_bottom'

        Returns:
            Path to the created video file
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        num_frames = int(fps * duration_seconds)

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if layout == 'side_by_side':
                # Left eye
                left_half = width // 2
                frame[:, :left_half, 0] = 100
                cv2.circle(frame, (left_half // 2, height // 2),
                          min(left_half, height) // 4, (255, 0, 0), -1)

                # Right eye
                frame[:, left_half:, 2] = 100
                cv2.circle(frame, (left_half + left_half // 2, height // 2),
                          min(left_half, height) // 4, (0, 0, 255), -1)
            else:  # top_bottom
                top_half = height // 2
                frame[:top_half, :, 0] = 100
                cv2.circle(frame, (width // 2, top_half // 2),
                          min(width, top_half) // 4, (255, 0, 0), -1)

                frame[top_half:, :, 2] = 100
                cv2.circle(frame, (width // 2, top_half + top_half // 2),
                          min(width, top_half) // 4, (0, 0, 255), -1)

            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def create_test_mask(
        width: int = 1920,
        height: int = 1080,
        mask_type: str = 'ellipse'
    ) -> np.ndarray:
        """
        Create a test segmentation mask.

        Args:
            width: Mask width
            height: Mask height
            mask_type: Type of mask ('ellipse', 'rectangle', 'polygon')

        Returns:
            Binary mask as numpy array
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if mask_type == 'ellipse':
            cv2.ellipse(mask, (width // 2, height // 2),
                       (width // 4, height // 3), 0, 0, 360, 255, -1)
        elif mask_type == 'rectangle':
            cv2.rectangle(mask, (width // 4, height // 4),
                         (3 * width // 4, 3 * height // 4), 255, -1)
        elif mask_type == 'polygon':
            pts = np.array([
                [width // 2, height // 4],
                [3 * width // 4, height // 2],
                [width // 2, 3 * height // 4],
                [width // 4, height // 2]
            ])
            cv2.fillPoly(mask, [pts], 255)

        return mask

    @staticmethod
    def create_batch_tensors(
        batch_size: int,
        channels: int = 3,
        height: int = 1080,
        width: int = 1920,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Create a batch of random tensors for testing.

        Args:
            batch_size: Number of samples in batch
            channels: Number of channels
            height: Tensor height
            width: Tensor width
            device: Device to create tensor on

        Returns:
            Batch tensor
        """
        return torch.rand(batch_size, channels, height, width, device=device)


class MockGPUContext:
    """Mock GPU context for testing without actual GPU."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.memory_allocated = 0
        self.memory_reserved = 0
        self.max_memory = 24 * 1024 * 1024 * 1024  # 24GB

    def allocate(self, size: int):
        """Simulate memory allocation."""
        if self.memory_allocated + size > self.max_memory:
            raise RuntimeError("Out of memory")
        self.memory_allocated += size

    def free(self, size: int):
        """Simulate memory deallocation."""
        self.memory_allocated = max(0, self.memory_allocated - size)

    def reset(self):
        """Reset memory counters."""
        self.memory_allocated = 0
        self.memory_reserved = 0


class PerformanceMonitor:
    """Monitor performance metrics during tests."""

    def __init__(self):
        self.metrics = {
            'fps': [],
            'latency_ms': [],
            'gpu_memory_mb': [],
            'cpu_memory_mb': [],
            'gpu_utilization': [],
            'cpu_utilization': []
        }

    def record_fps(self, fps: float):
        """Record FPS measurement."""
        self.metrics['fps'].append(fps)

    def record_latency(self, latency_ms: float):
        """Record latency measurement."""
        self.metrics['latency_ms'].append(latency_ms)

    def record_gpu_memory(self, memory_mb: float):
        """Record GPU memory usage."""
        self.metrics['gpu_memory_mb'].append(memory_mb)

    def record_cpu_memory(self, memory_mb: float):
        """Record CPU memory usage."""
        self.metrics['cpu_memory_mb'].append(memory_mb)

    def get_stats(self) -> dict:
        """Get statistical summary of metrics."""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        return stats

    def save_to_json(self, output_path: str):
        """Save metrics to JSON file."""
        stats = self.get_stats()
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)


class TempVideoFile:
    """Context manager for temporary video files."""

    def __init__(self, suffix: str = '.mp4', **video_params):
        self.suffix = suffix
        self.video_params = video_params
        self.file_path = None

    def __enter__(self):
        fd, self.file_path = tempfile.mkstemp(suffix=self.suffix)
        os.close(fd)
        TestDataGenerator.create_test_video(self.file_path, **self.video_params)
        return self.file_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path and os.path.exists(self.file_path):
            os.unlink(self.file_path)


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_device: Optional[str] = None,
    value_range: Optional[Tuple[float, float]] = None
):
    """
    Assert various properties of a tensor.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        expected_dtype: Expected data type
        expected_device: Expected device ('cuda' or 'cpu')
        value_range: Expected (min, max) value range
    """
    if expected_shape is not None:
        assert tensor.shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, \
            f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"

    if expected_device is not None:
        assert tensor.device.type == expected_device, \
            f"Device mismatch: expected {expected_device}, got {tensor.device.type}"

    if value_range is not None:
        min_val, max_val = value_range
        actual_min = tensor.min().item()
        actual_max = tensor.max().item()
        assert min_val <= actual_min, \
            f"Min value {actual_min} below expected {min_val}"
        assert actual_max <= max_val, \
            f"Max value {actual_max} above expected {max_val}"


def assert_video_properties(
    video_path: str,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_fps: Optional[float] = None,
    min_frames: Optional[int] = None
):
    """
    Assert properties of a video file.

    Args:
        video_path: Path to video file
        expected_width: Expected width in pixels
        expected_height: Expected height in pixels
        expected_fps: Expected frames per second
        min_frames: Minimum number of frames
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    cap = cv2.VideoCapture(video_path)

    if expected_width is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        assert width == expected_width, \
            f"Width mismatch: expected {expected_width}, got {width}"

    if expected_height is not None:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert height == expected_height, \
            f"Height mismatch: expected {expected_height}, got {height}"

    if expected_fps is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert abs(fps - expected_fps) < 1.0, \
            f"FPS mismatch: expected {expected_fps}, got {fps}"

    if min_frames is not None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= min_frames, \
            f"Too few frames: expected at least {min_frames}, got {frame_count}"

    cap.release()


def create_test_config(
    batch_size: int = 4,
    model_name: str = 'yolov8n-seg',
    input_size: Tuple[int, int] = (640, 640),
    device: str = 'cuda:0'
) -> dict:
    """
    Create a test configuration dictionary.

    Args:
        batch_size: Batch size for processing
        model_name: Name of segmentation model
        input_size: Input size for model
        device: Device to use

    Returns:
        Configuration dictionary
    """
    return {
        'batch_size': batch_size,
        'model': {
            'name': model_name,
            'input_size': input_size,
            'device': device
        },
        'preprocessing': {
            'resize_method': 'bilinear',
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'postprocessing': {
            'confidence_threshold': 0.5,
            'smoothing': True,
            'morphology': {
                'enabled': True,
                'kernel_size': 5
            }
        },
        'vr': {
            'layout': 'side_by_side',
            'process_both_eyes': True
        },
        'gpu': {
            'memory_fraction': 0.9,
            'allow_growth': True
        }
    }


def get_test_data_dir() -> Path:
    """Get the test data directory path."""
    test_dir = Path(__file__).parent
    data_dir = test_dir / 'test_data'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def cleanup_test_data():
    """Clean up test data directory."""
    data_dir = get_test_data_dir()
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
        data_dir.mkdir()
