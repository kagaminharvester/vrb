"""
Pytest configuration and fixtures for VR body segmentation tests.

This module provides shared fixtures, configuration, and hooks for all tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Generator

from test_utils import (
    TestDataGenerator,
    MockGPUContext,
    PerformanceMonitor,
    create_test_config,
    get_test_data_dir,
    cleanup_test_data
)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location."""
    for item in items:
        # Add markers based on file path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add GPU marker if test name contains 'gpu'
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Add slow marker for certain test patterns
        if any(x in item.nodeid.lower() for x in ['performance', 'stress', 'long']):
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide test data directory."""
    return get_test_data_dir()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(cuda_available) -> str:
    """Get the device to use for testing."""
    return 'cuda' if cuda_available else 'cpu'


@pytest.fixture(scope="session")
def gpu_info(cuda_available) -> dict:
    """Get GPU information if available."""
    if not cuda_available:
        return {}

    return {
        'device_count': torch.cuda.device_count(),
        'device_name': torch.cuda.get_device_name(0),
        'memory_total': torch.cuda.get_device_properties(0).total_memory,
        'capability': torch.cuda.get_device_capability(0)
    }


# ============================================================================
# Function-level Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config() -> dict:
    """Provide a default test configuration."""
    return create_test_config()


@pytest.fixture
def test_video_path(temp_dir) -> str:
    """Create a temporary test video."""
    video_path = str(temp_dir / "test_video.mp4")
    TestDataGenerator.create_test_video(
        video_path,
        width=1920,
        height=1080,
        fps=30,
        duration_seconds=1.0
    )
    yield video_path


@pytest.fixture
def test_vr_video_path(temp_dir) -> str:
    """Create a temporary VR stereo test video."""
    video_path = str(temp_dir / "test_vr_video.mp4")
    TestDataGenerator.create_vr_stereo_video(
        video_path,
        width=3840,
        height=1080,
        fps=30,
        duration_seconds=1.0
    )
    yield video_path


@pytest.fixture
def test_mask() -> np.ndarray:
    """Create a test segmentation mask."""
    return TestDataGenerator.create_test_mask(1920, 1080, 'ellipse')


@pytest.fixture
def test_batch_tensor(device) -> torch.Tensor:
    """Create a test batch tensor."""
    return TestDataGenerator.create_batch_tensors(
        batch_size=4,
        channels=3,
        height=1080,
        width=1920,
        device=device
    )


@pytest.fixture
def mock_gpu_context() -> MockGPUContext:
    """Provide a mock GPU context."""
    return MockGPUContext(device_id=0)


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Provide a performance monitor."""
    return PerformanceMonitor()


# ============================================================================
# Video Processing Fixtures
# ============================================================================

@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample video frame."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_frame_batch() -> np.ndarray:
    """Create a batch of sample video frames."""
    return np.random.randint(0, 255, (4, 1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_stereo_frame() -> np.ndarray:
    """Create a sample VR stereo frame."""
    return np.random.randint(0, 255, (1080, 3840, 3), dtype=np.uint8)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Provide a mock segmentation model."""
    class MockSegmentationModel:
        def __init__(self):
            self.device = 'cpu'
            self.input_size = (640, 640)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            batch_size = x.shape[0]
            # Return mock segmentation masks
            return torch.rand(batch_size, 1, 640, 640)

    return MockSegmentationModel()


# ============================================================================
# GPU Memory Fixtures
# ============================================================================

@pytest.fixture
def clear_gpu_memory(cuda_available):
    """Clear GPU memory before and after test."""
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    yield

    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# Parameterized Fixtures
# ============================================================================

@pytest.fixture(params=[
    (1920, 1080),
    (3840, 2160),
    (2560, 1440),
    (1280, 720)
])
def video_resolution(request):
    """Parameterized fixture for different video resolutions."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def batch_size(request):
    """Parameterized fixture for different batch sizes."""
    return request.param


@pytest.fixture(params=['side_by_side', 'top_bottom'])
def vr_layout(request):
    """Parameterized fixture for VR layouts."""
    return request.param


@pytest.fixture(params=[30, 60, 120])
def fps(request):
    """Parameterized fixture for different FPS values."""
    return request.param


# ============================================================================
# Cleanup Hooks
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test."""
    yield
    # Cleanup will happen here after test completes


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after entire test session."""
    # Clean up test data if needed
    pass


# ============================================================================
# Custom Assertions
# ============================================================================

@pytest.fixture
def assert_gpu_memory_released(cuda_available):
    """Assert that GPU memory is released after operation."""
    def _assert_released():
        if cuda_available:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            assert allocated == 0, f"GPU memory not released: {allocated} bytes still allocated"

    return _assert_released


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def benchmark_context(performance_monitor):
    """Provide context for performance benchmarking."""
    import time

    class BenchmarkContext:
        def __init__(self, monitor):
            self.monitor = monitor
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
            self.monitor.record_latency(elapsed)

    return BenchmarkContext(performance_monitor)


# ============================================================================
# Skip Conditions
# ============================================================================

skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

skip_if_insufficient_memory = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3,
    reason="Insufficient GPU memory (requires 8GB+)"
)

skip_if_no_tensorrt = pytest.mark.skipif(
    True,  # Check for TensorRT availability
    reason="TensorRT not available"
)


# ============================================================================
# Test Data Generation Hooks
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def generate_test_data(test_data_dir):
    """Generate test data once per session."""
    # Create sample videos in test data directory
    test_videos = {
        'short_1080p.mp4': {'width': 1920, 'height': 1080, 'duration_seconds': 1.0},
        'short_4k.mp4': {'width': 3840, 'height': 2160, 'duration_seconds': 1.0},
        'vr_sbs.mp4': {'width': 3840, 'height': 1080, 'duration_seconds': 1.0},
    }

    for filename, params in test_videos.items():
        video_path = test_data_dir / filename
        if not video_path.exists():
            TestDataGenerator.create_test_video(str(video_path), **params)

    yield

    # Cleanup is optional - can keep test data for debugging
    # cleanup_test_data()
