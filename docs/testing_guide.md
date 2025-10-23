# VR Body Segmentation - Testing Guide

## Overview

This document provides comprehensive guidance for testing the VR body segmentation application. The test suite includes unit tests, integration tests, and performance benchmarks designed to ensure quality, reliability, and optimal performance.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Performance Testing](#performance-testing)
5. [GPU Profiling](#gpu-profiling)
6. [CI/CD Integration](#cicd-integration)
7. [Writing Tests](#writing-tests)
8. [Troubleshooting](#troubleshooting)

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_utils.py            # Test utilities and helpers
├── unit/                    # Unit tests
│   ├── test_video_decoder.py
│   ├── test_preprocessor.py
│   ├── test_segmentation.py
│   ├── test_postprocessor.py
│   ├── test_gpu_utils.py
│   └── test_memory_manager.py
└── integration/             # Integration tests
    ├── test_full_pipeline.py
    ├── test_vr_processing.py
    └── test_performance.py
```

## Running Tests

### Quick Start

Run all tests:
```bash
./scripts/run_tests.sh
```

Run with coverage:
```bash
./scripts/run_tests.sh --coverage
```

### Test Categories

#### Unit Tests Only
```bash
./scripts/run_tests.sh -t unit
```

Tests individual components in isolation:
- Video decoder functionality
- Preprocessing operations
- Segmentation model interface
- Postprocessing operations
- GPU utilities
- Memory management

#### Integration Tests Only
```bash
./scripts/run_tests.sh -t integration
```

Tests complete workflows:
- Full video processing pipeline
- VR stereo processing
- End-to-end functionality

#### Performance Tests
```bash
./scripts/run_tests.sh -t performance
```

Benchmarks:
- FPS at different resolutions
- Latency measurements
- Throughput testing
- Memory usage profiling

### Advanced Options

#### Skip GPU Tests
```bash
./scripts/run_tests.sh --no-gpu
```

Useful for CPU-only environments or when GPU is unavailable.

#### Include Slow Tests
```bash
./scripts/run_tests.sh --slow
```

Runs comprehensive tests including:
- Long video processing (30+ minutes)
- High-resolution 4K/8K tests
- Stress testing

#### Generate HTML Report
```bash
./scripts/run_tests.sh --html -o my_results
```

Creates a detailed HTML report with test results and visualizations.

#### Custom Markers
```bash
./scripts/run_tests.sh -m "gpu and not slow"
```

Run tests matching specific pytest markers.

## Test Categories

### Unit Tests

#### Video Decoder (`test_video_decoder.py`)
- Video format handling (MP4, AVI, etc.)
- Frame extraction and seeking
- Resolution and FPS handling
- Error handling for corrupted files
- Hardware acceleration support

**Example:**
```python
def test_video_decoder_initialization(test_video_path):
    """Test video decoder can be initialized with valid video."""
    cap = cv2.VideoCapture(test_video_path)
    assert cap.isOpened()
```

#### Preprocessor (`test_preprocessor.py`)
- Frame resizing with different interpolations
- Normalization (mean/std, [0,1])
- Color space conversions (BGR, RGB, HSV)
- Tensor conversions (NumPy ↔ PyTorch)
- Batch processing

**Key Tests:**
- `test_resize_maintains_aspect_ratio`: Ensures proper aspect ratio handling
- `test_normalize_with_mean_std`: Validates ImageNet normalization
- `test_batch_preprocessing`: Tests batch efficiency

#### Segmentation (`test_segmentation.py`)
- Model loading and initialization
- Inference on single frames and batches
- Different input resolutions
- FP16/FP32 precision modes
- Memory management
- TensorRT optimization (when available)

**Coverage:**
- Model inference correctness
- Batch size handling
- GPU memory usage
- Error handling for invalid inputs

#### Postprocessor (`test_postprocessor.py`)
- Mask refinement (morphological operations)
- Smoothing (Gaussian, median, bilateral)
- Edge detection and refinement
- Temporal smoothing across frames
- Alpha blending and compositing

**Key Operations Tested:**
- Morphological opening/closing
- Contour detection and filling
- Mask resize and interpolation
- Multi-frame temporal filtering

#### GPU Utilities (`test_gpu_utils.py`)
- CUDA availability detection
- Device management
- Memory allocation and tracking
- Data transfer (CPU ↔ GPU)
- Multi-GPU support
- CUDA streams and events
- Mixed precision (AMP)

**Performance Tests:**
- Memory allocation overhead
- Transfer bandwidth
- Stream concurrency
- Synchronization costs

#### Memory Manager (`test_memory_manager.py`)
- Memory tracking (CPU and GPU)
- Memory pool management
- Leak detection
- Cache management
- Memory optimization strategies

**Validation:**
- No memory leaks in loops
- Proper cleanup after operations
- Memory growth within limits
- Cache eviction policies

### Integration Tests

#### Full Pipeline (`test_full_pipeline.py`)
- End-to-end video processing
- Different video formats and resolutions
- Batch processing efficiency
- Memory cleanup
- Error recovery

**Scenarios Tested:**
- Single frame pipeline
- Full video processing
- Batch processing with different sizes
- High-resolution (4K) videos
- Error handling and recovery

#### VR Processing (`test_vr_processing.py`)
- VR format detection (side-by-side, top-bottom)
- Eye separation and processing
- Stereo consistency
- VR-specific optimizations

**VR-Specific Tests:**
- Side-by-side frame splitting
- Top-bottom frame splitting
- Parallel eye processing
- Stereo recombination
- Quality consistency between eyes

#### Performance (`test_performance.py`)
- FPS benchmarks
- Latency measurements
- Throughput testing
- Memory profiling
- GPU utilization
- Scalability tests

**Metrics Collected:**
- Frames per second (FPS)
- Per-frame latency (ms)
- Peak memory usage
- GPU utilization (%)
- CPU utilization (%)

## Performance Testing

### Running Performance Benchmarks

```bash
# Run all performance tests
./scripts/run_tests.sh -t performance -v

# Run specific benchmark
pytest tests/integration/test_performance.py::TestFPSBenchmarks -v

# Profile with different batch sizes
pytest tests/integration/test_performance.py::TestThroughputBenchmarks -v
```

### Benchmark Suites

#### FPS Benchmarks
- Single frame processing
- Batch processing (1, 2, 4, 8, 16, 32)
- Different resolutions (640p, 720p, 1080p, 4K)

#### Latency Benchmarks
- Inference latency (model only)
- End-to-end latency (full pipeline)
- Percentile analysis (P50, P95, P99)

#### Throughput Benchmarks
- Sustained throughput over time
- Peak throughput with optimal batch size
- Long-running stability tests

#### Memory Benchmarks
- Peak GPU memory usage
- CPU memory growth over time
- Memory leak detection

### Expected Performance

**Target Hardware: NVIDIA RTX 3090 (24GB VRAM)**

| Resolution | Batch Size | Expected FPS | Latency (ms) |
|-----------|------------|--------------|--------------|
| 1080p     | 1          | 60+          | <17          |
| 1080p     | 4          | 120+         | <8           |
| 1080p     | 8          | 200+         | <5           |
| 4K        | 1          | 30+          | <33          |
| 4K        | 4          | 60+          | <17          |

*Note: Actual performance depends on model complexity and implementation.*

## GPU Profiling

### Running GPU Profiler

```bash
# Run all profiling tools
./scripts/profile_gpu.sh

# PyTorch profiler only
./scripts/profile_gpu.sh -t pytorch

# Nsight Systems profiler
./scripts/profile_gpu.sh -t nsight

# Custom duration and batch size
./scripts/profile_gpu.sh -d 30 -b 8
```

### Profiling Tools

#### PyTorch Profiler
Analyzes PyTorch operations:
- CUDA kernel execution times
- CPU operation times
- Memory allocations
- Data transfer overhead

**Output:** `profiling_results/pytorch_trace.json`

**View:** Open in Chrome at `chrome://tracing`

#### NVIDIA Nsight Systems
System-level profiling:
- GPU timeline visualization
- CPU/GPU synchronization
- CUDA API calls
- Memory transfers

**Output:** `profiling_results/nsight_report.qdrep`

**View:** `nsys-ui profiling_results/nsight_report.qdrep`

#### NVIDIA Nsight Compute
Kernel-level profiling:
- Kernel performance metrics
- Memory bandwidth utilization
- Occupancy analysis
- Optimization suggestions

**Output:** `profiling_results/ncu_report.ncu-rep`

**View:** `ncu-ui profiling_results/ncu_report.ncu-rep`

#### Memory Profiler
Tracks memory usage over time:
- Allocated memory
- Reserved memory
- Peak usage
- Memory timeline

**Output:** `profiling_results/memory_profile.json`

#### GPU Utilization Monitor
Real-time GPU metrics:
- GPU utilization (%)
- Memory utilization (%)
- Temperature
- Power consumption

**Output:** `profiling_results/gpu_utilization.log`

### Analyzing Results

#### Identifying Bottlenecks

1. **Low GPU Utilization (<80%)**:
   - Increase batch size
   - Check for CPU bottlenecks
   - Optimize data loading

2. **High Memory Usage (>90%)**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision (FP16)

3. **Low Throughput**:
   - Profile kernel execution times
   - Check for synchronization overhead
   - Optimize memory transfers

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        ./scripts/run_tests.sh --no-gpu --coverage

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./test_results/coverage.xml
```

### Jenkins Example

```groovy
pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh './scripts/run_tests.sh --coverage --html'
            }
        }

        stage('Performance') {
            when {
                branch 'main'
            }
            steps {
                sh './scripts/run_tests.sh -t performance'
            }
        }
    }

    post {
        always {
            junit 'test_results/junit.xml'
            publishHTML([
                reportDir: 'test_results',
                reportFiles: 'report.html',
                reportName: 'Test Report'
            ])
        }
    }
}
```

## Writing Tests

### Test Structure

```python
import pytest
import torch
import numpy as np

class TestMyFeature:
    """Test suite for my feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = my_function()
        assert result is not None

    @pytest.mark.gpu
    def test_gpu_functionality(self, cuda_available):
        """Test GPU-specific functionality."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(10, 10, device='cuda')
        result = process_on_gpu(tensor)
        assert result.device.type == 'cuda'

    @pytest.mark.parametrize("size", [640, 1280, 1920])
    def test_different_sizes(self, size):
        """Test with different input sizes."""
        result = process_image(size, size)
        assert result.shape[0] == size
```

### Best Practices

1. **Use Fixtures**: Leverage pytest fixtures for common setup
2. **Parameterize Tests**: Test multiple inputs with `@pytest.mark.parametrize`
3. **Mark Appropriately**: Use markers (`@pytest.mark.gpu`, `@pytest.mark.slow`)
4. **Clean Up**: Ensure resources are released after tests
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Document Tests**: Add clear docstrings explaining what is tested

### Available Fixtures

From `conftest.py`:

- `test_video_path`: Temporary test video
- `test_vr_video_path`: Temporary VR stereo video
- `test_mask`: Test segmentation mask
- `sample_frame`: Sample video frame
- `mock_model`: Mock segmentation model
- `device`: Device string ('cuda' or 'cpu')
- `performance_monitor`: Performance metrics collector
- `temp_dir`: Temporary directory for test files

## Troubleshooting

### Common Issues

#### Tests Fail with "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size in tests
export PYTEST_BATCH_SIZE=2

# Skip GPU tests
./scripts/run_tests.sh --no-gpu
```

#### Tests are Very Slow

**Solution:**
```bash
# Skip slow tests
./scripts/run_tests.sh -m "not slow"

# Run only unit tests
./scripts/run_tests.sh -t unit
```

#### Import Errors

**Solution:**
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or install individually
pip install pytest pytest-cov pytest-html opencv-python torch
```

#### Coverage Not Generated

**Solution:**
```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Verify source path
./scripts/run_tests.sh --coverage -v
```

### Getting Help

- Check test output for detailed error messages
- Run with `-v` flag for verbose output
- Check `test_results/` directory for reports
- Review test logs in `test_results/junit.xml`

## Test Coverage Goals

### Current Coverage Targets

- **Overall**: 80%+
- **Core modules**: 90%+
- **Utilities**: 70%+
- **Integration**: Functional coverage

### Measuring Coverage

```bash
# Generate coverage report
./scripts/run_tests.sh --coverage

# View HTML report
open test_results/coverage/index.html

# View terminal summary
pytest --cov=src --cov-report=term
```

## Continuous Improvement

### Adding New Tests

When adding new features:

1. Write unit tests for new functions/classes
2. Add integration tests for new workflows
3. Update performance benchmarks if applicable
4. Document any new test fixtures or utilities
5. Update this guide with new test categories

### Performance Regression Testing

Track performance over time:

```bash
# Baseline measurement
./scripts/run_tests.sh -t performance > baseline.txt

# After changes
./scripts/run_tests.sh -t performance > current.txt

# Compare
diff baseline.txt current.txt
```

## Appendix

### Test Dependencies

```
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-html>=3.1.0
opencv-python>=4.5.0
torch>=2.0.0
numpy>=1.20.0
psutil>=5.8.0
py3nvml>=0.2.7  # For GPU monitoring
```

### Useful Commands

```bash
# List all tests
pytest --collect-only

# Run specific test file
pytest tests/unit/test_preprocessor.py

# Run specific test
pytest tests/unit/test_preprocessor.py::TestFrameResizing::test_resize_frame

# Run with specific markers
pytest -m "gpu and not slow"

# Run in parallel (with pytest-xdist)
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Increase verbosity
pytest -vv
```

---

**Last Updated:** October 2025
**Version:** 1.0
**Maintainer:** VR Body Segmentation Team
