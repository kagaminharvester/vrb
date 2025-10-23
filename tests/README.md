# VR Body Segmentation - Test Suite

Comprehensive testing infrastructure for the VR body segmentation application.

## Quick Start

```bash
# Run all tests
./scripts/run_tests.sh

# Run with coverage
./scripts/run_tests.sh --coverage

# Run unit tests only
./scripts/run_tests.sh -t unit

# Run integration tests
./scripts/run_tests.sh -t integration

# Profile GPU performance
./scripts/profile_gpu.sh
```

## Test Organization

```
tests/
├── README.md                       # This file
├── conftest.py                     # Pytest configuration and fixtures
├── test_utils.py                   # Test utilities and helpers
│
├── unit/                           # Unit tests (80%+ coverage goal)
│   ├── test_video_decoder.py      # Video decoding tests
│   ├── test_preprocessor.py       # Preprocessing tests
│   ├── test_segmentation.py       # Segmentation model tests
│   ├── test_postprocessor.py      # Postprocessing tests
│   ├── test_gpu_utils.py          # GPU utilities tests
│   └── test_memory_manager.py     # Memory management tests
│
└── integration/                    # Integration tests
    ├── test_full_pipeline.py       # End-to-end pipeline tests
    ├── test_vr_processing.py       # VR-specific tests
    └── test_performance.py         # Performance benchmarks
```

## Test Categories

### Unit Tests (tests/unit/)

Test individual components in isolation:

- **Video Decoder**: Frame extraction, format handling, seeking
- **Preprocessor**: Resizing, normalization, color conversion
- **Segmentation**: Model inference, batch processing, optimization
- **Postprocessor**: Mask refinement, smoothing, compositing
- **GPU Utils**: CUDA operations, memory management, streams
- **Memory Manager**: Memory tracking, leak detection, optimization

### Integration Tests (tests/integration/)

Test complete workflows:

- **Full Pipeline**: End-to-end video processing
- **VR Processing**: Stereo handling, eye separation, recombination
- **Performance**: FPS benchmarks, latency tests, throughput

## Test Markers

Tests are marked for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.performance` - Performance benchmarks

Run specific markers:
```bash
# GPU tests only
./scripts/run_tests.sh -m gpu

# Skip slow tests
./scripts/run_tests.sh -m "not slow"

# GPU integration tests
./scripts/run_tests.sh -m "gpu and integration"
```

## Fixtures

Common fixtures available in all tests (from `conftest.py`):

### Session-level Fixtures
- `cuda_available` - Boolean indicating CUDA availability
- `device` - Device string ('cuda' or 'cpu')
- `gpu_info` - GPU properties dictionary
- `test_data_dir` - Path to test data directory

### Function-level Fixtures
- `test_video_path` - Temporary test video file
- `test_vr_video_path` - Temporary VR stereo video
- `test_mask` - Test segmentation mask
- `sample_frame` - Sample video frame (NumPy array)
- `sample_frame_batch` - Batch of sample frames
- `mock_model` - Mock segmentation model
- `performance_monitor` - Performance metrics collector
- `temp_dir` - Temporary directory for test outputs

### Parametrized Fixtures
- `video_resolution` - Different resolutions (640×480, 1080p, 4K)
- `batch_size` - Different batch sizes (1, 2, 4, 8)
- `vr_layout` - VR layouts ('side_by_side', 'top_bottom')
- `fps` - Different frame rates (30, 60, 120)

## Running Tests

### Basic Usage

```bash
# All tests
pytest tests/

# Specific file
pytest tests/unit/test_preprocessor.py

# Specific test
pytest tests/unit/test_preprocessor.py::TestFrameResizing::test_resize_frame

# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

### With Script

```bash
# Unit tests with coverage and HTML report
./scripts/run_tests.sh -t unit --coverage --html

# Integration tests without GPU
./scripts/run_tests.sh -t integration --no-gpu

# Performance tests with verbose output
./scripts/run_tests.sh -t performance -v

# Custom output directory
./scripts/run_tests.sh --coverage -o my_results
```

## Performance Testing

### Benchmarks

Run performance benchmarks:
```bash
./scripts/run_tests.sh -t performance
```

**Measured Metrics:**
- Frames per second (FPS)
- Per-frame latency (ms)
- End-to-end pipeline latency
- GPU memory usage
- CPU memory usage
- GPU utilization
- Throughput over time

### Expected Performance (RTX 3090)

| Test | Batch Size | Expected FPS | Notes |
|------|------------|--------------|-------|
| 1080p | 1 | 60+ | Single frame |
| 1080p | 4 | 120+ | Optimal batch |
| 1080p | 8 | 200+ | Max throughput |
| 4K | 1 | 30+ | High resolution |
| 4K | 4 | 60+ | Batch processing |
| VR Stereo | 4 | 60+ | Both eyes |

## GPU Profiling

### Run Profiler

```bash
# All profiling tools
./scripts/profile_gpu.sh

# PyTorch profiler only
./scripts/profile_gpu.sh -t pytorch

# Nsight Systems
./scripts/profile_gpu.sh -t nsight

# Custom duration and batch size
./scripts/profile_gpu.sh -d 30 -b 8 -o my_profile
```

### Profiling Tools

1. **PyTorch Profiler**
   - Output: `profiling_results/pytorch_trace.json`
   - View: Chrome at `chrome://tracing`

2. **NVIDIA Nsight Systems**
   - Output: `profiling_results/nsight_report.qdrep`
   - View: `nsys-ui profiling_results/nsight_report.qdrep`

3. **NVIDIA Nsight Compute**
   - Output: `profiling_results/ncu_report.ncu-rep`
   - View: `ncu-ui profiling_results/ncu_report.ncu-rep`

4. **Memory Profiler**
   - Output: `profiling_results/memory_profile.json`

5. **GPU Utilization**
   - Output: `profiling_results/gpu_utilization.log`

## Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=src --cov-report=term tests/

# HTML report
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html

# With script
./scripts/run_tests.sh --coverage
open test_results/coverage/index.html
```

### Coverage Goals

- Overall: **80%+**
- Core modules: **90%+**
- Utilities: **70%+**
- Integration: Functional coverage

## Writing Tests

### Example Test

```python
import pytest
import torch

class TestMyFeature:
    """Test suite for my feature."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_data)
        assert result is not None
        assert result.shape == expected_shape

    @pytest.mark.gpu
    def test_gpu_processing(self, cuda_available, device):
        """Test GPU-specific functionality."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(10, 10, device=device)
        result = process_on_gpu(tensor)
        assert result.device.type == device

    @pytest.mark.parametrize("size", [640, 1280, 1920])
    def test_different_sizes(self, size):
        """Test with multiple input sizes."""
        result = resize_image(size, size)
        assert result.shape[0] == size
        assert result.shape[1] == size

    @pytest.mark.slow
    def test_long_running(self):
        """Test with large dataset (marked as slow)."""
        # Long-running test...
        pass
```

### Best Practices

1. **Clear naming**: Descriptive test names explaining what is tested
2. **Single assertion**: Each test should verify one specific behavior
3. **Use fixtures**: Leverage pytest fixtures for setup/teardown
4. **Parametrize**: Test multiple inputs with `@pytest.mark.parametrize`
5. **Mark appropriately**: Use markers for GPU, slow, integration tests
6. **Clean up**: Ensure resources are properly released
7. **Document**: Add docstrings explaining test purpose

## Test Utilities

### TestDataGenerator

Create synthetic test data:

```python
from test_utils import TestDataGenerator

# Create test video
TestDataGenerator.create_test_video(
    "test.mp4",
    width=1920,
    height=1080,
    fps=30,
    duration_seconds=2.0
)

# Create VR stereo video
TestDataGenerator.create_vr_stereo_video(
    "vr_test.mp4",
    width=3840,
    height=1080,
    layout='side_by_side'
)

# Create test mask
mask = TestDataGenerator.create_test_mask(
    width=1920,
    height=1080,
    mask_type='ellipse'
)
```

### PerformanceMonitor

Track performance metrics:

```python
from test_utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Record metrics
monitor.record_fps(fps)
monitor.record_latency(latency_ms)
monitor.record_gpu_memory(memory_mb)

# Get statistics
stats = monitor.get_stats()
print(stats['fps']['mean'])

# Save to file
monitor.save_to_json('metrics.json')
```

### Assertions

Helper assertions:

```python
from test_utils import assert_tensor_properties, assert_video_properties

# Assert tensor properties
assert_tensor_properties(
    tensor,
    expected_shape=(1, 3, 640, 640),
    expected_dtype=torch.float32,
    expected_device='cuda',
    value_range=(0.0, 1.0)
)

# Assert video properties
assert_video_properties(
    video_path,
    expected_width=1920,
    expected_height=1080,
    expected_fps=30.0,
    min_frames=30
)
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          ./scripts/run_tests.sh --no-gpu --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh './scripts/run_tests.sh --coverage'
            }
        }
    }
    post {
        always {
            junit 'test_results/junit.xml'
        }
    }
}
```

## Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size or skip GPU tests
./scripts/run_tests.sh --no-gpu
```

**Tests are slow:**
```bash
# Skip slow tests
./scripts/run_tests.sh -m "not slow"

# Run unit tests only
./scripts/run_tests.sh -t unit
```

**Import errors:**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-html opencv-python torch
```

**Coverage not generated:**
```bash
# Install pytest-cov
pip install pytest-cov

# Verify path
./scripts/run_tests.sh --coverage -v
```

## Documentation

- **[Testing Guide](../docs/testing_guide.md)**: Comprehensive testing documentation
- **[Profiling Methodology](../docs/profiling_results.md)**: GPU profiling guide

## Dependencies

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or manually:
pip install pytest>=7.0.0 \
            pytest-cov>=3.0.0 \
            pytest-html>=3.1.0 \
            opencv-python>=4.5.0 \
            torch>=2.0.0 \
            numpy>=1.20.0 \
            psutil>=5.8.0 \
            py3nvml>=0.2.7
```

## Contributing

When adding new features:

1. Write unit tests for new functions/classes
2. Add integration tests for new workflows
3. Update performance benchmarks if needed
4. Ensure coverage goals are met
5. Document any new fixtures or utilities
6. Update this README if needed

## Support

For issues or questions:
- Check test output for detailed error messages
- Run with `-v` flag for verbose output
- Review `test_results/` directory for reports
- Consult the [Testing Guide](../docs/testing_guide.md)

---

**Version:** 1.0
**Last Updated:** October 2025
**Maintainer:** VR Body Segmentation Team
