# Testing Quick Start Guide

## Installation

```bash
# Install test dependencies
pip install -r requirements-test.txt
```

## Run Tests

```bash
# All tests
./scripts/run_tests.sh

# Unit tests only
./scripts/run_tests.sh -t unit

# Integration tests
./scripts/run_tests.sh -t integration

# Performance benchmarks
./scripts/run_tests.sh -t performance

# With coverage
./scripts/run_tests.sh --coverage

# Skip GPU tests
./scripts/run_tests.sh --no-gpu

# Skip slow tests
./scripts/run_tests.sh -m "not slow"
```

## GPU Profiling

```bash
# Run all profilers
./scripts/profile_gpu.sh

# PyTorch profiler only
./scripts/profile_gpu.sh -t pytorch

# Nsight profiler
./scripts/profile_gpu.sh -t nsight

# Custom settings
./scripts/profile_gpu.sh -d 30 -b 8
```

## Direct pytest Commands

```bash
# Run specific test file
pytest tests/unit/test_preprocessor.py -v

# Run specific test
pytest tests/unit/test_preprocessor.py::TestFrameResizing::test_resize_frame -v

# Run with markers
pytest -m "gpu and not slow" -v

# Show coverage
pytest --cov=src --cov-report=term tests/

# Generate HTML report
pytest --html=report.html --self-contained-html
```

## View Results

```bash
# Coverage report
open test_results/coverage/index.html

# Test report
open test_results/report.html

# PyTorch profiling trace
# Open chrome://tracing in Chrome browser
# Load: profiling_results/pytorch_trace.json

# Nsight Systems report
nsys-ui profiling_results/nsight_report.qdrep

# Nsight Compute report
ncu-ui profiling_results/ncu_report.ncu-rep
```

## Quick Test Examples

### Test a Specific Component

```bash
# Test video decoder
pytest tests/unit/test_video_decoder.py -v

# Test GPU utilities
pytest tests/unit/test_gpu_utils.py -v -m gpu

# Test VR processing
pytest tests/integration/test_vr_processing.py -v
```

### Performance Testing

```bash
# FPS benchmarks
pytest tests/integration/test_performance.py::TestFPSBenchmarks -v

# Latency tests
pytest tests/integration/test_performance.py::TestLatencyBenchmarks -v

# Memory benchmarks
pytest tests/integration/test_performance.py::TestMemoryBenchmarks -v
```

### Debug Failed Tests

```bash
# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Verbose with local variables
pytest tests/ -vv -l

# Enter debugger on failure
pytest tests/ --pdb
```

## Common Issues

### CUDA Out of Memory
```bash
# Skip GPU tests
./scripts/run_tests.sh --no-gpu
```

### Tests Too Slow
```bash
# Skip slow tests
./scripts/run_tests.sh -m "not slow"

# Run in parallel
pytest tests/ -n auto
```

### Import Errors
```bash
# Check Python path
export PYTHONPATH=/home/pi/vr-body-segmentation:$PYTHONPATH

# Reinstall dependencies
pip install -r requirements-test.txt
```

## Expected Performance (RTX 3090)

| Test | Batch | FPS | Latency |
|------|-------|-----|---------|
| 1080p | 1 | 60+ | <17ms |
| 1080p | 4 | 120+ | <8ms |
| 4K | 1 | 30+ | <33ms |
| VR Stereo | 4 | 60+ | <17ms |

## Test Coverage Goals

- Overall: 80%+
- Core modules: 90%+
- Utilities: 70%+

## Documentation

- **Full Guide**: [docs/testing_guide.md](docs/testing_guide.md)
- **Profiling**: [docs/profiling_results.md](docs/profiling_results.md)
- **Test README**: [tests/README.md](tests/README.md)

## Support

For issues:
1. Check test output messages
2. Run with `-v` for verbose output
3. Review test logs in `test_results/`
4. Consult documentation

---

Happy Testing! 🧪
