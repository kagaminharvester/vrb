# Troubleshooting Guide

## Common Issues and Solutions

### GPU Issues

#### Issue 1: CUDA Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

1. **Reduce Batch Size**:
```yaml
model:
  batch_size: 4  # Reduce from 8
```

2. **Clear CUDA Cache**:
```python
import torch
torch.cuda.empty_cache()
```

3. **Lower Input Resolution**:
```yaml
model:
  input_size: [512, 512]  # Reduce from [1024, 1024]
```

4. **Use Gradient Checkpointing** (if training):
```python
model.gradient_checkpointing_enable()
```

5. **Check for Memory Leaks**:
```bash
# Monitor memory over time
watch -n 1 nvidia-smi

# If memory keeps growing, there's a leak
```

**Prevention**:
```yaml
gpu:
  max_vram_usage: 0.85  # Use conservative limit
```

#### Issue 2: Low GPU Utilization (<50%)

**Symptoms**:
- GPU shows 20-40% utilization
- Low FPS despite available VRAM
- `nvidia-smi` shows underutilization

**Diagnosis**:
```bash
python scripts/optimize.py --detect-bottlenecks
```

**Solutions**:

1. **Increase Batch Size**:
```yaml
model:
  batch_size: 16  # Increase from 8
```

2. **Reduce CPU Preprocessing Overhead**:
```yaml
model:
  num_workers: 8  # Increase workers
  prefetch_factor: 4  # Increase prefetching
```

3. **Check Data Loading**:
```python
# Enable pin_memory for faster transfer
dataloader = DataLoader(dataset, pin_memory=True)
```

4. **Profile to Find Bottleneck**:
```bash
python -m torch.utils.bottleneck cli.py input.mp4 -o output.mp4
```

#### Issue 3: GPU Not Detected

**Symptoms**:
```
torch.cuda.is_available() returns False
```

**Solutions**:

1. **Check NVIDIA Driver**:
```bash
nvidia-smi
# Should show driver version and GPU info
```

2. **Reinstall CUDA-enabled PyTorch**:
```bash
pip uninstall torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify CUDA Installation**:
```bash
nvcc --version
```

4. **Check Environment Variables**:
```bash
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

#### Issue 4: CUDA Illegal Memory Access

**Symptoms**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**Solutions**:

1. **Update GPU Drivers**:
```bash
# Check current version
nvidia-smi

# Update to latest
sudo apt update
sudo apt install nvidia-driver-XXX
```

2. **Reduce Batch Size** (may be corruption):
```yaml
model:
  batch_size: 1  # Test with minimal batch
```

3. **Test GPU Health**:
```bash
# Run CUDA samples
cuda-samples/bin/x86_64/linux/release/deviceQuery
```

4. **Check for Hardware Issues**:
```bash
# Monitor temperature
nvidia-smi -l 1 --query-gpu=temperature.gpu --format=csv
```

### CPU Issues

#### Issue 5: High CPU Usage (>90%)

**Symptoms**:
- CPU at 95-100%
- Slow preprocessing
- System becomes unresponsive

**Diagnosis**:
```bash
htop  # Check which processes using CPU
```

**Solutions**:

1. **Reduce Number of Workers**:
```yaml
model:
  num_workers: 4  # Reduce from 8
processing:
  num_threads: 16  # Reduce from 32
```

2. **Disable Unnecessary Processing**:
```yaml
processing:
  use_multiprocessing: false  # Try sequential
```

3. **Optimize Preprocessing**:
```python
# Use vectorized operations instead of loops
import numpy as np
batch = np.array(frames)  # Vectorized
```

4. **Check for Runaway Processes**:
```bash
ps aux | grep python
kill -9 <PID>  # If needed
```

#### Issue 6: Memory Usage Too High

**Symptoms**:
- System memory at 95%+
- Swap usage increasing
- System slowdown

**Solutions**:

1. **Reduce Queue Sizes**:
```yaml
processing:
  max_queue_size: 50  # Reduce from 100
```

2. **Disable Caching**:
```yaml
processing:
  enable_caching: false
```

3. **Use Disk Cache Instead**:
```yaml
processing:
  cache_size_gb: 1.0  # Reduce memory cache
```

4. **Monitor Memory**:
```bash
watch -n 1 free -h
```

### Performance Issues

#### Issue 7: FPS Below Target

**Symptoms**:
- Achieving 60 FPS but target is 90 FPS
- Stuttering playback
- Inconsistent frame times

**Diagnosis**:
```bash
# Run benchmark
python benchmarks/benchmark_suite.py

# Detect bottlenecks
python scripts/optimize.py --detect-bottlenecks
```

**Solutions**:

1. **Auto-Optimize Configuration**:
```bash
python scripts/optimize.py --auto-optimize --save-config optimized.yaml
python cli.py input.mp4 -c optimized.yaml -o output.mp4
```

2. **Use Fast Profile**:
```bash
python cli.py input.mp4 --profile fast -o output.mp4
```

3. **Enable FP16**:
```yaml
gpu:
  precision: fp16
```

4. **Lower Input Resolution**:
```yaml
model:
  input_size: [512, 512]
```

5. **Skip Frames** (if acceptable):
```yaml
processing:
  frame_skip: 1  # Process every other frame
```

#### Issue 8: High Latency (>50ms per frame)

**Symptoms**:
- Individual frames take >50ms
- Real-time processing not possible
- VR experience degraded

**Solutions**:

1. **Reduce Batch Size** (lower latency):
```yaml
model:
  batch_size: 1  # Minimal latency
```

2. **Use Faster Model**:
```yaml
model:
  name: sam2_hiera_small  # Smaller, faster model
```

3. **Optimize Pipeline**:
```python
# Enable asynchronous processing
torch.backends.cudnn.benchmark = True
```

4. **Check System Latency**:
```bash
# Measure end-to-end latency
python -c "import time; start=time.time(); model(input); print(f'{(time.time()-start)*1000:.2f}ms')"
```

#### Issue 9: Inconsistent Performance

**Symptoms**:
- FPS varies wildly (30-120 FPS)
- Some frames very slow
- Unpredictable behavior

**Solutions**:

1. **Enable Warmup**:
```yaml
performance:
  warmup_iterations: 20  # Warm up before processing
```

2. **Disable Auto-Tuning** (if causing issues):
```yaml
gpu:
  cudnn_benchmark: false
performance:
  auto_tune: false
```

3. **Check Thermal Throttling**:
```bash
# Monitor GPU temperature
nvidia-smi dmon -s pucvt
# If >80°C, improve cooling
```

4. **Fix CPU Frequency Scaling**:
```bash
# Set performance governor
sudo cpupower frequency-set -g performance
```

### Application Issues

#### Issue 10: Application Crashes

**Symptoms**:
- Segmentation fault
- Python crash without error
- Sudden exit

**Diagnosis**:

1. **Run with Debug Logging**:
```bash
python cli.py input.mp4 -o output.mp4 --log-level DEBUG
```

2. **Check Logs**:
```bash
tail -f logs/vr_body_segmentation.log
```

3. **Enable Core Dumps**:
```bash
ulimit -c unlimited
# Run app, check core dump after crash
```

**Solutions**:

1. **Update Dependencies**:
```bash
pip install --upgrade torch torchvision
```

2. **Check Disk Space**:
```bash
df -h
# Ensure sufficient space for output
```

3. **Verify Input Files**:
```bash
ffprobe input.mp4  # Check if file is valid
```

4. **Reduce Complexity**:
```yaml
# Start with minimal config
model:
  batch_size: 1
processing:
  num_threads: 4
```

#### Issue 11: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'X'
```

**Solutions**:

1. **Install Requirements**:
```bash
pip install -r requirements.txt
```

2. **Check Python Path**:
```python
import sys
print(sys.path)
```

3. **Reinstall Package**:
```bash
pip uninstall package_name
pip install package_name
```

4. **Use Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue 12: Configuration Errors

**Symptoms**:
```
ValueError: Invalid configuration
```

**Solutions**:

1. **Validate Configuration**:
```bash
python -c "from src.utils.config_manager import ConfigManager; cm = ConfigManager(); cm.validate_config(cm.load_config())"
```

2. **Use Default Configuration**:
```bash
python cli.py input.mp4 -o output.mp4  # No -c flag
```

3. **Check YAML Syntax**:
```bash
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"
```

4. **Reset to Default**:
```bash
cp configs/default_config.yaml configs/my_config.yaml
```

### Video Processing Issues

#### Issue 13: Output Video Corrupted

**Symptoms**:
- Video won't play
- Artifacts in output
- Missing frames

**Solutions**:

1. **Check Codec Support**:
```yaml
output:
  codec: h264_nvenc  # GPU-accelerated
  # or
  codec: libx264  # CPU fallback
```

2. **Verify Bitrate**:
```yaml
output:
  bitrate: 50M  # Ensure sufficient quality
```

3. **Test with Different Format**:
```yaml
output:
  format: avi  # Try different container
```

4. **Check FFmpeg**:
```bash
ffmpeg -version
ffmpeg -codecs | grep h264
```

#### Issue 14: Color Space Issues

**Symptoms**:
- Colors look wrong
- Washed out appearance
- Incorrect gamma

**Solutions**:

1. **Match Input Color Space**:
```python
# Preserve color space
output.color_space = input.color_space
output.color_range = input.color_range
```

2. **Use Correct Normalization**:
```python
# RGB [0, 255] -> [0, 1]
normalized = image / 255.0
```

3. **Check VR-Specific Requirements**:
```yaml
vr:
  stereoscopic: true
  process_both_eyes: true
```

#### Issue 15: Audio/Video Sync Issues

**Symptoms**:
- Audio and video out of sync
- Drift over time
- Delayed audio

**Solutions**:

1. **Preserve Timestamps**:
```python
output.pts = input.pts  # Preserve presentation timestamps
```

2. **Use Frame Rate from Input**:
```python
output.fps = input.fps  # Don't change frame rate
```

3. **Check Processing FPS**:
```bash
# Ensure processing >= input FPS
# If input is 60 FPS, processing must be 60+ FPS
```

## Diagnostic Tools

### GPU Diagnostics

```bash
# GPU information
nvidia-smi

# Detailed GPU info
nvidia-smi -q

# Monitor in real-time
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# Test CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### CPU Diagnostics

```bash
# CPU info
lscpu

# CPU usage by core
mpstat -P ALL 1

# Process monitoring
htop

# Memory usage
free -h
vmstat 1
```

### Application Diagnostics

```bash
# Run with profiling
python -m cProfile -o profile.stats cli.py input.mp4 -o output.mp4

# Analyze profile
python -m pstats profile.stats

# Memory profiling
python -m memory_profiler cli.py input.mp4 -o output.mp4

# Bottleneck detection
python scripts/optimize.py --detect-bottlenecks

# Benchmarking
python benchmarks/benchmark_suite.py
```

## Error Messages Reference

### CUDA Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Insufficient VRAM | Reduce batch size |
| `CUDA error: device-side assert` | Invalid operation | Check tensor shapes |
| `CUDA illegal memory access` | Memory corruption | Update drivers, reduce batch |
| `CUDA unknown error` | Driver issue | Restart, update drivers |

### PyTorch Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: Expected tensor` | Wrong input type | Convert to tensor |
| `RuntimeError: shape mismatch` | Incompatible shapes | Check dimensions |
| `RuntimeError: Input type and weight type` | Dtype mismatch | Match dtypes |

### Application Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Missing file | Check path |
| `PermissionError` | No write access | Check permissions |
| `ValueError: Invalid config` | Bad configuration | Validate config |
| `ImportError` | Missing dependency | Install package |

## Getting Help

If issues persist:

1. **Check Logs**:
```bash
tail -n 100 logs/vr_body_segmentation.log
```

2. **Run Diagnostics**:
```bash
python scripts/optimize.py --detect-bottlenecks
```

3. **Gather System Info**:
```bash
# GPU info
nvidia-smi > gpu_info.txt

# System info
uname -a > system_info.txt
python --version >> system_info.txt
pip list >> system_info.txt
```

4. **Create Minimal Reproduction**:
```bash
# Test with smallest config
python cli.py input.mp4 -o output.mp4 --profile fast --batch-size 1
```

5. **Enable Debug Mode**:
```bash
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better errors
python cli.py input.mp4 -o output.mp4 --log-level DEBUG
```

## Prevention Tips

- ✓ Always run benchmarks before production
- ✓ Monitor GPU temperature and utilization
- ✓ Use version control for configurations
- ✓ Keep dependencies updated
- ✓ Test with sample data first
- ✓ Enable logging for debugging
- ✓ Regular system maintenance
- ✓ Backup important configurations
