"""
GPU Optimization Module for VR Body Segmentation
RTX 3090 + Threadripper 3990X Optimizations

This module provides comprehensive GPU acceleration infrastructure including:
- Custom CUDA kernels for video preprocessing
- TensorRT integration for maximum inference performance
- Intelligent memory management and VRAM monitoring
- Dynamic batching for optimal throughput
- Async CPU-GPU pipeline with multi-threading
- Performance profiling and bottleneck detection

Target Hardware:
- GPU: NVIDIA RTX 3090 (24GB VRAM, Ampere architecture)
- CPU: AMD Ryzen Threadripper 3990X (128 threads, 64 cores)
- RAM: 48GB
"""

from .cuda_kernels import (
    CUDAKernelProcessor,
    PinnedMemoryPool,
    FUSED_PREPROCESS_KERNEL,
    BATCH_NORM_KERNEL,
    RGB_BGR_CONVERT_KERNEL,
    MASK_POSTPROCESS_KERNEL
)

from .tensorrt_engine import TensorRTEngine

from .memory_manager import (
    VRAMMonitor,
    MemoryPool,
    ModelQuantizer,
    MixedPrecisionManager,
    AdaptiveBatchSizer,
    MemoryStats
)

from .batch_processor import (
    DynamicBatcher,
    StreamBatcher,
    BatchSizeOptimizer,
    BatchConfig,
    InferenceRequest,
    InferenceResponse
)

from .async_pipeline import (
    AsyncPipeline,
    VideoDecoder,
    GPUPreprocessor,
    GPUInferenceEngine,
    CPUPostprocessor,
    PipelineStage,
    PipelineItem,
    PipelineStats,
    LockFreeQueue
)

from .profiler import (
    Profiler,
    CUDAProfiler,
    MemoryProfiler,
    Timer,
    TimingRecord,
    ProfileStats
)

__version__ = "1.0.0"
__author__ = "VR Body Segmentation Team"

__all__ = [
    # CUDA Kernels
    "CUDAKernelProcessor",
    "PinnedMemoryPool",
    "FUSED_PREPROCESS_KERNEL",
    "BATCH_NORM_KERNEL",
    "RGB_BGR_CONVERT_KERNEL",
    "MASK_POSTPROCESS_KERNEL",

    # TensorRT
    "TensorRTEngine",

    # Memory Management
    "VRAMMonitor",
    "MemoryPool",
    "ModelQuantizer",
    "MixedPrecisionManager",
    "AdaptiveBatchSizer",
    "MemoryStats",

    # Batch Processing
    "DynamicBatcher",
    "StreamBatcher",
    "BatchSizeOptimizer",
    "BatchConfig",
    "InferenceRequest",
    "InferenceResponse",

    # Async Pipeline
    "AsyncPipeline",
    "VideoDecoder",
    "GPUPreprocessor",
    "GPUInferenceEngine",
    "CPUPostprocessor",
    "PipelineStage",
    "PipelineItem",
    "PipelineStats",
    "LockFreeQueue",

    # Profiling
    "Profiler",
    "CUDAProfiler",
    "MemoryProfiler",
    "Timer",
    "TimingRecord",
    "ProfileStats",
]


def get_system_info():
    """
    Get GPU and system information.

    Returns:
        Dictionary with system information
    """
    import torch
    import platform
    import psutil

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["num_gpus"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            }

    # CPU info
    info["cpu_count"] = psutil.cpu_count(logical=True)
    info["cpu_physical_cores"] = psutil.cpu_count(logical=False)

    # Memory info
    mem = psutil.virtual_memory()
    info["system_memory_gb"] = mem.total / (1024**3)
    info["available_memory_gb"] = mem.available / (1024**3)

    return info


def check_requirements():
    """
    Check if all required packages are installed.

    Returns:
        Dictionary with package availability
    """
    requirements = {}

    # PyTorch
    try:
        import torch
        requirements["torch"] = torch.__version__
    except ImportError:
        requirements["torch"] = "NOT INSTALLED"

    # CuPy
    try:
        import cupy
        requirements["cupy"] = cupy.__version__
    except ImportError:
        requirements["cupy"] = "NOT INSTALLED"

    # TensorRT
    try:
        import tensorrt
        requirements["tensorrt"] = tensorrt.__version__
    except ImportError:
        requirements["tensorrt"] = "NOT INSTALLED"

    # PyCUDA
    try:
        import pycuda
        requirements["pycuda"] = "INSTALLED"
    except ImportError:
        requirements["pycuda"] = "NOT INSTALLED"

    return requirements


def print_system_info():
    """Print system information."""
    info = get_system_info()

    print("="*80)
    print("GPU OPTIMIZATION MODULE - SYSTEM INFORMATION")
    print("="*80)
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"Number of GPUs: {info['num_gpus']}")

        for i in range(info['num_gpus']):
            gpu_info = info[f'gpu_{i}']
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu_info['name']}")
            print(f"  Memory: {gpu_info['total_memory_gb']:.2f} GB")
            print(f"  Compute Capability: {gpu_info['compute_capability']}")
            print(f"  Multiprocessors: {gpu_info['multi_processor_count']}")

    print(f"\nCPU Threads: {info['cpu_count']}")
    print(f"CPU Physical Cores: {info['cpu_physical_cores']}")
    print(f"System Memory: {info['system_memory_gb']:.2f} GB")
    print(f"Available Memory: {info['available_memory_gb']:.2f} GB")

    print("\n" + "="*80)
    print("PACKAGE REQUIREMENTS")
    print("="*80)

    requirements = check_requirements()
    for package, version in requirements.items():
        status = "✓" if version != "NOT INSTALLED" else "✗"
        print(f"{status} {package}: {version}")

    print("="*80)


# Print info on import (can be disabled)
if __name__ != "__main__":
    import os
    if os.environ.get("GPU_MODULE_VERBOSE", "0") == "1":
        print_system_info()
