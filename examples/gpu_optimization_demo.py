"""
GPU Optimization Demo
Complete example demonstrating all GPU optimization features

This script shows how to use the GPU optimization infrastructure for
high-performance VR body segmentation.
"""

import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gpu import (
    CUDAKernelProcessor,
    TensorRTEngine,
    VRAMMonitor,
    MemoryPool,
    ModelQuantizer,
    AdaptiveBatchSizer,
    DynamicBatcher,
    BatchConfig,
    InferenceRequest,
    AsyncPipeline,
    Profiler,
    CUDAProfiler,
    MemoryProfiler,
    print_system_info
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummySegmentationModel(nn.Module):
    """Dummy segmentation model for demonstration."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 2, 1)  # Binary segmentation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def demo_cuda_kernels():
    """Demonstrate custom CUDA kernel usage."""
    logger.info("="*80)
    logger.info("DEMO 1: Custom CUDA Kernels")
    logger.info("="*80)

    # Initialize processor
    processor = CUDAKernelProcessor(device_id=0)

    # Create test image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    logger.info(f"Input shape: {test_image.shape}")

    # Warmup
    for _ in range(10):
        result = processor.fused_preprocess(test_image, (512, 512))
        processor.synchronize()

    # Benchmark
    num_iterations = 100
    start = time.perf_counter()

    for _ in range(num_iterations):
        result = processor.fused_preprocess(test_image, (512, 512))
        processor.synchronize()

    elapsed = time.perf_counter() - start
    fps = num_iterations / elapsed

    logger.info(f"Fused preprocessing: {fps:.2f} FPS ({elapsed/num_iterations*1000:.2f} ms/frame)")
    logger.info(f"Output shape: {result.shape}")
    logger.info("")


def demo_tensorrt_optimization():
    """Demonstrate TensorRT optimization."""
    logger.info("="*80)
    logger.info("DEMO 2: TensorRT Optimization")
    logger.info("="*80)

    # Create model
    model = DummySegmentationModel()
    model.eval()

    # Define dynamic shapes
    input_shapes = {
        'input': {
            'min': (1, 3, 512, 512),
            'opt': (4, 3, 512, 512),
            'max': (8, 3, 512, 512)
        }
    }

    # Build TensorRT engine
    engine_path = '/tmp/demo_model_fp16.trt'

    logger.info("Building TensorRT engine (this may take a few minutes)...")
    engine = TensorRTEngine(
        max_workspace_size=4,
        fp16_mode=True,
        int8_mode=False,
        device_id=0
    )

    try:
        engine.build_engine_from_pytorch(
            model=model,
            input_shapes=input_shapes,
            output_path=engine_path,
            opset_version=13
        )

        # Benchmark
        results = engine.benchmark(num_iterations=100)
        logger.info(f"TensorRT Performance:")
        logger.info(f"  Average latency: {results['avg_latency_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['throughput_fps']:.2f} FPS")

    except Exception as e:
        logger.warning(f"TensorRT demo failed (this is normal if TensorRT not installed): {e}")

    logger.info("")


def demo_memory_management():
    """Demonstrate memory management."""
    logger.info("="*80)
    logger.info("DEMO 3: Memory Management")
    logger.info("="*80)

    # VRAM monitoring
    monitor = VRAMMonitor(device_id=0)
    monitor.log_stats()

    # Memory pooling
    pool = MemoryPool(device_id=0, pool_size_gb=2.0)

    logger.info("Allocating tensors from pool...")
    tensors = []
    for i in range(10):
        tensor = pool.allocate(1024 * 1024)
        tensors.append(tensor)

    logger.info(f"Pool stats: {pool.get_pool_stats()}")

    logger.info("Freeing tensors...")
    for tensor in tensors:
        pool.free(tensor)

    logger.info(f"Pool stats after free: {pool.get_pool_stats()}")

    # Model quantization
    model = DummySegmentationModel()
    size_fp32 = ModelQuantizer.measure_model_size(model)
    logger.info(f"FP32 model size: {size_fp32:.2f} MB")

    model_fp16 = ModelQuantizer.quantize_fp16(model)
    size_fp16 = ModelQuantizer.measure_model_size(model_fp16)
    reduction = (1 - size_fp16 / size_fp32) * 100
    logger.info(f"FP16 model size: {size_fp16:.2f} MB ({reduction:.1f}% reduction)")

    # Adaptive batch sizing
    batch_sizer = AdaptiveBatchSizer(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16
    )

    logger.info("Testing adaptive batch sizing...")
    for i in range(5):
        batch_size = batch_sizer.adjust_batch_size()
        logger.info(f"  Iteration {i}: batch_size = {batch_size}")

    logger.info("")


def demo_batch_processing():
    """Demonstrate batch processing."""
    logger.info("="*80)
    logger.info("DEMO 4: Dynamic Batch Processing")
    logger.info("="*80)

    # Dummy inference function
    def inference_fn(batch: np.ndarray) -> np.ndarray:
        # Simulate inference
        time.sleep(0.01)
        return np.zeros((batch.shape[0], 2, 512, 512), dtype=np.float32)

    # Configure batcher
    config = BatchConfig(
        min_batch_size=1,
        max_batch_size=8,
        timeout_ms=50.0,
        dynamic_batching=True
    )

    batcher = DynamicBatcher(config, inference_fn)
    batcher.start()

    # Submit requests
    logger.info("Submitting 20 inference requests...")
    results = []
    for i in range(20):
        request = InferenceRequest(
            data=np.random.randn(3, 512, 512).astype(np.float32),
            request_id=f"req_{i}",
            timestamp=time.perf_counter()
        )
        response = batcher.submit(request)
        results.append(response)

        if i % 5 == 0:
            logger.info(f"  Request {i}: latency={response.latency_ms:.2f}ms, batch_size={response.batch_size}")

    # Statistics
    batcher.log_stats()
    batcher.stop()

    logger.info("")


def demo_async_pipeline():
    """Demonstrate async pipeline."""
    logger.info("="*80)
    logger.info("DEMO 5: Async CPU-GPU Pipeline")
    logger.info("="*80)

    # Create model
    model = DummySegmentationModel().cuda().eval()

    # Configure pipeline
    config = {
        'decode_threads': 4,
        'preprocess_streams': 2,
        'batch_size': 4,
        'postprocess_threads': 8,
        'queue_size': 50,
        'target_size': (512, 512)
    }

    # Create pipeline
    pipeline = AsyncPipeline(model, config, device_id=0)
    pipeline.start()

    # Submit frames
    logger.info("Submitting 50 frames...")
    for i in range(50):
        pipeline.submit_frame(None, i)  # Dummy frame
        time.sleep(0.01)  # Simulate 100 FPS input

    # Get results
    logger.info("Retrieving results...")
    for i in range(50):
        result = pipeline.get_result(timeout=5.0)
        if result:
            if i % 10 == 0:
                logger.info(f"  Frame {result.frame_id}: latency={result.metadata.get('total_latency', 0)*1000:.2f}ms")

    # Statistics
    stats = pipeline.get_stats()
    logger.info(f"Pipeline Statistics:")
    logger.info(f"  FPS: {stats.fps:.2f}")
    logger.info(f"  Queue depths: {stats.queue_depths}")
    logger.info(f"  Bottleneck: {stats.bottleneck_stage}")

    pipeline.stop()

    logger.info("")


def demo_profiling():
    """Demonstrate profiling tools."""
    logger.info("="*80)
    logger.info("DEMO 6: Performance Profiling")
    logger.info("="*80)

    # Basic profiling
    profiler = Profiler(enabled=True)

    logger.info("Running profiled operations...")
    for i in range(50):
        with profiler.profile('preprocessing'):
            time.sleep(0.001)

        with profiler.profile('inference'):
            time.sleep(0.005)

        with profiler.profile('postprocessing'):
            time.sleep(0.002)

    # Print summary
    profiler.print_summary()

    # Export
    profiler.export_json('/tmp/profile_results.json')
    logger.info("Exported profiling data to /tmp/profile_results.json")

    # CUDA profiling (if available)
    if torch.cuda.is_available():
        logger.info("\nCUDA Profiling:")
        cuda_profiler = CUDAProfiler()

        model = nn.Linear(100, 10).cuda()
        input_tensor = torch.randn(32, 100).cuda()

        with cuda_profiler.profile():
            for _ in range(10):
                output = model(input_tensor)
                torch.cuda.synchronize()

        cuda_profiler.print_summary()

    # Memory profiling
    if torch.cuda.is_available():
        logger.info("\nMemory Profiling:")
        mem_profiler = MemoryProfiler(device_id=0)

        mem_profiler.snapshot('start')
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(10)]
        mem_profiler.snapshot('after_alloc')

        del tensors
        torch.cuda.empty_cache()
        mem_profiler.snapshot('after_free')

        mem_profiler.compare_snapshots('start', 'after_alloc')
        mem_profiler.compare_snapshots('after_alloc', 'after_free')

    logger.info("")


def main():
    """Run all demos."""
    logger.info("\n" + "="*80)
    logger.info("GPU OPTIMIZATION MODULE - DEMONSTRATION")
    logger.info("="*80 + "\n")

    # Print system info
    print_system_info()

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Some demos will be skipped.")
        return

    # Run demos
    try:
        demo_cuda_kernels()
    except Exception as e:
        logger.error(f"CUDA kernels demo failed: {e}", exc_info=True)

    try:
        demo_tensorrt_optimization()
    except Exception as e:
        logger.error(f"TensorRT demo failed: {e}", exc_info=True)

    try:
        demo_memory_management()
    except Exception as e:
        logger.error(f"Memory management demo failed: {e}", exc_info=True)

    try:
        demo_batch_processing()
    except Exception as e:
        logger.error(f"Batch processing demo failed: {e}", exc_info=True)

    try:
        demo_async_pipeline()
    except Exception as e:
        logger.error(f"Async pipeline demo failed: {e}", exc_info=True)

    try:
        demo_profiling()
    except Exception as e:
        logger.error(f"Profiling demo failed: {e}", exc_info=True)

    logger.info("="*80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
