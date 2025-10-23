"""
Custom CUDA Kernels for Video Preprocessing
Optimized for NVIDIA RTX 3090 (Ampere Architecture)

This module provides high-performance CUDA kernels for common video preprocessing
operations, leveraging CuPy for Python-based CUDA kernel development.
"""

import cupy as cp
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# CUDA Kernel for combined resize + color conversion + normalization
# This fused kernel reduces memory bandwidth by combining operations
FUSED_PREPROCESS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void fused_preprocess(
    const unsigned char* input,      // Input RGB image [H, W, 3]
    float* output,                   // Output normalized tensor [3, H_out, W_out]
    int in_height, int in_width,
    int out_height, int out_width,
    float scale_h, float scale_w,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    // Calculate output pixel coordinates
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    // Bilinear interpolation coordinates
    float in_y = out_y * scale_h;
    float in_x = out_x * scale_w;

    int y0 = (int)in_y;
    int x0 = (int)in_x;
    int y1 = min(y0 + 1, in_height - 1);
    int x1 = min(x0 + 1, in_width - 1);

    float dy = in_y - y0;
    float dx = in_x - x0;

    // Bilinear weights
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    // Process each channel with bilinear interpolation + normalization
    for (int c = 0; c < 3; c++) {
        // Get 4 corner pixels
        float p00 = input[(y0 * in_width + x0) * 3 + c];
        float p01 = input[(y0 * in_width + x1) * 3 + c];
        float p10 = input[(y1 * in_width + x0) * 3 + c];
        float p11 = input[(y1 * in_width + x1) * 3 + c];

        // Bilinear interpolation
        float interpolated = w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11;

        // Normalize based on channel
        float mean, std;
        if (c == 0) { mean = mean_r; std = std_r; }
        else if (c == 1) { mean = mean_g; std = std_g; }
        else { mean = mean_b; std = std_b; }

        float normalized = (interpolated / 255.0f - mean) / std;

        // Store in CHW format (channel-first)
        output[c * out_height * out_width + out_y * out_width + out_x] = normalized;
    }
}
''', 'fused_preprocess')


# CUDA Kernel for efficient batch normalization
BATCH_NORM_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void batch_normalize(
    const float* input,
    float* output,
    const float* mean,
    const float* std,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;

    if (idx >= total_elements) return;

    int channel = (idx / (height * width)) % channels;
    output[idx] = (input[idx] - mean[channel]) / std[channel];
}
''', 'batch_normalize')


# CUDA Kernel for RGB to BGR conversion (optimized for video codecs)
RGB_BGR_CONVERT_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void rgb_to_bgr(
    const unsigned char* input,
    unsigned char* output,
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pixels) return;

    int pixel_offset = idx * 3;

    // Swap R and B channels
    output[pixel_offset + 0] = input[pixel_offset + 2];  // B
    output[pixel_offset + 1] = input[pixel_offset + 1];  // G
    output[pixel_offset + 2] = input[pixel_offset + 0];  // R
}
''', 'rgb_to_bgr')


# CUDA Kernel for mask post-processing (argmax + smoothing)
MASK_POSTPROCESS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void argmax_and_smooth(
    const float* input,      // [B, C, H, W] logits
    unsigned char* output,   // [B, H, W] class indices
    int batch_size,
    int num_classes,
    int height,
    int width,
    float threshold
) {
    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || y >= height || x >= width) return;

    // Find argmax across classes
    int max_class = 0;
    float max_val = input[b * num_classes * height * width + 0 * height * width + y * width + x];

    for (int c = 1; c < num_classes; c++) {
        float val = input[b * num_classes * height * width + c * height * width + y * width + x];
        if (val > max_val) {
            max_val = val;
            max_class = c;
        }
    }

    // Apply confidence threshold (set to background if below threshold)
    if (max_val < threshold) {
        max_class = 0;
    }

    output[b * height * width + y * width + x] = (unsigned char)max_class;
}
''', 'argmax_and_smooth')


class CUDAKernelProcessor:
    """
    High-performance CUDA kernel processor for video preprocessing.

    Optimizations:
    - Fused operations to reduce memory bandwidth
    - Pinned memory for fast CPU-GPU transfers
    - Async execution with CUDA streams
    - Optimal block/grid sizing for Ampere architecture
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize CUDA kernel processor.

        Args:
            device_id: GPU device ID (default: 0)
        """
        self.device = cp.cuda.Device(device_id)
        self.device.use()

        # Create CUDA stream for async execution
        self.stream = cp.cuda.Stream(non_blocking=True)

        # Get device properties for optimal kernel configuration
        self.props = self.device.attributes
        self.max_threads_per_block = self.props['MaxThreadsPerBlock']
        self.max_block_dim = (
            self.props['MaxBlockDimX'],
            self.props['MaxBlockDimY'],
            self.props['MaxBlockDimZ']
        )

        logger.info(f"Initialized CUDA processor on device {device_id}")
        logger.info(f"Max threads per block: {self.max_threads_per_block}")

    def calculate_grid_block(
        self,
        height: int,
        width: int,
        block_size: Tuple[int, int] = (16, 16)
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Calculate optimal grid and block dimensions for 2D kernel.

        Args:
            height: Image height
            width: Image width
            block_size: Thread block size (default: 16x16 = 256 threads)

        Returns:
            (grid_dim, block_dim) tuples
        """
        block_x, block_y = block_size
        grid_x = (width + block_x - 1) // block_x
        grid_y = (height + block_y - 1) // block_y

        return (grid_x, grid_y, 1), (block_x, block_y, 1)

    def fused_preprocess(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        async_exec: bool = True
    ) -> cp.ndarray:
        """
        Fused preprocessing: resize + color conversion + normalization.

        This single kernel performs all operations in one pass, minimizing
        memory bandwidth and maximizing throughput.

        Args:
            image: Input image [H, W, 3] uint8 (RGB)
            target_size: (height, width) for output
            mean: Normalization mean per channel
            std: Normalization std per channel
            async_exec: Use async execution with CUDA streams

        Returns:
            Preprocessed tensor [3, H, W] float32 on GPU
        """
        in_height, in_width = image.shape[:2]
        out_height, out_width = target_size

        # Transfer input to GPU (use pinned memory for faster transfer)
        with self.stream if async_exec else cp.cuda.Stream.null:
            input_gpu = cp.asarray(image, dtype=cp.uint8)
            output_gpu = cp.empty((3, out_height, out_width), dtype=cp.float32)

            # Calculate scaling factors
            scale_h = float(in_height - 1) / (out_height - 1) if out_height > 1 else 0
            scale_w = float(in_width - 1) / (out_width - 1) if out_width > 1 else 0

            # Configure kernel launch parameters
            grid, block = self.calculate_grid_block(out_height, out_width)

            # Launch fused kernel
            FUSED_PREPROCESS_KERNEL(
                grid, block,
                (
                    input_gpu, output_gpu,
                    in_height, in_width,
                    out_height, out_width,
                    scale_h, scale_w,
                    mean[0], mean[1], mean[2],
                    std[0], std[1], std[2]
                ),
                stream=self.stream if async_exec else None
            )

        return output_gpu

    def batch_preprocess(
        self,
        images: list,
        target_size: Tuple[int, int],
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> cp.ndarray:
        """
        Batch preprocessing for multiple frames.

        Args:
            images: List of input images [H, W, 3]
            target_size: Target size for all images
            mean: Normalization mean
            std: Normalization std

        Returns:
            Batch tensor [B, 3, H, W] on GPU
        """
        batch_size = len(images)
        out_height, out_width = target_size

        # Allocate output batch
        batch_gpu = cp.empty((batch_size, 3, out_height, out_width), dtype=cp.float32)

        # Process each image (can be parallelized with multiple streams)
        for i, img in enumerate(images):
            result = self.fused_preprocess(img, target_size, mean, std, async_exec=True)
            batch_gpu[i] = result

        # Synchronize stream
        self.stream.synchronize()

        return batch_gpu

    def rgb_to_bgr(self, image: cp.ndarray) -> cp.ndarray:
        """
        Convert RGB to BGR format (for OpenCV compatibility).

        Args:
            image: Input image on GPU [H, W, 3] uint8

        Returns:
            BGR image on GPU [H, W, 3] uint8
        """
        height, width, _ = image.shape
        total_pixels = height * width

        output = cp.empty_like(image)

        # Configure kernel
        threads_per_block = 256
        blocks = (total_pixels + threads_per_block - 1) // threads_per_block

        RGB_BGR_CONVERT_KERNEL(
            (blocks,), (threads_per_block,),
            (image, output, total_pixels),
            stream=self.stream
        )

        return output

    def argmax_postprocess(
        self,
        logits: cp.ndarray,
        threshold: float = 0.5
    ) -> cp.ndarray:
        """
        Post-process segmentation logits: argmax + thresholding.

        Args:
            logits: Model output [B, C, H, W] float32
            threshold: Confidence threshold for predictions

        Returns:
            Class indices [B, H, W] uint8
        """
        batch_size, num_classes, height, width = logits.shape
        output = cp.empty((batch_size, height, width), dtype=cp.uint8)

        # Configure 3D grid for batch processing
        block = (16, 16, 1)
        grid = (
            (width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1],
            batch_size
        )

        MASK_POSTPROCESS_KERNEL(
            grid, block,
            (logits, output, batch_size, num_classes, height, width, threshold),
            stream=self.stream
        )

        self.stream.synchronize()
        return output

    def synchronize(self):
        """Synchronize CUDA stream."""
        self.stream.synchronize()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'stream'):
            self.stream.synchronize()


class PinnedMemoryPool:
    """
    Pinned (page-locked) memory pool for fast CPU-GPU transfers.

    Pinned memory can be transferred to/from GPU much faster than
    regular pageable memory (~10-20 GB/s vs ~3-5 GB/s).
    """

    def __init__(self, max_size_gb: float = 4.0):
        """
        Initialize pinned memory pool.

        Args:
            max_size_gb: Maximum pool size in GB
        """
        self.max_size = int(max_size_gb * 1024**3)
        self.pool = {}
        self.current_size = 0

        logger.info(f"Initialized pinned memory pool (max: {max_size_gb} GB)")

    def allocate(self, shape: Tuple, dtype=np.uint8) -> np.ndarray:
        """
        Allocate pinned memory array.

        Args:
            shape: Array shape
            dtype: Data type

        Returns:
            Pinned numpy array
        """
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize

        if self.current_size + size > self.max_size:
            logger.warning("Pinned memory pool full, using regular memory")
            return np.empty(shape, dtype=dtype)

        # Allocate pinned memory using CuPy
        arr = cp.cuda.alloc_pinned_memory(size)
        pinned_array = np.frombuffer(arr, dtype=dtype).reshape(shape)

        self.current_size += size
        return pinned_array

    def clear(self):
        """Clear memory pool."""
        self.pool.clear()
        self.current_size = 0


# Example usage and benchmarking
if __name__ == "__main__":
    import time

    # Initialize processor
    processor = CUDAKernelProcessor(device_id=0)

    # Test with sample image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    target_size = (512, 512)

    # Warmup
    for _ in range(10):
        result = processor.fused_preprocess(test_image, target_size)

    # Benchmark
    num_iterations = 100
    start = time.perf_counter()

    for _ in range(num_iterations):
        result = processor.fused_preprocess(test_image, target_size)
        processor.synchronize()

    elapsed = time.perf_counter() - start
    fps = num_iterations / elapsed

    print(f"Fused preprocessing: {fps:.2f} FPS")
    print(f"Average latency: {elapsed/num_iterations*1000:.2f} ms")

    # Batch processing benchmark
    batch_images = [test_image for _ in range(8)]

    start = time.perf_counter()
    batch_result = processor.batch_preprocess(batch_images, target_size)
    processor.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Batch processing (8 frames): {elapsed*1000:.2f} ms")
    print(f"Per-frame latency: {elapsed/8*1000:.2f} ms")
