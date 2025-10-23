"""
GPU Memory Management and Optimization
RTX 3090 VRAM Optimization (24GB)

This module provides intelligent VRAM management, memory pooling,
and model quantization for optimal memory utilization.
"""

import torch
import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import OrderedDict
import gc
import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_vram_gb: float
    allocated_vram_gb: float
    reserved_vram_gb: float
    free_vram_gb: float
    utilization_percent: float
    system_ram_gb: float
    system_ram_available_gb: float


class VRAMMonitor:
    """
    Real-time VRAM usage monitoring.

    Features:
    - Track allocation and deallocation
    - Memory leak detection
    - Usage alerts
    """

    def __init__(self, device_id: int = 0, alert_threshold: float = 0.9):
        """
        Initialize VRAM monitor.

        Args:
            device_id: GPU device ID
            alert_threshold: Alert when VRAM usage exceeds this ratio
        """
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.alert_threshold = alert_threshold

        # Get total VRAM
        self.total_vram = torch.cuda.get_device_properties(device_id).total_memory

    def get_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats object with current usage
        """
        # GPU memory
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        free = self.total_vram - reserved
        utilization = (allocated / self.total_vram) * 100

        # System RAM
        system_mem = psutil.virtual_memory()

        return MemoryStats(
            total_vram_gb=self.total_vram / (1024**3),
            allocated_vram_gb=allocated / (1024**3),
            reserved_vram_gb=reserved / (1024**3),
            free_vram_gb=free / (1024**3),
            utilization_percent=utilization,
            system_ram_gb=system_mem.total / (1024**3),
            system_ram_available_gb=system_mem.available / (1024**3)
        )

    def log_stats(self):
        """Log current memory statistics."""
        stats = self.get_stats()
        logger.info(
            f"VRAM: {stats.allocated_vram_gb:.2f}/{stats.total_vram_gb:.2f} GB "
            f"({stats.utilization_percent:.1f}%), "
            f"Reserved: {stats.reserved_vram_gb:.2f} GB, "
            f"Free: {stats.free_vram_gb:.2f} GB"
        )
        logger.info(
            f"System RAM: {stats.system_ram_available_gb:.2f}/"
            f"{stats.system_ram_gb:.2f} GB available"
        )

    def check_alert(self) -> bool:
        """
        Check if memory usage exceeds alert threshold.

        Returns:
            True if alert threshold exceeded
        """
        stats = self.get_stats()
        if stats.utilization_percent / 100 > self.alert_threshold:
            logger.warning(
                f"VRAM usage alert: {stats.utilization_percent:.1f}% "
                f"(threshold: {self.alert_threshold*100:.0f}%)"
            )
            return True
        return False

    def clear_cache(self):
        """Clear PyTorch CUDA cache."""
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device)


class MemoryPool:
    """
    Custom memory pool for efficient tensor allocation.

    Features:
    - Pre-allocated memory chunks
    - Fast allocation/deallocation
    - Reduced fragmentation
    """

    def __init__(
        self,
        device_id: int = 0,
        pool_size_gb: float = 4.0,
        chunk_sizes: List[int] = None
    ):
        """
        Initialize memory pool.

        Args:
            device_id: GPU device ID
            pool_size_gb: Total pool size in GB
            chunk_sizes: List of common tensor sizes to pre-allocate
        """
        self.device = torch.device(f'cuda:{device_id}')
        self.pool_size = int(pool_size_gb * 1024**3)

        # Default chunk sizes (common tensor shapes)
        if chunk_sizes is None:
            chunk_sizes = [
                1 * 3 * 512 * 512,    # Single frame
                4 * 3 * 512 * 512,    # Batch of 4
                8 * 3 * 512 * 512,    # Batch of 8
                1 * 64 * 512 * 512,   # Feature map
            ]

        self.chunk_sizes = sorted(chunk_sizes)
        self.pools = {size: [] for size in chunk_sizes}
        self.allocated = {}

        logger.info(f"Initialized memory pool with {len(chunk_sizes)} chunk sizes")

    def allocate(
        self,
        size: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Allocate tensor from pool.

        Args:
            size: Number of elements
            dtype: Data type

        Returns:
            Allocated tensor
        """
        # Find appropriate chunk size
        chunk_size = self._find_chunk_size(size)

        if chunk_size in self.pools and len(self.pools[chunk_size]) > 0:
            # Reuse from pool
            tensor = self.pools[chunk_size].pop()
            logger.debug(f"Reused tensor from pool (size: {chunk_size})")
        else:
            # Allocate new
            tensor = torch.empty(chunk_size, dtype=dtype, device=self.device)
            logger.debug(f"Allocated new tensor (size: {chunk_size})")

        # Track allocation
        tensor_id = id(tensor)
        self.allocated[tensor_id] = (chunk_size, dtype)

        return tensor[:size]  # Return view of requested size

    def free(self, tensor: torch.Tensor):
        """
        Return tensor to pool.

        Args:
            tensor: Tensor to free
        """
        tensor_id = id(tensor)

        if tensor_id in self.allocated:
            chunk_size, dtype = self.allocated.pop(tensor_id)

            if chunk_size in self.pools:
                # Return full tensor (not view) to pool
                full_tensor = tensor if tensor.numel() == chunk_size else tensor.storage().data_ptr()
                self.pools[chunk_size].append(tensor)
                logger.debug(f"Returned tensor to pool (size: {chunk_size})")
        else:
            logger.warning("Attempted to free untracked tensor")

    def _find_chunk_size(self, size: int) -> int:
        """Find smallest chunk size that fits requested size."""
        for chunk_size in self.chunk_sizes:
            if chunk_size >= size:
                return chunk_size
        # If no exact match, return requested size
        return size

    def clear(self):
        """Clear all pooled memory."""
        for size_pool in self.pools.values():
            size_pool.clear()
        self.allocated.clear()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleared memory pool")

    def get_pool_stats(self) -> Dict[int, int]:
        """Get pool statistics."""
        return {size: len(pool) for size, pool in self.pools.items()}


class ModelQuantizer:
    """
    Model quantization utilities for reducing memory footprint.

    Supports:
    - FP32 -> FP16 (2x reduction)
    - FP32 -> INT8 (4x reduction)
    - Dynamic quantization
    - Static quantization with calibration
    """

    @staticmethod
    def quantize_fp16(model: torch.nn.Module) -> torch.nn.Module:
        """
        Convert model to FP16 (half precision).

        Args:
            model: Input model (FP32)

        Returns:
            FP16 model
        """
        model_fp16 = model.half()
        logger.info("Converted model to FP16")
        return model_fp16

    @staticmethod
    def quantize_int8_dynamic(model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply dynamic INT8 quantization.

        Args:
            model: Input model (FP32)

        Returns:
            INT8 quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        logger.info("Applied dynamic INT8 quantization")
        return quantized_model

    @staticmethod
    def quantize_int8_static(
        model: torch.nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> torch.nn.Module:
        """
        Apply static INT8 quantization with calibration.

        Args:
            model: Input model (FP32)
            calibration_loader: DataLoader for calibration data
            device: Device for calibration

        Returns:
            INT8 quantized model
        """
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

        # Calibrate with representative data
        logger.info("Calibrating model for INT8 quantization...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                data = data.to(device)
                model(data)
                if batch_idx >= 10:  # Use 10 batches for calibration
                    break

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        logger.info("Applied static INT8 quantization")

        return model

    @staticmethod
    def measure_model_size(model: torch.nn.Module) -> float:
        """
        Measure model size in MB.

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)
        return size_mb

    @staticmethod
    def compare_quantization(
        model_fp32: torch.nn.Module,
        model_fp16: torch.nn.Module = None,
        model_int8: torch.nn.Module = None
    ):
        """
        Compare sizes of different quantization levels.

        Args:
            model_fp32: FP32 model
            model_fp16: FP16 model (optional)
            model_int8: INT8 model (optional)
        """
        size_fp32 = ModelQuantizer.measure_model_size(model_fp32)
        logger.info(f"FP32 model size: {size_fp32:.2f} MB")

        if model_fp16:
            size_fp16 = ModelQuantizer.measure_model_size(model_fp16)
            reduction = (1 - size_fp16 / size_fp32) * 100
            logger.info(f"FP16 model size: {size_fp16:.2f} MB ({reduction:.1f}% reduction)")

        if model_int8:
            size_int8 = ModelQuantizer.measure_model_size(model_int8)
            reduction = (1 - size_int8 / size_fp32) * 100
            logger.info(f"INT8 model size: {size_int8:.2f} MB ({reduction:.1f}% reduction)")


class MixedPrecisionManager:
    """
    Mixed precision training and inference manager.

    Uses automatic mixed precision (AMP) for optimal performance.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize mixed precision manager.

        Args:
            enabled: Enable mixed precision
        """
        self.enabled = enabled
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def autocast(self):
        """
        Context manager for automatic mixed precision.

        Returns:
            Autocast context
        """
        return torch.cuda.amp.autocast(enabled=self.enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.

        Args:
            loss: Unscaled loss

        Returns:
            Scaled loss
        """
        return self.scaler.scale(loss)

    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Step optimizer with gradient scaling.

        Args:
            optimizer: PyTorch optimizer
        """
        self.scaler.step(optimizer)
        self.scaler.update()


class AdaptiveBatchSizer:
    """
    Dynamically adjust batch size based on available VRAM.

    Features:
    - Automatic batch size tuning
    - OOM prevention
    - Maximize throughput
    """

    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 16,
        target_utilization: float = 0.85,
        device_id: int = 0
    ):
        """
        Initialize adaptive batch sizer.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_utilization: Target VRAM utilization (0-1)
            device_id: GPU device ID
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_utilization = target_utilization
        self.device_id = device_id

        self.monitor = VRAMMonitor(device_id)
        self.oom_count = 0

    def adjust_batch_size(self) -> int:
        """
        Adjust batch size based on current VRAM usage.

        Returns:
            New batch size
        """
        stats = self.monitor.get_stats()
        current_utilization = stats.utilization_percent / 100

        if current_utilization < self.target_utilization - 0.1:
            # VRAM underutilized, increase batch size
            new_batch_size = min(
                self.current_batch_size + 1,
                self.max_batch_size
            )
            if new_batch_size != self.current_batch_size:
                logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size

        elif current_utilization > self.target_utilization + 0.1:
            # VRAM overutilized, decrease batch size
            new_batch_size = max(
                self.current_batch_size - 1,
                self.min_batch_size
            )
            if new_batch_size != self.current_batch_size:
                logger.info(f"Decreasing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size

        return self.current_batch_size

    def handle_oom(self) -> int:
        """
        Handle out-of-memory error by reducing batch size.

        Returns:
            New (reduced) batch size
        """
        self.oom_count += 1
        logger.warning(f"OOM detected (count: {self.oom_count})")

        # Reduce batch size aggressively
        new_batch_size = max(
            self.current_batch_size // 2,
            self.min_batch_size
        )

        logger.info(f"Reducing batch size after OOM: {self.current_batch_size} -> {new_batch_size}")
        self.current_batch_size = new_batch_size

        # Clear cache
        torch.cuda.empty_cache()

        return self.current_batch_size

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size


# Example usage
if __name__ == "__main__":
    # Initialize VRAM monitor
    monitor = VRAMMonitor(device_id=0)
    monitor.log_stats()

    # Test memory pool
    pool = MemoryPool(device_id=0, pool_size_gb=2.0)

    tensors = []
    for i in range(10):
        tensor = pool.allocate(1024 * 1024)  # 1M elements
        tensors.append(tensor)

    logger.info(f"Pool stats: {pool.get_pool_stats()}")

    # Free tensors
    for tensor in tensors:
        pool.free(tensor)

    logger.info(f"Pool stats after free: {pool.get_pool_stats()}")

    # Test adaptive batch sizing
    batch_sizer = AdaptiveBatchSizer(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16
    )

    for i in range(5):
        batch_size = batch_sizer.adjust_batch_size()
        logger.info(f"Iteration {i}: batch_size = {batch_size}")

    monitor.log_stats()
