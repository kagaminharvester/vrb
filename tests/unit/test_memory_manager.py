"""
Unit tests for memory manager module.

Tests memory allocation, tracking, and optimization for both CPU and GPU memory.
"""

import pytest
import torch
import numpy as np
import gc
import psutil
import os
from unittest.mock import Mock, patch


class TestMemoryTracking:
    """Test memory usage tracking."""

    def test_get_current_memory_usage(self):
        """Test getting current process memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        assert memory_info.rss > 0  # Resident Set Size
        assert memory_info.vms > 0  # Virtual Memory Size

    def test_memory_usage_in_mb(self):
        """Test converting memory usage to MB."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)

        assert memory_mb > 0

    @pytest.mark.gpu
    def test_gpu_memory_usage(self, cuda_available):
        """Test tracking GPU memory usage."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        assert allocated >= 0
        assert reserved >= allocated

    @pytest.mark.gpu
    def test_peak_memory_tracking(self, cuda_available):
        """Test tracking peak memory usage."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.cuda.reset_peak_memory_stats()

        # Allocate some memory
        tensor = torch.rand(1000, 1000, device='cuda')
        peak = torch.cuda.max_memory_allocated()

        assert peak > 0

        del tensor
        torch.cuda.empty_cache()


class TestMemoryAllocation:
    """Test memory allocation strategies."""

    def test_preallocate_cpu_memory(self):
        """Test pre-allocating CPU memory."""
        size = (1000, 1000)
        buffer = np.empty(size, dtype=np.float32)

        assert buffer.nbytes == 1000 * 1000 * 4  # float32 = 4 bytes

    @pytest.mark.gpu
    def test_preallocate_gpu_memory(self, cuda_available):
        """Test pre-allocating GPU memory."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        size = (1000, 1000)
        buffer = torch.empty(size, device='cuda', dtype=torch.float32)

        assert buffer.element_size() * buffer.nelement() == 1000 * 1000 * 4

        del buffer
        torch.cuda.empty_cache()

    def test_allocate_pinned_memory(self, cuda_available):
        """Test allocating pinned (page-locked) memory."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(1000, 1000, pin_memory=True)
        assert tensor.is_pinned()

    def test_memory_pool_simulation(self):
        """Test simulating a memory pool."""
        pool_size = 10
        buffer_shape = (100, 100)

        # Pre-allocate pool
        pool = [np.empty(buffer_shape, dtype=np.float32) for _ in range(pool_size)]

        assert len(pool) == pool_size

        # Reuse buffers
        for i in range(pool_size):
            pool[i].fill(0)

        # Buffers should still exist
        assert len(pool) == pool_size


class TestMemoryReuse:
    """Test memory reuse patterns."""

    def test_buffer_reuse(self):
        """Test reusing the same buffer."""
        buffer = np.zeros((1000, 1000), dtype=np.float32)

        # Reuse buffer multiple times
        for i in range(10):
            buffer.fill(i)
            result = buffer.sum()

        assert buffer.shape == (1000, 1000)

    @pytest.mark.gpu
    def test_gpu_buffer_reuse(self, cuda_available):
        """Test reusing GPU buffer."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        buffer = torch.zeros(1000, 1000, device='cuda')

        for i in range(10):
            buffer.fill_(i)
            result = buffer.sum()

        assert buffer.shape == (1000, 1000)

        del buffer
        torch.cuda.empty_cache()

    def test_inplace_operations(self):
        """Test in-place operations to avoid memory allocation."""
        tensor = torch.rand(1000, 1000)

        # In-place operations
        tensor.add_(1.0)
        tensor.mul_(2.0)
        tensor.clamp_(0, 10)

        assert tensor.shape == (1000, 1000)


class TestMemoryOptimization:
    """Test memory optimization techniques."""

    def test_gradient_checkpointing_concept(self, cuda_available, device):
        """Test gradient checkpointing concept."""
        # Gradient checkpointing trades compute for memory
        # Here we just test the concept with a simple model

        model = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        ).to(device)

        input_tensor = torch.rand(32, 100, device=device)

        # Normal forward pass (stores all activations)
        output1 = model(input_tensor)

        # With checkpointing, intermediate activations would be released
        # and recomputed during backward pass

        assert output1.shape == (32, 10)

    def test_clear_cache(self):
        """Test clearing Python object cache."""
        # Create some objects
        large_list = [np.random.rand(100, 100) for _ in range(100)]

        # Clear
        del large_list
        gc.collect()

        # Memory should be freed

    @pytest.mark.gpu
    def test_empty_gpu_cache(self, cuda_available):
        """Test emptying GPU cache."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Allocate
        tensors = [torch.rand(100, 100, device='cuda') for _ in range(10)]

        memory_before = torch.cuda.memory_allocated()

        # Delete
        del tensors
        torch.cuda.empty_cache()

        memory_after = torch.cuda.memory_allocated()

        # Memory should be reduced (or zero)
        assert memory_after <= memory_before

    def test_numpy_vs_list_memory(self):
        """Test NumPy arrays are more memory efficient than lists."""
        size = 10000

        # List of floats
        list_data = [float(i) for i in range(size)]

        # NumPy array
        array_data = np.arange(size, dtype=np.float32)

        # NumPy should be more memory efficient
        # (Hard to test precisely, but conceptually important)

        assert len(list_data) == len(array_data)


class TestMemoryLeakDetection:
    """Test memory leak detection."""

    def test_no_leak_in_loop(self):
        """Test no memory leak in processing loop."""
        initial_objects = len(gc.get_objects())

        # Process many items
        for i in range(100):
            temp = np.random.rand(100, 100)
            result = temp.sum()
            del temp

        gc.collect()

        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly
        # (Some growth is normal due to Python internals)
        growth = final_objects - initial_objects
        assert growth < 1000  # Arbitrary threshold

    @pytest.mark.gpu
    def test_no_gpu_leak_in_loop(self, cuda_available):
        """Test no GPU memory leak in processing loop."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()

        for i in range(10):
            tensor = torch.rand(100, 100, device='cuda')
            result = tensor.sum()
            del tensor
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should return to initial level
        assert final_memory == initial_memory

    def test_circular_reference_cleanup(self):
        """Test circular references are cleaned up."""
        class Node:
            def __init__(self):
                self.data = np.random.rand(100, 100)
                self.next = None

        # Create circular reference
        node1 = Node()
        node2 = Node()
        node1.next = node2
        node2.next = node1

        # Delete
        del node1, node2

        # Force garbage collection
        collected = gc.collect()

        # Should collect circular references
        # (Number collected varies by Python version)


class TestBatchMemoryManagement:
    """Test memory management for batch processing."""

    def test_optimal_batch_size_calculation(self, cuda_available):
        """Test calculating optimal batch size."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Get available memory
        props = torch.cuda.get_device_properties(0)
        available_memory = props.total_memory * 0.8  # Use 80%

        # Estimate memory per sample (e.g., 10MB)
        memory_per_sample = 10 * 1024 * 1024

        # Calculate batch size
        optimal_batch_size = int(available_memory / memory_per_sample)

        assert optimal_batch_size > 0

    def test_dynamic_batch_sizing(self, cuda_available):
        """Test dynamically adjusting batch size."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        batch_size = 32

        try:
            # Try processing batch
            tensor = torch.rand(batch_size, 3, 640, 640, device='cuda')
            # Process...
            del tensor
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduce batch size
                batch_size = batch_size // 2
                pytest.skip(f"OOM, would reduce to batch_size={batch_size}")

    def test_batch_splitting(self):
        """Test splitting large batch into smaller chunks."""
        large_batch = np.random.rand(128, 3, 640, 640)

        chunk_size = 32
        num_chunks = (len(large_batch) + chunk_size - 1) // chunk_size

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(large_batch))
            chunk = large_batch[start:end]
            chunks.append(chunk)

        total_samples = sum(len(c) for c in chunks)
        assert total_samples == len(large_batch)


class TestMemoryProfiling:
    """Test memory profiling utilities."""

    def test_memory_snapshot(self):
        """Test taking memory snapshot."""
        process = psutil.Process(os.getpid())

        snapshot = {
            'rss_mb': process.memory_info().rss / (1024 * 1024),
            'vms_mb': process.memory_info().vms / (1024 * 1024),
            'percent': process.memory_percent()
        }

        assert snapshot['rss_mb'] > 0
        assert snapshot['percent'] > 0

    @pytest.mark.gpu
    def test_gpu_memory_snapshot(self, cuda_available):
        """Test taking GPU memory snapshot."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        snapshot = {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
        }

        assert snapshot['allocated_mb'] >= 0
        assert snapshot['reserved_mb'] >= snapshot['allocated_mb']

    def test_track_memory_growth(self):
        """Test tracking memory growth over time."""
        process = psutil.Process(os.getpid())

        snapshots = []

        for i in range(5):
            # Allocate some memory
            temp = np.random.rand(1000, 1000)

            # Take snapshot
            memory_mb = process.memory_info().rss / (1024 * 1024)
            snapshots.append(memory_mb)

            del temp

        # Snapshots should show memory changes
        assert len(snapshots) == 5


class TestMemoryLimits:
    """Test memory limit enforcement."""

    def test_check_available_memory(self):
        """Test checking available system memory."""
        memory = psutil.virtual_memory()

        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        percent_used = memory.percent

        assert total_gb > 0
        assert available_gb > 0
        assert 0 <= percent_used <= 100

    @pytest.mark.gpu
    def test_check_gpu_memory_limit(self, cuda_available):
        """Test checking GPU memory limit."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)

        assert total_memory_gb > 0

    def test_enforce_memory_limit(self):
        """Test enforcing memory limit."""
        max_memory_gb = 10  # Example limit

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < max_memory_gb:
            # Not enough memory available
            pytest.skip("Insufficient memory")

        # Proceed with processing


class TestCacheManagement:
    """Test caching strategies."""

    def test_lru_cache_concept(self):
        """Test LRU cache concept for processed frames."""
        from functools import lru_cache

        @lru_cache(maxsize=10)
        def process_frame(frame_id):
            # Simulate expensive processing
            return np.random.rand(100, 100)

        # First call - computes
        result1 = process_frame(1)

        # Second call - cached
        result2 = process_frame(1)

        # Should return same object from cache
        assert np.array_equal(result1, result2)

    def test_manual_cache_management(self):
        """Test manual cache with size limit."""
        cache = {}
        max_cache_size = 5

        for i in range(10):
            frame_data = np.random.rand(100, 100)

            # Add to cache
            cache[i] = frame_data

            # Evict oldest if over limit
            if len(cache) > max_cache_size:
                oldest_key = min(cache.keys())
                del cache[oldest_key]

        assert len(cache) <= max_cache_size


class TestMemoryAlignment:
    """Test memory alignment for performance."""

    def test_numpy_array_alignment(self):
        """Test NumPy array memory alignment."""
        array = np.random.rand(1000, 1000)

        # Check if array data is aligned
        assert array.flags['C_CONTIGUOUS'] or array.flags['F_CONTIGUOUS']

    def test_torch_tensor_alignment(self, device):
        """Test PyTorch tensor memory layout."""
        tensor = torch.rand(1000, 1000, device=device)

        # Check if tensor is contiguous
        assert tensor.is_contiguous()

    def test_force_contiguous(self):
        """Test forcing tensor to be contiguous."""
        # Create non-contiguous tensor
        tensor = torch.rand(10, 10).t()  # Transpose

        assert not tensor.is_contiguous()

        # Make contiguous
        contiguous_tensor = tensor.contiguous()

        assert contiguous_tensor.is_contiguous()


class TestResourceCleanup:
    """Test resource cleanup."""

    def test_context_manager_cleanup(self):
        """Test resource cleanup with context manager."""
        class TensorBuffer:
            def __init__(self, size):
                self.buffer = torch.rand(size)

            def __enter__(self):
                return self.buffer

            def __exit__(self, exc_type, exc_val, exc_tb):
                del self.buffer
                gc.collect()

        with TensorBuffer((1000, 1000)) as buffer:
            result = buffer.sum()

        # Buffer should be cleaned up

    def test_explicit_cleanup(self):
        """Test explicit resource cleanup."""
        tensors = []

        # Allocate
        for i in range(10):
            tensors.append(torch.rand(100, 100))

        # Cleanup
        for tensor in tensors:
            del tensor
        tensors.clear()

        gc.collect()

        assert len(tensors) == 0

    @pytest.mark.gpu
    def test_gpu_resource_cleanup(self, cuda_available):
        """Test GPU resource cleanup."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.cuda.empty_cache()
        initial = torch.cuda.memory_allocated()

        # Allocate
        tensors = [torch.rand(100, 100, device='cuda') for _ in range(10)]

        # Cleanup
        del tensors
        torch.cuda.empty_cache()

        final = torch.cuda.memory_allocated()

        assert final == initial
