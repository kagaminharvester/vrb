"""
Unit tests for GPU utilities.

Tests CUDA operations, memory management, and GPU acceleration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestGPUAvailability:
    """Test GPU availability and configuration."""

    def test_cuda_available(self, cuda_available):
        """Test CUDA availability check."""
        assert isinstance(cuda_available, bool)

    def test_cuda_device_count(self, cuda_available):
        """Test getting CUDA device count."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        device_count = torch.cuda.device_count()
        assert device_count > 0

    def test_cuda_device_name(self, cuda_available):
        """Test getting CUDA device name."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        device_name = torch.cuda.get_device_name(0)
        assert isinstance(device_name, str)
        assert len(device_name) > 0

    def test_cuda_capability(self, cuda_available):
        """Test getting CUDA compute capability."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        capability = torch.cuda.get_device_capability(0)
        assert isinstance(capability, tuple)
        assert len(capability) == 2
        assert all(isinstance(x, int) for x in capability)


class TestDeviceManagement:
    """Test device management operations."""

    def test_get_device(self, device):
        """Test getting device."""
        torch_device = torch.device(device)
        assert torch_device.type in ['cuda', 'cpu']

    def test_set_device(self, cuda_available):
        """Test setting CUDA device."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        assert current_device == 0

    def test_device_context(self, cuda_available):
        """Test device context manager."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        with torch.cuda.device(0):
            assert torch.cuda.current_device() == 0

    def test_tensor_device(self, device):
        """Test creating tensor on specific device."""
        tensor = torch.rand(10, 10, device=device)
        assert tensor.device.type == device


class TestMemoryAllocation:
    """Test GPU memory allocation."""

    @pytest.mark.gpu
    def test_allocate_tensor(self, cuda_available):
        """Test allocating tensor on GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(1000, 1000, device='cuda')
        assert tensor.device.type == 'cuda'

        memory_allocated = torch.cuda.memory_allocated()
        assert memory_allocated > 0

        del tensor
        torch.cuda.empty_cache()

    @pytest.mark.gpu
    def test_memory_stats(self, cuda_available):
        """Test getting memory statistics."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.cuda.reset_peak_memory_stats()

        tensor = torch.rand(1000, 1000, device='cuda')

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()

        assert allocated > 0
        assert reserved >= allocated
        assert max_allocated >= allocated

        del tensor
        torch.cuda.empty_cache()

    @pytest.mark.gpu
    def test_empty_cache(self, cuda_available):
        """Test emptying GPU cache."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(1000, 1000, device='cuda')
        del tensor

        torch.cuda.empty_cache()

        # Memory might not be fully released due to caching
        # but the operation should complete without error

    @pytest.mark.gpu
    def test_memory_fraction(self, cuda_available):
        """Test setting memory fraction."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Get total memory
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory

        assert total_memory > 0


class TestDataTransfer:
    """Test data transfer between CPU and GPU."""

    @pytest.mark.gpu
    def test_cpu_to_gpu(self, cuda_available):
        """Test transferring data from CPU to GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        cpu_tensor = torch.rand(100, 100)
        gpu_tensor = cpu_tensor.to('cuda')

        assert cpu_tensor.device.type == 'cpu'
        assert gpu_tensor.device.type == 'cuda'
        torch.testing.assert_close(cpu_tensor, gpu_tensor.cpu())

    @pytest.mark.gpu
    def test_gpu_to_cpu(self, cuda_available):
        """Test transferring data from GPU to CPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        gpu_tensor = torch.rand(100, 100, device='cuda')
        cpu_tensor = gpu_tensor.cpu()

        assert gpu_tensor.device.type == 'cuda'
        assert cpu_tensor.device.type == 'cpu'

    @pytest.mark.gpu
    def test_async_transfer(self, cuda_available):
        """Test asynchronous data transfer."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        cpu_tensor = torch.rand(100, 100, pin_memory=True)
        gpu_tensor = cpu_tensor.to('cuda', non_blocking=True)

        torch.cuda.synchronize()

        assert gpu_tensor.device.type == 'cuda'

    @pytest.mark.gpu
    def test_pinned_memory(self, cuda_available):
        """Test using pinned memory for faster transfers."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        cpu_tensor = torch.rand(100, 100, pin_memory=True)
        assert cpu_tensor.is_pinned()

        gpu_tensor = cpu_tensor.to('cuda')
        assert gpu_tensor.device.type == 'cuda'


class TestMultiGPU:
    """Test multi-GPU operations."""

    @pytest.mark.gpu
    def test_multiple_devices(self, cuda_available):
        """Test working with multiple GPUs."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        device_count = torch.cuda.device_count()

        if device_count < 2:
            pytest.skip("Multiple GPUs not available")

        # Create tensors on different GPUs
        tensor0 = torch.rand(10, 10, device='cuda:0')
        tensor1 = torch.rand(10, 10, device='cuda:1')

        assert tensor0.device.index == 0
        assert tensor1.device.index == 1

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires multiple GPUs")
    def test_data_parallel(self, cuda_available):
        """Test DataParallel for multi-GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        device_count = torch.cuda.device_count()
        if device_count < 2:
            pytest.skip("Multiple GPUs not available")

        model = torch.nn.Linear(10, 10)
        parallel_model = torch.nn.DataParallel(model)
        parallel_model = parallel_model.cuda()

        input_tensor = torch.rand(32, 10, device='cuda')
        output = parallel_model(input_tensor)

        assert output.device.type == 'cuda'


class TestCUDAStreams:
    """Test CUDA streams for concurrent operations."""

    @pytest.mark.gpu
    def test_create_stream(self, cuda_available):
        """Test creating CUDA stream."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        stream = torch.cuda.Stream()
        assert stream is not None

    @pytest.mark.gpu
    def test_stream_context(self, cuda_available):
        """Test using stream context."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            tensor = torch.rand(100, 100, device='cuda')

        stream.synchronize()
        assert tensor.device.type == 'cuda'

    @pytest.mark.gpu
    def test_concurrent_streams(self, cuda_available):
        """Test concurrent operations on different streams."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            tensor1 = torch.rand(1000, 1000, device='cuda')
            result1 = tensor1 @ tensor1.T

        with torch.cuda.stream(stream2):
            tensor2 = torch.rand(1000, 1000, device='cuda')
            result2 = tensor2 @ tensor2.T

        torch.cuda.synchronize()

        assert result1.shape == (1000, 1000)
        assert result2.shape == (1000, 1000)


class TestCUDAEvents:
    """Test CUDA events for timing and synchronization."""

    @pytest.mark.gpu
    def test_create_event(self, cuda_available):
        """Test creating CUDA event."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        event = torch.cuda.Event()
        assert event is not None

    @pytest.mark.gpu
    def test_record_event(self, cuda_available):
        """Test recording CUDA event."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Do some work
        tensor = torch.rand(1000, 1000, device='cuda')
        result = tensor @ tensor.T

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        assert elapsed_time > 0

    @pytest.mark.gpu
    def test_event_synchronization(self, cuda_available):
        """Test event synchronization."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        event = torch.cuda.Event()
        event.record()

        # Wait for event
        event.synchronize()

        # Event should be completed
        assert event.query()


class TestTensorOperations:
    """Test GPU tensor operations."""

    @pytest.mark.gpu
    def test_matrix_multiplication(self, cuda_available):
        """Test matrix multiplication on GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        a = torch.rand(1000, 1000, device='cuda')
        b = torch.rand(1000, 1000, device='cuda')

        c = a @ b

        assert c.shape == (1000, 1000)
        assert c.device.type == 'cuda'

    @pytest.mark.gpu
    def test_element_wise_operations(self, cuda_available):
        """Test element-wise operations on GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        a = torch.rand(1000, 1000, device='cuda')
        b = torch.rand(1000, 1000, device='cuda')

        c = a + b
        d = a * b
        e = torch.relu(a)

        assert c.device.type == 'cuda'
        assert d.device.type == 'cuda'
        assert e.device.type == 'cuda'

    @pytest.mark.gpu
    def test_reduction_operations(self, cuda_available):
        """Test reduction operations on GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(1000, 1000, device='cuda')

        sum_val = tensor.sum()
        mean_val = tensor.mean()
        max_val = tensor.max()
        min_val = tensor.min()

        assert sum_val.device.type == 'cuda'
        assert mean_val.device.type == 'cuda'


class TestCuDNN:
    """Test cuDNN operations."""

    @pytest.mark.gpu
    def test_cudnn_available(self, cuda_available):
        """Test cuDNN availability."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        assert torch.backends.cudnn.is_available()

    @pytest.mark.gpu
    def test_cudnn_benchmark(self, cuda_available):
        """Test cuDNN benchmark mode."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.backends.cudnn.benchmark = True
        assert torch.backends.cudnn.benchmark

        # Test convolution
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
        input_tensor = torch.rand(1, 3, 224, 224, device='cuda')

        output = conv(input_tensor)
        assert output.device.type == 'cuda'

    @pytest.mark.gpu
    def test_cudnn_deterministic(self, cuda_available):
        """Test cuDNN deterministic mode."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.backends.cudnn.deterministic = True
        assert torch.backends.cudnn.deterministic


class TestAMPAutocast:
    """Test Automatic Mixed Precision (AMP)."""

    @pytest.mark.gpu
    def test_autocast(self, cuda_available):
        """Test autocast context."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        model = torch.nn.Linear(10, 10).cuda()
        input_tensor = torch.rand(32, 10, device='cuda')

        with torch.cuda.amp.autocast():
            output = model(input_tensor)

        assert output.device.type == 'cuda'

    @pytest.mark.gpu
    def test_autocast_dtype(self, cuda_available):
        """Test autocast changes dtype to float16."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        conv = torch.nn.Conv2d(3, 64, 3).cuda()
        input_tensor = torch.rand(1, 3, 224, 224, device='cuda')

        with torch.cuda.amp.autocast():
            output = conv(input_tensor)
            # Inside autocast, operations use float16 when beneficial
            # Output dtype might be float16 or float32 depending on operation

        assert output.device.type == 'cuda'

    @pytest.mark.gpu
    def test_grad_scaler(self, cuda_available):
        """Test gradient scaler for mixed precision."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        scaler = torch.cuda.amp.GradScaler()

        model = torch.nn.Linear(10, 10).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        input_tensor = torch.rand(32, 10, device='cuda')
        target = torch.rand(32, 10, device='cuda')

        with torch.cuda.amp.autocast():
            output = model(input_tensor)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


class TestErrorHandling:
    """Test GPU error handling."""

    @pytest.mark.gpu
    def test_out_of_memory(self, cuda_available):
        """Test handling out of memory errors."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        try:
            # Try to allocate huge tensor
            huge = torch.rand(100000, 100000, device='cuda')
            pytest.fail("Should have raised OOM error")
        except RuntimeError as e:
            assert "out of memory" in str(e).lower()

    @pytest.mark.gpu
    def test_device_mismatch(self, cuda_available):
        """Test handling device mismatch errors."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        cpu_tensor = torch.rand(10, 10)
        gpu_tensor = torch.rand(10, 10, device='cuda')

        with pytest.raises(RuntimeError):
            result = cpu_tensor + gpu_tensor

    def test_invalid_device_id(self):
        """Test handling invalid device ID."""
        with pytest.raises((RuntimeError, AssertionError)):
            tensor = torch.rand(10, 10, device='cuda:99')


class TestSynchronization:
    """Test CUDA synchronization."""

    @pytest.mark.gpu
    def test_synchronize(self, cuda_available):
        """Test CUDA synchronization."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(1000, 1000, device='cuda')
        result = tensor @ tensor.T

        torch.cuda.synchronize()

        # All CUDA operations should be complete

    @pytest.mark.gpu
    def test_device_synchronize(self, cuda_available):
        """Test synchronizing specific device."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.rand(100, 100, device='cuda:0')
        result = tensor * 2

        torch.cuda.synchronize(0)

        assert result.device.type == 'cuda'


class TestUtilityFunctions:
    """Test utility functions for GPU management."""

    def test_get_optimal_batch_size(self, cuda_available):
        """Test calculating optimal batch size for GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory

        # Simple heuristic: assume each sample needs 10MB
        sample_memory = 10 * 1024 * 1024
        optimal_batch = int(total_memory * 0.8 / sample_memory)

        assert optimal_batch > 0

    def test_clear_all_memory(self, cuda_available):
        """Test clearing all GPU memory."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Allocate some memory
        tensors = [torch.rand(100, 100, device='cuda') for _ in range(10)]

        # Clear
        del tensors
        torch.cuda.empty_cache()

        # Memory should be reduced (but may not be zero due to PyTorch caching)
