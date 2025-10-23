"""
Integration tests for performance benchmarking.

Tests FPS, latency, throughput, and resource utilization.
"""

import pytest
import numpy as np
import torch
import time
import psutil
import os
from unittest.mock import Mock


class TestFPSBenchmarks:
    """Test frames per second benchmarks."""

    def test_single_frame_fps(self, sample_frame, mock_model, device, performance_monitor):
        """Test FPS for single frame processing."""
        num_iterations = 100

        start_time = time.perf_counter()

        for _ in range(num_iterations):
            resized = torch.from_numpy(sample_frame).permute(2, 0, 1).float() / 255.0
            resized = torch.nn.functional.interpolate(
                resized.unsqueeze(0),
                size=(640, 640),
                mode='bilinear',
                align_corners=False
            ).to(device)

            with torch.no_grad():
                mask = mock_model(resized)

            if device == 'cuda':
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        fps = num_iterations / elapsed

        performance_monitor.record_fps(fps)

        assert fps > 0
        print(f"\nSingle frame FPS: {fps:.2f}")

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_fps(self, mock_model, device, batch_size, performance_monitor):
        """Test FPS for different batch sizes."""
        num_iterations = 50

        # Create batch
        batch = torch.rand(batch_size, 3, 640, 640, device=device)

        start_time = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                masks = mock_model(batch)

            if device == 'cuda':
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        fps = (num_iterations * batch_size) / elapsed

        performance_monitor.record_fps(fps)

        print(f"\nBatch size {batch_size} FPS: {fps:.2f}")
        assert fps > 0

    @pytest.mark.parametrize("resolution", [
        (640, 640),
        (1280, 720),
        (1920, 1080),
    ])
    def test_resolution_fps(self, mock_model, device, resolution, performance_monitor):
        """Test FPS at different resolutions."""
        num_iterations = 50

        # Note: mock model always outputs same size
        # In real implementation, would need resolution-aware model
        input_tensor = torch.rand(1, 3, *resolution, device=device)

        start_time = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                mask = mock_model(input_tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        fps = num_iterations / elapsed

        performance_monitor.record_fps(fps)

        print(f"\nResolution {resolution} FPS: {fps:.2f}")
        assert fps > 0


class TestLatencyBenchmarks:
    """Test latency benchmarks."""

    def test_inference_latency(self, mock_model, device, performance_monitor):
        """Test inference latency."""
        num_warmup = 10
        num_iterations = 100

        input_tensor = torch.rand(1, 3, 640, 640, device=device)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = mock_model(input_tensor)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Measure
        latencies = []

        for _ in range(num_iterations):
            start = time.perf_counter()

            with torch.no_grad():
                mask = mock_model(input_tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            performance_monitor.record_latency(latency_ms)

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nLatency stats:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Std Dev: {std_latency:.2f}ms")
        print(f"  P50: {p50_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")

        assert avg_latency > 0

    def test_end_to_end_latency(self, sample_frame, mock_model, device, performance_monitor):
        """Test end-to-end pipeline latency."""
        import cv2

        num_iterations = 50
        latencies = []

        for _ in range(num_iterations):
            start = time.perf_counter()

            # Full pipeline
            resized = cv2.resize(sample_frame, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            mask_np = mask.squeeze().cpu().numpy()
            mask_resized = cv2.resize(mask_np, (sample_frame.shape[1], sample_frame.shape[0]))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

            if device == 'cuda':
                torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            performance_monitor.record_latency(latency_ms)

        avg_latency = np.mean(latencies)
        print(f"\nEnd-to-end latency: {avg_latency:.2f}ms")

        assert avg_latency > 0


class TestThroughputBenchmarks:
    """Test throughput benchmarks."""

    @pytest.mark.slow
    def test_sustained_throughput(self, mock_model, device, performance_monitor):
        """Test sustained throughput over time."""
        duration_seconds = 10
        batch_size = 4

        input_batch = torch.rand(batch_size, 3, 640, 640, device=device)

        start_time = time.perf_counter()
        frames_processed = 0

        while time.perf_counter() - start_time < duration_seconds:
            with torch.no_grad():
                masks = mock_model(input_batch)

            if device == 'cuda':
                torch.cuda.synchronize()

            frames_processed += batch_size

        elapsed = time.perf_counter() - start_time
        throughput = frames_processed / elapsed

        print(f"\nSustained throughput: {throughput:.2f} FPS over {duration_seconds}s")

        assert throughput > 0

    def test_peak_throughput(self, mock_model, device):
        """Test peak throughput with optimal batch size."""
        # Find optimal batch size for this device
        if device == 'cuda':
            # Try increasing batch sizes
            batch_sizes = [1, 2, 4, 8, 16, 32]
        else:
            batch_sizes = [1, 2, 4, 8]

        best_throughput = 0
        best_batch_size = 1

        for batch_size in batch_sizes:
            try:
                input_batch = torch.rand(batch_size, 3, 640, 640, device=device)

                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = mock_model(input_batch)

                if device == 'cuda':
                    torch.cuda.synchronize()

                # Measure
                start = time.perf_counter()
                num_iterations = 50

                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = mock_model(input_batch)

                if device == 'cuda':
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                throughput = (num_iterations * batch_size) / elapsed

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size

                print(f"Batch {batch_size}: {throughput:.2f} FPS")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Batch {batch_size}: OOM")
                    break
                raise

        print(f"\nBest throughput: {best_throughput:.2f} FPS at batch size {best_batch_size}")

        assert best_throughput > 0


class TestMemoryBenchmarks:
    """Test memory usage benchmarks."""

    @pytest.mark.gpu
    def test_gpu_memory_usage(self, mock_model, cuda_available, performance_monitor):
        """Test GPU memory usage during inference."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            torch.cuda.reset_peak_memory_stats()

            input_batch = torch.rand(batch_size, 3, 640, 640, device='cuda')

            with torch.no_grad():
                masks = mock_model(input_batch)

            torch.cuda.synchronize()

            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB

            performance_monitor.record_gpu_memory(peak)

            print(f"\nBatch {batch_size}:")
            print(f"  Allocated: {allocated:.2f} MB")
            print(f"  Peak: {peak:.2f} MB")

            del input_batch, masks
            torch.cuda.empty_cache()

    def test_cpu_memory_usage(self, sample_frame, mock_model, device, performance_monitor):
        """Test CPU memory usage."""
        import cv2

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / (1024**2)  # MB

        # Process frames
        for _ in range(100):
            resized = cv2.resize(sample_frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

        final_memory = process.memory_info().rss / (1024**2)  # MB
        memory_growth = final_memory - initial_memory

        performance_monitor.record_cpu_memory(final_memory)

        print(f"\nCPU memory:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Growth: {memory_growth:.2f} MB")

        # Memory growth should be minimal
        assert memory_growth < 100  # Less than 100MB growth


class TestGPUUtilization:
    """Test GPU utilization metrics."""

    @pytest.mark.gpu
    def test_gpu_utilization_monitoring(self, cuda_available):
        """Test monitoring GPU utilization."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        try:
            import py3nvml.py3nvml as nvml

            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            # Get GPU info
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)

            print(f"\nGPU Info:")
            print(f"  Memory Total: {info.total / (1024**3):.2f} GB")
            print(f"  Memory Used: {info.used / (1024**3):.2f} GB")
            print(f"  Memory Free: {info.free / (1024**3):.2f} GB")
            print(f"  GPU Utilization: {utilization.gpu}%")
            print(f"  Memory Utilization: {utilization.memory}%")

            nvml.nvmlShutdown()

        except ImportError:
            pytest.skip("py3nvml not available")
        except Exception as e:
            pytest.skip(f"NVML error: {e}")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_sustained_gpu_utilization(self, mock_model, cuda_available):
        """Test GPU utilization under sustained load."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        duration_seconds = 5
        batch_size = 8

        input_batch = torch.rand(batch_size, 3, 640, 640, device='cuda')

        start_time = time.perf_counter()

        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            utilizations = []

            while time.perf_counter() - start_time < duration_seconds:
                with torch.no_grad():
                    _ = mock_model(input_batch)

                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                utilizations.append(util.gpu)

            avg_util = np.mean(utilizations)
            print(f"\nAverage GPU utilization: {avg_util:.1f}%")

            nvml.nvmlShutdown()

        except ImportError:
            pytest.skip("py3nvml not available")


class TestScalabilityBenchmarks:
    """Test scalability with different configurations."""

    @pytest.mark.parametrize("num_streams", [1, 2, 4])
    @pytest.mark.gpu
    def test_multi_stream_processing(self, mock_model, cuda_available, num_streams):
        """Test processing multiple streams concurrently."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        input_tensors = [
            torch.rand(1, 3, 640, 640, device='cuda')
            for _ in range(num_streams)
        ]

        start_time = time.perf_counter()

        # Process on different streams
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                with torch.no_grad():
                    _ = mock_model(input_tensors[i])

        # Wait for all streams
        for stream in streams:
            stream.synchronize()

        elapsed = time.perf_counter() - start_time

        print(f"\n{num_streams} streams processed in {elapsed*1000:.2f}ms")

        assert elapsed > 0

    @pytest.mark.slow
    def test_long_running_stability(self, mock_model, device):
        """Test stability over long running period."""
        duration_seconds = 30
        batch_size = 4

        input_batch = torch.rand(batch_size, 3, 640, 640, device=device)

        start_time = time.perf_counter()
        iterations = 0
        errors = 0

        while time.perf_counter() - start_time < duration_seconds:
            try:
                with torch.no_grad():
                    _ = mock_model(input_batch)

                if device == 'cuda':
                    torch.cuda.synchronize()

                iterations += 1

            except Exception as e:
                errors += 1
                print(f"Error at iteration {iterations}: {e}")

        elapsed = time.perf_counter() - start_time
        fps = (iterations * batch_size) / elapsed

        print(f"\nLong running test:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Iterations: {iterations}")
        print(f"  Errors: {errors}")
        print(f"  Average FPS: {fps:.2f}")

        assert errors == 0
        assert iterations > 0


class TestOptimizationComparison:
    """Test different optimization strategies."""

    @pytest.mark.gpu
    def test_fp32_vs_fp16(self, cuda_available):
        """Compare FP32 vs FP16 performance."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        ).cuda()

        input_tensor = torch.rand(4, 3, 640, 640, device='cuda')

        # FP32
        model.float()
        input_fp32 = input_tensor.float()

        start = time.perf_counter()
        for _ in range(50):
            with torch.no_grad():
                _ = model(input_fp32)
        torch.cuda.synchronize()
        fp32_time = time.perf_counter() - start

        # FP16
        model.half()
        input_fp16 = input_tensor.half()

        start = time.perf_counter()
        for _ in range(50):
            with torch.no_grad():
                _ = model(input_fp16)
        torch.cuda.synchronize()
        fp16_time = time.perf_counter() - start

        speedup = fp32_time / fp16_time

        print(f"\nFP32 time: {fp32_time:.3f}s")
        print(f"FP16 time: {fp16_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        assert fp16_time > 0


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.slow
    def test_4k_video_processing(self, temp_dir, mock_model, device):
        """Test processing 4K video."""
        from test_utils import TestDataGenerator
        import cv2

        video_path = str(temp_dir / "4k.mp4")
        TestDataGenerator.create_test_video(
            video_path,
            width=3840,
            height=2160,
            duration_seconds=2.0
        )

        cap = cv2.VideoCapture(video_path)

        start_time = time.perf_counter()
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downsample for processing
            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            # Upsample mask back to 4K
            mask_upscaled = torch.nn.functional.interpolate(
                mask,
                size=(2160, 3840),
                mode='bilinear',
                align_corners=False
            )

            frames_processed += 1

        elapsed = time.perf_counter() - start_time
        cap.release()

        fps = frames_processed / elapsed

        print(f"\n4K processing:")
        print(f"  Frames: {frames_processed}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")

        assert fps > 0

    @pytest.mark.slow
    def test_vr_stereo_performance(self, test_vr_video_path, mock_model, device):
        """Test VR stereo processing performance."""
        import cv2

        cap = cv2.VideoCapture(test_vr_video_path)

        height, width, _ = cap.read()[1].shape
        half_width = width // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        start_time = time.perf_counter()
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Split eyes
            left_eye = frame[:, :half_width]
            right_eye = frame[:, half_width:]

            # Process both in batch
            left_resized = cv2.resize(left_eye, (640, 640))
            right_resized = cv2.resize(right_eye, (640, 640))

            batch = np.stack([left_resized, right_resized])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            tensor = tensor.to(device)

            with torch.no_grad():
                masks = mock_model(tensor)

            frames_processed += 1

        elapsed = time.perf_counter() - start_time
        cap.release()

        fps = frames_processed / elapsed

        print(f"\nVR stereo processing:")
        print(f"  Frames: {frames_processed}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")

        assert fps > 0
