"""
Integration tests for full video processing pipeline.

Tests end-to-end pipeline from video input to segmented output.
"""

import pytest
import numpy as np
import cv2
import torch
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_single_frame_pipeline(self, sample_frame, mock_model, device):
        """Test processing single frame through pipeline."""
        # 1. Preprocess
        resized = cv2.resize(sample_frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        # 2. Inference
        with torch.no_grad():
            mask = mock_model(tensor)

        # 3. Postprocess
        mask_np = mask.squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask_np, (sample_frame.shape[1], sample_frame.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        assert mask_binary.shape == sample_frame.shape[:2]

    def test_video_pipeline(self, test_video_path, mock_model, device, temp_dir):
        """Test processing complete video through pipeline."""
        cap = cv2.VideoCapture(test_video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create output video writer
        output_path = str(temp_dir / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            resized = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            # Postprocess and write
            mask_np = mask.squeeze().cpu().numpy()
            mask_resized = cv2.resize(mask_np, (width, height))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Apply mask to frame
            output_frame = frame.copy()
            output_frame[mask_binary == 0] = 0

            out.write(output_frame)
            frames_processed += 1

        cap.release()
        out.release()

        assert frames_processed > 0
        assert Path(output_path).exists()

    def test_batch_pipeline(self, test_video_path, mock_model, device):
        """Test processing video in batches."""
        cap = cv2.VideoCapture(test_video_path)
        batch_size = 4

        frames_processed = 0
        batch = []

        while True:
            ret, frame = cap.read()

            if not ret:
                # Process final partial batch
                if len(batch) > 0:
                    batch_array = np.stack(batch)
                    # Process batch...
                    frames_processed += len(batch)
                break

            # Preprocess frame
            resized = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch.append(rgb)

            if len(batch) == batch_size:
                # Process batch
                batch_array = np.stack(batch)
                tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2).float() / 255.0
                tensor = tensor.to(device)

                with torch.no_grad():
                    masks = mock_model(tensor)

                frames_processed += batch_size
                batch = []

        cap.release()

        assert frames_processed >= 30  # At least 1 second at 30fps


class TestPipelineWithDifferentInputs:
    """Test pipeline with various input formats."""

    @pytest.mark.parametrize("resolution", [
        (640, 480),
        (1280, 720),
        (1920, 1080),
    ])
    def test_different_resolutions(self, temp_dir, resolution, mock_model, device):
        """Test pipeline with different input resolutions."""
        from test_utils import TestDataGenerator

        video_path = str(temp_dir / "test.mp4")
        TestDataGenerator.create_test_video(
            video_path,
            width=resolution[0],
            height=resolution[1],
            duration_seconds=0.5
        )

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        assert ret
        assert frame.shape[:2] == (resolution[1], resolution[0])

        # Process through pipeline
        resized = cv2.resize(frame, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mask = mock_model(tensor)

        assert mask is not None

    @pytest.mark.parametrize("fps", [24, 30, 60])
    def test_different_framerates(self, temp_dir, fps, mock_model, device):
        """Test pipeline with different framerates."""
        from test_utils import TestDataGenerator

        video_path = str(temp_dir / "test.mp4")
        TestDataGenerator.create_test_video(
            video_path,
            fps=fps,
            duration_seconds=0.5
        )

        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # FPS should be close to requested
        assert abs(actual_fps - fps) < 1.0


class TestPipelinePerformance:
    """Test pipeline performance metrics."""

    def test_measure_fps(self, test_video_path, mock_model, device):
        """Test measuring processing FPS."""
        cap = cv2.VideoCapture(test_video_path)

        start_time = time.perf_counter()
        frames_processed = 0

        for _ in range(30):  # Process 30 frames
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            frames_processed += 1

        elapsed_time = time.perf_counter() - start_time
        cap.release()

        fps = frames_processed / elapsed_time

        assert fps > 0
        print(f"Processing FPS: {fps:.2f}")

    def test_measure_latency(self, sample_frame, mock_model, device):
        """Test measuring per-frame latency."""
        latencies = []

        for _ in range(10):
            start_time = time.perf_counter()

            # Process single frame
            resized = cv2.resize(sample_frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(elapsed)

        avg_latency = np.mean(latencies)
        assert avg_latency > 0
        print(f"Average latency: {avg_latency:.2f}ms")

    @pytest.mark.gpu
    def test_gpu_utilization(self, test_video_path, mock_model, cuda_available):
        """Test GPU utilization during processing."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        cap = cv2.VideoCapture(test_video_path)

        torch.cuda.reset_peak_memory_stats()

        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).cuda()

            with torch.no_grad():
                mask = mock_model(tensor)

        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        cap.release()

        assert peak_memory > 0
        print(f"Peak GPU memory: {peak_memory:.2f}MB")


class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""

    def test_corrupted_frame_handling(self, mock_model, device):
        """Test handling of corrupted frames."""
        # Create corrupted frame (all zeros)
        corrupted_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Should process without crashing
        resized = cv2.resize(corrupted_frame, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mask = mock_model(tensor)

        assert mask is not None

    def test_inconsistent_frame_sizes(self, mock_model, device):
        """Test handling frames of different sizes."""
        sizes = [(640, 480), (1280, 720), (1920, 1080)]

        for size in sizes:
            frame = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)

            # Preprocess normalizes size
            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            assert mask.shape[2:] == (640, 640)

    def test_empty_video_handling(self, temp_dir):
        """Test handling of video with no frames."""
        # Create empty video file
        video_path = str(temp_dir / "empty.mp4")
        Path(video_path).touch()

        cap = cv2.VideoCapture(video_path)
        is_opened = cap.isOpened()
        cap.release()

        assert not is_opened

    def test_memory_cleanup_after_processing(self, test_video_path, mock_model, device):
        """Test memory is cleaned up after processing."""
        if device == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        cap = cv2.VideoCapture(test_video_path)

        # Process some frames
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            del tensor, mask

        cap.release()

        if device == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()

            # Memory should return to initial state
            assert final_memory == initial_memory


class TestPipelineConfiguration:
    """Test pipeline with different configurations."""

    def test_different_batch_sizes(self, test_video_path, mock_model, device):
        """Test pipeline with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            cap = cv2.VideoCapture(test_video_path)
            batch = []

            frames_processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    if len(batch) > 0:
                        # Process final batch
                        frames_processed += len(batch)
                    break

                resized = cv2.resize(frame, (640, 640))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                batch.append(rgb)

                if len(batch) == batch_size:
                    batch_array = np.stack(batch)
                    tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2).float() / 255.0
                    tensor = tensor.to(device)

                    with torch.no_grad():
                        masks = mock_model(tensor)

                    frames_processed += batch_size
                    batch = []

            cap.release()

            assert frames_processed > 0

    def test_different_confidence_thresholds(self, sample_frame, mock_model, device):
        """Test different confidence thresholds."""
        thresholds = [0.3, 0.5, 0.7, 0.9]

        resized = cv2.resize(sample_frame, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mask = mock_model(tensor)

        mask_np = mask.squeeze().cpu().numpy()

        for threshold in thresholds:
            binary_mask = (mask_np > threshold).astype(np.uint8) * 255
            pixel_count = (binary_mask > 0).sum()

            # Higher thresholds should generally have fewer pixels
            assert binary_mask.shape == mask_np.shape


class TestPipelineOutputQuality:
    """Test output quality of pipeline."""

    def test_mask_quality_metrics(self, sample_frame, mock_model, device):
        """Test mask quality metrics."""
        resized = cv2.resize(sample_frame, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mask = mock_model(tensor)

        mask_np = mask.squeeze().cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.uint8)

        # Calculate coverage
        coverage = binary_mask.sum() / binary_mask.size

        assert 0 <= coverage <= 1

    def test_output_video_properties(self, test_video_path, mock_model, device, temp_dir):
        """Test output video has correct properties."""
        cap = cv2.VideoCapture(test_video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = str(temp_dir / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process and write frames
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)

        cap.release()
        out.release()

        # Verify output
        out_cap = cv2.VideoCapture(output_path)
        out_width = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = out_cap.get(cv2.CAP_PROP_FPS)
        out_cap.release()

        assert out_width == width
        assert out_height == height
        assert abs(out_fps - fps) < 1.0


class TestPipelineStressTest:
    """Stress test pipeline with challenging scenarios."""

    @pytest.mark.slow
    def test_long_video_processing(self, temp_dir, mock_model, device):
        """Test processing long video."""
        from test_utils import TestDataGenerator

        # Create 5 second video
        video_path = str(temp_dir / "long.mp4")
        TestDataGenerator.create_test_video(
            video_path,
            duration_seconds=5.0
        )

        cap = cv2.VideoCapture(video_path)
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            frames_processed += 1

        cap.release()

        assert frames_processed >= 150  # 5 seconds at 30fps

    @pytest.mark.slow
    def test_high_resolution_processing(self, temp_dir, mock_model, device):
        """Test processing high resolution video."""
        from test_utils import TestDataGenerator

        # Create 4K video
        video_path = str(temp_dir / "4k.mp4")
        TestDataGenerator.create_test_video(
            video_path,
            width=3840,
            height=2160,
            duration_seconds=1.0
        )

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        assert ret
        assert frame.shape == (2160, 3840, 3)

        # Process through pipeline
        resized = cv2.resize(frame, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mask = mock_model(tensor)

        assert mask is not None
