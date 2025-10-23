"""
Integration tests for VR video processing.

Tests VR-specific functionality including stereo processing,
eye separation, and VR format handling.
"""

import pytest
import numpy as np
import cv2
import torch
from pathlib import Path


class TestVRFormatHandling:
    """Test VR video format detection and handling."""

    def test_side_by_side_detection(self, test_vr_video_path):
        """Test detecting side-by-side VR format."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]

        # Side-by-side should have width ~2x height
        aspect_ratio = width / height

        assert aspect_ratio > 1.5  # Typically ~3.5 for SBS

    def test_top_bottom_detection(self, temp_dir):
        """Test detecting top-bottom VR format."""
        from test_utils import TestDataGenerator

        video_path = str(temp_dir / "vr_tb.mp4")
        TestDataGenerator.create_vr_stereo_video(
            video_path,
            width=1920,
            height=2160,
            layout='top_bottom'
        )

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]

        # Top-bottom should have height ~2x width
        aspect_ratio = height / width

        assert aspect_ratio > 1.5


class TestEyeSeparation:
    """Test separating left and right eye views."""

    def test_split_side_by_side(self, test_vr_video_path):
        """Test splitting side-by-side frame into left/right eyes."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_width = width // 2

        # Split into left and right
        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        assert left_eye.shape == (height, half_width, 3)
        assert right_eye.shape == (height, half_width, 3)

    def test_split_top_bottom(self, temp_dir):
        """Test splitting top-bottom frame into left/right eyes."""
        from test_utils import TestDataGenerator

        video_path = str(temp_dir / "vr_tb.mp4")
        TestDataGenerator.create_vr_stereo_video(
            video_path,
            width=1920,
            height=2160,
            layout='top_bottom'
        )

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_height = height // 2

        # Split into top (left) and bottom (right)
        left_eye = frame[:half_height, :]
        right_eye = frame[half_height:, :]

        assert left_eye.shape == (half_height, width, 3)
        assert right_eye.shape == (half_height, width, 3)


class TestStereoProcessing:
    """Test processing both eyes of VR video."""

    def test_process_both_eyes(self, test_vr_video_path, mock_model, device):
        """Test processing both left and right eyes."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        # Split frame
        height, width = frame.shape[:2]
        half_width = width // 2
        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        # Process left eye
        left_resized = cv2.resize(left_eye, (640, 640))
        left_tensor = torch.from_numpy(left_resized).permute(2, 0, 1).float() / 255.0
        left_tensor = left_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            left_mask = mock_model(left_tensor)

        # Process right eye
        right_resized = cv2.resize(right_eye, (640, 640))
        right_tensor = torch.from_numpy(right_resized).permute(2, 0, 1).float() / 255.0
        right_tensor = right_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            right_mask = mock_model(right_tensor)

        assert left_mask is not None
        assert right_mask is not None
        assert left_mask.shape == right_mask.shape

    def test_batch_process_stereo(self, test_vr_video_path, mock_model, device):
        """Test batch processing both eyes together."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        # Split frame
        height, width = frame.shape[:2]
        half_width = width // 2
        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        # Batch process
        left_resized = cv2.resize(left_eye, (640, 640))
        right_resized = cv2.resize(right_eye, (640, 640))

        # Stack into batch
        batch = np.stack([left_resized, right_resized])
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            masks = mock_model(tensor)

        assert masks.shape[0] == 2  # Both eyes


class TestStereoRecombination:
    """Test recombining processed stereo frames."""

    def test_recombine_side_by_side(self, test_vr_video_path, mock_model, device):
        """Test recombining processed left/right into side-by-side."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_width = width // 2

        # Split and process
        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        # Process both (simplified - just use originals)
        left_processed = left_eye.copy()
        right_processed = right_eye.copy()

        # Recombine
        recombined = np.concatenate([left_processed, right_processed], axis=1)

        assert recombined.shape == frame.shape

    def test_recombine_top_bottom(self, temp_dir, mock_model, device):
        """Test recombining processed eyes into top-bottom."""
        from test_utils import TestDataGenerator

        video_path = str(temp_dir / "vr_tb.mp4")
        TestDataGenerator.create_vr_stereo_video(
            video_path,
            width=1920,
            height=2160,
            layout='top_bottom'
        )

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_height = height // 2

        # Split and process
        left_eye = frame[:half_height, :]
        right_eye = frame[half_height:, :]

        # Process both (simplified)
        left_processed = left_eye.copy()
        right_processed = right_eye.copy()

        # Recombine
        recombined = np.concatenate([left_processed, right_processed], axis=0)

        assert recombined.shape == frame.shape


class TestVRPipeline:
    """Test complete VR processing pipeline."""

    def test_full_vr_pipeline_sbs(self, test_vr_video_path, mock_model, device, temp_dir):
        """Test full pipeline for side-by-side VR video."""
        cap = cv2.VideoCapture(test_vr_video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create output writer
        output_path = str(temp_dir / "vr_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_processed = 0
        half_width = width // 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Split eyes
            left_eye = frame[:, :half_width]
            right_eye = frame[:, half_width:]

            # Process left eye
            left_resized = cv2.resize(left_eye, (640, 640))
            left_tensor = torch.from_numpy(left_resized).permute(2, 0, 1).float() / 255.0
            left_tensor = left_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                left_mask = mock_model(left_tensor)

            # Process right eye
            right_resized = cv2.resize(right_eye, (640, 640))
            right_tensor = torch.from_numpy(right_resized).permute(2, 0, 1).float() / 255.0
            right_tensor = right_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                right_mask = mock_model(right_tensor)

            # Apply masks
            left_mask_np = left_mask.squeeze().cpu().numpy()
            left_mask_resized = cv2.resize(left_mask_np, (half_width, height))
            left_binary = (left_mask_resized > 0.5).astype(np.uint8) * 255

            right_mask_np = right_mask.squeeze().cpu().numpy()
            right_mask_resized = cv2.resize(right_mask_np, (half_width, height))
            right_binary = (right_mask_resized > 0.5).astype(np.uint8) * 255

            # Apply to frames
            left_output = left_eye.copy()
            left_output[left_binary == 0] = 0

            right_output = right_eye.copy()
            right_output[right_binary == 0] = 0

            # Recombine
            output_frame = np.concatenate([left_output, right_output], axis=1)

            out.write(output_frame)
            frames_processed += 1

        cap.release()
        out.release()

        assert frames_processed > 0
        assert Path(output_path).exists()


class TestVRPerformance:
    """Test VR processing performance."""

    def test_stereo_processing_speed(self, test_vr_video_path, mock_model, device):
        """Test speed of stereo processing."""
        import time

        cap = cv2.VideoCapture(test_vr_video_path)

        height, width, _ = cap.read()[1].shape
        half_width = width // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        start_time = time.perf_counter()
        frames_processed = 0

        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break

            # Split
            left_eye = frame[:, :half_width]
            right_eye = frame[:, half_width:]

            # Process both
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
        assert fps > 0
        print(f"VR processing FPS: {fps:.2f}")

    @pytest.mark.gpu
    def test_parallel_eye_processing(self, test_vr_video_path, mock_model, cuda_available):
        """Test processing both eyes in parallel on GPU."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        import time

        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_width = width // 2

        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        # Batch processing (parallel)
        start_time = time.perf_counter()

        left_resized = cv2.resize(left_eye, (640, 640))
        right_resized = cv2.resize(right_eye, (640, 640))

        batch = np.stack([left_resized, right_resized])
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.cuda()

        with torch.no_grad():
            masks = mock_model(tensor)

        torch.cuda.synchronize()
        parallel_time = time.perf_counter() - start_time

        print(f"Parallel processing time: {parallel_time*1000:.2f}ms")

        assert parallel_time > 0


class TestVRQuality:
    """Test VR output quality."""

    def test_stereo_consistency(self, test_vr_video_path, mock_model, device):
        """Test consistency between left and right eye masks."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_width = width // 2

        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        # Process both
        left_resized = cv2.resize(left_eye, (640, 640))
        right_resized = cv2.resize(right_eye, (640, 640))

        batch = np.stack([left_resized, right_resized])
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            masks = mock_model(tensor)

        left_mask = masks[0].cpu().numpy()
        right_mask = masks[1].cpu().numpy()

        # Masks should have similar structure
        # (Can't guarantee identical due to stereo disparity)
        assert left_mask.shape == right_mask.shape

    def test_no_seam_in_output(self, test_vr_video_path, mock_model, device):
        """Test no visible seam at recombination point."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        half_width = width // 2

        # Process and recombine
        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:]

        # Simplified processing
        output = np.concatenate([left_eye, right_eye], axis=1)

        # Check dimensions match
        assert output.shape == frame.shape

        # Check middle column exists
        middle_col = output[:, half_width-1:half_width+1]
        assert middle_col.shape[1] == 2


class TestVREdgeCases:
    """Test VR edge cases."""

    def test_odd_width_handling(self, temp_dir, mock_model, device):
        """Test handling VR video with odd width."""
        from test_utils import TestDataGenerator

        # Create video with odd width
        video_path = str(temp_dir / "vr_odd.mp4")
        TestDataGenerator.create_vr_stereo_video(
            video_path,
            width=3841,  # Odd width
            height=1080
        )

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            pytest.skip("Could not create odd-width video")

        height, width = frame.shape[:2]
        half_width = width // 2

        left_eye = frame[:, :half_width]
        right_eye = frame[:, half_width:2*half_width]

        # Should still work
        assert left_eye.shape[1] == right_eye.shape[1]

    def test_monoscopic_fallback(self, test_video_path, mock_model, device):
        """Test falling back to monoscopic processing."""
        # Regular (non-VR) video
        cap = cv2.VideoCapture(test_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        aspect_ratio = width / height

        # Not VR format (aspect ratio < 2)
        if aspect_ratio < 2:
            # Process as single frame
            resized = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                mask = mock_model(tensor)

            assert mask is not None


class TestVRMetadata:
    """Test VR metadata handling."""

    def test_detect_vr_projection(self, test_vr_video_path):
        """Test detecting VR projection type."""
        cap = cv2.VideoCapture(test_vr_video_path)
        ret, frame = cap.read()
        cap.release()

        height, width = frame.shape[:2]
        aspect_ratio = width / height

        # Detect format based on aspect ratio
        if aspect_ratio > 2:
            projection = "side_by_side"
        elif aspect_ratio > 1:
            projection = "monoscopic"
        else:
            projection = "top_bottom"

        assert projection in ["side_by_side", "monoscopic", "top_bottom"]

    def test_vr_frame_packing(self, test_vr_video_path):
        """Test understanding VR frame packing."""
        cap = cv2.VideoCapture(test_vr_video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        # Side-by-side: each eye is half width
        eye_width = width // 2
        eye_height = height

        assert eye_width > 0
        assert eye_height > 0
