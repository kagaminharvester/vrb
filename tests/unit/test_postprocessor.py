"""
Unit tests for postprocessing module.

Tests mask refinement, smoothing, and video composition.
"""

import pytest
import numpy as np
import cv2
import torch
from unittest.mock import Mock, patch


class TestMaskRefinement:
    """Test mask refinement operations."""

    def test_binary_threshold(self, test_mask):
        """Test binary thresholding of mask."""
        # Convert to probability map
        prob_mask = (test_mask / 255.0).astype(np.float32)

        # Apply threshold
        threshold = 0.5
        binary = (prob_mask > threshold).astype(np.uint8) * 255

        # Should only contain 0 and 255
        unique_values = np.unique(binary)
        assert all(v in [0, 255] for v in unique_values)

    def test_morphological_closing(self, test_mask):
        """Test morphological closing to fill holes."""
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel)

        assert closed.shape == test_mask.shape
        # Closing should not decrease white pixels
        assert closed.sum() >= test_mask.sum()

    def test_morphological_opening(self, test_mask):
        """Test morphological opening to remove noise."""
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(test_mask, cv2.MORPH_OPEN, kernel)

        assert opened.shape == test_mask.shape
        # Opening should not increase white pixels
        assert opened.sum() <= test_mask.sum()

    def test_erosion(self, test_mask):
        """Test erosion operation."""
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(test_mask, kernel, iterations=1)

        assert eroded.shape == test_mask.shape
        # Erosion should reduce white pixels
        assert eroded.sum() <= test_mask.sum()

    def test_dilation(self, test_mask):
        """Test dilation operation."""
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(test_mask, kernel, iterations=1)

        assert dilated.shape == test_mask.shape
        # Dilation should increase white pixels
        assert dilated.sum() >= test_mask.sum()

    @pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
    def test_variable_kernel_sizes(self, test_mask, kernel_size):
        """Test different kernel sizes for morphology."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel)

        assert processed.shape == test_mask.shape


class TestMaskSmoothing:
    """Test mask smoothing techniques."""

    def test_gaussian_blur(self, test_mask):
        """Test Gaussian blur smoothing."""
        blurred = cv2.GaussianBlur(test_mask, (5, 5), 0)

        assert blurred.shape == test_mask.shape
        assert blurred.dtype == test_mask.dtype

    def test_median_blur(self, test_mask):
        """Test median blur for noise removal."""
        blurred = cv2.medianBlur(test_mask, 5)

        assert blurred.shape == test_mask.shape
        # Median blur preserves edges better
        assert blurred.dtype == test_mask.dtype

    def test_bilateral_filter(self, test_mask):
        """Test bilateral filter for edge-preserving smoothing."""
        filtered = cv2.bilateralFilter(test_mask, 9, 75, 75)

        assert filtered.shape == test_mask.shape
        assert filtered.dtype == test_mask.dtype

    def test_box_filter(self, test_mask):
        """Test box filter smoothing."""
        blurred = cv2.blur(test_mask, (5, 5))

        assert blurred.shape == test_mask.shape
        assert blurred.dtype == test_mask.dtype

    @pytest.mark.parametrize("blur_size", [(3, 3), (5, 5), (7, 7), (9, 9)])
    def test_variable_blur_sizes(self, test_mask, blur_size):
        """Test different blur kernel sizes."""
        blurred = cv2.GaussianBlur(test_mask, blur_size, 0)

        assert blurred.shape == test_mask.shape


class TestEdgeDetection:
    """Test edge detection and refinement."""

    def test_canny_edge_detection(self, test_mask):
        """Test Canny edge detection."""
        edges = cv2.Canny(test_mask, 50, 150)

        assert edges.shape == test_mask.shape
        # Edges should be binary
        unique_values = np.unique(edges)
        assert all(v in [0, 255] for v in unique_values)

    def test_sobel_edge_detection(self, test_mask):
        """Test Sobel edge detection."""
        sobelx = cv2.Sobel(test_mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(test_mask, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx**2 + sobely**2)

        assert magnitude.shape == test_mask.shape

    def test_laplacian_edge_detection(self, test_mask):
        """Test Laplacian edge detection."""
        laplacian = cv2.Laplacian(test_mask, cv2.CV_64F)

        assert laplacian.shape == test_mask.shape


class TestContourProcessing:
    """Test contour detection and processing."""

    def test_find_contours(self, test_mask):
        """Test finding contours in mask."""
        contours, hierarchy = cv2.findContours(
            test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) > 0

    def test_largest_contour(self, test_mask):
        """Test finding largest contour."""
        contours, _ = cv2.findContours(
            test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            assert area > 0

    def test_contour_approximation(self, test_mask):
        """Test contour approximation."""
        contours, _ = cv2.findContours(
            test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            contour = contours[0]
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            assert len(approx) <= len(contour)

    def test_fill_contours(self, test_mask):
        """Test filling contours."""
        contours, _ = cv2.findContours(
            test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filled = np.zeros_like(test_mask)
        cv2.drawContours(filled, contours, -1, 255, -1)

        assert filled.shape == test_mask.shape


class TestTemporalSmoothing:
    """Test temporal smoothing across frames."""

    def test_running_average(self):
        """Test running average of masks."""
        num_frames = 5
        masks = [np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
                for _ in range(num_frames)]

        # Simple running average
        avg_mask = np.mean(masks, axis=0).astype(np.uint8)

        assert avg_mask.shape == masks[0].shape

    def test_exponential_moving_average(self):
        """Test exponential moving average."""
        alpha = 0.3
        current_mask = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        prev_avg = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # EMA
        ema = ((1 - alpha) * prev_avg + alpha * current_mask).astype(np.uint8)

        assert ema.shape == current_mask.shape

    def test_median_temporal_filter(self):
        """Test temporal median filter."""
        num_frames = 5
        masks = [np.random.randint(0, 256, (100, 100), dtype=np.uint8)
                for _ in range(num_frames)]

        # Stack and compute median
        stacked = np.stack(masks, axis=0)
        median_mask = np.median(stacked, axis=0).astype(np.uint8)

        assert median_mask.shape == masks[0].shape


class TestMaskResize:
    """Test mask resizing operations."""

    @pytest.mark.parametrize("target_size", [
        (640, 640),
        (1280, 720),
        (1920, 1080),
        (3840, 2160)
    ])
    def test_resize_mask(self, test_mask, target_size):
        """Test resizing mask to different sizes."""
        resized = cv2.resize(test_mask, target_size)

        assert resized.shape == (target_size[1], target_size[0])

    def test_resize_nearest_neighbor(self, test_mask):
        """Test nearest neighbor interpolation."""
        resized = cv2.resize(test_mask, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Should preserve binary values
        unique_values = np.unique(resized)
        assert all(v in [0, 255] for v in unique_values)

    def test_resize_with_aspect_ratio(self, test_mask):
        """Test resizing while maintaining aspect ratio."""
        height, width = test_mask.shape
        target_width = 640

        scale = target_width / width
        target_height = int(height * scale)

        resized = cv2.resize(test_mask, (target_width, target_height))

        assert resized.shape[1] == target_width
        assert resized.shape[0] == target_height


class TestAlphaBlending:
    """Test alpha blending operations."""

    def test_simple_alpha_blend(self, sample_frame, test_mask):
        """Test simple alpha blending."""
        # Resize mask to match frame
        mask = cv2.resize(test_mask, (sample_frame.shape[1], sample_frame.shape[0]))

        # Create alpha channel
        alpha = (mask / 255.0).astype(np.float32)

        # Create background
        background = np.zeros_like(sample_frame)

        # Blend
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        blended = (sample_frame * alpha_3ch + background * (1 - alpha_3ch)).astype(np.uint8)

        assert blended.shape == sample_frame.shape

    def test_feathered_edges(self, sample_frame, test_mask):
        """Test alpha blending with feathered edges."""
        # Resize mask
        mask = cv2.resize(test_mask, (sample_frame.shape[1], sample_frame.shape[0]))

        # Feather edges with Gaussian blur
        feathered = cv2.GaussianBlur(mask, (15, 15), 0)

        # Create alpha
        alpha = (feathered / 255.0).astype(np.float32)

        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_multi_layer_compositing(self, sample_frame, test_mask):
        """Test compositing multiple layers."""
        # Resize mask
        mask = cv2.resize(test_mask, (sample_frame.shape[1], sample_frame.shape[0]))
        alpha = (mask / 255.0).astype(np.float32)

        # Create layers
        foreground = sample_frame
        background = np.full_like(sample_frame, [0, 255, 0])  # Green background

        # Composite
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        result = (foreground * alpha_3ch + background * (1 - alpha_3ch)).astype(np.uint8)

        assert result.shape == sample_frame.shape


class TestMaskQuality:
    """Test mask quality metrics."""

    def test_mask_coverage(self, test_mask):
        """Test calculating mask coverage percentage."""
        total_pixels = test_mask.size
        masked_pixels = (test_mask > 0).sum()

        coverage = masked_pixels / total_pixels

        assert 0 <= coverage <= 1

    def test_mask_connectivity(self, test_mask):
        """Test checking mask connectivity."""
        # Find connected components
        num_labels, labels = cv2.connectedComponents(test_mask)

        # Number of components (excluding background)
        num_objects = num_labels - 1

        assert num_objects >= 0

    def test_mask_smoothness(self, test_mask):
        """Test measuring mask edge smoothness."""
        # Detect edges
        edges = cv2.Canny(test_mask, 50, 150)

        # Count edge pixels
        edge_pixels = (edges > 0).sum()

        # More edge pixels = less smooth
        assert edge_pixels >= 0


class TestColorCorrection:
    """Test color correction for composited output."""

    def test_brightness_adjustment(self, sample_frame):
        """Test brightness adjustment."""
        factor = 1.2
        brightened = np.clip(sample_frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        assert brightened.shape == sample_frame.shape
        assert brightened.dtype == np.uint8

    def test_contrast_adjustment(self, sample_frame):
        """Test contrast adjustment."""
        factor = 1.5
        mean = sample_frame.mean()
        adjusted = np.clip((sample_frame - mean) * factor + mean, 0, 255).astype(np.uint8)

        assert adjusted.shape == sample_frame.shape

    def test_gamma_correction(self, sample_frame):
        """Test gamma correction."""
        gamma = 1.5

        # Normalize to [0, 1]
        normalized = sample_frame / 255.0

        # Apply gamma
        corrected = np.power(normalized, gamma)

        # Scale back
        result = (corrected * 255).astype(np.uint8)

        assert result.shape == sample_frame.shape


class TestOutputGeneration:
    """Test final output generation."""

    def test_generate_masked_output(self, sample_frame, test_mask):
        """Test generating masked output."""
        # Resize mask
        mask = cv2.resize(test_mask, (sample_frame.shape[1], sample_frame.shape[0]))

        # Apply mask
        masked = sample_frame.copy()
        masked[mask == 0] = 0

        assert masked.shape == sample_frame.shape

    def test_generate_background_replacement(self, sample_frame, test_mask):
        """Test background replacement."""
        # Resize mask
        mask = cv2.resize(test_mask, (sample_frame.shape[1], sample_frame.shape[0]))

        # Create new background
        new_bg = np.full_like(sample_frame, [0, 255, 0])  # Green

        # Replace background
        result = sample_frame.copy()
        result[mask == 0] = new_bg[mask == 0]

        assert result.shape == sample_frame.shape

    def test_generate_alpha_matte(self, test_mask):
        """Test generating alpha matte."""
        # Smooth mask for better alpha
        smoothed = cv2.GaussianBlur(test_mask, (15, 15), 0)

        # Convert to float [0, 1]
        alpha = smoothed.astype(np.float32) / 255.0

        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0


class TestBatchPostprocessing:
    """Test batch postprocessing operations."""

    def test_batch_mask_refinement(self):
        """Test refining batch of masks."""
        batch_size = 4
        masks = [np.random.randint(0, 2, (640, 640), dtype=np.uint8) * 255
                for _ in range(batch_size)]

        # Apply morphology to all
        kernel = np.ones((5, 5), np.uint8)
        refined = [cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel) for m in masks]

        assert len(refined) == batch_size
        assert all(m.shape == (640, 640) for m in refined)

    def test_batch_temporal_smoothing(self):
        """Test temporal smoothing on batch."""
        num_frames = 10
        masks = [np.random.randint(0, 256, (100, 100), dtype=np.uint8)
                for _ in range(num_frames)]

        # Apply temporal smoothing
        window_size = 3
        smoothed = []

        for i in range(len(masks)):
            start = max(0, i - window_size // 2)
            end = min(len(masks), i + window_size // 2 + 1)

            window = masks[start:end]
            avg = np.mean(window, axis=0).astype(np.uint8)
            smoothed.append(avg)

        assert len(smoothed) == num_frames


class TestErrorHandling:
    """Test error handling in postprocessing."""

    def test_empty_mask_handling(self):
        """Test handling of empty mask."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        # Should not crash
        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.morphologyEx(empty_mask, cv2.MORPH_CLOSE, kernel)

        assert processed.shape == empty_mask.shape
        assert processed.sum() == 0

    def test_full_mask_handling(self):
        """Test handling of full mask (all ones)."""
        full_mask = np.ones((100, 100), dtype=np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

        assert processed.shape == full_mask.shape

    def test_mismatched_size_handling(self, sample_frame, test_mask):
        """Test handling of mismatched frame/mask sizes."""
        # Mask is different size than frame
        assert test_mask.shape != sample_frame.shape[:2]

        # Resize to match
        resized_mask = cv2.resize(test_mask, (sample_frame.shape[1], sample_frame.shape[0]))

        assert resized_mask.shape == sample_frame.shape[:2]
