"""
Mask Processor

Postprocessing module for segmentation masks.
Includes refinement, edge smoothing, temporal filtering, and format conversion.
"""

import logging
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from collections import deque

logger = logging.getLogger(__name__)


class MaskProcessorConfig:
    """Configuration for mask postprocessing."""

    def __init__(
        self,
        enable_refinement: bool = True,
        enable_edge_smoothing: bool = True,
        enable_temporal_filtering: bool = True,
        edge_smoothing_kernel: int = 5,
        morphology_kernel_size: int = 5,
        temporal_window_size: int = 5,
        temporal_alpha: float = 0.3,
        min_object_size: int = 1000,
        hole_filling: bool = True,
        edge_dilation: int = 2,
    ):
        self.enable_refinement = enable_refinement
        self.enable_edge_smoothing = enable_edge_smoothing
        self.enable_temporal_filtering = enable_temporal_filtering
        self.edge_smoothing_kernel = edge_smoothing_kernel
        self.morphology_kernel_size = morphology_kernel_size
        self.temporal_window_size = temporal_window_size
        self.temporal_alpha = temporal_alpha
        self.min_object_size = min_object_size
        self.hole_filling = hole_filling
        self.edge_dilation = edge_dilation


class MaskProcessor:
    """
    Postprocessor for segmentation masks.
    Refines masks with various filtering and smoothing techniques.
    """

    def __init__(self, config: MaskProcessorConfig):
        self.config = config
        self.temporal_buffer: deque = deque(maxlen=config.temporal_window_size)

        # Create morphology kernels
        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel_size, config.morphology_kernel_size)
        )

        logger.info("Mask processor initialized")

    def process(
        self,
        mask: np.ndarray,
        apply_temporal: bool = True
    ) -> np.ndarray:
        """
        Complete mask processing pipeline.

        Args:
            mask: Input segmentation mask
            apply_temporal: Whether to apply temporal filtering

        Returns:
            Processed mask
        """
        processed = mask.copy()

        # Refine mask
        if self.config.enable_refinement:
            processed = self.refine_mask(processed)

        # Smooth edges
        if self.config.enable_edge_smoothing:
            processed = self.smooth_edges(processed)

        # Apply temporal filtering
        if apply_temporal and self.config.enable_temporal_filtering:
            processed = self.apply_temporal_filter(processed)

        return processed

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine mask with morphological operations.

        Args:
            mask: Input mask

        Returns:
            Refined mask
        """
        refined = mask.copy()

        # Convert to binary if needed
        if refined.dtype != np.uint8:
            refined = (refined * 255).astype(np.uint8)

        # Remove small objects
        if self.config.min_object_size > 0:
            refined = self._remove_small_objects(refined, self.config.min_object_size)

        # Fill holes
        if self.config.hole_filling:
            refined = self._fill_holes(refined)

        # Morphological operations
        # Opening: erosion followed by dilation (removes noise)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, self.morphology_kernel)

        # Closing: dilation followed by erosion (fills small gaps)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, self.morphology_kernel)

        return refined

    def smooth_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Smooth mask edges.

        Args:
            mask: Input mask

        Returns:
            Mask with smoothed edges
        """
        # Convert to float for smoothing
        mask_float = mask.astype(np.float32)
        if mask_float.max() > 1.0:
            mask_float /= 255.0

        # Apply gaussian smoothing
        smoothed = gaussian_filter(
            mask_float,
            sigma=self.config.edge_smoothing_kernel / 3.0
        )

        # Convert back
        result = (smoothed * 255).astype(np.uint8) if mask.max() > 1 else smoothed

        return result

    def apply_temporal_filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply temporal filtering across frames.

        Args:
            mask: Current frame mask

        Returns:
            Temporally filtered mask
        """
        # Add current mask to buffer
        self.temporal_buffer.append(mask.copy())

        if len(self.temporal_buffer) < 2:
            # Not enough frames yet
            return mask

        # Exponential moving average
        alpha = self.config.temporal_alpha
        filtered = mask.astype(np.float32)

        for i, prev_mask in enumerate(reversed(list(self.temporal_buffer)[:-1])):
            weight = alpha * ((1 - alpha) ** i)
            filtered += weight * prev_mask.astype(np.float32)

        # Normalize
        total_weight = 1.0 + sum(
            alpha * ((1 - alpha) ** i)
            for i in range(len(self.temporal_buffer) - 1)
        )
        filtered /= total_weight

        # Convert back to original dtype
        if mask.dtype == np.uint8:
            filtered = filtered.astype(np.uint8)

        return filtered

    def _remove_small_objects(
        self,
        mask: np.ndarray,
        min_size: int
    ) -> np.ndarray:
        """Remove small connected components."""
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Create output mask
        output = np.zeros_like(mask)

        # Keep only large components
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = 255

        return output

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in mask."""
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        # Fill contours
        filled = mask.copy()
        for contour in contours:
            cv2.drawContours(filled, [contour], 0, 255, -1)

        return filled

    def extract_edges(
        self,
        mask: np.ndarray,
        dilation: int = 0
    ) -> np.ndarray:
        """
        Extract edges from mask.

        Args:
            mask: Input mask
            dilation: Edge dilation amount

        Returns:
            Edge mask
        """
        # Convert to binary if needed
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Find edges using Canny
        edges = cv2.Canny(mask, 50, 150)

        # Dilate edges if requested
        if dilation > 0:
            kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def create_soft_mask(
        self,
        mask: np.ndarray,
        feather_radius: int = 10
    ) -> np.ndarray:
        """
        Create soft mask with feathered edges.

        Args:
            mask: Input hard mask
            feather_radius: Feathering radius in pixels

        Returns:
            Soft mask with alpha channel
        """
        # Convert to float
        mask_float = mask.astype(np.float32)
        if mask_float.max() > 1.0:
            mask_float /= 255.0

        # Apply gaussian blur for feathering
        soft_mask = cv2.GaussianBlur(
            mask_float,
            (feather_radius * 2 + 1, feather_radius * 2 + 1),
            feather_radius / 3.0
        )

        return soft_mask

    def refine_with_trimap(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        edge_width: int = 10
    ) -> np.ndarray:
        """
        Refine mask using trimap-based matting.

        Args:
            image: Original image
            mask: Initial mask
            edge_width: Width of uncertain region

        Returns:
            Refined alpha mask
        """
        # Create trimap
        trimap = np.zeros(mask.shape, dtype=np.uint8)

        # Definite foreground
        kernel = np.ones((edge_width, edge_width), np.uint8)
        sure_fg = cv2.erode(mask, kernel, iterations=1)
        trimap[sure_fg == 255] = 255

        # Definite background
        sure_bg = cv2.dilate(mask, kernel, iterations=1)
        trimap[sure_bg == 0] = 0

        # Unknown region
        trimap[(trimap != 0) & (trimap != 255)] = 128

        # For simple implementation, use GrabCut-like refinement
        # In production, could use more sophisticated matting algorithms
        refined = self._simple_matting(image, trimap)

        return refined

    def _simple_matting(
        self,
        image: np.ndarray,
        trimap: np.ndarray
    ) -> np.ndarray:
        """Simple alpha matting implementation."""
        # This is a placeholder for simple matting
        # In production, use more sophisticated algorithms like:
        # - GrabCut
        # - Deep Image Matting
        # - Background Matting

        alpha = trimap.copy().astype(np.float32) / 255.0

        # Smooth unknown regions
        unknown_mask = (trimap == 128)
        if unknown_mask.any():
            alpha[unknown_mask] = gaussian_filter(
                alpha, sigma=2.0
            )[unknown_mask]

        return alpha

    def reset_temporal_buffer(self) -> None:
        """Reset temporal filtering buffer."""
        self.temporal_buffer.clear()
        logger.info("Temporal buffer reset")


class MaskVisualizer:
    """Visualizes segmentation masks with various overlay styles."""

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay mask on image with transparency.

        Args:
            image: Original image
            mask: Segmentation mask
            color: Overlay color (RGB)
            alpha: Transparency (0-1)

        Returns:
            Image with mask overlay
        """
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask / 255.0)

        # Create colored overlay
        overlay = image.copy()
        overlay[mask > 0.5] = color

        # Blend
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return result

    @staticmethod
    def create_boundary(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw mask boundaries on image.

        Args:
            image: Original image
            mask: Segmentation mask
            color: Boundary color (RGB)
            thickness: Line thickness

        Returns:
            Image with boundaries
        """
        # Convert mask to uint8
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)

        return result

    @staticmethod
    def create_side_by_side(
        image: np.ndarray,
        mask: np.ndarray,
        segmented: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create side-by-side comparison visualization.

        Args:
            image: Original image
            mask: Segmentation mask
            segmented: Optional segmented image

        Returns:
            Side-by-side comparison
        """
        # Convert mask to 3-channel for visualization
        if len(mask.shape) == 2:
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_vis = mask

        if segmented is not None:
            # Three-way comparison
            result = np.hstack([image, mask_vis, segmented])
        else:
            # Two-way comparison
            result = np.hstack([image, mask_vis])

        return result


class MaskConverter:
    """Converts masks between different formats."""

    @staticmethod
    def to_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert to binary mask."""
        if mask.dtype == np.uint8 and mask.max() > 1:
            threshold = threshold * 255

        return (mask > threshold).astype(np.uint8) * 255

    @staticmethod
    def to_probability(mask: np.ndarray) -> np.ndarray:
        """Convert to probability map (0-1)."""
        mask_float = mask.astype(np.float32)
        if mask_float.max() > 1.0:
            mask_float /= 255.0
        return mask_float

    @staticmethod
    def to_rgb(mask: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
        """Convert mask to RGB using colormap."""
        # Normalize to 0-255
        mask_norm = mask.astype(np.float32)
        if mask_norm.max() <= 1.0:
            mask_norm *= 255
        mask_norm = mask_norm.astype(np.uint8)

        # Apply colormap
        if colormap == 'viridis':
            colored = cv2.applyColorMap(mask_norm, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'jet':
            colored = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(mask_norm, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(mask_norm, cv2.COLORMAP_VIRIDIS)

        return colored

    @staticmethod
    def to_torch(mask: np.ndarray, device: str = 'cuda') -> torch.Tensor:
        """Convert numpy mask to torch tensor."""
        tensor = torch.from_numpy(mask).to(device)

        # Add batch and channel dimensions if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy mask."""
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor

        # Remove batch dimension
        while array.ndim > 2 and array.shape[0] == 1:
            array = array.squeeze(0)

        return array
