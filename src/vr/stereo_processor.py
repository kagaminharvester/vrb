"""
Stereo Processor

Handles stereoscopic video processing with stereo consistency enforcement.
Ensures segmentation masks are consistent across left and right eye views.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class StereoConsistencyConfig:
    """Configuration for stereo consistency enforcement."""

    def __init__(
        self,
        enable: bool = True,
        method: str = 'average',  # 'average', 'weighted', 'cross_check', 'optical_flow'
        weight_left: float = 0.5,
        weight_right: float = 0.5,
        consistency_threshold: float = 0.8,
        smoothing_kernel: int = 5,
        temporal_weight: float = 0.3,
    ):
        self.enable = enable
        self.method = method
        self.weight_left = weight_left
        self.weight_right = weight_right
        self.consistency_threshold = consistency_threshold
        self.smoothing_kernel = smoothing_kernel
        self.temporal_weight = temporal_weight


class StereoProcessor:
    """
    Processes stereo VR video frames with consistency enforcement.
    Ensures left and right eye segmentation masks are consistent.
    """

    def __init__(self, config: StereoConsistencyConfig):
        self.config = config
        self.prev_left_mask: Optional[np.ndarray] = None
        self.prev_right_mask: Optional[np.ndarray] = None

        logger.info(f"Stereo processor initialized: method={config.method}")

    def enforce_consistency(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        disparity_map: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce consistency between left and right eye masks.

        Args:
            left_mask: Left eye segmentation mask
            right_mask: Right eye segmentation mask
            disparity_map: Optional disparity map for better alignment

        Returns:
            (consistent_left_mask, consistent_right_mask)
        """
        if not self.config.enable:
            return left_mask, right_mask

        method = self.config.method

        if method == 'average':
            return self._enforce_average(left_mask, right_mask)
        elif method == 'weighted':
            return self._enforce_weighted(left_mask, right_mask)
        elif method == 'cross_check':
            return self._enforce_cross_check(left_mask, right_mask, disparity_map)
        elif method == 'optical_flow':
            return self._enforce_optical_flow(left_mask, right_mask)
        else:
            logger.warning(f"Unknown consistency method: {method}")
            return left_mask, right_mask

    def _enforce_average(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Average-based consistency enforcement.
        Simply averages the two masks.
        """
        # Convert to float for averaging
        left_float = left_mask.astype(np.float32)
        right_float = right_mask.astype(np.float32)

        # Average masks
        avg_mask = (left_float + right_float) / 2.0

        # Apply smoothing
        if self.config.smoothing_kernel > 0:
            avg_mask = gaussian_filter(
                avg_mask,
                sigma=self.config.smoothing_kernel / 3.0
            )

        # Convert back to original type
        consistent_left = avg_mask.astype(left_mask.dtype)
        consistent_right = avg_mask.astype(right_mask.dtype)

        return consistent_left, consistent_right

    def _enforce_weighted(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted consistency enforcement.
        Uses configurable weights for each eye.
        """
        left_float = left_mask.astype(np.float32)
        right_float = right_mask.astype(np.float32)

        # Weighted average
        w_left = self.config.weight_left
        w_right = self.config.weight_right
        weighted_mask = w_left * left_float + w_right * right_float

        # Normalize weights
        weighted_mask = weighted_mask / (w_left + w_right)

        # Apply smoothing
        if self.config.smoothing_kernel > 0:
            weighted_mask = gaussian_filter(
                weighted_mask,
                sigma=self.config.smoothing_kernel / 3.0
            )

        consistent_left = weighted_mask.astype(left_mask.dtype)
        consistent_right = weighted_mask.astype(right_mask.dtype)

        return consistent_left, consistent_right

    def _enforce_cross_check(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        disparity_map: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-check consistency enforcement.
        Only keeps consistent predictions between views.
        """
        # Calculate agreement between masks
        agreement = (left_mask == right_mask).astype(np.float32)

        # Calculate consistency score
        consistency_score = agreement.mean()

        if consistency_score < self.config.consistency_threshold:
            logger.warning(f"Low stereo consistency: {consistency_score:.3f}")

        # Create consistent mask (keep only agreeing regions)
        consistent_mask = np.where(
            agreement > 0.5,
            (left_mask.astype(np.float32) + right_mask.astype(np.float32)) / 2.0,
            0
        ).astype(left_mask.dtype)

        # Apply smoothing
        if self.config.smoothing_kernel > 0:
            consistent_mask = gaussian_filter(
                consistent_mask.astype(np.float32),
                sigma=self.config.smoothing_kernel / 3.0
            ).astype(left_mask.dtype)

        return consistent_mask, consistent_mask

    def _enforce_optical_flow(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optical flow-based consistency enforcement.
        Uses optical flow to warp masks for consistency.
        """
        import cv2

        # Convert masks to uint8 for optical flow
        left_uint8 = (left_mask * 255).astype(np.uint8)
        right_uint8 = (right_mask * 255).astype(np.uint8)

        # Compute optical flow from left to right
        try:
            flow = cv2.calcOpticalFlowFarneback(
                left_uint8, right_uint8,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Warp left mask to right view
            h, w = left_mask.shape[:2]
            flow_map = np.column_stack([
                (np.arange(w) + flow[..., 0]).flatten(),
                (np.arange(h).reshape(-1, 1) + flow[..., 1]).flatten()
            ])

            # Remap left mask
            warped_left = cv2.remap(
                left_uint8, flow[..., 0].astype(np.float32),
                flow[..., 1].astype(np.float32),
                cv2.INTER_LINEAR
            )

            # Average warped left and right
            consistent = ((warped_left.astype(np.float32) +
                          right_uint8.astype(np.float32)) / 2.0).astype(np.uint8)

            # Convert back to original range
            consistent_left = (consistent / 255.0).astype(left_mask.dtype)
            consistent_right = (consistent / 255.0).astype(right_mask.dtype)

            return consistent_left, consistent_right

        except Exception as e:
            logger.error(f"Optical flow failed: {e}, falling back to average")
            return self._enforce_average(left_mask, right_mask)

    def apply_temporal_consistency(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal consistency with previous frames.

        Args:
            left_mask: Current left eye mask
            right_mask: Current right eye mask

        Returns:
            Temporally smoothed masks
        """
        if self.prev_left_mask is None or self.prev_right_mask is None:
            # First frame, store and return
            self.prev_left_mask = left_mask.copy()
            self.prev_right_mask = right_mask.copy()
            return left_mask, right_mask

        # Temporal smoothing
        alpha = 1.0 - self.config.temporal_weight

        left_smoothed = (
            alpha * left_mask.astype(np.float32) +
            self.config.temporal_weight * self.prev_left_mask.astype(np.float32)
        ).astype(left_mask.dtype)

        right_smoothed = (
            alpha * right_mask.astype(np.float32) +
            self.config.temporal_weight * self.prev_right_mask.astype(np.float32)
        ).astype(right_mask.dtype)

        # Update previous masks
        self.prev_left_mask = left_smoothed.copy()
        self.prev_right_mask = right_smoothed.copy()

        return left_smoothed, right_smoothed

    def process_stereo_pair(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        apply_temporal: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete stereo processing pipeline.

        Args:
            left_mask: Left eye segmentation mask
            right_mask: Right eye segmentation mask
            apply_temporal: Whether to apply temporal consistency

        Returns:
            Processed (left_mask, right_mask)
        """
        # Enforce spatial consistency
        left_consistent, right_consistent = self.enforce_consistency(
            left_mask, right_mask
        )

        # Apply temporal consistency
        if apply_temporal and self.config.temporal_weight > 0:
            left_final, right_final = self.apply_temporal_consistency(
                left_consistent, right_consistent
            )
        else:
            left_final, right_final = left_consistent, right_consistent

        return left_final, right_final

    def reset_temporal_state(self) -> None:
        """Reset temporal consistency state."""
        self.prev_left_mask = None
        self.prev_right_mask = None
        logger.info("Temporal state reset")

    def get_consistency_metrics(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate consistency metrics between stereo masks.

        Args:
            left_mask: Left eye mask
            right_mask: Right eye mask

        Returns:
            Dictionary of metrics
        """
        # Agreement ratio
        agreement = (left_mask == right_mask).astype(np.float32).mean()

        # Dice coefficient
        intersection = np.logical_and(left_mask > 0.5, right_mask > 0.5).sum()
        union = np.logical_or(left_mask > 0.5, right_mask > 0.5).sum()
        dice = 2.0 * intersection / (union + intersection + 1e-8)

        # Mean absolute difference
        mad = np.abs(left_mask.astype(np.float32) - right_mask.astype(np.float32)).mean()

        # IoU
        iou = intersection / (union + 1e-8)

        return {
            'agreement': float(agreement),
            'dice': float(dice),
            'mad': float(mad),
            'iou': float(iou)
        }


class DisparityEstimator:
    """
    Estimates disparity between stereo views.
    Can be used for depth-aware processing.
    """

    def __init__(
        self,
        method: str = 'sgbm',  # 'bm', 'sgbm'
        num_disparities: int = 128,
        block_size: int = 11
    ):
        self.method = method
        self.num_disparities = num_disparities
        self.block_size = block_size

        # Initialize stereo matcher
        if method == 'bm':
            self.matcher = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
        elif method == 'sgbm':
            self.matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
        else:
            raise ValueError(f"Unknown stereo method: {method}")

        logger.info(f"Disparity estimator initialized: {method}")

    def estimate(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> np.ndarray:
        """
        Estimate disparity map from stereo pair.

        Args:
            left_image: Left eye image
            right_image: Right eye image

        Returns:
            Disparity map
        """
        import cv2

        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image

        # Compute disparity
        disparity = self.matcher.compute(left_gray, right_gray)

        # Normalize
        disparity = cv2.normalize(
            disparity, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return disparity


class StereoGeometryCorrector:
    """
    Corrects geometric distortions in stereo VR video.
    Handles lens distortion and stereo rectification.
    """

    def __init__(
        self,
        camera_matrix_left: Optional[np.ndarray] = None,
        camera_matrix_right: Optional[np.ndarray] = None,
        dist_coeffs_left: Optional[np.ndarray] = None,
        dist_coeffs_right: Optional[np.ndarray] = None,
    ):
        self.camera_matrix_left = camera_matrix_left
        self.camera_matrix_right = camera_matrix_right
        self.dist_coeffs_left = dist_coeffs_left
        self.dist_coeffs_right = dist_coeffs_right

        self.has_calibration = (
            camera_matrix_left is not None and
            camera_matrix_right is not None
        )

        if self.has_calibration:
            logger.info("Stereo geometry corrector initialized with calibration")
        else:
            logger.info("Stereo geometry corrector initialized without calibration")

    def undistort(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove lens distortion from stereo images.

        Args:
            left_image: Left eye image
            right_image: Right eye image

        Returns:
            Undistorted (left_image, right_image)
        """
        if not self.has_calibration:
            logger.warning("No calibration data, returning original images")
            return left_image, right_image

        import cv2

        # Undistort left
        left_undistorted = cv2.undistort(
            left_image,
            self.camera_matrix_left,
            self.dist_coeffs_left
        )

        # Undistort right
        right_undistorted = cv2.undistort(
            right_image,
            self.camera_matrix_right,
            self.dist_coeffs_right
        )

        return left_undistorted, right_undistorted
