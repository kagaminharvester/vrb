"""
Depth Estimator

Optional depth estimation for VR videos.
Provides depth information for depth-aware segmentation and effects.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import cv2

logger = logging.getLogger(__name__)


class DepthEstimatorConfig:
    """Configuration for depth estimation."""

    def __init__(
        self,
        model_name: str = 'midas',  # 'midas', 'dpt', 'stereo'
        model_type: str = 'DPT_Large',  # For MiDaS
        device: str = 'cuda',
        use_fp16: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        enable_smoothing: bool = True,
        smoothing_kernel: int = 5,
        normalize_depth: bool = True,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.use_fp16 = use_fp16
        self.target_size = target_size
        self.enable_smoothing = enable_smoothing
        self.smoothing_kernel = smoothing_kernel
        self.normalize_depth = normalize_depth


class MiDaSDepthEstimator:
    """
    Depth estimator using MiDaS (Monocular Depth Estimation).
    Provides relative depth information from single images.
    """

    def __init__(self, config: DepthEstimatorConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None

        self._load_model()

    def _load_model(self) -> None:
        """Load MiDaS model."""
        try:
            logger.info(f"Loading MiDaS model: {self.config.model_type}")

            # Load model
            model_type = self.config.model_type
            self.model = torch.hub.load('intel-isl/MiDaS', model_type)
            self.model.to(self.device)
            self.model.eval()

            # Load transform
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

            if model_type in ['DPT_Large', 'DPT_Hybrid']:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            # Optimize
            if self.config.use_fp16:
                self.model = self.model.half()

            logger.info("MiDaS model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from single image.

        Args:
            image: Input image (RGB, uint8)

        Returns:
            Depth map (normalized to 0-1)
        """
        with torch.no_grad():
            # Preprocess
            input_batch = self.transform(image).to(self.device)

            if self.config.use_fp16:
                input_batch = input_batch.half()

            # Predict
            prediction = self.model(input_batch)

            # Resize to original size
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

            # Convert to numpy
            depth = prediction.cpu().numpy()

            # Normalize
            if self.config.normalize_depth:
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # Apply smoothing
            if self.config.enable_smoothing:
                from scipy.ndimage import gaussian_filter
                depth = gaussian_filter(depth, sigma=self.config.smoothing_kernel / 3.0)

            return depth

    def estimate_stereo(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        method: str = 'average'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth for stereo pair.

        Args:
            left_image: Left eye image
            right_image: Right eye image
            method: Combination method ('average', 'left', 'right')

        Returns:
            (left_depth, right_depth)
        """
        if method == 'left':
            left_depth = self.estimate(left_image)
            right_depth = left_depth
        elif method == 'right':
            right_depth = self.estimate(right_image)
            left_depth = right_depth
        else:  # average
            left_depth = self.estimate(left_image)
            right_depth = self.estimate(right_image)

            # Average depths for consistency
            avg_depth = (left_depth + right_depth) / 2.0
            left_depth = avg_depth
            right_depth = avg_depth

        return left_depth, right_depth


class StereoDepthEstimator:
    """
    Depth estimator using stereo matching.
    Provides metric depth from stereo pairs.
    """

    def __init__(self, config: DepthEstimatorConfig):
        self.config = config
        self.stereo_matcher = None

        self._initialize_matcher()

    def _initialize_matcher(self) -> None:
        """Initialize stereo matcher."""
        logger.info("Initializing stereo depth estimator")

        # Create SGBM matcher for better quality
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        logger.info("Stereo matcher initialized")

    def estimate(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        baseline: float = 0.065,  # meters
        focal_length: float = 500.0  # pixels
    ) -> np.ndarray:
        """
        Estimate depth from stereo pair.

        Args:
            left_image: Left eye image
            right_image: Right eye image
            baseline: Stereo baseline in meters
            focal_length: Focal length in pixels

        Returns:
            Depth map in meters
        """
        # Convert to grayscale
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image

        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Convert disparity to depth
        # depth = (baseline * focal_length) / disparity
        valid_mask = disparity > 0
        depth = np.zeros_like(disparity)
        depth[valid_mask] = (baseline * focal_length) / disparity[valid_mask]

        # Clip unrealistic depths
        depth = np.clip(depth, 0.1, 100.0)

        # Normalize if requested
        if self.config.normalize_depth:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Apply smoothing
        if self.config.enable_smoothing:
            from scipy.ndimage import gaussian_filter
            depth = gaussian_filter(depth, sigma=self.config.smoothing_kernel / 3.0)

        return depth


class DepthGuidedProcessor:
    """
    Uses depth information to guide segmentation processing.
    Enables depth-aware effects and refinement.
    """

    def __init__(self, depth_estimator: Optional[Any] = None):
        self.depth_estimator = depth_estimator

    def refine_mask_with_depth(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        depth_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Refine segmentation mask using depth information.

        Args:
            mask: Segmentation mask
            depth: Depth map
            depth_threshold: Optional depth threshold for filtering

        Returns:
            Refined mask
        """
        refined_mask = mask.copy()

        if depth_threshold is not None:
            # Filter mask based on depth
            depth_mask = depth < depth_threshold
            refined_mask = np.logical_and(refined_mask, depth_mask).astype(mask.dtype)

        return refined_mask

    def create_depth_layers(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        num_layers: int = 3
    ) -> list:
        """
        Create depth-based layers for compositing.

        Args:
            image: Input image
            mask: Segmentation mask
            depth: Depth map
            num_layers: Number of depth layers

        Returns:
            List of (layer_image, layer_mask, layer_depth) tuples
        """
        # Divide depth into layers
        depth_min = depth.min()
        depth_max = depth.max()
        depth_range = depth_max - depth_min
        layer_size = depth_range / num_layers

        layers = []

        for i in range(num_layers):
            # Define depth range for this layer
            layer_min = depth_min + i * layer_size
            layer_max = depth_min + (i + 1) * layer_size

            # Create layer mask
            layer_depth_mask = np.logical_and(
                depth >= layer_min,
                depth < layer_max
            )
            layer_mask = np.logical_and(mask, layer_depth_mask).astype(np.uint8)

            # Extract layer image
            layer_image = image * layer_mask[..., np.newaxis]

            # Average depth for layer
            layer_depth = np.mean(depth[layer_depth_mask]) if layer_depth_mask.any() else 0.0

            layers.append({
                'image': layer_image,
                'mask': layer_mask,
                'depth': layer_depth,
                'depth_range': (layer_min, layer_max)
            })

        return layers

    def apply_depth_effect(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        effect_type: str = 'blur',
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Apply depth-based effect to image.

        Args:
            image: Input image
            depth: Depth map
            effect_type: Type of effect ('blur', 'fog', 'bokeh')
            intensity: Effect intensity

        Returns:
            Image with depth effect applied
        """
        if effect_type == 'blur':
            return self._apply_depth_blur(image, depth, intensity)
        elif effect_type == 'fog':
            return self._apply_depth_fog(image, depth, intensity)
        elif effect_type == 'bokeh':
            return self._apply_depth_bokeh(image, depth, intensity)
        else:
            logger.warning(f"Unknown effect type: {effect_type}")
            return image

    def _apply_depth_blur(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        intensity: float
    ) -> np.ndarray:
        """Apply depth-based blur (depth of field effect)."""
        result = image.copy()

        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Apply varying blur based on depth
        for i in range(10):
            depth_level = i / 10.0
            blur_size = int(1 + i * 2 * intensity)

            if blur_size % 2 == 0:
                blur_size += 1

            # Create mask for this depth level
            mask = np.abs(depth_norm - depth_level) < 0.1

            if mask.any() and blur_size > 1:
                blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
                result = np.where(mask[..., np.newaxis], blurred, result)

        return result

    def _apply_depth_fog(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        intensity: float
    ) -> np.ndarray:
        """Apply depth-based fog effect."""
        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Create fog
        fog_color = np.array([200, 200, 200], dtype=np.uint8)
        fog = np.ones_like(image) * fog_color

        # Blend based on depth
        alpha = depth_norm[..., np.newaxis] * intensity
        alpha = np.clip(alpha, 0, 1)

        result = (1 - alpha) * image + alpha * fog
        return result.astype(np.uint8)

    def _apply_depth_bokeh(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        intensity: float
    ) -> np.ndarray:
        """Apply depth-based bokeh effect."""
        # Similar to depth blur but with circular bokeh
        # For simplicity, using gaussian blur
        return self._apply_depth_blur(image, depth, intensity)


class DepthEstimatorFactory:
    """Factory for creating depth estimators."""

    @staticmethod
    def create(config: DepthEstimatorConfig):
        """
        Create depth estimator based on configuration.

        Args:
            config: Depth estimator configuration

        Returns:
            Depth estimator instance
        """
        model_name = config.model_name.lower()

        if 'midas' in model_name or 'dpt' in model_name:
            return MiDaSDepthEstimator(config)
        elif 'stereo' in model_name:
            return StereoDepthEstimator(config)
        else:
            raise ValueError(f"Unknown depth estimator: {config.model_name}")
