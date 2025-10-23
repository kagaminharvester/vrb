"""
Video Utility Functions

Common utilities for video processing, format detection, and conversion.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class VRFormat(Enum):
    """VR video format types."""
    SIDE_BY_SIDE = "side_by_side"
    OVER_UNDER = "over_under"
    EQUIRECTANGULAR = "equirectangular"
    MONO = "mono"
    UNKNOWN = "unknown"


class VideoCodec(Enum):
    """Supported video codecs."""
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"
    UNKNOWN = "unknown"


def detect_vr_format(
    width: int,
    height: int,
    metadata: Optional[Dict] = None
) -> VRFormat:
    """
    Detect VR video format based on dimensions and metadata.

    Args:
        width: Video width
        height: Video height
        metadata: Optional metadata dictionary

    Returns:
        Detected VR format
    """
    aspect_ratio = width / height

    # Check metadata first
    if metadata:
        format_hint = metadata.get('vr_format', '').lower()
        if 'side' in format_hint or 'sbs' in format_hint:
            return VRFormat.SIDE_BY_SIDE
        elif 'over' in format_hint or 'ou' in format_hint or 'tb' in format_hint:
            return VRFormat.OVER_UNDER
        elif 'equirect' in format_hint or '360' in format_hint:
            return VRFormat.EQUIRECTANGULAR

    # Detect based on aspect ratio
    if 3.5 < aspect_ratio < 4.5:  # ~4:1 for side-by-side
        logger.info(f"Detected side-by-side format (aspect ratio: {aspect_ratio:.2f})")
        return VRFormat.SIDE_BY_SIDE
    elif 0.9 < aspect_ratio < 1.1:  # ~1:1 for over-under
        logger.info(f"Detected over-under format (aspect ratio: {aspect_ratio:.2f})")
        return VRFormat.OVER_UNDER
    elif 1.9 < aspect_ratio < 2.1:  # ~2:1 for equirectangular
        logger.info(f"Detected equirectangular format (aspect ratio: {aspect_ratio:.2f})")
        return VRFormat.EQUIRECTANGULAR
    elif 1.7 < aspect_ratio < 1.8:  # ~16:9 for mono
        logger.info(f"Detected mono format (aspect ratio: {aspect_ratio:.2f})")
        return VRFormat.MONO
    else:
        logger.warning(f"Unknown format (aspect ratio: {aspect_ratio:.2f})")
        return VRFormat.UNKNOWN


def split_stereo_frame(
    frame: np.ndarray,
    vr_format: VRFormat
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split stereo frame into left and right eye views.

    Args:
        frame: Input stereo frame
        vr_format: VR format type

    Returns:
        Tuple of (left_eye, right_eye) frames
    """
    h, w = frame.shape[:2]

    if vr_format == VRFormat.SIDE_BY_SIDE:
        mid = w // 2
        left_eye = frame[:, :mid]
        right_eye = frame[:, mid:]
    elif vr_format == VRFormat.OVER_UNDER:
        mid = h // 2
        left_eye = frame[:mid, :]
        right_eye = frame[mid:, :]
    elif vr_format == VRFormat.EQUIRECTANGULAR:
        # For equirectangular, left and right are typically side-by-side
        mid = w // 2
        left_eye = frame[:, :mid]
        right_eye = frame[:, mid:]
    else:
        # Return same frame for both eyes if mono or unknown
        left_eye = frame
        right_eye = frame

    return left_eye, right_eye


def merge_stereo_frames(
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    vr_format: VRFormat
) -> np.ndarray:
    """
    Merge left and right eye frames back into stereo format.

    Args:
        left_eye: Left eye frame
        right_eye: Right eye frame
        vr_format: VR format type

    Returns:
        Combined stereo frame
    """
    if vr_format == VRFormat.SIDE_BY_SIDE:
        return np.hstack([left_eye, right_eye])
    elif vr_format == VRFormat.OVER_UNDER:
        return np.vstack([left_eye, right_eye])
    elif vr_format == VRFormat.EQUIRECTANGULAR:
        return np.hstack([left_eye, right_eye])
    else:
        return left_eye


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract video information using OpenCV and ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    info = {}

    try:
        # OpenCV info
        cap = cv2.VideoCapture(video_path)
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['fourcc'] = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        # Derived info
        info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
        info['aspect_ratio'] = info['width'] / info['height']
        info['vr_format'] = detect_vr_format(info['width'], info['height']).value

        logger.info(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, "
                   f"{info['frame_count']} frames, format: {info['vr_format']}")

    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise

    return info


def detect_codec(video_path: str) -> VideoCodec:
    """
    Detect video codec.

    Args:
        video_path: Path to video file

    Returns:
        Detected codec
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        # Convert fourcc to string
        codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        codec_str = codec_str.lower()

        if 'h264' in codec_str or 'avc' in codec_str:
            return VideoCodec.H264
        elif 'h265' in codec_str or 'hevc' in codec_str:
            return VideoCodec.H265
        elif 'vp9' in codec_str:
            return VideoCodec.VP9
        elif 'av1' in codec_str or 'av01' in codec_str:
            return VideoCodec.AV1
        else:
            logger.warning(f"Unknown codec: {codec_str}")
            return VideoCodec.UNKNOWN

    except Exception as e:
        logger.error(f"Failed to detect codec: {e}")
        return VideoCodec.UNKNOWN


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect: bool = True,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize frame to target size.

    Args:
        frame: Input frame
        target_size: Target (height, width)
        maintain_aspect: Whether to maintain aspect ratio
        interpolation: Interpolation method

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    target_h, target_w = target_size

    if maintain_aspect:
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        # Pad to target size
        if new_h < target_h or new_w < target_w:
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
    else:
        resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

    return resized


def convert_color_space(
    frame: np.ndarray,
    src_space: str = 'BGR',
    dst_space: str = 'RGB'
) -> np.ndarray:
    """
    Convert frame color space.

    Args:
        frame: Input frame
        src_space: Source color space (BGR, RGB, GRAY)
        dst_space: Destination color space

    Returns:
        Converted frame
    """
    conversion_map = {
        ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
        ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
        ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
        ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
        ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
        ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
    }

    key = (src_space.upper(), dst_space.upper())
    if key in conversion_map:
        return cv2.cvtColor(frame, conversion_map[key])
    else:
        logger.warning(f"No conversion for {src_space} -> {dst_space}, returning original")
        return frame


def normalize_frame(
    frame: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize frame for model input.

    Args:
        frame: Input frame (0-255)
        mean: Mean values for normalization
        std: Standard deviation values

    Returns:
        Normalized frame
    """
    # Convert to float and scale to [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # Normalize
    frame = (frame - np.array(mean)) / np.array(std)

    return frame


def denormalize_frame(
    frame: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize frame back to [0, 255].

    Args:
        frame: Normalized frame
        mean: Mean values used for normalization
        std: Standard deviation values

    Returns:
        Denormalized frame
    """
    # Denormalize
    frame = frame * np.array(std) + np.array(mean)

    # Scale to [0, 255]
    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)

    return frame


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = 'mp4v',
    is_color: bool = True
) -> cv2.VideoWriter:
    """
    Create OpenCV video writer.

    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: Frame size (width, height)
        codec: FourCC codec code
        is_color: Whether video is color

    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, is_color)

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    logger.info(f"Created video writer: {output_path} ({frame_size[0]}x{frame_size[1]} @ {fps}fps)")
    return writer


def get_optimal_thread_count(
    video_info: Dict[str, Any],
    available_threads: int = 128
) -> int:
    """
    Calculate optimal thread count for video processing.

    Args:
        video_info: Video information dictionary
        available_threads: Available CPU threads

    Returns:
        Optimal thread count
    """
    # Base calculation on resolution
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    pixels = width * height

    # Resolution-based thread allocation
    if pixels >= 7680 * 4320:  # 8K
        optimal = min(64, available_threads)
    elif pixels >= 5760 * 2880:  # 6K
        optimal = min(48, available_threads)
    elif pixels >= 3840 * 2160:  # 4K
        optimal = min(32, available_threads)
    elif pixels >= 2560 * 1440:  # 2K
        optimal = min(16, available_threads)
    else:  # 1080p or lower
        optimal = min(8, available_threads)

    logger.info(f"Optimal thread count: {optimal} (resolution: {width}x{height})")
    return optimal


def estimate_processing_time(
    video_info: Dict[str, Any],
    fps_processing: float
) -> float:
    """
    Estimate total processing time.

    Args:
        video_info: Video information dictionary
        fps_processing: Processing speed in FPS

    Returns:
        Estimated time in seconds
    """
    total_frames = video_info.get('frame_count', 0)
    if fps_processing <= 0:
        return float('inf')

    estimated_time = total_frames / fps_processing
    logger.info(f"Estimated processing time: {estimated_time:.2f}s "
               f"({total_frames} frames @ {fps_processing:.2f} fps)")
    return estimated_time


def validate_video_file(video_path: str) -> bool:
    """
    Validate video file can be opened and read.

    Args:
        video_path: Path to video file

    Returns:
        True if valid, False otherwise
    """
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return False

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        # Try to read first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error(f"Cannot read frames from video: {video_path}")
            return False

        logger.info(f"Video file validated: {video_path}")
        return True

    except Exception as e:
        logger.error(f"Video validation failed: {e}")
        return False
