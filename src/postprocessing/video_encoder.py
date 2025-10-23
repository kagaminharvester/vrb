"""
Video Encoder

Handles encoding of processed VR videos with segmentation masks.
Supports multiple output formats and codecs.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import subprocess
import json

from utils.video_utils import VRFormat, merge_stereo_frames, create_video_writer

logger = logging.getLogger(__name__)


class VideoEncoderConfig:
    """Configuration for video encoding."""

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        codec: str = 'h264',  # 'h264', 'h265', 'vp9', 'av1'
        quality: str = 'high',  # 'low', 'medium', 'high', 'lossless'
        bitrate: Optional[str] = None,  # e.g., '10M'
        preset: str = 'medium',  # 'ultrafast', 'fast', 'medium', 'slow', 'veryslow'
        pixel_format: str = 'yuv420p',
        hardware_acceleration: bool = True,
        vr_format: Optional[VRFormat] = None,
        preserve_metadata: bool = True,
        include_alpha: bool = False,
    ):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.bitrate = bitrate
        self.preset = preset
        self.pixel_format = pixel_format
        self.hardware_acceleration = hardware_acceleration
        self.vr_format = vr_format
        self.preserve_metadata = preserve_metadata
        self.include_alpha = include_alpha


class VideoEncoder:
    """
    Encodes processed video frames to output file.
    Supports various codecs and quality settings.
    """

    def __init__(self, config: VideoEncoderConfig, frame_size: Tuple[int, int]):
        self.config = config
        self.frame_size = frame_size  # (width, height)
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.use_ffmpeg = self._should_use_ffmpeg()

        self._initialize_encoder()

    def _should_use_ffmpeg(self) -> bool:
        """Determine if we should use ffmpeg instead of OpenCV."""
        # Use ffmpeg for:
        # - H.265/HEVC encoding
        # - VP9 encoding
        # - AV1 encoding
        # - Hardware acceleration
        # - High quality presets
        advanced_codecs = ['h265', 'hevc', 'vp9', 'av1']
        return (
            self.config.codec.lower() in advanced_codecs or
            self.config.hardware_acceleration or
            self.config.quality == 'lossless'
        )

    def _initialize_encoder(self) -> None:
        """Initialize video encoder."""
        if self.use_ffmpeg:
            logger.info(f"Using ffmpeg for encoding: {self.config.codec}")
            self._initialize_ffmpeg()
        else:
            logger.info(f"Using OpenCV for encoding: {self.config.codec}")
            self._initialize_opencv()

    def _initialize_opencv(self) -> None:
        """Initialize OpenCV video writer."""
        # Map codec names to fourcc codes
        codec_map = {
            'h264': 'mp4v',
            'h265': 'mp4v',
            'mjpeg': 'MJPG',
            'xvid': 'XVID',
        }

        fourcc = codec_map.get(self.config.codec.lower(), 'mp4v')

        self.writer = create_video_writer(
            self.config.output_path,
            self.config.fps,
            self.frame_size,
            codec=fourcc,
            is_color=True
        )

    def _initialize_ffmpeg(self) -> None:
        """Initialize ffmpeg pipe for encoding."""
        # Build ffmpeg command
        cmd = self._build_ffmpeg_command()

        logger.info(f"FFmpeg command: {' '.join(cmd)}")

        # Start ffmpeg process
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("FFmpeg process started")
        except Exception as e:
            logger.error(f"Failed to start ffmpeg: {e}")
            raise

    def _build_ffmpeg_command(self) -> list:
        """Build ffmpeg command for encoding."""
        width, height = self.frame_size

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.config.fps),
            '-i', '-',  # Input from stdin
        ]

        # Codec selection
        codec = self.config.codec.lower()
        if codec == 'h264':
            if self.config.hardware_acceleration:
                cmd.extend(['-c:v', 'h264_nvenc'])  # NVIDIA hardware encoding
            else:
                cmd.extend(['-c:v', 'libx264'])
        elif codec in ['h265', 'hevc']:
            if self.config.hardware_acceleration:
                cmd.extend(['-c:v', 'hevc_nvenc'])
            else:
                cmd.extend(['-c:v', 'libx265'])
        elif codec == 'vp9':
            cmd.extend(['-c:v', 'libvpx-vp9'])
        elif codec == 'av1':
            cmd.extend(['-c:v', 'libaom-av1'])
        else:
            cmd.extend(['-c:v', 'libx264'])

        # Preset
        if codec in ['h264', 'h265', 'hevc']:
            cmd.extend(['-preset', self.config.preset])

        # Quality settings
        if self.config.bitrate:
            cmd.extend(['-b:v', self.config.bitrate])
        else:
            # Use CRF for quality-based encoding
            quality_map = {
                'lossless': '0',
                'high': '18',
                'medium': '23',
                'low': '28'
            }
            crf = quality_map.get(self.config.quality, '23')

            if codec in ['h264', 'h265', 'hevc']:
                cmd.extend(['-crf', crf])
            elif codec == 'vp9':
                cmd.extend(['-crf', crf, '-b:v', '0'])

        # Pixel format
        cmd.extend(['-pix_fmt', self.config.pixel_format])

        # Output
        cmd.append(self.config.output_path)

        return cmd

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the output video.

        Args:
            frame: Frame to write (BGR format)

        Returns:
            True if successful
        """
        try:
            if self.use_ffmpeg:
                # Write to ffmpeg stdin
                self.ffmpeg_process.stdin.write(frame.tobytes())
            else:
                # Write using OpenCV
                self.writer.write(frame)

            self.frame_count += 1
            return True

        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            return False

    def write_stereo_frame(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> bool:
        """
        Write stereo frame pair.

        Args:
            left_frame: Left eye frame
            right_frame: Right eye frame

        Returns:
            True if successful
        """
        if self.config.vr_format is None:
            logger.error("VR format not specified for stereo encoding")
            return False

        # Merge stereo frames
        merged_frame = merge_stereo_frames(
            left_frame, right_frame, self.config.vr_format
        )

        return self.write_frame(merged_frame)

    def finalize(self) -> None:
        """Finalize encoding and close writer."""
        logger.info(f"Finalizing encoder ({self.frame_count} frames written)")

        if self.use_ffmpeg:
            # Close ffmpeg stdin
            if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait()

                # Check for errors
                _, stderr = self.ffmpeg_process.communicate()
                if self.ffmpeg_process.returncode != 0:
                    logger.error(f"FFmpeg error: {stderr.decode()}")
        else:
            # Release OpenCV writer
            if self.writer:
                self.writer.release()

        logger.info(f"Encoding complete: {self.config.output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()

    def __del__(self):
        """Destructor."""
        self.finalize()


class MaskVideoEncoder:
    """
    Specialized encoder for mask videos.
    Supports various mask visualization modes.
    """

    def __init__(
        self,
        config: VideoEncoderConfig,
        frame_size: Tuple[int, int],
        visualization_mode: str = 'binary'  # 'binary', 'alpha', 'overlay', 'side_by_side'
    ):
        self.config = config
        self.frame_size = frame_size
        self.visualization_mode = visualization_mode
        self.encoder = VideoEncoder(config, frame_size)

        logger.info(f"Mask video encoder initialized: {visualization_mode}")

    def write_mask_frame(
        self,
        original_frame: np.ndarray,
        mask: np.ndarray
    ) -> bool:
        """
        Write mask frame with visualization.

        Args:
            original_frame: Original video frame
            mask: Segmentation mask

        Returns:
            True if successful
        """
        # Create visualization based on mode
        if self.visualization_mode == 'binary':
            output_frame = self._create_binary_vis(mask)
        elif self.visualization_mode == 'alpha':
            output_frame = self._create_alpha_vis(original_frame, mask)
        elif self.visualization_mode == 'overlay':
            output_frame = self._create_overlay_vis(original_frame, mask)
        elif self.visualization_mode == 'side_by_side':
            output_frame = self._create_side_by_side_vis(original_frame, mask)
        else:
            logger.warning(f"Unknown visualization mode: {self.visualization_mode}")
            output_frame = original_frame

        return self.encoder.write_frame(output_frame)

    def _create_binary_vis(self, mask: np.ndarray) -> np.ndarray:
        """Create binary mask visualization."""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Convert to 3-channel
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return mask

    def _create_alpha_vis(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Create alpha-masked visualization (foreground only)."""
        if mask.max() > 1:
            mask = mask / 255.0

        # Apply mask
        result = frame * mask[..., np.newaxis]
        return result.astype(np.uint8)

    def _create_overlay_vis(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create overlay visualization."""
        from postprocessing.mask_processor import MaskVisualizer
        return MaskVisualizer.overlay_mask(frame, mask, color, alpha)

    def _create_side_by_side_vis(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Create side-by-side visualization."""
        from postprocessing.mask_processor import MaskVisualizer
        return MaskVisualizer.create_side_by_side(frame, mask)

    def finalize(self) -> None:
        """Finalize encoding."""
        self.encoder.finalize()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


class MultiOutputEncoder:
    """
    Encoder that writes to multiple outputs simultaneously.
    Useful for creating multiple versions at once.
    """

    def __init__(self, encoders: list):
        self.encoders = encoders

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write frame to all encoders."""
        results = [encoder.write_frame(frame) for encoder in self.encoders]
        return all(results)

    def write_stereo_frame(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> bool:
        """Write stereo frame to all encoders."""
        results = [
            encoder.write_stereo_frame(left_frame, right_frame)
            for encoder in self.encoders
        ]
        return all(results)

    def finalize(self) -> None:
        """Finalize all encoders."""
        for encoder in self.encoders:
            encoder.finalize()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


class VideoMetadataWriter:
    """Writes metadata for encoded videos."""

    @staticmethod
    def write_metadata(
        video_path: str,
        metadata: Dict[str, Any],
        metadata_path: Optional[str] = None
    ) -> None:
        """
        Write metadata to JSON file.

        Args:
            video_path: Path to video file
            metadata: Metadata dictionary
            metadata_path: Optional custom metadata path
        """
        if metadata_path is None:
            # Create metadata path from video path
            video_path_obj = Path(video_path)
            metadata_path = str(video_path_obj.with_suffix('.json'))

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata written to: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")

    @staticmethod
    def read_metadata(metadata_path: str) -> Optional[Dict[str, Any]]:
        """
        Read metadata from JSON file.

        Args:
            metadata_path: Path to metadata file

        Returns:
            Metadata dictionary or None if failed
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            logger.info(f"Metadata read from: {metadata_path}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            return None
