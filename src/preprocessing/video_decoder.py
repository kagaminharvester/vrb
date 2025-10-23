"""
VR Video Decoder

Handles decoding of VR videos with support for multiple formats and codecs.
Manages stereo video separation and frame extraction.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any
from queue import Queue
from threading import Thread, Event
import time

from utils.video_utils import (
    VRFormat, VideoCodec, detect_vr_format, split_stereo_frame,
    get_video_info, detect_codec, validate_video_file
)

logger = logging.getLogger(__name__)


class VideoDecoderConfig:
    """Configuration for video decoder."""

    def __init__(
        self,
        video_path: str,
        vr_format: Optional[VRFormat] = None,
        buffer_size: int = 30,
        use_threading: bool = True,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_skip: int = 0,
        decode_mode: str = 'stereo',  # 'stereo', 'left', 'right', 'mono'
    ):
        self.video_path = video_path
        self.vr_format = vr_format
        self.buffer_size = buffer_size
        self.use_threading = use_threading
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_skip = frame_skip
        self.decode_mode = decode_mode


class FrameBuffer:
    """Thread-safe frame buffer for decoded frames."""

    def __init__(self, max_size: int = 30):
        self.queue = Queue(maxsize=max_size)
        self.stopped = Event()

    def put(self, item: Any, timeout: float = 1.0) -> bool:
        """Add item to buffer."""
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except:
            return False

    def get(self, timeout: float = 1.0) -> Optional[Any]:
        """Get item from buffer."""
        try:
            return self.queue.get(timeout=timeout)
        except:
            return None

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.queue.empty()

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.queue.full()

    def size(self) -> int:
        """Get current buffer size."""
        return self.queue.qsize()

    def stop(self) -> None:
        """Signal buffer to stop."""
        self.stopped.set()

    def is_stopped(self) -> bool:
        """Check if buffer is stopped."""
        return self.stopped.is_set()

    def clear(self) -> None:
        """Clear all items from buffer."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break


class VRVideoDecoder:
    """
    VR video decoder with support for stereo formats.
    Handles frame extraction and stereo separation.
    """

    def __init__(self, config: VideoDecoderConfig):
        self.config = config
        self.video_info: Optional[Dict] = None
        self.vr_format: Optional[VRFormat] = None
        self.codec: Optional[VideoCodec] = None
        self.capture: Optional[cv2.VideoCapture] = None

        # Frame buffer for threaded decoding
        self.frame_buffer: Optional[FrameBuffer] = None
        self.decode_thread: Optional[Thread] = None

        # Statistics
        self.frames_decoded = 0
        self.decode_start_time = 0.0

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize video decoder."""
        # Validate video file
        if not validate_video_file(self.config.video_path):
            raise ValueError(f"Invalid video file: {self.config.video_path}")

        # Get video info
        self.video_info = get_video_info(self.config.video_path)

        # Detect or use provided VR format
        if self.config.vr_format:
            self.vr_format = self.config.vr_format
        else:
            self.vr_format = VRFormat(self.video_info['vr_format'])

        # Detect codec
        self.codec = detect_codec(self.config.video_path)

        logger.info(f"Video decoder initialized: {self.vr_format.value}, "
                   f"codec: {self.codec.value}")

        # Open video capture
        self.capture = cv2.VideoCapture(self.config.video_path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.config.video_path}")

        # Set start frame
        if self.config.start_frame > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)

    def _decode_worker(self) -> None:
        """Worker thread for decoding frames."""
        logger.info("Decode worker thread started")
        frame_idx = self.config.start_frame
        end_frame = self.config.end_frame or self.video_info['frame_count']

        while not self.frame_buffer.is_stopped() and frame_idx < end_frame:
            # Read frame
            ret, frame = self.capture.read()

            if not ret or frame is None:
                logger.warning(f"Failed to read frame {frame_idx}")
                break

            # Apply frame skip
            if self.config.frame_skip > 0:
                for _ in range(self.config.frame_skip):
                    frame_idx += 1
                    if frame_idx >= end_frame:
                        break
                    self.capture.read()

            # Process frame based on decode mode
            if self.config.decode_mode == 'stereo':
                left_eye, right_eye = split_stereo_frame(frame, self.vr_format)
                frame_data = {
                    'frame_idx': frame_idx,
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            elif self.config.decode_mode == 'left':
                left_eye, _ = split_stereo_frame(frame, self.vr_format)
                frame_data = {
                    'frame_idx': frame_idx,
                    'frame': left_eye,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            elif self.config.decode_mode == 'right':
                _, right_eye = split_stereo_frame(frame, self.vr_format)
                frame_data = {
                    'frame_idx': frame_idx,
                    'frame': right_eye,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            else:  # mono
                frame_data = {
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }

            # Put frame in buffer
            while not self.frame_buffer.is_stopped():
                if self.frame_buffer.put(frame_data, timeout=0.1):
                    break

            frame_idx += 1
            self.frames_decoded += 1

        # Signal end of stream
        self.frame_buffer.put(None)
        logger.info(f"Decode worker thread finished ({self.frames_decoded} frames)")

    def start_threaded_decode(self) -> None:
        """Start threaded frame decoding."""
        if not self.config.use_threading:
            logger.warning("Threading not enabled in config")
            return

        if self.decode_thread is not None and self.decode_thread.is_alive():
            logger.warning("Decode thread already running")
            return

        # Initialize buffer
        self.frame_buffer = FrameBuffer(max_size=self.config.buffer_size)

        # Start decode thread
        self.decode_thread = Thread(target=self._decode_worker, daemon=True)
        self.decode_start_time = time.time()
        self.decode_thread.start()

        logger.info("Threaded decoding started")

    def read_frame(self) -> Optional[Dict]:
        """
        Read next frame from buffer or capture.

        Returns:
            Frame data dictionary or None if end of stream
        """
        if self.config.use_threading:
            # Read from buffer
            if self.frame_buffer is None:
                raise RuntimeError("Buffer not initialized. Call start_threaded_decode() first")

            frame_data = self.frame_buffer.get()
            return frame_data
        else:
            # Read directly from capture
            ret, frame = self.capture.read()

            if not ret or frame is None:
                return None

            frame_idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Process frame based on decode mode
            if self.config.decode_mode == 'stereo':
                left_eye, right_eye = split_stereo_frame(frame, self.vr_format)
                return {
                    'frame_idx': frame_idx,
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            elif self.config.decode_mode == 'left':
                left_eye, _ = split_stereo_frame(frame, self.vr_format)
                return {
                    'frame_idx': frame_idx,
                    'frame': left_eye,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            elif self.config.decode_mode == 'right':
                _, right_eye = split_stereo_frame(frame, self.vr_format)
                return {
                    'frame_idx': frame_idx,
                    'frame': right_eye,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            else:  # mono
                return {
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'timestamp': self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }

    def frames(self) -> Iterator[Dict]:
        """
        Iterator for reading frames.

        Yields:
            Frame data dictionaries
        """
        while True:
            frame_data = self.read_frame()
            if frame_data is None:
                break
            yield frame_data

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information."""
        if self.frame_buffer is None:
            return {'enabled': False}

        return {
            'enabled': True,
            'size': self.frame_buffer.size(),
            'max_size': self.config.buffer_size,
            'is_full': self.frame_buffer.is_full(),
            'is_empty': self.frame_buffer.is_empty(),
        }

    def get_decode_stats(self) -> Dict[str, Any]:
        """Get decoding statistics."""
        elapsed_time = time.time() - self.decode_start_time if self.decode_start_time > 0 else 0
        fps = self.frames_decoded / elapsed_time if elapsed_time > 0 else 0

        return {
            'frames_decoded': self.frames_decoded,
            'elapsed_time': elapsed_time,
            'decode_fps': fps,
            'buffer_status': self.get_buffer_status()
        }

    def seek(self, frame_idx: int) -> bool:
        """
        Seek to specific frame.

        Args:
            frame_idx: Target frame index

        Returns:
            True if successful
        """
        if self.capture is None:
            return False

        try:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            return True
        except Exception as e:
            logger.error(f"Seek failed: {e}")
            return False

    def stop(self) -> None:
        """Stop decoder and cleanup resources."""
        logger.info("Stopping video decoder")

        # Stop buffer
        if self.frame_buffer is not None:
            self.frame_buffer.stop()

        # Wait for decode thread
        if self.decode_thread is not None and self.decode_thread.is_alive():
            self.decode_thread.join(timeout=2.0)

        # Release capture
        if self.capture is not None:
            self.capture.release()

        logger.info("Video decoder stopped")

    def __enter__(self):
        """Context manager entry."""
        if self.config.use_threading:
            self.start_threaded_decode()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Destructor."""
        self.stop()


class MultiVideoDecoder:
    """
    Decoder for processing multiple videos simultaneously.
    Useful for batch processing or multi-camera setups.
    """

    def __init__(self, video_paths: list, config_template: VideoDecoderConfig):
        self.video_paths = video_paths
        self.config_template = config_template
        self.decoders: list[VRVideoDecoder] = []

        # Initialize decoders
        for video_path in video_paths:
            config = VideoDecoderConfig(
                video_path=video_path,
                vr_format=config_template.vr_format,
                buffer_size=config_template.buffer_size,
                use_threading=config_template.use_threading,
                decode_mode=config_template.decode_mode
            )
            decoder = VRVideoDecoder(config)
            self.decoders.append(decoder)

        logger.info(f"Multi-video decoder initialized with {len(self.decoders)} videos")

    def start_all(self) -> None:
        """Start all decoders."""
        for decoder in self.decoders:
            if decoder.config.use_threading:
                decoder.start_threaded_decode()

    def read_all(self) -> list:
        """Read frames from all decoders."""
        frames = []
        for decoder in self.decoders:
            frame_data = decoder.read_frame()
            frames.append(frame_data)
        return frames

    def stop_all(self) -> None:
        """Stop all decoders."""
        for decoder in self.decoders:
            decoder.stop()

    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all()
