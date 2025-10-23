"""
VR Body Segmentation Application

Core application for body segmentation in VR videos.
"""

__version__ = "1.0.0"
__author__ = "VR Body Segmentation Team"

from core.video_pipeline import VRVideoSegmentationPipeline, PipelineConfig, BatchVideoPipeline
from core.segmentation_engine import SegmentationEngine, SegmentationEngineConfig
from models.model_loader import ModelLoader, ModelConfig, ModelManager
from preprocessing.video_decoder import VRVideoDecoder, VideoDecoderConfig
from preprocessing.frame_preprocessor import FramePreprocessor, PreprocessConfig
from postprocessing.mask_processor import MaskProcessor, MaskProcessorConfig
from postprocessing.video_encoder import VideoEncoder, VideoEncoderConfig
from vr.stereo_processor import StereoProcessor, StereoConsistencyConfig
from vr.depth_estimator import DepthEstimatorFactory, DepthEstimatorConfig
from utils.video_utils import VRFormat, VideoCodec

__all__ = [
    # Main Pipeline
    'VRVideoSegmentationPipeline',
    'PipelineConfig',
    'BatchVideoPipeline',

    # Segmentation
    'SegmentationEngine',
    'SegmentationEngineConfig',

    # Models
    'ModelLoader',
    'ModelConfig',
    'ModelManager',

    # Preprocessing
    'VRVideoDecoder',
    'VideoDecoderConfig',
    'FramePreprocessor',
    'PreprocessConfig',

    # Postprocessing
    'MaskProcessor',
    'MaskProcessorConfig',
    'VideoEncoder',
    'VideoEncoderConfig',

    # VR Specific
    'StereoProcessor',
    'StereoConsistencyConfig',
    'DepthEstimatorFactory',
    'DepthEstimatorConfig',

    # Utils
    'VRFormat',
    'VideoCodec',
]
