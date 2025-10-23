#!/usr/bin/env python3
"""
Advanced Features Example

Demonstrates advanced features including:
- Depth estimation
- Custom stereo processing
- Model ensembling
- Performance optimization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.segmentation_engine import SegmentationEngine, SegmentationEngineConfig
from models.model_loader import ModelConfig, ModelManager, ModelLoader
from preprocessing.video_decoder import VRVideoDecoder, VideoDecoderConfig
from preprocessing.frame_preprocessor import PreprocessConfig
from postprocessing.mask_processor import MaskProcessorConfig
from vr.depth_estimator import DepthEstimatorFactory, DepthEstimatorConfig
from vr.stereo_processor import StereoProcessor, StereoConsistencyConfig
import numpy as np


def example_depth_estimation():
    """Example: Using depth estimation."""
    print("\n" + "=" * 60)
    print("Example 1: Depth Estimation")
    print("=" * 60)

    # Create depth estimator
    depth_config = DepthEstimatorConfig(
        model_name='midas',
        model_type='DPT_Large',
        device='cuda',
        normalize_depth=True,
    )

    depth_estimator = DepthEstimatorFactory.create(depth_config)

    # Example frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    print("Estimating depth...")
    depth_map = depth_estimator.estimate(frame)

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")


def example_stereo_processing():
    """Example: Advanced stereo processing."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Stereo Processing")
    print("=" * 60)

    # Create stereo processor with different methods
    methods = ['average', 'weighted', 'cross_check']

    for method in methods:
        print(f"\nTesting {method} method...")

        config = StereoConsistencyConfig(
            enable=True,
            method=method,
            temporal_weight=0.3,
        )

        processor = StereoProcessor(config)

        # Example masks
        left_mask = np.random.rand(1080, 1920) > 0.5
        right_mask = np.random.rand(1080, 1920) > 0.5

        # Process
        left_consistent, right_consistent = processor.enforce_consistency(
            left_mask.astype(np.float32),
            right_mask.astype(np.float32)
        )

        # Get metrics
        metrics = processor.get_consistency_metrics(
            left_consistent,
            right_consistent
        )

        print(f"  Agreement: {metrics['agreement']:.3f}")
        print(f"  Dice: {metrics['dice']:.3f}")
        print(f"  IoU: {metrics['iou']:.3f}")


def example_model_ensemble():
    """Example: Model ensembling."""
    print("\n" + "=" * 60)
    print("Example 3: Model Ensemble")
    print("=" * 60)

    # Create model manager
    manager = ModelManager()

    # Load multiple models
    models_to_load = [
        ('deeplabv3', ModelConfig(model_name='deeplabv3', device='cuda')),
        ('bisenet', ModelConfig(model_name='bisenet', device='cuda')),
    ]

    for name, config in models_to_load:
        print(f"Loading {name}...")
        try:
            model = ModelLoader.load_model(config)
            manager.add_model(name, model)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # Get model info
    info = manager.get_model_info()
    print(f"\nLoaded models: {info['models']}")
    print(f"Active model: {info['active_model']}")

    # Example frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Ensemble prediction (if multiple models loaded)
    if len(manager.models) > 1:
        print("\nRunning ensemble prediction...")
        # This would require preprocessing first
        # ensemble_mask = manager.predict_ensemble(preprocessed_frame)
        print("Ensemble prediction ready!")


def example_performance_optimization():
    """Example: Performance optimization techniques."""
    print("\n" + "=" * 60)
    print("Example 4: Performance Optimization")
    print("=" * 60)

    # Configuration with optimizations
    optimized_config = ModelConfig(
        model_name='deeplabv3',
        device='cuda',
        use_fp16=True,           # Half precision
        use_tensorrt=False,      # TensorRT optimization (requires setup)
        use_torch_compile=True,  # PyTorch 2.0 compilation
        batch_size=4,            # Batch processing
        warmup_iterations=10,    # Warm up GPU
    )

    print("Configuration:")
    print(f"  FP16: {optimized_config.use_fp16}")
    print(f"  Torch Compile: {optimized_config.use_torch_compile}")
    print(f"  Batch Size: {optimized_config.batch_size}")
    print(f"  Warmup Iterations: {optimized_config.warmup_iterations}")

    # Create engine
    preprocess_config = PreprocessConfig(
        target_size=(1080, 1920),
        device='cuda',
        dtype='float16',
    )

    postprocess_config = MaskProcessorConfig()

    segmentation_config = SegmentationEngineConfig(
        model_config=optimized_config,
        preprocess_config=preprocess_config,
        postprocess_config=postprocess_config,
    )

    print("\nInitializing optimized engine...")
    try:
        engine = SegmentationEngine(segmentation_config)

        # Get performance stats
        stats = engine.get_performance_stats()
        print(f"\nEngine ready!")
        print(f"  Device: {stats['device']}")

    except Exception as e:
        print(f"Failed to initialize engine: {e}")


def example_custom_preprocessing():
    """Example: Custom preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Preprocessing")
    print("=" * 60)

    from preprocessing.frame_preprocessor import AugmentationPipeline

    # Create augmentation pipeline
    aug_pipeline = AugmentationPipeline(
        enable_flip=True,
        enable_rotation=True,
        enable_brightness=True,
        enable_contrast=True,
        enable_noise=True,
        max_rotation=10.0,
    )

    # Example frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    mask = np.random.rand(1080, 1920) > 0.5

    print("Applying augmentations...")
    augmented_frame, augmented_mask = aug_pipeline.augment(frame, mask.astype(np.uint8))

    print(f"Original shape: {frame.shape}")
    print(f"Augmented shape: {augmented_frame.shape}")
    print("Augmentation complete!")


def main():
    """Run all examples."""
    print("=" * 60)
    print("VR Body Segmentation - Advanced Features")
    print("=" * 60)

    try:
        # Run examples
        example_depth_estimation()
        example_stereo_processing()
        example_model_ensemble()
        example_performance_optimization()
        example_custom_preprocessing()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
