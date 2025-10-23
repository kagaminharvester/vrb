#!/usr/bin/env python3
"""
Optimization utilities for VR Body Segmentation application.

Provides automatic optimization, bottleneck detection, and performance tuning.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.cuda
    import psutil
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not installed. Some optimization features disabled.")
    TORCH_AVAILABLE = False

from src.utils.logger import setup_logger, LoggerConfig
from src.utils.config_manager import ConfigManager, AppConfig


class BottleneckDetector:
    """Detect performance bottlenecks in the pipeline."""

    def __init__(self, logger):
        self.logger = logger

    def check_gpu_utilization(self) -> Dict[str, Any]:
        """
        Check GPU utilization and identify issues.

        Returns:
            GPU utilization analysis
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'available': False}

        try:
            # Get GPU stats
            gpu_util = torch.cuda.utilization()
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            gpu_memory_reserved = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory * 100

            analysis = {
                'available': True,
                'utilization': gpu_util,
                'memory_allocated_percent': gpu_memory,
                'memory_reserved_percent': gpu_memory_reserved,
                'bottlenecks': []
            }

            # Identify bottlenecks
            if gpu_util < 50:
                analysis['bottlenecks'].append({
                    'type': 'low_gpu_utilization',
                    'severity': 'high',
                    'message': f'GPU utilization is low ({gpu_util}%). Consider increasing batch size.',
                    'recommendations': [
                        'Increase batch size',
                        'Reduce CPU preprocessing overhead',
                        'Check for CPU-GPU transfer bottlenecks'
                    ]
                })

            if gpu_memory_reserved > 90:
                analysis['bottlenecks'].append({
                    'type': 'high_memory_usage',
                    'severity': 'medium',
                    'message': f'GPU memory usage is high ({gpu_memory_reserved:.1f}%). Risk of OOM.',
                    'recommendations': [
                        'Reduce batch size',
                        'Enable gradient checkpointing',
                        'Use mixed precision training'
                    ]
                })

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to check GPU utilization: {e}")
            return {'available': True, 'error': str(e)}

    def check_cpu_bottleneck(self) -> Dict[str, Any]:
        """
        Check for CPU bottlenecks.

        Returns:
            CPU bottleneck analysis
        """
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()

        analysis = {
            'cpu_utilization': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'bottlenecks': []
        }

        if cpu_percent > 90:
            analysis['bottlenecks'].append({
                'type': 'high_cpu_usage',
                'severity': 'high',
                'message': f'CPU usage is very high ({cpu_percent}%).',
                'recommendations': [
                    'Reduce number of preprocessing workers',
                    'Optimize preprocessing pipeline',
                    'Move more computation to GPU'
                ]
            })

        if memory.percent > 90:
            analysis['bottlenecks'].append({
                'type': 'high_memory_usage',
                'severity': 'high',
                'message': f'System memory usage is high ({memory.percent}%).',
                'recommendations': [
                    'Reduce queue sizes',
                    'Reduce batch size',
                    'Enable disk caching'
                ]
            })

        return analysis

    def detect_all_bottlenecks(self) -> Dict[str, Any]:
        """
        Run comprehensive bottleneck detection.

        Returns:
            Complete bottleneck analysis
        """
        self.logger.info("Running bottleneck detection...")

        analysis = {
            'gpu': self.check_gpu_utilization(),
            'cpu': self.check_cpu_bottleneck(),
            'summary': []
        }

        # Collect all bottlenecks
        all_bottlenecks = []
        if 'bottlenecks' in analysis['gpu']:
            all_bottlenecks.extend(analysis['gpu']['bottlenecks'])
        if 'bottlenecks' in analysis['cpu']:
            all_bottlenecks.extend(analysis['cpu']['bottlenecks'])

        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_bottlenecks.sort(key=lambda x: severity_order.get(x['severity'], 3))

        analysis['summary'] = all_bottlenecks

        return analysis


class PerformanceOptimizer:
    """Automatic performance optimization."""

    def __init__(self, config_manager: ConfigManager, logger):
        self.config_manager = config_manager
        self.logger = logger
        self.detector = BottleneckDetector(logger)

    def optimize_batch_size(self, config: AppConfig, target_memory_percent: float = 80.0) -> int:
        """
        Automatically optimize batch size based on available VRAM.

        Args:
            config: Current configuration
            target_memory_percent: Target VRAM utilization

        Returns:
            Optimized batch size
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using default batch size")
            return config.model.batch_size

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        target_memory_gb = gpu_memory_gb * (target_memory_percent / 100.0)

        # Estimate memory per sample (rough heuristic)
        input_h, input_w = config.model.input_size
        memory_per_sample_gb = (input_h * input_w * 3 * 4) / (1024**3) * 3  # 3x for activations

        # Calculate optimal batch size
        optimal_batch_size = int(target_memory_gb / memory_per_sample_gb)
        optimal_batch_size = max(1, min(32, optimal_batch_size))  # Clamp between 1 and 32

        self.logger.info(f"Optimized batch size: {optimal_batch_size} (GPU: {gpu_memory_gb:.1f}GB)")

        return optimal_batch_size

    def optimize_num_workers(self, config: AppConfig) -> int:
        """
        Optimize number of data loader workers.

        Args:
            config: Current configuration

        Returns:
            Optimized number of workers
        """
        cpu_count = psutil.cpu_count()

        # Use 25% of CPUs for data loading, but at least 2 and at most 8
        optimal_workers = max(2, min(8, cpu_count // 4))

        self.logger.info(f"Optimized num_workers: {optimal_workers} (CPUs: {cpu_count})")

        return optimal_workers

    def optimize_precision(self, config: AppConfig) -> str:
        """
        Determine optimal precision mode.

        Args:
            config: Current configuration

        Returns:
            Optimal precision mode
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return "fp32"

        gpu_name = torch.cuda.get_device_name(0)

        # Modern GPUs benefit from FP16
        if any(x in gpu_name.upper() for x in ['RTX', 'A100', 'A40', 'V100', 'T4']):
            optimal_precision = "fp16"
            self.logger.info(f"Using FP16 precision for {gpu_name}")
        else:
            optimal_precision = "fp32"
            self.logger.info(f"Using FP32 precision for {gpu_name}")

        return optimal_precision

    def auto_optimize(self, config: AppConfig) -> AppConfig:
        """
        Automatically optimize all configuration parameters.

        Args:
            config: Current configuration

        Returns:
            Optimized configuration
        """
        self.logger.info("Running automatic optimization...")

        # Detect bottlenecks first
        bottlenecks = self.detector.detect_all_bottlenecks()

        # Optimize parameters
        config.model.batch_size = self.optimize_batch_size(config)
        config.model.num_workers = self.optimize_num_workers(config)
        config.gpu.precision = self.optimize_precision(config)

        # Adjust based on bottlenecks
        for bottleneck in bottlenecks['summary']:
            if bottleneck['type'] == 'low_gpu_utilization':
                # Increase batch size
                config.model.batch_size = min(32, config.model.batch_size * 2)
                self.logger.info(f"Increased batch size to {config.model.batch_size} due to low GPU utilization")

            elif bottleneck['type'] == 'high_memory_usage':
                # Reduce batch size
                config.model.batch_size = max(1, config.model.batch_size // 2)
                self.logger.info(f"Reduced batch size to {config.model.batch_size} due to high memory usage")

            elif bottleneck['type'] == 'high_cpu_usage':
                # Reduce workers
                config.model.num_workers = max(1, config.model.num_workers // 2)
                self.logger.info(f"Reduced num_workers to {config.model.num_workers} due to high CPU usage")

        self.logger.info("Optimization complete")

        return config

    def benchmark_config(self, config: AppConfig, duration_sec: float = 5.0) -> Dict[str, Any]:
        """
        Quick benchmark of a configuration.

        Args:
            config: Configuration to benchmark
            duration_sec: Benchmark duration in seconds

        Returns:
            Benchmark results
        """
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}

        self.logger.info(f"Benchmarking configuration for {duration_sec}s...")

        # Create dummy model and data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_h, input_w = config.model.input_size

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, 3, padding=1),
        ).to(device)

        if config.gpu.precision == "fp16":
            model = model.half()
            dtype = torch.float16
        else:
            dtype = torch.float32

        model.eval()

        # Benchmark
        iterations = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            while time.perf_counter() - start_time < duration_sec:
                dummy_input = torch.randn(config.model.batch_size, 3, input_h, input_w,
                                         dtype=dtype, device=device)
                _ = model(dummy_input)
                iterations += 1

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        fps = iterations / elapsed

        results = {
            'fps': fps,
            'iterations': iterations,
            'duration_sec': elapsed,
            'batch_size': config.model.batch_size,
            'precision': config.gpu.precision
        }

        self.logger.info(f"Benchmark: {fps:.2f} FPS")

        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="VR Body Segmentation Optimization Tool")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--profile", type=str, choices=['fast', 'balanced', 'quality'],
                       help="Load profile")
    parser.add_argument("--detect-bottlenecks", action="store_true",
                       help="Detect performance bottlenecks")
    parser.add_argument("--auto-optimize", action="store_true",
                       help="Automatically optimize configuration")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark current configuration")
    parser.add_argument("--save-config", type=str,
                       help="Save optimized configuration to file")
    parser.add_argument("--output-dir", type=str, default="./optimization_results",
                       help="Output directory")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "optimizer",
        LoggerConfig(log_dir=str(output_dir / "logs"))
    )

    config_manager = ConfigManager()

    # Load configuration
    if args.profile:
        config = config_manager.load_profile(args.profile)
        logger.info(f"Loaded profile: {args.profile}")
    elif args.config:
        config = config_manager.load_config(args.config)
        logger.info(f"Loaded config: {args.config}")
    else:
        config = config_manager.load_config()
        logger.info("Loaded default configuration")

    # Bottleneck detection
    if args.detect_bottlenecks:
        detector = BottleneckDetector(logger)
        bottlenecks = detector.detect_all_bottlenecks()

        logger.info("\n" + "="*60)
        logger.info("BOTTLENECK DETECTION RESULTS")
        logger.info("="*60)

        if bottlenecks['summary']:
            for i, bottleneck in enumerate(bottlenecks['summary'], 1):
                logger.warning(f"\n{i}. {bottleneck['type'].upper()} (Severity: {bottleneck['severity']})")
                logger.warning(f"   {bottleneck['message']}")
                logger.info("   Recommendations:")
                for rec in bottleneck['recommendations']:
                    logger.info(f"     - {rec}")
        else:
            logger.info("No significant bottlenecks detected!")

        # Save results
        with open(output_dir / "bottleneck_analysis.json", 'w') as f:
            json.dump(bottlenecks, f, indent=2)
        logger.info(f"\nResults saved to: {output_dir / 'bottleneck_analysis.json'}")

    # Auto-optimization
    if args.auto_optimize:
        optimizer = PerformanceOptimizer(config_manager, logger)
        optimized_config = optimizer.auto_optimize(config)

        logger.info("\n" + "="*60)
        logger.info("OPTIMIZED CONFIGURATION")
        logger.info("="*60)
        logger.info(f"Batch Size: {optimized_config.model.batch_size}")
        logger.info(f"Num Workers: {optimized_config.model.num_workers}")
        logger.info(f"Precision: {optimized_config.gpu.precision}")

        config = optimized_config

        if args.save_config:
            config_manager.save_config(config, args.save_config)
            logger.info(f"\nOptimized config saved to: {args.save_config}")

    # Benchmark
    if args.benchmark:
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available. Cannot run benchmark.")
        else:
            optimizer = PerformanceOptimizer(config_manager, logger)
            results = optimizer.benchmark_config(config)

            logger.info("\n" + "="*60)
            logger.info("BENCHMARK RESULTS")
            logger.info("="*60)
            logger.info(f"FPS: {results['fps']:.2f}")
            logger.info(f"Iterations: {results['iterations']}")
            logger.info(f"Duration: {results['duration_sec']:.2f}s")

            with open(output_dir / "benchmark_results.json", 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
