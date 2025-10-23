#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for VR Body Segmentation application.

Tests various configurations, resolutions, batch sizes, and precision modes
to identify optimal settings for the target hardware.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Warning: PyTorch not installed. Some benchmarks will be skipped.")
    torch = None

from src.utils.logger import setup_logger, LoggerConfig
from src.utils.config_manager import ConfigManager, AppConfig


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    config: Dict[str, Any]
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    gpu_utilization: float
    gpu_memory_mb: float
    gpu_memory_percent: float
    cpu_utilization: float
    throughput_frames: int
    total_time_s: float
    success: bool
    error_message: Optional[str] = None


class GPUBenchmark:
    """GPU performance benchmarking."""

    def __init__(self, logger):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_dummy_model(self, input_size: Tuple[int, int], complexity: str = "medium") -> nn.Module:
        """Create a dummy model for benchmarking."""
        if complexity == "simple":
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1),
            )
        elif complexity == "medium":
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 1, 3, padding=1),
            )
        else:  # complex
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 1, 3, padding=1),
            )

        return model.to(self.device)

    def benchmark_inference(self,
                          resolution: Tuple[int, int],
                          batch_size: int,
                          precision: str = "fp32",
                          num_iterations: int = 100,
                          warmup_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark model inference.

        Args:
            resolution: Input resolution (height, width)
            batch_size: Batch size
            precision: Precision mode (fp32, fp16)
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations

        Returns:
            Benchmark results
        """
        self.logger.info(f"Benchmarking: {resolution[0]}x{resolution[1]}, batch={batch_size}, precision={precision}")

        # Create model
        model = self.create_dummy_model((resolution[0], resolution[1]), "medium")
        model.eval()

        # Set precision
        if precision == "fp16":
            model = model.half()
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, resolution[0], resolution[1],
                                  dtype=dtype, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        gpu_memory_usage = []

        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency = time.perf_counter() - start_time
                latencies.append(latency)

                if torch.cuda.is_available():
                    gpu_memory_usage.append(torch.cuda.memory_allocated() / (1024**2))

        # Calculate statistics
        latencies = np.array(latencies)
        fps = 1.0 / latencies

        results = {
            'avg_fps': float(np.mean(fps)),
            'min_fps': float(np.min(fps)),
            'max_fps': float(np.max(fps)),
            'avg_latency_ms': float(np.mean(latencies) * 1000),
            'min_latency_ms': float(np.min(latencies) * 1000),
            'max_latency_ms': float(np.max(latencies) * 1000),
            'std_latency_ms': float(np.std(latencies) * 1000),
            'iterations': num_iterations,
        }

        if torch.cuda.is_available():
            results['avg_gpu_memory_mb'] = float(np.mean(gpu_memory_usage))
            results['max_gpu_memory_mb'] = float(np.max(gpu_memory_usage))
            results['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)

        return results


class BenchmarkSuite:
    """Comprehensive benchmark suite."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            "benchmark",
            LoggerConfig(
                log_dir=str(self.output_dir / "logs"),
                console_level=20,  # INFO
                file_level=10,  # DEBUG
            )
        )

        self.results: List[BenchmarkResult] = []

    def run_resolution_benchmarks(self, batch_size: int = 8, precision: str = "fp16"):
        """
        Benchmark different resolutions.

        Args:
            batch_size: Batch size to use
            precision: Precision mode
        """
        resolutions = [
            (1080, 1920, "1080p"),
            (2160, 3840, "4K"),
            (3240, 5760, "6K"),
            (4320, 7680, "8K"),
        ]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Resolution Benchmark (batch_size={batch_size}, precision={precision})")
        self.logger.info(f"{'='*60}")

        gpu_bench = GPUBenchmark(self.logger)

        for height, width, name in resolutions:
            try:
                results = gpu_bench.benchmark_inference(
                    resolution=(height, width),
                    batch_size=batch_size,
                    precision=precision,
                    num_iterations=50,
                    warmup_iterations=10
                )

                benchmark_result = BenchmarkResult(
                    name=f"resolution_{name}",
                    config={
                        'resolution': f"{width}x{height}",
                        'batch_size': batch_size,
                        'precision': precision
                    },
                    avg_fps=results['avg_fps'],
                    min_fps=results['min_fps'],
                    max_fps=results['max_fps'],
                    avg_latency_ms=results['avg_latency_ms'],
                    min_latency_ms=results['min_latency_ms'],
                    max_latency_ms=results['max_latency_ms'],
                    gpu_utilization=0.0,  # Not measured in simple benchmark
                    gpu_memory_mb=results.get('avg_gpu_memory_mb', 0),
                    gpu_memory_percent=results.get('avg_gpu_memory_mb', 0) / results.get('gpu_memory_total_mb', 1) * 100,
                    cpu_utilization=0.0,
                    throughput_frames=results['iterations'],
                    total_time_s=results['iterations'] / results['avg_fps'],
                    success=True
                )

                self.results.append(benchmark_result)
                self._print_result(benchmark_result)

            except Exception as e:
                self.logger.error(f"Failed to benchmark {name}: {e}")
                benchmark_result = BenchmarkResult(
                    name=f"resolution_{name}",
                    config={'resolution': f"{width}x{height}", 'batch_size': batch_size},
                    avg_fps=0, min_fps=0, max_fps=0,
                    avg_latency_ms=0, min_latency_ms=0, max_latency_ms=0,
                    gpu_utilization=0, gpu_memory_mb=0, gpu_memory_percent=0,
                    cpu_utilization=0, throughput_frames=0, total_time_s=0,
                    success=False,
                    error_message=str(e)
                )
                self.results.append(benchmark_result)

    def run_batch_size_benchmarks(self, resolution: Tuple[int, int] = (1080, 1920), precision: str = "fp16"):
        """
        Benchmark different batch sizes.

        Args:
            resolution: Input resolution
            precision: Precision mode
        """
        batch_sizes = [1, 4, 8, 16, 32]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch Size Benchmark (resolution={resolution[1]}x{resolution[0]}, precision={precision})")
        self.logger.info(f"{'='*60}")

        gpu_bench = GPUBenchmark(self.logger)

        for batch_size in batch_sizes:
            try:
                results = gpu_bench.benchmark_inference(
                    resolution=resolution,
                    batch_size=batch_size,
                    precision=precision,
                    num_iterations=50,
                    warmup_iterations=10
                )

                benchmark_result = BenchmarkResult(
                    name=f"batch_size_{batch_size}",
                    config={
                        'resolution': f"{resolution[1]}x{resolution[0]}",
                        'batch_size': batch_size,
                        'precision': precision
                    },
                    avg_fps=results['avg_fps'],
                    min_fps=results['min_fps'],
                    max_fps=results['max_fps'],
                    avg_latency_ms=results['avg_latency_ms'],
                    min_latency_ms=results['min_latency_ms'],
                    max_latency_ms=results['max_latency_ms'],
                    gpu_utilization=0.0,
                    gpu_memory_mb=results.get('avg_gpu_memory_mb', 0),
                    gpu_memory_percent=results.get('avg_gpu_memory_mb', 0) / results.get('gpu_memory_total_mb', 1) * 100,
                    cpu_utilization=0.0,
                    throughput_frames=results['iterations'] * batch_size,
                    total_time_s=results['iterations'] / results['avg_fps'],
                    success=True
                )

                self.results.append(benchmark_result)
                self._print_result(benchmark_result)

            except Exception as e:
                self.logger.error(f"Failed to benchmark batch_size={batch_size}: {e}")
                benchmark_result = BenchmarkResult(
                    name=f"batch_size_{batch_size}",
                    config={'batch_size': batch_size},
                    avg_fps=0, min_fps=0, max_fps=0,
                    avg_latency_ms=0, min_latency_ms=0, max_latency_ms=0,
                    gpu_utilization=0, gpu_memory_mb=0, gpu_memory_percent=0,
                    cpu_utilization=0, throughput_frames=0, total_time_s=0,
                    success=False,
                    error_message=str(e)
                )
                self.results.append(benchmark_result)

    def run_precision_benchmarks(self, resolution: Tuple[int, int] = (1080, 1920), batch_size: int = 8):
        """
        Benchmark different precision modes.

        Args:
            resolution: Input resolution
            batch_size: Batch size
        """
        precisions = ["fp32", "fp16"]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Precision Benchmark (resolution={resolution[1]}x{resolution[0]}, batch_size={batch_size})")
        self.logger.info(f"{'='*60}")

        gpu_bench = GPUBenchmark(self.logger)

        for precision in precisions:
            try:
                results = gpu_bench.benchmark_inference(
                    resolution=resolution,
                    batch_size=batch_size,
                    precision=precision,
                    num_iterations=50,
                    warmup_iterations=10
                )

                benchmark_result = BenchmarkResult(
                    name=f"precision_{precision}",
                    config={
                        'resolution': f"{resolution[1]}x{resolution[0]}",
                        'batch_size': batch_size,
                        'precision': precision
                    },
                    avg_fps=results['avg_fps'],
                    min_fps=results['min_fps'],
                    max_fps=results['max_fps'],
                    avg_latency_ms=results['avg_latency_ms'],
                    min_latency_ms=results['min_latency_ms'],
                    max_latency_ms=results['max_latency_ms'],
                    gpu_utilization=0.0,
                    gpu_memory_mb=results.get('avg_gpu_memory_mb', 0),
                    gpu_memory_percent=results.get('avg_gpu_memory_mb', 0) / results.get('gpu_memory_total_mb', 1) * 100,
                    cpu_utilization=0.0,
                    throughput_frames=results['iterations'],
                    total_time_s=results['iterations'] / results['avg_fps'],
                    success=True
                )

                self.results.append(benchmark_result)
                self._print_result(benchmark_result)

            except Exception as e:
                self.logger.error(f"Failed to benchmark precision={precision}: {e}")
                benchmark_result = BenchmarkResult(
                    name=f"precision_{precision}",
                    config={'precision': precision},
                    avg_fps=0, min_fps=0, max_fps=0,
                    avg_latency_ms=0, min_latency_ms=0, max_latency_ms=0,
                    gpu_utilization=0, gpu_memory_mb=0, gpu_memory_percent=0,
                    cpu_utilization=0, throughput_frames=0, total_time_s=0,
                    success=False,
                    error_message=str(e)
                )
                self.results.append(benchmark_result)

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        self.logger.info(f"\n{result.name}:")
        self.logger.info(f"  FPS: {result.avg_fps:.2f} (min: {result.min_fps:.2f}, max: {result.max_fps:.2f})")
        self.logger.info(f"  Latency: {result.avg_latency_ms:.2f}ms (min: {result.min_latency_ms:.2f}ms, max: {result.max_latency_ms:.2f}ms)")
        self.logger.info(f"  GPU Memory: {result.gpu_memory_mb:.2f}MB ({result.gpu_memory_percent:.1f}%)")
        if result.success:
            self.logger.info(f"  Status: SUCCESS")
        else:
            self.logger.error(f"  Status: FAILED - {result.error_message}")

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_file = self.output_dir / filename

        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'hardware': self._get_hardware_info(),
            'results': [asdict(r) for r in self.results]
        }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"\nResults saved to: {output_file}")

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        import platform
        import psutil

        hw_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
        }

        if torch and torch.cuda.is_available():
            hw_info['cuda_available'] = True
            hw_info['gpu_count'] = torch.cuda.device_count()
            hw_info['gpu_name'] = torch.cuda.get_device_name(0)
            hw_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hw_info['cuda_version'] = torch.version.cuda
            hw_info['cudnn_version'] = torch.backends.cudnn.version()
        else:
            hw_info['cuda_available'] = False

        return hw_info

    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        self.logger.info("Starting comprehensive benchmark suite...")

        # Resolution benchmarks
        self.run_resolution_benchmarks(batch_size=8, precision="fp16")

        # Batch size benchmarks
        self.run_batch_size_benchmarks(resolution=(1080, 1920), precision="fp16")

        # Precision benchmarks
        self.run_precision_benchmarks(resolution=(1080, 1920), batch_size=8)

        # Save results
        self.save_results()

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print benchmark summary."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info(f"{'='*60}")

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        self.logger.info(f"Total benchmarks: {len(self.results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")

        if successful:
            best_fps = max(successful, key=lambda x: x.avg_fps)
            lowest_latency = min(successful, key=lambda x: x.avg_latency_ms)

            self.logger.info(f"\nBest FPS: {best_fps.name} ({best_fps.avg_fps:.2f} FPS)")
            self.logger.info(f"Lowest Latency: {lowest_latency.name} ({lowest_latency.avg_latency_ms:.2f}ms)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="VR Body Segmentation Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--resolution-only", action="store_true",
                       help="Only run resolution benchmarks")
    parser.add_argument("--batch-only", action="store_true",
                       help="Only run batch size benchmarks")
    parser.add_argument("--precision-only", action="store_true",
                       help="Only run precision benchmarks")

    args = parser.parse_args()

    if torch is None:
        print("Error: PyTorch not installed. Please install PyTorch to run benchmarks.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Benchmarks may not be accurate.")

    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)

    # Run benchmarks
    if args.resolution_only:
        suite.run_resolution_benchmarks()
        suite.save_results("resolution_benchmark.json")
    elif args.batch_only:
        suite.run_batch_size_benchmarks()
        suite.save_results("batch_size_benchmark.json")
    elif args.precision_only:
        suite.run_precision_benchmarks()
        suite.save_results("precision_benchmark.json")
    else:
        suite.run_all_benchmarks()


if __name__ == "__main__":
    main()
