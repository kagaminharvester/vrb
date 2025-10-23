"""
Performance Profiling and Bottleneck Detection
RTX 3090 Optimization Tools

This module provides comprehensive profiling utilities for identifying
and analyzing performance bottlenecks in the segmentation pipeline.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """Single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileStats:
    """Aggregated profiling statistics."""
    name: str
    count: int = 0
    total_time_ms: float = 0.0
    mean_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    std_time_ms: float = 0.0
    percentile_50_ms: float = 0.0
    percentile_95_ms: float = 0.0
    percentile_99_ms: float = 0.0


class Timer:
    """
    High-precision timer with CUDA synchronization.

    Features:
    - Automatic CUDA synchronization for GPU operations
    - Context manager interface
    - Nested timing support
    """

    def __init__(self, name: str, cuda_sync: bool = True):
        """
        Initialize timer.

        Args:
            name: Timer name
            cuda_sync: Synchronize CUDA before timing
        """
        self.name = name
        self.cuda_sync = cuda_sync
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        """Start timer."""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer."""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.duration_ms = (end_time - self.start_time) * 1000

    def get_duration(self) -> float:
        """Get duration in milliseconds."""
        return self.duration_ms


class Profiler:
    """
    Comprehensive profiling tool for pipeline performance analysis.

    Features:
    - Hierarchical timing
    - Statistical analysis
    - Bottleneck detection
    - Export to various formats
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.

        Args:
            enabled: Enable profiling
        """
        self.enabled = enabled
        self.records: List[TimingRecord] = []
        self.active_timers: Dict[str, float] = {}
        self.stats_cache: Dict[str, ProfileStats] = {}
        self.cache_valid = False

        logger.info(f"Profiler initialized (enabled={enabled})")

    @contextmanager
    def profile(self, name: str, cuda_sync: bool = True, **metadata):
        """
        Profile code block.

        Args:
            name: Profile name
            cuda_sync: Synchronize CUDA
            **metadata: Additional metadata

        Example:
            with profiler.profile('inference'):
                output = model(input)
        """
        if not self.enabled:
            yield
            return

        # Start timing
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Stop timing
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Record timing
            record = TimingRecord(
                name=name,
                duration_ms=duration_ms,
                timestamp=start_time,
                metadata=metadata
            )
            self.records.append(record)
            self.cache_valid = False

    def start(self, name: str):
        """
        Start named timer.

        Args:
            name: Timer name
        """
        if not self.enabled:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_timers[name] = time.perf_counter()

    def stop(self, name: str, **metadata):
        """
        Stop named timer.

        Args:
            name: Timer name
            **metadata: Additional metadata
        """
        if not self.enabled or name not in self.active_timers:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        start_time = self.active_timers.pop(name)
        duration_ms = (end_time - start_time) * 1000

        record = TimingRecord(
            name=name,
            duration_ms=duration_ms,
            timestamp=start_time,
            metadata=metadata
        )
        self.records.append(record)
        self.cache_valid = False

    def compute_stats(self) -> Dict[str, ProfileStats]:
        """
        Compute aggregated statistics for all timings.

        Returns:
            Dictionary of profile statistics
        """
        if self.cache_valid:
            return self.stats_cache

        # Group records by name
        grouped = defaultdict(list)
        for record in self.records:
            grouped[record.name].append(record.duration_ms)

        # Compute statistics
        stats = {}
        for name, durations in grouped.items():
            durations_arr = np.array(durations)

            stats[name] = ProfileStats(
                name=name,
                count=len(durations),
                total_time_ms=float(np.sum(durations_arr)),
                mean_time_ms=float(np.mean(durations_arr)),
                min_time_ms=float(np.min(durations_arr)),
                max_time_ms=float(np.max(durations_arr)),
                std_time_ms=float(np.std(durations_arr)),
                percentile_50_ms=float(np.percentile(durations_arr, 50)),
                percentile_95_ms=float(np.percentile(durations_arr, 95)),
                percentile_99_ms=float(np.percentile(durations_arr, 99))
            )

        self.stats_cache = stats
        self.cache_valid = True

        return stats

    def get_bottleneck(self) -> Optional[str]:
        """
        Identify bottleneck (operation with highest total time).

        Returns:
            Name of bottleneck operation
        """
        stats = self.compute_stats()
        if not stats:
            return None

        bottleneck = max(stats.items(), key=lambda x: x[1].total_time_ms)
        return bottleneck[0]

    def print_summary(self):
        """Print profiling summary."""
        stats = self.compute_stats()

        if not stats:
            logger.info("No profiling data available")
            return

        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)

        # Sort by total time
        sorted_stats = sorted(stats.values(), key=lambda x: x.total_time_ms, reverse=True)

        print(f"\n{'Operation':<30} {'Count':>8} {'Total(ms)':>12} {'Mean(ms)':>12} {'P95(ms)':>12}")
        print("-"*80)

        for stat in sorted_stats:
            print(
                f"{stat.name:<30} {stat.count:>8} "
                f"{stat.total_time_ms:>12.2f} {stat.mean_time_ms:>12.4f} "
                f"{stat.percentile_95_ms:>12.4f}"
            )

        print("-"*80)

        # Bottleneck
        bottleneck = self.get_bottleneck()
        if bottleneck:
            print(f"\nBottleneck: {bottleneck} ({stats[bottleneck].total_time_ms:.2f} ms total)")

        print("="*80 + "\n")

    def export_json(self, filepath: str):
        """
        Export profiling data to JSON.

        Args:
            filepath: Output file path
        """
        stats = self.compute_stats()

        data = {
            'records': [
                {
                    'name': r.name,
                    'duration_ms': r.duration_ms,
                    'timestamp': r.timestamp,
                    'metadata': r.metadata
                }
                for r in self.records
            ],
            'statistics': {
                name: {
                    'count': s.count,
                    'total_time_ms': s.total_time_ms,
                    'mean_time_ms': s.mean_time_ms,
                    'min_time_ms': s.min_time_ms,
                    'max_time_ms': s.max_time_ms,
                    'std_time_ms': s.std_time_ms,
                    'percentile_50_ms': s.percentile_50_ms,
                    'percentile_95_ms': s.percentile_95_ms,
                    'percentile_99_ms': s.percentile_99_ms
                }
                for name, s in stats.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported profiling data to {filepath}")

    def plot_timeline(self, filepath: Optional[str] = None, max_records: int = 1000):
        """
        Plot timeline of operations.

        Args:
            filepath: Output file path (or None to show)
            max_records: Maximum records to plot
        """
        if not self.records:
            logger.warning("No profiling data to plot")
            return

        # Use subset if too many records
        records = self.records[-max_records:] if len(self.records) > max_records else self.records

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))

        # Get unique operation names
        unique_names = list(set(r.name for r in records))
        name_to_y = {name: i for i, name in enumerate(unique_names)}

        # Plot bars
        for record in records:
            y = name_to_y[record.name]
            ax.barh(
                y,
                record.duration_ms,
                left=record.timestamp * 1000,
                height=0.8,
                alpha=0.7
            )

        # Configure plot
        ax.set_yticks(range(len(unique_names)))
        ax.set_yticklabels(unique_names)
        ax.set_xlabel('Time (ms)')
        ax.set_title('Operation Timeline')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=150)
            logger.info(f"Saved timeline plot to {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_distribution(self, filepath: Optional[str] = None):
        """
        Plot distribution of operation times.

        Args:
            filepath: Output file path (or None to show)
        """
        stats = self.compute_stats()

        if not stats:
            logger.warning("No profiling data to plot")
            return

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Sort by total time
        sorted_stats = sorted(stats.values(), key=lambda x: x.total_time_ms, reverse=True)

        # 1. Total time comparison
        names = [s.name for s in sorted_stats[:10]]
        totals = [s.total_time_ms for s in sorted_stats[:10]]
        axes[0].barh(names, totals)
        axes[0].set_xlabel('Total Time (ms)')
        axes[0].set_title('Total Time by Operation')
        axes[0].invert_yaxis()

        # 2. Mean time comparison
        means = [s.mean_time_ms for s in sorted_stats[:10]]
        axes[1].barh(names, means)
        axes[1].set_xlabel('Mean Time (ms)')
        axes[1].set_title('Mean Time by Operation')
        axes[1].invert_yaxis()

        # 3. Count comparison
        counts = [s.count for s in sorted_stats[:10]]
        axes[2].barh(names, counts)
        axes[2].set_xlabel('Count')
        axes[2].set_title('Call Count by Operation')
        axes[2].invert_yaxis()

        # 4. Box plot of latencies
        data_for_box = []
        labels_for_box = []
        for stat in sorted_stats[:10]:
            durations = [r.duration_ms for r in self.records if r.name == stat.name]
            data_for_box.append(durations)
            labels_for_box.append(stat.name)

        axes[3].boxplot(data_for_box, labels=labels_for_box, vert=False)
        axes[3].set_xlabel('Duration (ms)')
        axes[3].set_title('Latency Distribution')

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=150)
            logger.info(f"Saved distribution plot to {filepath}")
        else:
            plt.show()

        plt.close()

    def reset(self):
        """Reset profiler state."""
        self.records.clear()
        self.active_timers.clear()
        self.stats_cache.clear()
        self.cache_valid = False
        logger.info("Profiler reset")


class CUDAProfiler:
    """
    CUDA-specific profiling using PyTorch profiler.

    Features:
    - Kernel-level profiling
    - Memory usage tracking
    - CPU-GPU overlap analysis
    """

    def __init__(self):
        """Initialize CUDA profiler."""
        self.profiler = None

    @contextmanager
    def profile(
        self,
        activities: List[str] = None,
        record_shapes: bool = True,
        with_stack: bool = True
    ):
        """
        Profile CUDA operations.

        Args:
            activities: Activities to profile (CPU, CUDA)
            record_shapes: Record tensor shapes
            with_stack: Record stack traces

        Example:
            with cuda_profiler.profile():
                output = model(input)
        """
        if activities is None:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=True
        )

        with self.profiler:
            yield self.profiler

    def export_chrome_trace(self, filepath: str):
        """
        Export trace in Chrome tracing format.

        Args:
            filepath: Output file path
        """
        if self.profiler is None:
            logger.warning("No profiling data available")
            return

        self.profiler.export_chrome_trace(filepath)
        logger.info(f"Exported Chrome trace to {filepath}")

    def print_summary(self):
        """Print profiling summary."""
        if self.profiler is None:
            logger.warning("No profiling data available")
            return

        print("\n" + "="*80)
        print("CUDA PROFILING SUMMARY")
        print("="*80)
        print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print("="*80 + "\n")


class MemoryProfiler:
    """
    GPU memory profiling and leak detection.

    Features:
    - Track allocations/deallocations
    - Detect memory leaks
    - Analyze memory fragmentation
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize memory profiler.

        Args:
            device_id: GPU device ID
        """
        self.device = torch.device(f'cuda:{device_id}')
        self.snapshots = []

    def snapshot(self, name: str):
        """
        Take memory snapshot.

        Args:
            name: Snapshot name
        """
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)

        snapshot = {
            'name': name,
            'timestamp': time.time(),
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'max_reserved': max_reserved
        }

        self.snapshots.append(snapshot)
        logger.info(
            f"Memory snapshot '{name}': "
            f"Allocated={allocated/(1024**2):.2f} MB, "
            f"Reserved={reserved/(1024**2):.2f} MB"
        )

    def compare_snapshots(self, name1: str, name2: str):
        """
        Compare two snapshots.

        Args:
            name1: First snapshot name
            name2: Second snapshot name
        """
        snap1 = next((s for s in self.snapshots if s['name'] == name1), None)
        snap2 = next((s for s in self.snapshots if s['name'] == name2), None)

        if snap1 is None or snap2 is None:
            logger.warning("Snapshot not found")
            return

        allocated_diff = snap2['allocated'] - snap1['allocated']
        reserved_diff = snap2['reserved'] - snap1['reserved']

        print(f"\nMemory difference between '{name1}' and '{name2}':")
        print(f"  Allocated: {allocated_diff/(1024**2):+.2f} MB")
        print(f"  Reserved: {reserved_diff/(1024**2):+.2f} MB")

        if allocated_diff > 1024**2:  # > 1 MB
            logger.warning(f"Potential memory leak detected: {allocated_diff/(1024**2):.2f} MB")

    def reset(self):
        """Reset memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device)
        self.snapshots.clear()


# Example usage and benchmarking
if __name__ == "__main__":
    # Test profiler
    profiler = Profiler(enabled=True)

    # Simulate operations
    for i in range(100):
        with profiler.profile('preprocess'):
            time.sleep(0.001)

        with profiler.profile('inference'):
            time.sleep(0.005)

        with profiler.profile('postprocess'):
            time.sleep(0.002)

    # Print summary
    profiler.print_summary()

    # Export data
    profiler.export_json('/tmp/profile.json')

    # Test CUDA profiler (if CUDA available)
    if torch.cuda.is_available():
        cuda_profiler = CUDAProfiler()

        model = torch.nn.Linear(100, 10).cuda()
        input_tensor = torch.randn(32, 100).cuda()

        with cuda_profiler.profile():
            for _ in range(10):
                output = model(input_tensor)
                torch.cuda.synchronize()

        cuda_profiler.print_summary()

    # Test memory profiler
    if torch.cuda.is_available():
        mem_profiler = MemoryProfiler(device_id=0)

        mem_profiler.snapshot('start')

        # Allocate memory
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(10)]

        mem_profiler.snapshot('after_alloc')

        # Free memory
        del tensors
        torch.cuda.empty_cache()

        mem_profiler.snapshot('after_free')

        mem_profiler.compare_snapshots('start', 'after_alloc')
        mem_profiler.compare_snapshots('after_alloc', 'after_free')
