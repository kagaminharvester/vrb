#!/usr/bin/env python3
"""
Performance analysis and visualization tools for VR Body Segmentation.

Analyzes benchmark results and generates performance reports with graphs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Plotting features disabled.")
    PLOTTING_AVAILABLE = False


class PerformanceAnalyzer:
    """Analyzes and visualizes performance benchmark results."""

    def __init__(self, results_file: str, output_dir: Optional[str] = None):
        """
        Initialize performance analyzer.

        Args:
            results_file: Path to benchmark results JSON
            output_dir: Output directory for reports and plots
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir) if output_dir else self.results_file.parent / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)

        self.results = self.data.get('results', [])
        self.hardware = self.data.get('hardware', {})

        # Setup plotting style
        if PLOTTING_AVAILABLE:
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 6)

    def analyze_resolution_impact(self) -> Dict[str, Any]:
        """
        Analyze impact of resolution on performance.

        Returns:
            Analysis results
        """
        resolution_results = [r for r in self.results if r['name'].startswith('resolution_')]

        if not resolution_results:
            return {'error': 'No resolution benchmarks found'}

        resolutions = []
        fps_values = []
        latency_values = []
        memory_values = []

        for result in resolution_results:
            if result['success']:
                resolutions.append(result['config']['resolution'])
                fps_values.append(result['avg_fps'])
                latency_values.append(result['avg_latency_ms'])
                memory_values.append(result['gpu_memory_mb'])

        analysis = {
            'resolutions': resolutions,
            'fps': fps_values,
            'latency_ms': latency_values,
            'gpu_memory_mb': memory_values,
            'fps_range': [min(fps_values), max(fps_values)],
            'latency_range': [min(latency_values), max(latency_values)],
            'memory_range': [min(memory_values), max(memory_values)],
        }

        # Calculate scaling efficiency
        if len(fps_values) >= 2:
            baseline_fps = fps_values[0]
            scaling_efficiency = [(fps / baseline_fps) for fps in fps_values]
            analysis['scaling_efficiency'] = scaling_efficiency

        return analysis

    def analyze_batch_size_impact(self) -> Dict[str, Any]:
        """
        Analyze impact of batch size on performance.

        Returns:
            Analysis results
        """
        batch_results = [r for r in self.results if r['name'].startswith('batch_size_')]

        if not batch_results:
            return {'error': 'No batch size benchmarks found'}

        batch_sizes = []
        fps_values = []
        latency_values = []
        memory_values = []

        for result in batch_results:
            if result['success']:
                batch_sizes.append(result['config']['batch_size'])
                fps_values.append(result['avg_fps'])
                latency_values.append(result['avg_latency_ms'])
                memory_values.append(result['gpu_memory_mb'])

        analysis = {
            'batch_sizes': batch_sizes,
            'fps': fps_values,
            'latency_ms': latency_values,
            'gpu_memory_mb': memory_values,
            'optimal_batch_size': batch_sizes[fps_values.index(max(fps_values))],
            'fps_range': [min(fps_values), max(fps_values)],
        }

        # Calculate throughput (frames per second * batch size)
        throughput = [fps * bs for fps, bs in zip(fps_values, batch_sizes)]
        analysis['throughput'] = throughput
        analysis['optimal_throughput_batch'] = batch_sizes[throughput.index(max(throughput))]

        return analysis

    def analyze_precision_impact(self) -> Dict[str, Any]:
        """
        Analyze impact of precision on performance.

        Returns:
            Analysis results
        """
        precision_results = [r for r in self.results if r['name'].startswith('precision_')]

        if not precision_results:
            return {'error': 'No precision benchmarks found'}

        precisions = []
        fps_values = []
        latency_values = []
        memory_values = []

        for result in precision_results:
            if result['success']:
                precisions.append(result['config']['precision'])
                fps_values.append(result['avg_fps'])
                latency_values.append(result['avg_latency_ms'])
                memory_values.append(result['gpu_memory_mb'])

        analysis = {
            'precisions': precisions,
            'fps': fps_values,
            'latency_ms': latency_values,
            'gpu_memory_mb': memory_values,
        }

        # Calculate speedup from FP32 to FP16
        if 'fp32' in precisions and 'fp16' in precisions:
            fp32_idx = precisions.index('fp32')
            fp16_idx = precisions.index('fp16')
            speedup = fps_values[fp16_idx] / fps_values[fp32_idx]
            memory_reduction = (memory_values[fp32_idx] - memory_values[fp16_idx]) / memory_values[fp32_idx] * 100
            analysis['fp16_speedup'] = speedup
            analysis['fp16_memory_reduction_percent'] = memory_reduction

        return analysis

    def generate_plots(self):
        """Generate performance visualization plots."""
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return

        # Resolution impact plot
        self._plot_resolution_impact()

        # Batch size impact plot
        self._plot_batch_size_impact()

        # Precision impact plot
        self._plot_precision_impact()

        # Overall summary plot
        self._plot_summary()

        print(f"Plots saved to: {self.output_dir}")

    def _plot_resolution_impact(self):
        """Plot resolution impact on performance."""
        analysis = self.analyze_resolution_impact()

        if 'error' in analysis:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # FPS plot
        axes[0].bar(range(len(analysis['resolutions'])), analysis['fps'], color='steelblue')
        axes[0].set_xlabel('Resolution')
        axes[0].set_ylabel('FPS')
        axes[0].set_title('FPS vs Resolution')
        axes[0].set_xticks(range(len(analysis['resolutions'])))
        axes[0].set_xticklabels(analysis['resolutions'], rotation=45)
        axes[0].axhline(y=90, color='r', linestyle='--', label='Target: 90 FPS')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Latency plot
        axes[1].bar(range(len(analysis['resolutions'])), analysis['latency_ms'], color='coral')
        axes[1].set_xlabel('Resolution')
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title('Latency vs Resolution')
        axes[1].set_xticks(range(len(analysis['resolutions'])))
        axes[1].set_xticklabels(analysis['resolutions'], rotation=45)
        axes[1].axhline(y=50, color='r', linestyle='--', label='Target: 50ms')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Memory plot
        axes[2].bar(range(len(analysis['resolutions'])), analysis['gpu_memory_mb'], color='green')
        axes[2].set_xlabel('Resolution')
        axes[2].set_ylabel('GPU Memory (MB)')
        axes[2].set_title('GPU Memory vs Resolution')
        axes[2].set_xticks(range(len(analysis['resolutions'])))
        axes[2].set_xticklabels(analysis['resolutions'], rotation=45)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'resolution_impact.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_batch_size_impact(self):
        """Plot batch size impact on performance."""
        analysis = self.analyze_batch_size_impact()

        if 'error' in analysis:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # FPS plot
        axes[0, 0].plot(analysis['batch_sizes'], analysis['fps'], marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].set_title('FPS vs Batch Size')
        axes[0, 0].axhline(y=90, color='r', linestyle='--', label='Target: 90 FPS')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Latency plot
        axes[0, 1].plot(analysis['batch_sizes'], analysis['latency_ms'], marker='s', linewidth=2, markersize=8, color='coral')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Latency vs Batch Size')
        axes[0, 1].axhline(y=50, color='r', linestyle='--', label='Target: 50ms')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Memory plot
        axes[1, 0].plot(analysis['batch_sizes'], analysis['gpu_memory_mb'], marker='^', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('GPU Memory (MB)')
        axes[1, 0].set_title('GPU Memory vs Batch Size')
        axes[1, 0].grid(True, alpha=0.3)

        # Throughput plot
        axes[1, 1].plot(analysis['batch_sizes'], analysis['throughput'], marker='d', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Throughput (frames/sec)')
        axes[1, 1].set_title('Throughput vs Batch Size')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_size_impact.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_precision_impact(self):
        """Plot precision impact on performance."""
        analysis = self.analyze_precision_impact()

        if 'error' in analysis:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # FPS comparison
        axes[0].bar(analysis['precisions'], analysis['fps'], color=['steelblue', 'lightcoral'])
        axes[0].set_xlabel('Precision')
        axes[0].set_ylabel('FPS')
        axes[0].set_title('FPS by Precision Mode')
        axes[0].axhline(y=90, color='r', linestyle='--', label='Target: 90 FPS')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Latency comparison
        axes[1].bar(analysis['precisions'], analysis['latency_ms'], color=['coral', 'lightgreen'])
        axes[1].set_xlabel('Precision')
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title('Latency by Precision Mode')
        axes[1].axhline(y=50, color='r', linestyle='--', label='Target: 50ms')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Memory comparison
        axes[2].bar(analysis['precisions'], analysis['gpu_memory_mb'], color=['green', 'orange'])
        axes[2].set_xlabel('Precision')
        axes[2].set_ylabel('GPU Memory (MB)')
        axes[2].set_title('GPU Memory by Precision Mode')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_impact.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_summary(self):
        """Plot overall performance summary."""
        successful_results = [r for r in self.results if r['success']]

        if not successful_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # FPS distribution
        fps_values = [r['avg_fps'] for r in successful_results]
        axes[0, 0].hist(fps_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=90, color='r', linestyle='--', linewidth=2, label='Target: 90 FPS')
        axes[0, 0].set_xlabel('FPS')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('FPS Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Latency distribution
        latency_values = [r['avg_latency_ms'] for r in successful_results]
        axes[0, 1].hist(latency_values, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=50, color='r', linestyle='--', linewidth=2, label='Target: 50ms')
        axes[0, 1].set_xlabel('Latency (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Latency Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # FPS vs Latency scatter
        axes[1, 0].scatter(latency_values, fps_values, c='green', s=100, alpha=0.6, edgecolors='black')
        axes[1, 0].set_xlabel('Latency (ms)')
        axes[1, 0].set_ylabel('FPS')
        axes[1, 0].set_title('FPS vs Latency')
        axes[1, 0].grid(True, alpha=0.3)

        # GPU Memory usage
        memory_values = [r['gpu_memory_mb'] for r in successful_results]
        names = [r['name'] for r in successful_results]
        axes[1, 1].barh(range(len(names)), memory_values, color='purple', alpha=0.7)
        axes[1, 1].set_yticks(range(len(names)))
        axes[1, 1].set_yticklabels(names, fontsize=8)
        axes[1, 1].set_xlabel('GPU Memory (MB)')
        axes[1, 1].set_title('GPU Memory Usage by Benchmark')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_file: Optional[str] = None):
        """
        Generate comprehensive performance report.

        Args:
            output_file: Output file path (default: performance_report.txt)
        """
        if output_file is None:
            output_file = self.output_dir / 'performance_report.txt'
        else:
            output_file = Path(output_file)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VR BODY SEGMENTATION - PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Timestamp
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Benchmark Date: {self.data.get('timestamp', 'Unknown')}\n\n")

            # Hardware info
            f.write("-" * 80 + "\n")
            f.write("HARDWARE CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            for key, value in self.hardware.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # Resolution analysis
            f.write("-" * 80 + "\n")
            f.write("RESOLUTION IMPACT ANALYSIS\n")
            f.write("-" * 80 + "\n")
            res_analysis = self.analyze_resolution_impact()
            if 'error' not in res_analysis:
                for i, res in enumerate(res_analysis['resolutions']):
                    f.write(f"\n{res}:\n")
                    f.write(f"  FPS: {res_analysis['fps'][i]:.2f}\n")
                    f.write(f"  Latency: {res_analysis['latency_ms'][i]:.2f}ms\n")
                    f.write(f"  GPU Memory: {res_analysis['gpu_memory_mb'][i]:.2f}MB\n")
            f.write("\n")

            # Batch size analysis
            f.write("-" * 80 + "\n")
            f.write("BATCH SIZE IMPACT ANALYSIS\n")
            f.write("-" * 80 + "\n")
            batch_analysis = self.analyze_batch_size_impact()
            if 'error' not in batch_analysis:
                f.write(f"Optimal Batch Size (FPS): {batch_analysis['optimal_batch_size']}\n")
                f.write(f"Optimal Batch Size (Throughput): {batch_analysis['optimal_throughput_batch']}\n\n")
                for i, bs in enumerate(batch_analysis['batch_sizes']):
                    f.write(f"\nBatch Size {bs}:\n")
                    f.write(f"  FPS: {batch_analysis['fps'][i]:.2f}\n")
                    f.write(f"  Latency: {batch_analysis['latency_ms'][i]:.2f}ms\n")
                    f.write(f"  Throughput: {batch_analysis['throughput'][i]:.2f} frames/sec\n")
            f.write("\n")

            # Precision analysis
            f.write("-" * 80 + "\n")
            f.write("PRECISION MODE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            prec_analysis = self.analyze_precision_impact()
            if 'error' not in prec_analysis:
                if 'fp16_speedup' in prec_analysis:
                    f.write(f"FP16 Speedup: {prec_analysis['fp16_speedup']:.2f}x\n")
                    f.write(f"FP16 Memory Reduction: {prec_analysis['fp16_memory_reduction_percent']:.1f}%\n\n")
                for i, prec in enumerate(prec_analysis['precisions']):
                    f.write(f"\n{prec.upper()}:\n")
                    f.write(f"  FPS: {prec_analysis['fps'][i]:.2f}\n")
                    f.write(f"  Latency: {prec_analysis['latency_ms'][i]:.2f}ms\n")
            f.write("\n")

            # Recommendations
            f.write("-" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            self._write_recommendations(f)

            f.write("\n" + "=" * 80 + "\n")

        print(f"Report saved to: {output_file}")

    def _write_recommendations(self, f):
        """Write performance recommendations to report."""
        # Analyze all results
        successful = [r for r in self.results if r['success']]

        if not successful:
            f.write("No successful benchmarks to analyze.\n")
            return

        # Find configurations meeting VR targets (90 FPS, <50ms latency)
        vr_suitable = [r for r in successful if r['avg_fps'] >= 90 and r['avg_latency_ms'] <= 50]

        if vr_suitable:
            f.write("Configurations meeting VR targets (90+ FPS, <50ms latency):\n\n")
            for result in vr_suitable:
                f.write(f"  - {result['name']}\n")
                for key, value in result['config'].items():
                    f.write(f"      {key}: {value}\n")
                f.write(f"      Performance: {result['avg_fps']:.1f} FPS, {result['avg_latency_ms']:.1f}ms\n\n")
        else:
            f.write("WARNING: No configurations meet VR performance targets!\n\n")
            best_fps = max(successful, key=lambda x: x['avg_fps'])
            f.write(f"Best achievable FPS: {best_fps['avg_fps']:.1f} ({best_fps['name']})\n")
            f.write("Recommendations:\n")
            f.write("  - Consider reducing input resolution\n")
            f.write("  - Use FP16 precision if not already enabled\n")
            f.write("  - Increase batch size for better GPU utilization\n")
            f.write("  - Optimize model architecture\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance Analysis for VR Body Segmentation")
    parser.add_argument("results_file", type=str, help="Path to benchmark results JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory for analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    # Create analyzer
    analyzer = PerformanceAnalyzer(args.results_file, args.output_dir)

    # Generate report
    if not args.report_only:
        print("Generating performance report...")
    analyzer.generate_report()

    # Generate plots
    if not args.no_plots and not args.report_only:
        print("Generating visualization plots...")
        analyzer.generate_plots()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
