#!/usr/bin/env python3
"""
Main CLI application for VR Body Segmentation.

Provides a comprehensive command-line interface with progress bars,
real-time metrics, and interactive mode.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
import signal

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    print("Warning: rich not installed. Install with: pip install rich")
    RICH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import psutil

from src.utils.logger import setup_logger, LoggerConfig
from src.utils.config_manager import ConfigManager, AppConfig, get_config_manager
from src.utils.cache_manager import CacheManager, get_cache_manager


class PerformanceMonitor:
    """Real-time performance monitoring."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.fps_history = []

    def update(self, frames_processed: int = 1):
        """Update frame count."""
        self.frame_count += frames_processed

    def get_fps(self) -> float:
        """Get current FPS."""
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        try:
            return {
                'utilization': torch.cuda.utilization(),
                'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2),
                'memory_percent': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            }
        except:
            return {}

    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics."""
        return {
            'utilization': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent
        }

    def reset(self):
        """Reset monitoring."""
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.fps_history = []


class CLIApplication:
    """Main CLI application."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.config_manager = ConfigManager()
        self.cache_manager = None
        self.logger = None
        self.monitor = PerformanceMonitor()
        self.running = True

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            self.running = False
            if self.console:
                self.console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            else:
                print("\nShutting down gracefully...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def print_header(self):
        """Print application header."""
        if self.console:
            self.console.print(Panel.fit(
                "[bold cyan]VR Body Segmentation[/bold cyan]\n"
                "[dim]High-Performance Real-Time Segmentation for VR Content[/dim]",
                border_style="cyan"
            ))
        else:
            print("="*60)
            print("VR Body Segmentation")
            print("High-Performance Real-Time Segmentation for VR Content")
            print("="*60)

    def print_hardware_info(self):
        """Print hardware information."""
        hw_info = self.config_manager.get_hardware_info()

        if self.console:
            table = Table(title="Hardware Configuration", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Details", style="white")

            table.add_row("CPU", f"{hw_info.get('cpu_count', 'N/A')} cores")
            table.add_row("Memory", f"{hw_info.get('memory_gb', 0):.1f} GB")

            if hw_info.get('cuda_available'):
                table.add_row("GPU", hw_info.get('gpu_name', 'N/A'))
                table.add_row("GPU Memory", f"{hw_info.get('gpu_memory_gb', 0):.1f} GB")
                table.add_row("CUDA Version", hw_info.get('cuda_version', 'N/A'))
            else:
                table.add_row("GPU", "[red]Not Available[/red]")

            self.console.print(table)
        else:
            print("\nHardware Configuration:")
            print(f"  CPU: {hw_info.get('cpu_count', 'N/A')} cores")
            print(f"  Memory: {hw_info.get('memory_gb', 0):.1f} GB")
            if hw_info.get('cuda_available'):
                print(f"  GPU: {hw_info.get('gpu_name', 'N/A')}")
                print(f"  GPU Memory: {hw_info.get('gpu_memory_gb', 0):.1f} GB")
            else:
                print("  GPU: Not Available")

    def print_config_summary(self, config: AppConfig):
        """Print configuration summary."""
        if self.console:
            table = Table(title="Configuration Summary", box=box.ROUNDED)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Profile", config.profile)
            table.add_row("Batch Size", str(config.model.batch_size))
            table.add_row("Precision", config.gpu.precision)
            table.add_row("Input Size", f"{config.model.input_size[0]}x{config.model.input_size[1]}")
            table.add_row("Workers", str(config.model.num_workers))
            table.add_row("Target FPS", str(config.performance.target_fps))

            self.console.print(table)
        else:
            print("\nConfiguration Summary:")
            print(f"  Profile: {config.profile}")
            print(f"  Batch Size: {config.model.batch_size}")
            print(f"  Precision: {config.gpu.precision}")
            print(f"  Input Size: {config.model.input_size[0]}x{config.model.input_size[1]}")
            print(f"  Workers: {config.model.num_workers}")
            print(f"  Target FPS: {config.performance.target_fps}")

    def create_performance_table(self) -> Table:
        """Create real-time performance monitoring table."""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        # FPS
        fps = self.monitor.get_fps()
        fps_color = "green" if fps >= 90 else "yellow" if fps >= 60 else "red"
        table.add_row("FPS", f"[{fps_color}]{fps:.1f}[/{fps_color}]")

        # Frames processed
        table.add_row("Frames", str(self.monitor.frame_count))

        # GPU stats
        gpu_stats = self.monitor.get_gpu_stats()
        if gpu_stats:
            gpu_util_color = "green" if gpu_stats['utilization'] >= 80 else "yellow"
            table.add_row("GPU Util", f"[{gpu_util_color}]{gpu_stats['utilization']:.1f}%[/{gpu_util_color}]")

            mem_color = "green" if gpu_stats['memory_percent'] < 80 else "yellow" if gpu_stats['memory_percent'] < 95 else "red"
            table.add_row("GPU Mem", f"[{mem_color}]{gpu_stats['memory_allocated_mb']:.0f}/{gpu_stats['memory_total_mb']:.0f} MB ({gpu_stats['memory_percent']:.1f}%)[/{mem_color}]")

        # CPU stats
        cpu_stats = self.monitor.get_cpu_stats()
        cpu_color = "green" if cpu_stats['utilization'] < 80 else "yellow"
        table.add_row("CPU Util", f"[{cpu_color}]{cpu_stats['utilization']:.1f}%[/{cpu_color}]")
        table.add_row("CPU Mem", f"{cpu_stats['memory_percent']:.1f}%")

        return table

    def process_video(self, input_path: str, output_path: str, config: AppConfig):
        """
        Process video with progress monitoring.

        Args:
            input_path: Input video path
            output_path: Output video path
            config: Application configuration
        """
        if not self.console:
            print(f"\nProcessing: {input_path}")
            print("Progress monitoring requires 'rich' package")
            return

        # Simulate processing (replace with actual processing)
        total_frames = 1000  # This would come from video metadata

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:

            task = progress.add_task("[cyan]Processing frames...", total=total_frames)

            for i in range(total_frames):
                # Simulate frame processing
                time.sleep(0.001)

                # Update monitor
                self.monitor.update(1)

                # Update progress
                progress.update(task, advance=1)

                # Break if interrupted
                if not self.running:
                    break

        self.console.print(f"\n[green]Processing complete![/green]")
        self.console.print(f"Output saved to: {output_path}")

    def interactive_mode(self):
        """Run interactive mode."""
        if not self.console:
            print("Interactive mode requires 'rich' package")
            return

        self.console.print("\n[bold cyan]Interactive Mode[/bold cyan]")
        self.console.print("Enter commands (type 'help' for available commands, 'exit' to quit)\n")

        while self.running:
            try:
                command = self.console.input("[bold cyan]>[/bold cyan] ")
                command = command.strip().lower()

                if command == "exit" or command == "quit":
                    break
                elif command == "help":
                    self.print_help()
                elif command == "hardware":
                    self.print_hardware_info()
                elif command == "config":
                    config = self.config_manager.get_config()
                    self.print_config_summary(config)
                elif command == "stats":
                    self.console.print(self.create_performance_table())
                elif command == "clear":
                    self.console.clear()
                elif command.startswith("profile "):
                    profile_name = command.split()[1]
                    try:
                        config = self.config_manager.load_profile(profile_name)
                        self.console.print(f"[green]Loaded profile: {profile_name}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Error: {e}[/red]")
                else:
                    self.console.print(f"[red]Unknown command: {command}[/red]")

            except (KeyboardInterrupt, EOFError):
                break

        self.console.print("\n[yellow]Exiting interactive mode[/yellow]")

    def print_help(self):
        """Print help information."""
        if self.console:
            table = Table(title="Available Commands", box=box.ROUNDED)
            table.add_column("Command", style="cyan")
            table.add_column("Description", style="white")

            table.add_row("help", "Show this help message")
            table.add_row("hardware", "Show hardware information")
            table.add_row("config", "Show current configuration")
            table.add_row("stats", "Show performance statistics")
            table.add_row("profile <name>", "Load a profile (fast, balanced, quality)")
            table.add_row("clear", "Clear screen")
            table.add_row("exit", "Exit interactive mode")

            self.console.print(table)
        else:
            print("\nAvailable Commands:")
            print("  help              - Show this help message")
            print("  hardware          - Show hardware information")
            print("  config            - Show current configuration")
            print("  stats             - Show performance statistics")
            print("  profile <name>    - Load a profile")
            print("  clear             - Clear screen")
            print("  exit              - Exit interactive mode")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="VR Body Segmentation - High-Performance Real-Time Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")

    # Configuration
    parser.add_argument("-c", "--config", help="Configuration file")
    parser.add_argument("-p", "--profile", choices=['fast', 'balanced', 'quality'],
                       help="Load preset profile")

    # Options
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark mode")
    parser.add_argument("--optimize", action="store_true",
                       help="Auto-optimize configuration")
    parser.add_argument("--show-hardware", action="store_true",
                       help="Show hardware info and exit")
    parser.add_argument("--show-config", action="store_true",
                       help="Show configuration and exit")

    # Performance
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--precision", choices=['fp32', 'fp16'],
                       help="Override precision mode")
    parser.add_argument("--workers", type=int, help="Override number of workers")

    # Logging
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")
    parser.add_argument("--log-dir", default="./logs", help="Log directory")

    args = parser.parse_args()

    # Create application
    app = CLIApplication()
    app.setup_signal_handlers()

    # Print header
    app.print_header()

    # Show hardware info
    if args.show_hardware:
        app.print_hardware_info()
        return

    # Load configuration
    if args.profile:
        config = app.config_manager.load_profile(args.profile)
    elif args.config:
        config = app.config_manager.load_config(args.config)
    else:
        config = app.config_manager.load_config()

    # Apply overrides
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.precision:
        config.gpu.precision = args.precision
    if args.workers:
        config.model.num_workers = args.workers

    # Setup logger
    config.logging.level = args.log_level
    config.logging.log_dir = args.log_dir
    app.logger = setup_logger(
        "vr_body_segmentation",
        LoggerConfig(
            log_dir=config.logging.log_dir,
            console_level=getattr(__import__('logging'), config.logging.level),
            json_logging=config.logging.json_logging
        )
    )

    # Setup cache manager
    app.cache_manager = get_cache_manager(
        memory_cache_mb=config.processing.cache_size_gb * 1024,
        enable_memory_cache=config.processing.enable_caching,
        enable_disk_cache=config.processing.enable_caching
    )

    # Show configuration
    if args.show_config:
        app.print_config_summary(config)
        return

    # Auto-optimize
    if args.optimize:
        from scripts.optimize import PerformanceOptimizer
        optimizer = PerformanceOptimizer(app.config_manager, app.logger)
        config = optimizer.auto_optimize(config)
        if app.console:
            app.console.print("[green]Configuration optimized![/green]")
        else:
            print("Configuration optimized!")

    # Print config summary
    app.print_config_summary(config)

    # Interactive mode
    if args.interactive:
        app.interactive_mode()
        return

    # Benchmark mode
    if args.benchmark:
        if app.console:
            app.console.print("\n[yellow]Running benchmark...[/yellow]")
        else:
            print("\nRunning benchmark...")

        # Run benchmark (placeholder)
        from benchmarks.benchmark_suite import BenchmarkSuite
        suite = BenchmarkSuite()
        suite.run_all_benchmarks()
        return

    # Process video
    if args.input:
        output = args.output or f"output_{Path(args.input).name}"
        app.process_video(args.input, output, config)
    else:
        if app.console:
            app.console.print("[yellow]No input file specified. Use --interactive or --help[/yellow]")
        else:
            print("No input file specified. Use --interactive or --help")


if __name__ == "__main__":
    main()
