"""
Advanced logging framework for VR Body Segmentation application.

Provides multi-level logging with performance metrics, GPU monitoring,
structured logging, and automatic log rotation.
"""

import logging
import logging.handlers
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from contextlib import contextmanager


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""

    def filter(self, record):
        if not hasattr(record, 'performance_data'):
            record.performance_data = {}
        return True


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add performance data if available
        if hasattr(record, 'performance_data') and record.performance_data:
            log_data['performance'] = record.performance_data

        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        formatted = super().format(record)

        # Reset levelname for subsequent handlers
        record.levelname = levelname

        return formatted


class PerformanceLogger:
    """Logger with built-in performance tracking capabilities."""

    def __init__(self, name: str = "vr_body_segmentation"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    @contextmanager
    def timer(self, operation: str, log_level: int = logging.DEBUG):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.log_performance(operation, elapsed, log_level)

    def log_performance(self, operation: str, elapsed_time: float,
                       log_level: int = logging.DEBUG):
        """Log performance metrics for an operation."""
        with self._lock:
            if operation not in self._timers:
                self._timers[operation] = []
            self._timers[operation].append(elapsed_time)

        perf_data = {
            'operation': operation,
            'elapsed_time': f"{elapsed_time:.4f}s",
            'elapsed_ms': f"{elapsed_time * 1000:.2f}ms"
        }

        extra = {'performance_data': perf_data}
        self.logger.log(log_level,
                       f"Performance: {operation} took {elapsed_time*1000:.2f}ms",
                       extra=extra)

    def increment_counter(self, counter_name: str):
        """Increment a named counter."""
        with self._lock:
            self._counters[counter_name] = self._counters.get(counter_name, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            stats = {
                'timers': {},
                'counters': dict(self._counters)
            }

            for operation, times in self._timers.items():
                if times:
                    stats['timers'][operation] = {
                        'count': len(times),
                        'total': sum(times),
                        'mean': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times)
                    }

            return stats

    def reset_stats(self):
        """Reset all performance statistics."""
        with self._lock:
            self._timers.clear()
            self._counters.clear()

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, extra={'extra_data': kwargs} if kwargs else {})

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, extra={'extra_data': kwargs} if kwargs else {})

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, extra={'extra_data': kwargs} if kwargs else {})

    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, extra={'extra_data': kwargs} if kwargs else {})

    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, extra={'extra_data': kwargs} if kwargs else {})

    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, extra={'extra_data': kwargs} if kwargs else {})


class LoggerConfig:
    """Configuration for logger setup."""

    def __init__(self,
                 log_dir: Optional[str] = None,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 json_logging: bool = False,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        self.console_level = console_level
        self.file_level = file_level
        self.json_logging = json_logging
        self.max_bytes = max_bytes
        self.backup_count = backup_count


def setup_logger(name: str = "vr_body_segmentation",
                config: Optional[LoggerConfig] = None) -> PerformanceLogger:
    """
    Setup and configure the application logger.

    Args:
        name: Logger name
        config: Logger configuration

    Returns:
        Configured PerformanceLogger instance
    """
    if config is None:
        config = LoggerConfig()

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear existing handlers

    # Add performance filter
    perf_filter = PerformanceFilter()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.console_level)
    console_formatter = ColoredConsoleFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(perf_filter)
    logger.addHandler(console_handler)

    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)

    # File handler with rotation
    log_file = os.path.join(config.log_dir, f'{name}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    file_handler.setLevel(config.file_level)

    if config.json_logging:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(perf_filter)
    logger.addHandler(file_handler)

    # Performance log file (always JSON)
    perf_log_file = os.path.join(config.log_dir, f'{name}_performance.json')
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    perf_handler.setLevel(logging.DEBUG)
    perf_handler.setFormatter(StructuredFormatter())
    perf_handler.addFilter(perf_filter)

    # Only log records with performance data
    class PerfOnlyFilter(logging.Filter):
        def filter(self, record):
            return hasattr(record, 'performance_data') and bool(record.performance_data)

    perf_handler.addFilter(PerfOnlyFilter())
    logger.addHandler(perf_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return PerformanceLogger(name)


# Global logger instance
_global_logger: Optional[PerformanceLogger] = None


def get_logger(name: Optional[str] = None) -> PerformanceLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (uses global logger if None)

    Returns:
        PerformanceLogger instance
    """
    global _global_logger

    if name is None:
        if _global_logger is None:
            _global_logger = setup_logger()
        return _global_logger
    else:
        return setup_logger(name)


# Convenience functions
def debug(msg: str, **kwargs):
    """Log debug message using global logger."""
    get_logger().debug(msg, **kwargs)


def info(msg: str, **kwargs):
    """Log info message using global logger."""
    get_logger().info(msg, **kwargs)


def warning(msg: str, **kwargs):
    """Log warning message using global logger."""
    get_logger().warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """Log error message using global logger."""
    get_logger().error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """Log critical message using global logger."""
    get_logger().critical(msg, **kwargs)


def exception(msg: str, **kwargs):
    """Log exception with traceback using global logger."""
    get_logger().exception(msg, **kwargs)


@contextmanager
def timer(operation: str, log_level: int = logging.DEBUG):
    """Context manager for timing operations using global logger."""
    with get_logger().timer(operation, log_level):
        yield


if __name__ == "__main__":
    # Example usage
    logger = setup_logger(config=LoggerConfig(console_level=logging.DEBUG))

    logger.info("Application started")

    with logger.timer("test_operation", logging.INFO):
        time.sleep(0.1)

    logger.increment_counter("frames_processed")
    logger.increment_counter("frames_processed")

    logger.debug("Debug message", frame_id=123, resolution="4K")
    logger.warning("Warning message", gpu_memory="80%")

    print("\nPerformance Statistics:")
    print(json.dumps(logger.get_stats(), indent=2))
