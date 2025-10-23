"""
Configuration management system for VR Body Segmentation application.

Provides YAML-based configuration with validation, profiles, and auto-detection
of optimal settings based on hardware capabilities.
"""

import os
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class PrecisionMode(Enum):
    """Precision modes for model inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


class ProcessingProfile(Enum):
    """Processing profiles for different use cases."""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"


@dataclass
class GPUConfig:
    """GPU-specific configuration."""
    device_id: int = 0
    max_vram_usage: float = 0.9  # Use up to 90% of VRAM
    precision: str = "fp16"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    allow_tf32: bool = True
    memory_fraction: Optional[float] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "sam2_hiera_large"
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    batch_size: int = 8
    input_size: tuple = (1024, 1024)
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    num_threads: int = 32
    max_queue_size: int = 100
    frame_skip: int = 0  # Skip every N frames (0 = no skip)
    use_multiprocessing: bool = True
    chunk_size: int = 100
    enable_caching: bool = True
    cache_size_gb: float = 4.0


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    target_fps: int = 90
    max_latency_ms: float = 50.0
    enable_profiling: bool = False
    auto_tune: bool = True
    warmup_iterations: int = 10
    benchmark_mode: bool = False


@dataclass
class OutputConfig:
    """Output configuration."""
    format: str = "mp4"
    codec: str = "h264_nvenc"
    bitrate: str = "50M"
    save_masks: bool = True
    save_visualizations: bool = True
    output_dir: str = "./output"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "./logs"
    json_logging: bool = False
    console_output: bool = True
    file_output: bool = True
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class VRConfig:
    """VR-specific configuration."""
    stereoscopic: bool = True
    ipd: float = 63.0  # Interpupillary distance in mm
    process_both_eyes: bool = True
    eye_separation_method: str = "split"  # 'split' or 'metadata'


@dataclass
class AppConfig:
    """Main application configuration."""
    profile: str = "balanced"
    gpu: GPUConfig = field(default_factory=GPUConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    vr: VRConfig = field(default_factory=VRConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary."""
        config = cls()

        # Update nested configs
        if 'gpu' in config_dict:
            config.gpu = GPUConfig(**config_dict['gpu'])
        if 'model' in config_dict:
            model_data = config_dict['model']
            if 'input_size' in model_data and isinstance(model_data['input_size'], list):
                model_data['input_size'] = tuple(model_data['input_size'])
            config.model = ModelConfig(**model_data)
        if 'processing' in config_dict:
            config.processing = ProcessingConfig(**config_dict['processing'])
        if 'performance' in config_dict:
            config.performance = PerformanceConfig(**config_dict['performance'])
        if 'output' in config_dict:
            config.output = OutputConfig(**config_dict['output'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        if 'vr' in config_dict:
            config.vr = VRConfig(**config_dict['vr'])

        if 'profile' in config_dict:
            config.profile = config_dict['profile']

        return config


class ConfigManager:
    """Manager for application configuration."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to configs directory in project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "configs"

        self.config_dir = Path(config_dir)
        self.profiles_dir = self.config_dir / "profiles"
        self.current_config: Optional[AppConfig] = None

    def load_config(self, config_path: Optional[str] = None) -> AppConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file (uses default if None)

        Returns:
            Loaded configuration
        """
        if config_path is None:
            config_path = self.config_dir / "default_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            # Return default config
            self.current_config = AppConfig()
            return self.current_config

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.current_config = AppConfig.from_dict(config_dict)
        return self.current_config

    def save_config(self, config: AppConfig, config_path: str):
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    def load_profile(self, profile_name: str) -> AppConfig:
        """
        Load a named profile.

        Args:
            profile_name: Name of profile (fast, balanced, quality)

        Returns:
            Configuration with profile applied
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"

        if not profile_path.exists():
            raise ValueError(f"Profile '{profile_name}' not found at {profile_path}")

        with open(profile_path, 'r') as f:
            profile_dict = yaml.safe_load(f)

        config = AppConfig.from_dict(profile_dict)
        config.profile = profile_name
        self.current_config = config
        return config

    def create_profile(self, profile_name: str, config: AppConfig):
        """
        Create a new profile.

        Args:
            profile_name: Name for the profile
            config: Configuration to save as profile
        """
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        self.save_config(config, profile_path)

    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Detect hardware capabilities.

        Returns:
            Dictionary with hardware information
        """
        import torch
        import psutil
        import multiprocessing

        hw_info = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            hw_info['gpu_count'] = torch.cuda.device_count()
            hw_info['gpu_name'] = torch.cuda.get_device_name(0)
            hw_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hw_info['cuda_version'] = torch.version.cuda
            hw_info['cudnn_version'] = torch.backends.cudnn.version()

        return hw_info

    def auto_tune_config(self, config: Optional[AppConfig] = None) -> AppConfig:
        """
        Automatically tune configuration based on hardware.

        Args:
            config: Base configuration to tune (uses current or default if None)

        Returns:
            Auto-tuned configuration
        """
        if config is None:
            config = self.current_config or AppConfig()
        else:
            config = copy.deepcopy(config)

        hw_info = self.get_hardware_info()

        # Tune based on CPU
        cpu_count = hw_info.get('cpu_count', 4)
        config.processing.num_threads = max(4, min(64, cpu_count))
        config.model.num_workers = max(2, min(8, cpu_count // 4))

        # Tune based on GPU
        if hw_info.get('cuda_available', False):
            gpu_memory_gb = hw_info.get('gpu_memory_gb', 8)

            # Adjust batch size based on VRAM
            if gpu_memory_gb >= 20:  # RTX 3090/4090
                config.model.batch_size = 16
                config.processing.cache_size_gb = 8.0
            elif gpu_memory_gb >= 12:  # RTX 3080/4070
                config.model.batch_size = 8
                config.processing.cache_size_gb = 4.0
            elif gpu_memory_gb >= 8:  # RTX 3070/4060
                config.model.batch_size = 4
                config.processing.cache_size_gb = 2.0
            else:
                config.model.batch_size = 2
                config.processing.cache_size_gb = 1.0

            # Use FP16 for better performance on modern GPUs
            if 'RTX' in hw_info.get('gpu_name', '') or 'A100' in hw_info.get('gpu_name', ''):
                config.gpu.precision = "fp16"
            else:
                config.gpu.precision = "fp32"

        # Tune based on total memory
        memory_gb = hw_info.get('memory_gb', 16)
        config.processing.max_queue_size = int(min(200, memory_gb * 10))

        self.current_config = config
        return config

    def validate_config(self, config: AppConfig) -> List[str]:
        """
        Validate configuration and return list of issues.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Validate GPU config
        if config.gpu.max_vram_usage <= 0 or config.gpu.max_vram_usage > 1:
            issues.append("GPU max_vram_usage must be between 0 and 1")

        if config.gpu.precision not in ["fp32", "fp16", "int8", "mixed"]:
            issues.append(f"Invalid GPU precision: {config.gpu.precision}")

        # Validate model config
        if config.model.batch_size <= 0:
            issues.append("Model batch_size must be positive")

        if config.model.num_workers < 0:
            issues.append("Model num_workers must be non-negative")

        # Validate processing config
        if config.processing.num_threads <= 0:
            issues.append("Processing num_threads must be positive")

        if config.processing.max_queue_size <= 0:
            issues.append("Processing max_queue_size must be positive")

        # Validate performance config
        if config.performance.target_fps <= 0:
            issues.append("Performance target_fps must be positive")

        if config.performance.max_latency_ms <= 0:
            issues.append("Performance max_latency_ms must be positive")

        # Validate output config
        valid_formats = ["mp4", "avi", "mov", "mkv"]
        if config.output.format not in valid_formats:
            issues.append(f"Output format must be one of {valid_formats}")

        # Validate logging config
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.level.upper() not in valid_levels:
            issues.append(f"Logging level must be one of {valid_levels}")

        return issues

    def merge_configs(self, base: AppConfig, override: Dict[str, Any]) -> AppConfig:
        """
        Merge override settings into base configuration.

        Args:
            base: Base configuration
            override: Dictionary with override values

        Returns:
            Merged configuration
        """
        config = copy.deepcopy(base)
        config_dict = config.to_dict()

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        merged_dict = deep_update(config_dict, override)
        return AppConfig.from_dict(merged_dict)

    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self.current_config is None:
            self.current_config = self.load_config()
        return self.current_config


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """
    Get or create global configuration manager.

    Args:
        config_dir: Configuration directory

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Convenience function to load configuration."""
    return get_config_manager().load_config(config_path)


def get_config() -> AppConfig:
    """Convenience function to get current configuration."""
    return get_config_manager().get_config()


if __name__ == "__main__":
    # Example usage
    manager = ConfigManager()

    # Load default config
    config = manager.load_config()
    print("Default configuration loaded")

    # Auto-tune based on hardware
    print("\nDetected hardware:")
    hw_info = manager.get_hardware_info()
    for key, value in hw_info.items():
        print(f"  {key}: {value}")

    print("\nAuto-tuning configuration...")
    tuned_config = manager.auto_tune_config(config)
    print(f"Batch size: {tuned_config.model.batch_size}")
    print(f"Num threads: {tuned_config.processing.num_threads}")
    print(f"Precision: {tuned_config.gpu.precision}")

    # Validate configuration
    issues = manager.validate_config(tuned_config)
    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid")
