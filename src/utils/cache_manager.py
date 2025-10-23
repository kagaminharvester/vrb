"""
Advanced caching system for VR Body Segmentation application.

Provides LRU caching, disk caching, model weight caching, and preprocessed data caching
with automatic memory management.
"""

import os
import pickle
import hashlib
import time
import threading
from pathlib import Path
from typing import Any, Optional, Dict, Callable, Tuple
from collections import OrderedDict
from functools import wraps
import weakref
import psutil


class LRUCache:
    """Thread-safe Least Recently Used (LRU) cache."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 1024.0):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict = OrderedDict()
        self.memory_usage: Dict[str, int] = {}
        self.total_memory = 0
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            import sys
            return sys.getsizeof(obj)
        except (TypeError, AttributeError):
            # Object doesn't support sizeof
            return 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Remove old entry if exists
            if key in self.cache:
                self.total_memory -= self.memory_usage.get(key, 0)
                del self.cache[key]

            # Estimate size
            size = self._estimate_size(value)

            # Evict items if necessary
            while (len(self.cache) >= self.max_size or
                   self.total_memory + size > self.max_memory_bytes) and self.cache:
                oldest_key, _ = self.cache.popitem(last=False)
                self.total_memory -= self.memory_usage.pop(oldest_key, 0)

            # Add new item
            self.cache[key] = value
            self.memory_usage[key] = size
            self.total_memory += size

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.memory_usage.clear()
            self.total_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.total_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class DiskCache:
    """Persistent disk-based cache."""

    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.lock = threading.RLock()
        self.index_file = self.cache_dir / ".cache_index.pkl"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
                # Cache file corrupted or missing, start fresh
                return {}
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            print(f"Warning: Failed to save cache index: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(info['size'] for info in self.index.values())

    def _evict_lru(self):
        """Evict least recently used items to stay under size limit."""
        while self._get_total_size() > self.max_size_bytes and self.index:
            # Find LRU item
            lru_key = min(self.index.items(), key=lambda x: x[1]['last_access'])[0]
            cache_path = self._get_cache_path(lru_key)

            # Remove file
            if cache_path.exists():
                cache_path.unlink()

            # Remove from index
            del self.index[lru_key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from disk cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key not in self.index:
                return None

            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                # Remove from index if file doesn't exist
                del self.index[key]
                return None

            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)

                # Update access time
                self.index[key]['last_access'] = time.time()
                self._save_index()

                return value
            except Exception as e:
                print(f"Warning: Failed to load cache for key '{key}': {e}")
                return None

    def put(self, key: str, value: Any):
        """
        Put item in disk cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            cache_path = self._get_cache_path(key)

            try:
                # Serialize to disk
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)

                # Update index
                self.index[key] = {
                    'last_access': time.time(),
                    'size': cache_path.stat().st_size
                }

                # Evict if necessary
                self._evict_lru()

                # Save index
                self._save_index()

            except Exception as e:
                print(f"Warning: Failed to cache key '{key}': {e}")
                if cache_path.exists():
                    cache_path.unlink()

    def clear(self):
        """Clear the cache."""
        with self.lock:
            # Remove all cache files
            for key in list(self.index.keys()):
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()

            self.index.clear()
            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = self._get_total_size()
            return {
                'size': len(self.index),
                'total_size_gb': total_size / (1024 ** 3),
                'max_size_gb': self.max_size_bytes / (1024 ** 3),
                'utilization': total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }


class ModelWeightCache:
    """Cache for model weights with automatic loading."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model weight cache.

        Args:
            cache_dir: Directory for cached models
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/vr_body_segmentation/models")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, weakref.ref] = {}
        self.lock = threading.RLock()

    def get_model_path(self, model_name: str, version: str = "latest") -> Path:
        """
        Get path to cached model.

        Args:
            model_name: Name of the model
            version: Model version

        Returns:
            Path to model file
        """
        return self.cache_dir / f"{model_name}_{version}.pth"

    def has_model(self, model_name: str, version: str = "latest") -> bool:
        """
        Check if model is cached.

        Args:
            model_name: Name of the model
            version: Model version

        Returns:
            True if model is cached
        """
        return self.get_model_path(model_name, version).exists()

    def save_model(self, model: Any, model_name: str, version: str = "latest"):
        """
        Save model to cache.

        Args:
            model: Model to save
            model_name: Name of the model
            version: Model version
        """
        import torch

        with self.lock:
            model_path = self.get_model_path(model_name, version)
            torch.save(model.state_dict(), model_path)

    def load_model(self, model_class: Callable, model_name: str,
                   version: str = "latest", **kwargs) -> Optional[Any]:
        """
        Load model from cache.

        Args:
            model_class: Model class to instantiate
            model_name: Name of the model
            version: Model version
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model or None if not cached
        """
        import torch

        with self.lock:
            cache_key = f"{model_name}_{version}"

            # Check if already loaded in memory
            if cache_key in self.loaded_models:
                model_ref = self.loaded_models[cache_key]
                model = model_ref()
                if model is not None:
                    return model

            # Load from disk
            model_path = self.get_model_path(model_name, version)
            if not model_path.exists():
                return None

            try:
                model = model_class(**kwargs)
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)

                # Store weak reference
                self.loaded_models[cache_key] = weakref.ref(model)

                return model
            except Exception as e:
                print(f"Warning: Failed to load model '{model_name}': {e}")
                return None

    def clear(self):
        """Clear model cache."""
        with self.lock:
            self.loaded_models.clear()
            for model_file in self.cache_dir.glob("*.pth"):
                model_file.unlink()


class CacheManager:
    """Unified cache manager for the application."""

    def __init__(self,
                 memory_cache_size: int = 100,
                 memory_cache_mb: float = 1024.0,
                 disk_cache_dir: Optional[str] = None,
                 disk_cache_gb: float = 10.0,
                 enable_memory_cache: bool = True,
                 enable_disk_cache: bool = True):
        """
        Initialize cache manager.

        Args:
            memory_cache_size: Maximum number of items in memory cache
            memory_cache_mb: Maximum memory cache size in MB
            disk_cache_dir: Directory for disk cache
            disk_cache_gb: Maximum disk cache size in GB
            enable_memory_cache: Enable memory caching
            enable_disk_cache: Enable disk caching
        """
        self.enable_memory_cache = enable_memory_cache
        self.enable_disk_cache = enable_disk_cache

        # Memory cache
        if enable_memory_cache:
            self.memory_cache = LRUCache(memory_cache_size, memory_cache_mb)
        else:
            self.memory_cache = None

        # Disk cache
        if enable_disk_cache:
            if disk_cache_dir is None:
                disk_cache_dir = os.path.expanduser("~/.cache/vr_body_segmentation/data")
            self.disk_cache = DiskCache(disk_cache_dir, disk_cache_gb)
        else:
            self.disk_cache = None

        # Model cache
        self.model_cache = ModelWeightCache()

    def get(self, key: str, use_disk: bool = True) -> Optional[Any]:
        """
        Get item from cache (memory first, then disk).

        Args:
            key: Cache key
            use_disk: Whether to check disk cache if not in memory

        Returns:
            Cached value or None if not found
        """
        # Try memory cache first
        if self.memory_cache is not None:
            value = self.memory_cache.get(key)
            if value is not None:
                return value

        # Try disk cache
        if use_disk and self.disk_cache is not None:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory cache
                if self.memory_cache is not None:
                    self.memory_cache.put(key, value)
                return value

        return None

    def put(self, key: str, value: Any, to_disk: bool = False):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
            to_disk: Whether to also save to disk cache
        """
        # Put in memory cache
        if self.memory_cache is not None:
            self.memory_cache.put(key, value)

        # Optionally put in disk cache
        if to_disk and self.disk_cache is not None:
            self.disk_cache.put(key, value)

    def clear(self):
        """Clear all caches."""
        if self.memory_cache is not None:
            self.memory_cache.clear()
        if self.disk_cache is not None:
            self.disk_cache.clear()
        self.model_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}

        if self.memory_cache is not None:
            stats['memory_cache'] = self.memory_cache.get_stats()

        if self.disk_cache is not None:
            stats['disk_cache'] = self.disk_cache.get_stats()

        # System memory stats
        memory = psutil.virtual_memory()
        stats['system_memory'] = {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'percent_used': memory.percent
        }

        return stats


def cached(key_func: Optional[Callable] = None,
          to_disk: bool = False,
          cache_manager: Optional[CacheManager] = None):
    """
    Decorator for caching function results.

    Args:
        key_func: Function to generate cache key from arguments
        to_disk: Whether to cache to disk
        cache_manager: Cache manager to use (creates default if None)

    Example:
        @cached(key_func=lambda x, y: f"add_{x}_{y}")
        def add(x, y):
            return x + y
    """
    if cache_manager is None:
        cache_manager = CacheManager()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"

            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.put(cache_key, result, to_disk=to_disk)

            return result

        return wrapper

    return decorator


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager(**kwargs) -> CacheManager:
    """
    Get or create global cache manager.

    Args:
        **kwargs: Arguments for CacheManager initialization

    Returns:
        CacheManager instance
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(**kwargs)
    return _global_cache_manager


if __name__ == "__main__":
    # Example usage
    import numpy as np

    print("Testing Cache Manager\n")

    # Create cache manager
    cache_mgr = CacheManager(
        memory_cache_size=10,
        memory_cache_mb=100.0,
        disk_cache_gb=1.0
    )

    # Test memory cache
    print("Memory Cache Test:")
    cache_mgr.put("test_key", {"data": [1, 2, 3, 4, 5]})
    result = cache_mgr.get("test_key")
    print(f"  Cached: {result}")

    # Test disk cache
    print("\nDisk Cache Test:")
    large_data = np.random.rand(1000, 1000)
    cache_mgr.put("large_data", large_data, to_disk=True)
    result = cache_mgr.get("large_data")
    print(f"  Retrieved shape: {result.shape if result is not None else None}")

    # Test decorator
    print("\nDecorator Test:")

    @cached(key_func=lambda x: f"square_{x}", cache_manager=cache_mgr)
    def expensive_operation(x):
        print(f"  Computing square of {x}...")
        time.sleep(0.1)
        return x ** 2

    # First call - computed
    result1 = expensive_operation(5)
    print(f"  Result: {result1}")

    # Second call - cached
    result2 = expensive_operation(5)
    print(f"  Result: {result2}")

    # Get statistics
    print("\nCache Statistics:")
    stats = cache_mgr.get_stats()
    for cache_type, cache_stats in stats.items():
        print(f"\n{cache_type}:")
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
