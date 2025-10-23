"""
Efficient Batch Processing Framework
Optimized for High Throughput on RTX 3090

This module provides intelligent batching strategies to maximize GPU utilization
while balancing latency and throughput requirements.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Callable
import queue
import threading
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    min_batch_size: int = 1
    max_batch_size: int = 8
    timeout_ms: float = 50.0  # Max wait time to accumulate batch
    dynamic_batching: bool = True
    priority_mode: bool = False  # Prioritize latency over throughput


@dataclass
class InferenceRequest:
    """Single inference request."""
    data: np.ndarray
    request_id: str
    timestamp: float
    priority: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class InferenceResponse:
    """Inference response."""
    result: np.ndarray
    request_id: str
    latency_ms: float
    batch_size: int
    metadata: Dict[str, Any] = None


class DynamicBatcher:
    """
    Dynamic batching engine for optimal GPU utilization.

    Features:
    - Automatic batch accumulation with timeout
    - Priority-based batching
    - Variable batch size support
    - Latency-throughput trade-off
    """

    def __init__(
        self,
        config: BatchConfig,
        inference_fn: Callable[[np.ndarray], np.ndarray],
        device_id: int = 0
    ):
        """
        Initialize dynamic batcher.

        Args:
            config: Batch configuration
            inference_fn: Function to run inference on batch
            device_id: GPU device ID
        """
        self.config = config
        self.inference_fn = inference_fn
        self.device = torch.device(f'cuda:{device_id}')

        # Request queues
        self.request_queue = queue.PriorityQueue() if config.priority_mode else queue.Queue()
        self.response_queues = {}  # {request_id: queue}

        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_latency_ms': 0.0,
            'throughput_fps': 0.0
        }

        # Control
        self.running = False
        self.batch_thread = None

        logger.info(f"Initialized DynamicBatcher with config: {config}")

    def start(self):
        """Start batch processing thread."""
        if self.running:
            logger.warning("Batcher already running")
            return

        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self.batch_thread.start()
        logger.info("Started batch processing thread")

    def stop(self):
        """Stop batch processing thread."""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        logger.info("Stopped batch processing thread")

    def submit(self, request: InferenceRequest) -> InferenceResponse:
        """
        Submit inference request (blocking).

        Args:
            request: Inference request

        Returns:
            Inference response
        """
        # Create response queue for this request
        response_queue = queue.Queue(maxsize=1)
        self.response_queues[request.request_id] = response_queue

        # Submit request
        if self.config.priority_mode:
            self.request_queue.put((-request.priority, request))
        else:
            self.request_queue.put(request)

        self.stats['total_requests'] += 1

        # Wait for response
        try:
            response = response_queue.get(timeout=30.0)  # 30s timeout
            return response
        finally:
            # Cleanup
            del self.response_queues[request.request_id]

    def submit_async(self, request: InferenceRequest):
        """
        Submit inference request (non-blocking).

        Args:
            request: Inference request
        """
        # Create response queue
        response_queue = queue.Queue(maxsize=1)
        self.response_queues[request.request_id] = response_queue

        # Submit request
        if self.config.priority_mode:
            self.request_queue.put((-request.priority, request))
        else:
            self.request_queue.put(request)

        self.stats['total_requests'] += 1

    def get_response(self, request_id: str, timeout: float = None) -> Optional[InferenceResponse]:
        """
        Get response for async request.

        Args:
            request_id: Request ID
            timeout: Timeout in seconds

        Returns:
            Response or None if timeout
        """
        if request_id not in self.response_queues:
            return None

        try:
            response = self.response_queues[request_id].get(timeout=timeout)
            del self.response_queues[request_id]
            return response
        except queue.Empty:
            return None

    def _batch_loop(self):
        """Main batch processing loop."""
        while self.running:
            try:
                # Accumulate batch
                batch = self._accumulate_batch()

                if len(batch) == 0:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue

                # Process batch
                self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in batch loop: {e}", exc_info=True)

    def _accumulate_batch(self) -> List[InferenceRequest]:
        """
        Accumulate batch of requests.

        Returns:
            List of requests to process
        """
        batch = []
        start_time = time.perf_counter()
        timeout_s = self.config.timeout_ms / 1000.0

        while len(batch) < self.config.max_batch_size:
            # Calculate remaining timeout
            elapsed = time.perf_counter() - start_time
            remaining_timeout = timeout_s - elapsed

            if remaining_timeout <= 0 and len(batch) >= self.config.min_batch_size:
                # Timeout reached with minimum batch size
                break

            try:
                # Try to get request with timeout
                wait_time = max(remaining_timeout, 0.001) if self.config.dynamic_batching else None

                if self.config.priority_mode:
                    _, request = self.request_queue.get(timeout=wait_time)
                else:
                    request = self.request_queue.get(timeout=wait_time)

                batch.append(request)

            except queue.Empty:
                # No more requests available
                if len(batch) >= self.config.min_batch_size:
                    break
                elif not self.config.dynamic_batching:
                    # In non-dynamic mode, wait for full batch
                    continue
                else:
                    break

        return batch

    def _process_batch(self, batch: List[InferenceRequest]):
        """
        Process batch of requests.

        Args:
            batch: List of requests
        """
        batch_start = time.perf_counter()
        batch_size = len(batch)

        try:
            # Stack inputs into batch
            batch_data = np.stack([req.data for req in batch], axis=0)

            # Run inference
            inference_start = time.perf_counter()
            batch_results = self.inference_fn(batch_data)
            inference_time = (time.perf_counter() - inference_start) * 1000  # ms

            # Distribute results
            for i, request in enumerate(batch):
                total_latency = (time.perf_counter() - request.timestamp) * 1000  # ms

                response = InferenceResponse(
                    result=batch_results[i],
                    request_id=request.request_id,
                    latency_ms=total_latency,
                    batch_size=batch_size,
                    metadata={
                        'inference_time_ms': inference_time,
                        'queue_time_ms': total_latency - inference_time
                    }
                )

                # Send response
                if request.request_id in self.response_queues:
                    self.response_queues[request.request_id].put(response)

            # Update statistics
            self._update_stats(batch_size, inference_time)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

            # Send error responses
            for request in batch:
                if request.request_id in self.response_queues:
                    error_response = InferenceResponse(
                        result=None,
                        request_id=request.request_id,
                        latency_ms=-1,
                        batch_size=batch_size,
                        metadata={'error': str(e)}
                    )
                    self.response_queues[request.request_id].put(error_response)

    def _update_stats(self, batch_size: int, inference_time_ms: float):
        """Update running statistics."""
        self.stats['total_batches'] += 1

        # Running average for batch size
        alpha = 0.1  # EMA smoothing factor
        self.stats['avg_batch_size'] = (
            alpha * batch_size + (1 - alpha) * self.stats['avg_batch_size']
        )

        # Running average for latency
        self.stats['avg_latency_ms'] = (
            alpha * inference_time_ms + (1 - alpha) * self.stats['avg_latency_ms']
        )

        # Throughput (FPS)
        if inference_time_ms > 0:
            batch_fps = (batch_size / inference_time_ms) * 1000
            self.stats['throughput_fps'] = (
                alpha * batch_fps + (1 - alpha) * self.stats['throughput_fps']
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats.copy()

    def log_stats(self):
        """Log current statistics."""
        stats = self.get_stats()
        logger.info(
            f"Batch Stats - "
            f"Requests: {stats['total_requests']}, "
            f"Batches: {stats['total_batches']}, "
            f"Avg Batch Size: {stats['avg_batch_size']:.2f}, "
            f"Avg Latency: {stats['avg_latency_ms']:.2f} ms, "
            f"Throughput: {stats['throughput_fps']:.2f} FPS"
        )


class StreamBatcher:
    """
    Stream-based batch processor using CUDA streams.

    Features:
    - Multiple CUDA streams for parallel execution
    - Overlapped data transfer and compute
    - Maximum GPU utilization
    """

    def __init__(
        self,
        batch_size: int,
        num_streams: int = 4,
        device_id: int = 0
    ):
        """
        Initialize stream batcher.

        Args:
            batch_size: Fixed batch size
            num_streams: Number of CUDA streams
            device_id: GPU device ID
        """
        self.batch_size = batch_size
        self.num_streams = num_streams
        self.device = torch.device(f'cuda:{device_id}')

        # Create CUDA streams
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]

        # Pinned memory buffers for fast H2D transfer
        self.input_buffers = []
        self.output_buffers = []

        logger.info(f"Initialized StreamBatcher with {num_streams} streams")

    def allocate_buffers(self, input_shape: Tuple, output_shape: Tuple):
        """
        Pre-allocate pinned memory buffers.

        Args:
            input_shape: Input tensor shape (excluding batch dim)
            output_shape: Output tensor shape (excluding batch dim)
        """
        for i in range(self.num_streams):
            # Allocate pinned memory for fast transfer
            input_buffer = torch.empty(
                (self.batch_size, *input_shape),
                dtype=torch.float32,
                device='cpu',
                pin_memory=True
            )
            output_buffer = torch.empty(
                (self.batch_size, *output_shape),
                dtype=torch.float32,
                device='cpu',
                pin_memory=True
            )

            self.input_buffers.append(input_buffer)
            self.output_buffers.append(output_buffer)

        logger.info("Allocated pinned memory buffers")

    def process_stream(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        stream_idx: int
    ) -> torch.Tensor:
        """
        Process batch on specific stream.

        Args:
            model: PyTorch model
            input_data: Input batch (on CPU)
            stream_idx: Stream index

        Returns:
            Output batch (on CPU)
        """
        stream = self.streams[stream_idx]
        input_buffer = self.input_buffers[stream_idx]
        output_buffer = self.output_buffers[stream_idx]

        with torch.cuda.stream(stream):
            # H2D transfer (async)
            input_buffer.copy_(input_data, non_blocking=True)
            input_gpu = input_buffer.to(self.device, non_blocking=True)

            # Inference
            with torch.no_grad():
                output_gpu = model(input_gpu)

            # D2H transfer (async)
            output_buffer.copy_(output_gpu, non_blocking=True)

        # Synchronize stream
        stream.synchronize()

        return output_buffer

    def process_pipeline(
        self,
        model: torch.nn.Module,
        data_iterator: iter
    ) -> List[torch.Tensor]:
        """
        Process data with pipelined execution across multiple streams.

        Args:
            model: PyTorch model
            data_iterator: Iterator yielding batches

        Returns:
            List of output batches
        """
        results = []
        stream_idx = 0

        for batch in data_iterator:
            # Process on next available stream
            result = self.process_stream(model, batch, stream_idx)
            results.append(result)

            # Round-robin stream selection
            stream_idx = (stream_idx + 1) % self.num_streams

        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()

        return results


class BatchSizeOptimizer:
    """
    Automatically find optimal batch size for maximum throughput.

    Uses binary search to find largest batch size that doesn't OOM.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        device_id: int = 0
    ):
        """
        Initialize batch size optimizer.

        Args:
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            device_id: GPU device ID
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.device = torch.device(f'cuda:{device_id}')

    def find_optimal_batch_size(
        self,
        model: torch.nn.Module,
        input_shape: Tuple,
        num_warmup: int = 5,
        num_test: int = 10
    ) -> int:
        """
        Find optimal batch size using binary search.

        Args:
            model: PyTorch model
            input_shape: Input shape (without batch dimension)
            num_warmup: Warmup iterations
            num_test: Test iterations per batch size

        Returns:
            Optimal batch size
        """
        logger.info("Finding optimal batch size...")

        model.eval()
        model.to(self.device)

        left = self.min_batch_size
        right = self.max_batch_size
        optimal = self.min_batch_size

        while left <= right:
            mid = (left + right) // 2

            try:
                # Test this batch size
                torch.cuda.empty_cache()
                success = self._test_batch_size(model, input_shape, mid, num_warmup, num_test)

                if success:
                    optimal = mid
                    left = mid + 1  # Try larger
                    logger.info(f"Batch size {mid}: SUCCESS")
                else:
                    right = mid - 1  # Try smaller
                    logger.info(f"Batch size {mid}: FAILED")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    right = mid - 1
                    torch.cuda.empty_cache()
                    logger.info(f"Batch size {mid}: OOM")
                else:
                    raise

        logger.info(f"Optimal batch size: {optimal}")
        return optimal

    def _test_batch_size(
        self,
        model: torch.nn.Module,
        input_shape: Tuple,
        batch_size: int,
        num_warmup: int,
        num_test: int
    ) -> bool:
        """
        Test if batch size works without OOM.

        Returns:
            True if successful, False if OOM
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape, device=self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy_input)

            # Test
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_test):
                    _ = model(dummy_input)
                    torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            throughput = (batch_size * num_test) / elapsed

            logger.info(f"  Throughput: {throughput:.2f} samples/sec")

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e):
                return False
            raise


# Example usage
if __name__ == "__main__":
    # Dummy inference function
    def dummy_inference(batch: np.ndarray) -> np.ndarray:
        time.sleep(0.01)  # Simulate 10ms inference
        return np.zeros((batch.shape[0], 10))  # Dummy output

    # Test dynamic batcher
    config = BatchConfig(
        min_batch_size=1,
        max_batch_size=8,
        timeout_ms=50.0,
        dynamic_batching=True
    )

    batcher = DynamicBatcher(config, dummy_inference)
    batcher.start()

    # Submit requests
    results = []
    for i in range(20):
        request = InferenceRequest(
            data=np.random.randn(3, 224, 224).astype(np.float32),
            request_id=f"req_{i}",
            timestamp=time.perf_counter()
        )
        response = batcher.submit(request)
        results.append(response)
        logger.info(f"Request {i}: latency={response.latency_ms:.2f}ms, batch_size={response.batch_size}")

    batcher.log_stats()
    batcher.stop()
