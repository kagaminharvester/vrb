"""
TensorRT Engine Management for Maximum Inference Performance
Optimized for NVIDIA RTX 3090

This module handles model conversion, optimization, and inference using TensorRT,
achieving 2-10x speedup over native PyTorch inference.
"""

import tensorrt as trt
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Dict, Optional, Union
import logging
import os
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class TensorRTEngine:
    """
    TensorRT engine manager with dynamic shape support and mixed precision.

    Features:
    - Automatic PyTorch -> ONNX -> TensorRT conversion
    - Dynamic batch size support
    - FP16/INT8 quantization
    - Multiple optimization profiles
    - Persistent engine caching
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        max_workspace_size: int = 8,  # GB
        fp16_mode: bool = True,
        int8_mode: bool = False,
        device_id: int = 0
    ):
        """
        Initialize TensorRT engine.

        Args:
            engine_path: Path to saved engine file (or None to build new)
            max_workspace_size: Max workspace in GB for TensorRT
            fp16_mode: Enable FP16 precision
            int8_mode: Enable INT8 precision (requires calibration)
            device_id: GPU device ID
        """
        self.engine_path = engine_path
        self.max_workspace_size = max_workspace_size * (1024**3)
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.device_id = device_id

        # TensorRT objects
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = None
        self.context = None

        # I/O buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Binding metadata
        self.binding_shapes = {}
        self.binding_dtypes = {}

        if engine_path and os.path.exists(engine_path):
            self.load_engine(engine_path)

    def build_engine_from_pytorch(
        self,
        model: torch.nn.Module,
        input_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
        output_path: str,
        opset_version: int = 13,
        calibration_data: Optional[np.ndarray] = None
    ):
        """
        Build TensorRT engine from PyTorch model.

        Process: PyTorch -> ONNX -> TensorRT

        Args:
            model: PyTorch model
            input_shapes: Dict of {input_name: {"min": shape, "opt": shape, "max": shape}}
            output_path: Path to save engine
            opset_version: ONNX opset version
            calibration_data: Data for INT8 calibration (if int8_mode=True)
        """
        logger.info("Starting PyTorch -> ONNX -> TensorRT conversion")

        # Step 1: Convert PyTorch to ONNX
        onnx_path = output_path.replace('.trt', '.onnx')
        self._export_to_onnx(model, input_shapes, onnx_path, opset_version)

        # Step 2: Build TensorRT engine from ONNX
        self.build_engine_from_onnx(onnx_path, input_shapes, output_path, calibration_data)

        logger.info(f"Engine saved to {output_path}")

    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
        output_path: str,
        opset_version: int
    ):
        """Export PyTorch model to ONNX format."""
        model.eval()
        device = torch.device(f'cuda:{self.device_id}')
        model.to(device)

        # Create dummy input with optimal shape
        dummy_inputs = {}
        for name, shapes in input_shapes.items():
            opt_shape = shapes['opt']
            dummy_inputs[name] = torch.randn(opt_shape, device=device)

        # Export with dynamic axes
        dynamic_axes = {}
        for name, shapes in input_shapes.items():
            # Typically batch dimension (0) is dynamic
            dynamic_axes[name] = {0: 'batch'}

        # Get output names
        with torch.no_grad():
            outputs = model(**dummy_inputs) if isinstance(dummy_inputs, dict) else model(*dummy_inputs.values())
            if isinstance(outputs, torch.Tensor):
                output_names = ['output']
            elif isinstance(outputs, (list, tuple)):
                output_names = [f'output_{i}' for i in range(len(outputs))]
            else:
                output_names = list(outputs.keys())

        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()) if len(dummy_inputs) > 1 else list(dummy_inputs.values())[0],
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        logger.info(f"Exported ONNX model to {output_path}")

    def build_engine_from_onnx(
        self,
        onnx_path: str,
        input_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
        output_path: str,
        calibration_data: Optional[np.ndarray] = None
    ):
        """
        Build TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX model
            input_shapes: Dynamic shape specifications
            output_path: Path to save TensorRT engine
            calibration_data: Data for INT8 calibration
        """
        logger.info(f"Building TensorRT engine from {onnx_path}")

        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size

        # Enable precision modes
        if self.fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")

        if self.int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data is not None:
                calibrator = self._create_int8_calibrator(calibration_data)
                config.int8_calibrator = calibrator
                logger.info("INT8 mode enabled with calibration")
            else:
                logger.warning("INT8 mode requested but no calibration data provided")

        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        for input_name, shapes in input_shapes.items():
            profile.set_shape(
                input_name,
                min=shapes['min'],
                opt=shapes['opt'],
                max=shapes['max']
            )
            logger.info(f"Set shape for {input_name}: min={shapes['min']}, opt={shapes['opt']}, max={shapes['max']}")

        config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Serialize and save
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT engine built and saved to {output_path}")

        # Load the engine
        self.load_engine(output_path)

    def load_engine(self, engine_path: str):
        """
        Load pre-built TensorRT engine.

        Args:
            engine_path: Path to serialized engine
        """
        logger.info(f"Loading TensorRT engine from {engine_path}")

        runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

        logger.info(f"Engine loaded successfully with {self.engine.num_bindings} bindings")

    def _allocate_buffers(self):
        """Allocate GPU memory for I/O tensors."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            # Store metadata
            self.binding_shapes[binding] = self.engine.get_binding_shape(binding)
            self.binding_dtypes[binding] = dtype

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
                logger.info(f"Input binding: {binding}, shape: {self.binding_shapes[binding]}, dtype: {dtype}")
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                logger.info(f"Output binding: {binding}, shape: {self.binding_shapes[binding]}, dtype: {dtype}")

    def set_dynamic_shape(self, input_idx: int, shape: Tuple[int, ...]):
        """
        Set dynamic input shape for this inference.

        Args:
            input_idx: Input index
            shape: New input shape
        """
        binding_name = self.engine.get_binding_name(input_idx)
        self.context.set_binding_shape(input_idx, shape)
        self.binding_shapes[binding_name] = shape

    def infer(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]],
        batch_size: Optional[int] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run inference on input data.

        Args:
            input_data: Input tensor(s) as numpy array or dict
            batch_size: Explicit batch size (optional)

        Returns:
            Output tensor(s) as numpy array or list
        """
        # Handle single input vs multiple inputs
        if isinstance(input_data, dict):
            inputs_list = list(input_data.values())
        elif isinstance(input_data, np.ndarray):
            inputs_list = [input_data]
        else:
            inputs_list = input_data

        # Update dynamic shapes if needed
        for i, input_tensor in enumerate(inputs_list):
            current_shape = input_tensor.shape
            if current_shape != tuple(self.binding_shapes[self.engine.get_binding_name(i)]):
                self.set_dynamic_shape(i, current_shape)

        # Copy input data to GPU
        for i, input_tensor in enumerate(inputs_list):
            np.copyto(self.inputs[i]['host'], input_tensor.ravel())
            cuda.memcpy_htod_async(
                self.inputs[i]['device'],
                self.inputs[i]['host'],
                self.stream
            )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output data from GPU
        outputs = []
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(
                self.outputs[i]['host'],
                self.outputs[i]['device'],
                self.stream
            )
            self.stream.synchronize()

            # Reshape output
            output_name = self.engine.get_binding_name(len(self.inputs) + i)
            output_shape = self.context.get_binding_shape(len(self.inputs) + i)
            output = self.outputs[i]['host'].reshape(output_shape)
            outputs.append(output)

        return outputs[0] if len(outputs) == 1 else outputs

    def infer_async(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]]
    ):
        """
        Start async inference (non-blocking).

        Args:
            input_data: Input tensor(s)
        """
        # Similar to infer() but without synchronization
        if isinstance(input_data, dict):
            inputs_list = list(input_data.values())
        elif isinstance(input_data, np.ndarray):
            inputs_list = [input_data]
        else:
            inputs_list = input_data

        # Update shapes
        for i, input_tensor in enumerate(inputs_list):
            current_shape = input_tensor.shape
            if current_shape != tuple(self.binding_shapes[self.engine.get_binding_name(i)]):
                self.set_dynamic_shape(i, current_shape)

        # Copy to GPU (async)
        for i, input_tensor in enumerate(inputs_list):
            np.copyto(self.inputs[i]['host'], input_tensor.ravel())
            cuda.memcpy_htod_async(
                self.inputs[i]['device'],
                self.inputs[i]['host'],
                self.stream
            )

        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

    def get_async_results(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get results from async inference (blocking).

        Returns:
            Output tensor(s)
        """
        outputs = []
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(
                self.outputs[i]['host'],
                self.outputs[i]['device'],
                self.stream
            )
            self.stream.synchronize()

            output_name = self.engine.get_binding_name(len(self.inputs) + i)
            output_shape = self.context.get_binding_shape(len(self.inputs) + i)
            output = self.outputs[i]['host'].reshape(output_shape)
            outputs.append(output)

        return outputs[0] if len(outputs) == 1 else outputs

    def warmup(self, num_iterations: int = 10):
        """
        Warmup engine with dummy inputs.

        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up TensorRT engine ({num_iterations} iterations)")

        # Create dummy input
        for input_buffer in self.inputs:
            input_buffer['host'].fill(0.5)

        for _ in range(num_iterations):
            self.infer(self.inputs[0]['host'].reshape(
                self.binding_shapes[self.engine.get_binding_name(0)]
            ))

        logger.info("Warmup complete")

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark engine performance.

        Args:
            num_iterations: Number of benchmark iterations

        Returns:
            Performance metrics
        """
        import time

        # Warmup
        self.warmup(10)

        # Create dummy input
        dummy_input = np.random.randn(
            *self.binding_shapes[self.engine.get_binding_name(0)]
        ).astype(np.float32)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            self.infer(dummy_input)
        elapsed = time.perf_counter() - start

        avg_latency = (elapsed / num_iterations) * 1000  # ms
        throughput = num_iterations / elapsed  # FPS

        results = {
            'avg_latency_ms': avg_latency,
            'throughput_fps': throughput,
            'total_time_s': elapsed,
            'iterations': num_iterations
        }

        logger.info(f"Benchmark results: {avg_latency:.2f} ms/frame, {throughput:.2f} FPS")

        return results

    def _create_int8_calibrator(self, calibration_data: np.ndarray):
        """Create INT8 calibrator for quantization."""
        class INT8Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data, cache_file='calibration.cache'):
                super().__init__()
                self.data = data
                self.cache_file = cache_file
                self.batch_size = data.shape[0]
                self.current_index = 0

                # Allocate device memory
                self.device_input = cuda.mem_alloc(data.nbytes)

            def get_batch_size(self):
                return self.batch_size

            def get_batch(self, names):
                if self.current_index < len(self.data):
                    batch = self.data[self.current_index:self.current_index + self.batch_size]
                    cuda.memcpy_htod(self.device_input, batch)
                    self.current_index += self.batch_size
                    return [int(self.device_input)]
                return None

            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, 'rb') as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)

        return INT8Calibrator(calibration_data)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'stream'):
            self.stream.synchronize()


# Example usage
if __name__ == "__main__":
    # Example: Build engine from PyTorch model
    # (Assuming you have a segmentation model)

    # Define dynamic input shapes
    input_shapes = {
        'input': {
            'min': (1, 3, 512, 512),
            'opt': (4, 3, 512, 512),  # Optimal batch size
            'max': (8, 3, 512, 512)
        }
    }

    # Initialize engine builder
    engine = TensorRTEngine(
        max_workspace_size=8,
        fp16_mode=True,
        int8_mode=False
    )

    # Uncomment to build from PyTorch model:
    # model = YourSegmentationModel()
    # engine.build_engine_from_pytorch(
    #     model=model,
    #     input_shapes=input_shapes,
    #     output_path='segmentation_fp16.trt'
    # )

    # Or load existing engine:
    # engine.load_engine('segmentation_fp16.trt')

    # Benchmark
    # results = engine.benchmark(num_iterations=100)
    # print(f"Performance: {results}")
