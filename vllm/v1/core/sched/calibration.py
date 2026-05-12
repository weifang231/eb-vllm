# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hardware parameter calibration for the P/D Competition Scheduler.

This module provides utilities to estimate hardware timing parameters
(α_p, β_p, α_d, β_d) by running benchmarks on the target hardware.

Usage:
    # Run calibration
    python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B --output params.json

    # Use calibrated parameters
    VLLM_PD_CALIBRATION_FILE=params.json vllm serve Qwen/Qwen3-8B
"""

import argparse
import json
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class HardwareParams:
    """Hardware timing parameters for the P/D Competition Scheduler.

    The timing models are:
        Prefill: T_p = α_p + β_p × L  (L = input tokens)
        Decode:  T_d = α_d + β_d × k  (k = batch size)

    All time values are in seconds.
    """
    # Core parameters
    alpha_p: float  # Prefill fixed overhead (seconds)
    beta_p: float   # Prefill per-token cost (seconds/token)
    alpha_d: float  # Decode fixed overhead (seconds)
    beta_d: float   # Decode per-batch cost (seconds/request)

    # Metadata
    model: str = ""
    device_name: str = ""
    dtype: str = "float16"
    timestamp: str = ""

    # Fitting quality metrics
    prefill_r2: float = 0.0  # R² score for prefill fit
    decode_r2: float = 0.0   # R² score for decode fit

    # Calibration config
    prefill_sizes: list[int] = field(default_factory=list)
    decode_counts: list[int] = field(default_factory=list)
    num_iterations: int = 0

    def save(self, filepath: str | Path) -> None:
        """Save parameters to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Hardware parameters saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "HardwareParams":
        """Load parameters from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)

    def __str__(self) -> str:
        return (
            f"HardwareParams(\n"
            f"  α_p={self.alpha_p:.6f}s, β_p={self.beta_p:.8f}s/token\n"
            f"  α_d={self.alpha_d:.6f}s, β_d={self.beta_d:.8f}s/req\n"
            f"  model={self.model}, device={self.device_name}\n"
            f"  prefill_r2={self.prefill_r2:.4f}, decode_r2={self.decode_r2:.4f}\n"
            f")"
        )


def get_device_name() -> str:
    """Get GPU device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or "unknown"


class GPUTimer:
    """Accurate GPU timing using CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()

    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Simple linear regression: y = alpha + beta * x.

    Returns:
        (alpha, beta, r2_score)
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute coefficients
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return float(y_mean), 0.0, 0.0

    beta = numerator / denominator
    alpha = y_mean - beta * x_mean

    # Compute R² score
    y_pred = alpha + beta * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return float(alpha), float(beta), float(r2)


class HardwareCalibrator:
    """Calibrates hardware parameters by running benchmarks."""

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        prefill_sizes: list[int] | None = None,
        decode_counts: list[int] | None = None,
        decode_context_len: int = 512,
        num_warmup: int = 5,
        num_iterations: int = 20,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 1024,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model = model
        self.dtype = dtype
        self.prefill_sizes = prefill_sizes or [256, 512, 1024, 2048, 4096]
        self.decode_counts = decode_counts or [
            16, 32, 64, 128, 256, 384, 512, 768, 1024]
        self.decode_context_len = decode_context_len
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization

        self.engine_core = None

    def setup(self):
        """Initialize the engine."""
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.torch_utils import set_default_torch_num_threads
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor import Executor

        logger.info(f"Initializing model: {self.model}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  max_num_batched_tokens: {self.max_num_batched_tokens}")
        logger.info(f"  max_num_seqs: {self.max_num_seqs}")

        engine_args = EngineArgs(
            model=self.model,
            dtype=self.dtype,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
        )

        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        with set_default_torch_num_threads(1):
            self.engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )

        logger.info("Engine initialized.")

    def _add_request(self, prompt_len: int, max_tokens: int = 1000) -> str:
        """Add a new request to the engine."""
        from vllm import SamplingParams
        from vllm.v1.engine import EngineCoreRequest

        assert self.engine_core is not None

        req_id = f"req_{uuid.uuid4().hex[:8]}"
        prompt_token_ids = list(np.random.randint(100, 10000, size=prompt_len))

        request = EngineCoreRequest(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        self.engine_core.add_request(
            *self.engine_core.preprocess_add_request(request)
        )
        return req_id

    def _run_single_step(self) -> tuple[int, int, int, float]:
        """Run a single step and measure time.

        Returns:
            (num_decode, num_prefill, total_tokens, time_ms)
        """
        assert self.engine_core is not None

        scheduler_output = self.engine_core.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return 0, 0, 0, 0.0

        num_decode = 0
        num_prefill = 0
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_tokens == 1:
                num_decode += 1
            else:
                num_prefill += 1

        total_tokens = scheduler_output.total_num_scheduled_tokens

        timer = GPUTimer()
        timer.start()
        model_output = self.engine_core.model_executor.execute_model(
            scheduler_output, non_block=False
        )
        if model_output is None:
            model_output = self.engine_core.model_executor.sample_tokens(None)
        elapsed = timer.stop()

        self.engine_core.scheduler.update_from_output(scheduler_output, model_output)

        return num_decode, num_prefill, total_tokens, elapsed

    def _cleanup_all_requests(self):
        """Clean up all requests."""
        if self.engine_core is None:
            return

        running_ids = [req.request_id for req in self.engine_core.scheduler.running]
        waiting_ids = [req.request_id for req in self.engine_core.scheduler.waiting]
        all_ids = running_ids + waiting_ids

        if all_ids:
            self.engine_core.abort_requests(all_ids)

        while self.engine_core.scheduler.has_requests():
            try:
                self.engine_core.step_fn()
            except Exception:
                break

    def _prepare_decode_requests(self, num_decode: int) -> bool:
        """Prepare decode requests by completing their prefill phase."""
        if num_decode == 0:
            return True

        assert self.engine_core is not None

        if num_decode > self.max_num_seqs:
            return False

        for _ in range(num_decode):
            self._add_request(self.decode_context_len, max_tokens=100000)

        tokens_to_process = self.decode_context_len * num_decode
        max_steps = (tokens_to_process // self.max_num_batched_tokens) + 100

        for _ in range(max_steps):
            num_running = len(self.engine_core.scheduler.running)
            if num_running >= num_decode:
                all_in_decode = all(
                    req.num_computed_tokens >= req.num_prompt_tokens
                    for req in self.engine_core.scheduler.running
                )
                if all_in_decode:
                    break
            try:
                self.engine_core.step_fn()
            except (KeyError, RuntimeError) as e:
                logger.warning(f"Error during prefill preparation: {e}")
                return False

        num_in_decode = sum(
            1 for req in self.engine_core.scheduler.running
            if req.num_computed_tokens >= req.num_prompt_tokens
        )

        return num_in_decode >= num_decode

    def _global_warmup(self):
        """Run a global warmup to ensure GPU is in hot state."""
        assert self.engine_core is not None

        num_warmup_iters = 20
        warmup_prefill_size = max(self.prefill_sizes)

        logger.info("Running global GPU warmup...")
        for _ in range(num_warmup_iters):
            self._cleanup_all_requests()
            self._add_request(warmup_prefill_size, max_tokens=10)
            for _ in range(5):
                if not self.engine_core.scheduler.has_requests():
                    break
                self.engine_core.step_fn()

        self._cleanup_all_requests()
        torch.cuda.synchronize()
        logger.info("Warmup complete.")

    def measure_prefill_times(self) -> list[tuple[int, float]]:
        """Measure prefill times for different input sizes.

        Returns:
            List of (prefill_size, time_ms) tuples
        """
        results = []

        logger.info("Measuring prefill times...")
        for prefill_size in self.prefill_sizes:
            times = []

            for i in range(self.num_warmup + self.num_iterations):
                self._cleanup_all_requests()

                # Small warmup before each measurement
                self._add_request(256, max_tokens=5)
                for _ in range(3):
                    if not self.engine_core.scheduler.has_requests():
                        break
                    self.engine_core.step_fn()
                self._cleanup_all_requests()

                # Actual measurement
                self._add_request(prefill_size, max_tokens=1)
                num_d, num_p, total_tokens, elapsed = self._run_single_step()

                if i >= self.num_warmup and total_tokens > 0:
                    times.append(elapsed)

            if times:
                median_time = float(np.median(times))
                results.append((prefill_size, median_time))
                logger.info(f"  Prefill {prefill_size} tokens: {median_time:.3f}ms")

        return results

    def measure_decode_times(self) -> list[tuple[int, float]]:
        """Measure decode times for different batch sizes.

        Returns:
            List of (batch_size, time_ms) tuples
        """
        results = []

        logger.info("Measuring decode times...")
        for num_decode in self.decode_counts:
            times = []

            for i in range(self.num_warmup + self.num_iterations):
                self._cleanup_all_requests()

                success = self._prepare_decode_requests(num_decode)
                if not success:
                    logger.warning(f"  Could not prepare {num_decode} decode requests")
                    break

                num_d, num_p, total_tokens, elapsed = self._run_single_step()

                if i >= self.num_warmup and num_d > 0:
                    times.append(elapsed)

            if times:
                median_time = float(np.median(times))
                results.append((num_decode, median_time))
                logger.info(f"  Decode batch={num_decode}: {median_time:.3f}ms")

        return results

    def calibrate(self) -> HardwareParams:
        """Run full calibration and return hardware parameters."""
        self._global_warmup()

        # Measure prefill and decode times
        prefill_data = self.measure_prefill_times()
        decode_data = self.measure_decode_times()

        if len(prefill_data) < 2:
            raise RuntimeError("Not enough prefill measurements for fitting")
        if len(decode_data) < 2:
            raise RuntimeError("Not enough decode measurements for fitting")

        # Convert to numpy arrays (times in seconds)
        prefill_sizes = np.array([d[0] for d in prefill_data])
        prefill_times = np.array([d[1] / 1000 for d in prefill_data])  # ms -> s

        decode_counts = np.array([d[0] for d in decode_data])
        decode_times = np.array([d[1] / 1000 for d in decode_data])  # ms -> s

        # Linear regression
        alpha_p, beta_p, r2_p = linear_regression(prefill_sizes, prefill_times)
        alpha_d, beta_d, r2_d = linear_regression(decode_counts, decode_times)

        # Ensure non-negative values
        alpha_p = max(0, alpha_p)
        beta_p = max(0, beta_p)
        alpha_d = max(0, alpha_d)
        beta_d = max(0, beta_d)

        params = HardwareParams(
            alpha_p=alpha_p,
            beta_p=beta_p,
            alpha_d=alpha_d,
            beta_d=beta_d,
            model=self.model,
            device_name=get_device_name(),
            dtype=self.dtype,
            timestamp=datetime.now().isoformat(),
            prefill_r2=r2_p,
            decode_r2=r2_d,
            prefill_sizes=self.prefill_sizes,
            decode_counts=self.decode_counts,
            num_iterations=self.num_iterations,
        )

        logger.info(f"\nCalibration complete:")
        logger.info(f"  Prefill: T_p = {alpha_p:.6f} + {beta_p:.8f} × L  (R²={r2_p:.4f})")
        logger.info(f"  Decode:  T_d = {alpha_d:.6f} + {beta_d:.8f} × k  (R²={r2_d:.4f})")

        return params


def calibrate_hardware_params(
    model: str,
    dtype: str = "float16",
    prefill_sizes: list[int] | None = None,
    decode_counts: list[int] | None = None,
    decode_context_len: int = 512,
    num_warmup: int = 5,
    num_iterations: int = 20,
    max_num_batched_tokens: int = 16384,
    max_num_seqs: int = 1024,
    gpu_memory_utilization: float = 0.9,
    output_file: str | None = None,
) -> HardwareParams:
    """
    Convenience function to run calibration.

    Args:
        model: Model name or path
        dtype: Model dtype
        prefill_sizes: List of prefill sizes to test
        decode_counts: List of decode batch sizes to test
        decode_context_len: Context length for decode requests
        num_warmup: Number of warmup iterations
        num_iterations: Number of measurement iterations
        max_num_batched_tokens: Max tokens per batch
        max_num_seqs: Max sequences
        gpu_memory_utilization: GPU memory utilization
        output_file: Optional path to save results

    Returns:
        HardwareParams with calibrated values
    """
    calibrator = HardwareCalibrator(
        model=model,
        dtype=dtype,
        prefill_sizes=prefill_sizes,
        decode_counts=decode_counts,
        decode_context_len=decode_context_len,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    calibrator.setup()
    params = calibrator.calibrate()

    if output_file:
        params.save(output_file)

    return params


def load_hardware_params(filepath: str | Path | None = None) -> HardwareParams | None:
    """
    Load hardware parameters from file.

    Checks in order:
    1. Provided filepath
    2. VLLM_PD_CALIBRATION_FILE environment variable

    Returns None if no file is found.
    """
    if filepath is None:
        filepath = os.environ.get("VLLM_PD_CALIBRATION_FILE", "")

    if not filepath:
        return None

    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"Calibration file not found: {filepath}")
        return None

    try:
        params = HardwareParams.load(filepath)
        logger.info(f"Loaded hardware parameters from {filepath}")
        return params
    except Exception as e:
        logger.warning(f"Failed to load calibration file: {e}")
        return None


def main():
    """CLI entry point for calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate hardware parameters for the P/D Competition Scheduler"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--prefill-sizes",
        type=str,
        default="256,512,1024,2048,4096",
        help="Comma-separated list of prefill sizes to test",
    )
    parser.add_argument(
        "--decode-counts",
        type=str,
        default="16,32,64,128,256,384,512,768,1024",
        help="Comma-separated list of decode batch sizes to test",
    )
    parser.add_argument(
        "--decode-context-len",
        type=int,
        default=512,
        help="Context length for decode requests (default: 512)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of measurement iterations (default: 20)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="Max tokens per batch (default: 16384)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1024,
        help="Max sequences (default: 1024)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: pd_exp/outputs/pd_calibration_<model_short>.json)",
    )

    args = parser.parse_args()

    # 默认输出路径按模型区分
    if args.output is None:
        model_short = args.model.split("/")[-1]
        default_output = Path(__file__).parent.parent.parent.parent.parent / "pd_exp" / "outputs" / f"pd_calibration_{model_short}.json"
        args.output = str(default_output)

    prefill_sizes = [int(x) for x in args.prefill_sizes.split(",")]
    decode_counts = [int(x) for x in args.decode_counts.split(",")]

    params = calibrate_hardware_params(
        model=args.model,
        dtype=args.dtype,
        prefill_sizes=prefill_sizes,
        decode_counts=decode_counts,
        decode_context_len=args.decode_context_len,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_file=args.output,
    )

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(params)
    print("=" * 60)
    print(f"\nTo use these parameters, set:")
    print(f"  export VLLM_PD_CALIBRATION_FILE={args.output}")
    print(f"\nOr set environment variables directly:")
    print(f"  export VLLM_PD_ALPHA_P={params.alpha_p}")
    print(f"  export VLLM_PD_BETA_P={params.beta_p}")
    print(f"  export VLLM_PD_ALPHA_D={params.alpha_d}")
    print(f"  export VLLM_PD_BETA_D={params.beta_d}")


if __name__ == "__main__":
    main()
