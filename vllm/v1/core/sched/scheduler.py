# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ruff: noqa: G004
#
# This file has ~20 PD/EB scheduler diagnostic loggers that use multi-line
# f-strings to format Greek symbols (θ, α, β, Δ, η), scientific-notation
# floats, and arrow notation for state transitions (old->new). Converting
# them to lazy `%`-formatting (the upstream vLLM convention) would (a)
# substantially obfuscate the messages and (b) risk a fat-fingered argument
# order change that silently corrupts the diagnostic output. The cost of
# lazy formatting (deferred string interpolation) is negligible here since
# these run at most every M completions, not per-token.
import atexit
import itertools
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Final

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.calibration import load_hardware_params
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

# Absolute lower bound on the adaptive theta* threshold. Used by both the
# raw bisection solver (`_compute_optimal_ratio`) as a numerical safety
# floor and by the online controllers as a hard min on top of the
# user-configurable VLLM_PD_THETA_FLOOR (whose default is also this value).
# Anywhere theta is computed or clamped, the result must be >= this.
_PD_THETA_HARD_MIN: Final[float] = 0.01


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        # Scheduler mode: "v1" (default; vLLM v1 mixed-batching), "eb"
        # (pure exclusive batching), "auto" (EB+, the v1↔eb crossover scheduler).
        # VLLM_PD_SCHEDULER_MODE takes precedence over VLLM_USE_PD_SCHEDULER.
        _mode_env = os.environ.get("VLLM_PD_SCHEDULER_MODE", "")
        if _mode_env:
            self.scheduler_mode = _mode_env.lower()
        elif os.environ.get("VLLM_USE_PD_SCHEDULER", "0") == "1":
            self.scheduler_mode = "eb"
        else:
            self.scheduler_mode = "v1"

        # Backward-compatible flag — True for both "eb" and "auto"
        # so all existing PD state initialization and guards work unchanged
        self.use_pd_scheduler = self.scheduler_mode in ("eb", "auto")

        # P/D competition scheduling state - initialize for "eb" and "auto" modes
        if self.use_pd_scheduler:
            # N: batch size - number of requests to prefill before starting decode
            self.pd_batch_size_N = self.max_num_running_reqs

            # k* mode selection:
            #   - "direct": k* computed directly (default)
            #               if VLLM_PD_K_STAR is set, use that fixed k*; otherwise
            #               compute via Proposition 1
            #   - "ratio":  k* = theta* x N (ratio mode)
            #               if VLLM_PD_K_RATIO is set, use that fixed theta*; otherwise
            #               compute theta* dynamically
            self.pd_k_mode = os.environ.get("VLLM_PD_K_MODE", "direct")
            # direct-mode param: if VLLM_PD_K_STAR is set, use that fixed k*
            _k_star_env = os.environ.get("VLLM_PD_K_STAR", "")
            self.pd_k_star_user_specified = bool(_k_star_env)
            self.pd_k_star_fixed = int(_k_star_env) if _k_star_env else 0
            # ratio-mode param: if VLLM_PD_K_RATIO is set, use that fixed theta*
            _k_ratio_env = os.environ.get("VLLM_PD_K_RATIO", "")
            self.pd_k_ratio_user_specified = bool(_k_ratio_env)
            self.pd_k_ratio = float(_k_ratio_env) if _k_ratio_env else 0.0

            # auto mode needs a non-zero theta to evaluate the EB↔MB
            # crossover (see _evaluate_mode_switch). In direct k_mode the
            # online controllers never write pd_k_ratio, so unless the user
            # also sets VLLM_PD_K_RATIO or VLLM_PD_K_STAR the crossover is
            # silently skipped and auto degenerates to the cold-start mode.
            if (
                self.scheduler_mode == "auto"
                and self.pd_k_mode == "direct"
                and not self.pd_k_ratio_user_specified
                and not self.pd_k_star_user_specified
            ):
                logger.warning(
                    "VLLM_PD_SCHEDULER_MODE=auto with VLLM_PD_K_MODE=direct "
                    "and no VLLM_PD_K_RATIO / VLLM_PD_K_STAR override: the "
                    "MB↔EB crossover will never fire (pd_k_ratio stays 0) "
                    "and auto will behave like cold-start mode. Set "
                    "VLLM_PD_K_MODE=ifr (recommended) or cfr to enable "
                    "online switching."
                )

            # Hardware timing parameters (Proposition 1):
            #   Prefill: T_p = α_p + β_p * L (L = input tokens)
            #   Decode:  T_d = α_d + β_d * k (per decode step with batch size k)
            # Priority: calibration file > environment variables
            # NOTE: No defaults - calibration is REQUIRED for accurate scheduling
            _hw_params = load_hardware_params()
            if _hw_params is not None:
                self.pd_alpha_p = _hw_params.alpha_p
                self.pd_beta_p = _hw_params.beta_p
                self.pd_alpha_d = _hw_params.alpha_d
                self.pd_beta_d = _hw_params.beta_d
                logger.info(
                    "Loaded hardware params from calibration file: model=%s, device=%s",
                    _hw_params.model,
                    _hw_params.device_name,
                )
            else:
                # Check environment variables - no defaults allowed
                _alpha_p = os.environ.get("VLLM_PD_ALPHA_P")
                _beta_p = os.environ.get("VLLM_PD_BETA_P")
                _alpha_d = os.environ.get("VLLM_PD_ALPHA_D")
                _beta_d = os.environ.get("VLLM_PD_BETA_D")

                if not all([_alpha_p, _beta_p, _alpha_d, _beta_d]):
                    raise ValueError(
                        "PD Scheduler requires hardware calibration parameters. "
                        "Please run calibration first:\n"
                        "  python -m vllm.v1.core.sched.calibration "
                        "--model <model_name>\n"
                        "Or set environment variable:\n"
                        "  export VLLM_PD_CALIBRATION_FILE="
                        "/path/to/pd_calibration.json\n"
                        "Or set all timing parameters manually:\n"
                        "  export VLLM_PD_ALPHA_P=<value>\n"
                        "  export VLLM_PD_BETA_P=<value>\n"
                        "  export VLLM_PD_ALPHA_D=<value>\n"
                        "  export VLLM_PD_BETA_D=<value>"
                    )
                # `all([...])` above narrows these to non-None strings.
                assert _alpha_p is not None
                assert _beta_p is not None
                assert _alpha_d is not None
                assert _beta_d is not None
                self.pd_alpha_p = float(_alpha_p)
                self.pd_beta_p = float(_beta_p)
                self.pd_alpha_d = float(_alpha_d)
                self.pd_beta_d = float(_beta_d)

            # Workload parameter p: cold-start value assuming mean output length ~100
            # Will be replaced by actual measurement after first N requests complete
            self.pd_p = 0.01

            # IFR (Increasing Failure Rate) mode parameters
            # Used for online adaptive threshold selection (alg:adaptive_joint)
            # Must be initialized before k* mode selection below.
            # CFR (Constant Failure Rate) mode shares the same online
            # estimator state but pins η ≡ 0 and uses the exact-θ formula
            # together with a closed-form midpoint construction (not
            # described in the camera-ready paper; see the journal version).
            if self.pd_k_mode in ("ifr", "cfr"):
                # Sliding window of output length samples for hazard
                # rate estimation.  Keeps only the most recent W samples
                # so the estimator adapts to distribution shifts within
                # O(W) completions.
                self.pd_ifr_window_size = int(
                    os.environ.get("VLLM_PD_IFR_WINDOW_SIZE", "500")
                )
                self.pd_ifr_samples: deque[int] = deque(maxlen=self.pd_ifr_window_size)
                # M: Update interval (re-estimate every M completions)
                self.pd_ifr_update_interval = int(
                    os.environ.get("VLLM_PD_IFR_UPDATE_INTERVAL", "100")
                )
                # W_min: Minimum samples before estimation starts
                self.pd_ifr_min_samples = int(
                    os.environ.get("VLLM_PD_IFR_MIN_SAMPLES", "30")
                )
                # θ_default: Default theta during cold-start phase
                self.pd_ifr_default_theta = float(
                    os.environ.get("VLLM_PD_IFR_DEFAULT_THETA", "0.70")
                )
                # c: Independent update counter
                self.pd_ifr_update_counter = 0
                # Estimated hazard rate parameters: h(t) = p_0 + η * t
                self.pd_hazard_p0 = 0.01  # Base hazard rate (will be estimated)
                self.pd_hazard_eta = 0.0  # Hazard rate slope (η >= 0 for IFR)
                # Maximum theta to prevent excessive waiting
                self.pd_theta_max = float(os.environ.get("VLLM_PD_THETA_MAX", "0.80"))
                # Lower bound on the adaptive theta* (paper's theta_min;
                # see model.tex clipping rule). Default 0.01: paper-time
                # near-no-floor. Workload-specific clipping (0.3/0.7/0.85)
                # is no longer needed because a runtime KV-aware
                # Phase-1->2 gate (not in the camera-ready paper; see
                # journal version) handles the underlying phase-thrashing
                # at the root.
                self.pd_theta_floor = float(
                    os.environ.get("VLLM_PD_THETA_FLOOR", "0.01")
                )
                # EMA smoothing for θ* to damp oscillations from noisy
                # hazard-rate estimates.  α=0.3 means ~70% weight on
                # previous θ*, providing stability while still tracking
                # distribution shifts.
                self.pd_ifr_theta_ema_alpha = float(
                    os.environ.get("VLLM_PD_IFR_THETA_EMA_ALPHA", "0.3")
                )
                self.pd_ifr_theta_initialized = False

            # Initialize k* based on mode
            if self.pd_k_mode == "direct":
                # If k* is user-specified, use that fixed value;
                # otherwise compute the optimum.
                if self.pd_k_star_user_specified:
                    self.pd_switch_threshold_k = max(1, self.pd_k_star_fixed)
                else:
                    self.pd_switch_threshold_k = self._compute_optimal_k()
            elif self.pd_k_mode == "ratio":
                # If ratio is not user-specified, compute the initial theta*
                # from the asymptotic formula.
                if not self.pd_k_ratio_user_specified:
                    self.pd_k_ratio = self._compute_optimal_ratio()
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
            elif self.pd_k_mode == "ifr":
                # IFR mode: use default theta during cold-start phase
                # Will adapt based on hazard rate estimation as samples accumulate
                self.pd_k_ratio = self.pd_ifr_default_theta
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
            elif self.pd_k_mode == "cfr":
                # CFR mode (closed-form midpoint construction; not in
                # the camera-ready paper, see journal version).
                #
                # Cold start: use IFR's default θ until enough samples accumulate
                # to estimate p_0.  Once p_0 is estimated each update period,
                # we recompute (θ_0, k̂, N̂) from the exact-θ formula
                # (Eq. theta_base) and the midpoint construction
                # (Eq. midpoint_k); η is forced to 0 (CFR assumption).
                self.pd_k_ratio = self.pd_ifr_default_theta
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
                self.pd_cfr_initialized = False
                # CFR-extra state: μ_O estimator (EMA of avg_output_tokens
                # already exists as pd_avg_output_tokens; mu_L tracking is
                # added below for both auto-mode and stats reporting).
            else:
                logger.warning("Unknown k mode '%s', using direct mode", self.pd_k_mode)
                self.pd_k_mode = "direct"
                self.pd_switch_threshold_k = self._compute_optimal_k()

            # Log PD scheduler configuration
            if self.pd_k_mode == "direct":
                dyn_tag = "fixed" if self.pd_k_star_user_specified else "auto"
                k_info = f"k*={self.pd_switch_threshold_k} ({dyn_tag})"
            elif self.pd_k_mode == "ratio":
                dyn_tag = "fixed" if self.pd_k_ratio_user_specified else "auto"
                k_info = (
                    f"θ*={self.pd_k_ratio:.4f}, "
                    f"k*={self.pd_switch_threshold_k} ({dyn_tag})"
                )
            elif self.pd_k_mode == "cfr":
                k_info = (
                    f"θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} "
                    f"(CFR midpoint, θ_max={self.pd_theta_max})"
                )
            else:  # ifr
                k_info = (
                    f"θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} "
                    f"(IFR adaptive, θ_max={self.pd_theta_max})"
                )
            logger.info(
                f"[P/D Competition Scheduler] Initialized: "
                f"N={self.pd_batch_size_N}, k_mode={k_info}, "
                f"α_p={self.pd_alpha_p}, β_p={self.pd_beta_p}, "
                f"α_d={self.pd_alpha_d}, β_d={self.pd_beta_d}"
            )

            # Phase tracking:
            # Phase 0: Initial prefill - prefill N requests
            # Phase 1: Decode - switch when min(q,N-n)/n >= θ*/(1-θ*)
            # Phase 2: Refill prefill - prefill min(q,N-n) requests (no decode)
            # Then back to Phase 1
            self.pd_phase = 0

            # Counters
            self.pd_prefilled_count = 0  # Prefills completed in current batch
            self.pd_completed_decode_count = 0  # Decodes completed since last switch
            self.pd_refill_target = 0  # Number of requests to prefill in Phase 2

            # Track which requests are in decode phase
            self.pd_decoding_requests: set[str] = set()

            # Unified parameter update interval for p, avg_output_tokens, k*, θ*
            # All parameters update together every pd_param_update_interval requests
            self.pd_param_update_interval = int(
                os.environ.get("VLLM_PD_PARAM_UPDATE_INTERVAL", "100")
            )
            self.pd_ema_alpha = 0.2  # EMA smoothing factor
            self.pd_total_completed = 0  # Total completed requests (all time)
            self.pd_param_initialized = False  # First batch: direct assign, not EMA
            # Batch accumulators (reset after each parameter update)
            self.pd_batch_completed_count = 0  # Completed in current batch
            self.pd_batch_total_output_tokens = 0  # Sum of output tokens in batch

            # Track average output tokens with EMA for adaptive thresholds
            self.pd_avg_output_tokens = 100.0  # Initial estimate (cold start)
            # Base reserve ratio (minimum fraction of KV cache to reserve for decode)
            self.pd_base_kv_reserve = float(
                os.environ.get("VLLM_PD_BASE_KV_RESERVE", "0")
            )
            # Safety margin multiplier for output token estimation
            self.pd_output_margin = float(
                os.environ.get("VLLM_PD_OUTPUT_MARGIN", "0.5")
            )

            # N RECOVERY cooldown to prevent frequent updates
            self.pd_last_n_update_time = 0.0  # timestamp of last N update
            self.pd_n_update_cooldown = float(
                os.environ.get("VLLM_PD_N_UPDATE_COOLDOWN", "2.0")
            )  # seconds

            # μ_L (mean prompt length) EMA — used by both CFR midpoint
            # construction and the adaptive selector diagnostic Δ(N).
            self.pd_avg_prompt_len = 512.0
            self.pd_avg_prompt_ema_alpha = 0.05  # slow EMA

            # CFR / midpoint configuration
            # Dynamic memory-safe N̂ (Eq. eq:Nstar / Proposition prop:memory):
            # if enabled, the scheduler periodically re-derives the maximum
            # batch size from the KV-cache budget and the OOM tolerance ε.
            self.pd_auto_compute_n = (
                os.environ.get("VLLM_PD_AUTO_COMPUTE_N", "0") == "1"
            )
            self.pd_oom_tolerance = float(
                os.environ.get("VLLM_PD_OOM_TOLERANCE", "0.01")
            )
            # Cumulative count of requests that ran out of KV memory and had
            # to be preempted — surfaced in stats for OOM-rate validation.
            self.pd_oom_event_count = 0
            # Per-update snapshot history (paper Algorithm 1 trace; used by
            # validation / adaptive-selector analyzers).
            self.pd_update_history: list[dict] = []
            # Last computed midpoint diagnostics (logged into stats).
            self.pd_theta_zero_last = 0.0
            self.pd_k_hat_midpoint_last = 0.0
            self.pd_n_hat_safe_last = 0.0
            self.pd_delta_diagnostic_last = 0.0

            # Log adaptive settings
            logger.info(
                f"[P/D Adaptive] Initial: avg_output={self.pd_avg_output_tokens:.0f}, "
                f"base_kv_reserve={self.pd_base_kv_reserve:.2f}, "
                f"output_margin={self.pd_output_margin:.1f}, "
                f"auto_compute_N={self.pd_auto_compute_n}, "
                f"OOM_tol_eps={self.pd_oom_tolerance}"
            )
        else:
            logger.info("[Scheduler] Using original vLLM scheduler")

        # --- EB+ auto mode state ---
        if self.scheduler_mode == "auto":
            # Current active sub-scheduler in the EB+ state machine:
            # "mb" (mixed batching) or "eb" (exclusive batching).
            self._active_scheduler = os.environ.get(
                "VLLM_PD_AUTO_COLD_START_MODE", "mb"
            )

            # CP effective marginal cost: f(r) = a + b*r + c*r² (offline profiled)
            self._mb_cost_a = float(os.environ.get("VLLM_PD_MB_COST_A", "0"))
            self._mb_cost_b = float(os.environ.get("VLLM_PD_MB_COST_B", "0"))
            self._mb_cost_c = float(os.environ.get("VLLM_PD_MB_COST_C", "0"))
            self._mb_cost_profiled = any(
                os.environ.get(k)
                for k in ["VLLM_PD_MB_COST_A", "VLLM_PD_MB_COST_B", "VLLM_PD_MB_COST_C"]
            )

            # α_MB: defaults to α_p (paper approximation α_p ≈ α_d ≈ α_MB)
            _acp = os.environ.get("VLLM_PD_ALPHA_MB", "")
            self._alpha_mb = float(_acp) if _acp else self.pd_alpha_p

            # Hysteresis band and cooldown for mode switching
            self._mode_switch_delta = float(
                os.environ.get("VLLM_PD_MODE_SWITCH_DELTA", "0.0001")
            )
            self._mode_cooldown_max = int(os.environ.get("VLLM_PD_MODE_COOLDOWN", "3"))
            self._mode_cooldown = 0

            # Batch occupancy EMA (N_obs) — uses asymmetric EMA in schedule()
            self._n_obs = float(self.max_num_running_reqs)

            # Average prompt length EMA (μ_L tracking)
            self._avg_prompt_len = 512.0  # initial estimate
            self._avg_prompt_ema_alpha = 0.05  # slow EMA for prompt length

            # Mode switch tracking for stats/debugging
            self._mode_switch_history: list[dict] = []
            self._mode_switch_count = 0

            logger.info(
                f"[EB+] Auto mode initialized: "
                f"cold_start={self._active_scheduler}, "
                f"mb_cost_profiled={self._mb_cost_profiled}, "
                f"alpha_mb={self._alpha_mb:.6f}, "
                f"delta={self._mode_switch_delta}, "
                f"cooldown={self._mode_cooldown_max}"
            )

        self.chunk_prefilling: list[Request] = []

        # N update history: (timestamp, old_N, new_N, reason)
        self.pd_n_update_history: list[dict] = []
        self._pd_start_time = time.monotonic()

        # Performance metrics for parameter updates
        self._param_update_count = 0  # Number of cold path updates
        self._param_update_total_us = 0.0  # Total time spent in cold path (μs)
        self._last_param_update_us = 0.0  # Last cold path duration (μs)

        # Schedule statistics collection for analysis
        self._schedule_stats_enabled = (
            os.environ.get("VLLM_COLLECT_SCHEDULE_STATS", "0") == "1"
        )
        self._schedule_stats: list[dict] = []
        self._schedule_stats_start_time: float | None = None  # Set on first record
        self._schedule_stats_file = os.environ.get(
            "VLLM_SCHEDULE_STATS_FILE", "schedule_stats.json"
        )

        # Register atexit handler to save stats on shutdown
        if self._schedule_stats_enabled:
            atexit.register(self._save_stats_on_exit)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

    # Phase name constants for logging
    PD_PHASE_NAMES = {0: "INITIAL_PREFILL", 1: "DECODE", 2: "REFILL_PREFILL"}

    def get_pd_stats(self) -> dict:
        """Get current P/D scheduling statistics for monitoring."""
        stats = {
            "phase": self.pd_phase,
            "k_star": self.pd_switch_threshold_k,
            "k_ratio": self.pd_k_ratio,
            "k_ratio_user_specified": self.pd_k_ratio_user_specified,
            "k_mode": self.pd_k_mode,
            "N": self.pd_batch_size_N,
            "prefilled_count": self.pd_prefilled_count,
            "completed_decode_count": self.pd_completed_decode_count,
            "refill_target": self.pd_refill_target,
            "decoding_requests": len(self.pd_decoding_requests),
            "running_requests": len(self.running),
            "waiting_requests": len(self.waiting),
            "p": self.pd_p,
            "total_completed": self.pd_total_completed,
            "avg_output_tokens": self.pd_avg_output_tokens,
            "adaptive_kv_threshold": self._compute_adaptive_kv_threshold(),
            "adaptive_N": self._compute_adaptive_N(),
        }
        # Add IFR-specific stats if in IFR mode
        if self.pd_k_mode == "ifr":
            stats.update(
                {
                    "hazard_p0": self.pd_hazard_p0,
                    "hazard_eta": self.pd_hazard_eta,
                    "ifr_sample_count": len(self.pd_ifr_samples),
                    "ifr_update_counter": self.pd_ifr_update_counter,
                    "ifr_update_interval": self.pd_ifr_update_interval,
                    "ifr_window_size": self.pd_ifr_window_size,
                    "theta_max": self.pd_theta_max,
                }
            )
        return stats

    # @cprofile("compute_optimal_k.prof")
    def _compute_optimal_k(self) -> int:
        """
        Compute optimal switching threshold k* using Proposition 1.

        k* is the smallest integer k satisfying:
            k * τ(N-k) - Σ_{j=N-k+1}^{N} τ(j) >= α_p

        This maximizes throughput = k / (E[T_d(k)] + E[T_p(k)])

        Optimized: O(N) instead of O(N²) via τ precomputation + incremental sum.
        """
        N = self.pd_batch_size_N

        # Precompute all τ values: τ[0], τ[1], ..., τ[N]
        # τ(j) = (α_d + β_d * j) / (1 - (1-p)^j), τ[0] = inf
        one_minus_p = 1.0 - self.pd_p
        tau = []
        power = 1.0  # (1-p)^j, updated incrementally
        for j in range(N + 1):
            if j == 0:
                tau.append(float("inf"))
            else:
                power *= one_minus_p  # power = (1-p)^j
                denom = 1.0 - power
                if denom <= 0:
                    tau.append(float("inf"))
                else:
                    tau.append((self.pd_alpha_d + self.pd_beta_d * j) / denom)

        # Search with incremental sum: sum_tau accumulates τ[N-k+1] to τ[N]
        sum_tau = 0.0
        for k in range(1, N + 1):
            # Incrementally add τ[N-k+1] to sum
            sum_tau += tau[N - k + 1]

            # LHS: k * τ[N-k]
            lhs = k * tau[N - k]

            # RHS: Σ τ[j] + α_p
            rhs = sum_tau + self.pd_alpha_p

            if lhs >= rhs:
                return max(1, k)

        # If no k satisfies the condition, use N/5 as fallback
        return max(1, N // 5)

    def _compute_k_from_ratio(self) -> int:
        """
        Compute k* as a ratio of N.

        k* = pd_k_ratio * N

        This makes k* automatically adapt when N changes (e.g., due to
        adaptive N learning based on avg output tokens).

        Returns:
            int: k* value (at least 1)
        """
        k = int(self.pd_k_ratio * self.pd_batch_size_N)
        return max(1, k)

    def _compute_optimal_ratio(self) -> float:
        """
        Compute optimal ratio θ* using asymptotic formula (Proposition 1).

        For long sequences (p << 1) and moderate batch sizes (N << 1/p),
        the normalized threshold θ* = k*/N satisfies:

            θ/(1-θ) + ln(1-θ) = p * α_p / α_d

        This is solved using bisection method.

        Returns:
            float: optimal ratio θ* in (0, 1)
        """
        import math

        # Compute RHS: C = p * α_p / α_d
        C = self.pd_p * self.pd_alpha_p / self.pd_alpha_d

        # f(θ) = θ/(1-θ) + ln(1-θ)
        # We need to solve f(θ) = C
        def f(theta: float) -> float:
            if theta <= 0 or theta >= 1:
                return float("inf")
            return theta / (1 - theta) + math.log(1 - theta)

        # f(θ) is monotonically increasing from f(0+) = 0 to f(1-) = +∞
        # Use bisection to find θ such that f(θ) = C

        # Edge case: if C is very small, θ* ≈ 0
        if C <= 1e-6:
            return _PD_THETA_HARD_MIN  # numerical safety floor

        # Bisection search
        lo, hi = 0.001, 0.999
        for _ in range(100):  # Enough iterations for the required precision
            mid = (lo + hi) / 2
            f_mid = f(mid)
            if abs(f_mid - C) < 1e-9:
                break
            if f_mid < C:
                lo = mid
            else:
                hi = mid

        theta_star = (lo + hi) / 2

        # Clamp to reasonable range: [_PD_THETA_HARD_MIN, 0.99]. The 0.99
        # upper bound is the math solver's domain limit (theta -> 1 makes
        # ln(1-theta) blow up); the policy-side cap pd_theta_max (default
        # 0.80) is applied separately in the online update path.
        theta_star = max(_PD_THETA_HARD_MIN, min(0.99, theta_star))

        logger.debug(
            f"Computed optimal ratio: θ*={theta_star:.4f} "
            f"(p={self.pd_p}, α_p={self.pd_alpha_p}, "
            f"α_d={self.pd_alpha_d}, C={C:.6f})"
        )

        return theta_star

    # ================================================================
    # CFR midpoint algorithm (closed-form construction not in the
    # camera-ready paper; see the journal version for full derivation)
    # ----------------------------------------------------------------
    # The four functions below implement the closed-form midpoint
    # construction.  They are independent of the existing IFR / ratio
    # code path so the legacy schedulers remain bit-for-bit reproducible.
    # ================================================================

    def _compute_theta_zero_exact(self, p_0: float) -> float:
        """θ_0 from the exact CFR formula (Eq. theta_base):

            θ/(1-θ) + ln(1-θ) = (-ln(1-p_0)) · α_p / α_d.

        Differs from `_compute_optimal_ratio` (which uses p_0 directly on the
        right-hand side, valid only as p_0 → 0) by the exact -ln(1-p_0)
        factor required by the paper's CFR proofs.
        """
        import math

        if p_0 <= 0.0:
            return _PD_THETA_HARD_MIN
        if p_0 >= 1.0 - 1e-9:
            return 0.99

        rhs = (-math.log(1.0 - p_0)) * self.pd_alpha_p / self.pd_alpha_d
        if rhs <= 1e-12:
            return _PD_THETA_HARD_MIN

        def f(theta: float) -> float:
            if theta <= 0.0 or theta >= 1.0:
                return float("inf")
            return theta / (1.0 - theta) + math.log(1.0 - theta)

        lo, hi = 1e-6, 1.0 - 1e-6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < rhs:
                lo = mid
            else:
                hi = mid
        theta_zero = 0.5 * (lo + hi)
        return max(_PD_THETA_HARD_MIN, min(0.99, theta_zero))

    def _compute_midpoint_k(
        self, theta_zero: float, p_0: float, n: int, mu_L: float
    ) -> tuple[float, int, int, int]:
        """Midpoint k̂ following the closed-form midpoint construction.

        The R̄(m) / TP_int(m) / k̂ definitions used below are introduced
        in the journal version of the paper, not the camera-ready
        proceedings paper.

        Returns
        -------
        (k_hat, M_minus, M_plus, M_star) where k_hat is the (real-valued)
        midpoint placement; the integer threshold actually used is
        max(1, round(k_hat)).
        """
        import math

        if p_0 <= 0.0 or p_0 >= 1.0 or theta_zero <= 0.0 or theta_zero >= 1.0:
            # Fall back to floor(θ_0 N).
            return float(max(1, int(theta_zero * n))), 0, 0, 0
        if n <= 1:
            return 1.0, 0, 0, 0

        # τ_R* = ln(1-θ_0) / ln(1-p_0)
        log_one_minus_theta = math.log(1.0 - theta_zero)
        log_one_minus_p = math.log(1.0 - p_0)
        if log_one_minus_p == 0.0:
            return float(max(1, int(theta_zero * n))), 0, 0, 0
        tau_real = log_one_minus_theta / log_one_minus_p
        m_minus = max(1, int(math.floor(tau_real)))
        m_plus = max(m_minus + 1, int(math.ceil(tau_real)))

        def r_bar(m: int) -> float:
            if m <= 0:
                return 0.0
            return n * (1.0 - (1.0 - p_0) ** m)

        def tp_int(m: int) -> float:
            r = r_bar(m)
            denom = (
                self.pd_alpha_d * m
                + self.pd_beta_d * r / max(p_0, 1e-9)
                + self.pd_alpha_p
                + self.pd_beta_p * r * mu_L
            )
            return r / denom if denom > 0 else 0.0

        m_star = m_minus if tp_int(m_minus) >= tp_int(m_plus) else m_plus
        # k̂ = ½ (R̄(M*-1) + R̄(M*))
        k_hat = 0.5 * (r_bar(max(0, m_star - 1)) + r_bar(m_star))
        # Guard rails: keep within [1, N-1].
        k_hat = max(1.0, min(float(n - 1), k_hat))
        return k_hat, m_minus, m_plus, m_star

    def _compute_memory_safe_n(self, theta_zero: float, p_0: float, mu_L: float) -> int:
        """Memory-safe N̂ from Proposition prop:memory.

        NOTE on paper-vs-code:
        The main paper (Eq. eq:Nstar) presents the supremum bound
            x_ε = ν · ln(1/ε),    ν = 1/(p_0² · μ_L)
        giving the linearised closed form
            N* = ⌊ (C - ν · ln(1/ε)) / D(θ) ⌋.
        This implementation instead solves the CLT-type concentration
            N · D(θ) + σ(θ) · sqrt(N · ln(1/ε)) ≤ C
        for N (returning the floor of the positive root, clamped to
        [1, max_num_running_reqs]). The two bounds are asymptotically
        equivalent for N ≫ 1; the CLT form is tighter in the moderate-N
        regime relevant to our experiments, and is the form actually used
        to produce all paper results. The journal version of the paper
        adopts this quadratic form.

        Here:
            D(θ) = μ_L + (1-θ)/(θ p_0) · Λ        with Λ = -ln(1-θ)
            σ²(θ) = 2Λ (1 + (p_0 μ_L + Λ/θ)²) / p_0²
        and ε = `pd_oom_tolerance`.
        """
        import math

        n_cap = int(self.max_num_running_reqs)
        if not self.pd_auto_compute_n:
            # Auto-compute disabled — keep current N (typically the user's
            # --max-num-seqs).
            return n_cap

        # Capacity C: total KV-cache token slots advertised by the kv-cache
        # manager (in tokens, i.e. block_size × num_blocks).
        try:
            num_blocks = int(self.kv_cache_manager.block_pool.num_gpu_blocks)
            block_size = int(self.block_size)
            capacity_tokens = num_blocks * block_size
        except Exception:
            return n_cap
        if capacity_tokens <= 0:
            return n_cap

        eps = max(min(self.pd_oom_tolerance, 0.5), 1e-6)
        log_inv_eps = math.log(1.0 / eps)
        if theta_zero <= 0.0 or theta_zero >= 1.0 or p_0 <= 0.0:
            return n_cap

        # D(θ) = μ_L + (1-θ)/(θ p_0) ln(1/(1-θ))
        # D_max(θ) ≈ D(θ) when p_0 D(θ) ≥ 1 (paper assumption — true for
        # all real workloads in the paper's experiments).  We use D(θ) as
        # D_max(θ) for the closed-form bound.
        Lambda = -math.log(1.0 - theta_zero)
        D_theta = mu_L + (1.0 - theta_zero) / (theta_zero * p_0) * Lambda
        if D_theta <= 0:
            return n_cap

        # σ²(θ) = 2 Λ (1 + (p_0 μ_L + λ_θ)²) / p_0², λ_θ = Λ/θ
        lambda_theta = Lambda / max(theta_zero, 1e-9)
        sigma_sq = (
            2.0 * Lambda * (1.0 + (p_0 * mu_L + lambda_theta) ** 2) / max(p_0**2, 1e-18)
        )
        sigma = math.sqrt(max(sigma_sq, 0.0))

        # Solve N · D + σ √(N · log(1/ε)) ≤ C  →
        #   √N = ( -σ√L + √(σ² L + 4 D C) ) / (2 D)
        sqrt_N = (
            -sigma * math.sqrt(log_inv_eps)
            + math.sqrt(sigma_sq * log_inv_eps + 4.0 * D_theta * capacity_tokens)
        ) / (2.0 * D_theta)
        n_safe = int(math.floor(max(0.0, sqrt_N) ** 2))
        return max(1, min(n_cap, n_safe))

    def _compute_diagnostic_delta(
        self,
        theta_zero: float,
        k_hat: float,
        n: int,
        mu_L: float,
        mu_O: float,
    ) -> float:
        """Diagnostic Δ(N) for the adaptive selector.

        Equivalent to the inequality in Eq. eq:comparison_condition
        (Prop. 4, prop:comparison) rearranged as LHS - RHS:
        Δ(N) < 0  iff  MB throughput > EB throughput.

        Δ(N) = (β_MB^e − β_EB^w)
             − (1/(μ_L+μ_O)) · [ (α_p − α_d ln(1−θ_0) μ_O)/k̂
                               − α_MB(1+μ_O)/N ]

        The kernel-cost terms (β_MB^e, α_MB) are read from environment-
        provided calibration constants (VLLM_PD_BETA_MB_E, VLLM_PD_ALPHA_MB)
        — defaulting to 0 when not profiled (in which case Δ degenerates
        but the sign of the second term still drives MB/EB choice).
        """
        import math

        if mu_L + mu_O <= 0 or k_hat <= 0 or n <= 0:
            return 0.0

        beta_eb_w = (self.pd_beta_p * mu_L + self.pd_beta_d * mu_O) / (mu_L + mu_O)
        beta_mb_e = float(os.environ.get("VLLM_PD_BETA_MB_E", str(beta_eb_w)))
        alpha_mb = float(os.environ.get("VLLM_PD_ALPHA_MB", str(self.pd_alpha_p)))

        log_one_minus_theta = math.log(max(1.0 - theta_zero, 1e-9))
        eb_term = (
            self.pd_alpha_p - self.pd_alpha_d * log_one_minus_theta * mu_O
        ) / k_hat
        mb_term = alpha_mb * (1.0 + mu_O) / n

        return (beta_mb_e - beta_eb_w) - (eb_term - mb_term) / (mu_L + mu_O)

    def _estimate_hazard_params(self) -> tuple[float, float]:
        """
        Estimate hazard rate parameters (p_0, η) from sliding window samples.

        The empirical hazard rate at iteration t is:
            ĥ(t) = #{O_i = t} / #{O_i >= t}

        We fit h(t) = p_0 + η * t via weighted least squares over
        t ∈ [t_start, t_95], where t_start is the 5th percentile of
        observed output lengths.  Fitting only over the support of the
        distribution avoids the zero-hazard prefix that arises with
        bounded-support distributions (e.g. uniform, gamma with large
        shape), which would otherwise drag p_0 negative.

        Returns:
            tuple[float, float]: (p_0, η) where η >= 0 for IFR distributions
        """
        # Use sliding window samples for estimation
        samples = self.pd_ifr_samples
        if len(samples) < self.pd_ifr_min_samples:
            # Not enough samples, return current estimates
            return self.pd_hazard_p0, self.pd_hazard_eta

        # Compute fitting range: [t_start, t_95]
        # t_start = 5th percentile — skips the zero-hazard region before
        # the distribution's effective support begins.
        # t_95 = 95th percentile — avoids noisy tail estimates.
        sorted_samples = sorted(samples)
        t_start = sorted_samples[max(0, int(len(sorted_samples) * 0.05))]
        t_start = max(t_start, 1)
        t_95 = sorted_samples[int(len(sorted_samples) * 0.95)]
        t_95 = max(t_95, t_start + 10)  # Ensure enough range

        # Count occurrences and survivors
        from collections import Counter

        counts = Counter(samples)
        max_t = max(samples)

        # Compute survivors: #{O_i >= t} for each t
        survivors = [0] * (max_t + 2)
        survivors[max_t + 1] = 0
        for t in range(max_t, 0, -1):
            survivors[t] = survivors[t + 1] + counts.get(t, 0)

        # Compute empirical hazard rate and perform weighted least squares
        # h(t) = p_0 + η * t
        # Minimize: Σ w_t * (ĥ(t) - p_0 - η * t)^2
        sum_w = 0.0
        sum_wt = 0.0
        sum_wt2 = 0.0
        sum_wh = 0.0
        sum_wth = 0.0

        for t in range(t_start, min(t_95 + 1, max_t + 1)):
            n_t = survivors[t]
            if n_t < 5:  # Skip unreliable estimates
                continue
            d_t = counts.get(t, 0)
            h_t = d_t / n_t  # Empirical hazard at t

            w = n_t  # Weight by number of survivors
            sum_w += w
            sum_wt += w * t
            sum_wt2 += w * t * t
            sum_wh += w * h_t
            sum_wth += w * t * h_t

        if sum_w < 10:
            # Not enough valid data points
            return self.pd_hazard_p0, self.pd_hazard_eta

        # Solve normal equations for weighted least squares
        # [sum_w    sum_wt ] [p_0]   [sum_wh ]
        # [sum_wt   sum_wt2] [η  ] = [sum_wth]
        det = sum_w * sum_wt2 - sum_wt * sum_wt
        if abs(det) < 1e-10:
            # Singular matrix, use sample mean based estimate
            sample_mean = sum(samples) / len(samples)
            p_0 = 1.0 / sample_mean if sample_mean > 0 else 0.01
            return p_0, 0.0

        p_0 = (sum_wt2 * sum_wh - sum_wt * sum_wth) / det
        eta = (sum_w * sum_wth - sum_wt * sum_wh) / det

        # Ensure valid ranges
        eta = max(0.0, eta)  # η >= 0 for IFR (clamp negative to CFR)

        # Floor p_0 at the mean-based completion rate 1/μ_o.
        # For strongly IFR distributions (e.g. Gamma shape≥2), the WLS
        # intercept p_0 is near zero because h(0)≈0.  Using the raw
        # estimate would make θ_cfr vanishingly small and cause the IFR
        # correction Δθ ∝ η/p_0² to explode.  The geometric rate 1/μ_o
        # is a natural lower bound: it is the completion rate of a
        # memoryless process with the same mean output length.
        sample_mean = sum(samples) / len(samples)
        p_0_floor = (1.0 / sample_mean) if sample_mean > 0 else 0.01
        p_0 = max(p_0, p_0_floor)

        return p_0, eta

    def _compute_ifr_correction(self, theta_cfr: float) -> float:
        """
        Compute IFR correction Δθ based on the IFR threshold theorem.

        See `thm:threshold_ifr` in the paper.

        For linear increasing hazard rate h(t) = p_0 + η * t with η > 0,
        the optimal threshold admits:
            θ*_IFR = θ*_CFR + Δθ

        where:
            Δθ = (η(1-θ*_CFR)²) / (p_0² * θ*_CFR) *
                 [Λ(θ*_CFR/(1-θ*_CFR) - Λ/2) + ρ(Λ - θ*_CFR)]

        with Λ = -ln(1-θ*_CFR) and ρ = β_d * N / α_d.

        Args:
            theta_cfr: The CFR baseline threshold θ*_CFR

        Returns:
            float: The correction Δθ (always >= 0)
        """
        if self.pd_hazard_eta <= 0 or theta_cfr <= 0 or theta_cfr >= 1:
            return 0.0

        import math

        p_0 = self.pd_hazard_p0
        eta = self.pd_hazard_eta

        # Λ = -ln(1 - θ*_CFR)
        Lambda = -math.log(1 - theta_cfr)

        # ρ = β_d * N / α_d (per-token cost ratio)
        rho = self.pd_beta_d * self.pd_batch_size_N / self.pd_alpha_d

        # Duration effect: Λ * (θ*_CFR/(1-θ*_CFR) - Λ/2)
        duration_effect = Lambda * (theta_cfr / (1 - theta_cfr) - Lambda / 2)

        # Per-token cost effect: ρ * (Λ - θ*_CFR)
        per_token_effect = rho * (Lambda - theta_cfr)

        # Δθ = (η(1-θ*_CFR)²) / (p_0² * θ*_CFR) * [duration + per_token]
        numerator = eta * (1 - theta_cfr) ** 2
        denominator = p_0**2 * theta_cfr

        if denominator < 1e-12:
            return 0.0

        delta_theta = (numerator / denominator) * (duration_effect + per_token_effect)

        # Ensure non-negative (should always be positive for IFR)
        delta_theta = max(0.0, delta_theta)

        # Cap Δθ at 5·θ_cfr.  The first-order expansion is derived for
        # small η; when η/p_0² is large the uncapped correction can
        # exceed 1, making θ* meaningless.  The factor 5 allows the IFR
        # correction to dominate the CFR base (up to θ* ≤ 6·θ_cfr) while
        # preventing runaway values.
        delta_theta = min(delta_theta, 5.0 * theta_cfr)

        return delta_theta

    def _compute_optimal_ratio_ifr(self) -> float:
        """
        Compute optimal ratio θ* with IFR correction.

        This implements the online adaptive threshold selection
        (`alg:adaptive_joint` in the paper):
        1. Estimate hazard rate parameters (p_0, η) from samples
        2. Compute CFR baseline θ*_CFR using Proposition 1
        3. If η > 0, apply IFR correction from the IFR threshold
           theorem (`thm:threshold_ifr`)
        4. Return θ* = min(θ*_CFR + Δθ, θ_max)

        Returns:
            float: Optimal ratio θ* in (0, θ_max]
        """
        # Step 1: Estimate hazard rate parameters
        p_0, eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = eta

        # Step 2: Compute CFR baseline using p_0 (not self.pd_p)
        # Temporarily set pd_p to p_0 for _compute_optimal_ratio
        old_p = self.pd_p
        self.pd_p = p_0
        theta_cfr = self._compute_optimal_ratio()
        self.pd_p = old_p

        # Step 3: Apply IFR correction if η > 0
        if eta > 0:
            delta_theta = self._compute_ifr_correction(theta_cfr)
            theta_star = theta_cfr + delta_theta
        else:
            theta_star = theta_cfr

        # Step 4: Clamp to θ_max
        theta_star = min(theta_star, self.pd_theta_max)
        theta_star = max(_PD_THETA_HARD_MIN, theta_star)

        logger.debug(
            f"IFR optimal ratio: θ*={theta_star:.4f} "
            f"(θ*_CFR={theta_cfr:.4f}, Δθ={theta_star - theta_cfr:.4f}, "
            f"p_0={p_0:.6f}, η={eta:.8f})"
        )

        return theta_star

    def _update_ifr_threshold(self) -> None:
        """
        Online joint adaptation of (k̂*, N̂*) — paper Algorithm 1
        (alg:adaptive_joint).

        Called every M completions when window has >= W_min samples.
        Estimates (p̂_0, η̂, μ̂_L), computes θ_0 + Δθ → θ̂*, applies the
        memory-safe N̂* (Eq. eq:Nstar) when pd_auto_compute_n is enabled,
        and finally sets k̂* = ⌊θ̂* · N̂*⌋.

        EMA smoothing is applied to θ̂* to damp oscillations caused by
        noisy hazard-rate estimates.
        """
        old_ratio = self.pd_k_ratio
        old_k = self.pd_switch_threshold_k
        old_n = self.pd_batch_size_N

        # Step 1: Estimate hazard rate parameters from sliding window.
        p_0, eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = eta

        # Step 2: Compute CFR baseline θ_0 using p_0 (Eq. eq:theta_base).
        old_p = self.pd_p
        self.pd_p = p_0
        theta_0 = self._compute_optimal_ratio()
        self.pd_p = old_p

        # Step 3: Apply IFR correction Δθ if η > 0 (Eq. eq:delta_theta).
        if eta > 0:
            delta_theta = self._compute_ifr_correction(theta_0)
            theta_star = theta_0 + delta_theta
        else:
            delta_theta = 0.0
            theta_star = theta_0

        # Step 4: Clamp to [theta_floor (>= _PD_THETA_HARD_MIN), pd_theta_max].
        # theta_floor (env VLLM_PD_THETA_FLOOR, default _PD_THETA_HARD_MIN)
        # is a regularising lower bound on the adaptive controller — see
        # __init__ comment for why.
        theta_star = max(
            _PD_THETA_HARD_MIN,
            self.pd_theta_floor,
            min(theta_star, self.pd_theta_max),
        )

        # Step 5: EMA smoothing to damp oscillations from noisy estimates.
        # During cold start (first update), assign directly.
        if not self.pd_ifr_theta_initialized:
            self.pd_k_ratio = theta_star
            self.pd_ifr_theta_initialized = True
        else:
            alpha = self.pd_ifr_theta_ema_alpha
            self.pd_k_ratio = alpha * theta_star + (1 - alpha) * self.pd_k_ratio

        # Step 6: Memory-safe N̂* (paper Algorithm 1 line 163,
        # Eq. eq:Nstar / Proposition prop:memory). When pd_auto_compute_n
        # is disabled, _compute_memory_safe_n short-circuits and returns
        # the current cap, so this branch is a no-op in that case.
        mu_L = max(1.0, float(self.pd_avg_prompt_len))
        mu_O = max(1.0, float(self.pd_avg_output_tokens))
        if self.pd_auto_compute_n:
            n_hat = self._compute_memory_safe_n(self.pd_k_ratio, p_0, mu_L)
            if n_hat != self.pd_batch_size_N:
                self._record_n_update(
                    self.pd_batch_size_N, n_hat, "ifr_memory_safe"
                )
                self.pd_batch_size_N = n_hat
        n_eff = max(1, self.pd_batch_size_N)

        # Step 7: k̂* = ⌊θ̂* · N̂*⌋ (paper Algorithm 1 line 164).
        self.pd_switch_threshold_k = max(1, int(self.pd_k_ratio * n_eff))

        # Diagnostic Δ(N) — shared with the adaptive selector / EB+ auto
        # mode; computed from the CFR baseline θ_0 (Prop. prop:comparison).
        delta_diag = self._compute_diagnostic_delta(
            theta_0, float(self.pd_switch_threshold_k), n_eff, mu_L, mu_O
        )

        # Surface diagnostics for stats / analysis.
        self.pd_theta_zero_last = theta_0
        self.pd_k_hat_midpoint_last = float(self.pd_switch_threshold_k)
        self.pd_n_hat_safe_last = float(n_eff)
        self.pd_delta_diagnostic_last = delta_diag

        self.pd_update_history.append(
            {
                "timestamp": time.monotonic() - self._pd_start_time,
                "p_0_estimate": p_0,
                "eta_estimate": eta,
                "mu_L_estimate": mu_L,
                "mu_O_estimate": mu_O,
                "theta_0": theta_0,
                "delta_theta": delta_theta,
                "theta_star": self.pd_k_ratio,
                "k_hat_int": int(self.pd_switch_threshold_k),
                "N_hat": int(n_eff),
                "delta_diagnostic": delta_diag,
                "samples_used": len(self.pd_ifr_samples),
                "oom_event_count": int(self.pd_oom_event_count),
            }
        )

        if (
            abs(self.pd_k_ratio - old_ratio) > 0.01
            or old_k != self.pd_switch_threshold_k
            or old_n != self.pd_batch_size_N
        ):
            logger.info(
                f"[P/D IFR] online update: θ_0={theta_0:.4f}, "
                f"θ̂={old_ratio:.4f}->{self.pd_k_ratio:.4f} "
                f"(Δθ={delta_theta:.4f}), "
                f"k̂={old_k}->{self.pd_switch_threshold_k}, "
                f"N̂={old_n}->{self.pd_batch_size_N} "
                f"(p_0={p_0:.6f}, η={eta:.8f}, "
                f"samples={len(self.pd_ifr_samples)}, Δ={delta_diag:.6f})"
            )

    def _update_cfr_threshold(self) -> None:
        """Online CFR midpoint update (closed-form midpoint construction
        not in the camera-ready paper; see the journal version).

        Estimates p_0 from the sliding window, recomputes the exact θ_0
        (Eq. eq:theta_base), the memory-safe N̂ (Eq. eq:Nstar) when
        enabled, and the midpoint k̂ (journal-only formula).  Also
        evaluates the diagnostic Δ(N) used by the adaptive selector.
        """
        old_ratio = self.pd_k_ratio
        old_k = self.pd_switch_threshold_k
        old_n = self.pd_batch_size_N

        # Step 1: Estimate p_0 (η pinned to 0 in CFR).
        p_0, _eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = 0.0  # CFR assumption

        mu_L = max(1.0, float(self.pd_avg_prompt_len))
        mu_O = max(1.0, float(self.pd_avg_output_tokens))

        # Step 2: θ_0 from the exact formula.
        theta_zero = self._compute_theta_zero_exact(p_0)
        theta_zero_clamped = max(_PD_THETA_HARD_MIN, min(self.pd_theta_max, theta_zero))

        # Step 3: Optional dynamic N̂ from memory-safe sizing (Prop. memory).
        if self.pd_auto_compute_n:
            n_hat = self._compute_memory_safe_n(theta_zero_clamped, p_0, mu_L)
            if n_hat != self.pd_batch_size_N:
                self._record_n_update(self.pd_batch_size_N, n_hat, "cfr_memory_safe")
                self.pd_batch_size_N = n_hat
        n_eff = max(1, self.pd_batch_size_N)

        # Step 4: Midpoint k̂.
        k_hat_real, m_minus, m_plus, m_star = self._compute_midpoint_k(
            theta_zero_clamped, p_0, n_eff, mu_L
        )

        # Step 5: Diagnostic Δ(N) (used by adaptive selector / auto mode).
        delta_diag = self._compute_diagnostic_delta(
            theta_zero_clamped, k_hat_real, n_eff, mu_L, mu_O
        )

        # Step 6: EMA smoothing on the realised θ̂ = k̂ / N̂ to damp noise.
        theta_hat = k_hat_real / float(n_eff)
        theta_hat = max(_PD_THETA_HARD_MIN, min(self.pd_theta_max, theta_hat))
        if not self.pd_ifr_theta_initialized:
            self.pd_k_ratio = theta_hat
            self.pd_ifr_theta_initialized = True
        else:
            alpha = self.pd_ifr_theta_ema_alpha
            self.pd_k_ratio = alpha * theta_hat + (1 - alpha) * self.pd_k_ratio
        # Switch threshold uses the smoothed ratio scaled to current N.
        self.pd_switch_threshold_k = max(1, int(self.pd_k_ratio * n_eff))

        # Surface diagnostics for stats / analysis.
        self.pd_theta_zero_last = theta_zero_clamped
        self.pd_k_hat_midpoint_last = k_hat_real
        self.pd_n_hat_safe_last = float(n_eff)
        self.pd_delta_diagnostic_last = delta_diag

        self.pd_update_history.append(
            {
                "timestamp": time.monotonic() - self._pd_start_time,
                "p_0_estimate": p_0,
                "mu_L_estimate": mu_L,
                "mu_O_estimate": mu_O,
                "theta_0": theta_zero_clamped,
                "k_hat_real": k_hat_real,
                "k_hat_int": int(self.pd_switch_threshold_k),
                "N_hat": int(n_eff),
                "M_minus": int(m_minus),
                "M_plus": int(m_plus),
                "M_star": int(m_star),
                "delta_diagnostic": delta_diag,
                "samples_used": len(self.pd_ifr_samples),
                "oom_event_count": int(self.pd_oom_event_count),
            }
        )

        if (
            abs(self.pd_k_ratio - old_ratio) > 0.01
            or old_k != self.pd_switch_threshold_k
            or old_n != self.pd_batch_size_N
        ):
            logger.info(
                f"[P/D CFR] midpoint update: θ_0={theta_zero_clamped:.4f}, "
                f"θ̂={old_ratio:.4f}->{self.pd_k_ratio:.4f}, "
                f"k̂={old_k}->{self.pd_switch_threshold_k}, "
                f"N̂={old_n}->{self.pd_batch_size_N} "
                f"(p_0={p_0:.6f}, μ_L={mu_L:.0f}, μ_O={mu_O:.0f}, "
                f"M*={m_star}, Δ={delta_diag:.6f})"
            )

    def _record_n_update(self, old_n: int, new_n: int, reason: str) -> None:
        """Record an N update event for trajectory tracking."""
        if old_n == new_n:
            return
        timestamp = time.monotonic() - self._pd_start_time
        self.pd_n_update_history.append(
            {
                "timestamp": timestamp,
                "old_N": old_n,
                "new_N": new_n,
                "reason": reason,
                "k_star": self.pd_switch_threshold_k,
                "avg_output_tokens": self.pd_avg_output_tokens,
            }
        )

    def _compute_adaptive_kv_threshold(self) -> float:
        """
        Compute adaptive KV cache threshold based on average output tokens.

        The idea: Reserve enough KV cache space for decoding phase.
        - If avg_output_tokens is high, reserve more space (higher threshold)
        - If avg_output_tokens is low, can use more cache for prefill

        Formula:
        - expected_decode_blocks = N * avg_output_tokens * margin / block_size
        - threshold = expected_decode_blocks / total_blocks + base_reserve

        Returns:
            float: KV cache threshold (fraction of total blocks to reserve)
        """
        if not hasattr(self.kv_cache_manager, "block_pool"):
            return 0.05  # Default fallback

        total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
        if total_blocks <= 0:
            return 0.05

        # Expected blocks needed for decode phase
        # Each request needs avg_output_tokens * margin for safety
        tokens_per_block = self.block_size
        expected_output_tokens = self.pd_avg_output_tokens * self.pd_output_margin

        # For N decoding requests, total blocks needed
        blocks_for_decode = (
            self.pd_batch_size_N * expected_output_tokens / tokens_per_block
        )

        # Compute threshold: reserve this fraction of total blocks
        reserve_ratio = blocks_for_decode / total_blocks

        # Add base reserve and clamp to reasonable bounds.
        # Upper bound is configurable via VLLM_PD_KV_THRESHOLD_MAX (default 0.6).
        # On r->1 workloads (very long outputs) the formula can saturate this
        # cap and cause kv_escape to fire repeatedly; in that case setting the
        # env var to ~0.3 makes kv_escape less aggressive. We keep 0.6 as the
        # default since lowering it didn't help once the IFR floor (below) is in.
        kv_threshold_max = float(os.environ.get("VLLM_PD_KV_THRESHOLD_MAX", "0.6"))
        threshold = reserve_ratio + self.pd_base_kv_reserve
        threshold = max(0.05, min(kv_threshold_max, threshold))

        return threshold

    def _compute_adaptive_N(self) -> int:
        """
        Compute adaptive batch size N based on KV cache capacity and avg output.

        The idea: N should be chosen such that:
        - All N requests can be prefilled
        - There's enough KV cache left for decode phase

        Constraint:
        - N * (avg_prompt + avg_output * margin) / block_size
            <= total_blocks * (1 - reserve)

        Solving for N:
        - N <= total_blocks * (1 - reserve) * block_size
                / (avg_prompt + avg_output * margin)

        Returns:
            int: Adaptive batch size N
        """
        if not hasattr(self.kv_cache_manager, "block_pool"):
            return self.max_num_running_reqs

        total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
        if total_blocks <= 0:
            return self.max_num_running_reqs

        # Estimate average prompt tokens from running/waiting requests
        avg_prompt_tokens = 512  # Default estimate
        sample_requests = list(self.running)[:50] + list(self.waiting)[:50]
        if sample_requests:
            total_prompt = sum(r.num_prompt_tokens for r in sample_requests)
            avg_prompt_tokens = total_prompt / len(sample_requests)

        # Expected tokens per request (prompt + output with margin)
        expected_output = self.pd_avg_output_tokens * self.pd_output_margin
        tokens_per_request = avg_prompt_tokens + expected_output

        # Available blocks (with base reserve for safety)
        available_blocks = total_blocks * (1.0 - self.pd_base_kv_reserve)

        # Compute adaptive N
        blocks_per_request = tokens_per_request / self.block_size
        adaptive_n = int(available_blocks / blocks_per_request)

        # Clamp to reasonable bounds
        min_n = max(16, self.max_num_running_reqs // 10)
        max_n = self.max_num_running_reqs
        adaptive_n = max(min_n, min(max_n, adaptive_n))

        return adaptive_n

    # @cprofile("update_params_online.prof")
    def _update_params_online(self, output_tokens: int) -> None:
        """
        Unified parameter update with configurable interval.

        Updates all parameters together: avg_output_tokens, p, k*, θ*
        Hot path (every request): Only two integer additions.
        Cold path (every pd_param_update_interval requests): Update all params.

        k* update behavior by mode:
        - "direct": k* computed via Proposition 1
                    (unless VLLM_PD_K_STAR is user-specified)
        - "ratio":  k* = theta* x N, theta* computed from p
                    (unless VLLM_PD_K_RATIO is user-specified)
        - "ifr":    k* = theta* x N, theta* from the IFR correction formula
                    (using the hazard-rate estimate)
        """
        # HOT PATH: Only integer operations (zero overhead)
        self.pd_batch_completed_count += 1
        self.pd_batch_total_output_tokens += output_tokens

        # IFR mode: online adaptive update (alg:adaptive_joint)
        if self.pd_k_mode == "ifr":
            # Append to sliding window (deque with maxlen auto-evicts)
            self.pd_ifr_samples.append(output_tokens)
            self.pd_ifr_update_counter += 1

            # Check if we should update threshold (independent of other params)
            if (
                self.pd_ifr_update_counter >= self.pd_ifr_update_interval
                and len(self.pd_ifr_samples) >= self.pd_ifr_min_samples
            ):
                self._update_ifr_threshold()
                self.pd_ifr_update_counter = 0

        # CFR mode: online midpoint update (closed-form construction
        # not in the camera-ready paper; see journal version)
        elif self.pd_k_mode == "cfr":
            self.pd_ifr_samples.append(output_tokens)
            self.pd_ifr_update_counter += 1
            if (
                self.pd_ifr_update_counter >= self.pd_ifr_update_interval
                and len(self.pd_ifr_samples) >= self.pd_ifr_min_samples
            ):
                self._update_cfr_threshold()
                self.pd_ifr_update_counter = 0

        # Check if we've reached the update interval
        if self.pd_batch_completed_count < self.pd_param_update_interval:
            return  # Fast exit - no expensive operations

        # COLD PATH: Reached threshold, do the expensive operations
        _cold_path_start = time.perf_counter()

        self.pd_total_completed += self.pd_batch_completed_count

        if self.pd_batch_total_output_tokens > 0:
            batch_mean_len = (
                self.pd_batch_total_output_tokens / self.pd_batch_completed_count
            )
            batch_p = 1.0 / batch_mean_len

            if not self.pd_param_initialized:
                # First batch: direct assignment
                self.pd_p = batch_p
                self.pd_avg_output_tokens = batch_mean_len
                self.pd_param_initialized = True
            else:
                # EMA update
                self.pd_p = (
                    self.pd_ema_alpha * batch_p + (1 - self.pd_ema_alpha) * self.pd_p
                )
                self.pd_avg_output_tokens = (
                    self.pd_ema_alpha * batch_mean_len
                    + (1 - self.pd_ema_alpha) * self.pd_avg_output_tokens
                )

            # Update k* based on mode (skip if user specified fixed value)
            if self.pd_k_mode == "direct" and not self.pd_k_star_user_specified:
                old_k = self.pd_switch_threshold_k
                # Recompute k* (depends on p and N)
                self.pd_switch_threshold_k = self._compute_optimal_k()

                if old_k != self.pd_switch_threshold_k:
                    logger.info(
                        f"[P/D] k* update: {old_k}->{self.pd_switch_threshold_k} "
                        f"(p={self.pd_p:.4f}, mean_len={batch_mean_len:.1f})"
                    )

            elif self.pd_k_mode == "ratio" and not self.pd_k_ratio_user_specified:
                old_ratio = self.pd_k_ratio
                old_k = self.pd_switch_threshold_k
                self.pd_k_ratio = self._compute_optimal_ratio()

                if self.pd_k_ratio != old_ratio:
                    self.pd_switch_threshold_k = self._compute_k_from_ratio()
                    logger.info(
                        f"[P/D] ratio update: "
                        f"θ*={old_ratio:.4f}->{self.pd_k_ratio:.4f}, "
                        f"k*={old_k}->{self.pd_switch_threshold_k} "
                        f"(p={self.pd_p:.4f}, mean_len={batch_mean_len:.1f})"
                    )

            # Note: IFR mode uses independent online update mechanism
            # (see _update_ifr_threshold called from hot path)

        # EB+ mode selection (only in auto mode)
        if self.scheduler_mode == "auto":
            self._evaluate_mode_switch()

        # Reset batch accumulators
        self.pd_batch_completed_count = 0
        self.pd_batch_total_output_tokens = 0

        # Record cold path timing
        _cold_path_elapsed = (time.perf_counter() - _cold_path_start) * 1e6
        self._param_update_count += 1
        self._param_update_total_us += _cold_path_elapsed
        self._last_param_update_us = _cold_path_elapsed

    # ================================================================
    # EB+ Adaptive Mode Selection (alg:adaptive_joint extension)
    # ================================================================

    def _evaluate_mode_switch(self) -> None:
        """Evaluate the EB-MB crossover condition (prop:comparison) and switch mode.

        Called every pd_param_update_interval completions from the cold path.
        Decision:
          LHS = β_MB^e(r̂) - β_EB_w
          RHS = (1/(μ_L+μ_o)) * [
                  (α_p - α_d·ln(1-θ₀)·μ_o)/(θ₀·N_obs)
                  - α_MB·(1+μ_o)/N_obs
                ]
          Switch to EB if LHS > RHS + δ  (contention dominates)
          Switch to CP if LHS < RHS - δ  (amortization dominates)
        """
        import math

        # Wait for enough samples before making decisions
        if not self.pd_param_initialized:
            return

        # --- Compute workload statistics ---
        mu_o = self.pd_avg_output_tokens
        mu_L = self._avg_prompt_len
        if mu_L + mu_o <= 0:
            return

        # Steady-state decode ratio: r = μ_o / (μ_L + μ_o)
        r = mu_o / (mu_L + mu_o)
        # Batch occupancy
        N_obs = max(1.0, self._n_obs)

        # --- LHS: β_MB^e(r̂) - β_EB_w ---
        # β_EB_w: workload-weighted exclusive marginal cost
        beta_EB_w = (self.pd_beta_p * mu_L + self.pd_beta_d * mu_o) / (mu_L + mu_o)

        # β_MB^e(r): CP effective marginal cost from offline profile
        if self._mb_cost_profiled:
            beta_MB_e = self._mb_cost_a + self._mb_cost_b * r + self._mb_cost_c * r * r
        else:
            # Fallback: no profiled CP cost → LHS = 0
            # Decision driven entirely by overhead comparison (RHS)
            beta_MB_e = beta_EB_w

        LHS = beta_MB_e - beta_EB_w

        # --- RHS: amortized fixed-cost comparison ---
        # θ₀: current ratio (from IFR or ratio estimator)
        theta0 = self.pd_k_ratio if hasattr(self, "pd_k_ratio") else 0.5
        if theta0 <= 0 or theta0 >= 1:
            return  # invalid, skip

        log_1_minus_theta = math.log(1 - theta0)  # negative
        eb_overhead = (self.pd_alpha_p - self.pd_alpha_d * log_1_minus_theta * mu_o) / (
            theta0 * N_obs
        )
        mb_overhead = self._alpha_mb * (1 + mu_o) / N_obs

        RHS = (1.0 / (mu_L + mu_o)) * (eb_overhead - mb_overhead)

        # --- Decision with hysteresis ---
        delta = self._mode_switch_delta

        if self._mode_cooldown > 0:
            self._mode_cooldown -= 1
            return

        old_mode = self._active_scheduler
        if RHS + delta < LHS:
            # Contention dominates → EB is better
            if self._active_scheduler != "eb":
                self._transition_to_eb()
                self._mode_cooldown = self._mode_cooldown_max
        elif RHS - delta > LHS and self._active_scheduler != "mb":
            # Amortization dominates → MB is better
            self._transition_to_mb()
            self._mode_cooldown = self._mode_cooldown_max

        # Log mode switch
        if self._active_scheduler != old_mode:
            self._mode_switch_count += 1
            switch_record = {
                "timestamp": time.monotonic() - self._pd_start_time,
                "old_mode": old_mode,
                "new_mode": self._active_scheduler,
                "LHS": LHS,
                "RHS": RHS,
                "delta": delta,
                "r": r,
                "mu_L": mu_L,
                "mu_o": mu_o,
                "N_obs": N_obs,
                "theta0": theta0,
                "beta_MB_e": beta_MB_e,
                "beta_EB_w": beta_EB_w,
                "total_completed": self.pd_total_completed,
            }
            self._mode_switch_history.append(switch_record)
            logger.info(
                f"[EB+] Mode switch: {old_mode} -> "
                f"{self._active_scheduler} | "
                f"LHS={LHS:.6f}, RHS={RHS:.6f}, r={r:.3f}, "
                f"N_obs={N_obs:.1f}, θ₀={theta0:.4f}, "
                f"β_MB^e={beta_MB_e:.6f}, β_EB_w={beta_EB_w:.6f}"
            )

    def _transition_to_eb(self) -> None:
        """Transition from MB mode to EB mode.

        Build pd_decoding_requests from current running set and set
        appropriate PD phase state.
        """
        self._active_scheduler = "eb"

        # Populate pd_decoding_requests from running requests in decode phase
        self.pd_decoding_requests.clear()
        for req in self.running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                self.pd_decoding_requests.add(req.request_id)

        num_decoding = len(self.pd_decoding_requests)
        has_waiting = len(self.waiting) > 0

        if num_decoding > 0:
            # We have decoding requests → enter Phase 1 (Decode)
            self.pd_phase = 1
            self.pd_completed_decode_count = 0
            self.pd_prefilled_count = num_decoding
            self.pd_refill_target = 0
            # Update N to reflect actual demand (running + waiting),
            # capped at max_num_running_reqs.  Without this, a v1→EB
            # transition under light running but heavy waiting (e.g. a
            # concurrency spike) would leave N tiny, starving prefill.
            self.pd_batch_size_N = min(
                num_decoding + len(self.waiting), self.max_num_running_reqs
            )
            self._update_k_star()
        elif has_waiting:
            # No decoding but have waiting → start fresh at Phase 0
            self._reset_pd_to_initial()
        else:
            # Nothing to do
            self.pd_phase = 0
            self.pd_prefilled_count = 0
            self.pd_completed_decode_count = 0
            self.pd_refill_target = 0

        logger.info(
            f"[EB+] MB -> EB: phase={self.pd_phase}, "
            f"decoding={num_decoding}, running={len(self.running)}, "
            f"N={self.pd_batch_size_N}, k*={self.pd_switch_threshold_k}"
        )

    def _transition_to_mb(self) -> None:
        """Transition from EB mode to MB mode.

        The running list is shared so v1 can immediately schedule all
        requests. Clear PD-specific tracking state.
        """
        self._active_scheduler = "mb"

        # Clear PD tracking state — MB does not use it.
        # Will be rebuilt if we switch back to EB.
        self.pd_decoding_requests.clear()
        self.pd_phase = 0
        self.pd_prefilled_count = 0
        self.pd_completed_decode_count = 0
        self.pd_refill_target = 0

        logger.info(
            f"[EB+] EB -> MB: running={len(self.running)}, waiting={len(self.waiting)}"
        )

    def _preempt_chunk_prefilling(self) -> tuple[int, int]:
        """Preempt all chunk_prefilling requests to free KV cache.

        Returns:
            Tuple of (num_preempted_chunks, num_preempted_tokens)
        """
        preempted_chunks = 0
        preempted_tokens = 0
        for req in list(self.chunk_prefilling):
            preempted_tokens += req.num_computed_tokens
            self.kv_cache_manager.free(req)
            if hasattr(self, "encoder_cache_manager"):
                self.encoder_cache_manager.free(req)
            req.status = RequestStatus.PREEMPTED
            req.num_computed_tokens = 0
            req.num_preemptions += 1
            self.running.remove(req)
            self.waiting.prepend_request(req)
            preempted_chunks += 1
        self.chunk_prefilling.clear()
        return preempted_chunks, preempted_tokens

    def _update_k_star(self) -> None:
        """Update k* (switch threshold) based on current mode.

        - direct mode: if user specified k*, don't update; otherwise recompute
        - ratio mode: if user specified ratio, don't update; otherwise recompute
        - ifr mode: uses independent online update (see _update_ifr_threshold)
        """
        if self.pd_k_mode == "direct":
            if not self.pd_k_star_user_specified:
                self.pd_switch_threshold_k = self._compute_optimal_k()
        elif self.pd_k_mode == "ratio":
            if not self.pd_k_ratio_user_specified:
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
        elif self.pd_k_mode == "ifr":
            # IFR mode uses independent online update mechanism
            # Only recalculate k* from current ratio (N may have changed)
            self.pd_switch_threshold_k = self._compute_k_from_ratio()

    def _apply_long_prefill_threshold(self, num_tokens: int) -> int:
        """Apply long prefill token threshold if configured."""
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_tokens:
            return threshold
        return num_tokens

    @staticmethod
    def _is_prefill(req: Request) -> bool:
        """Check if request is in prefill phase."""
        return req.num_computed_tokens < req.num_prompt_tokens

    # @cprofile("handle_phase_transition.prof")
    def _handle_phase_transition(self) -> None:
        """
        Handle P/D phase transitions based on current state.

        Updates self.pd_phase based on:
        - Phase 0 -> 1: When N prefilled or no more waiting
        - Phase 1 -> 2: When k decoded AND k waiting available
        - Phase 2 -> 1: When k prefilled or no more waiting
        - Reset to 0: When idle (no decode work, no running, has waiting)
        """
        # Cleanup orphans only if there's a size mismatch (fast check first)
        num_running = len(self.running)
        if len(self.pd_decoding_requests) > num_running:
            running_ids = {req.request_id for req in self.running}
            orphaned_ids = self.pd_decoding_requests - running_ids
            if orphaned_ids:
                logger.warning(
                    "[P/D] Cleaning %d orphaned decoding IDs", len(orphaned_ids)
                )
                self.pd_decoding_requests -= orphaned_ids

        num_pending_chunks = len(self.chunk_prefilling)
        if num_pending_chunks > 0:
            running_set = set(self.running)
            # Clean up: 1) requests not in running, 2) requests that completed prefill
            orphaned_chunks = [
                r
                for r in self.chunk_prefilling
                if r not in running_set or r.num_computed_tokens >= r.num_prompt_tokens
            ]
            if orphaned_chunks:
                logger.warning(
                    "[P/D] Cleaning %d orphaned/completed chunk_prefilling",
                    len(orphaned_chunks),
                )
                for req in orphaned_chunks:
                    self.chunk_prefilling.remove(req)
                    # If prefill completed, add to decoding set and count
                    if (
                        req in running_set
                        and req.num_computed_tokens >= req.num_prompt_tokens
                        and req.request_id not in self.pd_decoding_requests
                    ):
                        self.pd_decoding_requests.add(req.request_id)
                        self.pd_prefilled_count += 1
                num_pending_chunks = len(self.chunk_prefilling)

        num_decoding = len(self.pd_decoding_requests)
        has_decoding = num_decoding > 0
        waiting_count = len(self.waiting) + num_pending_chunks
        has_waiting = waiting_count > 0
        has_pending_chunks = num_pending_chunks > 0
        prev_phase = self.pd_phase

        # Check if KV cache usage exceeds adaptive threshold
        # The threshold is learned based on average output tokens:
        # - Higher avg output tokens -> higher threshold (reserve more for decode)
        # - Lower avg output tokens -> lower threshold (can use more for prefill)
        # When threshold exceeded, allow phase transitions even with pending chunks
        # to prevent deadlock - chunks will continue after decode frees memory
        kv_cache_full = False
        adaptive_threshold = self._compute_adaptive_kv_threshold()
        if hasattr(self.kv_cache_manager, "block_pool"):
            total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
            free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
            kv_cache_full = free_blocks < total_blocks * adaptive_threshold

        if self.pd_phase == 0:
            # Initial prefill -> decode when N prefilled OR no more waiting
            # When KV cache is full, allow transition even with pending chunks
            can_transition = not has_pending_chunks or kv_cache_full
            if self.pd_prefilled_count >= self.pd_batch_size_N:
                if can_transition:
                    self.pd_phase = 1
                    self.pd_completed_decode_count = 0
                else:
                    logger.info(
                        f"[P/D] Phase 0: waiting for {num_pending_chunks} "
                        f"chunked prefills to complete before decode"
                    )
            # KV cache full escape - transition to decode to free memory
            # Use adaptive N based on avg output tokens to prevent preemptions
            # IMPORTANT: Proactively preempt chunk_prefilling requests to free KV
            # cache. These requests cannot continue in Phase 1, and if we let them
            # sit idle, they will be preempted anyway when decode needs more space.
            # Better to free them now (proactive) than later (reactive).
            elif kv_cache_full and has_decoding:
                adaptive_n = self._compute_adaptive_N()
                # min_n floor controls how aggressively kv_escape may shrink N.
                # Configurable via VLLM_PD_MIN_N_FLOOR_DIV (default 10, the
                # original behaviour). Raising to 2 keeps N >= max_seqs/2 and
                # prevents collapse, but blocks legitimate gradual shrinking
                # (hurt WildChat). The IFR floor (VLLM_PD_THETA_FLOOR) is a
                # better lever for r->1 workloads.
                _floor_div = int(os.environ.get("VLLM_PD_MIN_N_FLOOR_DIV", "10"))
                min_n = max(16, self.max_num_running_reqs // _floor_div)
                # Use the smaller of adaptive_n and current prefilled_count
                # to ensure we don't oversubscribe KV cache
                new_n = min(adaptive_n, self.pd_prefilled_count)
                if new_n >= min_n:
                    self._update_batch_size_n(new_n, "kv_escape")

                # Proactively preempt chunk_prefilling requests
                preempted_chunks, preempted_tokens = self._preempt_chunk_prefilling()

                logger.info(
                    f"[P/D] KV cache threshold ({adaptive_threshold:.2%}) escape: "
                    f"phase 0->1 with {self.pd_prefilled_count} prefilled, "
                    f"{waiting_count} waiting, adaptive_N={adaptive_n}, "
                    f"avg_output={self.pd_avg_output_tokens:.1f}, "
                    f"preempted_chunks={preempted_chunks} "
                    f"(freed {preempted_tokens} computed tokens)"
                )
                self.pd_phase = 1
                self.pd_completed_decode_count = 0
            elif not has_waiting and has_decoding and can_transition:
                # Adjust N to actual prefilled count, but keep a minimum
                # to avoid cold-start degradation (min 10% of max or 16)
                min_n = max(16, self.max_num_running_reqs // 10)
                if self.pd_prefilled_count >= min_n:
                    self._update_batch_size_n(self.pd_prefilled_count, "cold_start")
                # If prefilled count is too low, don't adjust N down
                # This prevents cold-start from permanently reducing batch size
                self.pd_phase = 1
                self.pd_completed_decode_count = 0

        elif self.pd_phase == 1:
            # RECOVERY: If N is too small relative to demand (e.g. after a
            # MB→EB transition under light running but heavy waiting, or a
            # cold-start), scale N up so that the 1→2 transition can refill
            # enough requests to keep the pipeline busy.
            target_n = self.max_num_running_reqs
            if (
                self.pd_batch_size_N < target_n
                and waiting_count >= target_n // 2
                and (
                    time.monotonic() - self.pd_last_n_update_time
                    >= self.pd_n_update_cooldown
                )
            ):
                old_n = self.pd_batch_size_N
                self.pd_batch_size_N = target_n
                self._update_k_star()
                self.pd_last_n_update_time = time.monotonic()
                self._record_n_update(old_n, self.pd_batch_size_N, "recovery")
                logger.info(
                    f"[P/D] N RECOVERY: {old_n} -> {self.pd_batch_size_N} "
                    f"(queue filled, k*={self.pd_switch_threshold_k}, "
                    f"avg_out={self.pd_avg_output_tokens:.1f})"
                )

            # Decode -> prefill when ratio condition met:
            #   min(q, N-n) / n >= k* / (N-k*)  i.e.  θ*/(1-θ*)
            # Preserves steady-state ratio θ* regardless of batch size,
            # naturally degrading to continuous-batching behavior under light load.
            # Uses integer arithmetic to avoid float division:
            #   fillable * (N - k*) >= n * k*
            #
            # KV-AWARE GUARD (not in camera-ready paper; see journal): also require KV
            # cache to have room for the refill. If kv_cache_full, the ratio
            # condition may trigger but refill cannot allocate blocks ->
            # scheduler bounces back to DECODE in Phase 2 (kv_cache_full
            # escape path) -> rapid phase thrashing. Stay in DECODE until
            # existing requests finish and free KV space; natural EOS /
            # completion paths then refill.
            if num_decoding > 0 and not kv_cache_full:
                N = self.pd_batch_size_N
                k_star = self.pd_switch_threshold_k
                fillable = min(waiting_count, max(0, N - num_decoding))
                denom = N - k_star
                if denom > 0 and fillable * denom >= num_decoding * k_star:
                    self.pd_refill_target = fillable
                    self.pd_phase = 2
                    self.pd_prefilled_count = 0
            # RESET: All decode requests completed, go back to Phase 0
            # Note: running may still have chunked prefill requests that will
            # continue in Phase 0. We only check has_decoding (pd_decoding_requests)
            # instead of len(running)==0 to allow this.
            elif not has_decoding and has_waiting:
                self._reset_pd_to_initial()

        elif self.pd_phase == 2:
            # Refill prefill -> decode when refill target met OR no more waiting
            # OR KV cache is full (to prevent deadlock)
            ready_to_decode = (
                self.pd_prefilled_count >= self.pd_refill_target
                or (not has_waiting and has_decoding)
                or kv_cache_full
            )  # KV cache full escape
            if ready_to_decode:
                # When KV cache is full, must transition even with pending chunks
                # to free memory. Proactively preempt them to free KV cache.
                # Otherwise, wait for chunked prefills to complete.
                if not has_pending_chunks or kv_cache_full:
                    if kv_cache_full and has_pending_chunks:
                        preempted_chunks, preempted_tokens = (
                            self._preempt_chunk_prefilling()
                        )
                        logger.info(
                            f"[P/D] Phase 2->1: preempted {preempted_chunks} "
                            f"chunks (freed {preempted_tokens} computed tokens, "
                            f"KV full)"
                        )
                    self.pd_phase = 1
                    self.pd_completed_decode_count = 0
                else:
                    logger.info(
                        f"[P/D] Phase 2: waiting for {num_pending_chunks} "
                        f"chunked prefills to complete before decode"
                    )

        # Log phase transition
        if prev_phase != self.pd_phase:
            logger.info(
                f"[P/D] {self.PD_PHASE_NAMES[prev_phase]} -> "
                f"{self.PD_PHASE_NAMES[self.pd_phase]} | "
                f"prefilled={self.pd_prefilled_count}, "
                f"decoded={self.pd_completed_decode_count}, "
                f"k*={self.pd_switch_threshold_k}, "
                f"refill_target={self.pd_refill_target}, "
                f"decoding={num_decoding}, N={self.pd_batch_size_N}, "
                f"avg_out={self.pd_avg_output_tokens:.1f}, "
                f"kv_thresh={adaptive_threshold:.2%}"
            )

    def _update_batch_size_n(self, new_n: int, reason: str = "update") -> None:
        """Update batch size N and recompute k* (ratio-based or optimal)."""
        if new_n != self.pd_batch_size_N:
            old_n, old_k = self.pd_batch_size_N, self.pd_switch_threshold_k
            self.pd_batch_size_N = new_n
            self._update_k_star()
            self.pd_last_n_update_time = time.monotonic()
            self._record_n_update(old_n, new_n, reason)
            logger.info(
                f"[P/D] N update: {old_n}->{new_n}, "
                f"k*={old_k}->{self.pd_switch_threshold_k}"
            )

    def _reset_pd_to_initial(self) -> None:
        """Reset P/D scheduler to initial state."""
        old_n = self.pd_batch_size_N
        self.pd_batch_size_N = self.max_num_running_reqs
        self._update_k_star()
        self.pd_last_n_update_time = time.monotonic()
        self._record_n_update(old_n, self.pd_batch_size_N, "reset")
        logger.info(
            f"[P/D] RESET: phase {self.pd_phase}->0 | "
            f"N={old_n}->{self.pd_batch_size_N}, k*={self.pd_switch_threshold_k}, "
            f"avg_out={self.pd_avg_output_tokens:.1f}"
        )
        self.pd_phase = 0
        self.pd_prefilled_count = 0
        self.pd_completed_decode_count = 0
        self.pd_refill_target = 0

    def schedule_pd(self) -> SchedulerOutput:
        """
        P/D Competition Scheduler with batch-based switching:

        Phase 0 (Initial Prefill): Prefill N requests
        Phase 1 (Decode): Decode all requests until k complete
        Phase 2 (Refill Prefill): Prefill k new requests (no decode)
        Then back to Phase 1: Decode (N-k old + k new) until k complete
        ...repeat...

        k is the switching threshold (can be optimized later).
        """
        # Mirror _schedule_default: notify the KV cache manager that a new
        # scheduling step has started so per-step state (e.g. FullAttention's
        # cached_blocks_this_step dedup set) is reset. Required for correct
        # prefix-cache block accounting in EB mode.
        self.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        effective_lookahead_tokens = 0
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_timestamp = time.monotonic()

        # Handle phase transitions
        self._handle_phase_transition()

        # ===== PREFILL SCHEDULING (Phase 0 or Phase 2) =====
        if self.pd_phase in (0, 2):
            target = (
                self.pd_batch_size_N if self.pd_phase == 0 else self.pd_refill_target
            )
            remaining = target - self.pd_prefilled_count

            # Continue chunked prefills first
            # Note: We don't check `remaining > 0` here because we must continue
            # all existing chunked prefills to prevent deadlock. The `remaining`
            # counter only limits NEW prefills from the waiting queue.
            if self.scheduler_config.enable_chunked_prefill:
                req_index = 0
                while req_index < len(self.chunk_prefilling) and token_budget > 0:
                    request = self.chunk_prefilling[req_index]

                    if not self._is_prefill(request):
                        req_index += 1
                        continue
                    if request.request_id in num_scheduled_tokens:
                        req_index += 1
                        continue

                    num_new_tokens = request.num_tokens - request.num_computed_tokens
                    num_new_tokens = self._apply_long_prefill_threshold(num_new_tokens)
                    num_new_tokens = min(num_new_tokens, token_budget)
                    if num_new_tokens <= 0:
                        req_index += 1
                        continue

                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=effective_lookahead_tokens,
                    )
                    if new_blocks is None:
                        req_index += 1
                        continue

                    # Check if prefill completes
                    # Use num_prompt_tokens (not num_tokens) to match is_prefill() logic
                    will_complete = (
                        request.num_prompt_tokens - request.num_computed_tokens
                        <= num_new_tokens
                    )
                    if will_complete:
                        self.chunk_prefilling.remove(request)
                        self.pd_prefilled_count += 1
                        remaining -= 1
                        self.pd_decoding_requests.add(request.request_id)
                    else:
                        req_index += 1

                    scheduled_running_reqs.append(request)
                    req_to_new_blocks[request.request_id] = new_blocks
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens

            # Schedule new prefills from waiting queue
            skipped = create_request_queue(self.policy)
            while self.waiting and token_budget > 0 and remaining > 0:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                num_external_computed_tokens = 0
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_local = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )
                    num_computed_tokens = num_local + num_external_computed_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_local = 0
                    num_computed_tokens = request.num_computed_tokens

                num_new_tokens = request.num_tokens - num_computed_tokens
                num_new_tokens = self._apply_long_prefill_threshold(num_new_tokens)

                is_chunked = False
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > token_budget
                ):
                    self.waiting.pop_request()
                    skipped.prepend_request(request)
                    continue
                elif num_new_tokens > token_budget:
                    is_chunked = True

                num_new_tokens = min(num_new_tokens, token_budget)
                if num_new_tokens <= 0:
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_local,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                )
                if new_blocks is None:
                    break

                request = self.waiting.pop_request()
                self.running.append(request)

                if is_chunked:
                    self.chunk_prefilling.append(request)
                else:
                    # Prefill completes in one step
                    self.pd_prefilled_count += 1
                    remaining -= 1
                    self.pd_decoding_requests.add(request.request_id)

                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )

                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # NOTE: upstream removed Request.num_cached_tokens; the old EB
                # sentinel bookkeeping here was write-only (never read by any EB
                # decision), so it is dropped. Prefix-cache hit counts are now
                # surfaced via prefill_stats / RequestOutput.num_cached_tokens.

            if skipped:
                self.waiting.prepend_requests(skipped)

        # ===== DECODE SCHEDULING (Phase 1 only) =====
        elif self.pd_phase == 1:
            req_index = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]

                if request.request_id in num_scheduled_tokens:
                    req_index += 1
                    continue
                if self._is_prefill(request):
                    req_index += 1
                    continue
                # Only decode requests in pd_decoding_requests
                if request.request_id not in self.pd_decoding_requests:
                    req_index += 1
                    continue

                num_new_tokens = (
                    request.num_tokens_with_spec
                    + request.num_output_placeholders
                    - request.num_computed_tokens
                )
                num_new_tokens = self._apply_long_prefill_threshold(num_new_tokens)
                num_new_tokens = min(num_new_tokens, token_budget)

                max_total = min(
                    request.num_prompt_tokens + request.max_tokens, self.max_model_len
                )
                num_new_tokens = min(
                    num_new_tokens, max_total - 1 - request.num_computed_tokens
                )
                if num_new_tokens == 0:
                    req_index += 1
                    continue

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

                if new_blocks is None:
                    # Need to preempt
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running, key=lambda r: (r.priority, r.arrival_time)
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    preempted_req.num_preemptions += 1
                    # P/D scheduling: clean up tracking state and count OOM
                    # events (each preemption is a KV-cache exhaustion =
                    # the ε-rate event tracked by Prop. memory).
                    if self.use_pd_scheduler:
                        self.pd_oom_event_count += 1
                    self.pd_decoding_requests.discard(preempted_req.request_id)
                    if preempted_req in self.chunk_prefilling:
                        self.chunk_prefilling.remove(preempted_req)

                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp
                        )
                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break
                    continue

                scheduled_running_reqs.append(request)
                req_to_new_blocks[request.request_id] = new_blocks
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

        # Construct scheduler output
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request.request_id
                )
            )

        # Mirror _schedule_default: the v2 model runner treats resumed
        # requests as new and requires prefill_token_ids (req._all_token_ids)
        # on each NewRequestData (asserted in gpu/model_runner.add_requests).
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def schedule(self) -> SchedulerOutput:
        """Entry point for scheduling. Dispatches to P/D, default, or auto."""
        self.current_step += 1
        if self._schedule_stats_enabled:
            t_start = time.perf_counter()

        # Track demand EMA for auto mode (running + waiting = true load)
        # Using only len(running) is wrong under EB: phase separation
        # drains running while filling waiting, but total demand is constant.
        if self.scheduler_mode == "auto":
            demand = float(len(self.running) + len(self.waiting))
            # Asymmetric EMA: fast ramp-up (0.3), slow ramp-down (0.03)
            # so we react quickly to traffic surges but don't prematurely
            # switch back when demand dips briefly.
            a = 0.3 if demand > self._n_obs else 0.03
            self._n_obs = a * demand + (1 - a) * self._n_obs

            # Demand surge detection: if instantaneous demand is 2x the
            # current EMA AND we haven't evaluated recently, trigger
            # immediate mode evaluation without waiting for cold path.
            if (
                demand > self._n_obs * 2
                and self._mode_cooldown == 0
                and self.pd_param_initialized
            ):
                self._evaluate_mode_switch()

        # Dispatch based on scheduler mode
        if self.scheduler_mode == "auto":
            if self._active_scheduler == "eb":
                output = self.schedule_pd()
            else:
                output = self._schedule_default()
        elif self.use_pd_scheduler:
            output = self.schedule_pd()
        else:
            output = self._schedule_default()

        if self._schedule_stats_enabled:
            t_end = time.perf_counter()
            self._record_schedule_stats(output, t_end - t_start)

        return output

    def _schedule_default(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Skip request with pending mm encoding prefetches
                    if (
                        self.ec_connector is not None
                        and request.mm_features
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens
                        )
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats is not None:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                limit_lookahead_tokens = load_kv_async and self.use_eagle
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # P/D scheduling: clean up tracking state
        if self.use_pd_scheduler:
            self.pd_decoding_requests.discard(request.request_id)
        if request in self.chunk_prefilling:
            self.chunk_prefilling.remove(request)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            finish_reason = None
            if stopped:
                # P/D scheduling: count completed decode requests and feed
                # online parameter estimators BEFORE the request is freed/reset.
                if self.use_pd_scheduler:
                    if request.request_id in self.pd_decoding_requests:
                        # EB path: request was tracked in pd_decoding_requests
                        self.pd_completed_decode_count += 1
                        self.pd_decoding_requests.discard(request.request_id)
                        output_tokens = request.num_tokens - request.num_prompt_tokens
                        if output_tokens > 0:
                            self._update_params_online(output_tokens)
                        # Track prompt length (used by CFR midpoint and
                        # auto-mode selector for mu_L estimation).
                        pl = float(request.num_prompt_tokens)
                        a = self.pd_avg_prompt_ema_alpha
                        self.pd_avg_prompt_len = (
                            a * pl + (1 - a) * self.pd_avg_prompt_len
                        )
                        if self.scheduler_mode == "auto":
                            self._avg_prompt_len = self.pd_avg_prompt_len
                    elif (
                        self.scheduler_mode == "auto" and self._active_scheduler == "mb"
                    ):
                        # Auto+MB path: still feed output samples for
                        # parameter tracking and mode selection
                        output_tokens = request.num_tokens - request.num_prompt_tokens
                        if output_tokens > 0:
                            self._update_params_online(output_tokens)
                        # Track prompt length EMA for mode selection
                        prompt_len = float(request.num_prompt_tokens)
                        a = self.pd_avg_prompt_ema_alpha
                        self.pd_avg_prompt_len = (
                            a * prompt_len + (1 - a) * self.pd_avg_prompt_len
                        )
                        self._avg_prompt_len = self.pd_avg_prompt_len

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
            # P/D scheduling: also remove from decoding set and chunk_prefilling
            for req in running_requests_to_remove:
                if self.use_pd_scheduler:
                    self.pd_decoding_requests.discard(req.request_id)
                if req in self.chunk_prefilling:
                    self.chunk_prefilling.remove(req)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # For async scheduling, any output frames already in flight at
                # preemption time are now stale and must be discarded when they
                # return. num_output_placeholders is exactly that count: 0 if
                # the engine has drained (e.g. pause_generation(keep) waited
                # for idle), 1 for vanilla async mid-step, or 1 + spec/PP frames
                # otherwise.
                request.async_tokens_to_discard = request.num_output_placeholders
                request.num_output_placeholders = 0

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        # Save schedule stats if collection was enabled
        if self._schedule_stats_enabled and self._schedule_stats:
            stats_file = os.environ.get(
                "VLLM_SCHEDULE_STATS_FILE", "schedule_stats.json"
            )
            self.save_schedule_stats(stats_file)

        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_computed_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Blocks in-flight prefills still need to finish (their reservation).

        Sums remaining full-ISL blocks over `self._inflight_prefills` (running
        prefills + in-progress async loads). The candidate async load isn't yet
        in the set, so it's naturally excluded.
        """
        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids

    # =========================================================================
    # Schedule Statistics Collection (for performance analysis)
    # =========================================================================

    def _record_schedule_stats(
        self, output: SchedulerOutput, elapsed_time: float
    ) -> None:
        """Record statistics for a single schedule() call."""
        # Initialize start time on first call (after warmup/loading completes)
        if self._schedule_stats_start_time is None:
            self._schedule_stats_start_time = time.monotonic()

        timestamp = time.monotonic() - self._schedule_stats_start_time

        # Fix timing attribution: update PREVIOUS batch's execution time
        # The interval between schedule calls represents the previous batch's
        # model execution time, not the current batch's.
        if self._schedule_stats:
            prev_timestamp = self._schedule_stats[-1]["timestamp"]
            execution_time_us = (timestamp - prev_timestamp) * 1e6
            self._schedule_stats[-1]["execution_time_us"] = execution_time_us

        # Count prefill vs decode tokens.
        # Note: num_computed_tokens has already been updated by
        # _update_after_schedule, so we need to subtract num_tokens to get
        # the state BEFORE this scheduling step.
        prefill_tokens = 0
        decode_tokens = 0
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req:
                # Get the computed tokens BEFORE this step
                computed_before = req.num_computed_tokens - num_tokens
                if computed_before < req.num_prompt_tokens:
                    # Was in prefill phase at the start of this step
                    prefill_tokens += num_tokens
                else:
                    # Was in decode phase
                    decode_tokens += num_tokens
            else:
                # Request not found (possibly finished), count as decode
                decode_tokens += num_tokens

        # Count preemption statistics
        num_preempted_reqs = 0
        preempted_tokens = 0
        if output.preempted_req_ids:
            num_preempted_reqs = len(output.preempted_req_ids)
            for req_id in output.preempted_req_ids:
                req = self.requests.get(req_id)
                if req:
                    # This is the number of tokens that need to be re-prefilled
                    preempted_tokens += req.num_prompt_tokens

        self._schedule_stats.append(
            {
                "timestamp": timestamp,
                "elapsed_us": elapsed_time * 1e6,
                "execution_time_us": 0,  # Will be updated by next schedule() call
                "scheduler_type": (
                    self._active_scheduler
                    if self.scheduler_mode == "auto"
                    else ("pd" if self.use_pd_scheduler else "default")
                ),
                "scheduler_mode": self.scheduler_mode,
                "phase": self.pd_phase if self.use_pd_scheduler else -1,
                "total_tokens": output.total_num_scheduled_tokens,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "num_new_reqs": len(output.scheduled_new_reqs),
                "num_running_reqs": len(self.running),
                "num_waiting_reqs": len(self.waiting),
                "num_scheduled_reqs": len(output.num_scheduled_tokens),
                "k_star": self.pd_switch_threshold_k if self.use_pd_scheduler else 0,
                "k_ratio": self.pd_k_ratio if self.use_pd_scheduler else 0,
                "refill_target": self.pd_refill_target if self.use_pd_scheduler else 0,
                "N": self.pd_batch_size_N if self.use_pd_scheduler else 0,
                "num_decoding_reqs": len(self.pd_decoding_requests)
                if self.use_pd_scheduler
                else 0,
                "num_preempted_reqs": num_preempted_reqs,
                "preempted_tokens": preempted_tokens,
                # Adaptive scheduling values
                "avg_output_tokens": self.pd_avg_output_tokens
                if self.use_pd_scheduler
                else 0,
                "adaptive_kv_threshold": self._compute_adaptive_kv_threshold()
                if self.use_pd_scheduler
                else 0,
                # Hazard rate estimation (IFR / CFR online estimator)
                "hazard_p0": self.pd_hazard_p0
                if (self.use_pd_scheduler and self.pd_k_mode in ("ifr", "cfr"))
                else 0,
                "hazard_eta": self.pd_hazard_eta
                if (self.use_pd_scheduler and self.pd_k_mode in ("ifr", "cfr"))
                else 0,
                "ifr_sample_count": len(self.pd_ifr_samples)
                if (self.use_pd_scheduler and self.pd_k_mode in ("ifr", "cfr"))
                else 0,
                # CFR midpoint diagnostics (last update; 0 outside cfr/auto)
                "mu_L_estimate": self.pd_avg_prompt_len if self.use_pd_scheduler else 0,
                "mu_O_estimate": self.pd_avg_output_tokens
                if self.use_pd_scheduler
                else 0,
                "theta_zero_last": self.pd_theta_zero_last
                if self.use_pd_scheduler
                else 0,
                "k_hat_midpoint_last": self.pd_k_hat_midpoint_last
                if self.use_pd_scheduler
                else 0,
                "n_hat_safe_last": self.pd_n_hat_safe_last
                if self.use_pd_scheduler
                else 0,
                "delta_diagnostic_last": self.pd_delta_diagnostic_last
                if self.use_pd_scheduler
                else 0,
                "oom_event_count": self.pd_oom_event_count
                if self.use_pd_scheduler
                else 0,
                # Parameter update overhead (cold path)
                "param_update_count": self._param_update_count,
                "last_param_update_us": self._last_param_update_us,
                # EB+ auto mode stats
                "active_scheduler": (
                    self._active_scheduler if self.scheduler_mode == "auto" else ""
                ),
                "n_obs": (self._n_obs if self.scheduler_mode == "auto" else 0),
                "mode_switch_count": (
                    self._mode_switch_count if self.scheduler_mode == "auto" else 0
                ),
            }
        )

    def save_schedule_stats(self, filepath: str = "schedule_stats.json") -> None:
        """Save collected schedule statistics to a JSON file."""
        import json

        with open(filepath, "w") as f:
            json.dump(
                {
                    "stats": self._schedule_stats,
                    "summary": self.get_schedule_stats_summary(),
                    "n_update_history": self.pd_n_update_history
                    if self.use_pd_scheduler
                    else [],
                    "mode_switch_history": (
                        self._mode_switch_history
                        if self.scheduler_mode == "auto"
                        else []
                    ),
                    "update_history": (
                        self.pd_update_history if self.use_pd_scheduler else []
                    ),
                    "pd_config": {
                        "k_mode": (self.pd_k_mode if self.use_pd_scheduler else ""),
                        "scheduler_mode": self.scheduler_mode,
                        "alpha_p": (self.pd_alpha_p if self.use_pd_scheduler else 0),
                        "beta_p": (self.pd_beta_p if self.use_pd_scheduler else 0),
                        "alpha_d": (self.pd_alpha_d if self.use_pd_scheduler else 0),
                        "beta_d": (self.pd_beta_d if self.use_pd_scheduler else 0),
                        "auto_compute_n": (
                            self.pd_auto_compute_n if self.use_pd_scheduler else False
                        ),
                        "oom_tolerance": (
                            self.pd_oom_tolerance if self.use_pd_scheduler else 0
                        ),
                        "max_num_seqs": int(self.max_num_running_reqs),
                        "final_N": (
                            int(self.pd_batch_size_N) if self.use_pd_scheduler else 0
                        ),
                        "final_k_star": (
                            int(self.pd_switch_threshold_k)
                            if self.use_pd_scheduler
                            else 0
                        ),
                        "total_oom_events": (
                            int(self.pd_oom_event_count) if self.use_pd_scheduler else 0
                        ),
                    },
                },
                f,
                indent=2,
            )
        logger.info(
            f"[Schedule Stats] Saved {len(self._schedule_stats)} records to {filepath}"
        )

    def _save_stats_on_exit(self) -> None:
        """Atexit handler to save stats when server shuts down."""
        if self._schedule_stats:
            self.save_schedule_stats(self._schedule_stats_file)

    def get_schedule_stats_summary(self) -> dict:
        """Get summary statistics from collected data."""
        if not self._schedule_stats:
            return {}

        total_tokens = [s["total_tokens"] for s in self._schedule_stats]
        prefill_tokens = [s["prefill_tokens"] for s in self._schedule_stats]
        decode_tokens = [s["decode_tokens"] for s in self._schedule_stats]
        elapsed_us = [s["elapsed_us"] for s in self._schedule_stats]
        # Filter out the last entry (execution_time not yet measured) and zeros
        execution_time_us = [
            s.get("execution_time_us", 0)
            for s in self._schedule_stats
            if s.get("execution_time_us", 0) > 0
        ]
        num_scheduled = [s["num_scheduled_reqs"] for s in self._schedule_stats]
        num_preempted = [s.get("num_preempted_reqs", 0) for s in self._schedule_stats]
        preempted_tokens = [s.get("preempted_tokens", 0) for s in self._schedule_stats]

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0

        def safe_percentile(lst, p):
            if not lst:
                return 0
            sorted_lst = sorted(lst)
            idx = int(len(sorted_lst) * p / 100)
            return sorted_lst[min(idx, len(sorted_lst) - 1)]

        return {
            "num_schedule_calls": len(self._schedule_stats),
            "total_tokens": {
                "sum": sum(total_tokens),
                "mean": safe_mean(total_tokens),
                "p50": safe_percentile(total_tokens, 50),
                "p99": safe_percentile(total_tokens, 99),
            },
            "prefill_tokens": {
                "sum": sum(prefill_tokens),
                "mean": safe_mean(prefill_tokens),
            },
            "decode_tokens": {
                "sum": sum(decode_tokens),
                "mean": safe_mean(decode_tokens),
            },
            "schedule_time_us": {
                "mean": safe_mean(elapsed_us),
                "p50": safe_percentile(elapsed_us, 50),
                "p99": safe_percentile(elapsed_us, 99),
            },
            "execution_time_us": {
                "mean": safe_mean(execution_time_us),
                "p50": safe_percentile(execution_time_us, 50),
                "p99": safe_percentile(execution_time_us, 99),
                "sum_ms": sum(execution_time_us) / 1000 if execution_time_us else 0,
            },
            "batch_size": {
                "mean": safe_mean(num_scheduled),
                "p50": safe_percentile(num_scheduled, 50),
                "p99": safe_percentile(num_scheduled, 99),
            },
            "empty_schedules": sum(1 for t in total_tokens if t == 0),
            "preemption": {
                "total_preempted_reqs": sum(num_preempted),
                "total_preempted_tokens": sum(preempted_tokens),
                "schedules_with_preemption": sum(1 for n in num_preempted if n > 0),
                "preemption_rate": sum(1 for n in num_preempted if n > 0)
                / len(num_preempted)
                if num_preempted
                else 0,
            },
            "n_updates": {
                "total_updates": len(self.pd_n_update_history)
                if self.use_pd_scheduler
                else 0,
                "by_reason": self._count_n_updates_by_reason()
                if self.use_pd_scheduler
                else {},
            },
            "param_update_overhead": {
                "total_updates": self._param_update_count,
                "total_time_us": self._param_update_total_us,
                "mean_time_us": self._param_update_total_us / self._param_update_count
                if self._param_update_count > 0
                else 0,
            },
        }

    def _count_n_updates_by_reason(self) -> dict:
        """Count N updates grouped by reason."""
        counts: dict[str, int] = {}
        for update in self.pd_n_update_history:
            reason = update.get("reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        return counts
