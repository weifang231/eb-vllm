# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import itertools
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

from vllm import envs
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
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.utils.profiling import cprofile
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.calibration import load_hardware_params
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
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
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
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
            self.parallel_config.data_parallel_rank,
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
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

        # Scheduler mode: "cp" (default), "eb" (exclusive batching), "auto" (EB+)
        # VLLM_PD_SCHEDULER_MODE takes precedence over VLLM_USE_PD_SCHEDULER
        _mode_env = os.environ.get("VLLM_PD_SCHEDULER_MODE", "")
        if _mode_env:
            self.scheduler_mode = _mode_env.lower()
        elif os.environ.get("VLLM_USE_PD_SCHEDULER", "0") == "1":
            self.scheduler_mode = "eb"
        else:
            self.scheduler_mode = "cp"

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
            if (self.scheduler_mode == "auto"
                    and self.pd_k_mode == "direct"
                    and not self.pd_k_ratio_user_specified
                    and not self.pd_k_star_user_specified):
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
                logger.info(f"Loaded hardware params from calibration file: "
                            f"model={_hw_params.model}, device={_hw_params.device_name}")
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
                        "  python -m vllm.v1.core.sched.calibration --model <model_name>\n"
                        "Or set environment variable:\n"
                        "  export VLLM_PD_CALIBRATION_FILE=/path/to/pd_calibration.json\n"
                        "Or set all timing parameters manually:\n"
                        "  export VLLM_PD_ALPHA_P=<value>\n"
                        "  export VLLM_PD_BETA_P=<value>\n"
                        "  export VLLM_PD_ALPHA_D=<value>\n"
                        "  export VLLM_PD_BETA_D=<value>"
                    )
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
            # together with the midpoint construction (alg:midpoint;
            # journal-only).
            if self.pd_k_mode in ("ifr", "cfr"):
                # Sliding window of output length samples for hazard
                # rate estimation.  Keeps only the most recent W samples
                # so the estimator adapts to distribution shifts within
                # O(W) completions.
                self.pd_ifr_window_size = int(os.environ.get(
                    "VLLM_PD_IFR_WINDOW_SIZE", "500"))
                self.pd_ifr_samples: deque[int] = deque(
                    maxlen=self.pd_ifr_window_size)
                # M: Update interval (re-estimate every M completions)
                self.pd_ifr_update_interval = int(os.environ.get(
                    "VLLM_PD_IFR_UPDATE_INTERVAL", "100"))
                # W_min: Minimum samples before estimation starts
                self.pd_ifr_min_samples = int(os.environ.get(
                    "VLLM_PD_IFR_MIN_SAMPLES", "30"))
                # θ_default: Default theta during cold-start phase
                self.pd_ifr_default_theta = float(os.environ.get(
                    "VLLM_PD_IFR_DEFAULT_THETA", "0.70"))
                # c: Independent update counter
                self.pd_ifr_update_counter = 0
                # Estimated hazard rate parameters: h(t) = p_0 + η * t
                self.pd_hazard_p0 = 0.01  # Base hazard rate (will be estimated)
                self.pd_hazard_eta = 0.0  # Hazard rate slope (η >= 0 for IFR)
                # Maximum theta to prevent excessive waiting
                self.pd_theta_max = float(os.environ.get(
                    "VLLM_PD_THETA_MAX", "0.80"))
                # Lower bound on the adaptive theta* (paper's theta_min;
                # see model.tex clipping rule). Default 0.01: paper-time
                # near-no-floor. Workload-specific clipping (0.3/0.7/0.85)
                # is no longer needed because the runtime KV-aware phase
                # 1->2 gate (Appendix app:kv-aware-gate) handles the
                # underlying phase-thrashing at the root.
                self.pd_theta_floor = float(os.environ.get(
                    "VLLM_PD_THETA_FLOOR", "0.01"))
                # EMA smoothing for θ* to damp oscillations from noisy
                # hazard-rate estimates.  α=0.3 means ~70% weight on
                # previous θ*, providing stability while still tracking
                # distribution shifts.
                self.pd_ifr_theta_ema_alpha = float(os.environ.get(
                    "VLLM_PD_IFR_THETA_EMA_ALPHA", "0.3"))
                self.pd_ifr_theta_initialized = False

            # Initialize k* based on mode
            if self.pd_k_mode == "direct":
                # If k* is user-specified, use that fixed value; otherwise compute the optimum.
                if self.pd_k_star_user_specified:
                    self.pd_switch_threshold_k = max(1, self.pd_k_star_fixed)
                else:
                    self.pd_switch_threshold_k = self._compute_optimal_k()
            elif self.pd_k_mode == "ratio":
                # If ratio is not user-specified, compute the initial theta* from the asymptotic formula.
                if not self.pd_k_ratio_user_specified:
                    self.pd_k_ratio = self._compute_optimal_ratio()
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
            elif self.pd_k_mode == "ifr":
                # IFR mode: use default theta during cold-start phase
                # Will adapt based on hazard rate estimation as samples accumulate
                self.pd_k_ratio = self.pd_ifr_default_theta
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
            elif self.pd_k_mode == "cfr":
                # CFR mode (alg:midpoint; journal-only).
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
                logger.warning(f"Unknown k mode '{self.pd_k_mode}', using direct mode")
                self.pd_k_mode = "direct"
                self.pd_switch_threshold_k = self._compute_optimal_k()

            # Log PD scheduler configuration
            if self.pd_k_mode == "direct":
                dyn_tag = "fixed" if self.pd_k_star_user_specified else "auto"
                k_info = f"k*={self.pd_switch_threshold_k} ({dyn_tag})"
            elif self.pd_k_mode == "ratio":
                dyn_tag = "fixed" if self.pd_k_ratio_user_specified else "auto"
                k_info = f"θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} ({dyn_tag})"
            elif self.pd_k_mode == "cfr":
                k_info = (f"θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} "
                          f"(CFR midpoint, θ_max={self.pd_theta_max})")
            else:  # ifr
                k_info = (f"θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} "
                          f"(IFR adaptive, θ_max={self.pd_theta_max})")
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
            self.pd_param_update_interval = int(os.environ.get(
                "VLLM_PD_PARAM_UPDATE_INTERVAL", "100"))
            self.pd_ema_alpha = 0.2  # EMA smoothing factor
            self.pd_total_completed = 0  # Total completed requests (all time)
            self.pd_param_initialized = False  # First batch: direct assign, not EMA
            # Batch accumulators (reset after each parameter update)
            self.pd_batch_completed_count = 0  # Completed in current batch
            self.pd_batch_total_output_tokens = 0  # Sum of output tokens in batch

            # Track average output tokens with EMA for adaptive thresholds
            self.pd_avg_output_tokens = 100.0  # Initial estimate (cold start)
            # Base reserve ratio (minimum fraction of KV cache to reserve for decode)
            self.pd_base_kv_reserve = float(os.environ.get(
                "VLLM_PD_BASE_KV_RESERVE", "0"))
            # Safety margin multiplier for output token estimation
            self.pd_output_margin = float(os.environ.get(
                "VLLM_PD_OUTPUT_MARGIN", "0.5"))

            # N RECOVERY cooldown to prevent frequent updates
            self.pd_last_n_update_time = 0.0  # timestamp of last N update
            self.pd_n_update_cooldown = float(os.environ.get(
                "VLLM_PD_N_UPDATE_COOLDOWN", "2.0"))  # seconds

            # μ_L (mean prompt length) EMA — used by both CFR midpoint
            # construction and the adaptive selector diagnostic Δ(N).
            self.pd_avg_prompt_len = 512.0
            self.pd_avg_prompt_ema_alpha = 0.05  # slow EMA

            # CFR / midpoint configuration
            # Dynamic memory-safe N̂ (Eq. eq:Nstar / Proposition prop:memory):
            # if enabled, the scheduler periodically re-derives the maximum
            # batch size from the KV-cache budget and the OOM tolerance ε.
            self.pd_auto_compute_n = (os.environ.get(
                "VLLM_PD_AUTO_COMPUTE_N", "0") == "1")
            self.pd_oom_tolerance = float(os.environ.get(
                "VLLM_PD_OOM_TOLERANCE", "0.01"))
            # Cumulative count of requests that ran out of KV memory and had
            # to be preempted — surfaced in stats for OOM-rate validation.
            self.pd_oom_event_count = 0
            # Per-update snapshot history for CFR validation experiments.
            self.pd_cfr_update_history: list[dict] = []
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
            # Current active scheduler: "cp" or "eb"
            self._active_scheduler = os.environ.get(
                "VLLM_PD_AUTO_COLD_START_MODE", "cp")

            # CP effective marginal cost: f(r) = a + b*r + c*r² (offline profiled)
            self._cp_cost_a = float(os.environ.get("VLLM_PD_CP_COST_A", "0"))
            self._cp_cost_b = float(os.environ.get("VLLM_PD_CP_COST_B", "0"))
            self._cp_cost_c = float(os.environ.get("VLLM_PD_CP_COST_C", "0"))
            self._cp_cost_profiled = any(
                os.environ.get(k)
                for k in ["VLLM_PD_CP_COST_A", "VLLM_PD_CP_COST_B",
                           "VLLM_PD_CP_COST_C"])

            # α_CP: defaults to α_p (paper approximation α_p ≈ α_d ≈ α_CP)
            _acp = os.environ.get("VLLM_PD_ALPHA_CP", "")
            self._alpha_cp = float(_acp) if _acp else self.pd_alpha_p

            # Hysteresis band and cooldown for mode switching
            self._mode_switch_delta = float(os.environ.get(
                "VLLM_PD_MODE_SWITCH_DELTA", "0.0001"))
            self._mode_cooldown_max = int(os.environ.get(
                "VLLM_PD_MODE_COOLDOWN", "3"))
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
                f"cp_cost_profiled={self._cp_cost_profiled}, "
                f"alpha_cp={self._alpha_cp:.6f}, "
                f"delta={self._mode_switch_delta}, "
                f"cooldown={self._mode_cooldown_max}"
            )

        self.chunk_prefilling: list[Request] = []

        # N update history: (timestamp, old_N, new_N, reason)
        self.pd_n_update_history: list[dict] = []
        self._pd_start_time = time.monotonic()

        # Performance metrics for parameter updates
        self._param_update_count = 0          # Number of cold path updates
        self._param_update_total_us = 0.0     # Total time spent in cold path (μs)
        self._last_param_update_us = 0.0      # Last cold path duration (μs)

        # Schedule statistics collection for analysis
        self._schedule_stats_enabled = os.environ.get(
            "VLLM_COLLECT_SCHEDULE_STATS", "0") == "1"
        self._schedule_stats: list[dict] = []
        self._schedule_stats_start_time: float | None = None  # Set on first record
        self._schedule_stats_file = os.environ.get(
            "VLLM_SCHEDULE_STATS_FILE", "schedule_stats.json")

        # Register atexit handler to save stats on shutdown
        if self._schedule_stats_enabled:
            atexit.register(self._save_stats_on_exit)

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
            stats.update({
                "hazard_p0": self.pd_hazard_p0,
                "hazard_eta": self.pd_hazard_eta,
                "ifr_sample_count": len(self.pd_ifr_samples),
                "ifr_update_counter": self.pd_ifr_update_counter,
                "ifr_update_interval": self.pd_ifr_update_interval,
                "ifr_window_size": self.pd_ifr_window_size,
                "theta_max": self.pd_theta_max,
            })
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
                tau.append(float('inf'))
            else:
                power *= one_minus_p  # power = (1-p)^j
                denom = 1.0 - power
                if denom <= 0:
                    tau.append(float('inf'))
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
                return float('inf')
            return theta / (1 - theta) + math.log(1 - theta)

        # f(θ) is monotonically increasing from f(0+) = 0 to f(1-) = +∞
        # Use bisection to find θ such that f(θ) = C

        # Edge case: if C is very small, θ* ≈ 0
        if C <= 1e-6:
            return 0.01  # Use 1% as the lower bound

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

        # Clamp to reasonable range [0.01, 0.99]
        theta_star = max(0.01, min(0.99, theta_star))

        logger.debug(f"Computed optimal ratio: θ*={theta_star:.4f} "
                     f"(p={self.pd_p}, α_p={self.pd_alpha_p}, α_d={self.pd_alpha_d}, C={C:.6f})")

        return theta_star

    # ================================================================
    # CFR midpoint algorithm (Algorithm alg:midpoint)
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
            return 0.01
        if p_0 >= 1.0 - 1e-9:
            return 0.99

        rhs = (-math.log(1.0 - p_0)) * self.pd_alpha_p / self.pd_alpha_d
        if rhs <= 1e-12:
            return 0.01

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
        return max(0.01, min(0.99, theta_zero))

    def _compute_midpoint_k(
        self, theta_zero: float, p_0: float, n: int, mu_L: float
    ) -> tuple[float, int, int, int]:
        """Midpoint k̂ following Eqs. eq:Rbar / eq:TP_int / eq:midpoint_k.

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

    def _compute_memory_safe_n(
        self, theta_zero: float, p_0: float, mu_L: float
    ) -> int:
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
        sigma_sq = (2.0 * Lambda * (1.0 + (p_0 * mu_L + lambda_theta) ** 2)
                    / max(p_0 ** 2, 1e-18))
        sigma = math.sqrt(max(sigma_sq, 0.0))

        # Solve N · D + σ √(N · log(1/ε)) ≤ C  →
        #   √N = ( -σ√L + √(σ² L + 4 D C) ) / (2 D)
        sqrt_N = (-sigma * math.sqrt(log_inv_eps)
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
        """Diagnostic Δ(N) (Eq. eq:diagnostic) for the adaptive selector.

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

        beta_eb_w = ((self.pd_beta_p * mu_L + self.pd_beta_d * mu_O)
                     / (mu_L + mu_O))
        beta_mb_e = float(os.environ.get("VLLM_PD_BETA_MB_E", str(beta_eb_w)))
        alpha_mb = float(os.environ.get("VLLM_PD_ALPHA_MB",
                                          str(self.pd_alpha_p)))

        log_one_minus_theta = math.log(max(1.0 - theta_zero, 1e-9))
        eb_term = (self.pd_alpha_p
                   - self.pd_alpha_d * log_one_minus_theta * mu_O) / k_hat
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
        eta = max(0.0, eta)     # η >= 0 for IFR (clamp negative to CFR)

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
        Compute IFR correction Δθ based on the IFR threshold theorem (thm:threshold_ifr).

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
        denominator = p_0 ** 2 * theta_cfr

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

        This implements the online adaptive threshold selection (alg:adaptive_joint):
        1. Estimate hazard rate parameters (p_0, η) from samples
        2. Compute CFR baseline θ*_CFR using Proposition 1
        3. If η > 0, apply IFR correction from the IFR threshold theorem (thm:threshold_ifr)
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
        theta_star = max(0.01, theta_star)

        logger.debug(
            f"IFR optimal ratio: θ*={theta_star:.4f} "
            f"(θ*_CFR={theta_cfr:.4f}, Δθ={theta_star - theta_cfr:.4f}, "
            f"p_0={p_0:.6f}, η={eta:.8f})"
        )

        return theta_star

    def _update_ifr_threshold(self) -> None:
        """
        Online adaptive threshold update (alg:adaptive_joint).

        Called every M completions when window has >= W_min samples.
        Updates hazard rate parameters and recomputes θ*.
        EMA smoothing is applied to θ* to damp oscillations caused by
        noisy hazard-rate estimates.
        """
        old_ratio = self.pd_k_ratio
        old_k = self.pd_switch_threshold_k

        # Step 1: Estimate hazard rate parameters from sliding window
        p_0, eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = eta

        # Step 2: Compute CFR baseline θ_0 using p_0
        old_p = self.pd_p
        self.pd_p = p_0
        theta_0 = self._compute_optimal_ratio()
        self.pd_p = old_p

        # Step 3: Apply IFR correction if η > 0
        if eta > 0:
            delta_theta = self._compute_ifr_correction(theta_0)
            theta_star = theta_0 + delta_theta
        else:
            delta_theta = 0.0
            theta_star = theta_0

        # Step 4: Clamp to [max(0.01, theta_floor), θ_max].
        # theta_floor (env VLLM_PD_THETA_FLOOR, default 0.0) is a regularising
        # lower bound on the adaptive controller — see __init__ comment for why.
        theta_star = max(0.01, self.pd_theta_floor, min(theta_star, self.pd_theta_max))

        # Step 5: EMA smoothing to damp oscillations from noisy estimates.
        # During cold start (first update), assign directly.
        if not self.pd_ifr_theta_initialized:
            self.pd_k_ratio = theta_star
            self.pd_ifr_theta_initialized = True
        else:
            alpha = self.pd_ifr_theta_ema_alpha
            self.pd_k_ratio = alpha * theta_star + (1 - alpha) * self.pd_k_ratio

        self.pd_switch_threshold_k = max(
            1, int(self.pd_k_ratio * self.pd_batch_size_N))

        # Log significant changes
        if abs(self.pd_k_ratio - old_ratio) > 0.01 or old_k != self.pd_switch_threshold_k:
            logger.info(
                f"[P/D IFR] online update: θ*={old_ratio:.4f}->{self.pd_k_ratio:.4f} "
                f"(θ_CFR={theta_0:.4f}, Δθ={delta_theta:.4f}), "
                f"k*={old_k}->{self.pd_switch_threshold_k} "
                f"(p_0={p_0:.6f}, η={eta:.8f}, samples={len(self.pd_ifr_samples)})"
            )

    def _update_cfr_threshold(self) -> None:
        """Online CFR midpoint update (Algorithm alg:midpoint).

        Estimates p_0 from the sliding window, recomputes the exact θ_0
        (Eq. theta_base), the memory-safe N̂ (Eq. eq:Nstar) when enabled,
        and the midpoint k̂ (Eq. eq:midpoint_k).  Also evaluates the
        diagnostic Δ(N) used by the adaptive selector.
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
        theta_zero_clamped = max(0.01, min(self.pd_theta_max, theta_zero))

        # Step 3: Optional dynamic N̂ from memory-safe sizing (Prop. memory).
        if self.pd_auto_compute_n:
            n_hat = self._compute_memory_safe_n(theta_zero_clamped, p_0, mu_L)
            if n_hat != self.pd_batch_size_N:
                self._record_n_update(
                    self.pd_batch_size_N, n_hat, "cfr_memory_safe")
                self.pd_batch_size_N = n_hat
        n_eff = max(1, self.pd_batch_size_N)

        # Step 4: Midpoint k̂.
        k_hat_real, m_minus, m_plus, m_star = self._compute_midpoint_k(
            theta_zero_clamped, p_0, n_eff, mu_L)
        k_hat_int = max(1, min(n_eff - 1, int(round(k_hat_real))))

        # Step 5: Diagnostic Δ(N) (used by adaptive selector / auto mode).
        delta_diag = self._compute_diagnostic_delta(
            theta_zero_clamped, k_hat_real, n_eff, mu_L, mu_O)

        # Step 6: EMA smoothing on the realised θ̂ = k̂ / N̂ to damp noise.
        theta_hat = k_hat_real / float(n_eff)
        theta_hat = max(0.01, min(self.pd_theta_max, theta_hat))
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

        self.pd_cfr_update_history.append({
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
        })

        if (abs(self.pd_k_ratio - old_ratio) > 0.01
                or old_k != self.pd_switch_threshold_k
                or old_n != self.pd_batch_size_N):
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
        self.pd_n_update_history.append({
            "timestamp": timestamp,
            "old_N": old_n,
            "new_N": new_n,
            "reason": reason,
            "k_star": self.pd_switch_threshold_k,
            "avg_output_tokens": self.pd_avg_output_tokens,
        })

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
        if not hasattr(self.kv_cache_manager, 'block_pool'):
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
        kv_threshold_max = float(
            os.environ.get("VLLM_PD_KV_THRESHOLD_MAX", "0.6"))
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
        - N * (avg_prompt + avg_output * margin) / block_size <= total_blocks * (1 - reserve)

        Solving for N:
        - N <= total_blocks * (1 - reserve) * block_size / (avg_prompt + avg_output * margin)

        Returns:
            int: Adaptive batch size N
        """
        if not hasattr(self.kv_cache_manager, 'block_pool'):
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
        - "direct": k* computed via Proposition 1 (unless VLLM_PD_K_STAR is user-specified)
        - "ratio":  k* = theta* x N, theta* computed from p (unless VLLM_PD_K_RATIO is user-specified)
        - "ifr":    k* = theta* x N, theta* from the IFR correction formula (using the hazard-rate estimate)
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
            if (self.pd_ifr_update_counter >= self.pd_ifr_update_interval
                    and len(self.pd_ifr_samples) >= self.pd_ifr_min_samples):
                self._update_ifr_threshold()
                self.pd_ifr_update_counter = 0

        # CFR mode: online midpoint update (Algorithm alg:midpoint)
        elif self.pd_k_mode == "cfr":
            self.pd_ifr_samples.append(output_tokens)
            self.pd_ifr_update_counter += 1
            if (self.pd_ifr_update_counter >= self.pd_ifr_update_interval
                    and len(self.pd_ifr_samples) >= self.pd_ifr_min_samples):
                self._update_cfr_threshold()
                self.pd_ifr_update_counter = 0

        # Check if we've reached the update interval
        if self.pd_batch_completed_count < self.pd_param_update_interval:
            return  # Fast exit - no expensive operations

        # COLD PATH: Reached threshold, do the expensive operations
        _cold_path_start = time.perf_counter()

        self.pd_total_completed += self.pd_batch_completed_count

        if self.pd_batch_total_output_tokens > 0:
            batch_mean_len = (self.pd_batch_total_output_tokens /
                              self.pd_batch_completed_count)
            batch_p = 1.0 / batch_mean_len

            if not self.pd_param_initialized:
                # First batch: direct assignment
                self.pd_p = batch_p
                self.pd_avg_output_tokens = batch_mean_len
                self.pd_param_initialized = True
            else:
                # EMA update
                self.pd_p = (self.pd_ema_alpha * batch_p +
                             (1 - self.pd_ema_alpha) * self.pd_p)
                self.pd_avg_output_tokens = (
                    self.pd_ema_alpha * batch_mean_len +
                    (1 - self.pd_ema_alpha) * self.pd_avg_output_tokens)

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
                        f"[P/D] ratio update: θ*={old_ratio:.4f}->{self.pd_k_ratio:.4f}, "
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
          LHS = β_CP_e(r̂) - β_EB_w
          RHS = (1/(μ_L+μ_o)) * [
                  (α_p - α_d·ln(1-θ₀)·μ_o)/(θ₀·N_obs)
                  - α_CP·(1+μ_o)/N_obs
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

        # --- LHS: β_CP_e(r̂) - β_EB_w ---
        # β_EB_w: workload-weighted exclusive marginal cost
        beta_EB_w = ((self.pd_beta_p * mu_L + self.pd_beta_d * mu_o)
                     / (mu_L + mu_o))

        # β_CP_e(r): CP effective marginal cost from offline profile
        if self._cp_cost_profiled:
            beta_CP_e = (self._cp_cost_a
                         + self._cp_cost_b * r
                         + self._cp_cost_c * r * r)
        else:
            # Fallback: no profiled CP cost → LHS = 0
            # Decision driven entirely by overhead comparison (RHS)
            beta_CP_e = beta_EB_w

        LHS = beta_CP_e - beta_EB_w

        # --- RHS: amortized fixed-cost comparison ---
        # θ₀: current ratio (from IFR or ratio estimator)
        theta0 = self.pd_k_ratio if hasattr(self, 'pd_k_ratio') else 0.5
        if theta0 <= 0 or theta0 >= 1:
            return  # invalid, skip

        log_1_minus_theta = math.log(1 - theta0)  # negative
        eb_overhead = ((self.pd_alpha_p
                        - self.pd_alpha_d * log_1_minus_theta * mu_o)
                       / (theta0 * N_obs))
        cp_overhead = self._alpha_cp * (1 + mu_o) / N_obs

        RHS = (1.0 / (mu_L + mu_o)) * (eb_overhead - cp_overhead)

        # --- Decision with hysteresis ---
        delta = self._mode_switch_delta

        if self._mode_cooldown > 0:
            self._mode_cooldown -= 1
            return

        old_mode = self._active_scheduler
        if LHS > RHS + delta:
            # Contention dominates → EB is better
            if self._active_scheduler != "eb":
                self._transition_to_eb()
                self._mode_cooldown = self._mode_cooldown_max
        elif LHS < RHS - delta:
            # Amortization dominates → CP is better
            if self._active_scheduler != "cp":
                self._transition_to_cp()
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
                "beta_CP_e": beta_CP_e,
                "beta_EB_w": beta_EB_w,
                "total_completed": self.pd_total_completed,
            }
            self._mode_switch_history.append(switch_record)
            logger.info(
                f"[EB+] Mode switch: {old_mode} -> "
                f"{self._active_scheduler} | "
                f"LHS={LHS:.6f}, RHS={RHS:.6f}, r={r:.3f}, "
                f"N_obs={N_obs:.1f}, θ₀={theta0:.4f}, "
                f"β_CP_e={beta_CP_e:.6f}, β_EB_w={beta_EB_w:.6f}"
            )

    def _transition_to_eb(self) -> None:
        """Transition from CP mode to EB mode.

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
            # capped at max_num_running_reqs.  Without this, a CP→EB
            # transition under light running but heavy waiting (e.g. a
            # concurrency spike) would leave N tiny, starving prefill.
            self.pd_batch_size_N = min(
                num_decoding + len(self.waiting),
                self.max_num_running_reqs)
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
            f"[EB+] CP -> EB: phase={self.pd_phase}, "
            f"decoding={num_decoding}, running={len(self.running)}, "
            f"N={self.pd_batch_size_N}, k*={self.pd_switch_threshold_k}"
        )

    def _transition_to_cp(self) -> None:
        """Transition from EB mode to CP mode.

        The running list is shared so CP can immediately schedule all
        requests. Clear PD-specific tracking state.
        """
        self._active_scheduler = "cp"

        # Clear PD tracking state — CP doesn't use it.
        # Will be rebuilt if we switch back to EB.
        self.pd_decoding_requests.clear()
        self.pd_phase = 0
        self.pd_prefilled_count = 0
        self.pd_completed_decode_count = 0
        self.pd_refill_target = 0

        logger.info(
            f"[EB+] EB -> CP: running={len(self.running)}, "
            f"waiting={len(self.waiting)}"
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
            if hasattr(self, 'encoder_cache_manager'):
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
                logger.warning("[P/D] Cleaning %d orphaned decoding IDs",
                               len(orphaned_ids))
                self.pd_decoding_requests -= orphaned_ids

        num_pending_chunks = len(self.chunk_prefilling)
        if num_pending_chunks > 0:
            running_set = set(self.running)
            # Clean up: 1) requests not in running, 2) requests that completed prefill
            orphaned_chunks = [r for r in self.chunk_prefilling
                               if r not in running_set
                               or r.num_computed_tokens >= r.num_prompt_tokens]
            if orphaned_chunks:
                logger.warning("[P/D] Cleaning %d orphaned/completed chunk_prefilling",
                               len(orphaned_chunks))
                for req in orphaned_chunks:
                    self.chunk_prefilling.remove(req)
                    # If prefill completed, add to decoding set and count
                    if (req in running_set
                            and req.num_computed_tokens >= req.num_prompt_tokens
                            and req.request_id not in self.pd_decoding_requests):
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
        if hasattr(self.kv_cache_manager, 'block_pool'):
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
            # CP→EB transition under light running but heavy waiting, or a
            # cold-start), scale N up so that the 1→2 transition can refill
            # enough requests to keep the pipeline busy.
            target_n = self.max_num_running_reqs
            if (self.pd_batch_size_N < target_n
                    and waiting_count >= target_n // 2
                    and (time.monotonic() - self.pd_last_n_update_time
                         >= self.pd_n_update_cooldown)):
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
            # KV-AWARE GUARD (Appendix app:kv-aware-gate): also require KV
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
                or kv_cache_full)  # KV cache full escape
            if ready_to_decode:
                # When KV cache is full, must transition even with pending chunks
                # to free memory. Proactively preempt them to free KV cache.
                # Otherwise, wait for chunked prefills to complete.
                if not has_pending_chunks or kv_cache_full:
                    if kv_cache_full and has_pending_chunks:
                        preempted_chunks, preempted_tokens = (
                            self._preempt_chunk_prefilling())
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
                f"[P/D] N update: {old_n}->{new_n}, k*={old_k}->{self.pd_switch_threshold_k}"
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
            target = (self.pd_batch_size_N if self.pd_phase == 0
                      else self.pd_refill_target)
            remaining = target - self.pd_prefilled_count

            # Continue chunked prefills first
            # Note: We don't check `remaining > 0` here because we must continue
            # all existing chunked prefills to prevent deadlock. The `remaining`
            # counter only limits NEW prefills from the waiting queue.
            if self.scheduler_config.enable_chunked_prefill:
                req_index = 0
                while (req_index < len(self.chunk_prefilling)
                       and token_budget > 0):
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
                        request, num_new_tokens,
                        num_lookahead_tokens=effective_lookahead_tokens)
                    if new_blocks is None:
                        req_index += 1
                        continue

                    # Check if prefill completes
                    # Use num_prompt_tokens (not num_tokens) to match is_prefill() logic
                    will_complete = (request.num_prompt_tokens - request.num_computed_tokens
                                     <= num_new_tokens)
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
                        self.kv_cache_manager.get_computed_blocks(request))
                    num_computed_tokens = num_local + num_external_computed_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_local = 0
                    num_computed_tokens = request.num_computed_tokens

                num_new_tokens = request.num_tokens - num_computed_tokens
                num_new_tokens = self._apply_long_prefill_threshold(num_new_tokens)

                is_chunked = False
                if (not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget):
                    self.waiting.pop_request()
                    skipped.prepend_request(request)
                    continue
                elif num_new_tokens > token_budget:
                    is_chunked = True

                num_new_tokens = min(num_new_tokens, token_budget)
                if num_new_tokens <= 0:
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens + num_external_computed_tokens,
                    num_local, new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens)
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
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp)

                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

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

                num_new_tokens = (request.num_tokens_with_spec
                                  + request.num_output_placeholders
                                  - request.num_computed_tokens)
                num_new_tokens = self._apply_long_prefill_threshold(num_new_tokens)
                num_new_tokens = min(num_new_tokens, token_budget)

                max_total = min(request.num_prompt_tokens + request.max_tokens,
                                self.max_model_len)
                num_new_tokens = min(num_new_tokens,
                                     max_total - 1 - request.num_computed_tokens)
                if num_new_tokens == 0:
                    req_index += 1
                    continue

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)

                if new_blocks is None:
                    # Need to preempt
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running, key=lambda r: (r.priority, r.arrival_time))
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id]
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
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)
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
                    any_request.request_id))

        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs]

        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs, scheduled_resumed_reqs,
            num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks)

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
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes())

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def schedule(self) -> SchedulerOutput:
        """Entry point for scheduling. Dispatches to P/D, default, or auto."""
        if self._schedule_stats_enabled:
            t_start = time.perf_counter()

        # Track demand EMA for auto mode (running + waiting = true load)
        # Using only len(running) is wrong under EB: phase separation
        # drains running while filling waiting, but total demand is constant.
        if self.scheduler_mode == "auto":
            demand = float(len(self.running) + len(self.waiting))
            # Asymmetric EMA: fast ramp-up (0.3), slow ramp-down (0.03)
            # so we react quickly to traffic surges but don't prematurely
            # switch back when demand dips briefly
            if demand > self._n_obs:
                a = 0.3   # fast follow upward
            else:
                a = 0.03  # slow follow downward
            self._n_obs = a * demand + (1 - a) * self._n_obs

            # Demand surge detection: if instantaneous demand is 2x the
            # current EMA AND we haven't evaluated recently, trigger
            # immediate mode evaluation without waiting for cold path.
            if (demand > self._n_obs * 2
                    and self._mode_cooldown == 0
                    and self.pd_param_initialized):
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
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

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
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_tokens_to_restore = sum(
                                    preempted_req.get_num_encoder_tokens(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_tokens_to_restore
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
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
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
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
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

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
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
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

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
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
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

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self._update_connector_prefix_cache_stats(request)

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
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

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
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
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
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
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

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
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

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
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

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

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
            if self.use_pp:
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

        # Check remote cache first
        if self.ec_connector is not None:
            remote_cache_has_item = self.ec_connector.has_caches(request)
        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

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
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
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
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_tokens_to_schedule
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

            if self.ec_connector is not None and remote_cache_has_item[i]:
                mm_hashes_to_schedule.add(request.mm_features[i].identifier)
                external_load_encoder_input.append(i)
                num_tokens_to_schedule += num_encoder_tokens
                continue

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self,
        scheduler_output: SchedulerOutput,
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # PERF: in case of chunked prefill,
        # request might not include any new tokens.
        # Therefore, we might introduce some additional
        # cycle to fill in the bitmask, which could be a big no-op.
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id)) and req.use_structured_output
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
                kv_connector_output.invalid_block_ids
            )

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
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
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
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len, pooler_output)

            if stopped:
                # P/D scheduling: count completed decode requests
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
                            a * pl + (1 - a) * self.pd_avg_prompt_len)
                        if self.scheduler_mode == "auto":
                            self._avg_prompt_len = self.pd_avg_prompt_len
                    elif (self.scheduler_mode == "auto"
                          and self._active_scheduler == "cp"):
                        # Auto+CP path: still feed output samples for
                        # parameter tracking and mode selection
                        output_tokens = request.num_tokens - request.num_prompt_tokens
                        if output_tokens > 0:
                            self._update_params_online(output_tokens)
                        # Track prompt length EMA for mode selection
                        prompt_len = float(request.num_prompt_tokens)
                        a = self.pd_avg_prompt_ema_alpha
                        self.pd_avg_prompt_len = (
                            a * prompt_len
                            + (1 - a) * self.pd_avg_prompt_len)
                        self._avg_prompt_len = self.pd_avg_prompt_len

                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
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
                        num_cached_tokens=request.num_cached_tokens,
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
                spec_decoding_stats, kv_connector_stats, cudagraph_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
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

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids
                )
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

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

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

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
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

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
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
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
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> SpecDecodingStats | None:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        # Save schedule stats if collection was enabled
        if self._schedule_stats_enabled and self._schedule_stats:
            stats_file = os.environ.get(
                "VLLM_SCHEDULE_STATS_FILE", "schedule_stats.json")
            self.save_schedule_stats(stats_file)

        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def _update_connector_prefix_cache_stats(self, request: Request) -> None:
        if self.connector_prefix_cache_stats is None:
            return

        self.connector_prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=request.num_external_computed_tokens,
            preempted=request.num_preemptions > 0,
        )

    def _make_connector_prefix_cache_stats(self) -> PrefixCacheStats | None:
        if self.connector_prefix_cache_stats is None:
            return None
        stats = self.connector_prefix_cache_stats
        self.connector_prefix_cache_stats = PrefixCacheStats()
        return stats

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

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

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
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

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
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
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
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

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
                request.num_external_computed_tokens -= num_affected_tokens
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
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
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

        # Count prefill vs decode tokens
        # Note: num_computed_tokens has already been updated by _update_after_schedule,
        # so we need to subtract num_tokens to get the state BEFORE this scheduling step.
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

        self._schedule_stats.append({
            "timestamp": timestamp,
            "elapsed_us": elapsed_time * 1e6,
            "execution_time_us": 0,  # Will be updated by next schedule() call
            "scheduler_type": (
                self._active_scheduler
                if self.scheduler_mode == "auto"
                else ("pd" if self.use_pd_scheduler else "default")),
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
            "num_decoding_reqs": len(self.pd_decoding_requests) if self.use_pd_scheduler else 0,
            "num_preempted_reqs": num_preempted_reqs,
            "preempted_tokens": preempted_tokens,
            # Adaptive scheduling values
            "avg_output_tokens": self.pd_avg_output_tokens if self.use_pd_scheduler else 0,
            "adaptive_kv_threshold": self._compute_adaptive_kv_threshold() if self.use_pd_scheduler else 0,
            # Hazard rate estimation (IFR / CFR online estimator)
            "hazard_p0": self.pd_hazard_p0 if (self.use_pd_scheduler and self.pd_k_mode in ("ifr", "cfr")) else 0,
            "hazard_eta": self.pd_hazard_eta if (self.use_pd_scheduler and self.pd_k_mode in ("ifr", "cfr")) else 0,
            "ifr_sample_count": len(self.pd_ifr_samples) if (self.use_pd_scheduler and self.pd_k_mode in ("ifr", "cfr")) else 0,
            # CFR midpoint diagnostics (last update; 0 outside cfr/auto)
            "mu_L_estimate": self.pd_avg_prompt_len if self.use_pd_scheduler else 0,
            "mu_O_estimate": self.pd_avg_output_tokens if self.use_pd_scheduler else 0,
            "theta_zero_last": self.pd_theta_zero_last if self.use_pd_scheduler else 0,
            "k_hat_midpoint_last": self.pd_k_hat_midpoint_last if self.use_pd_scheduler else 0,
            "n_hat_safe_last": self.pd_n_hat_safe_last if self.use_pd_scheduler else 0,
            "delta_diagnostic_last": self.pd_delta_diagnostic_last if self.use_pd_scheduler else 0,
            "oom_event_count": self.pd_oom_event_count if self.use_pd_scheduler else 0,
            # Parameter update overhead (cold path)
            "param_update_count": self._param_update_count,
            "last_param_update_us": self._last_param_update_us,
            # EB+ auto mode stats
            "active_scheduler": (
                self._active_scheduler
                if self.scheduler_mode == "auto" else ""),
            "n_obs": (
                self._n_obs
                if self.scheduler_mode == "auto" else 0),
            "mode_switch_count": (
                self._mode_switch_count
                if self.scheduler_mode == "auto" else 0),
        })

    def save_schedule_stats(self, filepath: str = "schedule_stats.json") -> None:
        """Save collected schedule statistics to a JSON file."""
        import json
        with open(filepath, "w") as f:
            json.dump({
                "stats": self._schedule_stats,
                "summary": self.get_schedule_stats_summary(),
                "n_update_history": self.pd_n_update_history if self.use_pd_scheduler else [],
                "mode_switch_history": (
                    self._mode_switch_history
                    if self.scheduler_mode == "auto" else []),
                "cfr_update_history": (
                    self.pd_cfr_update_history
                    if self.use_pd_scheduler else []),
                "pd_config": {
                    "k_mode": (self.pd_k_mode if self.use_pd_scheduler else ""),
                    "scheduler_mode": self.scheduler_mode,
                    "alpha_p": (self.pd_alpha_p if self.use_pd_scheduler else 0),
                    "beta_p": (self.pd_beta_p if self.use_pd_scheduler else 0),
                    "alpha_d": (self.pd_alpha_d if self.use_pd_scheduler else 0),
                    "beta_d": (self.pd_beta_d if self.use_pd_scheduler else 0),
                    "auto_compute_n": (self.pd_auto_compute_n
                                        if self.use_pd_scheduler else False),
                    "oom_tolerance": (self.pd_oom_tolerance
                                        if self.use_pd_scheduler else 0),
                    "max_num_seqs": int(self.max_num_running_reqs),
                    "final_N": (int(self.pd_batch_size_N)
                                 if self.use_pd_scheduler else 0),
                    "final_k_star": (int(self.pd_switch_threshold_k)
                                      if self.use_pd_scheduler else 0),
                    "total_oom_events": (int(self.pd_oom_event_count)
                                          if self.use_pd_scheduler else 0),
                },
            }, f, indent=2)
        logger.info(f"[Schedule Stats] Saved {len(self._schedule_stats)} records to {filepath}")

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
        execution_time_us = [s.get("execution_time_us", 0) for s in self._schedule_stats
                            if s.get("execution_time_us", 0) > 0]
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
                "preemption_rate": sum(1 for n in num_preempted if n > 0) / len(num_preempted) if num_preempted else 0,
            },
            "n_updates": {
                "total_updates": len(self.pd_n_update_history) if self.use_pd_scheduler else 0,
                "by_reason": self._count_n_updates_by_reason() if self.use_pd_scheduler else {},
            },
            "param_update_overhead": {
                "total_updates": self._param_update_count,
                "total_time_us": self._param_update_total_us,
                "mean_time_us": self._param_update_total_us / self._param_update_count if self._param_update_count > 0 else 0,
            },
        }

    def _count_n_updates_by_reason(self) -> dict:
        """Count N updates grouped by reason."""
        counts: dict[str, int] = {}
        for update in self.pd_n_update_history:
            reason = update.get("reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        return counts
