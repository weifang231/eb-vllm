# Step-by-step reproduction recipe

Annotated end-to-end recipe for reproducing the THETA paper (ICML 2026).
Times below are wall-clock estimates on 8×RTX PRO 6000 Blackwell (96 GB each).

## 0. Environment

```bash
# Build vLLM from this fork (release/icml2026 branch, tagged icml2026-camera-ready)
cd <repo-root>
pip install -e .               # ~20 min if compiling kernels
.venv/bin/vllm --version       # smoke check

# Required Python deps for analysis / plotting
pip install matplotlib numpy pandas scipy
```

A `docker/` recipe is provided for a fully reproducible environment.

## 1. Generate per-(model, GPU) calibration (one-time)

```bash
cd reproduce
python -m vllm.v1.core.sched.calibration \
    --model Qwen/Qwen3-8B \
    --output calibration/pd_calibration_Qwen3-8B_$(./common/common_cfr.sh detect && echo $GPU_TAG).json
```

Repeat for any other (model, GPU) you intend to run. The sample
`pd_calibration_Qwen3-8B_H200.json` is provided; the runner falls back
to a per-GPU file first, then the legacy non-GPU name.

## 2. §3 — Cost model validation (≈30 min)

```bash
cd cost_model/linear_model
python benchmark_execution_time.py --output results.json
python analyze_prefill_linearity_all.py
python plot_execution_time.py --output execution_time.pdf

cd ../kernel_breakdown
python benchmark_flash_attn.py --output results_$(hostname).json
python plot_kernel_breakdown.py --output kernel_breakdown.pdf
```

## 3. §3 — Hazard rate (≈1 h)

```bash
cd ../../hazard_rate
./run_hazard_rate_experiment.sh 4    # 4 GPUs
python analyze_hazard_rate.py outputs/
python plot_cfr_ifr.py --output CFR_IFR.pdf
python plot_sharegpt_hazard_rate.py --output hazard_rate_comparison.pdf
```

## 4. §4.2 — Controller validation (≈30 min)

```bash
cd ../validation
MODEL=Qwen/Qwen3-8B ./run_validation_cfr.sh 8
python analyze_cfr_validation.py outputs/controller_validation/<GPU>_Qwen3-8B
python plot_validation_grid.py outputs/.../validation_summary.csv \
    --output validation_grid.pdf
```

## 5. §4.3.1 — Synthetic e2e grid (≈1.5–8 h depending on model)

```bash
cd ../synthetic_e2e
MODEL=Qwen/Qwen3-8B    ./run_grid_search_cfr.sh 8    # ~1.5 h
MODEL=Qwen/Qwen3-30B-A3B ./run_grid_search_cfr.sh 8  # ~7-8 h
python analyze_cfr_e2e.py outputs/e2e_grid_search/<GPU>_<MODEL>
python plot_synthetic_e2e.py --output fig_synthetic_e2e.pdf
```

## 6. §4.3.2/3 — Real workloads (≈6–10 h per (model, GPU))

```bash
cd ../real_workloads
MODEL=Qwen/Qwen3-8B GPUS=0,1,2,3 ./run_grid_search.sh
python analyze_grid_search.py outputs/<GPU>_<MODEL>/
python plot_real_workload_latency.py \
    --optimal-json outputs/<GPU>_<MODEL>/optimal_per_scheduler.json \
    --ttft-output ttft.pdf --tpot-output tpot.pdf
```

For multi-turn dialogue (WildChat) preprocessing, see
[`real_workloads/multiturn/`](real_workloads/multiturn/).

## 7. §4.4 — EB⁺ (≈30 min for traffic, ≈2 h for non-stationary)

```bash
cd ../eb_plus/traffic
MODEL=Qwen/Qwen3-8B ./run_adaptive_selector_cfr.sh 8
python analyze_cfr_selector.py outputs/adaptive_selector/<GPU>_<MODEL>

cd ../non_stationary
python generate_distribution_shift_dataset.py
./run_distribution_shift.sh
./run_concurrency_shift.sh
python plot_distribution_shift.py outputs/
```

## 8. §4.4 — Disaggregation comparison (≈4 h on 4 GPUs)

```bash
cd ../../disagg
./run_2gpu_comparison.sh
./run_4gpu_comparison.sh
./run_disagg_baseline.sh
```

## 9. §4.4 — Long context (≈2 h)

```bash
cd ../long_context
./run_long_context_comparison.sh
./run_128k_all_models.sh
python plot_long_context.py outputs/long_context_summary.csv \
    --output combined_ctx_comparison_tok1024.pdf
```

## 10. §4.5 — Scalability (reuses real_workloads, swap GPU/MODEL)

See [`scalability/README.md`](scalability/) for the GPU/model
matrix and the `plot_scalmodel.py` invocation.

## Total time

End-to-end full reproduction is **roughly one week** on 8×RTX 6000.
For a sampler tour, run §3 + §4.2 + §4.3.1 synthetic (≈4 h total).
