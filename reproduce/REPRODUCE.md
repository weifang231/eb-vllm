# Step-by-step reproduction recipe

Annotated end-to-end recipe for reproducing the EB(k̂*) paper (ICML 2026).
Times below are wall-clock estimates on 8×RTX PRO 6000 Blackwell (96 GB each).

## 0. Environment

Two install paths are documented below. **The source build is the
default recommendation** because it is immune to upstream wheel ABI
drift (see "Why not VLLM_USE_PRECOMPILED" below). The precompiled-wheel
path is faster (~30 s vs ~20 min) but requires manually pinning torch
and the wheel commit to a matching pair; it is documented for experts.

### 0.1 Source build (recommended, ~30 min)

> Verified end-to-end on H200 + torch 2.9.0+cu128 + Python 3.13.11
> (2026-05-19). Build time was ~30 min with `MAX_JOBS=16` on a 16-core
> host; expect longer on machines with fewer cores or when building for
> more CUDA archs.

```bash
cd <repo-root>             # i.e. eb-vllm/, the dir with setup.py
python3 -m venv .venv && source .venv/bin/activate

# Build deps (pip's --no-build-isolation skips PEP 517 isolation, so the
# build backend has to find these in the venv itself).
pip install --upgrade pip setuptools setuptools_scm wheel numpy

# Torch must match the pin in requirements/cuda.txt (torch==2.9.0). PyPI
# alone does not host torch+cu12.x wheels; you must use the pytorch.org
# index. cu128 also works (forward-compat with system 12.9 driver).
pip install torch==2.9.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# CUDA_HOME must point to a cu12.x toolkit (not cu13.x, which is ABI-
# incompatible with torch+cu12.x). Inspect /usr/local/ to find yours.
# SETUPTOOLS_SCM_PRETEND_VERSION is required when the repo is not a git
# checkout (e.g. extracted from a tarball) — scm cannot infer a version.
CUDA_HOME=/usr/local/cuda-12.8 \
  PATH=/usr/local/cuda-12.8/bin:$PATH \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+eb \
  MAX_JOBS=16 \
  pip install -e . --no-build-isolation

vllm --version              # smoke check

# Analysis / plotting deps
pip install matplotlib pandas scipy
```

A `docker/` recipe is also provided for a fully reproducible
environment; the `vllm-openai` base image already ships a known-good
toolchain.

### 0.2 Precompiled wheel (experts only, ~30 s — fragile)

`VLLM_USE_PRECOMPILED=1` makes setup.py fetch all 8 `.so` files from
`wheels.vllm.ai/<commit>/<cu-variant>/vllm/`. **Two latent failure
modes** (see "Why not VLLM_USE_PRECOMPILED" below for the full diagnosis):

1. The auto-detected base commit often has no wheels on
   `wheels.vllm.ai` → 404 during install.
2. Upstream's CI builds wheels against torch nightly, but the wheel's
   `Requires-Dist: torch==X` may name a torch version with a different
   `c10::cuda` ABI than the wheel's own `.so`. Installing the named
   torch then yields `ImportError: undefined symbol
   _ZN3c104cuda29c10_cuda_check_implementation...`.

The only reliable invocation pins both the wheel commit and a torch
build whose `libc10_cuda.so` matches the wheel's ABI. As of 2026-05-19,
the PyPI release wheel `vllm==0.21.0` works with `torch==2.11.0+cu129`:

```bash
cd <repo-root>
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools setuptools_scm wheel numpy

# Match what the PyPI vllm 0.21.0 wheel pins (Requires-Dist: torch==2.11.0)
pip install torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Use the PyPI release wheel locally instead of letting setup.py guess a
# wheels.vllm.ai commit (which is usually 404).
pip download --no-deps vllm==0.21.0 -d /tmp/

VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-0.21.0-*.whl \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+eb \
  pip install -e . --no-build-isolation
```

> **Note**: `requirements/cuda.txt` in this fork still pins
> `torch==2.9.0` (inherited from the upstream commit the fork was based
> on). pip will downgrade torch to 2.9.0 during `pip install -e .`
> because of that pin, which then breaks the 2.11-built wheel's ABI.
> Workaround: edit `requirements/cuda.txt` to `torch==2.11.0` before
> the precompiled install, or use `pip install --upgrade torch==2.11.0`
> immediately after to repair it.

### Why not `VLLM_USE_PRECOMPILED=1` as the default?

Four stacked failure modes make this path brittle:

1. **`wheels.vllm.ai` is sparse.** Only a handful of commits per day
   have CI-built wheels; the "base commit in main" that setup.py
   auto-detects is usually one of the missing ones → 404.
2. **PyTorch changed the `c10::cuda` ABI between 2.9 and 2.10.** The
   symbol `c10::cuda::c10_cuda_check_implementation(int, ..., unsigned
   long, bool)` had its 4th argument changed from `int` to `unsigned
   int`, renaming the mangled symbol from `..._ib` to `..._jb`.
3. **Upstream nightly wheels lie about their torch dependency.** They
   declare `Requires-Dist: torch==2.9.0` but are built against torch
   nightly (≥2.10) and require the `_jb` symbol. Installing torch 2.9.0
   stable as the metadata requests then yields ImportError on `vllm._C`.
4. **The official `VLLM_PRECOMPILED_WHEEL_COMMIT` env var was never
   documented in this README.** Without it, setup.py guesses, and the
   guess is almost always wrong (see failure mode 1).


## 1. Generate per-(model, GPU) calibration (one-time)

```bash
# Default --output auto-detects the GPU tag (H200 / RTXPRO6000 / ...) and
# writes to reproduce/calibration/pd_calibration_<model>_<GPU>.json — the
# same path the runner scripts (resolve_calibration in common_eb.sh) look
# up. Pass --output explicitly only if you want a non-standard location.
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
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
python benchmark_flash_attn_sweep.py --output results_$(hostname).json
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
MODEL=Qwen/Qwen3-8B ./run_validation.sh 8
python analyze_validation.py outputs/controller_validation/<GPU>_Qwen3-8B
python plot_validation_grid.py outputs/.../validation_summary.csv \
    --output validation_grid.pdf
```

## 5. §4.3.1 — Synthetic e2e grid (≈1.5–8 h depending on model)

```bash
cd ../synthetic_e2e
MODEL=Qwen/Qwen3-8B    ./run_grid_search.sh 8    # ~1.5 h
MODEL=Qwen/Qwen3-30B-A3B ./run_grid_search.sh 8  # ~7-8 h
python analyze_e2e.py outputs/e2e_grid_search/<GPU>_<MODEL>
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

**Workload protocol** (paper §4.1; set `IGNORE_EOS` and `CUSTOM_OUTPUT_LEN`
to match the per-workload truncation cap):

| Workload | `CUSTOM_OUTPUT_LEN` | `IGNORE_EOS` | Notes |
|----------|---------------------|--------------|-------|
| ShareGPT | `-1` (dataset-native, cap ≤500) | `true` | output mean ≈ 280 tok |
| LongBench | `20` | `true` | forced short output |
| NumimaMath | `4000` | `true` | forced full chain-of-thought |
| WildChat | n/a | n/a | multi-turn chat-mode, see `multiturn/` |

Single-cell example (optimal `(B, N)` from Appendix `tab:optimal-config-h200`):
```bash
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=500 MODEL=Qwen/Qwen3-8B \
    ./run_optimal_only.sh ../outputs/sharegpt_prompts.jsonl 1
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=20 MODEL=Qwen/Qwen3-8B \
    ./run_optimal_only.sh ../outputs/longbench_prefill.jsonl 1
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=4000 MODEL=Qwen/Qwen3-8B \
    ./run_optimal_only.sh ../outputs/numina_math_prompts.jsonl 1
```

For multi-turn dialogue (WildChat) preprocessing, see
[`real_workloads/multiturn/`](real_workloads/multiturn/).

## 7. §4.4 — EB⁺ (≈30 min for traffic, ≈2 h for non-stationary)

**EB⁺ requires offline mixed-batch cost calibration** (Appendix
`app:eb-plus-calibration`). The default coefficients for H200 Qwen3-8B
are shipped in `reproduce/calibration/beta_mb_Qwen3-8B_H200.json`. For
other (model, GPU) pairs, refit from v1 grid data and update env vars:

```bash
# H200 Qwen3-8B (provided)
export VLLM_PD_MB_COST_A=2.494e-05
export VLLM_PD_MB_COST_B=5.193e-05
export VLLM_PD_MB_COST_C=1.478e-05
# Mode-switch hysteresis δ — match observed |LHS-RHS| magnitude
export VLLM_PD_MODE_SWITCH_DELTA=1e-5
```

Then run the EB⁺ experiments:
```bash
cd ../eb_plus/traffic
MODEL=Qwen/Qwen3-8B ./run_adaptive_selector.sh 8
python analyze_selector.py outputs/adaptive_selector/<GPU>_<MODEL>

cd ../non_stationary
python generate_distribution_shift_dataset.py
./run_distribution_shift.sh 0    # positional arg = GPU_ID, not MAX_GPUS
./run_concurrency_shift.sh 1
python plot_distribution_shift.py outputs/
```

If `ebplus_stats.json` shows `mode_switch_count = 0`, the crossover
hysteresis `VLLM_PD_MODE_SWITCH_DELTA` is too large for the observed
LHS-RHS magnitude — try `1e-6` (or refit `(a, b, c)`).

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
