# eb-vllm: Exclusive Batching scheduler for vLLM

Code release for the ICML 2026 paper:

> **Threshold-Based Exclusive Batching for Memory-Bandwidth-Constrained LLM Inference**
> Weifang Zhang*, Yuzhou Nie*, Bowen Pang, Guangrui Ma, Shining Wu
> (\*equal contribution)

This repository extends [vLLM](https://github.com/vllm-project/vllm) (at commit
[`d1e1fb436`](https://github.com/vllm-project/vllm/commit/d1e1fb436), 2025-12-10)
with the **EB (Exclusive Batching) scheduler family**:

- **EB(k̂\*)**: exclusive batching with online (k̂\*, N̂\*) calibration; the
  phase-switching threshold k̂\* and memory-safe batch size N̂\* are updated
  online from observed workload statistics.
- **EB⁺**: adaptive selector that switches online between mixed batching
  (vLLM v1's default) and EB based on a closed-form crossover criterion.

The new scheduler code lives in [`vllm/v1/core/sched/`](vllm/v1/core/sched/).
Reproduction scripts for every figure and table in the paper live in
[`reproduce/`](reproduce/).

## Installation

```bash
git clone https://github.com/weifang231/eb-vllm.git
cd eb-vllm
# Use uv or pip; same install flow as upstream vLLM
pip install -e .
```

Pinning to the camera-ready state:

```bash
git checkout icml2026-camera-ready    # tag at the paper-time snapshot
```

The `main` branch tracks subsequent fixes and improvements. For exact
paper-faithful reproduction, use the tag.

## Reproducing the paper

See [`reproduce/README.md`](reproduce/README.md) for the per-section map of
scripts and a step-by-step replication recipe.

Quickstart:

```bash
cd reproduce/synthetic_e2e
./run_grid_search_cfr.sh 4               # 4 GPUs, ~2 h for Qwen3-8B
python analyze_cfr_e2e.py outputs/e2e_grid_search/<GPU>_Qwen3-8B
python plot_synthetic_e2e.py             # produces the §4.3.1 figure
```

## Repository layout

```
eb-vllm/
├── vllm/                              # vLLM package (upstream + EB scheduler)
│   └── v1/core/sched/                 # our scheduler additions
│       ├── calibration.py             # online (k̂*, N̂*) cost-model estimation
│       ├── calibration_full_iter.py
│       ├── scheduler.py               # EB scheduler + EB+ controller
│       ├── async_scheduler.py
│       ├── interface.py / output.py / request_queue.py / utils.py
│       └── __init__.py
├── reproduce/                         # paper reproduction harness
│   ├── README.md                      # paper section → subdir mapping
│   ├── REPRODUCE.md                   # recommended run order
│   ├── common/   calibration/  docker/
│   ├── cost_model/
│   ├── hazard_rate/
│   ├── validation/
│   ├── synthetic_e2e/
│   ├── real_workloads/
│   ├── eb_plus/
│   ├── disagg/
│   ├── long_context/
│   └── scalability/
└── ...upstream vLLM files (CMakeLists, csrc/, tests/, docs/, ...)
```

## Citation

```bibtex
@inproceedings{zhang2026theta,
  title  = {Threshold-Based Exclusive Batching for Memory-Bandwidth-Constrained
            LLM Inference},
  author = {Zhang, Weifang and Nie, Yuzhou and Pang, Bowen and
            Ma, Guangrui and Wu, Shining},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year   = {2026}
}
```

## License

Apache-2.0, inherited from upstream vLLM. See [`LICENSE`](LICENSE).

## Acknowledgement

This work builds on [vLLM](https://github.com/vllm-project/vllm) by the vLLM
team. We thank the vLLM community for the foundational serving infrastructure.
