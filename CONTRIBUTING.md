# Contributing to eb-vllm

This repository is the research code companion to the paper
*Threshold-Based Exclusive Batching for Memory-Bandwidth-Constrained LLM
Inference* (ICML 2026), built on top of
[vllm-project/vllm](https://github.com/vllm-project/vllm).

## Where to send what

- **Reproduction questions or bug reports about this paper's
  experiments** — open an issue on
  [weifang231/eb-vllm](https://github.com/weifang231/eb-vllm/issues).
- **Contributions to the EB / MB scheduler** (`vllm/v1/core/sched/`) or
  to the reproduction harness (`reproduce/`) — pull requests are welcome
  on this repository. The `release/icml2026` branch tracks the
  camera-ready snapshot and is kept frozen for reproducibility; active
  development happens on a separate branch (see the README for the
  current development target).
- **Contributions to upstream vLLM** (anything outside the paths above)
  — please file them at
  [vllm-project/vllm](https://github.com/vllm-project/vllm) following
  [their contribution guide](https://docs.vllm.ai/en/latest/contributing).
  This fork does not feed back to upstream automatically.

## Reproducing the paper

See [`reproduce/README.md`](reproduce/README.md) for the full
reproduction recipe and
[`reproduce/REPRODUCTION_REPORT.md`](reproduce/REPRODUCTION_REPORT.md)
for the per-figure status of what we have verified.

## License

By contributing you agree your contributions are licensed under
[Apache-2.0](LICENSE), matching the rest of the repository.
