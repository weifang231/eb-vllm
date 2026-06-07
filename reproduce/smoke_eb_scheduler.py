#!/usr/bin/env python
"""Minimal end-to-end smoke test for the EB / EB+ (P/D competition) scheduler.

Boots a tiny model under the EB scheduler and runs a batch of generations,
exercising schedule_pd()'s prefill->decode phase machine. Used to validate the
port of the EB scheduler onto upstream vLLM (does NOT check paper numbers).

Run (single process; see NOTEs below):

    CUDA_VISIBLE_DEVICES=6 VLLM_USE_FLASHINFER_SAMPLER=0 \
        taskset -c 0-15 python reproduce/smoke_eb_scheduler.py

NOTEs / environment gotchas discovered while bringing this up:
  * VLLM_USE_PD_SCHEDULER=1 (or VLLM_PD_SCHEDULER_MODE=eb/auto) selects the EB
    scheduler; VLLM_PD_K_MODE=ifr|cfr|ratio|direct picks the k* controller.
  * The EB scheduler requires hardware timing params; provide a calibration file
    via VLLM_PD_CALIBRATION_FILE, or set VLLM_PD_ALPHA_P/BETA_P/ALPHA_D/BETA_D.
  * `taskset -c 0-15`: on big NUMA boxes (192 cores here) torch's worker CPU
    affinity setup could hang in the kernel (sched_setaffinity); pinning to a
    small core set avoids it. Not EB-specific.
  * VLLM_USE_FLASHINFER_SAMPLER=0 avoids a runtime flashinfer JIT build (needs
    `ninja` on PATH); also not EB-specific.
"""

import os

os.environ.setdefault("VLLM_USE_PD_SCHEDULER", "1")
os.environ.setdefault("VLLM_PD_K_MODE", "ifr")
# Dummy-but-positive hardware timing params so __init__ doesn't require a
# calibration file. Replace with a real calibration for meaningful scheduling.
os.environ.setdefault("VLLM_PD_ALPHA_P", "1e-4")
os.environ.setdefault("VLLM_PD_BETA_P", "1e-6")
os.environ.setdefault("VLLM_PD_ALPHA_D", "1e-4")
os.environ.setdefault("VLLM_PD_BETA_D", "1e-6")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("EB_SMOKE_MODEL", "facebook/opt-125m")


def main() -> None:
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.25,
        max_model_len=1024,
        enforce_eager=True,
        disable_log_stats=True,
    )
    prompts = [
        "The capital of France is",
        "In a galaxy far far away,",
        "The meaning of life is",
        "Once upon a time",
    ] * 4  # 16 requests
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64))

    print("=== EB SMOKE RESULTS ===")
    for o in outs[:3]:
        print(repr(o.prompt), "->", repr(o.outputs[0].text[:60]))
    ok = len(outs) == len(prompts) and all(o.outputs[0].text for o in outs)
    print(f"num outputs: {len(outs)} | all have text: {ok}")
    assert ok, "EB smoke test failed: missing outputs"
    print("EB_SMOKE_OK")


if __name__ == "__main__":
    main()
