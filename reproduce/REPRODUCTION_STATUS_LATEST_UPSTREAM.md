# Reproduction status — EB/EB⁺ port on latest upstream vLLM (H200)

Branch `port-eb-onto-upstream` = `vllm-project/vllm@4dcd10eb0` + EB/EB⁺ port.
8×H200, Qwen3-8B. Reproduced 2026-06-08.

## Box-specific run infra (NOT EB issues)

A vLLM wrapper is installed as `.venv/bin/vllm` (real entry point → `vllm-real`). On this
192-core host, concurrent vLLM processes deadlock the kernel in `sched_setaffinity`
(OpenMP thread pinning). The wrapper fixes it for every vllm process:
`OMP_PROC_BIND=false`, `OMP_NUM_THREADS=8`, `KMP_AFFINITY=disabled`, and taskset-pins
`serve` to a per-GPU 24-core slice / `bench` to cores 0-15. Verified GPU-bound
(32-core == 64-core throughput), so pinning does **not** throttle.

## Results vs paper (H200, Qwen3-8B)

| Section | Status | Result |
|---|---|---|
| §3 CFR/IFR figure | ✅ exact | deterministic formula plot |
| §4.2 validation (Fig 3) | ✅ ran / ⚠️ margin | EB(k̂*) ties best-k (+0.1/−1.6/−0.4%); EB > v1. Paper's +8/+3.6/+0.6% margin does NOT reproduce — see diagnostic below. |
| §4.3.1 synthetic e2e (Fig 4) | ✅ | 180/180, EB > v1 all 3 workloads (+2.4/+4.7/+2.5%). H200 margins small, as paper. |
| §4.3.2/3 real workloads (Tables 2/3) | ✅ | sharegpt/longbench/numina/wildchat (v1+eb). EB ≈ v1 — matches the paper's **H200** column (not RTX-6000). longbench abs nearly exact (15.73 vs 15.77). |
| §4.4 EB⁺ (Tables 4/5) | ✅ | traffic sweep c∈{32,512,2048} + non-stationary (dist/conc shift). Needs `VLLM_PD_MB_COST_A/B/C` exported from the beta_mb json (not auto-loaded). EB⁺ ≈ v1 on H200. |
| §3 cost model (linear) | ✅ runs | fixed `EngineCoreRequest(eos_token_id=)` drift; multi-model R² aggregator needs all 7 models. |

The only metric that does not reproduce is the §4.2 EB-over-best-k **margin**. Everything
else reproduces in direction and (mostly) magnitude. The large EB wins (+41.9%) are on the
bandwidth-constrained RTX PRO 6000, which this box does not have — on high-bandwidth H200 the
paper itself shows EB ≈ v1.

## §4.2 diagnostic — is the missing margin a port bug?  NO.

We rebuilt the **original `main`** (58-commit fork + old vllm base 5d64fd8db = PR #29439,
the paper-time validated code) from source and ran the same §4.2 on this H200:

| scenario | old-main EB-vs-best-k | port EB-vs-best-k | paper | old-main eb | port eb | paper eb |
|---|---:|---:|---:|---:|---:|---:|
| decode_heavy | −2.2% | +0.1% | +8.0% | 11,941 | 13,385 | 15,725 |
| balanced | −3.0% | −1.6% | +3.6% | 18,220 | 20,290 | 20,937 |
| prefill_heavy | −7.7% | −0.4% | +0.6% | 30,115 | 34,019 | 32,863 |

The original code **also fails to reproduce the +8% margin on this box** (it is in fact worse
than the port, with lower absolute throughput). Therefore the missing §4.2 margin is an
**environment / measurement-setup difference** between this H200 machine and the paper authors'
H200 setup — **not** a regression from the upstream port. The port is healthy and lands closer
to the paper's absolute numbers than the original code does here.
