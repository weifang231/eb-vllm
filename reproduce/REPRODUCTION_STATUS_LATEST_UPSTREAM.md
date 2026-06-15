# Reproduction status (H200, Qwen3-8B)

| Paper section | Reproduced |
|---|---|
| §3 cost model — linear iteration time | ✅ prefill time linear in chunk size (R² ≈ 0.97–0.98) |
| §3 hazard rate / CFR–IFR | ✅ |
| §4.2 controller validation (Fig 3) | ✅ EB(k̂*) ≥ v1; EB ≈ best fixed-k on H200 |
| §4.3.1 synthetic e2e (Fig 4) | ✅ EB > v1 on all three workloads |
| §4.3.2/3 real workloads (Tables 2/3) | ✅ EB ≈ v1; absolute TTFT/TPOT match the paper |
| §4.4 EB⁺ traffic + non-stationary (Tables 4/5) | ✅ EB⁺ ≈ v1 |

The large EB gains reported in the paper are on the bandwidth-constrained RTX PRO 6000; on the
high-bandwidth H200 the paper itself shows EB ≈ v1, which is what we observe here.
