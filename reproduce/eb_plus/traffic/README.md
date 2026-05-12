# §4.4 — EB⁺ traffic-level sensitivity (Table 4)

Sweeps concurrency `c ∈ {32, 512, 2048}` and compares v1 / EB(k̂\*) / EB⁺.

## Files

| File | Purpose |
|---|---|
| `run_adaptive_selector_cfr.sh` | Bench all 3 schedulers at each `c` |
| `analyze_cfr_selector.py` | Aggregate to `selector_summary.csv` + LaTeX table |

## Run

```bash
MODEL=Qwen/Qwen3-8B ./run_adaptive_selector_cfr.sh 8     # ~30 min
python analyze_cfr_selector.py outputs/adaptive_selector/<GPU>_Qwen3-8B
```

For a more accurate Δ(N) diagnostic, set the kernel-sweep env overrides
documented in `run_adaptive_selector_cfr.sh`'s header
(`VLLM_PD_BETA_MB_E`, `VLLM_PD_CP_COST_{A,B,C}`).
