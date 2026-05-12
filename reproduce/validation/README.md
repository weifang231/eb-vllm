# §4.2 — Controller validation

Validates that the online (k̂\*, N̂\*) controller matches the throughput of
exhaustive fixed-k sweeps and outperforms v1 (mixed batching). Backs
Fig. `validation_grid.pdf` (3 scenarios × 3 metrics).

## Files

| File | Purpose |
|---|---|
| `run_validation_cfr.sh` | Fixed-k sweep + EB(k̂\*) + v1 on 3 synthetic CFR workloads |
| `analyze_cfr_validation.py` | Aggregate bench JSONs into `validation_summary.csv` |
| `plot_validation_grid.py` | 3×3 grid (scenario × metric); accepts the CSV or `--demo` |

## Run

```bash
MODEL=Qwen/Qwen3-8B ./run_validation_cfr.sh 8        # ~30 min on 8 GPUs
python analyze_cfr_validation.py outputs/controller_validation/<GPU>_Qwen3-8B
python plot_validation_grid.py \
    outputs/controller_validation/<GPU>_Qwen3-8B/validation_summary.csv \
    --output validation_grid.pdf
```

`SKIP_EXISTING=1` (default) lets you re-run incrementally. See
`run_validation_cfr.sh` header for env knobs (e.g. `VLLM_PD_OOM_TOLERANCE`).

## Expected output

3 rows (decode-heavy / balanced / prefill-heavy) × 3 columns (throughput,
mean TPOT, p99 TPOT). Green points = fixed-k sweep with `k*` annotated,
red horizontal = v1, blue dashed = EB(k̂\*).
