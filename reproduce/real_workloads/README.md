# §4.3.2-3 — Real-world workloads and latency

`(B, N)` grid search of v0 / v1 / EB(k̂\*) on ShareGPT, LongBench, WildChat,
NuminaMath across Qwen3-8B / Qwen3-30B-A3B × RTX PRO 6000 / H200. Produces
the §4.3.2 throughput Tables 2-3 and §4.3.3 latency Figs. `ttft.pdf`, `tpot.pdf`.

## Files

| File | Purpose |
|---|---|
| `run_grid_search.sh`, `run_grid_search_v0.sh` | Grid search over all 4 workloads × selected schedulers |
| `analyze_grid_search.py` | Aggregate to `optimal_per_scheduler.json` (per workload, per scheduler) |
| `plot_real_workload_latency.py` | Bar charts for TTFT and TPOT (accepts the JSON or `--demo`) |
| `multiturn/` | WildChat multi-turn dialogue preprocessing |

## Run

```bash
# Per (GPU, model). Typically 6–10 h per combination.
MODEL=Qwen/Qwen3-8B    ./run_grid_search.sh 4
python analyze_grid_search.py outputs/<GPU>_Qwen3-8B/
python plot_real_workload_latency.py \
    --optimal-json outputs/<GPU>_Qwen3-8B/optimal_per_scheduler.json \
    --gpu <GPU> --model Qwen3-8B \
    --ttft-output ttft.pdf --tpot-output tpot.pdf
```

For WildChat preprocessing (one-time), see `multiturn/README.md`.

## Tables 2 / 3 — throughput

`analyze_grid_search.py` also writes a LaTeX table (`optimal_table.tex`)
matching the format of Tables 2 (RTX PRO 6000) and 3 (H200) in the paper.
