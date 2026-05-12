"""
Approximate plot of §4.3.1 fig_synthetic_e2e.pdf.

NOTE: The paper figure was drawn from `outputs/.../optimal_per_scheduler.csv`
(produced by analyze_cfr_e2e.py), comparing three schedulers v0 / v1 / EB(k̂*) across
three workloads (decode-heavy / balanced / prefill-heavy) on two GPUs.

This script uses the older but committed grid-search data in
`experiments/serve/online_khat/grid_search_20260111_121930/`, which compares
"baseline" (= v1) vs "pd" (≈ EB) across the same three input/output regimes.
The labels and exact metric layout differ slightly from the paper figure, but
the shape (3 workloads × 2 metrics, bars per scheduler) is preserved.

When the new syn_cfr data is generated, swap the data-loader for
`outputs/e2e_grid_search/<GPU>_<MODEL>/optimal_per_scheduler.csv`
(columns: scheduler, scenario, throughput, ttft_ms, tpot_ms, ...).

Input:
  --grid-search-dir  experiments/serve/online_khat/grid_search_20260111_121930

Output:
  fig_synthetic_e2e.pdf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = [
    ("in128_out1024", "Decode-heavy\n(128/1024)"),
    ("in512_out512", "Balanced\n(512/512)"),
    ("in1024_out128", "Prefill-heavy\n(1024/128)"),
]


def best_per_scheduler(grid_dir: Path, scenario: str) -> dict:
    """Find best-throughput (B, N) for each scheduler in this scenario."""
    best = {}
    for tb_dir in grid_dir.glob("tb*"):
        for bs_dir in tb_dir.glob("bs*"):
            scen_dir = bs_dir / scenario
            if not scen_dir.exists():
                continue
            for fname, sched in [
                ("bench_baseline.json", "baseline (v1)"),
                ("bench_pd.json", "pd (EB)"),
            ]:
                f = scen_dir / fname
                if not f.exists():
                    continue
                with open(f) as fp:
                    d = json.load(fp)
                tput = d.get("request_throughput") or 0
                if tput > best.get(sched, {}).get("request_throughput", 0):
                    best[sched] = d
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid-search-dir",
        default="outputs/e2e_grid_search",
        help="Grid search output dir from analyze_cfr_e2e.py "
             "(typically outputs/e2e_grid_search/<GPU>_<MODEL>/)",
    )
    p.add_argument("--output", default="fig_synthetic_e2e.pdf")
    args = p.parse_args()

    grid_dir = Path(args.grid_search_dir)
    if not grid_dir.exists():
        raise SystemExit(
            f"Grid dir not found: {grid_dir}\n"
            "Generate it first with:\n"
            "  ./run_grid_search_cfr.sh\n"
            "  python analyze_cfr_e2e.py outputs/e2e_grid_search/<GPU>_<MODEL>"
        )

    # For each scenario, find best throughput per scheduler
    rows = []
    for key, label in SCENARIOS:
        best = best_per_scheduler(grid_dir, key)
        rows.append((label, best))

    metrics = [
        ("request_throughput", "Throughput (RPS)", False),
        ("mean_ttft_ms", "Mean TTFT (ms)", False),
        ("mean_tpot_ms", "Mean TPOT (ms)", False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    schedulers = ["baseline (v1)", "pd (EB)"]
    colors = {"baseline (v1)": "tab:red", "pd (EB)": "tab:blue"}
    x = np.arange(len(SCENARIOS))
    width = 0.35

    for ax, (mkey, mlabel, _) in zip(axes, metrics):
        for i, sched in enumerate(schedulers):
            vals = []
            for _, best in rows:
                vals.append(best.get(sched, {}).get(mkey, np.nan))
            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=sched,
                color=colors[sched],
                edgecolor="black",
                linewidth=0.5,
            )
            # Annotate best with a star
        best_idx = [
            int(np.nanargmin([rows[j][1].get(s, {}).get(mkey, np.inf) for s in schedulers]))
            if "ttft" in mkey or "tpot" in mkey
            else int(np.nanargmax([rows[j][1].get(s, {}).get(mkey, -np.inf) for s in schedulers]))
            for j in range(len(rows))
        ]
        for j, bi in enumerate(best_idx):
            sched = schedulers[bi]
            val = rows[j][1].get(sched, {}).get(mkey)
            if val is not None:
                offset = (bi - 0.5) * width
                ax.text(j + offset, val * 1.02, "★", ha="center", va="bottom", fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _ in rows])
        ax.set_ylabel(mlabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Synthetic workloads, H200 (older online_khat data — approximation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
