#!/usr/bin/env python3
"""Analyse CFR online-controller validation results.

Reads the *_stats.json files written by run_validation.sh, plus the
benchmark JSONs, and reports per-workload:
  (i)   estimation accuracy of p̂_0 vs ground-truth p_0 = 1/E[O];
        estimation accuracy of μ̂_L vs ground-truth E[L];
  (ii)  realised TP vs the fluid-optimal TP at the controller's
        (k̂, N̂) — the closed-form throughput model
        TP_fluid = N·θ_0 / (α_d τ_R + β_d N θ_0/p_0 + α_p + β_p N θ_0 μ_L);
  (iii) OOM event rate vs the prescribed ε.

Outputs:
  validation_summary.csv         (one row per workload + scheduler)
  validation_table.tex           (LaTeX table for the paper)
  per-workload trace plots in plots/<scenario>/...
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


GROUND_TRUTH = {
    "decode_heavy":  {"input_len": 128,  "output_len": 1024},
    "balanced":      {"input_len": 512,  "output_len": 512},
    "prefill_heavy": {"input_len": 1024, "output_len": 128},
}


def fluid_tp(theta: float, p_0: float, n: int, mu_L: float,
              alpha_p: float, beta_p: float,
              alpha_d: float, beta_d: float) -> float:
    """Closed-form fluid throughput TP(θ, N) (paper Eq. 9)."""
    if theta <= 0 or theta >= 1 or p_0 <= 0 or n <= 0:
        return 0.0
    tau = math.log(1 - theta) / math.log(1 - p_0)
    decode_time = alpha_d * tau + beta_d * n * theta / p_0
    prefill_time = alpha_p + beta_p * n * theta * mu_L
    served_tokens = n * theta * (mu_L + 1.0 / p_0)
    return served_tokens / max(decode_time + prefill_time, 1e-9)


def load_run(workload_dir: Path, sched: str) -> dict | None:
    bench = workload_dir / f"bench_{sched}.json"
    stats = workload_dir / f"{sched}_stats.json"
    if not bench.exists():
        return None
    with open(bench) as f:
        b = json.load(f)
    s = {}
    if stats.exists():
        with open(stats) as f:
            s = json.load(f)
    return {"bench": b, "stats": s, "bench_path": bench, "stats_path": stats}


STEADY_STATE_LO = 0.2  # drop first 20% by time
STEADY_STATE_HI = 0.8  # drop last 20% by time


def _steady_state_window(steps: list[dict]) -> tuple[float, float]:
    """Return (t_lo, t_hi) defining the steady-state middle 60% by time."""
    if not steps:
        return (0.0, 0.0)
    t0 = steps[0].get("timestamp", 0.0)
    t_end = steps[-1].get("timestamp", 0.0)
    span = t_end - t0
    if span <= 0:
        return (t0, t_end)
    return (t0 + STEADY_STATE_LO * span, t0 + STEADY_STATE_HI * span)


def per_cycle_oom_rate(stats_blob: dict, steady_only: bool = True
                       ) -> tuple[int, int, float]:
    """Compute per-cycle OOM rate from the per-step `stats` array.

    A cycle = one decode span (phase == 1) bounded by transitioning out of
    phase 1 (to phase 0/2). A cycle is counted as OOM if any preemption
    fires during the decode span. Matches Prop. memory event
    Pr(X_max > C) > 0 over the cycle.

    With `steady_only=True`, only cycles whose decode span ENDS inside the
    steady-state middle 60% time window are counted.

    Returns (n_cycles, n_oom_cycles, oom_rate).
    """
    steps = stats_blob.get("stats") or []
    if not steps:
        return 0, 0, float("nan")
    t_lo, t_hi = _steady_state_window(steps) if steady_only else (
        float("-inf"), float("inf"))
    cycles = 0
    oom_cycles = 0
    cur_preempt = 0
    in_decode = False
    for s in steps:
        ph = s.get("phase", -1)
        npre = s.get("num_preempted_reqs", 0) or 0
        ts = s.get("timestamp", 0.0)
        if ph == 1:
            in_decode = True
            cur_preempt += npre
        elif in_decode:
            if t_lo <= ts <= t_hi:
                cycles += 1
                if cur_preempt > 0:
                    oom_cycles += 1
            cur_preempt = 0
            in_decode = False
    if in_decode:
        ts_end = steps[-1].get("timestamp", 0.0)
        if t_lo <= ts_end <= t_hi:
            cycles += 1
            if cur_preempt > 0:
                oom_cycles += 1
    rate = oom_cycles / cycles if cycles > 0 else float("nan")
    return cycles, oom_cycles, rate


def steady_state_throughput(stats_blob: dict) -> tuple[float, float]:
    """Compute steady-state token throughput from per-step stats.

    Sums (prefill_tokens + decode_tokens) over steps falling in the middle
    60% time window, divided by the window length. Returns (tp, window_s).
    """
    steps = stats_blob.get("stats") or []
    if not steps:
        return float("nan"), 0.0
    t_lo, t_hi = _steady_state_window(steps)
    window = t_hi - t_lo
    if window <= 0:
        return float("nan"), 0.0
    tokens = 0
    for s in steps:
        ts = s.get("timestamp", 0.0)
        if t_lo <= ts <= t_hi:
            tokens += int(s.get("total_tokens", 0) or 0)
    return tokens / window, window


def discover_schedulers(workload_dir: Path) -> list[str]:
    """Return scheduler names in a stable order: v1, eb, sweep cells."""
    names = {p.stem.removeprefix("bench_") for p in workload_dir.glob("bench_*.json")}
    ordered: list[str] = []
    for primary in ("v1", "eb"):
        if primary in names:
            ordered.append(primary)
    fixed = sorted(
        (n for n in names if n.startswith("eb_fixed_k_")),
        key=lambda n: int(n.removeprefix("eb_fixed_k_")),
    )
    ordered.extend(fixed)
    kratio = sorted(
        (n for n in names if n.startswith("eb_kratio_")),
        key=lambda n: float(n.removeprefix("eb_kratio_")),
    )
    ordered.extend(kratio)
    return ordered


def summarise(workload_dir: Path, scen: str, cfg: dict) -> list[dict]:
    rows = []
    for sched in discover_schedulers(workload_dir):
        run = load_run(workload_dir, sched)
        if run is None:
            continue
        b = run["bench"]
        s = run["stats"]
        hist = s.get("update_history") or s.get("cfr_update_history") or []
        last = hist[-1] if hist else {}
        pd_cfg = s.get("pd_config", {})
        is_fixed_k = sched.startswith("eb_fixed_k_")
        is_kratio = sched.startswith("eb_kratio_")
        table1_cf = cfg.get("table1_closed_form", {}).get(scen, {})

        # Ground truth
        gt_o = GROUND_TRUTH[scen]["output_len"]
        gt_l = GROUND_TRUTH[scen]["input_len"]
        gt_p_0 = 1.0 / gt_o

        # Online estimates (NaN for sweep cells — controller is disabled)
        out_lens = b.get("output_lens") or []
        in_lens = b.get("input_lens") or []
        realised_o = float(np.mean(out_lens)) if out_lens else 0.0
        realised_l = float(np.mean(in_lens)) if in_lens else 0.0
        bs_pinned = int(pd_cfg.get("max_num_seqs", 0))
        if is_fixed_k:
            p_hat = float("nan")
            mu_l_hat = float("nan")
            theta_0 = float("nan")
            k_hat = int(sched.removeprefix("eb_fixed_k_"))
        elif is_kratio:
            # TABLE1 sweep cell: k pinned via ratio, no estimator updates.
            p_hat = float("nan")
            mu_l_hat = float("nan")
            theta_0 = float("nan")
            ratio = float(sched.removeprefix("eb_kratio_"))
            k_hat = max(1, int(round(ratio * bs_pinned)))
        else:
            p_hat = last.get("p_0_estimate", 0.0)
            mu_l_hat = last.get("mu_L_estimate", 0.0)
            theta_0 = last.get("theta_0", 0.0)
            k_hat = int(last.get("k_hat_int", pd_cfg.get("final_k_star", 0)))
            # In TABLE1 mode the EB row is also run as eb_kratio internally
            # (no cfr update). Recover closed-form θ_0 / k̂ from config.
            if sched == "eb" and table1_cf and (not theta_0 or math.isnan(theta_0)):
                theta_0 = float(table1_cf.get("theta_0", 0.0))
                k_hat = int(table1_cf.get("k_hat", 0)) or k_hat

        # Throughput attainment (only meaningful for the adaptive controller)
        n_hat = int(last.get("N_hat", bs_pinned))
        if sched == "eb" and table1_cf:
            n_hat = int(table1_cf.get("N_hat", n_hat))
        cal = cfg.get("calibration_params", {})
        if is_fixed_k or is_kratio or not isinstance(theta_0, (int, float)) or math.isnan(theta_0):
            tp_fluid = float("nan")
            attainment = float("nan")
        else:
            tp_fluid = fluid_tp(
                theta_0, gt_p_0, n_hat, gt_l,
                float(cal.get("alpha_p", 0.0)),
                float(cal.get("beta_p", 0.0)),
                float(cal.get("alpha_d", 0.0)),
                float(cal.get("beta_d", 0.0)),
            )
            attainment = tp_real_attain(b, tp_fluid)
        tp_real = b.get("total_token_throughput", 0.0)
        mean_tpot_ms = b.get("mean_tpot_ms", float("nan"))
        # vLLM bench reports percentiles via separate top-level fields when
        # --percentile-metrics tpot is set (default); newer versions also
        # expose percentiles_tpot_ms as a list[(percentile, ms)].
        p99_tpot_ms = b.get("p99_tpot_ms")
        if p99_tpot_ms is None:
            for pct, val in b.get("percentiles_tpot_ms") or []:
                if abs(pct - 99) < 1e-6:
                    p99_tpot_ms = val
                    break
        if p99_tpot_ms is None:
            p99_tpot_ms = float("nan")

        # OOM rate (per-request, legacy) + per-cycle steady-state (paper-faithful)
        completed = b.get("completed", 0)
        oom = pd_cfg.get("total_oom_events", 0)
        oom_rate = oom / max(completed, 1)
        eps = float(cfg.get("oom_tolerance", 0.01))
        n_cycles, n_oom_cycles, oom_rate_cycle = per_cycle_oom_rate(s, steady_only=True)
        tp_steady, steady_window_s = steady_state_throughput(s)

        rows.append({
            "scenario": scen,
            "scheduler": sched,
            "ground_truth_E_L": gt_l,
            "ground_truth_E_O": gt_o,
            "ground_truth_p_0": gt_p_0,
            "realised_E_L": realised_l,
            "realised_E_O": realised_o,
            "p_hat_final": p_hat,
            "mu_L_hat_final": mu_l_hat,
            "p_hat_relerr_pct": (abs(p_hat - gt_p_0) / gt_p_0 * 100)
                if (gt_p_0 > 0 and not math.isnan(p_hat)) else float("nan"),
            "mu_L_hat_relerr_pct": (abs(mu_l_hat - gt_l) / gt_l * 100)
                if (gt_l > 0 and not math.isnan(mu_l_hat)) else float("nan"),
            "theta_0_final": theta_0,
            "k_hat_final": k_hat,
            "N_hat_final": n_hat,
            "tp_real": tp_real,
            "mean_tpot_ms": mean_tpot_ms,
            "p99_tpot_ms": p99_tpot_ms,
            "tp_fluid": tp_fluid,
            "attainment_pct": attainment * 100 if not math.isnan(attainment) else float("nan"),
            "oom_events": oom,
            "completed": completed,
            "oom_rate": oom_rate,
            "oom_rate_pct": oom_rate * 100,
            "n_cycles_steady": n_cycles,
            "n_oom_cycles_steady": n_oom_cycles,
            "oom_rate_cycle_steady": oom_rate_cycle,
            "oom_rate_cycle_steady_pct": (oom_rate_cycle * 100)
                if not math.isnan(oom_rate_cycle) else float("nan"),
            "tp_steady": tp_steady,
            "steady_window_s": steady_window_s,
            "eps_target": eps,
            "n_updates": len(hist),
        })
    return rows


def _annotate_sweep_optimum(rows: list[dict]) -> None:
    """Per scenario, find theta*_sweep = argmax_k TP_steady among sweep cells
    and compute TP ratio = TP_steady(EB at theta_0) / TP_steady(sweep argmax).

    Uses steady-state throughput (middle-60% window) so the comparison
    isn't polluted by cold-start. The TP ratio is only paper-faithful
    when the sweep's N matches the EB controller's N_hat (caption:
    "k-sweep at fixed N_hat") — TABLE1 mode arranges this; otherwise
    tp_ratio_to_sweep is NaN.

    Sweep cells include both eb_fixed_k_* (legacy integer-k) and
    eb_kratio_* (TABLE1 per-workload ratio sweep).
    """
    by_scen: dict[str, list[dict]] = {}
    for r in rows:
        by_scen.setdefault(r["scenario"], []).append(r)
    for scen, scen_rows in by_scen.items():
        sweep = [
            r for r in scen_rows
            if r["scheduler"].startswith("eb_fixed_k_")
            or r["scheduler"].startswith("eb_kratio_")
        ]
        eb_row = next((r for r in scen_rows if r["scheduler"] == "eb"), None)
        if not sweep or eb_row is None:
            continue
        # Filter to cells with a valid steady-state TP measurement.
        sweep_valid = [r for r in sweep if r["tp_steady"] > 0
                       and not math.isnan(r["tp_steady"])]
        if not sweep_valid:
            continue
        best = max(sweep_valid, key=lambda r: r["tp_steady"])
        sweep_n = int(best["N_hat_final"])  # all sweep cells share the same N
        # Recover k_sweep for both naming conventions.
        sname = best["scheduler"]
        if sname.startswith("eb_kratio_"):
            theta_sweep = float(sname.removeprefix("eb_kratio_"))
            k_sweep = int(round(theta_sweep * sweep_n))
        else:
            k_sweep = int(sname.removeprefix("eb_fixed_k_"))
            theta_sweep = k_sweep / sweep_n if sweep_n > 0 else float("nan")
        eb_n_hat = int(eb_row["N_hat_final"])
        tp_ratio = (
            eb_row["tp_steady"] / best["tp_steady"]
            if best["tp_steady"] > 0 and sweep_n == eb_n_hat
               and not math.isnan(eb_row["tp_steady"])
            else float("nan")
        )
        eb_row["theta_sweep_star"] = theta_sweep
        eb_row["k_sweep_star"] = k_sweep
        eb_row["sweep_N"] = sweep_n
        eb_row["tp_at_sweep_star"] = best["tp_steady"]
        eb_row["tp_ratio_to_sweep"] = tp_ratio


def tp_real_attain(b: dict, tp_fluid: float) -> float:
    tp_real = b.get("total_token_throughput", 0.0)
    return tp_real / tp_fluid if tp_fluid > 0 else 0.0


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(empty)\n")
        return
    # Use union of keys (EB rows carry extra sweep-annotation columns).
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen_set:
                seen.append(k)
                seen_set.add(k)
    with open(path, "w") as f:
        f.write(",".join(seen) + "\n")
        for r in rows:
            f.write(",".join(
                repr(r[k]) if isinstance(r.get(k), str) else f"{r.get(k, '')}"
                for k in seen
            ) + "\n")


def write_latex(rows: list[dict], path: Path, eps: float) -> None:
    """Render the paper Table 1 row schema:
    $\\hat\\theta_0$, $\\theta^*_{\\text{sweep}}$, $\\hat N$, TP ratio, OOM
    (per cycle). Sweep columns are NaN if the k-sweep wasn't run at the
    controller's $\\hat N$ (run_validation.sh's TABLE1 mode lines them up).
    """
    eb_rows = [r for r in rows if r["scheduler"] == "eb"]
    eb_rows.sort(key=lambda r: ("decode_heavy", "balanced", "prefill_heavy").index(r["scenario"]))
    def _fmt(v: float, spec: str, na: str = "TBD") -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return na
        return format(v, spec)
    with open(path, "w") as f:
        f.write("% Auto-generated by analyze_validation.py\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Workload & $\\hat\\theta_0$ & $\\theta^*_{\\text{sweep}}$ "
                "& $\\hat N$ & TP ratio & OOM \\\\\n")
        f.write("\\midrule\n")
        for r in eb_rows:
            tp_ratio = r.get("tp_ratio_to_sweep", float("nan"))
            theta_sweep = r.get("theta_sweep_star", float("nan"))
            tp_ratio_pct = (tp_ratio * 100) if isinstance(tp_ratio, (int, float)) and not math.isnan(tp_ratio) else float("nan")
            oom_cycle_pct = r.get("oom_rate_cycle_steady_pct", float("nan"))
            f.write(f"{r['scenario'].replace('_', '-')} & "
                    f"{_fmt(r['theta_0_final'], '.3f')} & "
                    f"{_fmt(theta_sweep, '.3f')} & "
                    f"{r['N_hat_final']} & "
                    f"{_fmt(tp_ratio_pct, '.1f')}\\% & "
                    f"{_fmt(oom_cycle_pct, '.1f')}\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"% Prescribed OOM tolerance ε = {eps} (per-cycle definition: "
                f"fraction of decode cycles with any preemption).\n")


def maybe_plot_traces(workload_dir: Path, scen: str, cfg: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    run = load_run(workload_dir, "eb")
    if run is None:
        return
    hist = (
        run["stats"].get("update_history")
        or run["stats"].get("cfr_update_history")
        or []
    )
    if not hist:
        return
    times = [h["timestamp"] for h in hist]
    p_hat = [h["p_0_estimate"] for h in hist]
    mu_l_hat = [h["mu_L_estimate"] for h in hist]
    k_hat = [h["k_hat_int"] for h in hist]
    n_hat = [h["N_hat"] for h in hist]
    gt = GROUND_TRUTH[scen]
    plot_dir = workload_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(times, p_hat, label=r"$\hat p_0$")
    axes[0, 0].axhline(1 / gt["output_len"], color="r", linestyle="--",
                        label=r"truth $1/\mu_O$")
    axes[0, 0].set_title("p_0 estimate"); axes[0, 0].legend()
    axes[0, 1].plot(times, mu_l_hat, label=r"$\hat\mu_L$")
    axes[0, 1].axhline(gt["input_len"], color="r", linestyle="--",
                        label=r"truth $\mu_L$")
    axes[0, 1].set_title("mu_L estimate"); axes[0, 1].legend()
    axes[1, 0].plot(times, k_hat, color="C2")
    axes[1, 0].set_title(r"$\hat k$ over time")
    axes[1, 1].plot(times, n_hat, color="C3")
    axes[1, 1].set_title(r"$\hat N$ over time")
    for ax in axes.flat:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{scen}: online estimator trace")
    fig.tight_layout()
    out = plot_dir / "estimator_trace.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiment_dir", type=Path)
    args = p.parse_args()
    root = args.experiment_dir
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    cfg = json.loads((root / "experiment_config.json").read_text())
    eps = float(cfg.get("oom_tolerance", 0.01))

    rows: list[dict] = []
    for scen in ("decode_heavy", "balanced", "prefill_heavy"):
        sub = sorted(root.glob(f"{scen}_in*_out*"))
        if not sub:
            continue
        wd = sub[0]
        rows.extend(summarise(wd, scen, cfg))
        maybe_plot_traces(wd, scen, cfg)

    _annotate_sweep_optimum(rows)
    write_csv(rows, root / "validation_summary.csv")
    write_latex(rows, root / "validation_table.tex", eps)
    print(f"\nWrote {root / 'validation_summary.csv'}")
    print(f"Wrote {root / 'validation_table.tex'}")
    print()
    if rows:
        print(f"  {'workload':14s} {'sched':8s} "
              f"{'p̂_0':>10s} {'gt':>10s} "
              f"{'μ̂_L':>7s} {'gt':>5s} "
              f"{'TP':>9s} {'TP_flu':>9s} {'attain':>7s} "
              f"{'OOM%':>7s} {'ε':>6s}")
        for r in rows:
            print(f"  {r['scenario']:14s} {r['scheduler']:8s} "
                  f"{r['p_hat_final']:>10.5f} "
                  f"{r['ground_truth_p_0']:>10.5f} "
                  f"{r['mu_L_hat_final']:>7.0f} "
                  f"{r['ground_truth_E_L']:>5d} "
                  f"{r['tp_real']:>9.0f} "
                  f"{r['tp_fluid']:>9.0f} "
                  f"{r['attainment_pct']:>6.1f}% "
                  f"{r['oom_rate_pct']:>6.2f}% "
                  f"{r['eps_target']*100:>5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
