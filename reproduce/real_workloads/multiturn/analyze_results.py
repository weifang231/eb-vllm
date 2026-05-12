#!/usr/bin/env python3
"""
分析 multi-turn benchmark 的 TB × BS 网格搜索实验结果

用法:
    python pd_exp/multiturn/analyze_results.py <experiment_dir>

输出:
    - grid_summary.json: 完整数据汇总
    - heatmap_{metric}.png: 热力图
    - optimal_comparison.png: 最优配置对比图
    - analysis_report.txt: 文本分析报告
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，跳过绑图")


def load_bench_result(filepath: Path) -> Optional[Dict]:
    """加载 benchmark 结果文件

    对于大文件（10MB+），只读取前 100KB 和后 100KB 来提取关键指标，
    避免加载整个文件到内存。

    注意：vLLM benchmark JSON 文件结构为：
    - 开头包含：吞吐量指标、completed、failed 等
    - 结尾包含：TTFT、TPOT、ITL 等延迟指标
    """
    if not filepath.exists():
        return None
    try:
        file_size = filepath.stat().st_size

        # 小文件直接加载
        if file_size < 10 * 1024 * 1024:  # < 10MB
            with open(filepath) as f:
                return json.load(f)

        # 大文件：读取开头和结尾
        import re
        with open(filepath, 'r') as f:
            header = f.read(100 * 1024)  # 读取前 100KB
            # 读取后 100KB（延迟指标在文件末尾）
            f.seek(max(0, file_size - 100 * 1024))
            footer = f.read()

        # 合并开头和结尾的内容用于搜索
        combined = header + footer

        result = {}

        # 需要提取的字段
        # 吞吐量等在开头，延迟指标在结尾
        fields = [
            'request_throughput', 'output_throughput', 'total_token_throughput',
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
            'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
            'mean_itl_ms', 'median_itl_ms', 'p99_itl_ms',
            'mean_e2e_latency_ms', 'median_e2e_latency_ms', 'p99_e2e_latency_ms',
            'completed', 'failed', 'num_prompts'
        ]

        for field in fields:
            # 匹配 "field": value 模式
            pattern = rf'"{field}":\s*([^,\}}\]]+)'
            match = re.search(pattern, combined)
            if match:
                value_str = match.group(1).strip()
                try:
                    result[field] = json.loads(value_str)
                except json.JSONDecodeError:
                    pass

        return result if result else None

    except Exception:
        return None


def extract_metrics(bench_result: Dict) -> Dict[str, float]:
    """从 benchmark 结果提取关键指标"""
    return {
        "throughput": bench_result.get("request_throughput", 0),
        "output_throughput": bench_result.get("output_throughput", 0),
        "mean_ttft_ms": bench_result.get("mean_ttft_ms", 0),
        "median_ttft_ms": bench_result.get("median_ttft_ms", 0),
        "p99_ttft_ms": bench_result.get("p99_ttft_ms", 0),
        "mean_tpot_ms": bench_result.get("mean_tpot_ms", 0),
        "median_tpot_ms": bench_result.get("median_tpot_ms", 0),
        "p99_tpot_ms": bench_result.get("p99_tpot_ms", 0),
        "mean_itl_ms": bench_result.get("mean_itl_ms", 0),
        "median_itl_ms": bench_result.get("median_itl_ms", 0),
        "p99_itl_ms": bench_result.get("p99_itl_ms", 0),
        "mean_e2e_latency_ms": bench_result.get("mean_e2e_latency_ms", 0),
        "median_e2e_latency_ms": bench_result.get("median_e2e_latency_ms", 0),
        "p99_e2e_latency_ms": bench_result.get("p99_e2e_latency_ms", 0),
    }


def collect_grid_results(exp_dir: Path) -> Dict:
    """收集网格搜索结果

    Returns:
        {
            "tb_values": [...],
            "bs_values": [...],
            "results": {
                (tb, bs): {"baseline": metrics, "pd_ratio": metrics, ...}
            }
        }
    """
    tb_values = set()
    bs_values = set()
    results = {}

    # 遍历目录结构: tb{TB}/bs{BS}/
    for tb_dir in sorted(exp_dir.iterdir()):
        if not tb_dir.is_dir() or not tb_dir.name.startswith("tb"):
            continue

        try:
            tb = int(tb_dir.name[2:])
        except ValueError:
            continue
        tb_values.add(tb)

        for bs_dir in sorted(tb_dir.iterdir()):
            if not bs_dir.is_dir() or not bs_dir.name.startswith("bs"):
                continue

            try:
                bs = int(bs_dir.name[2:])
            except ValueError:
                continue
            bs_values.add(bs)

            key = (tb, bs)
            if key not in results:
                results[key] = {}

            # 动态检测所有 bench_*.json 文件
            for bench_file in bs_dir.glob("bench_*.json"):
                # 从文件名提取调度器名称: bench_pd_ifr_1.json -> pd_ifr_1
                scheduler = bench_file.stem[6:]  # 去掉 "bench_" 前缀
                bench_result = load_bench_result(bench_file)
                if bench_result:
                    results[key][scheduler] = extract_metrics(bench_result)

    return {
        "tb_values": sorted(tb_values),
        "bs_values": sorted(bs_values),
        "results": results
    }


def find_optimal_configs(data: Dict) -> Dict:
    """找到每个 scheduler 的最优配置"""
    optimal = {}
    # 动态检测所有调度器
    all_schedulers = set()
    for sched_results in data["results"].values():
        all_schedulers.update(sched_results.keys())

    for scheduler in sorted(all_schedulers):
        best_key = None
        best_throughput = 0

        for (tb, bs), sched_results in data["results"].items():
            if scheduler in sched_results:
                tp = sched_results[scheduler].get("throughput", 0)
                if tp > best_throughput:
                    best_throughput = tp
                    best_key = (tb, bs)

        if best_key:
            optimal[scheduler] = {
                "tb": best_key[0],
                "bs": best_key[1],
                "metrics": data["results"][best_key][scheduler]
            }

    return optimal


def compute_improvement_grid(data: Dict, metric: str, pd_scheduler: str = "pd_ifr",
                             higher_better: bool = True) -> Tuple[np.ndarray, list, list]:
    """计算 PD 相对 baseline 的改进网格

    Returns:
        (improvement_matrix, tb_values, bs_values)
    """
    tb_values = data["tb_values"]
    bs_values = data["bs_values"]
    results = data["results"]

    matrix = np.full((len(tb_values), len(bs_values)), np.nan)

    for i, tb in enumerate(tb_values):
        for j, bs in enumerate(bs_values):
            key = (tb, bs)
            if key in results:
                baseline = results[key].get("baseline", {}).get(metric, 0)
                pd = results[key].get(pd_scheduler, {}).get(metric, 0)

                if baseline > 0 and pd > 0:
                    if higher_better:
                        improvement = (pd - baseline) / baseline * 100
                    else:
                        improvement = (baseline - pd) / baseline * 100
                    matrix[i, j] = improvement

    return matrix, tb_values, bs_values


def plot_heatmaps(data: Dict, output_dir: Path):
    """绘制热力图"""
    if not HAS_MATPLOTLIB:
        return

    metrics_to_plot = [
        ("throughput", "Throughput Improvement (%)", True),
        ("mean_itl_ms", "Mean ITL Improvement (%)", False),
        ("mean_ttft_ms", "Mean TTFT Improvement (%)", False),
        ("output_throughput", "Output Throughput Improvement (%)", True),
    ]

    # 检测可用的 PD scheduler
    pd_schedulers = []
    for key in ["pd_ifr", "pd_ratio", "pd"]:
        for sched_results in data["results"].values():
            if key in sched_results:
                pd_schedulers.append(key)
                break

    if not pd_schedulers:
        print("未找到 PD 调度器结果，跳过热力图")
        return

    for pd_sched in pd_schedulers:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            matrix, tb_vals, bs_vals = compute_improvement_grid(
                data, metric, pd_sched, higher_better)

            # 设置颜色范围 (绿色=改进, 红色=退步)
            vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
            vmin = -vmax

            cmap = plt.cm.RdYlGn
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

            # 设置刻度
            ax.set_xticks(range(len(bs_vals)))
            ax.set_xticklabels([str(b) for b in bs_vals], rotation=45)
            ax.set_yticks(range(len(tb_vals)))
            ax.set_yticklabels([str(t) for t in tb_vals])

            ax.set_xlabel('max_num_seqs (BS)')
            ax.set_ylabel('max_num_batched_tokens (TB)')
            ax.set_title(title)

            # 添加数值标注
            for i in range(len(tb_vals)):
                for j in range(len(bs_vals)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                color=color, fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle(f'Multi-Turn: {pd_sched} vs Baseline Improvement', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"heatmap_{pd_sched}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_optimal_comparison(optimal: Dict, output_dir: Path):
    """绘制最优配置对比图"""
    if not HAS_MATPLOTLIB:
        return

    metrics_to_plot = [
        ("throughput", "Request Throughput (req/s)", True),
        ("output_throughput", "Output Throughput (tok/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
        ("mean_ttft_ms", "Mean TTFT (ms)", False),
        ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ]

    # 调度器配置: (key, color, label)
    schedulers = [
        ("baseline", '#2ecc71', 'Baseline'),
        ("pd_ratio", '#3498db', 'PD (ratio)'),
        ("pd_ratio_auto", '#9b59b6', 'PD (ratio auto)'),
        ("pd_ifr", '#1abc9c', 'PD (IFR)'),
        ("pd", '#5dade2', 'PD (legacy)'),
    ]

    # 过滤掉没有数据的调度器
    available_scheds = [(key, color, label) for key, color, label in schedulers if key in optimal]

    if len(available_scheds) < 2:
        print("可用调度器少于 2 个，跳过对比图")
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for j, (metric, ylabel, higher_better) in enumerate(metrics_to_plot):
        ax = axes[j]

        values = []
        colors = []
        labels = []
        for key, color, label in available_scheds:
            opt = optimal[key]
            val = opt["metrics"].get(metric, 0)
            values.append(val)
            colors.append(color)
            labels.append(f'{label}\nTB={opt["tb"]}\nBS={opt["bs"]}')

        x = np.arange(len(values))
        width = 0.6 if len(values) <= 3 else 0.4
        bars = ax.bar(x, values, width=width, color=colors)

        # 计算 PD 相对 baseline 的改进
        baseline_val = optimal.get("baseline", {}).get("metrics", {}).get(metric, 0)
        pd_ifr_val = optimal.get("pd_ifr", {}).get("metrics", {}).get(metric, 0)
        if baseline_val > 0 and pd_ifr_val > 0:
            if higher_better:
                improvement = (pd_ifr_val - baseline_val) / baseline_val * 100
            else:
                improvement = (baseline_val - pd_ifr_val) / baseline_val * 100
            imp_str = f"PD(direct) vs Base: {improvement:+.1f}%"
        else:
            imp_str = ""
            improvement = 0

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ylabel, fontsize=10)

        # 在柱子上方标注数值
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=7)

        # 标注改进百分比
        if imp_str:
            ax.annotate(imp_str, xy=(0.5, 0.95), xycoords='axes fraction',
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       color='green' if improvement > 0 else 'red' if improvement < 0 else 'gray')

        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle('Multi-Turn: Optimal Configuration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "optimal_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(data: Dict, optimal: Dict) -> str:
    """生成文本分析报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("Multi-Turn TB × BS 网格搜索分析报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"TB 值: {data['tb_values']}")
    lines.append(f"BS 值: {data['bs_values']}")
    lines.append("")

    # 动态检测并收集可用的调度器
    def get_display_name(key: str) -> str:
        """将调度器 key 转换为显示名称"""
        name_map = {
            "baseline": "Baseline",
            "pd_ratio": "PD (ratio)",
            "pd_ratio_auto": "PD (ratio auto)",
            "pd_ifr": "PD (IFR)",
            "pd": "PD (legacy)",
        }
        if key in name_map:
            return name_map[key]
        # 对于带后缀的调度器 (如 pd_ifr_1)，生成友好名称
        if key.startswith("pd_"):
            return f"PD ({key[3:]})"
        return key

    available = []
    for key in sorted(optimal.keys()):
        available.append((key, get_display_name(key), optimal[key]))

    if len(available) >= 2:
        lines.append("=" * 80)
        lines.append("最优配置")
        lines.append("=" * 80)
        lines.append("")
        for key, display_name, opt in available:
            lines.append(f"  {display_name:>15}: TB={opt['tb']}, BS={opt['bs']}")
        lines.append("")

        # 获取特定调度器的最优配置
        baseline_opt = optimal.get("baseline", {})
        pd_ifr_opt = optimal.get("pd_ifr", {})
        pd_ratio_opt = optimal.get("pd_ratio", {})

        # 对比关键指标
        header = f"{'Metric':<25}"
        for key, display_name, opt in available:
            header += f" {display_name:<15}"
        if baseline_opt and pd_ifr_opt:
            header += f" {'PD(dir) vs Base':<15}"
        if baseline_opt and pd_ratio_opt:
            header += f" {'PD(rat) vs Base':<15}"
        lines.append(header)

        extra_cols = (1 if baseline_opt and pd_ifr_opt else 0) + (1 if baseline_opt and pd_ratio_opt else 0)
        lines.append("-" * (25 + 15 * len(available) + 15 * extra_cols))

        metrics_compare = [
            ("throughput", "Throughput (req/s)", True),
            ("output_throughput", "Output (tok/s)", True),
            ("mean_itl_ms", "Mean ITL (ms)", False),
            ("mean_ttft_ms", "Mean TTFT (ms)", False),
            ("p99_itl_ms", "P99 ITL (ms)", False),
            ("p99_ttft_ms", "P99 TTFT (ms)", False),
            ("mean_e2e_latency_ms", "Mean E2E (ms)", False),
        ]

        for metric, metric_name, higher_better in metrics_compare:
            row = f"{metric_name:<25}"
            for key, display_name, opt in available:
                val = opt["metrics"].get(metric, 0)
                row += f" {val:<15.2f}"

            # 计算 PD(IFR) vs Baseline 改进
            if baseline_opt and pd_ifr_opt:
                b_val = baseline_opt["metrics"].get(metric, 0)
                p_val = pd_ifr_opt["metrics"].get(metric, 0)
                if b_val > 0:
                    if higher_better:
                        improvement = (p_val - b_val) / b_val * 100
                    else:
                        improvement = (b_val - p_val) / b_val * 100
                    imp_str = f"{improvement:+.2f}%"
                else:
                    imp_str = "N/A"
                row += f" {imp_str:<15}"

            # 计算 PD(ratio) vs Baseline 改进
            if baseline_opt and pd_ratio_opt:
                b_val = baseline_opt["metrics"].get(metric, 0)
                p_val = pd_ratio_opt["metrics"].get(metric, 0)
                if b_val > 0:
                    if higher_better:
                        improvement = (p_val - b_val) / b_val * 100
                    else:
                        improvement = (b_val - p_val) / b_val * 100
                    imp_str = f"{improvement:+.2f}%"
                else:
                    imp_str = "N/A"
                row += f" {imp_str:<15}"

            lines.append(row)

        # 判断胜负 (基于吞吐量)
        throughputs = {display_name: opt["metrics"].get("throughput", 0)
                      for key, display_name, opt in available}
        winner = max(throughputs, key=throughputs.get)
        best_tp = throughputs[winner]

        lines.append("")
        lines.append(f"结论: {winner} 在吞吐量上胜出 ({best_tp:.2f} req/s)")

    # 总结
    lines.append("")
    lines.append("=" * 80)
    lines.append("总体结论")
    lines.append("=" * 80)

    # 统计胜出 (动态检测所有 pd_ 开头的调度器)
    baseline_tp = optimal.get("baseline", {}).get("metrics", {}).get("throughput", 0)
    best_pd_tp = 0
    best_pd_name = None
    for v, opt_data in optimal.items():
        if v.startswith("pd_"):
            tp = opt_data.get("metrics", {}).get("throughput", 0)
            if tp > best_pd_tp:
                best_pd_tp = tp
                best_pd_name = v

    lines.append("")
    if baseline_tp > 0 and best_pd_tp > 0:
        if best_pd_tp > baseline_tp:
            improvement = (best_pd_tp - baseline_tp) / baseline_tp * 100
            lines.append(f"总体: {best_pd_name} 胜出 (吞吐量提升 {improvement:+.2f}%)")
        elif baseline_tp > best_pd_tp:
            improvement = (baseline_tp - best_pd_tp) / baseline_tp * 100
            lines.append(f"总体: Baseline 胜出 (PD 吞吐量下降 {improvement:.2f}%)")
        else:
            lines.append("总体: Baseline 与 PD 持平")
    else:
        lines.append("总体: 数据不足，无法比较")

    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_results.py <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"目录不存在: {exp_dir}")
        sys.exit(1)

    print(f"分析实验目录: {exp_dir}")
    print("")

    # 收集数据
    data = collect_grid_results(exp_dir)

    if not data["results"]:
        print("未找到任何实验结果")
        sys.exit(1)

    print(f"找到 {len(data['tb_values'])} 个 TB 值: {data['tb_values']}")
    print(f"找到 {len(data['bs_values'])} 个 BS 值: {data['bs_values']}")
    print("")

    # 找最优配置
    optimal = find_optimal_configs(data)

    # 生成报告
    report = generate_report(data, optimal)
    print(report)

    # 保存报告
    report_path = exp_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")

    # 绘图
    plot_heatmaps(data, exp_dir)
    plot_optimal_comparison(optimal, exp_dir)

    # 保存汇总 JSON
    summary = {
        "tb_values": data["tb_values"],
        "bs_values": data["bs_values"],
        "optimal": optimal,
        "all_results": {
            f"tb{tb}_bs{bs}": sched_results
            for (tb, bs), sched_results in data["results"].items()
        }
    }

    summary_path = exp_dir / "grid_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"汇总数据已保存: {summary_path}")


if __name__ == "__main__":
    main()
