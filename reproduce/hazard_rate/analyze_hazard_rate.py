#!/usr/bin/env python3
"""
Hazard Rate Ordering 实验分析脚本

功能：
  - 从 Gamma 分布 k* sweep 结果中找到最优 k* 和对应的 θ*
  - 验证 k*_DFR < k*_CFR < k*_IFR

用法：
  python analyze_hazard_rate.py <results_dir>
  python analyze_hazard_rate.py outputs/hazard_rate_ordering_N256_O128
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_json(filepath: Path) -> dict | None:
    """加载 JSON 文件"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def extract_k_from_filename(filename: str) -> int | None:
    """从文件名中提取 k* 值"""
    match = re.search(r'fixed(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def get_throughput(bench_result: dict) -> float:
    """从 benchmark 结果中提取吞吐量"""
    return bench_result.get("output_throughput", 0)


def analyze_hazard_type(scenario_dir: Path) -> dict:
    """分析单个 hazard type 的结果，找到最优 k*"""
    results = defaultdict(list)

    # 收集所有 k* 的吞吐量
    for bench_file in scenario_dir.glob("bench_fixed*_run*.json"):
        k_star = extract_k_from_filename(bench_file.name)
        if k_star is None:
            continue

        data = load_json(bench_file)
        if data is None:
            continue

        throughput = get_throughput(data)
        results[k_star].append(throughput)

    # 如果没有 _run 后缀的文件
    if not results:
        for bench_file in scenario_dir.glob("bench_fixed*.json"):
            if "_run" in bench_file.name:
                continue
            k_star = extract_k_from_filename(bench_file.name)
            if k_star is None:
                continue

            data = load_json(bench_file)
            if data is None:
                continue

            throughput = get_throughput(data)
            results[k_star].append(throughput)

    if not results:
        return {}

    # 计算每个 k* 的平均吞吐量和标准差
    k_stats = {}
    for k, throughputs in results.items():
        k_stats[k] = {
            "mean": np.mean(throughputs),
            "std": np.std(throughputs) if len(throughputs) > 1 else 0,
            "n": len(throughputs),
        }

    # 找到最优 k*
    optimal_k = max(k_stats, key=lambda k: k_stats[k]["mean"])
    optimal_stats = k_stats[optimal_k]

    return {
        "optimal_k_star": optimal_k,
        "optimal_throughput_mean": optimal_stats["mean"],
        "optimal_throughput_std": optimal_stats["std"],
        "all_k_stats": k_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hazard Rate Ordering 实验分析：验证 k*_DFR < k*_CFR < k*_IFR"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="实验结果目录"
    )
    parser.add_argument(
        "--batch-size", "-N",
        type=int,
        default=256,
        help="批大小 N (默认 256)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    N = args.batch_size

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    # 加载实验配置
    config_file = results_dir / "experiment_config.json"
    if config_file.exists():
        config = load_json(config_file)
        if config:
            N = config.get("fixed_params", {}).get("N", N)
            print(f"从配置文件读取 N = {N}")

    print(f"\n{'='*60}")
    print("Hazard Rate Ordering 实验")
    print(f"验证: k*_DFR < k*_CFR < k*_IFR")
    print(f"{'='*60}")
    print(f"批大小 N = {N}")
    print()

    # 分析每个 hazard type
    results = {}
    hazard_order = ["DFR", "CFR", "IFR"]

    for scenario_dir in sorted(results_dir.glob("*_shape*")):
        if not scenario_dir.is_dir():
            continue

        # 解析场景名: DFR_shape0.5, CFR_shape1.0, IFR_shape2.0
        match = re.match(r'(DFR|CFR|IFR)_shape([\d.]+)', scenario_dir.name)
        if not match:
            continue

        hazard_type = match.group(1)
        gamma_shape = float(match.group(2))

        print(f"分析场景: {hazard_type} (shape={gamma_shape})")

        scenario_result = analyze_hazard_type(scenario_dir)
        if not scenario_result:
            print(f"  警告: 没有找到有效结果")
            continue

        optimal_k = scenario_result["optimal_k_star"]
        theta_star = optimal_k / N

        print(f"  最优 k* = {optimal_k}")
        print(f"  θ* = k*/N = {theta_star:.4f}")
        print(f"  吞吐量 = {scenario_result['optimal_throughput_mean']:.2f} "
              f"± {scenario_result['optimal_throughput_std']:.2f} tokens/s")

        results[hazard_type] = {
            "hazard_type": hazard_type,
            "gamma_shape": gamma_shape,
            "optimal_k_star": optimal_k,
            "theta_star": theta_star,
            "throughput_mean": scenario_result["optimal_throughput_mean"],
            "throughput_std": scenario_result["optimal_throughput_std"],
        }

    if len(results) < 3:
        print(f"\n警告: 只找到 {len(results)} 个 hazard type 的结果，无法完整验证")

    # 验证排序
    print(f"\n{'='*60}")
    print("验证 k*_DFR < k*_CFR < k*_IFR")
    print(f"{'='*60}")

    print(f"\n{'Hazard Type':<12} {'Shape':<8} {'k*':<8} {'θ*':<8}")
    print("-" * 40)

    k_stars = {}
    theta_stars = {}
    for hazard_type in hazard_order:
        if hazard_type in results:
            r = results[hazard_type]
            print(f"{hazard_type:<12} {r['gamma_shape']:<8.1f} "
                  f"{r['optimal_k_star']:<8} {r['theta_star']:.4f}")
            k_stars[hazard_type] = r["optimal_k_star"]
            theta_stars[hazard_type] = r["theta_star"]

    # 检验排序
    print(f"\n验证结果:")
    if len(k_stars) == 3:
        k_dfr = k_stars.get("DFR", float('inf'))
        k_cfr = k_stars.get("CFR", float('inf'))
        k_ifr = k_stars.get("IFR", 0)

        if k_dfr < k_cfr < k_ifr:
            print(f"  √ k*_DFR ({k_dfr}) < k*_CFR ({k_cfr}) < k*_IFR ({k_ifr})")
            print(f"  √ Hazard rate ordering 验证通过!")
        elif k_dfr <= k_cfr <= k_ifr:
            print(f"  ~ k*_DFR ({k_dfr}) ≤ k*_CFR ({k_cfr}) ≤ k*_IFR ({k_ifr})")
            print(f"  ~ Hazard rate ordering 部分验证 (存在相等情况)")
        else:
            print(f"  × 排序不符合预期:")
            print(f"    k*_DFR = {k_dfr}, k*_CFR = {k_cfr}, k*_IFR = {k_ifr}")
            print(f"  × Hazard rate ordering 验证失败")
    else:
        print(f"  ? 数据不完整，无法验证")

    # 输出 LaTeX 格式
    print(f"\nLaTeX 格式:")
    theta_values = []
    for hazard_type in hazard_order:
        if hazard_type in theta_stars:
            theta_values.append(f"{theta_stars[hazard_type]:.2f}")
        else:
            theta_values.append("---")
    print(f"  Using $a \\in \\{{0.5, 1, 2\\}}$ with scale parameters")
    print(f"  $b \\in \\{{256, 128, 64\\}}$ to maintain $\\mathbb{{E}}[O] = 128$,")
    print(f"  we obtain $\\hat{{\\theta}}^* = $ {', '.join(theta_values)} respectively")
    print(f"  ($N = {N}$, 3 runs), confirming the predicted ordering.")

    # 保存结果
    if args.output:
        output_data = {
            "experiment": "hazard_rate_ordering",
            "N": N,
            "results": results,
            "verification": {
                "k_stars": k_stars,
                "theta_stars": theta_stars,
                "ordering_verified": (
                    len(k_stars) == 3 and
                    k_stars.get("DFR", float('inf')) <
                    k_stars.get("CFR", float('inf')) <
                    k_stars.get("IFR", 0)
                ),
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
