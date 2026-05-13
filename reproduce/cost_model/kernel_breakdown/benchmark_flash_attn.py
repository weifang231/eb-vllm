"""
FlashAttention Microbenchmark: Pure Prefill vs Pure Decode vs Mixed

Tests FlashAttention kernel performance under different batch compositions,
Used to analyze why the PD scheduler is more effective on A6000 than on H200.

Usage:
    python reproduce/cost_model/kernel_breakdown/benchmark_flash_attn.py \
        --batch-size 512 \
        --prefill-len 512 \
        --context-len 512 \
        --output results.json
"""
import torch
import argparse
from typing import List, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def create_attention_inputs(
    batch_size: int,
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, ...]:
    """Create input tensors for FlashAttention."""

    total_tokens = sum(query_lens)
    max_context = max(context_lens)
    max_blocks = (max_context + block_size - 1) // block_size

    # Query tensor: [total_tokens, num_heads, head_size]
    query = torch.randn(total_tokens, num_heads, head_size,
                       device=device, dtype=dtype)

    # KV cache: [2, num_blocks, block_size, num_kv_heads, head_size]
    num_blocks = batch_size * max_blocks
    kv_cache = torch.randn(2, num_blocks, block_size, num_kv_heads, head_size,
                          device=device, dtype=dtype)

    # cu_seqlens_q: cumulative query lengths
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(torch.tensor(query_lens, device=device), dim=0)

    # seqused_k: context length per request
    seqused_k = torch.tensor(context_lens, dtype=torch.int32, device=device)

    # block_table: [batch_size, max_blocks]
    block_table = torch.arange(batch_size * max_blocks, device=device, dtype=torch.int32)
    block_table = block_table.view(batch_size, max_blocks)

    # Output tensor
    output = torch.empty(total_tokens, num_heads, head_size,
                        device=device, dtype=dtype)

    return (query, kv_cache, cu_seqlens_q, seqused_k, block_table, output,
            max(query_lens), max_context)


def benchmark_scenario(
    scenario_name: str,
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    block_size: int = 16,
    warmup: int = 10,
    repeat: int = 100,
) -> dict:
    """Benchmark a single scenario"""

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch_size = len(query_lens)
    query, kv_cache, cu_seqlens_q, seqused_k, block_table, output, max_q, max_k = \
        create_attention_inputs(
            batch_size, query_lens, context_lens,
            num_heads, num_kv_heads, head_size, block_size
        )

    key_cache = kv_cache[0]
    value_cache = kv_cache[1]

    # Warmup
    for _ in range(warmup):
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_k,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=True,
            block_table=block_table,
        )

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start_event.record()
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_k,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=True,
            block_table=block_table,
        )
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    total_tokens = sum(query_lens)
    avg_time = sum(times) / len(times)

    return {
        "scenario": scenario_name,
        "batch_size": batch_size,
        "total_query_tokens": total_tokens,
        "total_context_tokens": sum(context_lens),
        "max_query_len": max_q,
        "max_context_len": max_k,
        "avg_time_ms": avg_time,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "tokens_per_sec": total_tokens / (avg_time / 1000),
        "query_lens_sample": query_lens[:5],
    }


def plot_results(results: List[dict], output_path: str, gpu_name: str):
    """Plot benchmark results and compute slopes."""
    # Extract data
    pure_decode = next(r for r in results if r['scenario'] == 'pure_decode')
    pure_decode_time = pure_decode['avg_time_ms']

    # Collect mixed-scenario data (only 10%, 20%, 40%, 80%)
    prefill_pcts = []
    slowdowns = []
    times = []

    for pct in [10, 20, 40, 80]:
        scenario_name = f"mixed_{pct}pct_prefill"
        mixed = next((r for r in results if r['scenario'] == scenario_name), None)
        if mixed:
            prefill_pcts.append(pct)
            slowdowns.append(mixed['avg_time_ms'] / pure_decode_time)
            times.append(mixed['avg_time_ms'])

    # Linear-regression slope
    fit_x = np.linspace(0, 100, 100)
    if len(prefill_pcts) >= 2:
        pcts_arr = np.array(prefill_pcts)
        slowdowns_arr = np.array(slowdowns)
        times_arr = np.array(times)

        # Slope computation: y = slope * x + intercept
        slope_slowdown, intercept_slowdown = np.polyfit(pcts_arr, slowdowns_arr, 1)
        slope_time, intercept_time = np.polyfit(pcts_arr, times_arr, 1)

        print(f"\n{'='*70}")
        print(f"LINEAR REGRESSION (Mixed scenarios: 10%, 20%, 40%, 80%)")
        print(f"{'='*70}")
        print(f"Slowdown slope:    {slope_slowdown:.4f}x per 1% prefill increase")
        print(f"Time slope:        {slope_time:.4f} ms per 1% prefill increase")
        print(f"Slowdown at 50%:   {slope_slowdown * 50 + intercept_slowdown:.2f}x (predicted)")
    else:
        slope_slowdown = slope_time = intercept_slowdown = intercept_time = 0

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fig 1: Slowdown vs Prefill Percentage
    ax1 = axes[0]
    ax1.plot(prefill_pcts, slowdowns, 'bo-', linewidth=2, markersize=10, label='Measured')

    # Plot the fit line
    if len(prefill_pcts) >= 2:
        fit_x = np.linspace(0, 100, 100)
        fit_y = slope_slowdown * fit_x + intercept_slowdown
        ax1.plot(fit_x, fit_y, 'r--', linewidth=1.5, alpha=0.7,
                label=f'Linear fit (slope={slope_slowdown:.3f})')

    ax1.set_xlabel('Prefill Percentage (%)', fontsize=12)
    ax1.set_ylabel('Slowdown (vs Pure Decode)', fontsize=12)
    ax1.set_title(f'FlashAttention Slowdown vs Prefill Ratio\n({gpu_name})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)
    ax1.legend(loc='upper left')

    # Annotate key points
    for pct, sd in zip(prefill_pcts, slowdowns):
        ax1.annotate(f'{sd:.2f}x', (pct, sd), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    # Fig 2: Kernel Time vs Prefill Percentage
    ax2 = axes[1]
    ax2.plot(prefill_pcts, times, 'go-', linewidth=2, markersize=10, label='Measured')

    # Plot the fit line
    if len(prefill_pcts) >= 2:
        fit_y = slope_time * fit_x + intercept_time
        ax2.plot(fit_x, fit_y, 'r--', linewidth=1.5, alpha=0.7,
                label=f'Linear fit (slope={slope_time:.4f} ms/%)')

    ax2.set_xlabel('Prefill Percentage (%)', fontsize=12)
    ax2.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax2.set_title(f'FlashAttention Kernel Time vs Prefill Ratio\n({gpu_name})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.legend(loc='upper left')

    # Annotate key points
    for pct, t in zip(prefill_pcts, times):
        ax2.annotate(f'{t:.2f}ms', (pct, t), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")

    return {
        "slope_slowdown": slope_slowdown,
        "slope_time_ms": slope_time,
        "intercept_slowdown": intercept_slowdown,
        "intercept_time_ms": intercept_time,
    }


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Microbenchmark")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size (number of sequences)")
    parser.add_argument("--prefill-len", type=int, default=512,
                       help="Prefill sequence length")
    parser.add_argument("--context-len", type=int, default=512,
                       help="Context length (KV cache size per request)")
    parser.add_argument("--num-heads", type=int, default=32,
                       help="Number of attention heads (Qwen3-8B: 32)")
    parser.add_argument("--num-kv-heads", type=int, default=8,
                       help="Number of KV heads (Qwen3-8B: 8)")
    parser.add_argument("--head-size", type=int, default=128,
                       help="Head dimension (Qwen3-8B: 128)")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: batch_size={args.batch_size}, prefill_len={args.prefill_len}, "
          f"context_len={args.context_len}")
    print(f"Model: num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}, "
          f"head_size={args.head_size}")

    results = []

    # Scenario 1: Pure Prefill
    num_prefill_seqs = args.batch_size // args.prefill_len
    if num_prefill_seqs < 1:
        num_prefill_seqs = 1
    print(f"\n=== Scenario 1: Pure Prefill ({num_prefill_seqs} seqs x {args.prefill_len} tokens) ===")
    result = benchmark_scenario(
        "pure_prefill",
        query_lens=[args.prefill_len] * num_prefill_seqs,
        context_lens=[args.context_len] * num_prefill_seqs,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Scenario 2: Pure Decode
    print(f"\n=== Scenario 2: Pure Decode ({args.batch_size} seqs x 1 token) ===")
    result = benchmark_scenario(
        "pure_decode",
        query_lens=[1] * args.batch_size,
        context_lens=[args.context_len] * args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Scenario 3: Mixed (1 prefill + rest decode)
    num_decode = args.batch_size - 1
    print(f"\n=== Scenario 3: Mixed (1 prefill x {args.prefill_len} + {num_decode} decode x 1) ===")
    result = benchmark_scenario(
        "mixed_1prefill",
        query_lens=[args.prefill_len] + [1] * num_decode,
        context_lens=[args.context_len] * args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Scenario 4-7: Mixed with varying prefill percentages (10%, 20%, 40%, 80%)
    for pct in [10, 20, 40, 80]:
        num_prefill = max(1, args.batch_size * pct // 100)
        num_decode = args.batch_size - num_prefill
        print(f"\n=== Scenario: Mixed {pct}% ({num_prefill} prefill + {num_decode} decode) ===")
        result = benchmark_scenario(
            f"mixed_{pct}pct_prefill",
            query_lens=[args.prefill_len] * num_prefill + [1] * num_decode,
            context_lens=[args.context_len] * args.batch_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(result)
        print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Scenario':<25} {'Time(ms)':<12} {'Tokens':<10} {'Tok/s':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['scenario']:<25} {r['avg_time_ms']:<12.3f} {r['total_query_tokens']:<10} {r['tokens_per_sec']:<15.0f}")

    # Analysis: Compare mixed vs pure_decode
    pure_decode = next(r for r in results if r['scenario'] == 'pure_decode')
    mixed_1 = next(r for r in results if r['scenario'] == 'mixed_1prefill')

    print("\n" + "="*70)
    print("ANALYSIS: Mixed vs Pure Decode (slowdown ratio)")
    print("="*70)
    print(f"{'Scenario':<25} {'Time(ms)':<12} {'Slowdown':<10}")
    print("-"*70)
    print(f"{'pure_decode':<25} {pure_decode['avg_time_ms']:<12.3f} {'1.00x':<10}")
    print(f"{'mixed_1prefill':<25} {mixed_1['avg_time_ms']:<12.3f} "
          f"{mixed_1['avg_time_ms']/pure_decode['avg_time_ms']:.2f}x")

    analysis = {
        "pure_decode_time_ms": pure_decode['avg_time_ms'],
        "mixed_1prefill_slowdown": mixed_1['avg_time_ms'] / pure_decode['avg_time_ms'],
    }

    for pct in [10, 20, 40, 80]:
        scenario_name = f"mixed_{pct}pct_prefill"
        mixed = next((r for r in results if r['scenario'] == scenario_name), None)
        if mixed:
            slowdown = mixed['avg_time_ms'] / pure_decode['avg_time_ms']
            print(f"{scenario_name:<25} {mixed['avg_time_ms']:<12.3f} {slowdown:.2f}x")
            analysis[f"mixed_{pct}pct_slowdown"] = slowdown

    # Save results
    gpu_name = torch.cuda.get_device_name()
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "gpu": gpu_name,
            "results": results,
            "analysis": analysis,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

        # Generate plots
        plot_path = args.output.replace('.json', '.png')
        slope_info = plot_results(results, plot_path, gpu_name)

        # Update JSON with slope information
        output_data["slope_analysis"] = slope_info
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
