#!/usr/bin/env python3
"""Offline closed-form helper for Table 1 (controller validation).

Given (alpha_p, beta_p, alpha_d, beta_d, p_0, mu_L, C, eps), print the
deterministic closed-form (theta_0, N_hat, k_hat) from Prop. threshold_cfr
and Prop. memory. Mirrors the math in scheduler.py:_compute_theta_zero_exact
and _compute_memory_safe_n so the offline value matches what the live
scheduler would compute given the same pinned inputs.

Usage:
  compute_closed_form.py --alpha-p F --beta-p F --alpha-d F --beta-d F \
      --p0 F --mu-l F --capacity-tokens INT --eps F [--json]
"""
import argparse
import json
import math
import sys


def theta_zero_exact(p_0: float, alpha_p: float, alpha_d: float) -> float:
    """Newton solve of theta/(1-theta) + ln(1-theta) = -ln(1-p_0)*alpha_p/alpha_d."""
    if p_0 <= 0 or p_0 >= 1:
        raise ValueError(f"p_0 out of range: {p_0}")
    rhs = -math.log(1 - p_0) * alpha_p / alpha_d
    th = 0.5
    for _ in range(200):
        f = th / (1 - th) + math.log(1 - th) - rhs
        df = th / (1 - th) ** 2  # d/dth [th/(1-th)+ln(1-th)] = th/(1-th)^2
        if abs(df) < 1e-14:
            break
        step = f / df
        th_new = th - step
        if th_new <= 0:
            th_new = th * 0.5
        elif th_new >= 1:
            th_new = (th + 1) * 0.5
        if abs(th_new - th) < 1e-14:
            th = th_new
            break
        th = th_new
    return th


def memory_safe_n(theta: float, p_0: float, mu_L: float,
                  capacity_tokens: int, eps: float) -> int:
    """N_hat from Prop. memory (CLT-form quadratic, matches scheduler.py)."""
    if not (0 < theta < 1) or p_0 <= 0 or capacity_tokens <= 0:
        raise ValueError("invalid inputs")
    eps_clamped = max(min(eps, 0.5), 1e-6)
    Lambda = -math.log(1 - theta)
    D_theta = mu_L + (1 - theta) / (theta * p_0) * Lambda
    if D_theta <= 0:
        raise ValueError("D(theta) <= 0")
    lambda_theta = Lambda / theta
    sigma_sq = (2.0 * Lambda * (1.0 + (p_0 * mu_L + lambda_theta) ** 2)
                / max(p_0 ** 2, 1e-18))
    sigma = math.sqrt(max(sigma_sq, 0.0))
    log_inv_eps = math.log(1.0 / eps_clamped)
    sqrt_N = (-sigma * math.sqrt(log_inv_eps)
              + math.sqrt(sigma_sq * log_inv_eps + 4.0 * D_theta * capacity_tokens)
              ) / (2.0 * D_theta)
    return max(1, int(math.floor(max(0.0, sqrt_N) ** 2)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-p", type=float, required=True)
    p.add_argument("--beta-p", type=float, required=True)
    p.add_argument("--alpha-d", type=float, required=True)
    p.add_argument("--beta-d", type=float, required=True)
    p.add_argument("--p0", type=float, required=True)
    p.add_argument("--mu-l", type=float, required=True)
    p.add_argument("--capacity-tokens", type=int, required=True)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    theta = theta_zero_exact(args.p0, args.alpha_p, args.alpha_d)
    n_hat = memory_safe_n(theta, args.p0, args.mu_l, args.capacity_tokens, args.eps)
    k_hat = max(1, int(round(theta * n_hat)))

    out = {
        "theta_0": theta,
        "N_hat": n_hat,
        "k_hat": k_hat,
        "k_ratio": theta,  # alias for VLLM_PD_K_RATIO
    }
    if args.json:
        print(json.dumps(out))
    else:
        print(f"theta_0={theta:.6f}")
        print(f"N_hat={n_hat}")
        print(f"k_hat={k_hat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
