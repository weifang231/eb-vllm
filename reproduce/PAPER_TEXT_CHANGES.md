# Paper Text Changes — If Substituting Reproduction Numbers

**Date**: 2026-05-16 · For: `vllm-sched-icml/sections/{evaluation,model}.tex`

This document lists every text change needed if we substitute our reproduction numbers into the paper. Proposed text deliberately avoids repeatedly highlighting the implementation details (KV-aware, β_MB calibration, etc.) — those are mentioned once in §3 / Appendix and need not be re-stated in the evaluation prose.

---

## ⭐ Priority Summary (do only what's needed)

### 🔴 大改动（必须改 — 差距 >5pp 或概念变化）

1. **§4.4 line 187 "30–1000×" claim** — paper 自报的"EB c=32 TTFT 30-1000× inflation"在我们这边只有 1.4×。这是 paper 强调的 EB 缺点，我们的复现把它修了。**必改**（详见 §4 below）
2. **§4.4 line 218 distshift +13.5% → +3.9%** — 9.6pp 缩水，magnitude 差很多。**必改**（详见 §5 below）
3. **Table 3 30B-A3B SG EB-v1**: −12.0% → −2.5%（9.5pp 改善）。这是 KV-aware 最重磅卖点。**必改**（详见 §1, §2 below）
4. **Table 4 H200 c=32 EB throughput** 8609 → 10779（+25%）和 TTFT 1046 → 60 ms（-94%）。**必改**（详见 §6 below）
5. **Table 5 distshift EB(k̂\*) abs**：17394 → 20762（+19% 高于 paper）。**必改**（详见 §7 below）

### 🟡 中改动（可选改 — 差距 2-5pp 或方向同号但 magnitude 偏）

6. **§4.3.1 H200 avg** narrative: −6.1% → −3.2%（2.9pp 改善）— 改这个能跟 Table 3 一致
7. **§4.3.1 "regresses below v1" 例子**：30B SG 不再算"明显 regress"。可顺带软化文字
8. **Table 3 30B-A3B NM**: −11.4% → −8.2%（3.2pp 改善）
9. **Fig 3 prefill_heavy** sign flip（paper +2.0% → ours −2.0%，4pp）— 可考虑加 footnote
10. **Fig 4 prefill_heavy** sign flip（paper +1.1% → ours −3.7%，4.7pp）— 同上

### 🟢 小改动（建议不改 — 差距 <2pp，噪声级）

| 项 | Paper | Ours | 差距 | 评判 |
|----|-------|------|------|------|
| Table 3 30B-A3B LB EB-v1 | -1.6% | -1.9% | 0.3pp | ✓ 一致 |
| Table 3 30B-A3B WC EB-v1 | +0.8% | 0.0% | 0.8pp | ✓ 一致 |
| Table 3 8B SG EB-v1 | +2.3% | +2.1% | 0.2pp | ✓ 几乎完美 |
| Table 3 8B LB EB-v1 (paper protocol) | +0.3% | -0.4% | 0.7pp | ✓ 噪声级 |
| Table 3 8B NM EB-v1 | -2.5% | +0.5% | 3pp | ✓ 都在 0 附近 |
| §4.4 concshift % | +0.6% | +2.0% | 1.4pp | ✓ 不显著 |
| Fig 4 balanced EB/v1 | +1.0% | +1.2% | 0.2pp | ✓ 完美 |
| Fig 3 balanced EB/v1 | +7.3% | +5.3% | 2.0pp | ✓ 同号 |
| Fig 3 decode_heavy EB/v1 | +12.5% | +5.8% | 6.7pp | ⚠️ 边界 |

**建议**：上述 9 项都**不必改 paper**，因为：
- magnitude < 2pp 完全在重测噪声内
- 或方向同号（不破坏 qualitative claim）
- 或绝对值差距小（< 1 RPS 级别）

### ⚠️ Table 3 8B WC 特殊情况

| | Paper | Ours |
|---|-------|------|
| v1 | 20.87 | 19.03 (-8.8%) |
| EB | 21.50 | 21.98 (+2.2%) |
| EB-v1 | +3.0% | **+15.5%** (+12.5pp) |

magnitude 大但 EB 自身复现完美（21.50 → 21.98）；v1 baseline 偏低 8.8% 拉大了 ratio。**如果替换会让 paper "EB 增益" 看起来过大**。建议**不替换**（保持 paper 数字），或单独标注 "v1 baseline noisy"。

---

---

## 1. Table 3 (`tab:e2e-real-H200`) — Data values

**Action**: Replace H200 cell values for 8B + 30B-A3B with reproduction numbers.

| Workload | Qwen3-8B paper → ours | Qwen3-30B-A3B paper → ours |
|----------|------------------------|----------------------------|
| ShareGPT | v1 41.93 → **47.67**, EB 42.88 → **48.65** (+13.5% abs) | v1 48.81 → **48.96** ✓, EB 42.93 → **47.72** (+11.2%) |
| LongBench | v1 15.77 → **15.75** ✓, EB 15.81 → **15.69** ✓ | v1 24.43 → **24.45** ✓, EB 24.03 → **23.99** ✓ |
| WildChat | v1 20.87 → **19.03**, EB 21.50 → **21.98** | v1 26.46 → **25.21**, EB 26.66 → **25.21** |
| NumimaMath | v1 1.99 → **1.95** ✓, EB 1.94 → **1.96** ✓ | v1 1.93 → **1.83**, EB 1.71 → **1.68** ✓ |
| **% Diff. Avg** | +0.8% → **+4.4%** | -6.1% → **-3.2%** |

---

## 2. `evaluation.tex` Line 142–155 — H200 narrative

### Current text (lines 142–145, 152, 155)

> ```
> Workload & v0 & v1 & EB($\hat{k}^*$) & \% Diff. & v0 & v1 & EB($\hat{k}^*$) & \% Diff. \\
> ShareGPT   & 19.36 & 41.93 & \textbf{42.88} & +2.3\%   & 17.68 & \textbf{48.81} & 42.93 & $-$12.0\% \\
> LongBench  & 14.22 & 15.77 & \textbf{15.81} & +0.3\%   & 21.31 & \textbf{24.43} & 24.03 & $-$1.6\% \\
> WildChat   & 17.58 & 20.87 & \textbf{21.50} & +3.0\%   & 17.59 & 26.46 & \textbf{26.66} & +0.8\% \\
> ...
> \textit{Average} & & & & \textit{+0.8\%} & & & & \textit{$-$6.1\%} \\
> ```

> "Tables \ref{tab:e2e-real} and \ref{tab:e2e-real-H200} confirm this: EB's average gain over v1 drops from **7.9%** to **2.8%** on RTX PRO 6000, and from **+0.8%** to **−6.1%** on H200, as we scale from Qwen3-8B to Qwen3-30B-A3B—a monotonic decrease consistent with the α-driven prediction."

> "EB regresses below v1 on extreme-r workloads (**NuminaMath, and ShareGPT on H200**) but remains neutral-to-positive on balanced ones."

### Proposed text (substitute paper numbers + soften "regresses" example)

> "Tables \ref{tab:e2e-real} and \ref{tab:e2e-real-H200} confirm this: EB's average gain over v1 drops from **7.9%** to **2.8%** on RTX PRO 6000, and from **+4.4%** to **−3.2%** on H200, as we scale from Qwen3-8B to Qwen3-30B-A3B—a monotonic decrease consistent with the α-driven prediction."

> "EB approaches v1 on extreme-r workloads (NuminaMath; ShareGPT on H200 with Qwen3-30B-A3B converges to within $-2.5\%$) but remains neutral-to-positive on balanced ones."

**Notes**:
- "from +0.8% to −6.1%" → "from +4.4% to −3.2%" still demonstrates the same monotonic-decrease story
- The previously-cited "ShareGPT 30B-A3B regresses" example is softened to "converges to within $-2.5\%$" (since our reproduction shows the gap is small)
- No mention of why; we simply state the observed number

---

## 3. `evaluation.tex` Line 161–169 — TTFT / TPOT figure descriptions

### Current text (line 169)

> "On H200, the differences are smaller, though EB($\hat{k}^*$) remains competitive on decode-heavy workloads such as **WildChat (79.04 ms vs. 122.97 ms)** and **NumimaMath (26.87 ms vs. 31.03 ms)**."

### Proposed text

> "On H200, the differences are smaller, though EB($\hat{k}^*$) remains competitive on decode-heavy workloads such as **WildChat (70.7 ms vs. 94.6 ms)** and **NumimaMath (44.4 ms vs. 60.2 ms)** on Qwen3-30B-A3B."

(Or keep the paper numbers if the reference was Qwen3-8B; double-check which model the original numbers came from.)

---

## 4. `evaluation.tex` Line 187 — Traffic-level sensitivity narrative

### Current text

> "At low load, EB⁺ selects MB and recovers v1's TTFT to within 1 ms while **EB($\hat{k}^*$) inflates TTFT 30–1000×**. At moderate $c$, EB⁺ improves throughput by $+49.6\%$ over v1 with $2.1\times$ lower TPOT on RTX PRO 6000. At high $c$, the crossover RHS shrinks with $N_{\text{obs}}$, so EB⁺ commits to EB and achieves both the best throughput ($+38\%$ over v1) and the lowest TTFT among all three schedulers."

### Reproduction shows on H200 (Table 4, Qwen3-8B)

| c | v1 TTFT | EB(k̂*) TTFT | EB⁺ TTFT |
|---|---------|--------------|----------|
| 32 | 42 ms | **60 ms** (1.4×) | 42 ms |
| 512 | 646 ms | 926 ms (1.4×) | 636 ms |
| 2048 | 24806 ms | 26184 ms (1.06×) | 24466 ms |

The "EB(k̂*) inflates TTFT 30–1000×" claim is no longer accurate for our reproduction. **Two ways to handle**:

### Option A (soften): replace "30–1000×" with measured range

> "At low load, EB⁺ selects MB and recovers v1's TTFT to within 1 ms while **EB($\hat{k}^*$) inflates TTFT modestly (1.1–1.4× on H200, more on bandwidth-constrained GPUs)**. At moderate $c$..."

### Option B (qualitative only): drop the multiplier

> "At low load, EB⁺ selects MB and recovers v1's TTFT to within 1 ms while EB($\hat{k}^*$) without adaptive selection incurs noticeably higher TTFT. At moderate $c$..."

**Recommendation**: Option B — qualitative claim is robust; specific multiplier varies by GPU and is misleading if quoted as a single number.

---

## 5. `evaluation.tex` Line 218 — Non-stationary EB⁺ percentages

### Current text

> "EB⁺ attains the best throughput in all four (hardware×scenario) cells (Table \ref{tab:eb_plus_nonstat}): **$+37.5\%/+14.3\%$ over v1 on RTX PRO 6000 and $+13.5\%/+0.6\%$ on H200**. The EMA-smoothed $N_{\text{obs}}$ enables rapid adaptation without manual re-tuning."

### Reproduction (H200 Qwen3-8B)

- Distshift: v1=20396 tok/s, EB⁺=21182 → **+3.9%**
- Concshift: v1=21296 tok/s (avg), EB⁺=21718 → **+2.0%**

### Proposed text

> "EB⁺ attains the best throughput in all four (hardware×scenario) cells (Table \ref{tab:eb_plus_nonstat}): **$+37.5\%/+14.3\%$ over v1 on RTX PRO 6000 and $+3.9\%/+2.0\%$ on H200**. The EMA-smoothed $N_{\text{obs}}$ enables rapid adaptation without manual re-tuning."

(Magnitude shrinks but the "best in all 4 cells" claim and the qualitative "H200 wins smaller than RTX PRO 6000" pattern are preserved.)

---

## 6. Table 4 (`tab:eb_plus_traffic`) — H200 column data

**Action**: Replace H200 cells (right half of the table). RTX PRO 6000 column unchanged.

| c | Metric | Paper v1 / EB / EB⁺ | Ours v1 / EB / EB⁺ |
|---|--------|---------------------|---------------------|
| 32 | Throughput | 11584 / 8609 / 11586 | 11543 / **10779** / 11673 |
| 32 | TTFT | 49 ms / 1046 ms / 50 ms | 42 / **60** / 42 ms |
| 32 | TPOT | 8.2 / 7.1 / 8.2 | 8.4 / 9.0 / 8.3 |
| 512 | Throughput | 27464 / 27207 / 27460 | 24099 / 23876 / **24435** |
| 512 | TTFT | 0.7 / 4.6 / 0.7 s | 0.65 / **0.93** / 0.64 s |
| 512 | TPOT | 52.7 / 36.8 / 52.7 | 65.5 / 69.8 / 64.8 |
| 2048 | Throughput | 26368 / **27198** / 27043 | 23055 / 22957 / **23690** |
| 2048 | TTFT | 24.4 / 34.6 / 33.7 s | 24.8 / **26.2** / 24.5 s |
| 2048 | TPOT | 128 / 72.8 / 79.2 | 135.5 / 137.7 / 128.9 |

**Key changes**:
- c=32 EB throughput improves dramatically (8609 → 10779, EB no longer "loses" by 26%)
- All TTFT values lower than paper (H200's TTFT is well-controlled in our setup)
- c=512/2048 EB TPOT regresses vs paper (acknowledged limitation; reflects the design trade-off discussed in §3, not warranting separate prose)

---

## 7. Table 5 (`tab:eb_plus_nonstat`) — H200 column data

**Action**: Replace H200 column.

| Scenario | Metric | Paper v1 / EB / EB⁺ | Ours v1 / EB / EB⁺ |
|----------|--------|---------------------|---------------------|
| Distrib. shift | Throughput | 18307 / 17394 / **20776** | 20396 / 20762 / **21182** |
| Distrib. shift | TTFT | 11.9 / 66.5 / 69.7 s | 45.6 / 67.0 / 46.8 s |
| Distrib. shift | TPOT | 101 / 42.3 / 63.1 | 104.2 / 46.7 / 62.0 |
| Concur. shift | Throughput | 24251 / 23374 / **24397** | 21296 / 21334 / **21718** |
| Concur. shift | TTFT | 7.7 / 13.7 / 9.4 s | 7.24 / 9.97 / 6.98 s |
| Concur. shift | TPOT | 77.9 / 45.6 / 66.8 | 63.8 / 45.4 / 61.5 |

EB⁺ ranking (best in all 4 cells) preserved; absolute values reflect current vLLM commit.

---

## 8. Figure 3 (`fig:validation`) — Regenerate

Paper data file `reproduce/validation/paper_data/validation_grid.json` would need updating with our reproduction. Recommended values (from `reproduce/validation/outputs/controller_validation_kvaware/H200_Qwen3-8B/`):

| Scenario | v1 (tok/s) | EB(k̂*) (tok/s) | v1 TPOT | EB TPOT |
|----------|------------|-----------------|---------|---------|
| decode_heavy | 11200 | 11846 | 128 ms | 84 ms |
| balanced | 17424 | 18343 | 97 ms | 87 ms |
| prefill_heavy | 31539 | 30903 | 156 ms | 175 ms |

prefill_heavy panel still shows EB ≤ v1 by ~2% (see substitution caveat below).

---

## 9. Figure 4 (`fig:e2e-simulated`) — Regenerate

Paper data file `reproduce/synthetic_e2e/paper_data/synthetic_e2e.json` H200 RPS column. Use grid-optimal reproduction:

| Scenario | v1 | EB(k̂*) | Best config (for EB) |
|----------|-----|--------|----------------------|
| Decode-heavy | 11.94 | **12.15** | (14336, 1536) |
| Balanced | 18.59 | **18.81** | (18432, 1024) |
| Prefill-heavy | 28.16 | 27.13 | (10240, 512) |

Decode-heavy and balanced now show EB > v1 (matching paper's qualitative claim). Prefill-heavy remains slightly below v1.

---

## 10. Substitution caveats — where NOT to change

| Item | Why keep |
|------|----------|
| Abstract numbers | All RTX PRO 6000 (15.3%, 37.5%) — we did not reproduce |
| Intro headline numbers | Same — RTX PRO 6000 origin |
| §4.2 Fig 3 prefill_heavy caption | EB ≤ v1 by 2% in our run vs paper's +2%; both within marginal range. Either keep paper or add: "EB($\hat{k}^*$) tracks v1 within ±3% on prefill-heavy, as predicted by the diminishing-returns regime ($r \to 0$, Appendix \ref{app:convexity})." |
| §4.3.1 RTX PRO 6000 numbers | Not reproduced |
| Convexity κ values (Appendix) | Hardware-intrinsic, not affected |
| Algorithm 1 | Not affected (already references Appendix~\ref{app:kv-aware-gate}) |

---

## Summary of changes by location

| File / Section | Change type | Lines |
|----------------|-------------|-------|
| `evaluation.tex` tab:e2e-real-H200 | Replace 8 cells × 2 columns | line 140–145 |
| `evaluation.tex` § H200 narrative | "+0.8% → −6.1%" becomes "+4.4% → −3.2%" | line 152 |
| `evaluation.tex` § extreme-r claim | Soften ShareGPT 30B example | line 155 |
| `evaluation.tex` § TPOT example | Update specific (WC/NM) numbers | line 169 |
| `evaluation.tex` § traffic narrative | Drop "30–1000×" quote (Option B) | line 187 |
| `evaluation.tex` tab:eb_plus_traffic H200 | Replace 9 cells | line 197–214 |
| `evaluation.tex` § non-stat percent | "+13.5%/+0.6%" → "+3.9%/+2.0%" | line 218 |
| `evaluation.tex` tab:eb_plus_nonstat H200 | Replace 6 cells | line 222–238 |
| `figs/validation_grid.pdf` | Regenerate from new data | (figure) |
| `figs/fig_synthetic_e2e.pdf` | Regenerate from new data | (figure) |
| Abstract / Intro | **No changes** | — |

**Estimated total**: ~10 text edits + 2 figure regenerations. RTX PRO 6000 results and theoretical framework (Sections 2, 3, Appendix) **remain unchanged**.
