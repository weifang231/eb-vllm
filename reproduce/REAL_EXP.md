## 只运行ifr(ours)
```bash
SCHEDULERS=pd_ifr ./pd_exp/run_all_experiments.sh Qwen/Qwen3-8B 4
```
如果需要重复运行，可以加上VERSION=1
```bash
VERSION=1 SCHEDULERS=pd_ifr SKIP_EXISTING=1 ./pd_exp/run_all_experiments.sh Qwen/Qwen3-8B 4
```
## 一键运行所有实验

```bash
# 运行所有实验 (校准 + 数据集导出 + 4个实验)
./pd_exp/run_all_experiments.sh Qwen/Qwen3-8B 4
./pd_exp/run_all_experiments.sh Qwen/Qwen3-30B-A3B 4 # ongoing
DTYPE=bfloat16 ./pd_exp/run_all_experiments.sh google/gemma-3-1b-it 4 # ongoing
./pd_exp/run_all_experiments.sh openai/gpt-oss-20b 5 # 
DTYPE=bfloat16 EXPERIMENTS=wildchat ./pd_exp/run_all_experiments.sh openai/gpt-oss-120b 5 # 

# 可选参数
SKIP_CALIBRATION=true ./pd_exp/run_all_experiments.sh ...  # 跳过校准
SKIP_EXPORT=true ./pd_exp/run_all_experiments.sh ...       # 跳过数据集导出
EXPERIMENTS="sharegpt numina_math" ./pd_exp/run_all_experiments.sh ...  # 只跑指定实验
```

## 分析所有实验结果

```bash
python pd_exp/generate_summary.py
```

输出报告保存到 `pd_exp/eoutputs/evaluation/report_xxx.md`

---

## 0. 硬件校准 (必须先运行)

PD Scheduler 需要硬件校准参数才能准确调度。每个模型需要单独校准一次。

```bash
# 运行校准 (校准文件自动按模型命名: pd_calibration_<model_short>.json)
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
# -> 保存到 pd_exp/outputs/pd_calibration_Qwen3-8B.json

# 如果跑其他模型，需要单独校准
python -m vllm.v1.core.sched.calibration --model meta-llama/Llama-3.1-8B
# -> 保存到 pd_exp/outputs/pd_calibration_Llama-3.1-8B.json

# 实验脚本会自动查找对应模型的校准文件
# 也可以手动指定: VLLM_PD_CALIBRATION_FILE=/path/to/file.json
```

---

## ShareGPT

```shell
# A6000完成 baseline, pd_naive
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python pd_exp/export_dataset.py \
      --dataset sharegpt \
      --model Qwen/Qwen3-8B \
      --num-samples 4000 \
      --output ./pd_exp/outputs/sharegpt_prompts.jsonl
rm -rf ShareGPT_V3_unfiltered_cleaned_split.json

# ShareGPT: balanced workload, 关闭 thinking，输出长度 500
ENABLE_THINKING=false CUSTOM_OUTPUT_LEN=500 \
    ./pd_exp/real/run_grid_search.sh ./pd_exp/outputs/sharegpt_prompts.jsonl 4

# Grid search 结果分析 (目录名包含模型短名)
python pd_exp/real/analyze_grid_search.py pd_exp/outputs/grid_search_sharegpt_prompts_Qwen3-8B_Con_2048_Prompts_4000

# Input/Output 长度统计
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_sharegpt_prompts_Qwen3-8B_Con_2048_Prompts_4000 --summary-only
```

## numina_math

```bash
# A6000完成baseline, pd_naive
python pd_exp/export_dataset.py \
    --dataset numina_math \
    --model Qwen/Qwen3-8B \
    --num-samples 4000 \
    --min-output-len 800 \
    --output ./pd_exp/outputs/numina_math_prompts.jsonl

# numina_math: 开启 thinking (默认)，输出长度 4000
CUSTOM_OUTPUT_LEN=4000 \
    ./pd_exp/real/run_grid_search.sh ./pd_exp/outputs/numina_math_prompts.jsonl 4

# Grid search 结果分析
python pd_exp/real/analyze_grid_search.py pd_exp/outputs/grid_search_numina_math_prompts_Qwen3-8B_Con_2048_Prompts_4000

# Input/Output 长度统计 (查看真实的 decode-heavy 程度)
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_numina_math_prompts_Qwen3-8B_Con_2048_Prompts_4000 --summary-only
```

## longbench

```bash
# A6000完成 - baseline, pd_ratio
python pd_exp/export_dataset.py \
      --dataset longbench \
      --model Qwen/Qwen3-8B \
      --num-samples 4000 \
      --min-input-len 1000 \
      --max-input-len 4000 \
      --output ./pd_exp/outputs/longbench_prefill.jsonl

# longbench: prefill-heavy, 关闭 thinking，输出长度 20
ENABLE_THINKING=false CUSTOM_OUTPUT_LEN=20 \
    ./pd_exp/real/run_grid_search.sh ./pd_exp/outputs/longbench_prefill.jsonl 4

# Grid search 结果分析
python pd_exp/real/analyze_grid_search.py pd_exp/outputs/grid_search_longbench_prefill_Qwen3-8B_Con_2048_Prompts_4000

# Input/Output 长度统计 (确认是否 prefill-heavy)
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_longbench_prefill_Qwen3-8B_Con_2048_Prompts_4000 --summary-only
```

## WildChat (Prefix Cache Testing)

多轮对话场景，用于测试 prefix cache 效果。对比 baseline、pd_ratio、pd_ifr 三种 scheduler。

```bash
# A6000完成 - baseline, pd_ratio
# 导出多轮对话数据 (筛选至少 8 轮的对话)
python pd_exp/multiturn/export_dataset.py \
    --dataset wildchat \
    --model Qwen/Qwen3-8B \
    --num-conversations 3000 \
    --min-turns 6 \
    --output ./pd_exp/outputs/wildchat_multiturn.json

# 运行实验 (对比三种 scheduler)
./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4

# 结果分析 (scheduler 对比)
python pd_exp/multiturn/analyze_results.py pd_exp/outputs/multiturn_wildchat_multiturn_Qwen3-8B_Clients_8_MaxTurns_10
```

### 脚本参数 (环境变量)

```bash
# 自定义参数运行
NUM_CLIENTS=16 MAX_TURNS=12 LIMIT_MAX_TOKENS=512 K_RATIO=0.7 \
BS_VALUES="512 1024" TB_VALUES="8192 16384" \
    ./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4

# 恢复中断的实验 (从队列文件继续)
RESUME=true ./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4
```

- `NUM_CLIENTS`: 并发客户端数 (默认 8)
- `MAX_TURNS`: 每个对话最多执行的轮数 (默认 10)
- `LIMIT_MAX_TOKENS`: 每轮最大输出 token 数 (默认 256)
- `K_RATIO`: pd_ratio scheduler 的 θ* 值 (默认 0.8)
- `BS_VALUES`: 测试的 batch size 列表 (默认 "256 512 1024 1536 2048")
- `TB_VALUES`: 测试的 max_num_batched_tokens 列表 (默认 "4096 8192 10240 14336 16384 18432")
- `RESUME`: 设为 true 时从现有队列恢复实验 (默认 false)

### 导出参数

- `--min-turns`: 最小对话轮数，筛选多轮对话 (默认 8)
- `--num-conversations`: 导出的对话数量

### Scheduler 对比

每个 (TB, BS) 配置会测试三种 scheduler:

- **baseline**: vLLM 默认调度器
- **pd_ratio**: PD scheduler，固定 θ*=K_RATIO
- **pd_ifr**: PD scheduler，IFR 模式 (基于 hazard rate 自适应 θ*)

### Prefix Cache 工作原理

对于一个 n 轮对话：

- 第 1 轮: 0% cached (新对话)
- 第 2 轮: ~50% cached (复用第 1 轮历史)
- 第 3 轮: ~67% cached (复用第 1-2 轮历史)
- 第 n 轮: ~(n-1)/n cached

Benchmark 输出的 `approx_cached_percent` 指标反映了 prefix cache 的实际效果。