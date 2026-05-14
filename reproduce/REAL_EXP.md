## Run only ifr (ours)
```bash
SCHEDULERS=pd_ifr ./pd_exp/run_all_experiments.sh Qwen/Qwen3-8B 4
```
For repeated runs, add VERSION=1
```bash
VERSION=1 SCHEDULERS=pd_ifr SKIP_EXISTING=1 ./pd_exp/run_all_experiments.sh Qwen/Qwen3-8B 4
```
## Run all experiments in one command

```bash
# Run all experiments (calibration + dataset export + 4 experiments)
./pd_exp/run_all_experiments.sh Qwen/Qwen3-8B 4
./pd_exp/run_all_experiments.sh Qwen/Qwen3-30B-A3B 4 # ongoing
DTYPE=bfloat16 ./pd_exp/run_all_experiments.sh google/gemma-3-1b-it 4 # ongoing
./pd_exp/run_all_experiments.sh openai/gpt-oss-20b 5 # 
DTYPE=bfloat16 EXPERIMENTS=wildchat ./pd_exp/run_all_experiments.sh openai/gpt-oss-120b 5 # 

# Optional arguments
SKIP_CALIBRATION=true ./pd_exp/run_all_experiments.sh ...  # skip calibration
SKIP_EXPORT=true ./pd_exp/run_all_experiments.sh ...       # skip dataset export
EXPERIMENTS="sharegpt numina_math" ./pd_exp/run_all_experiments.sh ...  # only run specified experiments
```

## Analyze all experiment results

```bash
python pd_exp/generate_summary.py
```

Output report is saved to `pd_exp/eoutputs/evaluation/report_xxx.md`

---

## 0. Hardware calibration (must be run first)

PD Scheduler requires hardware calibration parameters for accurate scheduling. Each model needs to be calibrated once.

```bash
# Run calibration (calibration file is named automatically by model: pd_calibration_<model_short>.json)
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
# -> saved to pd_exp/outputs/pd_calibration_Qwen3-8B.json

# For other models, calibrate separately
python -m vllm.v1.core.sched.calibration --model meta-llama/Llama-3.1-8B
# -> saved to pd_exp/outputs/pd_calibration_Llama-3.1-8B.json

# The experiment script will automatically look up the calibration file for the model
# You can also specify manually: VLLM_PD_CALIBRATION_FILE=/path/to/file.json
```

---

## ShareGPT

```shell
# A6000 done: baseline, pd_naive
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python pd_exp/export_dataset.py \
      --dataset sharegpt \
      --model Qwen/Qwen3-8B \
      --num-samples 4000 \
      --output ./pd_exp/outputs/sharegpt_prompts.jsonl
rm -rf ShareGPT_V3_unfiltered_cleaned_split.json

# ShareGPT: balanced workload, thinking disabled, output length 500
ENABLE_THINKING=false CUSTOM_OUTPUT_LEN=500 \
    ./pd_exp/real/run_grid_search.sh ./pd_exp/outputs/sharegpt_prompts.jsonl 4

# Grid search result analysis (directory name contains model short name)
python pd_exp/real/analyze_grid_search.py pd_exp/outputs/grid_search_sharegpt_prompts_Qwen3-8B_Con_2048_Prompts_4000

# Input/Output length stats
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_sharegpt_prompts_Qwen3-8B_Con_2048_Prompts_4000 --summary-only
```

## numina_math

```bash
# A6000 done: baseline, pd_naive
python pd_exp/export_dataset.py \
    --dataset numina_math \
    --model Qwen/Qwen3-8B \
    --num-samples 4000 \
    --min-output-len 800 \
    --output ./pd_exp/outputs/numina_math_prompts.jsonl

# numina_math: thinking enabled (default), output length 4000
CUSTOM_OUTPUT_LEN=4000 \
    ./pd_exp/real/run_grid_search.sh ./pd_exp/outputs/numina_math_prompts.jsonl 4

# Grid search result analysis
python pd_exp/real/analyze_grid_search.py pd_exp/outputs/grid_search_numina_math_prompts_Qwen3-8B_Con_2048_Prompts_4000

# Input/Output length stats (check how decode-heavy it really is)
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_numina_math_prompts_Qwen3-8B_Con_2048_Prompts_4000 --summary-only
```

## longbench

```bash
# A6000 done - baseline, pd_ratio
python pd_exp/export_dataset.py \
      --dataset longbench \
      --model Qwen/Qwen3-8B \
      --num-samples 4000 \
      --min-input-len 1000 \
      --max-input-len 4000 \
      --output ./pd_exp/outputs/longbench_prefill.jsonl

# longbench: prefill-heavy, thinking disabled, output length 20
ENABLE_THINKING=false CUSTOM_OUTPUT_LEN=20 \
    ./pd_exp/real/run_grid_search.sh ./pd_exp/outputs/longbench_prefill.jsonl 4

# Grid search result analysis
python pd_exp/real/analyze_grid_search.py pd_exp/outputs/grid_search_longbench_prefill_Qwen3-8B_Con_2048_Prompts_4000

# Input/Output length stats (confirm prefill-heavy)
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_longbench_prefill_Qwen3-8B_Con_2048_Prompts_4000 --summary-only
```

## WildChat (Prefix Cache Testing)

Multi-turn conversation workload for testing prefix cache effectiveness. Compares baseline, pd_ratio, and pd_ifr schedulers.

```bash
# A6000 done - baseline, pd_ratio
# Export multi-turn conversation data (filter to conversations with at least 8 turns)
python pd_exp/multiturn/export_dataset.py \
    --dataset wildchat \
    --model Qwen/Qwen3-8B \
    --num-conversations 3000 \
    --min-turns 6 \
    --output ./pd_exp/outputs/wildchat_multiturn.json

# Run experiment (compare three schedulers)
./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4

# Result analysis (scheduler comparison)
python pd_exp/multiturn/analyze_results.py pd_exp/outputs/multiturn_wildchat_multiturn_Qwen3-8B_Clients_8_MaxTurns_10
```

### Script arguments (environment variables)

```bash
# Custom arguments
NUM_CLIENTS=16 MAX_TURNS=12 LIMIT_MAX_TOKENS=512 K_RATIO=0.7 \
BS_VALUES="512 1024" TB_VALUES="8192 16384" \
    ./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4

# Resume interrupted experiment (continue from queue file)
RESUME=true ./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4
```

- `NUM_CLIENTS`: number of concurrent clients (default 8)
- `MAX_TURNS`: max turns to execute per conversation (default 10)
- `LIMIT_MAX_TOKENS`: max output tokens per turn (default 256)
- `K_RATIO`: θ* value for pd_ratio scheduler (default 0.8)
- `BS_VALUES`: list of batch sizes to test (default "256 512 1024 1536 2048")
- `TB_VALUES`: list of max_num_batched_tokens to test (default "4096 8192 10240 14336 16384 18432")
- `RESUME`: set to true to resume experiment from existing queue (default false)

### Export arguments

- `--min-turns`: minimum number of turns; filters multi-turn conversations (default 8)
- `--num-conversations`: number of conversations to export

### Scheduler comparison

Each (TB, BS) configuration tests three schedulers:

- **baseline**: vLLM default scheduler
- **pd_ratio**: PD scheduler, fixed θ*=K_RATIO
- **pd_ifr**: PD scheduler, IFR mode (adaptive θ* based on hazard rate)

### How prefix cache works

For an n-turn conversation:

- Turn 1: 0% cached (new conversation)
- Turn 2: ~50% cached (reuses turn 1 history)
- Turn 3: ~67% cached (reuses turns 1-2 history)
- Turn n: ~(n-1)/n cached

The `approx_cached_percent` metric in the benchmark output reflects the actual effectiveness of the prefix cache.