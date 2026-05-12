#!/bin/bash
# 改进版 Hazard Rate 实验
# 主要改进:
# 1. 增加运行次数 (3 → 5)
# 2. 更细的 k* 分辨率 (step=8 而非 17)
# 3. 每个配置之间加入冷却时间
# 4. 添加更多的 warmup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/hazard_rate_ordering_v2_N256_O128}"
NUM_PROMPTS=3000
NUM_REPEATS=5  # 增加到 5 次
MAX_CONCURRENCY=3000
WARMUP_REQUESTS=200  # 增加 warmup
COOL_DOWN=30  # 配置之间的冷却时间 (秒)

# 更细的 k* sweep (step=8)
K_STAR_VALUES=(8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256)

# Gamma 配置
declare -A GAMMA_CONFIGS
GAMMA_CONFIGS["DFR"]="0.5 256"  # shape=0.5, scale=256, mean=128
GAMMA_CONFIGS["CFR"]="1.0 128"  # shape=1.0, scale=128, mean=128
GAMMA_CONFIGS["IFR"]="2.0 64"   # shape=2.0, scale=64, mean=128

mkdir -p "$OUTPUT_DIR"

# 保存配置
cat > "$OUTPUT_DIR/experiment_config_v2.json" << EOF
{
    "experiment": "hazard_rate_ordering_v2",
    "improvements": [
        "增加运行次数到 5",
        "更细的 k* 分辨率 (step=8)",
        "增加 warmup 到 200 请求",
        "配置之间加入冷却时间"
    ],
    "N": 256,
    "E_O": 128,
    "num_prompts": $NUM_PROMPTS,
    "num_repeats": $NUM_REPEATS,
    "warmup_requests": $WARMUP_REQUESTS,
    "cool_down_seconds": $COOL_DOWN,
    "k_star_values": [$(IFS=,; echo "${K_STAR_VALUES[*]}")],
    "gamma_configs": {
        "DFR": {"shape": 0.5, "scale": 256},
        "CFR": {"shape": 1.0, "scale": 128},
        "IFR": {"shape": 2.0, "scale": 64}
    }
}
EOF

echo "实验配置已保存到 $OUTPUT_DIR/experiment_config_v2.json"
echo "改进点:"
echo "  - 运行次数: $NUM_REPEATS"
echo "  - k* 步长: 8 (共 ${#K_STAR_VALUES[@]} 个点)"
echo "  - Warmup: $WARMUP_REQUESTS 请求"
echo "  - 冷却时间: ${COOL_DOWN}s"
echo ""
echo "总实验数: $((3 * ${#K_STAR_VALUES[@]} * NUM_REPEATS)) 次运行"
echo ""

# 运行实验的函数
run_experiment() {
    local hazard_type=$1
    local shape=$2
    local scale=$3
    local k_star=$4
    local run_id=$5

    local config_dir="$OUTPUT_DIR/${hazard_type}_shape${shape}"
    mkdir -p "$config_dir"

    echo "[$(date '+%H:%M:%S')] Running $hazard_type k*=$k_star run$run_id"

    # 这里添加实际的实验命令
    # python pd_exp/syn/run_single_experiment.py \
    #     --output-dir "$config_dir" \
    #     --gamma-shape $shape \
    #     --gamma-scale $scale \
    #     --k-star $k_star \
    #     --run-id $run_id \
    #     --num-prompts $NUM_PROMPTS \
    #     --warmup-requests $WARMUP_REQUESTS
}

echo "提示: 此脚本仅生成配置。实际运行需要集成到现有的实验框架中。"
echo ""
echo "建议的实验流程:"
echo "1. 对每个 hazard type (DFR, CFR, IFR):"
echo "   a. 启动服务器"
echo "   b. 运行 warmup ($WARMUP_REQUESTS 请求)"
echo "   c. 对每个 k* 值运行 $NUM_REPEATS 次"
echo "   d. 每个配置之间等待 ${COOL_DOWN}s 冷却"
echo ""
echo "2. 分析时使用 trimmed mean 或 median 去除异常值"
