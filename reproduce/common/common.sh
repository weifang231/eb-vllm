#!/bin/bash
# 公共函数库 - 被实验脚本 source 使用

GPU_MEM_THRESHOLD=${GPU_MEM_THRESHOLD:-10000}

# 选择要使用的 GPU
# 参数: $1 = MAX_GPUS
# 环境变量: GPUS = 指定GPU列表 (如 "0,1,2")
# 输出: 设置 GPUS_TO_USE 数组
select_gpus() {
    local max_gpus=${1:-4}

    if [ -n "${GPUS:-}" ]; then
        IFS=',' read -ra GPUS_TO_USE <<< "$GPUS"
        echo "使用指定的 GPU: ${GPUS_TO_USE[*]}"
        return
    fi

    # 自动检测可用 GPU (内存使用低于阈值)
    local available=()
    while IFS=, read -r gpu_id _ _ mem_used; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' MiB')
        [ "$mem_used" -lt "$GPU_MEM_THRESHOLD" ] && available+=("$gpu_id")
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null)

    [ ${#available[@]} -eq 0 ] && { echo "错误: 没有可用的 GPU"; exit 1; }

    GPUS_TO_USE=("${available[@]:0:$max_gpus}")
    echo "使用 ${#GPUS_TO_USE[@]} 张 GPU: ${GPUS_TO_USE[*]}"
}

# 检查端口是否可用 (只检查 LISTEN 状态，避免 TIME_WAIT/ESTABLISHED 误判)
check_port_available() {
    local port=$1 gpu_id=${2:-""}
    if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[GPU $gpu_id] 端口 $port 被占用"
        return 1
    fi
}

# 使用 Python 真正尝试 bind 来判断端口是否可用（不依赖 lsof 权限）
port_is_free() {
    local port=$1
    python - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

# 从起始端口开始查找可用端口
find_free_port() {
    local start_port=$1
    local max_tries=${2:-50}
    local port=$start_port
    local i=0
    while [ $i -lt $max_tries ]; do
        if port_is_free "$port"; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
        i=$((i + 1))
    done
    return 1
}

# 等待 GPU 内存释放
wait_for_gpu_memory() {
    local gpu_id=$1 max_wait=${2:-30} i=0
    while [ $i -lt $max_wait ]; do
        local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
        [ -n "$mem" ] && [ "$mem" -lt "$GPU_MEM_THRESHOLD" ] && return 0
        sleep 1; ((i++))
    done
    return 1
}

# 等待服务启动
wait_for_server() {
    local port=$1 server_pid=$2 max_wait=${3:-180} log_file=${4:-""} i=0
    while [ $i -lt $max_wait ]; do
        curl -s "http://localhost:${port}/health" >/dev/null 2>&1 && return 0
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "服务进程意外退出"
            [ -n "$log_file" ] && [ -f "$log_file" ] && sed 's/^/  /' "$log_file"
            return 1
        fi
        sleep 1; ((i++))
    done
    echo "服务启动超时 (${max_wait}s)"
    [ -n "$log_file" ] && [ -f "$log_file" ] && sed 's/^/  /' "$log_file"
    return 1
}

# 终止服务进程
kill_server() {
    local server_pid=$1 gpu_id=${2:-""}
    local max_wait=${3:-15}  # 默认等待 15 秒让服务优雅退出

    # 优雅终止 - 给服务足够时间保存 stats 等清理工作
    kill -TERM $server_pid 2>/dev/null || true

    # 等待进程自然退出
    local i=0
    while [ $i -lt $max_wait ]; do
        if ! kill -0 $server_pid 2>/dev/null; then
            # 进程已退出
            break
        fi
        sleep 1
        ((i++))
    done

    # 如果进程还在运行，强制终止
    if kill -0 $server_pid 2>/dev/null; then
        echo "进程 $server_pid 未响应 SIGTERM，强制终止..."
        kill -9 $server_pid 2>/dev/null || true
        # 杀死服务进程的子进程
        pkill -9 -P $server_pid 2>/dev/null || true
    fi

    # 清理GPU上的残留进程
    if [ -n "$gpu_id" ]; then
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu_id 2>/dev/null | tr -d ' '); do
            [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
        done
        wait_for_gpu_memory "$gpu_id" 30 || true
    else
        sleep 2
    fi
}

# 从队列获取下一个实验 (带锁)
get_next_experiment() {
    local queue_file=$1 lock_file=$2
    (
        flock -x 200
        local exp=$(head -n 1 "$queue_file" 2>/dev/null)
        if [ -n "$exp" ]; then
            tail -n +2 "$queue_file" > "${queue_file}.tmp"
            mv "${queue_file}.tmp" "$queue_file"
            echo "$exp"
        fi
    ) 200>"$lock_file"
}

# 更新进度
update_progress() {
    local status=$1 progress_file=$2 lock_file=$3 total=$4
    (
        flock -x 200
        echo "$status" >> "$progress_file"
        local done=$(wc -l < "$progress_file")
        echo "进度: $done / $total"
    ) 200>"$lock_file"
}

# 确保硬件校准文件存在，不存在则自动运行校准
# 用法: ensure_calibration <model> <model_short>
# 结果: 设置 VLLM_PD_CALIBRATION_FILE 环境变量
ensure_calibration() {
    local model=$1 model_short=$2
    local _common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local calibration_dir="${_common_dir}/outputs"

    if [ -n "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
        if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
            return 0
        else
            echo "错误: 指定的校准文件不存在: $VLLM_PD_CALIBRATION_FILE"
            return 1
        fi
    fi

    local calibration_file="${calibration_dir}/pd_calibration_${model_short}.json"
    if [ -f "$calibration_file" ]; then
        export VLLM_PD_CALIBRATION_FILE="$calibration_file"
        return 0
    fi

    echo "未找到校准文件，自动运行硬件校准..."
    mkdir -p "$calibration_dir"
    python3 -m vllm.v1.core.sched.calibration \
        --model "$model" \
        --output "$calibration_file" || return 1
    export VLLM_PD_CALIBRATION_FILE="$calibration_file"
    echo "校准完成: $calibration_file"
}

# 初始化实验环境
init_experiment_env() {
    # common.sh is at reproduce/common/common.sh; repo root is ../.. (eb-vllm/).
    local _common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local venv_path=${1:-"${_common_dir}/../../.venv"}
    ulimit -n 65535 2>/dev/null || true
    if [ -f "${venv_path}/bin/activate" ]; then
        source "${venv_path}/bin/activate"
    elif ! command -v vllm >/dev/null 2>&1; then
        echo "[WARN] No venv found at ${venv_path}/bin/activate and 'vllm' not in PATH." >&2
        echo "       Either activate a venv with vllm installed, or symlink it to:" >&2
        echo "         ln -s /path/to/your/.venv ${_common_dir}/../../.venv" >&2
    fi
}

# 打印实验结果统计
print_summary() {
    local progress_file=$1 total=$2 output_dir=$3
    local done=$(wc -l < "$progress_file" 2>/dev/null || echo 0)
    local ok=$(grep -c "^OK|" "$progress_file" 2>/dev/null || echo 0)
    local fail=$(grep -c "^FAIL|" "$progress_file" 2>/dev/null || echo 0)

    echo ""
    echo "========================================"
    echo "实验完成: $done / $total (成功: $ok, 失败: $fail)"
    echo "结果目录: $output_dir"
    echo "========================================"
}
