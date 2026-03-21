#!/bin/bash
# run.sh — tmux 多窗口挂载运行全部实验
# Usage: bash run.sh [GPU_ID]
#   GPU_ID: 指定使用的 GPU，默认 0
#
# 会创建一个名为 ecf 的 tmux session，包含以下窗口：
#   0: train-cloud   — 训练 Cloud 模型
#   1: train-edge    — 训练 Edge 模型（cloud 完成后自动启动）
#   2: exp1          — Experiment 1: 方法对比
#   3: exp2          — Experiment 2: t* 扫描
#   4: exp3          — Experiment 3: 压缩率扫描
#   5: vis           — 可视化
#   6: monitor       — GPU 监控
#
# 实验窗口会等待 checkpoint 就绪后自动启动

set -e

GPU=${1:-0}
SESSION="ecf"
CKPT_DIR="checkpoints"
LOG_DIR="logs"

# ── 检查 tmux ──────────────────────────────────────────────────
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux not found. Install with: sudo apt install tmux"
    exit 1
fi

# ── 检查 python ────────────────────────────────────────────────
if command -v python3 &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null; then
    PY=python
else
    echo "Error: python not found"
    exit 1
fi

echo "Using: $PY ($($PY --version 2>&1))"
echo "GPU: $GPU"
echo "Session: $SESSION"
echo ""

# ── 准备目录 ───────────────────────────────────────────────────
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── 安装依赖（前台完成再起 tmux）─────────────────────────────
echo "========== Installing dependencies =========="
$PY -m pip install -r requirements.txt -q
echo "Done."
echo ""

# ── 如果同名 session 已存在则先杀掉 ──────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── 公共环境变量前缀 ──────────────────────────────────────────
ENV="export CUDA_VISIBLE_DEVICES=$GPU PY=$PY CKPT_DIR=$CKPT_DIR LOG_DIR=$LOG_DIR"

# 等待 checkpoint 就绪的辅助函数（写成 shell 片段，嵌入各窗口）
WAIT_CLOUD="echo 'Waiting for cloud checkpoint...'; \
until [ -f \$CKPT_DIR/cloud_best.pt ]; do sleep 30; done; \
echo 'cloud_best.pt ready.'"

WAIT_BOTH="echo 'Waiting for both checkpoints...'; \
until [ -f \$CKPT_DIR/cloud_best.pt ] && [ -f \$CKPT_DIR/edge_best.pt ]; do sleep 30; done; \
echo 'Both checkpoints ready.'"

# ── 创建 session，第一个窗口: train-cloud ─────────────────────
tmux new-session -d -s "$SESSION" -n "train-cloud"
tmux send-keys -t "$SESSION:train-cloud" "$ENV" Enter
tmux send-keys -t "$SESSION:train-cloud" \
    "$PY train.py --mode cloud --epochs 100 --save_dir \$CKPT_DIR 2>&1 | tee \$LOG_DIR/train_cloud.log" Enter

# ── 窗口 1: train-edge（等 cloud 完成后启动）─────────────────
tmux new-window -t "$SESSION" -n "train-edge"
tmux send-keys -t "$SESSION:train-edge" "$ENV" Enter
tmux send-keys -t "$SESSION:train-edge" \
    "$WAIT_CLOUD && $PY train.py --mode edge --epochs 100 --save_dir \$CKPT_DIR 2>&1 | tee \$LOG_DIR/train_edge.log" Enter

# ── 窗口 2: exp1 ──────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "exp1"
tmux send-keys -t "$SESSION:exp1" "$ENV" Enter
tmux send-keys -t "$SESSION:exp1" \
    "$WAIT_BOTH && echo '=== Exp1: Direction Correction vs Baselines ===' && \
    $PY experiment.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
        --t_star 0.5 --num_samples 10000 \
    2>&1 | tee \$LOG_DIR/exp1.log" Enter

# ── 窗口 3: exp2 ──────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "exp2"
tmux send-keys -t "$SESSION:exp2" "$ENV" Enter
tmux send-keys -t "$SESSION:exp2" \
    "$WAIT_BOTH && echo '=== Exp2: t* Sweep ===' && \
    $PY sweep_t_star.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
        --num_samples 5000 \
    2>&1 | tee \$LOG_DIR/exp2.log" Enter

# ── 窗口 4: exp3 ──────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "exp3"
tmux send-keys -t "$SESSION:exp3" "$ENV" Enter
tmux send-keys -t "$SESSION:exp3" \
    "$WAIT_BOTH && echo '=== Exp3: Compression Sweep ===' && \
    $PY experiment.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
        --t_star 0.5 --num_samples 5000 --compression_sweep \
    2>&1 | tee \$LOG_DIR/exp3.log" Enter

# ── 窗口 5: vis ───────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "vis"
tmux send-keys -t "$SESSION:vis" "$ENV" Enter
tmux send-keys -t "$SESSION:vis" \
    "$WAIT_BOTH && echo '=== Visualization ===' && \
    $PY visualize.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
    2>&1 | tee \$LOG_DIR/vis.log" Enter

# ── 窗口 6: monitor ───────────────────────────────────────────
tmux new-window -t "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" "watch -n 2 nvidia-smi" Enter

# ── 切回第一个窗口 ────────────────────────────────────────────
tmux select-window -t "$SESSION:train-cloud"

echo "========== tmux session '$SESSION' started =========="
echo ""
echo "常用命令:"
echo "  tmux attach -t $SESSION          # 进入 session"
echo "  Ctrl+b, w                         # 窗口列表"
echo "  Ctrl+b, 0~6                       # 切换到指定窗口"
echo "  Ctrl+b, d                         # 后台挂起（detach）"
echo "  tmux kill-session -t $SESSION    # 终止所有任务"
echo ""
echo "日志文件在 $LOG_DIR/ 目录下"
echo ""

# 自动 attach（可注释掉让脚本直接返回）
tmux attach -t "$SESSION"
