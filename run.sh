#!/bin/bash
# run.sh — tmux 多窗口挂载运行全部实验（4xGPU 并行版）
# Usage: bash run.sh
#
# GPU 分配:
#   GPU 0: train-cloud
#   GPU 1: train-edge     (与 cloud 并行训练)
#   GPU 2: exp1 + vis
#   GPU 3: exp2 + exp3
#
# 窗口列表:
#   0: train-cloud   — 训练 Cloud 模型 (GPU 0)
#   1: train-edge    — 训练 Edge 模型  (GPU 1, 与 cloud 同步启动)
#   2: exp1          — Experiment 1   (GPU 2, 等 checkpoint 就绪)
#   3: exp2          — Experiment 2   (GPU 3, 等 checkpoint 就绪)
#   4: exp3          — Experiment 3   (GPU 2, exp1 完成后)
#   5: vis           — 可视化         (GPU 3, exp2 完成后)
#   6: monitor       — GPU 监控

set -e

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

# ── 公共变量 ──────────────────────────────────────────────────
BASE_ENV="PY=$PY CKPT_DIR=$CKPT_DIR LOG_DIR=$LOG_DIR"

WAIT_BOTH="echo 'Waiting for both checkpoints...'; \
until [ -f \$CKPT_DIR/cloud_best.pt ] && [ -f \$CKPT_DIR/edge_best.pt ]; \
do sleep 30; done; echo 'Both checkpoints ready.'"

WAIT_EXP1="echo 'Waiting for exp1 to finish...'; \
until [ -f \$OUTPUT_DIR/experiment1_results.json ]; \
do sleep 60; done; echo 'exp1 done.'"

WAIT_EXP2="echo 'Waiting for exp2 to finish...'; \
until [ -f \$OUTPUT_DIR/sweep_t_star_results.json ]; \
do sleep 60; done; echo 'exp2 done.'"

# ── 窗口 0: train-cloud (GPU 0, batch_size=512) ───────────────
tmux new-session -d -s "$SESSION" -n "train-cloud"
tmux send-keys -t "$SESSION:train-cloud" \
    "export CUDA_VISIBLE_DEVICES=0 $BASE_ENV OUTPUT_DIR=results/" Enter
tmux send-keys -t "$SESSION:train-cloud" \
    "$PY train.py --mode cloud --epochs 100 --batch_size 512 --save_dir \$CKPT_DIR \
    2>&1 | tee \$LOG_DIR/train_cloud.log" Enter

# ── 窗口 1: train-edge (GPU 1, batch_size=512, 与 cloud 并行) ─
tmux new-window -t "$SESSION" -n "train-edge"
tmux send-keys -t "$SESSION:train-edge" \
    "export CUDA_VISIBLE_DEVICES=1 $BASE_ENV OUTPUT_DIR=results/" Enter
tmux send-keys -t "$SESSION:train-edge" \
    "$PY train.py --mode edge --epochs 100 --batch_size 512 --save_dir \$CKPT_DIR \
    2>&1 | tee \$LOG_DIR/train_edge.log" Enter

# ── 窗口 2: exp1 (GPU 2) ──────────────────────────────────────
tmux new-window -t "$SESSION" -n "exp1"
tmux send-keys -t "$SESSION:exp1" \
    "export CUDA_VISIBLE_DEVICES=2 $BASE_ENV OUTPUT_DIR=results/" Enter
tmux send-keys -t "$SESSION:exp1" \
    "$WAIT_BOTH && echo '=== Exp1: Direction Correction vs Baselines ===' && \
    $PY experiment.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
        --t_star 0.5 --num_samples 10000 --batch_size 512 \
        --output_dir \$OUTPUT_DIR \
    2>&1 | tee \$LOG_DIR/exp1.log && \
    echo '=== Exp3: Compression Sweep ===' && \
    $PY experiment.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
        --t_star 0.5 --num_samples 5000 --batch_size 512 \
        --output_dir \$OUTPUT_DIR --compression_sweep \
    2>&1 | tee \$LOG_DIR/exp3.log" Enter

# ── 窗口 3: exp2 + vis (GPU 3) ────────────────────────────────
tmux new-window -t "$SESSION" -n "exp2"
tmux send-keys -t "$SESSION:exp2" \
    "export CUDA_VISIBLE_DEVICES=3 $BASE_ENV OUTPUT_DIR=results/" Enter
tmux send-keys -t "$SESSION:exp2" \
    "$WAIT_BOTH && echo '=== Exp2: t* Sweep ===' && \
    $PY sweep_t_star.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
        --num_samples 5000 --batch_size 512 \
    2>&1 | tee \$LOG_DIR/exp2.log && \
    echo '=== Visualization ===' && \
    $PY visualize.py \
        --cloud_ckpt \$CKPT_DIR/cloud_best.pt \
        --edge_ckpt \$CKPT_DIR/edge_best.pt \
    2>&1 | tee \$LOG_DIR/vis.log" Enter

# ── 窗口 4: monitor ───────────────────────────────────────────
tmux new-window -t "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" "watch -n 2 nvidia-smi" Enter

# ── 切回第一个窗口 ────────────────────────────────────────────
tmux select-window -t "$SESSION:train-cloud"

echo "========== tmux session '$SESSION' started =========="
echo ""
echo "GPU 分配:"
echo "  GPU 0  train-cloud"
echo "  GPU 1  train-edge    (并行训练)"
echo "  GPU 2  exp1 → exp3"
echo "  GPU 3  exp2 → vis"
echo ""
echo "常用命令:"
echo "  tmux attach -t $SESSION       # 进入 session"
echo "  Ctrl+b, w                      # 窗口列表"
echo "  Ctrl+b, d                      # 后台挂起"
echo "  tmux kill-session -t $SESSION # 终止所有任务"
echo ""
echo "日志文件在 $LOG_DIR/ 目录下"
echo ""

tmux attach -t "$SESSION"
