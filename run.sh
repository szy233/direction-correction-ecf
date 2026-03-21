#!/bin/bash
# run.sh — tmux 多窗口挂载运行全部实验（4xGPU 并行版）
# Usage: bash run.sh
#
# GPU 分配:
#   GPU 0: train-cloud
#   GPU 1: train-edge     (与 cloud 并行训练)
#   GPU 2: exp1 → exp3
#   GPU 3: exp2 → vis
#
# 窗口列表:
#   0: train-cloud   — 训练 Cloud 模型 (GPU 0)
#   1: train-edge    — 训练 Edge 模型  (GPU 1)
#   2: exp1+exp3     — Experiment 1&3  (GPU 2)
#   3: exp2+vis      — Experiment 2 & 可视化 (GPU 3)
#   4: monitor       — GPU 监控

SESSION="ecf"
CKPT_DIR="$(pwd)/checkpoints"
LOG_DIR="$(pwd)/logs"
OUTPUT_DIR="$(pwd)/results"
WORK_DIR="$(pwd)"

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
echo "Work dir: $WORK_DIR"
echo ""

# ── 准备目录 ───────────────────────────────────────────────────
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# ── 安装依赖（前台完成再起 tmux）─────────────────────────────
echo "========== Installing dependencies =========="
$PY -m pip install -r requirements.txt -q
echo "Done."
echo ""

# ── 如果同名 session 已存在则先杀掉 ──────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── 写任务脚本到 logs/，避免 tmux 内联命令引号地狱 ───────────
cat > "$LOG_DIR/task_train_cloud.sh" << SCRIPT
#!/bin/bash
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0
echo "=== Train Cloud (GPU 0) ==="
$PY train.py --mode cloud --epochs 100 --batch_size 512 --save_dir "$CKPT_DIR" \
    2>&1 | tee "$LOG_DIR/train_cloud.log"
echo "=== DONE ==="
exec bash
SCRIPT

cat > "$LOG_DIR/task_train_edge.sh" << SCRIPT
#!/bin/bash
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=1
echo "=== Train Edge (GPU 1) ==="
$PY train.py --mode edge --epochs 100 --batch_size 512 --save_dir "$CKPT_DIR" \
    2>&1 | tee "$LOG_DIR/train_edge.log"
echo "=== DONE ==="
exec bash
SCRIPT

cat > "$LOG_DIR/task_exp1_exp3.sh" << SCRIPT
#!/bin/bash
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=2
echo "Waiting for both checkpoints..."
until [ -f "$CKPT_DIR/cloud_best.pt" ] && [ -f "$CKPT_DIR/edge_best.pt" ]; do
    sleep 30
done
echo "Checkpoints ready."

echo "=== Exp1: Direction Correction vs Baselines (GPU 2) ==="
$PY experiment.py \
    --cloud_ckpt "$CKPT_DIR/cloud_best.pt" \
    --edge_ckpt  "$CKPT_DIR/edge_best.pt" \
    --t_star 0.5 --num_samples 10000 --batch_size 512 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/exp1.log"

echo "=== Exp3: Compression Sweep (GPU 2) ==="
$PY experiment.py \
    --cloud_ckpt "$CKPT_DIR/cloud_best.pt" \
    --edge_ckpt  "$CKPT_DIR/edge_best.pt" \
    --t_star 0.5 --num_samples 5000 --batch_size 512 \
    --output_dir "$OUTPUT_DIR" --compression_sweep \
    2>&1 | tee "$LOG_DIR/exp3.log"

echo "=== ALL DONE (GPU 2) ==="
exec bash
SCRIPT

cat > "$LOG_DIR/task_exp2_vis.sh" << SCRIPT
#!/bin/bash
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=3
echo "Waiting for both checkpoints..."
until [ -f "$CKPT_DIR/cloud_best.pt" ] && [ -f "$CKPT_DIR/edge_best.pt" ]; do
    sleep 30
done
echo "Checkpoints ready."

echo "=== Exp2: t* Sweep (GPU 3) ==="
$PY sweep_t_star.py \
    --cloud_ckpt "$CKPT_DIR/cloud_best.pt" \
    --edge_ckpt  "$CKPT_DIR/edge_best.pt" \
    --num_samples 5000 --batch_size 512 \
    2>&1 | tee "$LOG_DIR/exp2.log"

echo "=== Visualization (GPU 3) ==="
$PY visualize.py \
    --cloud_ckpt "$CKPT_DIR/cloud_best.pt" \
    --edge_ckpt  "$CKPT_DIR/edge_best.pt" \
    2>&1 | tee "$LOG_DIR/vis.log"

echo "=== ALL DONE (GPU 3) ==="
exec bash
SCRIPT

chmod +x "$LOG_DIR"/task_*.sh

# ── 创建 tmux session ─────────────────────────────────────────
tmux new-session  -d -s "$SESSION" -n "train-cloud" "bash $LOG_DIR/task_train_cloud.sh"
tmux new-window      -t "$SESSION" -n "train-edge"  "bash $LOG_DIR/task_train_edge.sh"
tmux new-window      -t "$SESSION" -n "exp1+exp3"   "bash $LOG_DIR/task_exp1_exp3.sh"
tmux new-window      -t "$SESSION" -n "exp2+vis"    "bash $LOG_DIR/task_exp2_vis.sh"
tmux new-window      -t "$SESSION" -n "monitor"     "watch -n 2 nvidia-smi"

tmux select-window -t "$SESSION:train-cloud"

echo "========== tmux session '$SESSION' started =========="
echo ""
echo "GPU 分配:"
echo "  GPU 0  train-cloud"
echo "  GPU 1  train-edge    (并行训练)"
echo "  GPU 2  exp1 → exp3   (训练完自动启动)"
echo "  GPU 3  exp2 → vis    (训练完自动启动)"
echo ""
echo "常用命令:"
echo "  tmux attach -t $SESSION       # 进入 session"
echo "  Ctrl+b, w                      # 窗口列表"
echo "  Ctrl+b, d                      # 后台挂起"
echo "  tmux kill-session -t $SESSION # 终止所有任务"
echo ""
echo "日志: $LOG_DIR/"
echo ""

tmux attach -t "$SESSION"
