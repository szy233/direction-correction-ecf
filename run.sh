#!/bin/bash
# run.sh — 一键运行全部实验
# Usage: bash run.sh

set -e

# 自动检测 python 命令
if command -v python3 &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null; then
    PY=python
else
    echo "Error: python not found"
    exit 1
fi

echo "Using: $PY ($($PY --version 2>&1))"
echo ""

# Step 0: 安装依赖
echo "========== Installing dependencies =========="
$PY -m pip install -r requirements.txt

# Step 1: 训练
echo ""
echo "========== Training Cloud Model =========="
$PY train.py --mode cloud --epochs 100 --save_dir checkpoints/

echo ""
echo "========== Training Edge Model =========="
$PY train.py --mode edge --epochs 100 --save_dir checkpoints/

# Step 2: 核心实验
echo ""
echo "========== Experiment 1: Direction Correction vs Baselines =========="
$PY experiment.py \
    --cloud_ckpt checkpoints/cloud_best.pt \
    --edge_ckpt checkpoints/edge_best.pt \
    --t_star 0.5 --num_samples 10000

# Step 3: t* 扫描
echo ""
echo "========== Experiment 2: t* Sweep =========="
$PY sweep_t_star.py \
    --cloud_ckpt checkpoints/cloud_best.pt \
    --edge_ckpt checkpoints/edge_best.pt \
    --num_samples 5000

# Step 4: 压缩率实验
echo ""
echo "========== Experiment 3: Compression Sweep =========="
$PY experiment.py \
    --cloud_ckpt checkpoints/cloud_best.pt \
    --edge_ckpt checkpoints/edge_best.pt \
    --t_star 0.5 --num_samples 5000 --compression_sweep

# Step 5: 可视化
echo ""
echo "========== Visualization =========="
$PY visualize.py \
    --cloud_ckpt checkpoints/cloud_best.pt \
    --edge_ckpt checkpoints/edge_best.pt

echo ""
echo "========== All done! Results in results/ =========="
