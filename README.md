# Direction Correction for Edge-Cloud Collaborative Flow Model Inference

## 核心命题
Rectified Flow 的线性轨迹性质使得端云协同可以从「状态传输」简化为「方向校正」，
大幅降低通信开销的同时保持生成质量。

## 项目结构
```
direction_correction_exp/
├── README.md
├── requirements.txt
├── models.py          # 云端/端侧 UNet 模型定义
├── rectified_flow.py  # Rectified Flow 训练与采样
├── train.py           # 训练脚本（云端全量 + 端侧压缩）
├── experiment.py      # 核心实验：方向修正 vs 状态接力 vs 纯端侧
├── sweep_t_star.py    # 实验二：t* 切分点扫描
├── compression.py     # δv 压缩策略（稀疏化 + 量化）
├── metrics.py         # FID 计算
└── visualize.py       # 轨迹可视化
```

## 环境要求
```bash
pip install -r requirements.txt
```

## 快速开始
```bash
# Step 1: 训练云端全量模型 + 端侧压缩模型
python train.py --mode cloud --epochs 100 --save_dir checkpoints/
python train.py --mode edge  --epochs 100 --save_dir checkpoints/

# Step 2: 运行核心实验（方向修正 vs 状态接力 vs 纯端侧）
python experiment.py --cloud_ckpt checkpoints/cloud_best.pt \
                     --edge_ckpt checkpoints/edge_best.pt \
                     --t_star 0.5 --num_samples 10000

# Step 3: t* 扫描
python sweep_t_star.py --cloud_ckpt checkpoints/cloud_best.pt \
                       --edge_ckpt checkpoints/edge_best.pt

# Step 4: 压缩率实验
python experiment.py --cloud_ckpt checkpoints/cloud_best.pt \
                     --edge_ckpt checkpoints/edge_best.pt \
                     --t_star 0.5 --compression_sweep
```
