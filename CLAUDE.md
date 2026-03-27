# Direction Correction for Edge-Cloud Collaborative Flow Model Inference

## Project Identity

**核心命题**: Rectified Flow 的线性轨迹特性使端云协同推理可从「状态传输」简化为「方向校正」——传输轻量 δv 替代完整中间状态 x_{t*}，在低通信预算下实现 State Relay 无法覆盖的质量提升。

**目标会议**: INFOCOM

**数据集**: CIFAR-10 (32x32x3)

**模型配置**:
- Cloud: UNet 35M params, base_ch=128, ch_mults=(1,2,2,2), w/ attention
- Edge: UNet 4M params, base_ch=48, ch_mults=(1,2,2), no attention
- 独立 FID: Cloud ~28, Edge ~61

---

## Architecture Overview

```
train.py
  → checkpoints/cloud_best.pt, edge_best.pt
    ↓
experiment.py (Exp1,3,4,5)         sweep_t_star.py (Exp2)
experiment_why_flow.py (Exp6-9)    ddim.py (DDIM对照框架)
    ↓
JSON results (Exp1-9)
    ↓ loaded by
adadc_v2.py (AdaDC v2 协议)
    ↓
experiment_v2.py (Exp10-14 系统实验)
  + network_model.py    (带宽/延迟仿真)
  + real_traces.py      (3GPP/Markov trace)
  + latency_profiler.py (推理延迟拆解)
  + sd_latent_ext.py    (SD latent 空间扩展)
```

## File Reference

### Core ML

| File | Purpose | Key Exports |
|------|---------|-------------|
| `models.py` | Cloud/Edge UNet 定义 | `build_cloud_model()`, `build_edge_model()`, `count_params()` |
| `rectified_flow.py` | RF 训练目标 + ODE 采样 + DC 采样器 | `RectifiedFlowTrainer`, `RectifiedFlowSampler`, `DirectionCorrectionSampler` |
| `train.py` | 训练脚本 | `train_model()`, `get_cifar10_loader()` |
| `compression.py` | δv 稀疏化 + 量化 | `sparsify()`, `quantize()`, `make_compress_fn()`, `compute_transmitted_size_kb()` |
| `metrics.py` | FID 计算 | `compute_fid()`, `save_samples_to_dir()`, `prepare_cifar10_reference()` |
| `ddim.py` | DDIM 对照框架 (ε-prediction) | `DDIMTrainer`, `DDIMSampler`, `DCOnDDIMSampler`, `compute_trajectory_straightness()`, `compute_delta_consistency()`, `compute_delta_spectrum()` |

### Experiments

| File | Experiments | CLI |
|------|-------------|-----|
| `experiment.py` | Exp1(基线), Exp3(压缩), Exp4(多点DC), Exp5(多点+压缩) | `--compression_sweep`, `--multi_point`, `--multi_point_compress` |
| `sweep_t_star.py` | Exp2(t\*扫描) | default |
| `experiment_why_flow.py` | Exp6(RF vs DDIM DC), Exp7(TSS), Exp8(一致性), Exp9(SVD) | `--mode {train_ddim,exp6,exp7,exp8,exp9,all}` |
| `experiment_v2.py` | Exp10-14(系统实验) | per-experiment functions |
| `visualize.py` | 轨迹可视化、δv 热图 | — |

### System Layer

| File | Purpose |
|------|---------|
| `adadc_protocol.py` | AdaDC v1: Pareto frontier + 静态配置选择 |
| `adadc_v2.py` | AdaDC v2: Drift estimator + BW predictor + Hybrid scheduler (4种调度器: Offline/Greedy/Optimal/Hybrid) |
| `network_model.py` | AR(1) 对数正态带宽仿真 (3G/4G/WiFi/5G profiles) |
| `real_traces.py` | 3GPP TR 38.901 trace 生成, Markov trace, TraceReplaySimulator |
| `latency_profiler.py` | GPU-aware 推理延迟 profiling (CUDA events) |
| `sd_latent_ext.py` | SD-1.5/SDXL latent 空间通信开销估算 |

---

## Experimental Results Summary

### Exp1-5: Algorithm Validation (Complete)

| Exp | 结论 | 关键数据 |
|-----|------|---------|
| Exp1 | 单点 DC 比 SR 差 ~14 FID，但通信量可压缩 | DC=51.0, SR=36.8, Edge=61.4 |
| Exp2 | DC 最优 t\*=0.7, SR 最优 t\*=0.1, 互补 | 交叉点 t\*≈0.8 |
| Exp3 | 50% sparse + 4-bit 仅增 0.7 FID, 压缩至 3.76KB | 12KB → 3.76KB |
| Exp4 | 3-point DC 逼近 SR (41.67 vs 39.60) | 每增一点 FID 降 ~5 |
| Exp5 | 3pt + 压缩: FID=42.75, 仅 11.27KB | 与 SR 的 12KB 相当 |

**核心论点**: DC 开辟了 SR 无法覆盖的 0-5KB 低通信工作区间。

### Exp6-9: Why Flow Models (Complete)

| Exp | 结论 | 关键数据 |
|-----|------|---------|
| Exp6 | DC 1pt 在 DDIM 上完全失效(负增益) | RF: Edge-10 FID, DDIM: Edge+0.6 FID |
| Exp7 | RF 轨迹近乎直线 | RF TSS=0.978, DDIM=0.912 |
| Exp8 | DDIM 的 δε 远离 t\* 时发散更严重 | t=0.9: DDIM 2.31 vs RF 1.88 |
| Exp9 | RF 的 δv 首个 SV=8.67, DDIM 仅 6.93 | RF 能量更集中 |

**DC 依赖 RF 的三个独有性质**:
1. 轨迹线性 → δv 时间一致性（全程施加有效）
2. ODE 确定性 → 端云可对齐（seed sync）
3. δv 结构化稀疏 → 高压缩率

### Exp10-14: System Layer (Framework Ready)

| Exp | 内容 | 状态 |
|-----|------|------|
| Exp10 | 真实 trace 协议对比 (Hybrid/Offline/Greedy/Optimal) | 待运行 |
| Exp11 | 调度策略 (drift-aware vs periodic vs static) | 待运行 |
| Exp12 | 跨分辨率 (CIFAR → SD-1.5 → SDXL) | 待运行 |
| Exp13 | Seed-sync 消融 | 待运行 |
| Exp14 | 动态带宽压力测试 | 待运行 |

---

## Key Design Decisions

### δv 施加策略
v1(单次) → v2(指数衰减) → **v3(α=1 全程恒定)**。v3 最优因为 RF 速度场沿轨迹近似恒定。

### 压缩策略
两层: Top-K 稀疏化 → 均匀标量量化。4-bit 和 8-bit 几乎无差别 (FID < 0.05)，统一用 4-bit。

### 时间映射 (RF vs DDIM)
- RF: t=0 纯噪声, t=1 干净图像, 正向积分
- DDIM: t=T 纯噪声, t=0 干净图像, 反向采样
- DC on DDIM 中 `t_star=0.7` 映射为 DDIM 的 `ddim_t_split = 1.0 - 0.7 = 0.3`

### AdaDC v2 Hybrid 策略
带宽充足时自动切换 State Relay（质量更高），低带宽时用 DC（通信量可控）。通过 drift estimator 判断校正时机，bandwidth predictor 做前瞻调度。

---

## Running Experiments

```bash
# Training
python train.py --mode both --epochs 100 --batch_size 512 --save_dir checkpoints/
python experiment_why_flow.py --mode train_ddim --epochs 100

# Core experiments
python experiment.py --cloud_ckpt checkpoints/cloud_best.pt \
    --edge_ckpt checkpoints/edge_best.pt --t_star 0.5 --num_samples 10000
python experiment.py ... --compression_sweep
python experiment.py ... --multi_point
python experiment.py ... --multi_point_compress
python sweep_t_star.py --cloud_ckpt ... --edge_ckpt ...

# Why-Flow ablation
python experiment_why_flow.py --mode all \
    --rf_cloud_ckpt checkpoints/cloud_best.pt \
    --rf_edge_ckpt checkpoints/edge_best.pt \
    --ddim_cloud_ckpt checkpoints/ddim_cloud_best.pt \
    --ddim_edge_ckpt checkpoints/ddim_edge_best.pt

# System experiments (Exp10-14)
python experiment_v2.py  # per-experiment entry points
```

---

## Conventions

- Checkpoints: `checkpoints/{cloud,edge,ddim_cloud,ddim_edge}_best.pt`
- Results: `results/` (gitignored, JSON files force-added for key experiments)
- FID 计算: `pytorch-fid`, CIFAR-10 reference cached at `data/cifar10_ref/`
- Random seed: 实验间统一 `torch.manual_seed(42)` 保证可比性
- 通信量单位: KB (per sample)
- 时间归一化: 所有模型接受 t ∈ [0, 1]

---

## Paper Narrative Structure

1. **Motivation**: 端云协同的通信瓶颈 → DC 在低带宽区间开辟新可能
2. **Why Flow**: RF 线性轨迹是 DC 有效的必要条件 (Exp6-9 证明)
3. **Method**: DC 协议 + 多点扩展 + 压缩 (Exp1-5)
4. **System**: AdaDC v2 自适应协议 (Exp10-14)
5. **Evaluation**: 真实 trace + 跨分辨率 + 压力测试
