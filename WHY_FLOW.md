# 为什么必须是流模型？——Direction Correction 的理论基础与实验验证

> 本文档阐述 Direction Correction (DC) 方法对 Rectified Flow (RF) 的依赖性，
> 以及为什么该方法不能直接迁移到扩散模型 (DDPM/DDIM) 上。

---

## 一、核心论点

**Direction Correction 的有效性建立在三个 RF 独有的性质之上：**

1. **轨迹线性 (Trajectory Straightness)** → δv 的时间一致性
2. **ODE 确定性 (Deterministic ODE)** → 端云轨迹可对齐
3. **δv 的结构化稀疏性** → 高压缩率下仍保持校正效果

这三条性质缺一不可，且都是扩散模型所不具备（或仅部分具备）的。

---

## 二、性质一：轨迹线性 → δv 时间一致性

### 2.1 RF 的恒定速度场

Rectified Flow 的训练目标是：
```
v_θ(x_t, t) ≈ x_1 - x_0    (常数，与 t 无关)
```

其 ODE 轨迹 `x_t = (1-t)·x_0 + t·x_1` 是连接噪声和数据的**直线**。
经过 reflow 后，学到的速度场沿轨迹近似恒定：

```
v_θ(x_t, t) ≈ v_θ(x_{t*}, t*)    ∀t ∈ [t*, 1]
```

**推论**：在 t\* 处测量的 δv = v_cloud - v_edge 在整个后续路径上都是有效的校正量。

### 2.2 DDIM 的弯曲轨迹

DDIM 的 ODE 形式为：
```
dx_t = [f(t)·x_t + g²(t)·s_θ(x_t, t) / (2σ_t)] dt
```

其中 score function `s_θ(x_t, t)` 沿轨迹**剧烈变化**（尤其在高噪声阶段），
导致轨迹高度弯曲。在 t\* 处测量的 δε = ε_cloud - ε_edge
到 t\*+Δt 处就不再是一个有效的校正量。

### 2.3 已有实验证据

我们的 Exp1 中，α=1（全程恒定施加 δv）效果**优于**指数衰减和单次施加：

| 施加策略 | FID |
|---------|-----|
| 仅在 t\* 施加一次 | 60.35 |
| 指数衰减 | 56.41 |
| **α=1 全程恒定** | **50.99** |

这直接验证了 RF 速度场的时间不变性——如果速度场沿轨迹变化剧烈，全程施加同一个 δv 应该反而更差。

---

## 三、性质二：ODE 确定性 → 端云轨迹可对齐

### 3.1 Seed Sync 的可行性

RF 使用纯 ODE 采样（无随机性注入），因此：
- 给定相同初始噪声 x_0，edge 和 cloud 的轨迹完全由各自模型决定
- δv = v_cloud(x_t, t) - v_edge(x_t, t) 严格表示**模型能力差异**
- Edge 和 Cloud 可通过共享 random seed 来隐式对齐初始状态，无需传输 x_0

### 3.2 DDPM 的随机轨迹问题

DDPM 使用 SDE 采样，每步注入 `σ_t · z` 随机噪声：
```
x_{t-1} = μ_θ(x_t, t) + σ_t · z,    z ~ N(0, I)
```

即使 edge 和 cloud 从相同 x_T 出发，由于每步注入不同的 z，轨迹会**随机分叉**。
此时 δε 不再纯粹反映模型能力差异，还混入了随机采样路径差异。

**注**：DDIM (η=0) 虽然也是 ODE，但其确定性轨迹是**弯曲**的（性质一不满足），
所以即使端云对齐了，DC 仍然失效。

---

## 四、性质三：δv 的结构化稀疏性 → 可压缩

### 4.1 为什么 δv 是稀疏的

RF 的速度场目标是 `v = x_1 - x_0`，cloud 和 edge 学习的都是同一个目标。
两个不同容量的模型对同一目标的近似差异 δv 通常是：
- **低秩的**：主要差异集中在 edge 模型容量不足的几个方向
- **空间结构化的**：差异集中在图像的高频/细节区域

我们的 Exp3 验证了这一点：
- top-20% sparsification + 4-bit quantization → 压缩至 1.5 KB，FID 仅从 51.0 涨至 57.6

### 4.2 扩散模型的 δε 为什么难压缩

扩散模型的 noise prediction `ε_θ(x_t, t)` 在高噪声阶段（小 t，即 RF 的大 t 附近）
本身就是一个高频、非结构化的信号。两个不同容量模型的差异 δε 同样是高频散射的，
稀疏化会丢失关键信息。

---

## 五、对照实验设计

### Exp6：DC on DDIM vs DC on RF

**目标**：用控制变量法证明 DC 的有效性依赖于 RF 的线性轨迹。

**实验设置**：
- 使用**完全相同的 UNet 架构**（cloud: 35M, edge: 4M）
- 分别用 RF 目标和 DDIM 目标训练
- 在两种框架下执行相同的 DC 协议
- 对比 FID

**预期结果**：

| 方法 | RF | DDIM |
|------|-----|------|
| Cloud Only | ~28 | ~28 |
| Edge Only | ~61 | ~61 |
| DC (single, t\*=0.7) | ~52 | **>>52** |
| DC (3-point) | ~42 | **>>42** |
| State Relay | ~40 | ~40 |

DC on DDIM 的 FID 预期显著高于 DC on RF，而 State Relay 在两者上差距不大——
因为 State Relay 传输的是完整状态，不依赖速度场的时间一致性。

### Exp7：轨迹直线度 (Straightness) 量化

**指标**：Trajectory Straightness Score (TSS)

```
TSS = E_{x_0} [ ||x_1 - x_0||₂ / L(trajectory) ]
```

其中 L 是轨迹弧长（Euler 步积分）。TSS=1 表示完美直线。

**预期**：RF 的 TSS 接近 1，DDIM 的 TSS 显著小于 1。

### Exp8：δv / δε 时间一致性曲线

在多个时间点 t ∈ {0.1, 0.2, ..., 0.9} 测量 δv(t) 或 δε(t)，
计算与 t\*=0.5 处的差值 ||δv(t) - δv(t\*)||₂ / ||δv(t\*)||₂。

**预期**：RF 上该曲线近似水平（δv 恒定），DDIM 上随 |t-t\*| 增大而急剧上升。

### Exp9：δv 能量谱分析

对 δv（RF）和 δε（DDIM）做 SVD，比较前 k 个奇异值的能量占比。

**预期**：RF 的 δv 能量集中在前几个奇异值（低秩），DDIM 的 δε 能量分散。

---

## 六、论文叙事建议

### 引言中的 Motivation

不要以"flow model is good"开头，而是以**问题驱动**的方式引入：

> "Edge-cloud collaborative inference for generative models faces a
> fundamental tension: state relay achieves high quality but requires
> transmitting the full intermediate representation; edge-only inference
> avoids communication but suffers from limited model capacity.
> **We observe that Rectified Flow's linear trajectory property creates
> a third option**: instead of transmitting state, we can transmit a
> lightweight direction correction vector that remains valid throughout
> the remaining trajectory. This is fundamentally impossible for
> diffusion models whose curved trajectories invalidate any fixed
> correction after a few steps."

### Section 3: Why Flow Models

建议独立一个 section（或 subsection）用来论证 RF 的不可替代性：

1. **Proposition 1** (Trajectory Linearity)：形式化 RF 的速度恒定性
2. **Proposition 2** (Correction Validity)：推导 DC 有效的充分条件
3. **Theorem 1** (Compression Bound)：δv 的稀疏性上界与轨迹直线度的关系
4. **Remark**：为什么 DDIM 不满足上述条件

### 实验部分

Exp6 作为 **ablation study** 放在实验的最后一个 subsection，
标题建议："Why not diffusion? — The necessity of flow matching"

---

## 七、文件清单

| 文件 | 用途 |
|------|------|
| `ddim.py` | DDIM 训练器 + 采样器 + DC-on-DDIM 实现 |
| `experiment_why_flow.py` | Exp6-9 实验入口 |
| `WHY_FLOW.md` | 本文档 |
