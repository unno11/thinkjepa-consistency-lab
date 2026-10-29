# 方法与实现

本项目扩展 ThinkJEPA 的轨迹读出路径，研究输出运动和几何一致性。它不训练机器人控制策略，也不构成动作条件闭环规划实验。

## 输入与输出

缓存的 V-JEPA 特征承载视觉动态，缓存的 VLM 多层特征提供语义 guidance。训练设置观察 32 帧，Direct32 一次产生未来 32 帧、52 个关节的世界坐标预测。这里的 Direct32 不能写成逐步因果自回归 rollout。

## MG 损失

设预测轨迹为 x，目标轨迹为 y。MG 损失由归一化骨长差、关节角差，以及相邻帧一阶／二阶差分的误差组成，整体形式为：

$$L = L_{pred} + L_{task} + s(e)\left(0.0006L_{bone} + 0.45L_{velocity} + 0.2L_{acceleration} + 0.0035L_{angle}\right).$$

掌骨项实现保留，但该冻结 MG 配置权重为 0；不能声称正式实验优化了所有骨组。s(e) 从 epoch 2 开始，在三个 epoch 内线性升至 1。具体公式、归一化和零长度处理以 [physics_losses.py](../cache_train/physics_losses.py) 为准。

**梯度范围：** predictor 输出在送到轨迹读出头前 detach。因此 predictor 通过原有预测目标训练，MG 和任务损失更新 readout。不能将这组实验解释为 MG 已改善 JEPA 的潜空间物理建模。

## KProj 强对照

[kinematic_projection.py](../evaluation/kinematic_projection.py) 只读取 B 的预测轨迹，逐骨计算 32 帧预测长度的时间中位数，保留每帧骨方向，并从两个前臂根重建。它不读取观测或未来真实关节，不跨样本计算，也不训练参数。

KProj 使用完整预测时域，是离线后处理，不能称作在线因果滤波。它按构造抑制骨长漂移，但不会修复预测骨方向，因此角度误差不变。它用来检查“增加损失带来的效果是否只是固定骨长即可得到”。

## Guidance 与公平比较

冻结代码保留 batch 轴和明确的 VLM 层映射，并在 eval 模式禁用 SDPA dropout。8 个缓存层映射到 12 个 predictor blocks 的规则是本项目明确采用的实现约定，不等同于论文逐项指定的唯一映射。

B 与 MG 使用相同的数据、初始化流程、优化参数、guidance 路径和修正，只在 MG 权重上不同。所有 seed 的配置均保留，checkpoint 按验证 ADE 选择；这仍然只是 validation 比较。

## 训练与指标的区别

训练的速度／加速度损失作用于预测与目标的差分。报告中的 VelocityRatio、AccelerationRatio 则是运动幅度比，并在逐样本层面变换成 abs(log ratio) 后汇总；二者不是同一个数值。BoneDriftAll 是预测骨长的时间方差，而非骨长与真值的差。FDE 取最后一个时间步，不是最后一个关节。
