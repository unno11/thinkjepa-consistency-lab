# ThinkJEPA Consistency Lab

**基于 ThinkJEPA 的手部运动预测与运动—几何一致性扩展。** 比较训练时约束与预测后几何投影，研究轨迹精度、骨骼几何和动态一致性的权衡。

![Architecture and gradient scope](docs/assets/pipeline.svg)

## 方法

- **B**：ThinkJEPA 轨迹预测基线。
- **MG**：对轨迹读出头施加骨长、关节角、速度和加速度约束。读出路径接收 detach 后的 predictor 特征，MG 梯度作用于读出头。
- **KProj**：使用预测序列自身的骨长中位数做几何投影，不读取未来真实关节，也不需要额外训练。

## 核心代码

```text
src/
  predictor.py             多层 VLM guidance 与潜特征预测
  models.py                轨迹读出头
  losses.py                运动—几何一致性损失与权重调度
  metrics.py               轨迹、几何与动态评测指标
  kinematic_projection.py  预测序列骨长投影
  topology.py              52 关节拓扑
  vendor/                  predictor 必需的 4 个 V-JEPA2 支持模块
evaluate.py                轨迹评测入口
```

本仓库发布核心算法实现，便于阅读和集成。特征提取、数据加载、完整 DDP 训练与 checkpoint 推理流程使用 [ThinkJEPA 上游工程](https://github.com/Hai-chao-Zhang/ThinkJEPA)；本仓库不提供完整训练复现包。

### 轨迹评测

Python 3.10+：

```sh
pip install -r requirements.txt
python evaluate.py --input trajectory.npz --kproj
```

输入文件包含 `prediction` 和 `target` 两个数组，形状均为 `[T, 52, 3]`，坐标单位为米、关节顺序见 `src/topology.py`。入口评估已有预测轨迹，不运行视频模型推理。零目标运动导致的未定义比值输出为 `null`。

使用模型和损失模块时，另行安装与运行环境匹配的 PyTorch 和 timm。核心接口：

```python
from src.models import TrajectoryReadoutMLP
from src.predictor import CortexGuidedVideoPredictor
from src.losses import compute_motion_geometry_losses, get_physics_loss_weight_scale
from src.metrics import compute_trajectory_metrics
```

### 训练设置

归档实验使用缓存的预训练视觉与 VLM 特征，训练 predictor 与轨迹读出头。B 与 MG 采用相同数据、优化设置和 guidance 修正，checkpoint 按验证 ADE 选择。

| 设置 | 取值 |
| --- | --- |
| 输入 / 预测 | 32 帧 / 32 帧，52 关节，步幅 1 |
| 训练 / 验证 | 1,600 / 200 个样本 |
| 训练种子 | 42、43、44 |
| 训练周期 / 全局 batch | 100 epochs / 8（4 卡，每卡 2） |
| MG 权重 | 骨长 0.0006，掌骨 0，速度 0.45，加速度 0.20，角度 0.0035 |
| MG 调度 | 从 epoch 2 开始，3 epochs 线性 warmup，epoch 从 0 计数 |

## 验证结果

同一组 200 个验证样本、13 类任务，三个训练种子的均值；所有指标越低越好。

| 指标 | B | KProj | MG |
| --- | ---: | ---: | ---: |
| ADE · m | 0.064185 | 0.064220 | **0.063693** |
| FDE · m | **0.060596** | 0.060958 | 0.061635 |
| 整体骨长误差 · mm | 5.890 | 4.053 | **2.504** |
| 关节角误差 · ° | 19.681 | 19.681 | **9.257** |
| 速度偏差 · abs(log ratio) | 1.425 | 1.422 | **1.232** |
| 加速度偏差 · abs(log ratio) | 2.122 | 2.120 | **1.906** |

![Three-seed comparison](docs/assets/results.svg)

MG 的几何与动态一致性在三个种子中均改善；ADE 仅部分种子改善，FDE 均值上升。KProj 使用完整预测时域，是离线后处理。

上述结果是历史验证实验汇总，并非官方论文表格的直接复现成绩。独立最终测试、跨数据集泛化和机器人闭环控制尚未验证；MG 不向 predictor 传播梯度。速度与加速度偏差在逐样本层面进行 `abs(log ratio)` 变换；种子标准差如图所示。当前代码精简未重新训练模型。

## 来源与许可

This software includes or is derived from **[ThinkJEPA](https://github.com/Hai-chao-Zhang/ThinkJEPA)**, originally developed at Northeastern University, NEU SMILE LAB by Haichao Zhang and Yun Raymond Fu. This is an independent, modified implementation, not an official ThinkJEPA release.

使用相关工作时请引用 [ThinkJEPA 论文](https://arxiv.org/abs/2603.22281)，引用条目见 [CITATION.bib](CITATION.bib)。本项目整理所依据的研究代码快照为 `0385f7057497a553f3094553f69aac68802572fe`；核心模型、损失和指标从该快照整理，导入路径按精简目录调整。

保留 [LICENSE](LICENSE)、[NOTICE](NOTICE) 及 [V-JEPA2 支持模块许可](src/vendor/LICENSE)。根许可为 `LicenseRef-ThinkJEPA-Attribution`；第三方组件适用各自许可。
