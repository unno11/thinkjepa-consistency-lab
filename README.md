# ThinkJEPA Consistency Lab

**面向手部运动预测的运动—几何一致性研究。** 基于 ThinkJEPA，比较训练时的一致性正则与无需训练的骨长投影，评估轨迹精度、骨骼几何和运动动态之间的权衡。

[方法](docs/METHOD.md) · [训练与评测](docs/REPRODUCTION.md) · [实验结果](results/README.md) · [适用边界](docs/RESULTS_AND_LIMITATIONS.md)

![Model pipeline and gradient scope](docs/assets/pipeline.svg)

## 研究设计

在相同的缓存特征、数据划分和训练设置下，对比三种方法：

| 方法 | 实现 | 目的 |
| --- | --- | --- |
| **B** | ThinkJEPA 轨迹预测基线 | 建立统一参考 |
| **MG** | 在轨迹读出头加入骨长、关节角、速度和加速度约束 | 改善输出的一致性 |
| **KProj** | 对 B 的预测进行目标无关的骨长投影 | 区分学习约束与几何后处理的收益 |

MG 正则更新轨迹读出头；predictor 输出在进入读出路径前已 detach。KProj 使用预测序列自身的骨长统计，不读取未来真实关节。实现还包含 guidance 样本隔离、评估阶段 attention dropout 和指标定义的检查。

## 实验结果

**Direct32**：观察 32 帧，一次预测未来 32 帧、52 个关节。训练集 1,600 个样本，验证集 200 个样本、覆盖 13 类任务。下表为训练种子 42 / 43 / 44 的均值，所有指标越低越好。

| 指标 | B | KProj | MG |
| --- | ---: | ---: | ---: |
| ADE · m | 0.064185 | 0.064220 | **0.063693** |
| FDE · m | **0.060596** | 0.060958 | 0.061635 |
| 整体骨长误差 · mm | 5.890 | 4.053 | **2.504** |
| 关节角误差 · ° | 19.681 | 19.681 | **9.257** |
| 速度偏差 · abs(log ratio) | 1.425 | 1.422 | **1.232** |
| 加速度偏差 · abs(log ratio) | 2.122 | 2.120 | **1.906** |

![Three-seed validation comparison](docs/assets/results.svg)

MG 在三个种子中均改善整体骨长、角度和动态一致性，但 ADE 仅在部分种子改善，FDE 均值上升。KProj 则以无需训练的后处理消除预测骨长的时间漂移。结果表明，一致性收益需要结合轨迹精度与具体指标共同评估。

这些结果来自相同验证集上的三训练种子实验；独立最终测试、跨数据集泛化和机器人闭环控制尚未验证。逐样本指标、种子标准差及统计说明见 [实验数据](results/README.md)。

## 快速开始

仅复算公开结果，无需 GPU、模型权重或原始数据。使用 Python 3.10+：

```sh
git clone https://github.com/unno11/thinkjepa-consistency-lab.git
cd thinkjepa-consistency-lab
python scripts/summarize_results.py
```

脚本检查文件哈希、样本配对和重复基线的一致性，并从逐样本指标重算三种子汇总。`--write` 可更新汇总表。

运行轻量检查和单元测试：

```sh
python -m pip install -r requirements-analysis.txt
python scripts/check_release.py
python -m unittest discover -s tests -v
```

训练与模型推理需要 PyTorch、匹配的运行环境及特征缓存。完整步骤见 [复现指南](docs/REPRODUCTION.md)，检查覆盖范围见 [验证说明](docs/VALIDATION.md)。

## 代码导航

| 目录 | 内容 |
| --- | --- |
| [`cache_train/`](cache_train/) | 缓存特征训练、predictor、轨迹读出与 MG 损失 |
| [`evaluation/`](evaluation/) | Direct32 推理、轨迹指标、KProj 与对照分析 |
| [`configs/`](configs/) | 训练配置、三种子评测协议与轻量验证配置 |
| [`scripts/`](scripts/) | 训练入口、缓存检查、guidance 诊断与结果复算 |
| [`results/`](results/) | 逐样本指标与三种子汇总 |
| [`tests/`](tests/) | 指标、投影、配置、guidance 和损失测试 |

`egodex/` 与 `vjepa2/` 仅保留当前路径所需的第三方组件及许可。原始特征提取、其他下游任务和完整预训练代码请参考上游项目。

## 来源与引用

This software includes or is derived from **[ThinkJEPA](https://github.com/Hai-chao-Zhang/ThinkJEPA)**, originally developed at Northeastern University, NEU SMILE LAB by Haichao Zhang and Yun Raymond Fu. This repository is an independent, modified research implementation, not an official ThinkJEPA release.

使用相关工作时请引用 [ThinkJEPA 论文](https://arxiv.org/abs/2603.22281)。上游模型、训练框架与本项目扩展的来源见 [来源说明](docs/PROVENANCE.md)；引用条目见 [CITATION.bib](CITATION.bib)。

保留 [LICENSE](LICENSE)、[NOTICE](NOTICE) 与第三方许可。根许可为 `LicenseRef-ThinkJEPA-Attribution`；第三方组件分别适用其目录中的许可。
