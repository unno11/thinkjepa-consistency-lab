# 工程案例

## 保留 batch 轴

原探索过程中，多层 VLM guidance 的预处理曾折叠 batch 轴，使样本之间可能互相影响。验证方式包括单独预测样本 A、与另一个样本 B 组 batch、只改变 B，以及重复 A。修正保留 `[B,L,S,D]` 维度，并为每个 predictor block 指定源层映射。

这里的关键是样本隔离，而不是某一次 loss 更低。公开实现和测试位于 `evaluation/guidance_contract.py`、`tests/test_guidance_contract.py` 与 `tests/test_guidance_semantics_torch.py`。

## eval 不自动关闭 functional SDPA dropout

模块处于 eval 状态并不自动改写 functional SDPA 的 `dropout_p` 参数。冻结版本在 eval 明确传入零，同时保留训练行为；对应检查见 `tests/test_attention_eval_semantics.py`。本次轻量环境没有 PyTorch，这部分测试跳过，不将历史 GPU 检查冒充本次重跑。

## FDE 必须取最后一个时间步

轨迹误差是 `[T,J]` 时，`error[-1]` 才是最后时刻；取 `error[:, -1]` 会变成最后一个关节。当前指标实现独立于训练器，见 `evaluation/metrics.py` 和对应测试。不同评估器必须先对齐坐标、单位和轴，再比较数值。

## 正结果与强对照

只看骨长漂移会把 MG 的价值夸大。KProj 不训练也能固定长度，因此实验继续比较角度、动态、掌骨以及空间端点误差，并报告优势和代价。无效或退化指标是方法说明的一部分。
