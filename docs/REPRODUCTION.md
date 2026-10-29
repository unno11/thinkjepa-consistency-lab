# 训练与评测复现

代码来源为冻结研究提交 `0385f7057497a553f3094553f69aac68802572fe`，基线锚点为上游 `9b95d358f7c7a1eaf73fda0efbcc8e28c44ce086`。本次仅复算归档指标和运行本地 CPU 检查，未在新环境中重训六个模型。

## 环境与数据

轻量指标和配置检查只需 Python 3.10+ 与 NumPy。训练需按硬件安装匹配的 PyTorch、torchvision、torchaudio，然后安装 `requirements-public.txt`。本仓库聚焦缓存特征训练；从原始视频提取特征请使用上游完整实现。具体上游环境请参考 [ThinkJEPA](https://github.com/Hai-chao-Zhang/ThinkJEPA)。未验证的硬件组合不保证直接运行。

数据和缓存来自 [上游缓存页面](https://huggingface.co/datasets/haichaozhang/cache)。本仓库不分发视频、缓存、权重或未来真实关节轨迹。提供的是精确训练／验证成员的相对文件名和归档评测指标；使用上游最新缓存并不保证与冻结缓存完全相同，应核对层索引和命名约定。

将本机缓存组织为 `DATA_ROOT/cache/part2/...npz`。protocol/splits 中的路径形如 `data/cache/part2/...npz`，加载器会在数据根目录下解析。公开的冻结成员为 train 1600、validation 200；最终测试成员不包含在发布中。

## 模型训练

默认只打印参数和配置，不启动训练：

```sh
python scripts/run_experiment.py \
  --config configs/thinkjepa_MG_alllayers_100ep.json \
  --data-dir /path/to/data \
  --train-manifest protocol/splits/train.txt \
  --val-manifest protocol/splits/val.txt \
  --output-root outputs \
  --gpus 0,1,2,3 --nproc-per-node 4
```

检查数据、环境和资源后加入 `--execute` 启动。B 使用 `thinkjepa_B_alllayers_100ep.json`；种子 43/44 使用同目录中的相应后缀配置。原实验四卡、每卡 batch 2、全局 batch 8，100 epochs；改变卡数、累积步数或 batch 不应仍声称完全相同的协议。

这里“from_scratch”描述被训练 predictor 和轨迹读出模块的初始化，视觉／VLM 使用预训练缓存，不意味着所有大模型从零训练。实验使用修正后 guidance 语义，不能用修正前的 checkpoint 冒充同一组结果。

训练器遗留参数名 `--test_manifest` 在包装器中接收的是 validation 成员，checkpoint 选择只依据 validation ADE；这不是执行最终测试。输出目录存在时包装器拒绝覆盖。

## Direct32 评测

以种子 42 为例，先准备自己训练、满足配置要求的 checkpoint：

```sh
python evaluation/evaluate_direct32.py \
  --protocol configs/direct32_B_MG_alllayers_gate0fixed_seed42_validation_v2.json \
  --role B --checkpoint /path/to/B/ckpt_best.pt \
  --data-dir /path/to/data --cache-dir /path/to/data/cache \
  --output-dir outputs/B_seed42_validation --gpu 0 \
  --kinematic-projection-control
```

MG 改为 `--role MG`，提供匹配 checkpoint 并去掉 KProj 开关。seed43、seed44 使用各自冻结的 validation protocol。评估会检查角色、种子、选择 epoch、数据哈希和模型参数；任意后续训练不一定满足历史 epoch 条件，应明确作为新的实验协议，而不是放宽旧协议冒充复现。

未来相机外参仅用于坐标变换；未来真实关节不作为投影输入。该评测并非部署时只接收机器人实时观测的控制协议。

## 结果复算范围

运行 `python scripts/summarize_results.py` 可从公开 CSV 核对三种子描述性结果。原始预测与目标关节数组未分发，因此该步骤不能重新计算每条轨迹的底层几何指标；它检查的是保存指标的配对、变换和汇总。独立底层指标实现见 `evaluation/metrics.py`，需要实际预测轨迹才能重新评估。
