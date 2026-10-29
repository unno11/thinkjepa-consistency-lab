# 验证结果与复算

`per_sample/` 包含每个训练种子下 B、MG、KProj 的 200 条 validation 指标，以及单独 KProj 评估时重复计算的 B。共 12 份 CSV，重复 B 用来检查两条评估路径等价，不增加独立样本数。

- `model_seed_summary.csv`：原冻结三方比较汇总。
- `per_seed_pairwise.csv`：原冻结逐种子配对结果与校正信息。
- `B_MG_KProj_multiseed_summary.json`：原冻结分析身份和完整摘要。
- `summary.csv`：从逐样本 CSV 重新计算的三方法均值和种子标准差。
- `verification.json`：本次复算范围与差异检查。

运行 `python scripts/summarize_results.py` 可核对源文件 SHA256、样本配对、重复 B 等价、逐样本 abs(log ratio) 变换，以及三种子描述性汇总。增加 `--write` 可再生成后两份文件。

CSV 是保存的指标，不是原始视频、缓存或关节轨迹。复算不会运行模型，也不会重新计算底层轨迹误差。每种方法的 600 条记录来自相同 200 个验证样本和三训练种子；不能把它们当作 600 个不同环境。最终测试未执行。
