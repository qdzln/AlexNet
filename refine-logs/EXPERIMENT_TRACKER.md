# 第五阶段运行清单

`configs/stage5_manifest.json` 给出每个运行的名称、分组与配置。运行状态、已尝试超参数和各轮指标由 `outputs/<实验名>/` 记录。归档后可从 `results/stage5/` 核对汇总表及各次试验的配置和指标。

| 顺序 | 运行 | 用途 |
| --- | --- | --- |
| 0 | `stage4_baseline_5epochs` | 已完成的 Adam 基线 |
| 1 | `stage5_bn` | BN 对照 |
| 2 | `stage5_dropout` | Dropout 对照 |
| 3–4 | `stage5_sgd_lr001`、`stage5_sgd_lr003` | SGD 学习率搜索 |
| 5–6 | `stage5_sgdm_lr001`、`stage5_sgdm_lr003` | SGD + 动量学习率搜索 |
| 7 | `stage5_attention` | 自注意力分类头 |
| 8 | `stage5_finalist` | 对验证集选出的配置训练 12 轮 |

开始测试前生成 `selection.json`，记录参与选择的配置和最终候选。测试阶段仅执行推理与计数。
