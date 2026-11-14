# 第六阶段：验收、复现与 GitHub 交付

本项目用 PyTorch 基础层实现灰度图 AlexNet。验证集选出的自注意力版网络使用第 10 轮权重，在官方测试集上正确分类 **9,201／10,000 张，准确率 92.01%**。

| 集合 | 正确数／样本数 | 准确率 |
| --- | ---: | ---: |
| 训练 | 52,296／54,000 | 96.84% |
| 验证 | 5,520／6,000 | 92.00% |
| 测试 | 9,201／10,000 | 92.01% |

## 作业要求对应材料

| 要求 | 交付材料 |
| --- | --- |
| 1. 每类展示 10 张 | [100 张灰度图网格](../results/class_examples_seed_42.png)及[选图索引](../results/class_examples_seed_42.json) |
| 2–3. 手写网络并打印结构 | [网络实现](../src/model.py)、[基线结构](../results/model_summary_cuda_bn0_dropout0_batch64_seed42.txt)、[注意力结构](../results/model_summary_cpu_bn0_dropout0_batch8_seed42_attention.txt) |
| 4. 记录调参过程 | [运行清单](../configs/stage5_manifest.json)、[各组配置与指标](../results/stage5/comparison.csv)、[逐轮记录](../results/stage5/runs/) |
| 5. 三个集合准确率，测试准确率超过 90% | [最终候选训练与验证记录](../results/stage5/runs/stage5_finalist/metrics.json)、[官方测试记录](../results/stage5/runs/stage5_finalist/best_test_evaluation.json) |
| 6–9. BN、Dropout、动量、自注意力 | [对照分析与曲线](../results/stage5/README.md)、[实现教程](05_experiments.md) |

样本和标签使用 Fashion-MNIST 原始数据。输入保留单通道，训练时从 28×28 插值到 96×96。训练与验证共用划分种子 42，正式测试前先封存每组验证结果和最佳权重的 SHA-256。详见[数据教程](02_data.md)和[选模记录](../results/stage5/selection.json)。

## 在新副本复现最终配置

在 Windows PowerShell 的项目根目录运行以下命令。基线环境为 Python 3.13.16、PyTorch 2.14.1+cu126、TorchVision 0.29.1+cu126，实测 GPU 为 RTX 3050 Ti Laptop。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe -m src.train --config configs/stage5/finalist.json --run-name reproduce_final
.\.venv\Scripts\python.exe -m src.evaluate outputs/reproduce_final/best.pt --split test
```

训练命令运行 12 轮，按本次运行的验证准确率和验证损失保存 `best.pt`。复用本地实验时，原始权重可从 `outputs/stage5_finalist/best.pt` 加载。重新训练使用新实验名。设备参数默认 `auto`，也可指定 `--device cpu`，需为系统安装对应版本。固定种子和版本便于复现，硬件或软件变化可能造成小幅差异。报告的 92.01% 对应已归档的检查点哈希。

如需复现全部对照，按[第五阶段教程](05_experiments.md)运行实验清单。新副本将训练清单中缺失的运行。若不同环境的 5 轮胜出配置发生变化，可复制清单，给候选指定新的配置路径和实验名，再用新的 `--selection` 路径保存选模记录。这样可以同时保留本次报告中的候选配置。

## 提交与检查

```powershell
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
git status --short
git log --oneline -6
```

GitHub 仓库提交代码、配置、分阶段教程、测试和实验记录。`data/`、`.venv/` 和 `outputs/` 由本地命令生成。模型权重从相应训练运行恢复。样例图出自 [Fashion-MNIST 官方项目](https://github.com/zalandoresearch/fashion-mnist)。
