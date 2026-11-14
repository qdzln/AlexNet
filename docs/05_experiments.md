# 第五阶段：做对照实验并验收测试集

用上一阶段的基线当参照。每个 5 轮实验使用相同的数据划分、批次大小、种子与输入处理。设计和运行顺序见[实验计划](../refine-logs/EXPERIMENT_PLAN.md)。

## BN 和 Dropout 改变哪一层

打开 `src/model.py` 找 `AlexNet.__init__()`。`use_bn=True` 时，卷积输出先经过 `BatchNorm2d`，然后经过 ReLU。BN 训练时按通道使用当前批次的统计量，并更新运行统计，评估时使用已保存的统计量。卷积层省略偏置，因为 BN 有可学习的偏移参数。

在两个隐藏全连接层的 ReLU 后，`Dropout(p=0.5)` 训练时随机将约一半激活置零，其余激活缩放。进入评估模式后随机丢弃关闭。比较这组与基线时，使用各自最佳检查点在评估模式下重新测量训练准确率，也比较测试准确率。这样可以区分训练过程中随机丢弃带来的差异和固定模型的预测差异。

## SGD 中动量怎样参与更新

打开 `src/train.py` 找优化器创建处。以 `g` 表示当前梯度，`v` 表示动量缓冲，`η` 表示学习率。在本项目的 SGD 配置下，可以把更新写为 `v ← μv + g`，`w ← w − ηv`。当 `μ=0`，每次只沿当前梯度更新。当 `μ=0.9`，前面几批的梯度会累积到本次更新中。参考 [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)。

实验同时试 `0.01` 与 `0.03`。先在同一学习率下比较有无动量，再分别用验证集选出纯 SGD 和动量 SGD 的最佳学习率。Adam 的 `0.0003` 是原基线配置，它的自适应更新规则与 SGD 不同。

## 注意力怎样替换全连接层

线性分类头先把 `[N,256,2,2]` 展平为 `[N,1024]`，再进入 `fc1`。注意力版本用 `AttentionPool` 替换这段。四个空间位置变成 `[N,4,256]`，每个位置有 256 个通道的特征，加上可学习的位置参数。

`AttentionPool.forward()` 以同一组向量作为 Q、K、V，交给四头自注意力计算。每个位置根据与其他位置的关系汇总信息。把注意力输出加回输入，再做层归一化，对四个位置取均值，得到 `[N,256]`。剩下的 `fc2` 和 `fc3` 输出 10 类分数。`describe_shapes()` 将 `AttentionPool` 作为整体报告，参数量包含位置参数、注意力投影和层归一化。参考 [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)。

这个对照同时改变了空间汇总方式、分类头宽度和参数量，测量的是整套替换的效果。运行下列命令可以观察形状与梯度。

```powershell
.\.venv\Scripts\python.exe -m src.inspect_model --head attention --device cpu
```

## 跟读实验入口

1. `read_json()` 读取清单和配置。`run_comparison()` 训练某个配置，或复用名称、配置一致且已完成的运行。
2. `ranked_runs()` 按验证准确率排序，同分时看验证损失，再按实验名排序。
3. `train_matrix()` 跑完短对照，将验证胜出的配置从头训练 12 轮，按同一规则写出最终候选和各检查点哈希。
4. `test_matrix()` 按选模记录验证权重哈希，加载各组最佳模型评估 10,000 张测试图像。
5. `summarize_experiments.py` 中的 `collect_run()` 归档单次记录，`plot_validation()` 画曲线，`summarize()` 汇总指标及相对基线的差值。`main()` 读取命令行参数。

```powershell
.\.venv\Scripts\python.exe -m src.run_experiments train --device cuda
.\.venv\Scripts\python.exe -m src.run_experiments test --device cuda
.\.venv\Scripts\python.exe -m src.summarize_experiments --output results/stage5
```

当前项目已完成这组实验。`outputs/stage5_selection.json` 已存在时，训练命令会提示选模记录已保存。可直接查看已有结果。要做新实验，可以复制清单、更换新运行的实验名与候选配置路径，再用 `--manifest` 和 `--selection` 指向新文件。

在 `comparison.csv` 中，准确率以 0 到 1 的小数保存，`delta_pp` 是相对于基线的百分点差。`runs/` 下有每次配置、状态与逐轮历史。最终准确率用选定模型的测试正确数除以 10,000。所有对照都使用一个固定训练种子，细小差距需要多种子复验证明其稳定性。见[实测结果](../results/stage5/README.md)。
