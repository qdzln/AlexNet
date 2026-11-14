# 第四阶段：训练、验证和保存最佳模型

上一阶段已经算出梯度；本阶段加入优化器，实际用梯度更新权重。使用同一个 AlexNet，
对 54,000 张训练图像训练 5 轮，每轮结束用固定的 6,000 张验证图像检查效果。

## 1. 先跑一遍，再读代码

在项目根目录执行。已有第二阶段数据即可，无需联网或安装新依赖。

```powershell
# 使用新的实验名，避免与已完成的基线记录重名
.\.venv\Scripts\python.exe -m src.train --config configs/baseline.json --run-name my_baseline

# 用独立入口重新加载最佳权重，评估验证集
.\.venv\Scripts\python.exe -m src.evaluate outputs/my_baseline/best.pt --split val

# 从历史记录重新画图，不重新训练
.\.venv\Scripts\python.exe -m src.plot_history outputs/my_baseline
```

也可以用项目解释器直接运行 `src/train.py` 和 `src/evaluate.py`，参数相同。
设备默认 `auto`，有 CUDA 时使用 GPU；也可加 `--device cuda` 或 `--device cpu`。
同名实验目录会报错；要再跑一次，换一个 `--run-name`。本机的示范训练约 109 秒，
其他设备耗时会不同。

本次完整配置在 `configs/baseline.json`：批次大小 64、Adam 学习率 0.0003、
BN 关闭、Dropout 概率 0，不做数据增强。`seed` 控制模型初始化和训练顺序，
`split_seed` 决定哪些图片属于训练集和验证集。调参时不要顺手改 `split_seed`，
否则不同配置的指标就可能来自不同图片。固定种子有助于复现，不保证跨软硬件结果逐位相同。

## 2. 一批图片怎样变成一次参数更新

从 `src/engine.py` 的 `run_epoch()` 开始。一个 epoch 指完整遍历一次数据集。
以 54,000 张图、每批 64 张为例，共 844 批，最后一批只有 48 张，也要处理。

```python
optimizer.zero_grad(set_to_none=True)  # 清除上一批留下的梯度
logits = model(images)                 # 前向：每张图得到 10 个类别分数
loss = criterion(logits, labels)       # 交叉熵：根据正确标签衡量预测误差
loss.backward()                        # 反向：计算每个参数的梯度
optimizer.step()                       # 依据梯度更新权重
```

这是训练分支的核心。调用前先用 `prepare_images()` 把原始批次从 `[N,1,28,28]`
转换为 `[N,1,96,96]`，并把像素映射到 `[-1,1]`。每个集合沿用同一套处理。
`criterion` 接收 logits 与整数标签，不需要先做 softmax。梯度描述怎样改变参数会
影响损失；学习率控制更新大小。Adam 根据梯度历史调整更新幅度，配置中的 `momentum`
是 SGD 的可选参数；后续做动量对照时会分别设置 SGD 学习率。

## 3. 验证、统计和选最佳轮次

同一个 `run_epoch()` 能评估验证集：不传 `optimizer` 就进入 `model.eval()`，
同时关闭梯度记录。`eval()` 切换 BN 和 Dropout 的行为；关闭梯度记录避免为反向传播
保存中间结果。验证只计算预测和指标，不更新模型。

| 指标 | 计算方法 |
| --- | --- |
| 准确率 | 正确预测的样本数 ÷ 实际处理的样本数 |
| 平均损失 | 每批平均损失 × 该批样本数，求和后 ÷ 总样本数 |

如果一批有 64 张、另一批有 48 张，直接平均两批损失会给后面的小批次过高权重。
乘样本数再求和，才能让每张图片对总损失贡献相同。`argmax(dim=1)` 对每张图
取分数最高的类别编号，用它和标签比较，累计正确数。

`is_better()` 优先比较验证准确率；相同则选验证损失更低的轮次，完全相同不替换。
`best.pt` 保存目前验证成绩最佳的模型，`last.pt` 保存最后完成一轮的模型。
这次最佳恰好是第 5 轮；如果第 4 轮表现更好，两份文件就会不同。

训练曲线中的准确率由每次更新前的预测累计，整轮期间模型一直在变。第 5 轮
训练过程准确率为 91.94%；训练结束后，用固定的最佳权重重新遍历训练集，得到
93.62%。两者回答的是不同问题：训练过程中总体预测如何，以及训练完的模型现在能答对多少。

## 4. 按文件理解各函数

| 文件 | 函数和作用 |
| --- | --- |
| `src/engine.py` | `seed_everything()` 设置随机种子；`resolve_device()` 选择 CPU/GPU；`training_subsets()` 复原并核对数据划分；`run_epoch()` 遍历一轮；`is_better()` 判定是否刷新最佳；`save_checkpoint()` 先写临时文件再替换；`load_checkpoint()` 从配置重建网络并恢复权重。 |
| `src/train.py` | `validate_config()` 检查配置字段和范围；`write_history()` 保存 JSON 与 CSV；`train()` 依次创建实验目录、训练、验证、保存检查点、重评估和作图；`main()` 读取命令行参数。 |
| `src/evaluate.py` | `evaluate()` 重新加载指定检查点，核对对应划分后独立评估；`main()` 解析路径、集合和运行设备。 |
| `src/plot_history.py` | `plot_history()` 读取已有历史并画出损失和准确率；`main()` 接收实验目录，调用绘图函数。 |

建议在编辑器中跟读 `train()`：先看 epoch 循环，再跳到 `run_epoch()` 理解批循环；
最后看如何用 `load_checkpoint()` 加载选出的权重，重新计算完整训练集与验证集指标。
检查点保存 `state_dict`、模型配置和轮次信息，读取时使用 `weights_only=True`。
当前已支持读取检查点评估；自动断点续训尚未实现。

## 5. 找到结果并确认口径

每次运行在 `outputs/<实验名>/` 保存以下内容：

- `config.json` 和 `metadata.json`：超参数、运行环境、划分指纹和源文件指纹。
- `history.json` 和 `history.csv`：每轮损失、准确率、样本数、学习率和耗时。
- `best.pt` 和 `last.pt`：检查点；大型权重不进入 Git。
- `metrics.json`：最佳模型在训练、验证集上的重新评估结果。
- `learning_curves.png`：损失与准确率曲线。
- `status.json`：运行状态；捕获到错误会记录失败原因，已写入的日志与检查点保留。

实测记录见 [第四阶段基线实验](../results/stage4_baseline_5epochs/README.md)。
模型第 5 轮最优：训练集 50,556／54,000，验证集 5,456／6,000。
独立评估重新加载 `best.pt` 后得到相同的验证结果。

本阶段未评估官方测试集。下一阶段先在验证集上完成参数选择和对照设计；
方案确定后，再使用独立评估入口的 `--split test` 计算完整 10,000 张的测试成绩。
测试成绩必须与对应的检查点一起记录；验证集 90.93% 还不能用于判断测试目标是否达成。

自查：为什么要清梯度？验证调用为什么不传优化器？为什么最后一批必须按样本数加权？
如果最后一轮的验证准确率下降，你会加载哪份权重？

参考：[PyTorch 优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、
[PyTorch 模型保存和读取](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。
