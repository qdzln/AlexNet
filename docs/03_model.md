# 第三阶段：手写 AlexNet，理解每一层在做什么

本阶段实现网络、打印结构，验证真实数据的前向和反向传播。尚不进行参数更新，
也不评估分类准确率。阅读顺序：本教程 → `src/model.py` → `src/inspect_model.py`。

## 1. 为什么采用适配版

Fashion-MNIST 原图为 28×28 单通道。如果直接套用大图 AlexNet 的卷积与池化参数，
空间尺寸会过早变小。这里先放大为 96×96，再进入网络；放大不会增加新的图像信息，
只是让输入尺寸匹配网络的卷积与池化结构。

本项目明确使用 **AlexNet-FashionMNIST-96 适配版**，不是原论文逐项复刻：

| 项目 | 本项目设计 |
| --- | --- |
| 主体结构 | 保留 5 个卷积层、3 个全连接层，使用 ReLU 与最大池化 |
| 输入 | 单通道 96×96，适应灰度数据与本机算力 |
| 卷积输出通道 | 96、256、384、384、256 |
| 卷积核 | 11×11、5×5、3×3、3×3、3×3 |
| 隐藏全连接层 | 从原论文的 4096 缩小到 512，降低参数量 |
| 输出 | 10 类，而非 ImageNet 的 1000 类 |
| 分组连接与 LRN | 不复刻原论文双 GPU 分组连接，不使用局部响应归一化 LRN |
| BN | 可选，卷积后、ReLU 前；基线关闭 |
| Dropout | 可选，两个隐藏全连接层之后；基线概率为 0 |

原论文使用 Dropout；本项目默认关闭是为了建立可比较的实验基线，不是声称原版没有它。
隐藏层大小和输入尺寸是当前工程选择，不是已经通过验证集选出的最优配置。
后续正式实验固定这套结构，再分别比较 BN、Dropout、动量和注意力。

## 2. 先看完整数据流

```text
原始批次 [N,1,28,28]，像素 [0,1]
  → prepare_images：放大到 96×96，固定映射到 [-1,1]
  → features：5 次卷积、ReLU、3 次池化
  → flatten：保留批次维，展平每张图的特征
  → classifier：3 个全连接层
  → logits [N,10]：每张图对应 10 个原始类别分数
```

`N` 是一批图片的数量，网络不要求它固定为 64。
`prepare_images` 必须放在训练、验证、测试的共同处理流程中，不能只用于训练。
`(x-0.5)/0.5` 是固定缩放，不是从全数据估计均值，不会使用测试统计量。
不要对已经变成 [-1,1] 的数据重复调用该函数。

## 3. 卷积与池化怎样改变尺寸

以下层的 dilation 均为 1，使用向下取整，空间尺寸计算公式为：

```text
输出尺寸 = floor((输入尺寸 + 2×padding - kernel_size) / stride) + 1
```

第一层例子：`floor((96 + 2×2 - 11) / 4) + 1 = 23`。

| 层 | 核 / 步长 / 填充 | 输出形状（省略批次 N） | 作用 |
| --- | --- | --- | --- |
| 输入 | — | 1×96×96 | 灰度图 |
| conv1 + ReLU | 11 / 4 / 2 | 96×23×23 | 学习局部模式 |
| pool1 | 3 / 2 / 0 | 96×11×11 | 缩小空间尺寸 |
| conv2 + ReLU | 5 / 1 / 2 | 256×11×11 | 组合前层特征 |
| pool2 | 3 / 2 / 0 | 256×5×5 | 再次压缩 |
| conv3 + ReLU | 3 / 1 / 1 | 384×5×5 | 学习更复杂的模式 |
| conv4 + ReLU | 3 / 1 / 1 | 384×5×5 | 继续组合特征 |
| conv5 + ReLU | 3 / 1 / 1 | 256×5×5 | 形成分类特征 |
| pool5 | 3 / 2 / 0 | 256×2×2 | 最后一次空间压缩 |
| flatten | — | 1024 | 把特征图展平 |
| fc1 + ReLU + Dropout | 1024 → 512 | 512 | 学习分类组合 |
| fc2 + ReLU + Dropout | 512 → 512 | 512 | 继续转换特征 |
| fc3 | 512 → 10 | 10 | 输出类别分数 |

ReLU 把负数变成 0，提供非线性，不改变张量形状。
最大池化在局部窗口取最大值，不含可训练参数。
`Flatten(start_dim=1)` 从通道维开始展平，保留第 0 维的批次，不能把不同图片混在一起。

## 4. 逐函数理解 model.py

### prepare_images(images)

检查数据的通道、形状、类型和范围，使用双线性插值放大，最后固定缩放。
输出仍保留在输入所在设备，CPU 与 CUDA 都能使用。

### AlexNet.__init__(use_bn, dropout)

`AlexNet` 继承 `nn.Module`，使 PyTorch 能自动管理参数、设备和训练状态。
`super().__init__()` 必须先初始化父类。所有训练层在这里创建，不能每次 forward 都重新创建。

用配置列表逐层构造 `nn.Conv2d`，没有调用任何现成 AlexNet。
`OrderedDict` 给层命名，所以输出可读为 `features.conv1` 而不是一串难记的数字。
`nn.Sequential` 负责按顺序执行已经创建的层。

开启 BN 后使用 `Conv → BN → ReLU`。BN 有自己的可学习偏移，因此卷积不再保留偏置。
Dropout 的 `p=0.5` 表示训练时随机丢弃约一半激活，不是永久删除一半神经元。

### AlexNet.forward(images)

定义输入怎样流过已有层。通常使用 `model(images)`，由 PyTorch 调用 forward，
这样框架还能执行 hook 等机制。不建议在业务代码里直接调用 `model.forward(...)`。

输出是 logits，可能为负，不要求和为 1。
`CrossEntropyLoss` 内部处理所需的对数概率运算，训练前不要再套一层 softmax。
需要概率展示时可使用 `logits.softmax(dim=1)`；预测编号使用 `logits.argmax(dim=1)`。

### describe_shapes(model, images)

临时安装 forward hook，在各层运行后收集形状和参数量。
检查期间用 `eval()` 与 `no_grad()`，避免更新 BN 统计或构建梯度图。
`finally` 恢复各层模式并移除 hook，检查失败也不会污染后续训练。

## 5. inspect_model.py 为什么计算损失

`inspect_model(...)` 从真实训练子集中取一批图片，建立网络、打印结构，执行：

```python
logits = model(images)                      # 前向：从图像得到类别分数
loss = nn.CrossEntropyLoss()(logits, labels) # 损失：分数与正确标签相差多少
loss.backward()                            # 反向：求每个参数的梯度
```

这里没有 `optimizer.step()`，权重没有更新。开启 BN 的验收前向会更新其运行统计，
但这只是一次性检查对象，不保存模型权重供后续训练使用。
报告中的 `initial_batch_loss` 是随机初始化网络在一批样本上的损失，不能据此比较模型优劣。

`model.eval()` 和 `torch.no_grad()` 不相同：前者改变 BN/Dropout 行为，后者关闭梯度记录。
正式评估通常两者一起使用。正式训练会使用 `model.train()` 并允许记录梯度。

## 6. 亲自运行

在项目根目录运行以下命令，已有第二阶段数据即可，不需要联网或安装新依赖。

```powershell
# 自动选择 GPU；默认一次检查 8 张真实训练图像
.\.venv\Scripts\python.exe -m src.inspect_model

# 也支持编辑器直接运行这个入口文件
.\.venv\Scripts\python.exe src/inspect_model.py

# 检查开启 BN 和 Dropout 后，网络仍然能够运行
.\.venv\Scripts\python.exe -m src.inspect_model --batch-norm --dropout 0.5

# 显式指定 CPU，或检查 GPU 上 64 张图片的批次
.\.venv\Scripts\python.exe -m src.inspect_model --device cpu --batch-size 2
.\.venv\Scripts\python.exe -m src.inspect_model --device cuda --batch-size 64

# 运行项目测试
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

入口会在 `results/model_summary_*.json` 保存逐层形状、参数量、验收配置与梯度检查结果，
并在同名 `.txt` 文件保存打印的结构。命令可重复执行，修改选项会得到不同文件名。
CUDA 报告的 `peak_allocated_mib` 是该检查过程 PyTorch 张量峰值分配量，
不是整张显卡占用，也不是正式训练显存保证；优化器状态还会占用额外内存。

## 7. 本阶段自查

1. 为什么最后的 256×2×2 要变成 1024，而不是包含批次 N 的更大数字？
2. `__init__` 与 `forward` 分别负责什么？
3. 为什么训练时不在 fc3 后面添加 softmax？
4. 为什么 `loss.backward()` 通过还不等于已经训练好模型？

下一阶段编写训练循环、验证流程、最佳权重保存和训练曲线。
达到测试准确率目标需要真实训练验证，本阶段没有给出准确率结论。

## 实际验收记录

- 基线参数总数：4,516,554；开启 BN 后为 4,517,930。
- CPU：2 张真实训练图像完成前向与反向传播，全部参数梯度存在且为有限值。
- CUDA：64 张真实训练图像完成基线前向与反向，输出形状为 `[64,10]`。
- CUDA：开启 BN 与 Dropout=0.5，8 张真实图像的前向与反向通过。
- 完整测试集成检查：12 项测试通过，其中本阶段新增 6 项模型测试。
- Ruff 代码规范检查通过；模块入口与直接运行入口均实际执行成功。
- 以上验收均没有优化器更新，报告中的 `optimizer_steps` 为 0。

网络权重使用 PyTorch 基础层默认初始化；没有加载预训练权重。
验收取训练子集的首批样本，仅用于接口与梯度检查；正式训练必须打乱批次顺序。

参考：[AlexNet 原论文](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)、
[PyTorch Conv2d 说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。
