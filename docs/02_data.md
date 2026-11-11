# 第二阶段：从图片到神经网络可以读取的批次

本阶段完成数据准备和每类 10 张图像展示，不进行网络训练。

## 1. 先理解数据

Fashion-MNIST 的一条样本由“图像”和“标签”组成。图像是 28×28 的灰度像素，
标签是 0～9 的整数。数字是类别编号，没有大小顺序的含义。

| 标签 | 英文 | 中文 |
| --- | --- | --- |
| 0 | T-shirt/top | T 恤/上衣 |
| 1 | Trouser | 裤子 |
| 2 | Pullover | 套头衫 |
| 3 | Dress | 连衣裙 |
| 4 | Coat | 外套 |
| 5 | Sandal | 凉鞋 |
| 6 | Shirt | 衬衫 |
| 7 | Sneaker | 运动鞋 |
| 8 | Bag | 包 |
| 9 | Ankle boot | 短靴 |

原始像素为 uint8 整数，范围是 0～255。`ToTensor()` 将它变成 float32，
除以 255 后落在 [0,1]，并把形状变成 `[1,28,28]`。这里的 1 是灰度通道数。
这一步只是缩放像素，并没有做均值/标准差标准化，也没有把图片复制为 RGB 三通道。

## 2. 三个集合为什么分开

```text
官方训练集 60,000 张
  ├─ 训练子集 54,000 张：更新模型参数，每类 5,400 张
  └─ 验证子集  6,000 张：选超参数与最佳轮次，每类 600 张
官方测试集 10,000 张：最终评价，不参与调参，每类 1,000 张
```

可以把训练集理解为练习题，验证集是调整学习方法时使用的模拟题，测试集是最终考试。
如果看着测试成绩不断调参数，最后的成绩就不再是可信的独立评价。
本阶段只检查测试集的大小和标签数量，不计算模型预测结果。

## 3. 按函数阅读 src/data.py

1. `load_dataset(train, download)`：调用 TorchVision 的数据读取接口。
   `train=True` 指官方 60,000 张数据，不是已经划分好的 54,000 张子集。
   下载和解码使用现成接口，后续 AlexNet 网络结构仍由我们手写。
2. `stratified_split(targets, val_per_class, seed)`：对每类分别打乱，先取 600 个
   索引做验证，剩下做训练。这叫分层划分，能够保持类别平衡。
3. `make_loader(...)`：将数据集分批读取，训练时打乱顺序，评估时保持顺序。
4. `save_json(...)`：保存人可以阅读的记录格式，方便检查数据与复现实验。
5. `prepare_data(...)`：串联以上步骤，保存完整索引和摘要，并检查一个训练批次。
6. `main()`：解析 `--download` 和 `--seed`，把终端参数传给业务函数。

索引是图片在原始数据中的位置，标签是它的类别，两者不能混用。
`Subset(dataset, indices)` 用索引形成一个视图，不必把图片复制到新的文件夹。
随机种子让同一程序产生相同随机顺序；不是“模型一定得到同样准确率”的保证。
我们还保存完整索引和 SHA-256 摘要：摘要像指纹，用来核对划分是否发生变化。
同种子已有索引与新划分不一致时程序报错，避免静默更换实验数据。

## 4. Dataset 与 DataLoader 的区别

`Dataset` 回答“第 i 张图是什么”；`DataLoader` 回答“下一批图是什么”。

当 batch_size=64 时：

```text
images.shape = [64, 1, 28, 28]
                │  │   │   └─ 宽度
                │  │   └───── 高度
                │  └───────── 通道数
                └──────────── 每批图片数
labels.shape = [64]
```

`shuffle=True` 打乱读取顺序，不会改变训练/验证成员。
`drop_last=False` 保留最后不足 64 张的一批，尤其评估时不能漏掉样本。
`num_workers=0` 表示在当前进程读取，便于 Windows 初学阶段调试。
`next(iter(loader))` 只取第一批检查形状，尚未进行任何模型训练。

## 5. 按函数阅读 src/visualize.py

- `select_examples(...)`：仅从训练子集随机选出每类 10 个索引。
- `plot_examples(...)`：创建 10 行×10 列的图，每行同一类别；保存 PNG 和选图索引。
- `main()`：读取随机种子并确定输出路径。

用原始图像展示，便于观察真实数据；统一使用 0～255 的灰度范围。
类别使用英文，避免不同电脑缺少中文字体造成方框；中文对照表见上文。
`Agg` 后端不需要弹出图形窗口，因此远程机器和自动测试也能保存图片。

## 6. 亲自运行

在项目根目录执行：

```powershell
# 第一次允许下载；已经下载过会复用本地文件
.\.venv\Scripts\python.exe -m src.data --download

# 后续可离线检查并生成相同的数据划分
.\.venv\Scripts\python.exe -m src.data

# 生成每类 10 张图片
.\.venv\Scripts\python.exe -m src.visualize

# 运行关键逻辑测试和代码规范检查
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

`-m src.data` 表示把 `src` 包中的 `data` 模块当作程序执行。
这样项目内部导入路径保持统一，不需要手动修改 `sys.path`。

产物：

- `data/FashionMNIST/raw/`：原始数据，不进入 Git。
- `data/splits/seed_42.json`：完整划分索引，不进入 Git，可按命令重建。
- `results/data_summary_seed_42.json`：大小、类别计数、张量形状和划分指纹。
- `results/class_examples_seed_42.png`：100 张图像网格，供作业展示。
- `results/class_examples_seed_42.json`：这 100 张图的原始索引。

## 7. 本阶段自查

1. 为什么 `train=True` 不直接等于我们的训练子集？
2. `[64,1,28,28]` 的四个数字分别是什么？
3. 为什么不能根据测试集成绩选择学习率？
4. 为什么每类分别划分，而不是截取原始数据的前 6,000 张作为验证集？

## 实际验收记录（2026-11-11）

- 官方数据下载成功，训练/验证/测试数量为 54,000 / 6,000 / 10,000。
- 每类分别为 5,400 / 600 / 1,000 张，标签范围为 0～9。
- 首批图像为 float32、形状 `[64,1,28,28]`、像素范围 [0,1]；标签为 int64。
- 重复离线运行数据准备后，划分 SHA-256 与首次一致，完整值见结果摘要。
- 5 项单元测试通过，覆盖划分隔离、完整性、种子、边界输入、选图与批次尾部。
- Ruff 代码检查与 pip 依赖一致性检查通过。
- 已查看生成的 10×10 网格，100 张图片及每行类别标签完整可见。
- 还没有训练网络，因此本阶段不报告分类准确率。

下一阶段手写 AlexNet，并逐层计算输出形状。当前的 28×28 仅用于数据观察，
不代表后续直接使用原始 AlexNet 的大卷积核和池化参数。

接口参考：[TorchVision FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)。
数据来源：[Fashion-MNIST 官方项目](https://github.com/zalandoresearch/fashion-mnist)，数据许可见其 MIT LICENSE。
