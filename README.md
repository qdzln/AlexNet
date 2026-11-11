# AlexNet 实现 Fashion-MNIST 分类

这是一个逐阶段完成的教学项目。当前阶段：数据读取、划分与可视化。
已完成环境搭建和数据模块；模型、训练和实验报告将在后续阶段编写，尚无准确率结果。

## 学习顺序与验收

| 阶段 | 编写内容 | 验收产物 |
| --- | --- | --- |
| 1 环境 | 虚拟环境、Git 规范、环境检查 | CPU/GPU 真实计算通过 |
| 2 数据 | `src/data.py`、`src/visualize.py` | 每类 10 张，共 100 张图；固定数据划分 |
| 3 网络 | `src/model.py` | 手写 AlexNet；打印结构与张量形状 |
| 4 训练 | `src/train.py`、`src/evaluate.py` | 检查点、训练曲线、三个集合的准确率 |
| 5 对照 | BN、Dropout、动量、注意力 | 配置与每次结果记录，含失败实验 |
| 6 交付 | 报告、复现说明、GitHub | 完整测试集准确率目标 >90%，代码提交 |

## 目录职责

- `scripts/`：环境检查等辅助脚本。
- `src/`：后续的数据处理、网络、训练、评估代码。
- `configs/`：实验超参数配置。
- `tests/`：模型形状、数据划分、评估计算等关键逻辑测试。
- `docs/`：每阶段的学习讲解与最终报告。
- `results/`：用于提交的实验结果表和精选图片。
- `data/`、`outputs/`：运行时数据、权重和日志，Git 忽略。
- `.venv/`：本项目独立 Python 环境，Git 忽略。

## 第一阶段运行方式（PowerShell）

以下命令均在项目根目录执行。已经创建环境后，不必重复创建。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe scripts/check_environment.py
```

直接指定虚拟环境解释器，避免终端激活策略与全局 Python 混淆。
实际依赖版本已记录在 `requirements-lock.txt`。相同平台重建时，使用锁定版本：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

该清单对应 Windows、Python 3.13、CUDA 12.6 构建；其他平台需要重新选择兼容构建。
GPU 依赖体积较大，首次下载需要时间。

## 第二阶段运行方式

```powershell
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe -m src.visualize
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

逐函数说明见 [第二阶段教程](docs/02_data.md)。首次准备后可省略 `--download` 离线运行。
划分索引保存在 `data/splits/`，摘要和样例图保存在 `results/`。
第二阶段增加了 Matplotlib 绘图依赖和 Ruff 开发工具，版本统一保存在锁定清单。

## 实验规则

1. Fashion-MNIST 是 10 类、28×28 单通道灰度图；首层输入通道必须匹配。
2. 从官方 60,000 张训练数据中按类别固定划出 6,000 张验证集，其余 54,000 张训练。
3. 官方 10,000 张测试数据只用于最终评估，不用于选模型、调参和选训练轮次。
4. 手写网络层组合，禁止直接调用 `torchvision.models.alexnet` 或加载其预训练模型。
5. 网络设计阶段明确记录对原始 AlexNet 的输入尺寸、通道、分类头等适配。
6. 每次实验保存配置、种子、划分、最佳轮次、指标和耗时；用验证集选检查点。
7. 训练集最终准确率也用评估模式计算，避免 Dropout 和数据增强干扰比较。
8. 分别比较 BN、Dropout、SGD 动量；更换优化器后重新用验证集调学习率。
9. 注意力实验真正替换部分全连接结构，并记录参数量和速度变化。
10. 对照实验预先定义，固定划分与预算；最终测试结果不得用于反向选择超参数。
11. 测试准确率为正确数量 / 10000；是否达到 90% 以上由真实结果决定。

## 开发约定

使用清晰的 snake_case 命名、类型提示、中文函数文档字符串；注释解释原因和关键步骤。
每个阶段验证后提交一次 Git，代码与生成数据分开管理。
GitHub 仓库计划命名 `AlexNet`；所属账号和可见性待提交阶段确定。

参考：[PyTorch 安装说明](https://pytorch.org/get-started/locally/)、
[Fashion-MNIST 官方项目](https://github.com/zalandoresearch/fashion-mnist)。
