# 第一阶段：先建立一个可复现的工作环境

## 1. 为什么使用虚拟环境

Python 是解释器，PyTorch 是解释器加载的第三方库。不同项目可能需要不同版本的库。
`python -m venv .venv` 为当前项目创建独立环境；并不会把模型或数据放进去。
`.venv/Scripts/python.exe` 就是本项目应该使用的解释器。
编辑器中也应选择这个解释器，而不是系统 Python。

`python -m pip` 的含义是“让这个 Python 执行它对应的 pip”，防止依赖装错环境。
PyTorch 用于定义网络、自动求导和训练；TorchVision 用于读取图像数据集。
使用 TorchVision 的数据集读取功能不等于调用现成的 AlexNet 模型。

## 2. 读懂第一个 Python 文件

打开 `scripts/check_environment.py`，按以下顺序阅读：

1. 文件开头的三引号是模块说明，介绍文件用途与运行方法。
2. `import` 导入标准库和第三方库，不会开始训练。
3. `show_versions()` 打印版本与解释器路径，先排除“装好了却导入失败”的环境混淆。
4. `check_tensor_computation(device)` 把同一个计算放到指定设备执行，减少重复代码。
5. `torch.tensor` 创建张量，`device` 决定张量位于内存还是显存。
6. `matrix @ matrix` 做矩阵乘法，这是神经网络中的基础运算。
7. `assert_close` 比较实际结果和已知答案；只有真正计算成功才能通过检查。
8. `main()` 组织执行顺序，先验证 CPU，再验证 CUDA。
9. `if __name__ == "__main__"` 是脚本入口，避免被其他文件导入时自动运行。

函数签名中的 `device: torch.device` 是参数类型提示；`-> None` 表示没有返回值。
类型提示帮助阅读和静态检查，本身不执行自动类型转换。

## 3. 为什么不能只看 nvidia-smi

`nvidia-smi` 能看到显卡，说明驱动能识别硬件；不代表当前 Python 已安装 GPU 版 PyTorch。
因此需要同时检查 `torch.cuda.is_available()` 并实际执行 CUDA 张量计算。
`nvidia-smi` 的 CUDA 字段和 `torch.version.cuda` 不一定相同；分别描述驱动支持与库运行时。

## 4. Git 与 GitHub

Git 记录本地代码历史；GitHub 保存远程仓库并方便提交作业。
`.gitignore` 排除虚拟环境、原始数据和大体积权重，防止仓库膨胀。
最终提交的是可复现代码、依赖说明、实验配置与报告。
后续阶段会讲 `git status`、`git add`、`git commit`、`git push` 的作用。

## 5. 本阶段掌握标准

能找到项目解释器，能运行检查脚本，并能解释以下三个问题：

- 为什么依赖装在 `.venv`，而不是全部装进系统 Python？
- `check_tensor_computation` 为什么接收 `device` 参数？
- 为什么检查 GPU 时还要执行一次真实计算？

下一阶段才开始下载数据、认识标签、可视化每类 10 张图像。
