# 第一阶段验证记录

验证日期：2026-11-10。

| 项目 | 实际结果 |
| --- | --- |
| Python | 3.13.16，项目 `.venv` |
| PyTorch | 2.14.1+cu126 |
| TorchVision | 0.29.1+cu126 |
| NumPy | 2.5.4 |
| PyTorch CUDA 运行时 | 12.6 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU，4 GB |
| CPU 矩阵乘法 | 通过 |
| CUDA 矩阵乘法 | 通过 |
| Python 语法检查（警告作为错误） | 通过 |
| Git 忽略规则 | 虚拟环境、字节码均被忽略 |

检查命令：`.\.venv\Scripts\python.exe scripts/check_environment.py`。
两个设备都得到 `[[7.0, 10.0], [15.0, 22.0]]`，并通过断言。

为解决沙箱与本机账号的文件所有者差异，将本项目的确切路径加入了 Git 的
`safe.directory` 配置。没有设置通配符信任。

这里只验证了环境，还没有下载数据、训练网络或测量分类准确率。
