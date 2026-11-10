r"""第一课：检查 Python、项目依赖和 GPU 是否能真正执行计算。

从项目根目录执行：.\.venv\Scripts\python.exe scripts/check_environment.py
本脚本不下载数据，也不训练模型。
"""

import importlib.metadata
import platform
import sys

import torch
import torchvision


def show_versions() -> None:
    """打印解释器路径和依赖版本，帮助确认程序使用的是项目虚拟环境。

    参数：无。
    返回：无；将检查结果输出到终端。
    """
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"NumPy: {importlib.metadata.version('numpy')}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")


def check_tensor_computation(device: torch.device) -> None:
    """在指定设备执行矩阵乘法，并核对计算结果。

    参数：device，执行计算的设备，例如 CPU 或 CUDA 显卡。
    返回：无；结果不正确时抛出异常，防止误报环境正常。
    """
    # 张量可以理解为多维数组；这里创建一个 2 行 2 列的矩阵。
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    # @ 表示矩阵乘法，不能与逐元素相乘的 * 混淆。
    result = matrix @ matrix
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]], device=device)
    torch.testing.assert_close(result, expected)
    print(f"Matrix multiplication passed on {device}: {result.tolist()}")


def main() -> None:
    """依次显示版本、验证 CPU，再在 CUDA 可用时验证显卡计算。

    参数：无。
    返回：无；没有可用显卡时明确报告 CPU 模式。
    """
    show_versions()
    check_tensor_computation(torch.device("cpu"))
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        check_tensor_computation(torch.device("cuda:0"))
    else:
        print("CUDA unavailable: CPU works, but GPU training is not ready.")


# 直接运行文件时进入 main；其他文件导入它时，不自动执行检查。
if __name__ == "__main__":
    main()
