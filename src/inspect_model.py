"""第三课入口：打印网络，使用真实训练图像验证前向与反向传播。"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # 支持编辑器直接运行，也支持 python -m src.inspect_model。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import Subset

from src.data import PROJECT_ROOT, load_dataset, make_loader, save_json, stratified_split
from src.model import AlexNet, describe_shapes, prepare_images


def inspect_model(
    batch_size: int = 8,
    device_name: str = "auto",
    use_bn: bool = False,
    dropout: float = 0.0,
    seed: int = 42,
    head: str = "linear",
) -> dict:
    """检查模型与真实数据是否兼容，保存结构及报告并返回报告字典。

    batch_size 控制验收批次；device_name 可为 auto/cpu/cuda；其余参数设置网络和种子。
    只求一次梯度，不创建优化器、不更新权重，也不报告分类准确率。
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset = load_dataset(train=True)
    train_indices, _ = stratified_split(dataset.targets, seed=seed)
    raw_images, labels = next(iter(make_loader(Subset(dataset, train_indices), batch_size)))
    images = prepare_images(raw_images.to(device))
    labels = labels.to(device)
    model = AlexNet(use_bn=use_bn, dropout=dropout, head=head).to(device)
    print(model)
    shapes = describe_shapes(model, images)
    for row in shapes:
        print(f"{row['name']:25s} -> {str(row['output_shape']):20s} {row['parameters']:>10,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # 前向：得到分数；损失：比较分数与标签；反向：计算参数应该如何调整的梯度。
    model.train()
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    report = {
        "model": "AlexNet-FashionMNIST-96",
        "seed": seed,
        "device": str(device),
        "use_bn": use_bn,
        "dropout": dropout,
        "head": head,
        "batch_size": len(labels),
        "input_shape": list(images.shape),
        "output_shape": list(logits.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_batch_loss": loss.item(),
        "all_gradients_finite": True,
        "optimizer_steps": 0,
        "layers": shapes,
    }
    if device.type == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    # 文件名区分网络选项与设备，防止不同验收配置相互覆盖。
    tag = f"{device.type}_bn{int(use_bn)}_dropout{dropout:g}_batch{batch_size}_seed{seed}"
    if head != "linear":
        tag += f"_{head}"
    output = PROJECT_ROOT / "results" / f"model_summary_{tag}.json"
    save_json(output, report)
    output.with_suffix(".txt").write_text(str(model) + "\n", encoding="utf-8")
    print(f"Parameters: {report['parameters']:,}; backward passed; optimizer steps: 0")
    print(f"Saved: {output}")
    return report


def main() -> None:
    """读取命令行选项并执行网络验收，缺少数据时提示先完成第二阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head", choices=("linear", "attention"), default="linear")
    args = parser.parse_args()
    inspect_model(args.batch_size, args.device, args.batch_norm, args.dropout, args.seed, args.head)


if __name__ == "__main__":
    main()
