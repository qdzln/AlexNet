"""从训练子集中每类选 10 张原始图像，保存带类别标签的 10×10 网格。"""

import argparse
from pathlib import Path

import matplotlib

# 使用不依赖桌面窗口的后端，终端和自动化环境都能生成图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import CLASS_NAMES, PROJECT_ROOT, load_dataset, save_json, stratified_split


def select_examples(
    targets: torch.Tensor,
    train_indices: list[int],
    per_class: int = 10,
    seed: int = 42,
) -> list[list[int]]:
    """仅从训练索引中按类随机选图，返回每一类的原始数据索引列表。

    per_class 指每类数量，seed 控制选图；候选不足时抛出 ValueError。
    """
    if per_class < 1:
        raise ValueError("per_class must be positive")
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.tensor(train_indices, dtype=torch.long)
    selected = []
    for label in range(10):
        indices = candidates[targets[candidates] == label]
        if len(indices) < per_class:
            raise ValueError(f"Not enough training examples for class {label}")
        order = torch.randperm(len(indices), generator=generator)[:per_class]
        selected.append(indices[order].tolist())
    return selected


def plot_examples(output_path: Path, seed: int = 42) -> None:
    """绘制并保存每类 10 张图像和索引清单；output_path 为 PNG 路径。

    seed 同时指定数据划分与选图随机种子；要求先运行数据准备命令。
    """
    dataset = load_dataset(train=True)
    train_indices, _ = stratified_split(dataset.targets, seed=seed)
    selected = select_examples(dataset.targets, train_indices, seed=seed)
    figure, axes = plt.subplots(10, 10, figsize=(12, 12))
    for label, indices in enumerate(selected):
        for column, index in enumerate(indices):
            axis = axes[label, column]
            # 用原始 uint8 图像绘图，固定色阶，避免每张图独立拉伸对比度。
            axis.imshow(dataset.data[index].numpy(), cmap="gray", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.set_ylabel(
                    f"{label}: {CLASS_NAMES[label]}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=10,
                )
    figure.suptitle("Fashion-MNIST | 10 training examples per class", fontsize=16)
    figure.subplots_adjust(left=0.18, right=0.99, top=0.94, bottom=0.02, wspace=0.12, hspace=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    save_json(output_path.with_suffix(".json"), {"seed": seed, "indices_by_class": selected})
    print(f"Saved 100 training images: {output_path}")


def main() -> None:
    """读取选图种子，生成对应的图片及 JSON 索引清单，无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot_examples(PROJECT_ROOT / "results" / f"class_examples_seed_{args.seed}.png", args.seed)


if __name__ == "__main__":
    main()
