"""从保存的训练历史绘制损失与准确率曲线，无需重新训练。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history_path: Path, output_path: Path) -> None:
    """读取历史 JSON 并保存两幅曲线；训练曲线是更新过程指标，验证曲线是轮末指标。"""
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError("No completed epochs to plot")
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric in zip(axes, ("loss", "accuracy"), strict=True):
        for split, label in (("train", "Training (during updates)"), ("val", "Validation")):
            values = [row[f"{split}_{metric}"] for row in history]
            axis.plot(epochs, values, marker="o", label=label)
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), xticks=epochs)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_ylim(0, 1)
    figure.suptitle("AlexNet / Fashion-MNIST")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """接收训练输出目录，将 history.json 转为 learning_curves.png。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plot_history(args.run_dir / "history.json", args.run_dir / "learning_curves.png")


if __name__ == "__main__":
    main()
