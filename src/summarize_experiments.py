"""归档实验配置、逐轮指标和评估结果，并生成对照表与验证曲线。"""

import argparse
import csv
import json
import re
from pathlib import Path
from shutil import copyfile

import matplotlib

from src.data import PROJECT_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS = (
    "config.json",
    "metadata.json",
    "status.json",
    "history.json",
    "history.csv",
    "metrics.json",
    "best_test_evaluation.json",
)
CONFIG_FIELDS = (
    "epochs",
    "batch_size",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "momentum",
    "use_bn",
    "dropout",
    "head",
    "seed",
    "split_seed",
)
METRIC_FIELDS = ("accuracy", "loss", "correct", "samples")


def read_optional(path: Path, default: dict | list) -> dict | list:
    """读取现有 JSON，尚未产生该文件时返回调用者指定的默认值。"""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def collect_run(entry: dict, output: Path) -> tuple[dict, list[dict]]:
    """归档一组轻量记录，返回汇总行与逐轮历史，缺失指标留空。"""
    name = entry["name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError(f"Invalid run name: {name}")
    source = PROJECT_ROOT / "outputs" / name
    archive = output / "runs" / name
    archive.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACTS:
        if (source / filename).is_file():
            copyfile(source / filename, archive / filename)
    config = read_optional(source / "config.json", {})
    planned = read_optional(PROJECT_ROOT / entry["config"], {})
    if config and config != planned:
        raise ValueError(f"Saved config differs from manifest: {name}")
    status = read_optional(source / "status.json", {})
    metrics = read_optional(source / "metrics.json", {})
    scores = {split: metrics.get(split) or {} for split in ("train", "val", "test")}
    test = read_optional(source / "best_test_evaluation.json", {})
    if test:
        if test["epoch"] != metrics.get("best_epoch") or test["metrics"]["samples"] != 10000:
            raise ValueError(f"Test evaluation does not match best epoch/full test split: {name}")
        scores["test"] = test["metrics"]
    history = read_optional(source / "history.json", [])
    row = {
        "name": name,
        "group": entry["group"],
        "status": status.get("status", "missing"),
        "error": status.get("error"),
        "best_epoch": metrics.get("best_epoch"),
        "seconds": metrics.get("seconds"),
        "split_sha256": metrics.get("split_sha256"),
        "checkpoint_sha256": test.get("checkpoint_sha256"),
    }
    row.update({field: (config or planned).get(field) for field in CONFIG_FIELDS})
    row["head"] = row["head"] or "linear"
    for split, score in scores.items():
        for field in METRIC_FIELDS:
            row[f"{split}_{field}"] = score.get(field)
    return row, history


def plot_validation(histories: list[tuple[dict, list[dict]]], output: Path) -> None:
    """分别绘制 Adam 分类头与 SGD 动量学习率的验证准确率曲线。"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for row, history in histories:
        if not history or row["group"] == "finalist":
            continue
        is_sgd = row["optimizer"] == "sgd"
        label = (
            f"SGD m={row['momentum']:g}, lr={row['learning_rate']:g}"
            if is_sgd
            else {"bn": "BN"}.get(row["group"], row["group"].capitalize())
        )
        axes[int(is_sgd)].plot(
            [epoch["epoch"] for epoch in history],
            [epoch["val_accuracy"] * 100 for epoch in history],
            marker="o",
            label=label,
        )
    titles = ("Adam and architecture changes", "SGD and momentum")
    for axis, title in zip(axes, titles, strict=True):
        axis.set(title=title, xlabel="Epoch", ylabel="Validation accuracy (%)", xticks=range(1, 6))
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(loc="lower right")
    axes[0].set_ylim(80, 94)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def summarize(manifest_path: Path, output: Path) -> None:
    """将清单中的每组结果写入 comparison.csv 并保存验证曲线。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    histories = [collect_run(entry, output) for entry in manifest["runs"]]
    rows = [row for row, _ in histories]
    (baseline,) = [row for row in rows if row["group"] == "baseline"]
    for row in rows:
        for split in ("val", "test"):
            score, base = row[f"{split}_accuracy"], baseline[f"{split}_accuracy"]
            row[f"{split}_delta_pp"] = (
                None if score is None or base is None else (score - base) * 100
            )
            row[f"{split}_relative_percent"] = (
                None if score is None or not base else (score / base - 1) * 100
            )
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_validation(histories, output / "validation_curves.png")


def main() -> None:
    """解析清单和输出目录，使用已有实验文件生成汇总。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "configs/stage5_manifest.json"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/stage5")
    args = parser.parse_args()
    summarize(args.manifest, args.output)


if __name__ == "__main__":
    main()
