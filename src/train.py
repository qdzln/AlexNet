"""第四课：训练、验证、保存最佳权重与绘图；不加载测试集。"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import PROJECT_ROOT, make_loader, save_json
from src.engine import (
    is_better,
    load_checkpoint,
    resolve_device,
    run_epoch,
    save_checkpoint,
    seed_everything,
    training_subsets,
)
from src.model import AlexNet
from src.plot_history import plot_history


def validate_config(config: dict) -> None:
    """检查配置字段与数值范围，提前拒绝拼写错误及非法超参数。"""
    required = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "momentum",
        "use_bn",
        "dropout",
        "seed",
        "split_seed",
    }
    if not required <= set(config) <= (required | {"head"}):
        raise ValueError(f"Config requires {sorted(required)}; head is optional")
    if config.get("head", "linear") not in ("linear", "attention"):
        raise ValueError("head must be linear or attention")
    for key in ("epochs", "batch_size", "seed", "split_seed"):
        if type(config[key]) is not int or config[key] < (
            1 if key in ("epochs", "batch_size") else 0
        ):
            raise ValueError(f"Invalid integer: {key}")
    if config["seed"] >= 2**32 or config["split_seed"] >= 2**32:
        raise ValueError("Seeds must be smaller than 2**32")
    for key in ("learning_rate", "weight_decay", "momentum", "dropout"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"Invalid numeric value: {key}")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if not 0 <= config["dropout"] < 1 or not 0 <= config["momentum"] < 1:
        raise ValueError("dropout and momentum must be in [0,1)")
    if config["optimizer"] not in ("adam", "sgd") or type(config["use_bn"]) is not bool:
        raise ValueError("optimizer must be adam/sgd and use_bn must be boolean")
    if config["optimizer"] == "adam" and config["momentum"] != 0:
        raise ValueError("momentum option applies only to SGD; Adam uses its own default betas")


def write_history(run_dir: Path, history: list[dict]) -> None:
    """同时保存 JSON 与 CSV 逐轮历史，便于程序读取和人工查看。"""
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(config: dict, run_name: str, device_name: str = "auto") -> dict:
    """按配置完整训练并返回最佳模型的训练/验证评估；拒绝覆盖已有实验目录。

    每轮保存最后模型，按验证准确率选最佳模型；失败时保留配置、历史和状态记录。
    此阶段不实现自动断点续训，last.pt 可用于恢复模型做评估。
    """
    validate_config(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run_name must contain only letters, digits, underscore or hyphen")
    run_dir = PROJECT_ROOT / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    started = time.perf_counter()
    history, best_metrics, best_epoch = [], None, 0
    save_json(
        run_dir / "status.json",
        {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()},
    )
    try:
        seed_everything(config["seed"])
        device = resolve_device(device_name)
        train_set, val_set, split_hash = training_subsets(config["split_seed"])
        train_loader = make_loader(train_set, config["batch_size"], True, config["seed"])
        val_loader = make_loader(val_set, config["batch_size"])
        save_json(
            run_dir / "metadata.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "split_sha256": split_hash,
                "train_size": len(train_set),
                "val_size": len(val_set),
                "source_sha256": {
                    str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted((PROJECT_ROOT / "src").glob("*.py"))
                },
            },
        )
        head_options = {"head": config["head"]} if "head" in config else {}
        model = AlexNet(config["use_bn"], config["dropout"], **head_options).to(device)
        options = {"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), **options)
        else:
            optimizer = torch.optim.SGD(model.parameters(), momentum=config["momentum"], **options)
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device)
            improved = is_better(val_metrics, best_metrics)
            if improved:
                best_metrics, best_epoch = val_metrics, epoch
            checkpoint = {
                "format_version": 1,
                "epoch": epoch,
                "config": config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "split_sha256": split_hash,
                "val_metrics": val_metrics,
                "best_epoch": best_epoch,
                "best_val_metrics": best_metrics,
            }
            save_checkpoint(run_dir / "last.pt", checkpoint)
            if improved:
                save_checkpoint(run_dir / "best.pt", checkpoint)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "train_samples": train_metrics["samples"],
                    "val_samples": val_metrics["samples"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - epoch_start,
                }
            )
            write_history(run_dir, history)
            print(
                f"Epoch {epoch}/{config['epochs']} | train {train_metrics['accuracy']:.2%} "
                f"| val {val_metrics['accuracy']:.2%} | {history[-1]['seconds']:.1f}s "
                f"| best epoch {best_epoch}",
                flush=True,
            )
        # 使用最佳验证模型重新评估；与训练过程中不断变化的模型指标区别开。
        del optimizer, model
        best_model, _ = load_checkpoint(run_dir / "best.pt", device)
        train_eval = run_epoch(best_model, make_loader(train_set, config["batch_size"]), device)
        val_eval = run_epoch(best_model, val_loader, device)
        result = {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "split_sha256": split_hash,
            "train": train_eval,
            "val": val_eval,
            "test": None,
            "test_status": "not_evaluated",
            "seconds": time.perf_counter() - started,
        }
        save_json(run_dir / "metrics.json", result)
        plot_history(run_dir / "history.json", run_dir / "learning_curves.png")
        save_json(
            run_dir / "status.json", {"status": "completed", "completed_epochs": len(history)}
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    except BaseException as error:
        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "completed_epochs": len(history),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    """解析配置路径、实验名与设备后启动训练；所有入口默认使用同一份配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.json")
    parser.add_argument(
        "--run-name", default=datetime.now(timezone.utc).strftime("baseline_%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    train(json.loads(args.config.read_text(encoding="utf-8")), args.run_name, args.device)


if __name__ == "__main__":
    main()
