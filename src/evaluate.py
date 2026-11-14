"""读取已保存模型，独立评估指定集合；默认评估验证集。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_dataset, make_loader, save_json
from src.engine import load_checkpoint, resolve_device, run_epoch, training_subsets


def evaluate(checkpoint_path: Path, split: str = "val", device_name: str = "auto") -> dict:
    """重建网络并评估完整集合，返回指标；只有显式 split=test 才加载测试集。

    train/val 使用检查点保存的划分种子并核对哈希。测试集必须完整覆盖 10,000 张。
    输出 JSON 同时记录检查点 SHA-256，避免把不同模型的结果混在一起。
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    if split == "test":
        dataset = load_dataset(train=False)
        if len(dataset) != 10000:
            raise ValueError("Test evaluation requires all 10000 examples")
    else:
        train_set, val_set, digest = training_subsets(config["split_seed"])
        if digest != checkpoint["split_sha256"]:
            raise ValueError("Checkpoint split does not match current data")
        dataset = train_set if split == "train" else val_set
    metrics = run_epoch(model, make_loader(dataset, config["batch_size"]), device)
    result = {
        "split": split,
        "epoch": checkpoint["epoch"],
        "metrics": metrics,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    destination = checkpoint_path.with_name(f"{checkpoint_path.stem}_{split}_evaluation.json")
    save_json(destination, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    """接收检查点路径和评估集合，默认 val；实验设计锁定后才使用 test。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.device)


if __name__ == "__main__":
    main()
