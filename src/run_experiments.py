"""执行第五阶段实验清单，根据验证集封存选择，然后用固定权重测试。"""

import argparse
import hashlib
import json
from pathlib import Path

from src.data import PROJECT_ROOT, save_json
from src.evaluate import evaluate
from src.train import train


def read_json(path: Path) -> dict:
    """读取 UTF-8 JSON 对象，用于实验清单、配置和运行记录。"""
    return json.loads(path.read_text(encoding="utf-8"))


def ranked_runs(records: list[dict]) -> list[dict]:
    """按验证准确率降序、验证损失升序、实验名排序，返回新的列表。"""
    return sorted(
        records, key=lambda row: (-row["val"]["accuracy"], row["val"]["loss"], row["name"])
    )


def run_comparison(entry: dict, device: str) -> dict:
    """新配置从头训练，已完成的同配置续用记录；未完成目录保留用于排查。"""
    name = entry["name"]
    config = read_json(PROJECT_ROOT / entry["config"])
    run_dir = PROJECT_ROOT / "outputs" / name
    if not run_dir.exists():
        train(config, name, device)
    if read_json(run_dir / "config.json") != config:
        raise ValueError(f"Existing run config differs: {name}")
    if read_json(run_dir / "status.json")["status"] != "completed":
        raise RuntimeError(f"Inspect status.json before reusing unfinished run: {name}")
    metrics = read_json(run_dir / "metrics.json")
    return {
        "name": name,
        "group": entry["group"],
        "config": config,
        "best_epoch": metrics["best_epoch"],
        "val": metrics["val"],
        "split_sha256": metrics["split_sha256"],
    }


def train_matrix(manifest: dict, selection_path: Path, device: str) -> None:
    """完成短对照与候选长训练，保存只依据验证结果生成的选模记录。"""
    if selection_path.exists():
        raise FileExistsError(f"Selection already saved: {selection_path}")
    comparisons = [entry for entry in manifest["runs"] if entry["group"] != "finalist"]
    records = [run_comparison(entry, device) for entry in comparisons]
    if len({record["split_sha256"] for record in records}) != 1:
        raise ValueError("Comparison runs must use the same training/validation split")
    short_winner = ranked_runs(records)[0]
    (finalist,) = [entry for entry in manifest["runs"] if entry["group"] == "finalist"]
    finalist_config = short_winner["config"] | {"epochs": 12}
    finalist_path = PROJECT_ROOT / finalist["config"]
    if finalist_path.exists() and read_json(finalist_path) != finalist_config:
        raise ValueError("Existing finalist config differs from validation winner")
    save_json(finalist_path, finalist_config)
    print(f"5-epoch validation winner: {short_winner['name']}", flush=True)
    records.append(run_comparison(finalist, device))
    for record in records:
        checkpoint = PROJECT_ROOT / "outputs" / record["name"] / "best.pt"
        record["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ordered = ranked_runs(records)
    save_json(
        selection_path,
        {
            "rule": "val accuracy descending, val loss ascending, run name ascending",
            "five_epoch_winner": short_winner["name"],
            "selected_run": ordered[0]["name"],
            "runs": ordered,
        },
    )
    print(f"Final validation selection: {ordered[0]['name']}", flush=True)


def test_matrix(selection_path: Path, device: str) -> None:
    """按已封存记录评估各组完整测试集，复核每份检查点的哈希。"""
    selection = read_json(selection_path)
    for record in selection["runs"]:
        checkpoint = PROJECT_ROOT / "outputs" / record["name"] / "best.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint differs from saved selection: {record['name']}")
        print(f"Test evaluation: {record['name']}", flush=True)
        evaluate(checkpoint, split="test", device_name=device)


def main() -> None:
    """解析设备与阶段，训练时生成选模记录，测试时读取该记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "test"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "configs/stage5_manifest.json"
    )
    parser.add_argument(
        "--selection", type=Path, default=PROJECT_ROOT / "outputs/stage5_selection.json"
    )
    args = parser.parse_args()
    if args.phase == "train":
        train_matrix(read_json(args.manifest), args.selection, args.device)
    else:
        test_matrix(args.selection, args.device)


if __name__ == "__main__":
    main()
