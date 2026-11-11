"""读取 Fashion-MNIST，生成可复现的分层划分，并演示批量读取。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_dataset(train: bool, download: bool = False) -> datasets.FashionMNIST:
    """读取官方训练或测试数据，返回可按索引访问的数据集。

    train=True 读取官方 60,000 张数据，False 读取独立的 10,000 张测试数据。
    download 控制缺少文件时是否联网。这里只转成张量，暂不缩放或标准化。
    """
    # ToTensor 将 uint8 的 [0,255] 像素转为 float32 的 [0,1]，并增加通道维。
    return datasets.FashionMNIST(
        root=DATA_ROOT, train=train, download=download, transform=transforms.ToTensor()
    )


def stratified_split(
    targets: torch.Tensor, val_per_class: int = 600, seed: int = 42
) -> tuple[list[int], list[int]]:
    """按类别随机划分，返回互不重叠的训练索引和验证索引。

    targets 是一维整数标签；val_per_class 是每类验证样本数；seed 固定随机顺序。
    要求标签恰为 0 到 9，并且每类划分后仍有训练样本；无效输入抛出 ValueError。
    """
    if targets.ndim != 1 or targets.dtype != torch.long:
        raise ValueError("targets must be a one-dimensional int64 tensor")
    if sorted(targets.unique().tolist()) != list(range(10)):
        raise ValueError("targets must contain all labels from 0 to 9")
    if val_per_class < 1:
        raise ValueError("val_per_class must be positive")
    # 独立的生成器不会改变其他代码的全局随机状态。
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = torch.where(targets == label)[0]
        if len(indices) <= val_per_class:
            raise ValueError(f"class {label} needs more than {val_per_class} samples")
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(shuffled[:val_per_class].tolist())
        train_indices.extend(shuffled[val_per_class:].tolist())
    return train_indices, val_indices


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """把数据集包装为批量读取器，返回每批图像与标签。

    batch_size 控制每批数量；训练时 shuffle=True，评估时 False。
    seed 固定打乱顺序；Windows 教学阶段使用单进程，避免多进程入口问题。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


def save_json(path: Path, content: dict) -> None:
    """将字典保存为 UTF-8 JSON；自动创建父目录，无返回值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_data(download: bool = False, seed: int = 42) -> dict:
    """准备数据并保存划分与统计，返回摘要；不训练模型或计算测试准确率。

    download 控制首次下载，seed 控制划分。摘要记录索引哈希，以核对实验数据一致性。
    """
    full_train = load_dataset(train=True, download=download)
    test = load_dataset(train=False, download=download)
    if len(full_train) != 60000 or len(test) != 10000:
        raise ValueError("Unexpected Fashion-MNIST dataset size")
    train_indices, val_indices = stratified_split(full_train.targets, seed=seed)
    split = {"seed": seed, "train_indices": train_indices, "val_indices": val_indices}
    split_path = DATA_ROOT / "splits" / f"seed_{seed}.json"
    # 相同种子不允许静默替换已有划分；算法变动后应明确创建新实验。
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != split:
        raise ValueError(f"Existing split differs: {split_path}")
    save_json(split_path, split)
    digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode("utf-8")).hexdigest()
    train_set = Subset(full_train, train_indices)
    images, labels = next(iter(make_loader(train_set, shuffle=True, seed=seed)))
    summary = {
        "dataset": "FashionMNIST",
        "seed": seed,
        "split_sha256": digest,
        "split_file": split_path.relative_to(PROJECT_ROOT).as_posix(),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test),
        "class_names": CLASS_NAMES,
        "train_class_counts": torch.bincount(full_train.targets[train_indices]).tolist(),
        "val_class_counts": torch.bincount(full_train.targets[val_indices]).tolist(),
        "test_class_counts": torch.bincount(test.targets).tolist(),
        "batch_shape": list(images.shape),
        "labels_shape": list(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "batch_pixel_range": [images.min().item(), images.max().item()],
    }
    save_json(PROJECT_ROOT / "results" / f"data_summary_seed_{seed}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    """解析命令行参数并准备数据；用 --download 明确开启首次下载。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="下载缺失的数据文件")
    parser.add_argument("--seed", type=int, default=42, help="划分和批次打乱的随机种子")
    args = parser.parse_args()
    prepare_data(download=args.download, seed=args.seed)


if __name__ == "__main__":
    main()
