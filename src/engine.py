"""训练与评估共用的小函数：指标统计、固定划分与检查点读取。"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data import DATA_ROOT, load_dataset, stratified_split
from src.model import AlexNet, prepare_images


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 随机种子，减少同环境重复运行的随机差异。

    不保证跨硬件或跨 PyTorch 版本逐位一致；CUDA 卷积禁用自动测速选算法。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    """将 auto/cpu/cuda 转为设备；显式要求不可用的 CUDA 时立即报错。"""
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def training_subsets(split_seed: int) -> tuple[Subset, Subset, str]:
    """读取官方训练数据，返回训练/验证子集和划分哈希；完全不加载测试集。

    与第二阶段相同算法重建划分，并核对已保存的索引，避免实验划分悄悄变化。
    """
    dataset = load_dataset(train=True)
    if len(dataset) != 60000:
        raise ValueError("Expected 60000 official training examples")
    train_indices, val_indices = stratified_split(dataset.targets, seed=split_seed)
    split = {"seed": split_seed, "train_indices": train_indices, "val_indices": val_indices}
    saved = DATA_ROOT / "splits" / f"seed_{split_seed}.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != split:
        raise ValueError("Saved split differs from reconstructed split")
    digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode("utf-8")).hexdigest()
    return Subset(dataset, train_indices), Subset(dataset, val_indices), digest


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """完整遍历 loader，返回按样本加权的 loss、accuracy、correct、samples。

    提供 optimizer 时训练并更新参数，否则进入 eval/no_grad 评估模式。
    训练指标累计自每次更新前的预测，最终固定模型的训练准确率需另外评估。
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, samples = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for raw_images, labels in loader:
            images = prepare_images(raw_images.to(device))
            labels = labels.to(device)
            if training:
                # PyTorch 默认累积梯度，每个批次开始前要清空上一次的梯度。
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; run stopped without reporting success")
            if training:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            # 最后一批可能不足 batch_size，不能对每批平均损失再次直接取平均。
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += count
    if samples == 0:
        raise ValueError("Cannot measure an empty dataset")
    return {
        "loss": total_loss / samples,
        "accuracy": correct / samples,
        "correct": correct,
        "samples": samples,
    }


def is_better(candidate: dict, best: dict | None) -> bool:
    """验证准确率更高则替换最佳模型；相同时比较更低的验证损失。"""
    return best is None or (candidate["accuracy"], -candidate["loss"]) > (
        best["accuracy"],
        -best["loss"],
    )


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """先写临时文件再替换目标，降低写入中断损坏已有检查点的风险。"""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[AlexNet, dict]:
    """读取本项目检查点，根据保存的配置重建网络，返回评估模式模型与记录。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = checkpoint["config"]
    # 第四阶段的检查点没有 head 字段，仍按原来的线性分类头重建。
    head_options = {"head": config["head"]} if "head" in config else {}
    model = AlexNet(use_bn=config["use_bn"], dropout=config["dropout"], **head_options)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
