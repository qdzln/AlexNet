"""用小批合成数据核对训练、评估统计与模型检查点，不运行完整训练。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.data import PROJECT_ROOT, make_loader
from src.engine import is_better, load_checkpoint, run_epoch, save_checkpoint
from src.model import AlexNet
from src.train import train, validate_config

CPU = torch.device("cpu")


def read_baseline() -> dict:
    """读取和训练入口使用的同一份基线配置，供测试改变单个字段。"""
    return json.loads((PROJECT_ROOT / "configs" / "baseline.json").read_text(encoding="utf-8"))


class TrainingTests(unittest.TestCase):
    """验证训练流程会影响实验结论的行为。"""

    def test_metrics_count_every_sample_and_weight_tail_batch(self) -> None:
        """对比已知 logits 的逐样本统计，确保不足一批的最后一张也计入。"""
        pixels = torch.tensor([0.0, 0.25, 1.0, 0.25, 0.75])
        labels = torch.tensor([1, 0, 0, 1, 1])
        images = pixels.view(-1, 1, 1, 1).expand(-1, 1, 28, 28)
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2))
        with torch.no_grad():
            model[-1].weight.copy_(torch.tensor([[2.0], [0.0]]))
            model[-1].bias.zero_()
        logits = torch.tensor([[-2.0, 0], [-1.0, 0], [2.0, 0], [-1.0, 0], [1.0, 0]])
        expected_loss = nn.functional.cross_entropy(logits, labels).item()
        metrics = run_epoch(model, make_loader(TensorDataset(images, labels), 2), CPU)
        self.assertEqual((metrics["correct"], metrics["samples"]), (3, 5))
        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)

    def test_training_updates_parameters_but_evaluation_does_not(self) -> None:
        """训练更新权重和 BN 统计；评估不更新它们，也不计算新梯度。"""
        model = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 2),
        )
        images = torch.full((3, 1, 28, 28), 0.75)
        loader = make_loader(TensorDataset(images, torch.tensor([0, 0, 0])), 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        old_bias = model[-1].bias.detach().clone()
        run_epoch(model, loader, CPU, optimizer)
        self.assertTrue(model.training)
        self.assertEqual(model[0].num_batches_tracked.item(), 2)
        self.assertFalse(torch.equal(old_bias, model[-1].bias))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer.zero_grad(set_to_none=True)
        run_epoch(model, loader, CPU)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_empty_split_and_non_finite_loss_fail(self) -> None:
        """拒绝空数据和无效预测，避免给没有意义的模型记录成功指标。"""
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2))
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            run_epoch(
                model,
                make_loader(
                    TensorDataset(torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long))
                ),
                CPU,
            )
        with torch.no_grad():
            model[-1].bias.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Non-finite loss"):
            run_epoch(
                model, make_loader(TensorDataset(torch.zeros(1, 1, 28, 28), torch.tensor([0]))), CPU
            )

    def test_best_uses_accuracy_before_loss(self) -> None:
        """准确率优先、同准确率比较损失、完全相同不替换已有权重。"""
        best = {"accuracy": 0.8, "loss": 0.3}
        self.assertTrue(is_better(best, None))
        self.assertTrue(is_better({"accuracy": 0.9, "loss": 0.4}, best))
        self.assertFalse(is_better({"accuracy": 0.7, "loss": 0.1}, best))
        self.assertTrue(is_better({"accuracy": 0.8, "loss": 0.2}, best))
        self.assertFalse(is_better(best, best))

    def test_checkpoint_reconstructs_options_weights_and_predictions(self) -> None:
        """启用 BN 和 Dropout 的模型保存后，可按原结构与参数恢复。"""
        model = AlexNet(use_bn=True, dropout=0.4).eval()
        config = read_baseline() | {"use_bn": True, "dropout": 0.4}
        payload = {
            "format_version": 1,
            "config": config,
            "epoch": 2,
            "model_state": model.state_dict(),
        }
        images = torch.ones(1, 1, 96, 96)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, payload)
            restored, info = load_checkpoint(path, CPU)
            self.assertFalse(path.with_suffix(".tmp").exists())
            self.assertEqual(info["epoch"], 2)
            self.assertTrue(restored.use_bn)
            self.assertEqual(restored.dropout, 0.4)
            self.assertFalse(restored.training)
            with torch.no_grad():
                torch.testing.assert_close(restored(images), model(images))
            save_checkpoint(path, payload | {"format_version": 999})
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                load_checkpoint(path, CPU)

    def test_invalid_config_and_interrupted_run_leave_clear_status(self) -> None:
        """拒绝非法配置；数据错误写入失败记录，重名不会覆盖该记录。"""
        config = read_baseline()
        for values in ({"epochs": True}, {"learning_rate": float("nan")}, {"momentum": 0.9}):
            with self.assertRaises(ValueError):
                validate_config(config | values)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("src.train.PROJECT_ROOT", Path(directory)),
            patch("src.train.training_subsets", side_effect=RuntimeError("test data error")),
        ):
            with self.assertRaisesRegex(RuntimeError, "test data error"):
                train(config, "failed_run", "cpu")
            status = Path(directory) / "outputs" / "failed_run" / "status.json"
            text = status.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text)["status"], "failed")
            with self.assertRaises(FileExistsError):
                train(config, "failed_run", "cpu")
            self.assertEqual(status.read_text(encoding="utf-8"), text)


if __name__ == "__main__":
    unittest.main()
