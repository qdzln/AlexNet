"""验证网络与数据的接口、梯度和训练/评估模式，不下载数据。"""

import unittest

import torch
from torch import nn

from src.model import AlexNet, describe_shapes, prepare_images


class ModelTests(unittest.TestCase):
    """覆盖基线、BN/Dropout 选项及模型检查工具的状态隔离。"""

    def test_preprocessing(self) -> None:
        """黑白图映射到 -1/+1，并保持批次、通道与目标尺寸。"""
        source = torch.stack([torch.zeros(1, 28, 28), torch.ones(1, 28, 28)])
        result = prepare_images(source)
        self.assertEqual(tuple(result.shape), (2, 1, 96, 96))
        torch.testing.assert_close(result[0], -torch.ones_like(result[0]))
        torch.testing.assert_close(result[1], torch.ones_like(result[1]))

    def test_invalid_inputs(self) -> None:
        """拒绝 RGB、未成批、越界、非有限和整数像素，避免静默使用错误数据。"""
        for source in (
            torch.zeros(2, 3, 28, 28),
            torch.zeros(1, 28, 28),
            torch.full((1, 1, 28, 28), 2.0),
            torch.full((1, 1, 28, 28), float("nan")),
            torch.zeros(1, 1, 28, 28, dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError):
                prepare_images(source)
        with self.assertRaises(ValueError):
            AlexNet()(torch.zeros(1, 1, 28, 28))
        for dropout in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                AlexNet(dropout=dropout)

    def test_shapes_and_backward(self) -> None:
        """基线及 BN/Dropout 版本都输出 10 类，并使全部参数得到有限梯度。"""
        for use_bn, dropout in ((False, 0.0), (True, 0.5)):
            with self.subTest(use_bn=use_bn):
                model = AlexNet(use_bn, dropout)
                output = model(prepare_images(torch.rand(2, 1, 28, 28)))
                self.assertEqual(tuple(output.shape), (2, 10))
                nn.CrossEntropyLoss()(output, torch.tensor([0, 9])).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_architecture_and_shape_trace(self) -> None:
        """检查 5 卷积/3 全连接与关键空间尺寸，避免展平层误吞批次维。"""
        model = AlexNet()
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 5)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.modules()), 3)
        rows = describe_shapes(model, torch.zeros(1, 1, 96, 96))
        shapes = {row["name"]: row["output_shape"] for row in rows}
        self.assertEqual(shapes["features.pool1"], [1, 96, 11, 11])
        self.assertEqual(shapes["features.pool2"], [1, 256, 5, 5])
        self.assertEqual(shapes["features.pool5"], [1, 256, 2, 2])
        self.assertEqual(shapes["flatten"], [1, 1024])

    def test_inspection_preserves_bn_state_and_removes_hooks(self) -> None:
        """打印结构不能修改 BN 均值或模式，失败后也不能遗留 hook。"""
        model = AlexNet(use_bn=True, dropout=0.5)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        describe_shapes(model, torch.zeros(2, 1, 96, 96))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            describe_shapes(model, torch.zeros(1, 1, 28, 28))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_eval_is_repeatable_with_dropout(self) -> None:
        """评估模式下关闭 Dropout 随机丢弃，同一输入得到同一输出。"""
        model = AlexNet(use_bn=True, dropout=0.5).eval()
        images = torch.randn(2, 1, 96, 96)
        with torch.no_grad():
            torch.testing.assert_close(model(images), model(images))


if __name__ == "__main__":
    unittest.main()
