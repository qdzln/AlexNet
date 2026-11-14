"""覆盖自注意力分类头与旧模型检查点的接口。"""

import unittest

import torch

from src.model import AlexNet, describe_shapes


class AttentionTests(unittest.TestCase):
    """验证分类、梯度与结构报告对新增网络分支的支持。"""

    def test_attention_forward_backward_and_parameter_count(self) -> None:
        """验证四个空间位置汇总后能分类，位置和注意力参数能接收梯度。"""
        model = AlexNet(head="attention")
        images = torch.randn(2, 1, 96, 96)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
        for parameter in model.attention_pool.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        shapes = describe_shapes(model, images)
        rows = {row["name"]: row for row in shapes}
        self.assertEqual(rows["attention_pool"]["input_shape"], [2, 256, 2, 2])
        self.assertEqual(rows["attention_pool"]["output_shape"], [2, 256])
        self.assertEqual(
            sum(row["parameters"] for row in shapes), sum(p.numel() for p in model.parameters())
        )
