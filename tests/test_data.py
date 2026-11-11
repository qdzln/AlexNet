"""使用合成标签验证划分和选图逻辑，不联网，也不依赖真实数据文件。"""

import unittest

import torch

from src.data import make_loader, stratified_split
from src.visualize import select_examples


class DataTests(unittest.TestCase):
    """验证影响实验可信度的数据隔离、可复现性和样本覆盖。"""

    def setUp(self) -> None:
        """为每个测试创建 10 类、每类 20 个样本的合成标签。"""
        self.targets = torch.arange(10).repeat_interleave(20)

    def test_split_is_disjoint_complete_and_balanced(self) -> None:
        """验证无重复、无交集、无遗漏，并保持每类的训练与验证数量。"""
        train, val = stratified_split(self.targets, val_per_class=4)
        self.assertEqual(len(set(train)), 160)
        self.assertEqual(len(set(val)), 40)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(200)))
        self.assertEqual(torch.bincount(self.targets[train]).tolist(), [16] * 10)
        self.assertEqual(torch.bincount(self.targets[val]).tolist(), [4] * 10)

    def test_seed_reproduces_split(self) -> None:
        """相同种子重现划分，不同种子改变划分，且不改变全局随机状态。"""
        state = torch.random.get_rng_state().clone()
        first = stratified_split(self.targets, 4, 42)
        self.assertEqual(first, stratified_split(self.targets, 4, 42))
        self.assertNotEqual(first, stratified_split(self.targets, 4, 43))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_invalid_split_is_rejected(self) -> None:
        """拒绝空验证集、无训练样本、缺少类别或错误标签形状。"""
        for count in (0, 20, 21):
            with self.assertRaises(ValueError):
                stratified_split(self.targets, count)
        for targets in (self.targets[:20], self.targets.reshape(10, 20)):
            with self.assertRaises(ValueError):
                stratified_split(targets, 4)

    def test_visualization_uses_only_training_examples(self) -> None:
        """检查选图每类 10 张、标签正确、无重复且不含验证样本。"""
        train, val = stratified_split(self.targets, 4)
        selected = select_examples(self.targets, train)
        self.assertEqual(selected, select_examples(self.targets, train))
        for label, indices in enumerate(selected):
            self.assertEqual(len(set(indices)), 10)
            self.assertTrue(set(indices) <= set(train))
            self.assertFalse(set(indices) & set(val))
            self.assertEqual(self.targets[indices].tolist(), [label] * 10)
        with self.assertRaises(ValueError):
            select_examples(self.targets, train, per_class=17)

    def test_loader_keeps_last_partial_batch(self) -> None:
        """评估不能丢弃最后不足一批的样本，并验证打乱后无遗漏。"""
        dataset = torch.utils.data.TensorDataset(torch.arange(10))
        batches = list(make_loader(dataset, batch_size=4, shuffle=True))
        self.assertEqual([len(batch[0]) for batch in batches], [4, 4, 2])
        self.assertEqual(
            sorted(torch.cat([batch[0] for batch in batches]).tolist()), list(range(10))
        )


if __name__ == "__main__":
    unittest.main()
