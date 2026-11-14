"""手写适配 Fashion-MNIST 的 AlexNet，不调用现成模型或预训练权重。"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = 96


def prepare_images(images: torch.Tensor) -> torch.Tensor:
    """将 ToTensor 后的图像批次转换为模型输入，返回 [N,1,96,96]。

    images 必须是 float32、[N,1,H,W]、像素在 [0,1] 的有限张量。
    双线性插值放大图像后用固定公式 (x-0.5)/0.5 映射到 [-1,1]。
    不估计任何数据统计量，因此训练、验证、测试可使用完全相同的处理。
    """
    if images.ndim != 4 or images.shape[1] != 1 or min(images.shape) < 1:
        raise ValueError("Expected a nonempty [N,1,H,W] grayscale batch")
    if images.dtype != torch.float32:
        raise ValueError("Expected float32 images from ToTensor")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Expected finite pixels in [0,1]; do not normalize twice")
    resized = F.interpolate(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (resized - 0.5) / 0.5


class AttentionPool(nn.Module):
    """让四个空间位置交换信息，再汇总为一个分类特征向量。"""

    def __init__(self) -> None:
        """创建位置参数、四头自注意力和残差后的层归一化。"""
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, 4, 256))
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """把 [N,256,2,2] 变成四个 token，并返回 [N,256] 的汇总特征。"""
        # 每个 token 对应一个空间位置，包含该位置的全部 256 个通道。
        tokens = features.flatten(2).transpose(1, 2) + self.position
        # Q、K、V 均来自这组 token，所以称为自注意力。
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + attended).mean(dim=1)


class AlexNet(nn.Module):
    """灰度图 AlexNet 适配版：5 个卷积层、3 个全连接层，输出 10 类原始分数。

    不复刻原论文的双 GPU 分组连接与 LRN；隐藏全连接层缩为 512。
    输入固定为 [N,1,96,96]，BN 与 Dropout 可独立开关用于后续实验。
    """

    def __init__(self, use_bn: bool = False, dropout: float = 0.0, head: str = "linear") -> None:
        """创建所有可训练层；use_bn 控制批归一化，dropout 是随机丢弃概率。

        默认关闭 BN 和 Dropout，作为对照实验基线；不代表原论文的默认设置。
        Dropout 必须满足 0 <= dropout < 1。head 选择线性或注意力分类头。
        """
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if head not in ("linear", "attention"):
            raise ValueError("head must be linear or attention")
        self.use_bn = use_bn
        self.dropout = dropout
        self.head = head
        # 每项依次为：输入通道、输出通道、卷积核大小、步长、填充。
        specifications = [
            (1, 96, 11, 4, 2),
            (96, 256, 5, 1, 2),
            (256, 384, 3, 1, 1),
            (384, 384, 3, 1, 1),
            (384, 256, 3, 1, 1),
        ]
        layers = OrderedDict()
        for number, (in_channels, out_channels, kernel, stride, padding) in enumerate(
            specifications, start=1
        ):
            layers[f"conv{number}"] = nn.Conv2d(
                in_channels, out_channels, kernel, stride, padding, bias=not use_bn
            )
            # BN 放在卷积之后、ReLU 之前；BN 已有可学习偏移，此时省略卷积偏置。
            if use_bn:
                layers[f"bn{number}"] = nn.BatchNorm2d(out_channels)
            layers[f"relu{number}"] = nn.ReLU()
            if number in (1, 2, 5):
                layers[f"pool{number}"] = nn.MaxPool2d(kernel_size=3, stride=2)
        self.features = nn.Sequential(layers)
        if head == "attention":
            self.attention_pool = AttentionPool()
            self.classifier = nn.Sequential(
                OrderedDict(
                    [
                        ("fc2", nn.Linear(256, 512)),
                        ("relu7", nn.ReLU()),
                        ("dropout2", nn.Dropout(dropout)),
                        ("fc3", nn.Linear(512, 10)),
                    ]
                )
            )
            return
        self.flatten = nn.Flatten(start_dim=1)
        # 最后一层池化后为 [N,256,2,2]，展平后每张图有 1024 个特征。
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(256 * 2 * 2, 512)),
                    ("relu6", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(512, 512)),
                    ("relu7", nn.ReLU()),
                    ("dropout2", nn.Dropout(dropout)),
                    ("fc3", nn.Linear(512, 10)),
                ]
            )
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """执行前向传播，把 [N,1,96,96] 图像转换为 [N,10] logits。

        调用前先执行 prepare_images；不在这里做 softmax，因为训练使用的
        CrossEntropyLoss 接收原始分数。预测类别可取 logits.argmax(dim=1)。
        """
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Expected [N,1,96,96]; call prepare_images first")
        features = self.features(images)
        vector = (
            self.attention_pool(features) if self.head == "attention" else self.flatten(features)
        )
        return self.classifier(vector)


def describe_shapes(model: AlexNet, images: torch.Tensor) -> list[dict]:
    """通过真实前向计算记录各层形状与参数量，返回可保存为 JSON 的记录。

    model 与 images 应在同一设备。临时使用评估模式，避免检查时修改 BN 统计量；
    finally 恢复每个模块原先的训练状态并移除 hook，即使计算失败也会清理。
    """
    records = []
    handles = []
    modes = {module: module.training for module in model.modules()}

    def record_shape(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """记录叶子层或整个注意力汇总模块的输入、输出形状与参数量。"""
        records.append(
            {
                "name": names[module],
                "type": type(module).__name__,
                "input_shape": list(inputs[0].shape),
                "output_shape": list(output.shape),
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
            }
        )

    names = {module: name for name, module in model.named_modules() if name}
    try:
        for module in names:
            # 注意力内部有位置参数和多返回值子层，整体打印可避免漏算或重复计算。
            if names[module].startswith("attention_pool."):
                continue
            if isinstance(module, AttentionPool) or not list(module.children()):
                handles.append(module.register_forward_hook(record_shape))
        model.eval()
        with torch.no_grad():
            model(images)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training
    return records
