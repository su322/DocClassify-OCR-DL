import torch
import torch.nn as nn
from torchvision import models


class DocumentCNN(nn.Module):
    """
    文档分类的 CNN 模型（Baseline）

    使用预训练 ResNet18 作为骨干网络，替换最后的全连接层为文档类别数。
    直接从原始像素图像进行分类，不依赖 OCR 或图结构。

    结构: ResNet18 (预训练) → FC(num_classes)
    输入: RGB 图像 (3, 224, 224)
    输出: 文档类别 logits
    """

    def __init__(self, num_classes: int):
        super(DocumentCNN, self).__init__()
        # 加载预训练 ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
