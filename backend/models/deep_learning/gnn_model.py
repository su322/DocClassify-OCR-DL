import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm


class DocumentGNN(nn.Module):
    """
    文档分类的图神经网络模型

    结构: 3层 GCNConv + BatchNorm + ReLU + Dropout → global_mean_pool → FC
    输入: 节点特征 (文本嵌入384维 + 空间特征4维 + 版面类型10维 = 398维)
    输出: 文档类别 logits
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DocumentGNN, self).__init__()
        # 图卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # 批归一化层（加速收敛，稳定训练）
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        # 分类头
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = self.dropout(self.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x
