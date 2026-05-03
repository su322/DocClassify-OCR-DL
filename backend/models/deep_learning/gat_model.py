import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


class DocumentGAT(nn.Module):
    """
    文档分类的图注意力网络模型（GAT）

    与 DocumentGNN (GCN) 的对比:
    - GCN: 所有邻居节点权重相同（平均聚合）
    - GAT: 通过注意力机制自动学习邻居节点的重要性权重

    结构: 3层 GATConv + BatchNorm + ReLU + Dropout → global_mean_pool → FC
    输入: 节点特征 (文本嵌入384维 + 空间特征4维 + 版面类型10维 = 398维)
    输出: 文档类别 logits
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        """
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出类别数
            heads: 注意力头数（多头注意力），最终会拼接后投影回 hidden_channels
        """
        super(DocumentGAT, self).__init__()
        # 图注意力层（多头注意力）
        self.conv1 = GATConv(
            in_channels, hidden_channels // heads, heads=heads, concat=True
        )
        self.conv2 = GATConv(
            hidden_channels, hidden_channels // heads, heads=heads, concat=True
        )
        self.conv3 = GATConv(
            hidden_channels, hidden_channels // heads, heads=heads, concat=True
        )
        # 批归一化层
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
