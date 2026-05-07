import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm


class DocumentGIN(nn.Module):
    """
    文档分类的图同构网络模型（GIN）

    与 GCN / GAT 的对比:
    - GCN: 按邻居度归一化平均聚合
    - GAT: 注意力加权聚合
    - GIN: 带可学习参数 ε 的 MLP 聚合，理论上表达能力最强
           (能区分 GCN/GAT 无法区分的某些图结构)

    结构: 3层 GINConv + BatchNorm + ReLU + Dropout → global_mean_pool → FC
    输入: 节点特征 (文本嵌入384维 + 空间特征4维 + 版面类型10维 = 398维)
    输出: 文档类别 logits
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(DocumentGIN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_channels if i == 0 else hidden_channels
            mlp = Sequential(
                Linear(input_dim, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            # train_eps=True: 让 ε 可学习，区分自身特征和邻居特征
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden_channels))

        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = self.dropout(self.relu(bn(conv(x, edge_index))))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x
