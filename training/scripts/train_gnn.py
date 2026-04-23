import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.classification_service import ClassificationService
from backend.services.ocr_service import ocr_service

class DocumentGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DocumentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

def prepare_train_data(data_dir):
    """
    准备训练数据
    data_dir: 训练数据目录，包含多个子目录，每个子目录对应一个类别
    """
    classification_service = ClassificationService()
    train_data = []
    
    # 遍历每个类别目录
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # 获取类别索引
        if class_name not in classification_service.class_to_idx:
            continue
        label = classification_service.class_to_idx[class_name]
        
        # 处理每个文档
        for filename in os.listdir(class_dir):
            if not filename.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                continue
            
            file_path = os.path.join(class_dir, filename)
            print(f"Processing {file_path}...")
            
            # 进行OCR处理
            try:
                doc_id = f"train_{filename}"
                ocr_result = ocr_service.process_document(file_path, filename, doc_id)
                
                # 构建图
                graph = classification_service._build_graph(ocr_result.regions, ocr_result.tables)
                
                # 添加到训练数据
                train_data.append((graph, label))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return train_data

def train(train_data, epochs=100, learning_rate=0.01, model_path="training/models/gnn_model.pth"):
    """
    训练GNN模型
    """
    if not train_data:
        print("No training data found!")
        return
    
    # 准备数据
    sample_graph = train_data[0][0]
    in_channels = sample_graph['node_features'].shape[1]
    hidden_channels = 128
    out_channels = 5  # 5个文档类别
    
    # 定义模型
    model = DocumentGNN(in_channels, hidden_channels, out_channels)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for graph, label in train_data:
            # 准备数据
            x = torch.tensor(graph['node_features'], dtype=torch.float32)
            edge_index = torch.tensor(graph['edges'], dtype=torch.long)
            batch = torch.zeros(graph['num_nodes'], dtype=torch.long)
            y = torch.tensor([label], dtype=torch.long)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(x, edge_index, batch)
            loss = criterion(output, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算损失和准确率
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == y).sum().item()
            total += 1
        
        # 打印训练信息
        avg_loss = total_loss / len(train_data)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    # 准备训练数据
    data_dir = "training/data/train"  # 训练数据目录
    train_data = prepare_train_data(data_dir)
    
    # 训练模型
    model = train(train_data, epochs=100)
    
    print("Training completed!")
