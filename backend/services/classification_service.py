from backend.schemas.classification import ClassificationRequest, ClassificationResponse
from typing import List, Dict, Any, Optional
from backend.schemas.ocr import OCRRegion, TableRegion, TableCell
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 导入深度学习模型
from backend.models.deep_learning.gnn_model import DocumentGNN

class ClassificationService:
    def __init__(self):
        # 初始化文本嵌入模型
        print("初始化 SentenceTransformer 模型...")
        self.text_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.use_sentence_transformer = True
        print("SentenceTransformer 模型初始化成功")
        # 文档类别
        self.document_classes = ['公文', '信件', '表单', '报告', '合同']
        # 类别到索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.document_classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.document_classes)}
        # 加载训练好的模型
        self.trained_model = None
        self._load_trained_model()

    def _calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个文本框之间的欧氏距离
        """
        # 计算中心点
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        # 计算欧氏距离
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _build_graph(self, regions: List[OCRRegion], tables: Optional[List[TableRegion]] = None) -> Dict[str, Any]:
        """
        构建图结构
        """
        # 节点特征
        node_features = []
        # 边
        edges = []
        
        # 构建文本区域节点
        for region in regions:
            # 文本嵌入
            text_embedding = self.text_encoder.encode(region.text, convert_to_tensor=False)
            
            # 空间特征 (归一化坐标)
            box = region.box
            # 计算框的中心坐标和大小
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            # 空间特征向量
            spatial_features = np.array([center_x, center_y, width, height])
            # 归一化空间特征
            if np.max(spatial_features) > 0:
                spatial_features = spatial_features / np.max(spatial_features)
            # 合并文本和空间特征
            node_feature = np.concatenate([text_embedding, spatial_features])
            node_features.append(node_feature)
        
        # 构建表格区域节点
        if tables:
            for table in tables:
                # 表格文本内容（从HTML或单元格中提取）
                table_text = ""
                if table.cells:
                    table_text = " ".join([cell.text for cell in table.cells if cell.text])
                
                # 文本嵌入
                table_embedding = self.text_encoder.encode(table_text, convert_to_tensor=False)
                
                # 空间特征
                box = table.box
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                spatial_features = np.array([center_x, center_y, width, height])
                if np.max(spatial_features) > 0:
                    spatial_features = spatial_features / np.max(spatial_features)
                # 合并特征
                node_feature = np.concatenate([table_embedding, spatial_features])
                node_features.append(node_feature)
        
        # 构建边
        distance_threshold = 100  # 距离阈值
        for i in range(len(node_features)):
            for j in range(i + 1, len(node_features)):
                # 对于文本区域，使用其box计算距离
                if i < len(regions):
                    box1 = regions[i].box
                else:
                    # 对于表格区域
                    table_idx = i - len(regions)
                    box1 = tables[table_idx].box if tables and table_idx < len(tables) else [0, 0, 0, 0]
                
                if j < len(regions):
                    box2 = regions[j].box
                else:
                    # 对于表格区域
                    table_idx = j - len(regions)
                    box2 = tables[table_idx].box if tables and table_idx < len(tables) else [0, 0, 0, 0]
                
                distance = self._calculate_distance(box1, box2)
                if distance < distance_threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # 无向图
        
        return {
            'node_features': np.array(node_features),
            'edges': np.array(edges).T if edges else np.array([[], []]),
            'num_nodes': len(node_features)
        }

    def _load_trained_model(self):
        """
        加载训练好的GNN模型
        """
        import os
        
        model_path = "training/models/gnn_model.pth"
        if os.path.exists(model_path):
            try:
                # 计算输入通道数（文本嵌入维度 + 空间特征维度）
                # 文本嵌入维度：384 (paraphrase-multilingual-MiniLM-L12-v2)
                # 空间特征维度：4 (center_x, center_y, width, height)
                in_channels = 384 + 4
                hidden_channels = 128
                out_channels = len(self.document_classes)
                
                # 初始化模型并加载权重
                self.trained_model = DocumentGNN(in_channels, hidden_channels, out_channels)
                self.trained_model.load_state_dict(torch.load(model_path))
                self.trained_model.eval()
                print("加载训练好的GNN模型成功")
            except Exception as e:
                print(f"加载模型失败: {e}")
                self.trained_model = None
        else:
            print("未找到训练好的模型，使用默认模型")

    def _graph_classification(self, graph: Dict[str, Any], model=None) -> Dict[str, Any]:
        """
        图分类 (使用 GNN)
        """
        # 获取图特征
        node_features = graph['node_features']
        edges = graph['edges']
        num_nodes = graph['num_nodes']
        
        if num_nodes == 0:
            # 没有节点，返回默认分类
            return {
                'predicted_class': '未知类型',
                'confidence': 0.5
            }
        
        # 准备数据
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)  # 单个图
        
        # 使用提供的模型或创建新模型
        if model is None:
            # 初始化模型
            in_channels = x.shape[1]
            hidden_channels = 128
            out_channels = len(self.document_classes)
            model = DocumentGNN(in_channels, hidden_channels, out_channels)
        
        # 前向传播
        with torch.no_grad():
            output = model(x, edge_index, batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = self.idx_to_class[predicted_idx.item()]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence.item()
        }

    def predict(self, request: ClassificationRequest) -> ClassificationResponse:
        """
        深度学习特征预处理与推理核心枢纽
        """
        regions: List[OCRRegion] = request.ocr_regions
        tables: Optional[List[TableRegion]] = request.tables
        self.regions = regions  # 保存 regions 供分类使用

        # 1. 图节点构建 (Graph Nodes Construction)
        # 2. 边构建 (Edges Construction)
        graph = self._build_graph(regions, tables)

        # 3. GNN 模型前向传播推理
        if self.trained_model is not None:
            # 使用训练好的模型
            result = self._graph_classification(graph, self.trained_model)
        else:
            # 使用未训练的模型
            result = self._graph_classification(graph)

        return ClassificationResponse(
            document_id=request.document_id,
            predicted_class=result['predicted_class'],
            confidence=result['confidence']
        )

# 单例实例
classification_service = ClassificationService()
