import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from backend.schemas.classification import ClassificationRequest, ClassificationResponse
from typing import List, Dict, Any, Optional, Tuple
from backend.schemas.ocr import OCRRegion, TableRegion, TableCell
from backend.core.config import settings
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 导入深度学习模型
from backend.models.deep_learning.gnn_model import DocumentGNN


class ClassificationService:
    def __init__(self, document_classes: Optional[List[str]] = None):
        # 初始化文本嵌入模型
        print("初始化 SentenceTransformer 模型...")
        self.text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.use_sentence_transformer = True
        print("SentenceTransformer 模型初始化成功")
        # 文档类别（支持外部传入，默认从全局配置读取）
        self.document_classes = (
            document_classes if document_classes else settings.DOCUMENT_CLASSES
        )
        # 类别到索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.document_classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.document_classes)}

        # 版面类型编码表（PP-StructureV3 layout_det_res 的 label 值）
        # 用于将 region_type 转为 one-hot 向量，作为节点特征的一部分
        self.layout_types = [
            "text",
            "title",
            "figure",
            "caption",
            "header",
            "footer",
            "table",
            "reference",
            "equation",
            "general",
        ]
        self.layout_type_to_idx = {t: i for i, t in enumerate(self.layout_types)}
        self.spatial_dim = 4  # 空间特征维度
        self.layout_type_dim = len(self.layout_types)  # 版面类型 one-hot 维度
        self.text_embed_dim = 384  # SentenceTransformer 输出维度

        # 加载训练好的模型
        self.trained_model = None
        self._load_trained_model()

    @property
    def in_channels(self) -> int:
        """节点特征总维度 = 文本嵌入 + 空间特征 + 版面类型 one-hot"""
        return self.text_embed_dim + self.spatial_dim + self.layout_type_dim

    def _encode_layout_type(self, region_type: str) -> np.ndarray:
        """
        将版面类型编码为 one-hot 向量

        Args:
            region_type: 版面类型标签，如 "title", "text", "figure" 等

        Returns:
            one-hot 向量，长度为 layout_type_dim
        """
        one_hot = np.zeros(self.layout_type_dim, dtype=np.float32)
        idx = self.layout_type_to_idx.get(
            region_type, self.layout_type_to_idx.get("general", 0)
        )
        one_hot[idx] = 1.0
        return one_hot

    def _extract_spatial_features(
        self, box: List[float], doc_width: float, doc_height: float
    ) -> np.ndarray:
        """
        提取归一化的空间特征

        使用文档宽高进行归一化，使不同分辨率的文档具有一致的空间表示。
        特征: [相对中心x, 相对中心y, 相对宽度, 相对高度]
        """
        center_x = float(box[0] + box[2]) / 2
        center_y = float(box[1] + box[3]) / 2
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])

        # 用文档尺寸归一化，得到 0~1 范围的相对坐标
        if float(doc_width) > 0 and float(doc_height) > 0:
            spatial_features = np.array(
                [
                    center_x / doc_width,
                    center_y / doc_height,
                    width / doc_width,
                    height / doc_height,
                ]
            )
        else:
            # 兜底：无文档尺寸信息时，用图内最大值归一化
            spatial_features = np.array([center_x, center_y, width, height])
            max_val = np.max(spatial_features)
            if max_val > 0:
                spatial_features = spatial_features / max_val

        return spatial_features

    def _build_spatial_edges(self, boxes, doc_width, doc_height):
        """边策略1: 原始空间距离阈值法，距离 < 对角线 15% 的连接"""
        edges = []
        if len(boxes) < 2:
            return edges

        if doc_width and doc_height and doc_width > 0 and doc_height > 0:
            diagonal = np.sqrt(doc_width**2 + doc_height**2)
        else:
            all_x = [b[0] for b in boxes] + [b[2] for b in boxes]
            all_y = [b[1] for b in boxes] + [b[3] for b in boxes]
            diagonal = np.sqrt(
                (max(all_x) - min(all_x)) ** 2 + (max(all_y) - min(all_y)) ** 2
            )

        distance_threshold = diagonal * 0.15

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                distance = self._calculate_distance(boxes[i], boxes[j])
                if distance < distance_threshold:
                    edges.append([i, j])
                    edges.append([j, i])

        return edges

    def _build_reading_order_edges(self, boxes):
        """
        边策略2: 阅读顺序连接

        模拟文档的阅读顺序（从上到下、从左到右）:
        1. 按 y 坐标将区域分组到同一"行"
        2. 行内按 x 坐标从左到右排序
        3. 连接阅读顺序上相邻的区域
        """
        if len(boxes) < 2:
            return []

        # 估算行高（用区域高度的中位数的一半作为行内 y 容差）
        heights = [box[3] - box[1] for box in boxes]
        median_height = float(np.median(heights)) if heights else 1.0
        row_threshold = max(median_height * 0.5, 1.0)

        # 为每个区域计算 (row_key, x_center, index)
        # row_key = floor(center_y / row_threshold)，将相近 y 的区域归入同一行
        indexed = []
        for i, box in enumerate(boxes):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            row_key = int(cy / row_threshold)
            indexed.append((row_key, cx, i))

        # 按 (行, 列) 排序 = 阅读顺序
        indexed.sort(key=lambda x: (x[0], x[1]))

        # 连接排序后相邻的区域
        edges = []
        sorted_indices = [item[2] for item in indexed]
        for k in range(len(sorted_indices) - 1):
            i, j = sorted_indices[k], sorted_indices[k + 1]
            edges.append([i, j])
            edges.append([j, i])

        return edges

    def _build_same_row_col_edges(self, boxes):
        """
        边策略3: 同行/同列连接

        连接在同一行（y 轴重叠大）或同一列（x 轴重叠大）的区域对，
        用于捕捉文档中的表格、多栏等布局结构。
        """
        if len(boxes) < 2:
            return []

        edges = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box_i, box_j = boxes[i], boxes[j]

                # 计算 y 轴重叠比例（判断是否同行）
                y_overlap = min(box_i[3], box_j[3]) - max(box_i[1], box_j[1])
                if y_overlap > 0:
                    y_i_height = box_i[3] - box_i[1]
                    y_j_height = box_j[3] - box_j[1]
                    min_h = min(y_i_height, y_j_height)
                    y_iou = y_overlap / min_h if min_h > 0 else 0
                else:
                    y_iou = 0

                # 计算 x 轴重叠比例（判断是否同列）
                x_overlap = min(box_i[2], box_j[2]) - max(box_i[0], box_j[0])
                if x_overlap > 0:
                    x_i_width = box_i[2] - box_i[0]
                    x_j_width = box_j[2] - box_j[0]
                    min_w = min(x_i_width, x_j_width)
                    x_iou = x_overlap / min_w if min_w > 0 else 0
                else:
                    x_iou = 0

                if y_iou > 0.3 or x_iou > 0.3:
                    edges.append([i, j])
                    edges.append([j, i])

        return edges

    def _build_hybrid_edges(self, boxes, doc_width, doc_height):
        """
        边策略4: 混合策略

        取 spatial + reading_order + same_row_col 三种边的并集，
        去重后返回。
        """
        all = []
        all.extend(self._build_spatial_edges(boxes, doc_width, doc_height))
        all.extend(self._build_reading_order_edges(boxes))
        all.extend(self._build_same_row_col_edges(boxes))

        # 去重（以 (min, max) 为 key）
        edge_set = set()
        for u, v in all:
            if u < v:
                edge_set.add((u, v))

        edges = []
        for u, v in edge_set:
            edges.append([u, v])
            edges.append([v, u])

        return edges

    def _build_graph(
        self,
        regions: List[OCRRegion],
        tables: Optional[List[TableRegion]] = None,
        doc_width: Optional[float] = None,
        doc_height: Optional[float] = None,
        edge_strategy: str = "spatial",
    ) -> Dict[str, Any]:
        """
        构建图结构

        节点特征 = 文本嵌入(384) + 空间特征(4) + 版面类型 one-hot(10) = 398 维
        边策略由 edge_strategy 参数控制

        Args:
            edge_strategy: 边构建策略
                - "spatial": 原始空间距离阈值法
                - "reading_order": 阅读顺序连接
                - "same_row_col": 同行/同列连接
                - "hybrid": 上述三种的并集
        """
        doc_width = float(doc_width) if doc_width else 0
        doc_height = float(doc_height) if doc_height else 0
        node_features = []
        all_boxes = []

        # 构建文本区域节点
        for region in regions:
            text_embedding = self.text_encoder.encode(
                region.text, convert_to_tensor=False
            )
            box = region.box
            spatial_features = self._extract_spatial_features(
                box, doc_width or 0, doc_height or 0
            )
            layout_one_hot = self._encode_layout_type(region.region_type or "general")
            node_feature = np.concatenate(
                [text_embedding, spatial_features, layout_one_hot]
            )
            node_features.append(node_feature)
            all_boxes.append(box)

        # 构建表格区域节点
        if tables:
            for table in tables:
                table_text = ""
                if table.cells:
                    table_text = " ".join(
                        [cell.text for cell in table.cells if cell.text]
                    )

                table_embedding = self.text_encoder.encode(
                    table_text, convert_to_tensor=False
                )
                box = table.box
                spatial_features = self._extract_spatial_features(
                    box, doc_width or 0, doc_height or 0
                )
                layout_one_hot = self._encode_layout_type("table")
                node_feature = np.concatenate(
                    [table_embedding, spatial_features, layout_one_hot]
                )
                node_features.append(node_feature)
                all_boxes.append(box)

        # 根据策略构建边
        strategy_map = {
            "spatial": self._build_spatial_edges,
            "reading_order": self._build_reading_order_edges,
            "same_row_col": self._build_same_row_col_edges,
            "hybrid": self._build_hybrid_edges,
        }
        build_fn = strategy_map.get(edge_strategy)
        if build_fn is None:
            raise ValueError(
                f"未知边策略: {edge_strategy}，可选: {list(strategy_map.keys())}"
            )

        if edge_strategy in ("spatial", "hybrid"):
            edges_list = build_fn(all_boxes, doc_width, doc_height)
        else:
            edges_list = build_fn(all_boxes)

        return {
            "node_features": np.array(node_features),
            "edges": np.array(edges_list).T if edges_list else np.array([[], []]),
            "num_nodes": len(node_features),
            "edge_strategy": edge_strategy,
        }

    def _calculate_distance(self, box1, box2) -> float:
        """计算两个文本框中心点之间的欧氏距离"""
        center1 = [(float(box1[0]) + float(box1[2])) / 2, (float(box1[1]) + float(box1[3])) / 2]
        center2 = [(float(box2[0]) + float(box2[2])) / 2, (float(box2[1]) + float(box2[3])) / 2]
        return float(np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2))

    def _load_trained_model(self):
        """加载训练好的GNN模型"""
        import os

        model_path = "training/models/gnn_model.pth"
        if os.path.exists(model_path):
            try:
                hidden_channels = 128
                out_channels = len(self.document_classes)

                self.trained_model = DocumentGNN(
                    self.in_channels, hidden_channels, out_channels
                )
                self.trained_model.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
                self.trained_model.eval()
                print(f"加载训练好的GNN模型成功 (in_channels={self.in_channels})")
            except Exception as e:
                print(f"加载模型失败: {e}")
                self.trained_model = None
        else:
            print("未找到训练好的模型，使用默认模型")

    def _graph_classification(
        self, graph: Dict[str, Any], model=None
    ) -> Dict[str, Any]:
        """图分类 (使用 GNN)"""
        node_features = graph["node_features"]
        edges = graph["edges"]
        num_nodes = graph["num_nodes"]

        if num_nodes == 0:
            return {"predicted_class": "未知类型", "confidence": 0.5}

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        if model is None:
            in_channels = x.shape[1]
            hidden_channels = 128
            out_channels = len(self.document_classes)
            model = DocumentGNN(in_channels, hidden_channels, out_channels)

        with torch.no_grad():
            output = model(x, edge_index, batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = self.idx_to_class[predicted_idx.item()]

        return {"predicted_class": predicted_class, "confidence": confidence.item()}

    def predict(self, request: ClassificationRequest) -> ClassificationResponse:
        """深度学习特征预处理与推理核心枢纽"""
        regions: List[OCRRegion] = request.ocr_regions
        tables: Optional[List[TableRegion]] = request.tables
        self.regions = regions

        graph = self._build_graph(regions, tables)

        if self.trained_model is not None:
            result = self._graph_classification(graph, self.trained_model)
        else:
            result = self._graph_classification(graph)

        return ClassificationResponse(
            document_id=request.document_id,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )


# 单例实例
classification_service = ClassificationService()
