import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from backend.schemas.classification import ClassificationRequest, ClassificationResponse
from typing import List, Dict, Any, Optional, Tuple
from backend.schemas.ocr import OCRRegion, TableRegion, TableCell
from backend.core.config import settings
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 导入深度学习模型
from backend.models.deep_learning.gnn_model import DocumentGNN
from backend.models.deep_learning.gat_model import DocumentGAT
from backend.models.deep_learning.gin_model import DocumentGIN

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained", "registry.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained")

# 模型架构工厂
MODEL_FACTORY = {
    "gcn": lambda in_c, h_c, out_c: DocumentGNN(in_c, h_c, out_c),
    "gat": lambda in_c, h_c, out_c: DocumentGAT(in_c, h_c, out_c, heads=4),
    "gin": lambda in_c, h_c, out_c: DocumentGIN(in_c, h_c, out_c),
}


class ClassificationService:
    def __init__(self, document_classes: Optional[List[str]] = None):
        # 初始化文本嵌入模型
        print("初始化 SentenceTransformer 模型...")
        self.text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.use_sentence_transformer = True
        print("SentenceTransformer 模型初始化成功")

        # 文档类别
        self.document_classes = (
            document_classes if document_classes else settings.DOCUMENT_CLASSES
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.document_classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.document_classes)}

        # 版面类型编码
        self.layout_types = [
            "text", "title", "figure", "caption", "header",
            "footer", "table", "reference", "equation", "general",
        ]
        self.layout_type_to_idx = {t: i for i, t in enumerate(self.layout_types)}
        self.spatial_dim = 4
        self.layout_type_dim = len(self.layout_types)
        self.text_embed_dim = 384

        # 模型注册表和缓存
        self.registry: Dict[str, dict] = {}
        self._loaded_models: Dict[str, torch.nn.Module] = {}
        self._load_registry()

    @property
    def in_channels(self) -> int:
        return self.text_embed_dim + self.spatial_dim + self.layout_type_dim

    def list_models(self) -> List[Dict[str, Any]]:
        """返回所有可用模型列表（给前端 API 使用）"""
        models = []
        for mid, info in self.registry.items():
            models.append({
                "model_id": mid,
                "name": info["name"],
                "model_type": info["model_type"],
                "edge_strategy": info["edge_strategy"],
                "description": info["description"],
                "loaded": mid in self._loaded_models,
            })
        return models

    def _load_registry(self):
        """加载模型注册表"""
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r") as f:
                self.registry = json.load(f)
            print(f"模型注册表已加载，共 {len(self.registry)} 个模型")
        else:
            print(f"警告: 未找到模型注册表 {REGISTRY_PATH}")

        # 默认加载最优模型
        default_id = "gcn_reading_order"
        self._get_model(default_id)

    def _get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """按 model_id 加载并缓存模型"""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        info = self.registry.get(model_id)
        if info is None:
            print(f"未知模型: {model_id}")
            return None

        model_path = os.path.join(MODELS_DIR, info["file"])
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None

        try:
            model_type = info["model_type"]
            in_c = info.get("in_channels", self.in_channels) or self.in_channels
            h_c = info.get("hidden_channels", 128) or 128
            out_c = len(self.document_classes)

            factory = MODEL_FACTORY.get(model_type)
            if factory is None:
                print(f"不支持的模型类型: {model_type}")
                return None

            model = factory(in_c, h_c, out_c)
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            self._loaded_models[model_id] = model
            print(f"模型加载成功: {model_id} ({model_path})")
            return model
        except Exception as e:
            print(f"模型加载失败 [{model_id}]: {e}")
            return None

    # ── 以下方法保持不变 ──────────────────────────────────

    def _encode_layout_type(self, region_type: str) -> np.ndarray:
        one_hot = np.zeros(self.layout_type_dim, dtype=np.float32)
        idx = self.layout_type_to_idx.get(
            region_type, self.layout_type_to_idx.get("general", 0)
        )
        one_hot[idx] = 1.0
        return one_hot

    def _extract_spatial_features(
        self, box: List[float], doc_width: float, doc_height: float
    ) -> np.ndarray:
        center_x = float(box[0] + box[2]) / 2
        center_y = float(box[1] + box[3]) / 2
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])

        if float(doc_width) > 0 and float(doc_height) > 0:
            spatial_features = np.array([
                center_x / doc_width, center_y / doc_height,
                width / doc_width, height / doc_height,
            ])
        else:
            spatial_features = np.array([center_x, center_y, width, height])
            max_val = np.max(spatial_features)
            if max_val > 0:
                spatial_features = spatial_features / max_val

        return spatial_features

    def _build_spatial_edges(self, boxes, doc_width, doc_height):
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
        if len(boxes) < 2:
            return []

        heights = [box[3] - box[1] for box in boxes]
        median_height = float(np.median(heights)) if heights else 1.0
        row_threshold = max(median_height * 0.5, 1.0)

        indexed = []
        for i, box in enumerate(boxes):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            row_key = int(cy / row_threshold)
            indexed.append((row_key, cx, i))

        indexed.sort(key=lambda x: (x[0], x[1]))

        edges = []
        sorted_indices = [item[2] for item in indexed]
        for k in range(len(sorted_indices) - 1):
            i, j = sorted_indices[k], sorted_indices[k + 1]
            edges.append([i, j])
            edges.append([j, i])

        return edges

    def _build_same_row_col_edges(self, boxes):
        if len(boxes) < 2:
            return []

        edges = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box_i, box_j = boxes[i], boxes[j]

                y_overlap = min(box_i[3], box_j[3]) - max(box_i[1], box_j[1])
                if y_overlap > 0:
                    y_i_height = box_i[3] - box_i[1]
                    y_j_height = box_j[3] - box_j[1]
                    min_h = min(y_i_height, y_j_height)
                    y_iou = y_overlap / min_h if min_h > 0 else 0
                else:
                    y_iou = 0

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
        all_edges = []
        all_edges.extend(self._build_spatial_edges(boxes, doc_width, doc_height))
        all_edges.extend(self._build_reading_order_edges(boxes))
        all_edges.extend(self._build_same_row_col_edges(boxes))

        edge_set = set()
        for u, v in all_edges:
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
        edge_strategy: str = "reading_order",
    ) -> Dict[str, Any]:
        doc_width = float(doc_width) if doc_width else 0
        doc_height = float(doc_height) if doc_height else 0
        node_features = []
        all_boxes = []

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
        center1 = [(float(box1[0]) + float(box1[2])) / 2, (float(box1[1]) + float(box1[3])) / 2]
        center2 = [(float(box2[0]) + float(box2[2])) / 2, (float(box2[1]) + float(box2[3])) / 2]
        return float(np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2))

    def _graph_classification(
        self, graph: Dict[str, Any], model: torch.nn.Module
    ) -> Dict[str, Any]:
        node_features = graph["node_features"]
        edges = graph["edges"]
        num_nodes = graph["num_nodes"]

        if num_nodes == 0:
            return {"predicted_class": "未知类型", "confidence": 0.5}

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        with torch.no_grad():
            output = model(x, edge_index, batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = self.idx_to_class[predicted_idx.item()]

        return {"predicted_class": predicted_class, "confidence": confidence.item()}

    def predict(
        self, request: ClassificationRequest, model_id: str = "gcn_reading_order"
    ) -> ClassificationResponse:
        regions: List[OCRRegion] = request.ocr_regions
        tables: Optional[List[TableRegion]] = request.tables

        # 获取模型信息和对应的边策略
        info = self.registry.get(model_id)
        if info is None:
            raise ValueError(f"未知模型: {model_id}")

        # 根据 model_id 选择边策略
        edge_strategy = info.get("edge_strategy", "reading_order") or "reading_order"

        # 构建图
        graph = self._build_graph(
            regions, tables,
            edge_strategy=edge_strategy,
        )

        # 获取模型
        model = self._get_model(model_id)
        if model is None:
            raise RuntimeError(f"模型加载失败: {model_id}")

        result = self._graph_classification(graph, model)

        return ClassificationResponse(
            document_id=request.document_id,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )


# 单例实例
classification_service = ClassificationService()
