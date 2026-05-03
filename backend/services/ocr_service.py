import os
from typing import List, Dict, Any, Optional, Tuple
from backend.schemas.ocr import OCRRegion, OCRResponse, TableRegion, TableCell
from paddleocr import PPStructureV3


class OCRService:
    def __init__(self):
        # 避免在 Windows 上的 OneDNN 报错
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["PADDLE_ENABLE_ONEDNN"] = "0"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        # 设置模型缓存目录为当前项目目录
        os.environ["PPDET_HOME"] = os.path.join(os.getcwd(), ".paddleocr")
        os.environ["PADDLEX_HOME"] = os.path.join(os.getcwd(), ".paddlex")

        # 初始化 PP-StructureV3 模型以获取版面分析（支持 text/title/table/figure 等）
        print("初始化 PP-StructureV3 模型...")
        self.ocr_model = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_table_recognition=True,
            use_formula_recognition=True,
            use_region_detection=True,
        )
        print("PP-StructureV3 模型初始化成功")

    @staticmethod
    def _assign_region_type(
        box: List[float], layout_boxes: List[Dict[str, Any]]
    ) -> str:
        """
        根据版面检测结果，判断文本框所属的版面类型。

        通过计算文本框与各版面区域的 IoU（交并比）来确定最匹配的版面标签。
        如果没有匹配的版面区域，默认返回 "text"。

        Args:
            box: 文本框坐标 [xmin, ymin, xmax, ymax]
            layout_boxes: layout_det_res 中的 boxes 列表

        Returns:
            版面类型标签，如 "title", "text", "figure", "table", "header", "footer" 等
        """
        if not layout_boxes or not box or len(box) < 4:
            return "text"

        best_label = "text"
        best_iou = 0.15  # IoU 阈值，低于此值认为不匹配

        for box_info in layout_boxes:
            label = box_info.get("label", "")
            if label == "table":
                continue  # 表格区域单独处理，跳过

            coord = box_info.get("coordinate")
            if not coord or len(coord) < 4:
                continue

            # 计算 IoU
            iou = OCRService._compute_iou(box, coord)
            if iou > best_iou:
                best_iou = iou
                best_label = label

        return best_label

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """计算两个轴对齐矩形的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def process_document(
        self, file_path: str, filename: str, document_id: str
    ) -> OCRResponse:
        """
        调用 PaddleOCR 对给定路径的文档进行脱机识别。
        自动兼容并处理单页图片（png/jpg等）和多页文档（pdf、长图tiff等）。
        """
        # PP-StructureV3 使用 predict 接口
        results = self.ocr_model.predict(input=file_path)

        # 兼容 PaddleX 管道返回的生成器对象
        if not isinstance(results, list):
            results = list(results)

        regions = []
        tables = []
        width = None
        height = None

        if results:
            for page_idx, page_res in enumerate(results):
                # 尝试将其安全转换为字典以便解析
                try:
                    if hasattr(page_res, "to_dict"):
                        res_dict = page_res.to_dict()
                    elif isinstance(page_res, dict):
                        res_dict = page_res
                    else:
                        res_dict = page_res.__dict__

                    # 获取文档尺寸
                    if width is None:
                        width = res_dict.get("width")
                    if height is None:
                        height = res_dict.get("height")

                    # 获取版面检测结果（用于标注 region_type）
                    layout_res = res_dict.get("layout_det_res", {})
                    layout_boxes = layout_res.get("boxes", [])

                    # 处理整体OCR结果
                    overall_ocr = res_dict.get("overall_ocr_res", {})
                    rec_texts = overall_ocr.get("rec_texts", [])
                    rec_scores = overall_ocr.get("rec_scores", [])
                    rec_boxes = overall_ocr.get("rec_boxes", [])
                    rec_polys = overall_ocr.get("rec_polys", [])

                    # 处理文本区域（利用版面检测结果标注 region_type）
                    for i, text in enumerate(rec_texts):
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        box = rec_boxes[i] if i < len(rec_boxes) else [0, 0, 0, 0]
                        polygon = rec_polys[i] if i < len(rec_polys) else None

                        # 根据版面检测结果自动判断 region_type
                        region_type = self._assign_region_type(box, layout_boxes)

                        region = OCRRegion(
                            text=text,
                            confidence=confidence,
                            box=box,
                            polygon=polygon,
                            region_type=region_type,
                        )
                        regions.append(region)

                    # 处理表格区域
                    table_res_list = res_dict.get("table_res_list", [])
                    for table_res in table_res_list:
                        pred_html = table_res.get("pred_html", "")
                        cell_box_list = table_res.get("cell_box_list", [])
                        table_ocr = table_res.get("table_ocr_pred", {})
                        table_rec_texts = table_ocr.get("rec_texts", [])

                        # 构建单元格列表
                        cells = []
                        for j, cell_box in enumerate(cell_box_list):
                            text = (
                                table_rec_texts[j] if j < len(table_rec_texts) else None
                            )
                            cell = TableCell(box=cell_box, text=text)
                            cells.append(cell)

                        # 获取表格坐标和置信度
                        table_box = None
                        table_confidence = 0.9

                        for box_info in layout_boxes:
                            if box_info.get("label") == "table":
                                table_box = box_info.get("coordinate", [0, 0, 0, 0])
                                table_confidence = box_info.get("score", 0.9)
                                break

                        if table_box:
                            table_region = TableRegion(
                                box=table_box,
                                html=pred_html,
                                cells=cells,
                                confidence=table_confidence,
                            )
                            tables.append(table_region)

                except Exception as parse_e:
                    print(f"解析页面时出现跳过的内容块: {parse_e}")
                    continue

        return OCRResponse(
            document_id=document_id,
            filename=filename,
            regions=regions,
            tables=tables if tables else None,
            width=width,
            height=height,
        )


# 单例实例
ocr_service = OCRService()
