import os
from typing import List, Dict, Any
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
            use_region_detection=True
        )
        print("PP-StructureV3 模型初始化成功")

    def process_document(self, file_path: str, filename: str, document_id: str) -> OCRResponse:
        """
        调用 PaddleOCR 对给定路径的文档进行脱机识别。
        自动兼容并处理单页图片（png/jpg等）和多页文档（pdf、长图tiff等）。
        未来若需支持 Word/Excel，可在此处预留拓展，将其先转为 PDF 或图片流再输入。
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
                        # Fallback 获取其内置属性
                        res_dict = page_res.__dict__

                    # 获取文档尺寸
                    if width is None:
                        width = res_dict.get("width")
                    if height is None:
                        height = res_dict.get("height")

                    # 处理整体OCR结果
                    overall_ocr = res_dict.get("overall_ocr_res", {})
                    rec_texts = overall_ocr.get("rec_texts", [])
                    rec_scores = overall_ocr.get("rec_scores", [])
                    rec_boxes = overall_ocr.get("rec_boxes", [])
                    rec_polys = overall_ocr.get("rec_polys", [])

                    # 处理文本区域
                    for i, text in enumerate(rec_texts):
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        box = rec_boxes[i] if i < len(rec_boxes) else [0, 0, 0, 0]
                        polygon = rec_polys[i] if i < len(rec_polys) else None

                        region = OCRRegion(
                            text=text,
                            confidence=confidence,
                            box=box,
                            polygon=polygon,
                            region_type="text"
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
                            text = table_rec_texts[j] if j < len(table_rec_texts) else None
                            cell = TableCell(
                                box=cell_box,
                                text=text
                            )
                            cells.append(cell)

                        # 获取表格坐标和置信度
                        layout_res = res_dict.get("layout_det_res", {})
                        boxes = layout_res.get("boxes", [])
                        table_box = None
                        table_confidence = 0.9

                        for box_info in boxes:
                            if box_info.get("label") == "table":
                                table_box = box_info.get("coordinate", [0, 0, 0, 0])
                                table_confidence = box_info.get("score", 0.9)
                                break

                        if table_box:
                            table_region = TableRegion(
                                box=table_box,
                                html=pred_html,
                                cells=cells,
                                confidence=table_confidence
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
            height=height
        )

# 单例实例
ocr_service = OCRService()
