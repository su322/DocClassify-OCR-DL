import os
from typing import List, Dict, Any
from backend.schemas.ocr import OCRRegion, OCRResponse
from paddleocr import PPStructureV3

class OCRService:
    def __init__(self):
        # 避免在 Windows 上的 OneDNN 报错 todo 可能没用
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["PADDLE_ENABLE_ONEDNN"] = "0"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        # 初始化 PP-StructureV3 模型以获取版面分析（支持 text/title/table/figure 等）todo 待检查参数
        self.ocr_model = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )

    def process_document(self, file_path: str, filename: str, document_id: str) -> OCRResponse:
        """
        调用 PaddleOCR 对给定路径的文档进行脱机识别。
        自动兼容并处理单页图片（png/jpg等）和多页文档（pdf、长图tiff等）。 todo 下面要看看
        未来若需支持 Word/Excel，可在此处预留拓展，将其先转为 PDF 或图片流再输入。
        """
        # res 格式：图片返回 [ [line1, line2...] ]，多页 PDF 返回 [ page1_lines, page2_lines... ]
        try:
            # PP-StructureV3 使用 predict 接口
            results = self.ocr_model.predict(input=file_path)

            # 兼容 PaddleX 管道返回的生成器对象
            if not isinstance(results, list):
                results = list(results)
        except Exception as e:
            # 捕获 PaddleOCR 不支持的文件格式异常
            raise ValueError(f"无法解析该文件格式或文件损坏: {filename}, 内部错误: {str(e)}")

        regions = []
        if results:
            for page_idx, page_res in enumerate(results):
                # PPStructureV3 / PaddleX 返回的 page_res 通常带有 layout 属性或可以直接转dict
                # 对于版面分析结果，我们提取每个内容块
                try:
                    # 尝试将其安全转换为字典以便解析
                    if hasattr(page_res, "to_dict"):
                        res_dict = page_res.to_dict()
                    elif isinstance(page_res, dict):
                        res_dict = page_res
                    else:
                        # Fallback 获取其内置属性
                        res_dict = page_res.__dict__

                    # 如果不能直接获取到布局块，尝试从其常见的存放路径提取
                    layout_boxes = res_dict.get("layout", []) or res_dict.get("html", []) or []

                    # 某些版本的返回格式直接是 list
                    if isinstance(page_res, list):
                        layout_boxes = page_res

                    for block in layout_boxes:
                        # PP-Structure 返回的块格式一般包含 type, bbox, res
                        b_type = block.get('type', 'text')
                        b_box = block.get('bbox', [0, 0, 0, 0])  # [xmin, ymin, xmax, ymax]

                        # 内部可能有多行文字，合并它们
                        inner_texts = []
                        confidences = []
                        polygons = []

                        inner_res = block.get('res', [])
                        if inner_res and isinstance(inner_res, list):
                            for line in inner_res:
                                if "text" in line:
                                    inner_texts.append(line["text"])
                                    confidences.append(line.get("confidence", 1.0))
                                    if "text_region" in line:
                                        polygons.extend(line["text_region"])

                        # 合并块内的文本
                        merged_text = "\n".join(inner_texts) if inner_texts else ""
                        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

                        region = OCRRegion(
                            text=merged_text,
                            confidence=avg_conf,
                            box=b_box,
                            polygon=polygons if polygons else None,
                            region_type=b_type
                        )
                        regions.append(region)

                except Exception as parse_e:
                    print(f"解析页面时出现跳过的内容块: {parse_e}")
                    continue

        return OCRResponse(
            document_id=document_id,
            filename=filename,
            regions=regions
        )

# 单例实例
ocr_service = OCRService()
