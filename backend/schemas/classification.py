from pydantic import BaseModel, Field
from typing import List, Optional
from backend.schemas.ocr import OCRRegion, TableRegion


class ClassificationRequest(BaseModel):
    document_id: str = Field(..., description="文档的唯一标识符")
    ocr_regions: List[OCRRegion] = Field(..., description="前期提取出的结构化 OCR 数据")
    tables: Optional[List[TableRegion]] = Field(None, description="识别出的表格列表")


class ClassificationResponse(BaseModel):
    document_id: str
    predicted_class: str = Field(
        ...,
        description="深度学习模型(GNN)预测出的最终文档分类，例如：'命令_决定', '通知_公告', '合同_协议'",
    )
    confidence: float = Field(..., description="分类结果的置信度 (0~1之间)")
