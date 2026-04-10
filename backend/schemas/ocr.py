from typing import List, Optional
from pydantic import BaseModel, Field

class OCRRegion(BaseModel):
    """
    单个识别区域的标准格式 (OCR Region) todo 待确认
    """
    text: str = Field(..., description="识别出的文本内容")
    confidence: float = Field(..., description="识别置信度分数 (0~1之间)")
    box: List[int] = Field(..., description="标准四点坐标框 [xmin, ymin, xmax, ymax]")
    polygon: Optional[List[List[int]]] = Field(None, description="多边形包围盒 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    region_type: Optional[str] = Field("general", description="版面类型（若使用版面分析则是 title/text/table/figure 等）")

class OCRResponse(BaseModel):
    document_id: str = Field(..., description="文档的全局唯一标识符 (UUID)")
    filename: str = Field(..., description="原始文件名")
    regions: List[OCRRegion] = Field(..., description="提取出的全体结构化数据序列")

class AsyncOCRResponse(BaseModel):
    """异步调用返回的数据体"""
    document_id: str = Field(..., description="文档全局唯一标识符")
    task_id: str = Field(..., description="Celery 后台任务 ID")
