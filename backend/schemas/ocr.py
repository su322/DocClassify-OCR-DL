from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class OCRRegion(BaseModel):
    """
    单个识别区域的标准格式 (OCR Region)
    """
    text: str = Field(..., description="识别出的文本内容")
    confidence: float = Field(..., description="识别置信度分数 (0~1之间)")
    box: List[float] = Field(..., description="标准四点坐标框 [xmin, ymin, xmax, ymax]")
    polygon: Optional[List[List[float]]] = Field(None, description="多边形包围盒 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    region_type: Optional[str] = Field("general", description="版面类型（若使用版面分析则是 title/text/table/figure 等）")

class TableCell(BaseModel):
    """
    表格单元格信息
    """
    box: List[float] = Field(..., description="单元格坐标 [xmin, ymin, xmax, ymax]")
    text: Optional[str] = Field(None, description="单元格文本内容")

class TableRegion(BaseModel):
    """
    表格区域信息
    """
    box: List[float] = Field(..., description="表格坐标 [xmin, ymin, xmax, ymax]")
    html: str = Field(..., description="表格HTML内容")
    cells: Optional[List[TableCell]] = Field(None, description="表格单元格列表")
    confidence: float = Field(..., description="表格识别置信度")

class OCRResponse(BaseModel):
    document_id: str = Field(..., description="文档的全局唯一标识符 (UUID)")
    filename: str = Field(..., description="原始文件名")
    regions: List[OCRRegion] = Field(..., description="提取出的全体结构化数据序列")
    tables: Optional[List[TableRegion]] = Field(None, description="识别出的表格列表")
    width: Optional[int] = Field(None, description="文档宽度")
    height: Optional[int] = Field(None, description="文档高度")

class AsyncOCRResponse(BaseModel):
    """
    异步调用返回的数据体
    """
    document_id: str = Field(..., description="文档全局唯一标识符")
    task_id: str = Field(..., description="后台任务 ID")
