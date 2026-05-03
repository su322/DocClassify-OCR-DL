from fastapi import APIRouter
from backend.schemas.classification import ClassificationRequest, ClassificationResponse
from backend.schemas.base.response import BaseResponse, success_response
from backend.services.classification_service import classification_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=BaseResponse[ClassificationResponse],
    summary="深度学习文档分类接口",
    description="接收前端传来的结构化 OCR 特征序列（或从数据库提取），输入并调用图神经网络输出最终文档分类。",
)
async def classify_document(request: ClassificationRequest):
    """
    此接口是 OCR 数据向 GNN 转换的枢纽。
    在后期的端云协同中，你不会直接用 HTTP 请求它，而是整个流程被 Celery 将 ocr.py 的返回值接力传给此函数。
    如果想要单独测试模型，也可以把之前 OCR 成功生成的 JSON 直接作为 Body POST 进来。
    """
    result = classification_service.predict(request)
    return success_response(data=result, msg="分类成功")
