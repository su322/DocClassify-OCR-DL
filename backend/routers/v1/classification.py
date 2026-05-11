from fastapi import APIRouter, Query
from backend.schemas.classification import ClassificationRequest, ClassificationResponse
from backend.schemas.base.response import BaseResponse, success_response
from backend.services.classification_service import classification_service

router = APIRouter()


@router.get(
    "/models",
    response_model=BaseResponse[list],
    summary="获取可用模型列表",
    description="返回所有已训练并注册的模型信息，前端可用于模型选择器。",
)
async def list_models():
    models = classification_service.list_models()
    return success_response(data=models, msg="获取成功")


@router.post(
    "/predict",
    response_model=BaseResponse[ClassificationResponse],
    summary="深度学习文档分类接口",
    description="接收前端传来的结构化 OCR 特征序列，使用指定模型进行分类。",
)
async def classify_document(
    request: ClassificationRequest,
    model_id: str = Query("gcn_reading_order", description="模型标识，默认最优 GCN (reading_order)"),
):
    result = classification_service.predict(request, model_id=model_id)
    return success_response(data=result, msg="分类成功")
