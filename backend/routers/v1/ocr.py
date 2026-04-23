import os
import shutil
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends

from backend.schemas.ocr import OCRResponse
from backend.schemas.classification import ClassificationRequest, ClassificationResponse
from backend.schemas.base.response import BaseResponse, success_response

from backend.services.ocr_service import ocr_service
from backend.services.classification_service import classification_service
from backend.core.database import get_db
from backend.core.config import settings
from backend.crud import document as doc_crud
from backend.models.enums.document_status import DocumentStatus

router = APIRouter()

def _handle_file_upload(file: UploadFile) -> str:
    """处理文件上传和校验"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # 格式白名单校验
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}。支持的格式: {', '.join(settings.ALLOWED_EXTENSIONS)}")

    # 将上传的文件保存到临时目录
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 大小校验
        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
        if ext == ".pdf" and file_size_mb > settings.MAX_PDF_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"PDF 文件大小 ({file_size_mb:.2f}MB) 超出限制 ({settings.MAX_PDF_SIZE_MB}MB)")
        elif ext != ".pdf" and file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"图片文件大小 ({file_size_mb:.2f}MB) 超出限制 ({settings.MAX_IMAGE_SIZE_MB}MB)")

        return temp_file_path
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise

@router.post("/process", response_model=BaseResponse[OCRResponse], summary="解析文档图像基础接口", description="支持PDF/PNG/JPG/BMP/TIF等格式。单文件≤200MB(PDF)或10MB(图片)。")
async def process_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    前端上传一张单页文档图片或PDF，后端通过 PaddleOCR 解析并返回标准的 Pydantic JSON 格式。
    """
    temp_file_path = _handle_file_upload(file)

    try:
        # 核心：生成全局唯一的 Document ID
        doc_id = str(uuid.uuid4())

        # 落地第一条数据库记录
        doc_crud.create_document(db, doc_id, file.filename)

        # 调用 OCR 服务进行处理
        result = ocr_service.process_document(temp_file_path, file.filename, document_id=doc_id)

        # 统一返回体
        return success_response(data=result, msg="解析成功")

    except Exception as e:
        # 更新数据库状态为失败
        if 'doc_id' in locals():
            doc_crud.update_document_failed(db, doc_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 处理成功后清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/classify", response_model=BaseResponse[ClassificationResponse], summary="文档分类完整流程", description="上传文档，自动完成 OCR 解析和分类")
async def classify_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    完整流程：上传文档 → OCR 解析 → 分类 → 返回结果
    """
    temp_file_path = _handle_file_upload(file)

    try:
        # 核心：生成全局唯一的 Document ID
        doc_id = str(uuid.uuid4())

        # 落地第一条数据库记录
        doc_crud.create_document(db, doc_id, file.filename)

        # 1. 调用 OCR 服务进行处理
        ocr_result = ocr_service.process_document(temp_file_path, file.filename, document_id=doc_id)

        # 2. 调用分类服务进行分类
        classification_request = ClassificationRequest(
            document_id=doc_id,
            ocr_regions=ocr_result.regions,
            tables=ocr_result.tables
        )
        classification_result = classification_service.predict(classification_request)

        # 3. 更新数据库记录
        ocr_results_dict = [region.model_dump() for region in ocr_result.regions]
        doc_crud.update_document_success(
            db, 
            doc_id, 
            classification_result.predicted_class, 
            classification_result.confidence, 
            ocr_results_dict
        )

        # 统一返回体
        return success_response(data=classification_result, msg="分类成功")

    except Exception as e:
        # 更新数据库状态为失败
        if 'doc_id' in locals():
            doc_crud.update_document_failed(db, doc_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 处理成功后清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
