import os
import shutil
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends

from backend.schemas.ocr import OCRResponse, AsyncOCRResponse
from backend.schemas.response import BaseResponse, success_response

from backend.services.ocr_service import ocr_service
from backend.worker.document_tasks import process_document_workflow
from backend.core.database import get_db
from backend.core.config import settings
from backend.crud import document as doc_crud

router = APIRouter()

@router.post("/process", response_model=BaseResponse[OCRResponse], summary="解析文档图像基础接口", description="支持PDF/PNG/JPG/BMP/TIF等格式。单文件≤200MB(PDF)或10MB(图片)。")
async def process_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    前端上传一张单页文档图片或PDF，后端通过 PaddleOCR 解析并返回标准的 Pydantic JSON 格式。
    在引入 Celery 之前，这里暂时是同步阻塞调用。
    """
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

        # 核心：生成全局唯一的 Document ID
        doc_id = str(uuid.uuid4())

        # 落地第一条数据库记录
        doc_crud.create_document(db, doc_id, file.filename)

        # 调用 OCR 服务进行处理
        result = ocr_service.process_document(temp_file_path, file.filename, document_id=doc_id)

        # 统一返回体
        return success_response(data=result, msg="解析成功")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # TODO: 处理成功后可能暂不清理，以备查看
        if os.path.exists(temp_file_path):
            pass # os.remove(temp_file_path)

@router.post("/process_async", response_model=BaseResponse[AsyncOCRResponse], summary="异步解析文档图像接口", description="支持PDF/PNG/JPG等格式。将图像处理任务发送到 Celery 队列而不阻塞。")
async def process_document_async(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    异步方式：前端上传图片，后端只返回一个 task_id，由后台 Mac 的 Celery 节点慢慢处理。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
        if ext == ".pdf" and file_size_mb > settings.MAX_PDF_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"PDF 文件大小超出")
        elif ext != ".pdf" and file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"图片文件大小超出")

        doc_id = str(uuid.uuid4())

        # 落地第一条任务调度初始数据库记录
        doc_crud.create_document(db, doc_id, file.filename)

        # 核心：发给 Celery 任务队列（不会阻塞在此处）
        task = process_document_workflow.delay(temp_file_path, file.filename, doc_id)

        async_data = AsyncOCRResponse(document_id=doc_id, task_id=task.id)
        return success_response(data=async_data, msg="任务已提交队列")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
