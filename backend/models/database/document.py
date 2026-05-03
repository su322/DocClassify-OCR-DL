from sqlalchemy import Column, String, Text, Float, DateTime, JSON
from datetime import datetime, timezone
import uuid

from backend.core.database import Base
from backend.models.enums.document_status import DocumentStatus


def get_utc_now():
    return datetime.now(timezone.utc)


class DocumentRecord(Base):
    __tablename__ = "documents"

    # UUID 作为主键，长度 36
    id = Column(
        String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4())
    )

    # 原始文件名
    filename = Column(String(255), index=True, nullable=False)

    # OCR 与分类的处理状态
    status = Column(String(50), default=DocumentStatus.PENDING.value, nullable=False)

    # 深度学习预测出的分类结果
    predicted_class = Column(String(100), nullable=True)

    # 置信度
    confidence = Column(Float, nullable=True)

    # 将 PaddleOCR 提取出的原始结构化数据序列化存入 (MySQL JSON 类型)
    ocr_results = Column(JSON, nullable=True)

    # 如果处理失败，保存报错信息
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)
