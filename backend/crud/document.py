from sqlalchemy.orm import Session
from backend.models.database.document import DocumentRecord
from backend.models.enums.document_status import DocumentStatus
from typing import List, Dict, Any, Optional


def create_document(db: Session, doc_id: str, filename: str) -> DocumentRecord:
    """初始化记录"""
    db_doc = DocumentRecord(
        id=doc_id, filename=filename, status=DocumentStatus.PENDING.value
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


def update_document_status(
    db: Session, doc_id: str, status: str
) -> Optional[DocumentRecord]:
    """修改为 PROCESSING / PENDING 状态"""
    db_doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if db_doc:
        db_doc.status = status
        db.commit()
        db.refresh(db_doc)
    return db_doc


def reset_document_status(db: Session, doc_id: str) -> Optional[DocumentRecord]:
    return update_document_status(db, doc_id, DocumentStatus.PROCESSING.value)


def update_document_success(
    db: Session,
    doc_id: str,
    predicted_class: str,
    confidence: float,
    ocr_results: List[Dict[str, Any]] = None,
) -> Optional[DocumentRecord]:
    """处理完毕，写入核心分类及特征结果"""
    db_doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if db_doc:
        db_doc.status = DocumentStatus.SUCCESS.value
        db_doc.predicted_class = predicted_class
        db_doc.confidence = confidence
        if ocr_results:
            db_doc.ocr_results = ocr_results
        db.commit()
        db.refresh(db_doc)
    return db_doc


def update_document_failed(
    db: Session, doc_id: str, error_msg: str
) -> Optional[DocumentRecord]:
    """处理失败的异常挂起处理"""
    db_doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if db_doc:
        db_doc.status = DocumentStatus.FAILED.value
        if error_msg:
            # 截断以防 MySQL TEXT 或 VARCHAR 越界
            db_doc.error_message = error_msg[:2000]
        db.commit()
        db.refresh(db_doc)
    return db_doc


def get_document(db: Session, doc_id: str) -> Optional[DocumentRecord]:
    """供前端轮询查询状态使用"""
    return db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
