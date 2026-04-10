import traceback
from backend.core.celery_app import celery_app

# 只有这个模块（运行在 Mac 上的独立 Python 进程）才会执行这些重度运算
from backend.services.ocr_service import ocr_service
from backend.services.classification_service import classification_service
from backend.schemas.classification import ClassificationRequest

@celery_app.task(bind=True, name="process_document_workflow")
def process_document_workflow(self, file_path: str, filename: str, document_id: str):
    """
    这是跑在 Mac M4 上的核心异步工作流：
    1. 从本地或共享挂载中读取文件执行 OCR。
    2. 无缝将 OCR 提取的列表丢进 GNN 图网络。
    3. 将最终的分类结果和坐标框写回云端共享数据库 (待实现 CRUD)。
    """
    print(f"[{document_id}] Worker 开始处理文件: {filename}...")
    try:
        # 第一阶段：PaddleOCR 版面分析与文本提取
        ocr_response = ocr_service.process_document(file_path, filename, document_id)

        # 将 OCR 结果转换为图神经网络（GNN）需要的数据格式
        gnn_request = ClassificationRequest(
            document_id=ocr_response.document_id,
            ocr_regions=ocr_response.regions
        )

        # 第二阶段：输入 GNN / 大模型进行深度推理分类
        gnn_response = classification_service.predict(gnn_request)

        # 第三阶段：落库闭环（此时我们可以通过 SQLAlchemy 把包含 document_id 的结果插入 MySQL 中）
        # result_to_save = gnn_response.model_dump()
        # save_to_mysql(result_to_save)

        print(f"[{document_id}] 处理完成！分类结果: {gnn_response.predicted_class} (置信度:{gnn_response.confidence})")

        # 返回分类字典，该信息将自动保存到 Celery Backend(Redis) 中
        return gnn_response.model_dump()

    except Exception as e:
        error_msg = f"处理失败 {filename}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # 触发重试等错误恢复机制
        raise e
