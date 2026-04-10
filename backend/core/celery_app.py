from celery import Celery

from backend.core.config import settings

# 初始化 Celery 实例
celery_app = Celery(
    "docclassify_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,     # 用于存储任务执行的最终结果或状态
    include=["backend.worker.document_tasks"] # 注册要被执行的任务模块
)

# 配置 Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    # 如果处理极度耗时的 AI 任务，可以限制 worker 的并发数以防止 Mac 内存溢出
    worker_concurrency=1,
)
