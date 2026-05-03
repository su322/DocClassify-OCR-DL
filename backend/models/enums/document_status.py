from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "PENDING"  # 待处理
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
