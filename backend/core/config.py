from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set, List, Dict


class Settings(BaseSettings):
    # --- 项目基础配置 ---
    PROJECT_NAME: str = "DocClassify-OCR-DL"

    # --- 文件上传配置 ---
    ALLOWED_EXTENSIONS: Set[str] = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_PDF_SIZE_MB: int = 200

    # --- 数据库配置 ---
    DATABASE_URL: str = "sqlite:///./docclassify.db"

    # --- 默认文档分类（推理服务使用） ---
    DOCUMENT_CLASSES: List[str] = [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific_report",
        "scientific_publication",
        "specification",
        "file_folder",
        "news_article",
        "budget",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo",
    ]

    # --- 数据集配置 ---
    DATASETS: Dict[str, dict] = {
        "rvl_cdip": {
            "description": "RVL-CDIP: 47996张文档图像, 16类, 英文, 文档分类领域标准benchmark",
            "classes": [
                "letter",
                "form",
                "email",
                "handwritten",
                "advertisement",
                "scientific_report",
                "scientific_publication",
                "specification",
                "file_folder",
                "news_article",
                "budget",
                "invoice",
                "presentation",
                "questionnaire",
                "resume",
                "memo",
            ],
            "source_dir": "training/data/rvl_cdip/data",
            "train_dir": "training/data/rvl_cdip/train",
            "val_dir": "training/data/rvl_cdip/val",
            "test_dir": "training/data/rvl_cdip/test",
        },
    }

    # Pydantic 自动从项目中加载 .env 变量覆盖上述默认值
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
