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
    # 训练时通过 --dataset 参数选择使用哪个数据集
    # 每个数据集定义: 名称 -> {类别列表, 训练数据目录, 描述}
    DATASETS: Dict[str, dict] = {
        "rvl_cdip": {
            "description": "RVL-CDIP: 40万张文档图像, 16类, 英文, 文档分类领域标准benchmark",
            "classes": [
                "letter",  # 信件
                "form",  # 表单
                "email",  # 电子邮件
                "handwritten",  # 手写文档
                "advertisement",  # 广告
                "scientific_report",  # 科学报告
                "scientific_publication",  # 科学出版物
                "specification",  # 规格说明书
                "file_folder",  # 文件夹
                "news_article",  # 新闻文章
                "budget",  # 预算
                "invoice",  # 发票
                "presentation",  # 演示文稿
                "questionnaire",  # 问卷
                "resume",  # 简历
                "memo",  # 备忘录
            ],
            "data_dir": "training/data/rvl_cdip/train",
        },
    }

    # Pydantic 自动从项目中加载 .env 变量覆盖上述默认值
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
