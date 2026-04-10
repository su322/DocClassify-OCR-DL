from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set

class Settings(BaseSettings):
    # --- 项目基础配置 ---
    PROJECT_NAME: str = "DocClassify-OCR-DL"

    # --- 中间件/Celery/Redis 配置 ---
    # 本地测试时使用 localhost，部署时你的 Mac 会在 .env 填上腾讯云的 Redis 公网 IP
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/0"

    # --- 文件上传配置 ---
    # 官方体验站的限制常量参考
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_PDF_SIZE_MB: int = 200

    # --- 数据库配置 ---
    # MySQL 协议（需预先建立数据库）
    DATABASE_URL: str = "mysql+pymysql://root:password@127.0.0.1:3306/docclassify"

    # Pydantic 自动从项目中加载 .env 变量覆盖上述默认值
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
