from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set

class Settings(BaseSettings):
    # --- 项目基础配置 ---
    PROJECT_NAME: str = "DocClassify-OCR-DL"

    # --- 文件上传配置 ---
    # 官方体验站的限制常量参考
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_PDF_SIZE_MB: int = 200

    # --- 数据库配置 ---
    # SQLite 本地数据库
    DATABASE_URL: str = "sqlite:///./docclassify.db"

    # Pydantic 自动从项目中加载 .env 变量覆盖上述默认值 todo 我没用
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
