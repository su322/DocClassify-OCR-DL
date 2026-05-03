import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings

# 创建数据库引擎
# SQLite 不需要连接池和回收机制
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 多线程支持
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 所有数据模型的基类
Base = declarative_base()


# FastAPI 依赖项，用于在每次 API 请求时获取/关闭数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
