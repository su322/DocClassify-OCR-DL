import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings

# 创建数据库引擎
# 固定开启连接池探测机制，防止数据库自动断线
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,      # 检查连接是否可用
    pool_recycle=3600        # 每小时回收重置连接，避免 MySQL server has gone away
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
