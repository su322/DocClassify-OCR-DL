from fastapi import FastAPI

from backend.core.database import engine, Base
from backend.routers.api_v1 import api_v1_router

# 必须在这里导入所有的数据库模型，Base 才能"看到"它们
# 添加 noqa 注释是为了告诉 IDE (如 PyCharm) 忽略"未使用导入"的警告，因为我们在利用它导入时的副作用（注册表结构）。
import backend.models.database.document  # noqa

# 系统启动时自动建表（如果表已存在则不处理）
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(api_v1_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
