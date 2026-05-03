from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    code: int = Field(200, description="业务状态码，200表示成功，非200表示异常")
    msg: str = Field("success", description="响应提示信息")
    data: Optional[T] = Field(None, description="核心数据载荷")


def success_response(data: Any = None, msg: str = "success") -> BaseResponse:
    """快速构建成功返回体的辅助函数"""
    return BaseResponse(code=200, msg=msg, data=data)


def error_response(
    code: int = 500, msg: str = "error", data: Any = None
) -> BaseResponse:
    """快速构建错误返回体的辅助函数"""
    return BaseResponse(code=code, msg=msg, data=data)
