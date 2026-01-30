from fastapi import APIRouter
from backend.routers.v1 import ocr

router = APIRouter()

PREFIX = "/api/v1/"

router.include_router(ocr.router, prefix=PREFIX+"ocr", tags=["ocr"])
