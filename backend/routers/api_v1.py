from fastapi import APIRouter
from backend.routers.v1 import ocr
from backend.routers.v1 import classification

api_v1_router = APIRouter()

PREFIX = "/api/v1"

api_v1_router.include_router(ocr.router, prefix=PREFIX+"/ocr", tags=["ocr"])
api_v1_router.include_router(classification.router, prefix=PREFIX+"/classification", tags=["classification"])
