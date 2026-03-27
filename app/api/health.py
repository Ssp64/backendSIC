# app/api/health.py
import time
from fastapi import APIRouter
from app.core.face_engine import face_engine

router = APIRouter()
START_TIME = time.time()


@router.get("/")
async def health():
    return {
        "status": "ok",
        "engine_ready": face_engine._initialized,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@router.get("/ready")
async def readiness():
    """Kubernetes readiness probe — only returns 200 when models are loaded."""
    if not face_engine._initialized:
        from fastapi import Response
        return Response(status_code=503, content="Engine not ready")
    return {"ready": True}
