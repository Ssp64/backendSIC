# app/main.py — Memoire AI Backend
# Production-grade FastAPI server with ArcFace/InsightFace face intelligence

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.face_engine import face_engine
from app.api import faces, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: warm up models before accepting traffic ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Memoire AI Backend starting — warming face engine...")
    t0 = time.perf_counter()
    await face_engine.initialize()
    logger.info(f"✅ Face engine ready in {time.perf_counter()-t0:.2f}s")
    yield
    logger.info("🛑 Shutting down — releasing GPU/CPU resources")
    face_engine.shutdown()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Memoire AI",
    description="Production face recognition & clustering API for event photos",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

# CORS — allow your Supabase-hosted frontend and localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter()-t0)*1000:.1f}ms"
    return response


# ─── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(faces.router, prefix="/api/v1/faces", tags=["faces"])
