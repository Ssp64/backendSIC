# app/core/config.py — Environment-driven settings
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    DEBUG: bool = False
    PORT: int = 8000
    WORKERS: int = 1  # keep at 1 — InsightFace models are not fork-safe

    # ── Security ──────────────────────────────────────────────────────────────
    # A shared secret that your frontend sends in the Authorization header.
    # Generate: python -c "import secrets; print(secrets.token_hex(32))"
    API_SECRET_KEY: str = "change-me-in-production"

    # ── Supabase (server-side) ────────────────────────────────────────────────
    # Use the SERVICE ROLE key here — never expose it to the frontend.
    SUPABASE_URL: str = "https://ogbrblkfqroxlnulgyvg.supabase.co"
    SUPABASE_SERVICE_KEY: str = ""  # set via env var SUPABASE_SERVICE_KEY

    # ── Face Engine ───────────────────────────────────────────────────────────
    # Model name — buffalo_l is the best accuracy/speed trade-off
    # Options: buffalo_l (best), buffalo_m (balanced), buffalo_s (fast)
    INSIGHTFACE_MODEL: str = "buffalo_l"

    # Detection thresholds
    DETECTION_THRESHOLD: float = 0.30   # face detection confidence (0-1)
    MATCHING_THRESHOLD: float = 0.50    # ArcFace cosine distance (lower = stricter)
    CLUSTER_EPSILON: float = 0.50       # DBSCAN initial pass — kept tight; merge pass handles re-joining
    CLUSTER_MIN_SAMPLES: int = 1        # min faces per cluster (1 = include singletons)

    # Performance
    BATCH_SIZE: int = 8                # images processed per batch
    MAX_IMAGE_DIM: int = 1280          # resize long edge to this before detection
    MAX_FILE_SIZE_MB: int = 20

    # ── Caching ───────────────────────────────────────────────────────────────
    REDIS_URL: str = ""                # optional: redis://localhost:6379/0

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
