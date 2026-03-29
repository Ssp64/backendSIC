# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    DEBUG:   bool = False
    PORT:    int  = 8000
    WORKERS: int  = 1   # keep at 1 — InsightFace models are not fork-safe

    # Security
    API_SECRET_KEY: str = "change-me-in-production"

    # Supabase (server-side service role key — never expose to browser)
    SUPABASE_URL:         str = "https://ogbrblkfqroxlnulgyvg.supabase.co"
    SUPABASE_SERVICE_KEY: str = ""

    # Face engine
    INSIGHTFACE_MODEL:    str   = "buffalo_l"   # buffalo_l = best accuracy
    DETECTION_THRESHOLD:  float = 0.15          # low = catches dark/angled/small faces
    MATCHING_THRESHOLD:   float = 0.50          # cosine distance cutoff for matching
    CLUSTER_EPSILON:      float = 0.60          # agglomerative distance threshold
    CLUSTER_MIN_SAMPLES:  int   = 1

    # Performance
    BATCH_SIZE:       int = 8
    MAX_IMAGE_DIM:    int = 1920              # larger = better detection on hi-res photos
    MAX_FILE_SIZE_MB: int = 20

    class Config:
        env_file       = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
