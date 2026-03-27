# app/models/schemas.py — Request / Response schemas
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


# ─── Shared ───────────────────────────────────────────────────────────────────
class FaceResult(BaseModel):
    """A single detected face with its ArcFace embedding."""
    embedding: List[float] = Field(..., description="512-d L2-normalized ArcFace embedding")
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] face bounding box")
    det_score: float = Field(..., description="RetinaFace detection confidence 0-1")
    pose: List[float] = Field(default=[0.0, 0.0, 0.0], description="[yaw, pitch, roll] head pose in degrees")


# ─── Indexing ────────────────────────────────────────────────────────────────
class IndexImageRequest(BaseModel):
    """Index a single image by URL (called after upload to Supabase storage)."""
    media_id: str
    url: str
    event_id: str


class IndexImageResponse(BaseModel):
    media_id: str
    faces_found: int
    face_results: List[FaceResult]
    success: bool
    error: Optional[str] = None


class BatchIndexRequest(BaseModel):
    """Index multiple images in one request (more efficient than N individual calls)."""
    items: List[IndexImageRequest]


class BatchIndexResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[IndexImageResponse]


# ─── Matching ────────────────────────────────────────────────────────────────
class GalleryItem(BaseModel):
    """A media row with its stored embeddings (fetched from Supabase)."""
    id: str
    url: str
    file_name: str
    file_type: str = "image"
    mime_type: Optional[str] = None
    storage_path: Optional[str] = None
    # face_embeddings is a JSON array of arrays: [[...512d...], [...512d...], ...]
    face_embeddings: Optional[List[List[float]]] = None


class MatchRequest(BaseModel):
    """Match a selfie (bytes sent as base64) against an event's gallery."""
    # Base64-encoded image (data URL or raw base64)
    image_base64: str = Field(..., description="Base64-encoded selfie image")
    gallery: List[GalleryItem] = Field(..., description="Indexed media rows for the event")
    threshold: Optional[float] = Field(None, description="Override default matching threshold")


class MatchResult(BaseModel):
    media_id: str
    distance: float
    score: int = Field(..., description="Match confidence 0-100")
    url: str
    file_name: str
    file_type: str
    mime_type: Optional[str] = None
    storage_path: Optional[str] = None


class MatchResponse(BaseModel):
    matches: List[MatchResult]
    total_gallery: int
    indexed_gallery: int
    probe_found: bool
    threshold_used: float


# ─── Clustering ──────────────────────────────────────────────────────────────
class ClusterRequest(BaseModel):
    """Cluster all faces in an event into person groups."""
    media_items: List[GalleryItem]
    epsilon: Optional[float] = Field(None, description="DBSCAN epsilon (cosine distance)")
    min_samples: Optional[int] = Field(None, description="DBSCAN min_samples")


class PersonCluster(BaseModel):
    person_index: int
    label: str
    photo_ids: List[str]
    photo_count: int
    face_count: int
    centroid: List[float] = Field(..., description="Averaged L2-normalized embedding")
    representative_url: str
    representative_photo_id: str


class ClusterResponse(BaseModel):
    people: List[PersonCluster]
    total_faces: int
    total_people: int
    epsilon_used: float
