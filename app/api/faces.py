# app/api/faces.py — Face Intelligence API endpoints
#
#   POST /api/v1/faces/index          — index a single image
#   POST /api/v1/faces/index/batch    — batch index
#   POST /api/v1/faces/match          — match selfie against gallery
#   POST /api/v1/faces/cluster        — cluster faces into person groups

import asyncio
import base64
import logging
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.config import settings
from app.core.face_engine import face_engine
from app.models.schemas import (
    BatchIndexRequest,
    BatchIndexResponse,
    ClusterRequest,
    ClusterResponse,
    IndexImageRequest,
    IndexImageResponse,
    MatchRequest,
    MatchResponse,
    MatchResult,
    PersonCluster,
)
from app.services.supabase_client import supabase_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Auth ──────────────────────────────────────────────────────────────────────
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")


# ── POST /index ───────────────────────────────────────────────────────────────
@router.post("/index", response_model=IndexImageResponse,
             dependencies=[Depends(verify_api_key)])
async def index_image(req: IndexImageRequest) -> IndexImageResponse:
    """Download image, detect faces, save embeddings to Supabase."""
    image_bytes = await supabase_service.download_image(req.url)
    if not image_bytes:
        return IndexImageResponse(media_id=req.media_id, faces_found=0,
                                  face_results=[], success=False,
                                  error="Could not download image")

    face_results = await face_engine.extract_embeddings_from_bytes(image_bytes, req.url)
    saved = await supabase_service.save_face_embeddings(req.media_id, face_results)

    return IndexImageResponse(
        media_id=req.media_id,
        faces_found=len(face_results),
        face_results=face_results,
        success=saved,
    )


# ── POST /index/batch ─────────────────────────────────────────────────────────
@router.post("/index/batch", response_model=BatchIndexResponse,
             dependencies=[Depends(verify_api_key)])
async def batch_index(req: BatchIndexRequest) -> BatchIndexResponse:
    """
    Batch index images.

    Strategy:
      - Download all images in a batch concurrently (I/O bound)
      - Detect faces sequentially (CPU bound, model not thread-safe)
      - Save embeddings concurrently (I/O bound)
    """
    results: List[IndexImageResponse] = []
    batch_size = settings.BATCH_SIZE

    for batch_start in range(0, len(req.items), batch_size):
        batch = req.items[batch_start: batch_start + batch_size]

        # 1. Download all images in this batch concurrently
        image_bytes_list = await asyncio.gather(
            *[supabase_service.download_image(item.url) for item in batch],
            return_exceptions=True,
        )

        # 2. Detect faces sequentially, collect (media_id, face_results) pairs
        detection_results = []
        for item, image_bytes in zip(batch, image_bytes_list):
            if isinstance(image_bytes, Exception) or image_bytes is None:
                results.append(IndexImageResponse(
                    media_id=req.media_id if hasattr(req, "media_id") else item.media_id,
                    faces_found=0, face_results=[], success=False,
                    error="Download failed",
                ))
                # Use a sentinel so we skip saving for this item
                detection_results.append((item.media_id, None))
                continue

            try:
                face_results = await face_engine.extract_embeddings_from_bytes(
                    image_bytes, item.url
                )
                results.append(IndexImageResponse(
                    media_id=item.media_id,
                    faces_found=len(face_results),
                    face_results=face_results,
                    success=True,
                ))
                detection_results.append((item.media_id, face_results))
            except Exception as exc:
                logger.error(f"Detection failed for {item.media_id}: {exc}")
                results.append(IndexImageResponse(
                    media_id=item.media_id,
                    faces_found=0, face_results=[], success=False,
                    error=str(exc),
                ))
                detection_results.append((item.media_id, None))

        # 3. Save embeddings concurrently — build coroutines first, then gather
        save_coros = [
            supabase_service.save_face_embeddings(media_id, face_results)
            for media_id, face_results in detection_results
            if face_results is not None  # skip failed downloads/detections
        ]
        if save_coros:
            save_outcomes = await asyncio.gather(*save_coros, return_exceptions=True)
            for outcome in save_outcomes:
                if isinstance(outcome, Exception):
                    logger.error(f"Save embeddings failed: {outcome}")

        logger.info(
            f"Batch indexed {min(batch_start + batch_size, len(req.items))}"
            f"/{len(req.items)}"
        )

    succeeded = sum(1 for r in results if r.success)
    return BatchIndexResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


# ── POST /match ───────────────────────────────────────────────────────────────
@router.post("/match", response_model=MatchResponse,
             dependencies=[Depends(verify_api_key)])
async def match_face(req: MatchRequest) -> MatchResponse:
    """Match a selfie against an event gallery."""
    try:
        img_data = req.image_base64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        image_bytes = base64.b64decode(img_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    probe = await face_engine.extract_probe_embedding(image_bytes)
    if probe is None:
        return MatchResponse(
            matches=[], total_gallery=len(req.gallery),
            indexed_gallery=0, probe_found=False,
            threshold_used=req.threshold or settings.MATCHING_THRESHOLD,
        )

    gallery_dicts = [item.model_dump() for item in req.gallery]
    indexed       = [g for g in gallery_dicts if g.get("face_embeddings")]
    raw_matches   = face_engine.match(probe, indexed, threshold=req.threshold)

    return MatchResponse(
        matches=[MatchResult(**m) for m in raw_matches],
        total_gallery=len(req.gallery),
        indexed_gallery=len(indexed),
        probe_found=True,
        threshold_used=req.threshold or settings.MATCHING_THRESHOLD,
    )


# ── POST /cluster ─────────────────────────────────────────────────────────────
@router.post("/cluster", response_model=ClusterResponse,
             dependencies=[Depends(verify_api_key)])
async def cluster_faces(req: ClusterRequest) -> ClusterResponse:
    """Cluster all faces in an event into person groups."""
    eps         = req.epsilon or settings.CLUSTER_EPSILON
    media_dicts = [item.model_dump() for item in req.media_items]

    loop      = asyncio.get_event_loop()
    people_raw = await loop.run_in_executor(
        None,
        lambda: face_engine.cluster_faces(media_dicts, epsilon=eps,
                                          min_samples=req.min_samples),
    )

    total_faces = sum(p["face_count"] for p in people_raw)
    return ClusterResponse(
        people=[PersonCluster(**p) for p in people_raw],
        total_faces=total_faces,
        total_people=len(people_raw),
        epsilon_used=eps,
    )
