# app/api/faces.py — Face Intelligence API endpoints
#
# Endpoints:
#   POST /api/v1/faces/index          — index a single image
#   POST /api/v1/faces/index/batch    — batch index (recommended for bulk upload)
#   POST /api/v1/faces/match          — match a selfie against an event gallery
#   POST /api/v1/faces/cluster        — cluster all faces into people groups
#   POST /api/v1/faces/probe          — extract probe embedding only (no matching)

import asyncio
import base64
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status

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


# ─── Auth dependency ──────────────────────────────────────────────────────────
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Simple shared-secret auth. Replace with JWT validation if needed."""
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ─── POST /index — Index a single image ──────────────────────────────────────
@router.post(
    "/index",
    response_model=IndexImageResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Index faces in a single image",
)
async def index_image(req: IndexImageRequest) -> IndexImageResponse:
    """
    Download an image from its public URL, detect all faces,
    extract ArcFace embeddings, and save them back to the media row in Supabase.

    Call this from your frontend after each file upload completes.
    """
    image_bytes = await supabase_service.download_image(req.url)
    if not image_bytes:
        return IndexImageResponse(
            media_id=req.media_id,
            faces_found=0,
            face_results=[],
            success=False,
            error="Could not download image from URL",
        )

    face_results = await face_engine.extract_embeddings_from_bytes(image_bytes, req.url)

    # Persist to Supabase asynchronously
    await supabase_service.save_face_embeddings(req.media_id, face_results)

    return IndexImageResponse(
        media_id=req.media_id,
        faces_found=len(face_results),
        face_results=face_results,
        success=True,
    )


# ─── POST /index/batch — Batch index (production path) ───────────────────────
@router.post(
    "/index/batch",
    response_model=BatchIndexResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Batch index multiple images",
)
async def batch_index(req: BatchIndexRequest) -> BatchIndexResponse:
    """
    Index multiple images concurrently (bounded by BATCH_SIZE to avoid OOM).
    This is the recommended path after a bulk upload.

    Processing strategy:
      - Download images concurrently (I/O bound → fast)
      - Run face detection sequentially in the thread pool (CPU bound → safe)
      - Save embeddings concurrently (I/O bound → fast)
    """
    results: List[IndexImageResponse] = []
    items = req.items
    batch_size = settings.BATCH_SIZE

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Download all images in batch concurrently
        download_tasks = [supabase_service.download_image(item.url) for item in batch]
        image_bytes_list = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Process each image sequentially (face engine is not thread-safe for >1 concurrent)
        save_tasks = []
        for item, image_bytes in zip(batch, image_bytes_list):
            if isinstance(image_bytes, Exception) or image_bytes is None:
                results.append(
                    IndexImageResponse(
                        media_id=item.media_id,
                        faces_found=0,
                        face_results=[],
                        success=False,
                        error="Download failed",
                    )
                )
                continue

            try:
                face_results = await face_engine.extract_embeddings_from_bytes(
                    image_bytes, item.url
                )
                save_tasks.append(
                    supabase_service.save_face_embeddings(item.media_id, face_results)
                )
                results.append(
                    IndexImageResponse(
                        media_id=item.media_id,
                        faces_found=len(face_results),
                        face_results=face_results,
                        success=True,
                    )
                )
            except Exception as e:
                logger.error(f"Face extraction failed for {item.media_id}: {e}")
                results.append(
                    IndexImageResponse(
                        media_id=item.media_id,
                        faces_found=0,
                        face_results=[],
                        success=False,
                        error=str(e),
                    )
                )

        # Save all embeddings in this batch concurrently
        if save_tasks:
            await asyncio.gather(*save_tasks, return_exceptions=True)

        logger.info(f"Batch indexed {min(i + batch_size, len(items))}/{len(items)}")

    succeeded = sum(1 for r in results if r.success)
    return BatchIndexResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


# ─── POST /match — Match selfie against gallery ───────────────────────────────
@router.post(
    "/match",
    response_model=MatchResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Match a selfie against an event gallery",
)
async def match_face(req: MatchRequest) -> MatchResponse:
    """
    Extract a probe embedding from the selfie, then match it against
    all indexed faces in the gallery.

    The gallery should be fetched client-side from Supabase (the frontend
    already has the face_embeddings column data). This avoids a round-trip
    to Supabase from the API server.

    Alternatively, pass event_id and let the API fetch the gallery from Supabase.
    """
    # Decode base64 image
    try:
        img_data = req.image_base64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        image_bytes = base64.b64decode(img_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Extract probe embedding
    probe = await face_engine.extract_probe_embedding(image_bytes)
    if probe is None:
        return MatchResponse(
            matches=[],
            total_gallery=len(req.gallery),
            indexed_gallery=0,
            probe_found=False,
            threshold_used=req.threshold or settings.MATCHING_THRESHOLD,
        )

    # Filter gallery to only indexed items
    gallery_dicts = [item.model_dump() for item in req.gallery]
    indexed = [g for g in gallery_dicts if g.get("face_embeddings")]

    # Run matching (CPU-bound but fast — pure numpy, no model inference)
    raw_matches = face_engine.match(probe, indexed, threshold=req.threshold)
    matches = [MatchResult(**m) for m in raw_matches]

    return MatchResponse(
        matches=matches,
        total_gallery=len(req.gallery),
        indexed_gallery=len(indexed),
        probe_found=True,
        threshold_used=req.threshold or settings.MATCHING_THRESHOLD,
    )


# ─── POST /cluster — Cluster all faces into people ────────────────────────────
@router.post(
    "/cluster",
    response_model=ClusterResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Cluster all faces in an event into person groups",
)
async def cluster_faces(req: ClusterRequest) -> ClusterResponse:
    """
    Google Photos-style people grouping using DBSCAN on ArcFace embeddings.

    Pass all indexed media rows for an event. Returns person clusters sorted
    by photo count (most-photographed person first).

    Tip: Cache the cluster result in your frontend for the session — it's
    expensive to recompute but only needs to update when new photos are indexed.
    """
    eps = req.epsilon or settings.CLUSTER_EPSILON
    media_dicts = [item.model_dump() for item in req.media_items]

    loop = asyncio.get_event_loop()
    people_raw = await loop.run_in_executor(
        None,  # use default executor
        lambda: face_engine.cluster_faces(media_dicts, epsilon=eps, min_samples=req.min_samples),
    )

    total_faces = sum(p["face_count"] for p in people_raw)
    people = [PersonCluster(**p) for p in people_raw]

    return ClusterResponse(
        people=people,
        total_faces=total_faces,
        total_people=len(people),
        epsilon_used=eps,
    )


# ─── POST /probe — Extract embedding only (for advanced client-side matching) ─
@router.post(
    "/probe",
    dependencies=[Depends(verify_api_key)],
    summary="Extract ArcFace embedding from a selfie",
)
async def extract_probe(image_base64: str):
    """
    Extract a normalized ArcFace embedding from a selfie.
    Returns the 512-d vector so the client can do local matching
    without sending the full gallery to the server.
    """
    try:
        img_data = image_base64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        image_bytes = base64.b64decode(img_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    probe = await face_engine.extract_probe_embedding(image_bytes)
    return {
        "embedding": probe,
        "found": probe is not None,
        "dimensions": len(probe) if probe else 0,
    }
