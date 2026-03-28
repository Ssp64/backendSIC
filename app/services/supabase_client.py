# app/services/supabase_client.py — Server-side Supabase client
# Uses SERVICE ROLE key — bypasses RLS. Backend only, never expose to browser.

import json
import logging
from typing import List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class SupabaseServiceClient:

    def __init__(self):
        self._base    = settings.SUPABASE_URL.rstrip("/")
        self._headers = {
            "apikey":        settings.SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        }
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                headers=self._headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def save_face_embeddings(self, media_id: str, face_results: List[dict]) -> bool:
        """Store ArcFace embeddings into the media row (jsonb column)."""
        payload = {
            "face_embeddings": [r["embedding"] for r in face_results],
            "face_count":      len(face_results),
            "face_metadata":   [
                {
                    "bbox":      r["bbox"],
                    "det_score": r["det_score"],
                    "pose":      r.get("pose", [0, 0, 0]),
                }
                for r in face_results
            ],
        }
        resp = await self.client.patch(
            f"/rest/v1/media?id=eq.{media_id}",
            json=payload,
        )
        if resp.status_code not in (200, 204):
            logger.error(f"save_face_embeddings failed for {media_id}: {resp.text}")
            return False
        return True

    async def get_event_gallery(self, event_id: str) -> List[dict]:
        """Fetch all indexed media rows for an event."""
        resp = await self.client.get(
            "/rest/v1/media",
            params={
                "event_id":  f"eq.{event_id}",
                "file_type": "eq.image",
                "select":    "id,url,file_name,file_type,mime_type,storage_path,face_embeddings,face_count",
                "order":     "created_at.desc",
            },
        )
        if resp.status_code != 200:
            logger.error(f"get_event_gallery failed: {resp.text}")
            return []

        rows   = resp.json()
        result = []
        for row in rows:
            raw = row.get("face_embeddings")
            if raw:
                try:
                    embs = json.loads(raw) if isinstance(raw, str) else raw
                    row["face_embeddings"] = embs
                except Exception:
                    row["face_embeddings"] = []
            else:
                row["face_embeddings"] = []
            result.append(row)
        return result

    async def download_image(self, url: str) -> Optional[bytes]:
        """Download an image from a public Supabase storage URL."""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, follow_redirects=True)
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.warning(f"download_image failed ({url}): {e}")
        return None


supabase_service = SupabaseServiceClient()
