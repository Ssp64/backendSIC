# app/core/face_engine.py — Production InsightFace/ArcFace Engine
#
# This module wraps InsightFace's buffalo_l pack which bundles:
#   • RetinaFace  — state-of-the-art face detector (outperforms MTCNN, SSD)
#   • ArcFace     — 128-d/512-d embeddings trained on MS1M-ArcFace (99.8% LFW)
#
# Key design decisions:
#   • Singleton pattern — models are heavy; load once and reuse
#   • asyncio-safe — all CPU-bound work runs in a thread pool executor
#   • Adaptive preprocessing — auto-resize, CLAHE, augmentation for hard cases
#   • No GPU required — works on CPU; add ctx_id=0 for CUDA

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# InsightFace is imported lazily to avoid startup errors when the package
# isn't installed yet (e.g., during CI lint checks).
_insightface = None
_sklearn_dbscan = None


def _lazy_imports():
    global _insightface, _sklearn_dbscan
    if _insightface is None:
        import insightface
        from insightface.app import FaceAnalysis
        _insightface = FaceAnalysis
    if _sklearn_dbscan is None:
        from sklearn.cluster import DBSCAN
        _sklearn_dbscan = DBSCAN
    return _insightface, _sklearn_dbscan


# ─── Face Engine Singleton ────────────────────────────────────────────────────
class FaceEngine:
    """
    Thread-safe, async-compatible wrapper around InsightFace's ArcFace pipeline.

    Usage:
        engine = FaceEngine()
        await engine.initialize()

        # index a photo
        faces = await engine.extract_embeddings_from_bytes(image_bytes)

        # match a selfie
        results = engine.match(probe_embedding, gallery_embeddings, threshold=0.40)

        # cluster an event
        people = engine.cluster_faces(gallery_embeddings, event_photo_ids)
    """

    def __init__(self):
        self._app = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face-worker")
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Load InsightFace models. Call once at startup."""
        async with self._lock:
            if self._initialized:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._load_models)
            self._initialized = True

    def _load_models(self):
        FaceAnalysis, _ = _lazy_imports()
        self._app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL,
            # providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU
            providers=["CPUExecutionProvider"],
        )
        # ctx_id=-1 → CPU. Use ctx_id=0 for CUDA GPU.
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info(f"InsightFace {settings.INSIGHTFACE_MODEL} loaded on CPU")

    def shutdown(self):
        self._executor.shutdown(wait=False)

    # ─── Public async API ─────────────────────────────────────────────────────

    async def extract_embeddings_from_bytes(
        self, image_bytes: bytes, url: str = ""
    ) -> List[dict]:
        """
        Detect all faces in an image and return their ArcFace embeddings.

        Returns a list of face dicts:
            {
                "embedding": List[float],    # 512-d ArcFace vector (normalized)
                "bbox": [x1, y1, x2, y2],   # face bounding box in original image
                "det_score": float,          # detection confidence 0-1
                "pose": [yaw, pitch, roll],  # head pose in degrees
            }
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_sync,
            image_bytes,
            url,
        )

    async def extract_probe_embedding(self, image_bytes: bytes) -> Optional[List[float]]:
        """
        Extract a single averaged/augmented embedding for a guest selfie probe.
        Uses multiple augmented views to improve robustness.

        Returns a normalized 512-d embedding, or None if no face found.
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_probe_sync,
            image_bytes,
        )

    # ─── Sync implementations (run in thread pool) ────────────────────────────

   def _load_image(self, image_bytes: bytes) -> np.ndarray:
    """Decode bytes → BGR numpy array, auto-rotating via EXIF, resizing large images."""
    img = Image.open(io.BytesIO(image_bytes))

    # Auto-rotate based on EXIF orientation tag
    try:
        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                from PIL import ExifTags
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
                    break
    except Exception:
        pass

    img = img.convert("RGB")
    w, h = img.size
    max_dim = settings.MAX_IMAGE_DIM
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _enhance(self, bgr: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
        luminance channel. This dramatically helps detection in dark / uneven
        lighting conditions without over-saturating bright images.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _detect_faces(self, bgr: np.ndarray) -> list:
        """Run RetinaFace detection, fall back to upscaled if tiny faces missed."""
        faces = self._app.get(bgr)

        # If no faces found and image is small, try 2× upscale
        if not faces:
            h, w = bgr.shape[:2]
            if max(h, w) < 400:
                big = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                faces = self._app.get(big)

        return [f for f in faces if f.det_score >= settings.DETECTION_THRESHOLD]

    def _extract_sync(self, image_bytes: bytes, url: str) -> List[dict]:
        """Full detection + embedding pipeline for one image."""
        try:
            bgr = self._load_image(image_bytes)
        except Exception as e:
            logger.warning(f"Image decode failed ({url}): {e}")
            return []

        # Try raw first, then CLAHE-enhanced
        faces = self._detect_faces(bgr)
        if not faces:
            faces = self._detect_faces(self._enhance(bgr))

        results = []
        for face in faces:
            emb = face.normed_embedding  # already L2-normalized 512-d float32
            if emb is None:
                continue
            bbox = face.bbox.astype(int).tolist()
            pose = face.pose.tolist() if hasattr(face, "pose") and face.pose is not None else [0, 0, 0]
            results.append(
                {
                    "embedding": emb.tolist(),
                    "bbox": bbox,
                    "det_score": float(face.det_score),
                    "pose": pose,
                }
            )

        logger.debug(f"Detected {len(results)} face(s) in {url or 'image'}")
        return results

    def _extract_probe_sync(self, image_bytes: bytes) -> Optional[List[float]]:
        """
        Multi-augmentation probe extraction for guest selfies.
        
        Strategy: run detection on 5 augmented views and average the embeddings.
        Augmentations cover brightness/contrast variations + horizontal flip.
        This gives a more robust probe for matching across different photo conditions.
        """
        try:
            bgr = self._load_image(image_bytes)
        except Exception as e:
            logger.warning(f"Probe image decode failed: {e}")
            return None

        augments = [
            bgr,
            self._enhance(bgr),
            self._apply_brightness(bgr, 1.15, 1.05),
            self._apply_brightness(bgr, 0.85, 1.10),
            cv2.flip(bgr, 1),  # horizontal flip
        ]

        embeddings = []
        for aug in augments:
            faces = self._detect_faces(aug)
            if not faces:
                # No face in this augment — try CLAHE
                faces = self._detect_faces(self._enhance(aug))

            if faces:
                # Pick the largest / most confident face
                best = max(faces, key=lambda f: f.det_score * (
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                ))
                if best.normed_embedding is not None:
                    embeddings.append(best.normed_embedding)

        if not embeddings:
            logger.info("Probe extraction: no face found in any augmentation")
            return None

        # Average and re-normalize
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        return avg.tolist()

    @staticmethod
    def _apply_brightness(bgr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """alpha = contrast [0.5-2.0], beta = brightness [0.5-2.0]."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * alpha * beta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ─── Matching ─────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine distance in [0, 2] space.
        Since ArcFace embeddings are L2-normalized, this equals:
            1 - dot(a, b)
        Values: 0.0 = identical, ~0.4 = same person threshold, >0.6 = different
        """
        return float(1.0 - np.dot(a, b))

    def match(
        self,
        probe: List[float],
        gallery: List[dict],
        threshold: Optional[float] = None,
    ) -> List[dict]:
        """
        Match a probe embedding against a gallery of indexed media rows.

        gallery items must have:
            { "id": str, "face_embeddings": [[...512d...], ...], ...rest }

        Returns sorted list (best match first):
            { "media_id": str, "distance": float, "score": int (0-100), ...row }
        """
        thresh = threshold or settings.MATCHING_THRESHOLD
        probe_arr = np.array(probe, dtype=np.float32)

        results = []
        for item in gallery:
            embeddings = item.get("face_embeddings") or []
            if not embeddings:
                continue

            best_dist = float("inf")
            for emb in embeddings:
                dist = self.cosine_distance(probe_arr, np.array(emb, dtype=np.float32))
                if dist < best_dist:
                    best_dist = dist

            if best_dist <= thresh:
                score = int(max(0, (1 - best_dist / thresh) * 100))
                results.append(
                    {
                        "media_id": item["id"],
                        "distance": round(best_dist, 4),
                        "score": score,
                        "url": item.get("url"),
                        "file_name": item.get("file_name"),
                        "file_type": item.get("file_type"),
                        "mime_type": item.get("mime_type"),
                        "storage_path": item.get("storage_path"),
                    }
                )

        results.sort(key=lambda x: x["distance"])
        return results

    # ─── Clustering ───────────────────────────────────────────────────────────

    def cluster_faces(
        self,
        media_items: List[dict],
        epsilon: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> List[dict]:
        """
        Google Photos-style person clustering using DBSCAN on ArcFace embeddings.

        Algorithm:
          1. Flatten all face embeddings from all photos into one array
          2. Run DBSCAN with cosine metric (epsilon ≈ 0.45)
          3. Each cluster = one person
          4. Map cluster membership back to photo IDs
          5. Sort by photo count descending

        Why DBSCAN over k-means:
          - No need to specify k (number of people) in advance
          - Handles noise/outliers gracefully (label -1)
          - Non-convex cluster shapes (embeddings aren't always spherical)
          - Works at any scale: 2 people or 200

        Returns sorted list of person clusters:
            {
                "person_id": str,         # deterministic hash of centroid
                "label": str,             # "Person 1", "Person 2", ...
                "photo_ids": [str, ...],  # unique photo IDs containing this person
                "photo_count": int,
                "face_count": int,
                "centroid": [float, ...], # averaged embedding (for matching)
                "representative_url": str,
                "representative_photo_id": str,
            }
        """
        DBSCAN, _ = _lazy_imports()[1], None
        from sklearn.cluster import DBSCAN as _DBSCAN

        eps = epsilon or settings.CLUSTER_EPSILON
        min_s = min_samples or settings.CLUSTER_MIN_SAMPLES

        # Build flat list of (face_embedding, media_id, photo_url)
        face_records = []
        for item in media_items:
            embs = item.get("face_embeddings") or []
            for emb in embs:
                face_records.append(
                    {
                        "embedding": np.array(emb, dtype=np.float32),
                        "media_id": item["id"],
                        "url": item.get("url", ""),
                    }
                )

        if not face_records:
            return []

        X = np.stack([r["embedding"] for r in face_records])  # shape (N, 512)

        # DBSCAN with cosine metric
        # epsilon=0.45 means faces with cosine distance ≤ 0.45 are neighbours.
        # This is equivalent to cosine similarity ≥ 0.55, which is conservative
        # enough to avoid false merges while still grouping the same person across
        # very different photos (angles, lighting, expressions).
        db = _DBSCAN(eps=eps, min_samples=min_s, metric="cosine", n_jobs=-1)
        labels = db.fit_predict(X)

        # Group by cluster label
        cluster_map: dict[int, dict] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                # DBSCAN noise — create a singleton cluster so we don't lose faces
                label = f"noise_{idx}"

            if label not in cluster_map:
                cluster_map[label] = {
                    "embeddings": [],
                    "photo_ids": set(),
                    "representative_media_id": face_records[idx]["media_id"],
                    "representative_url": face_records[idx]["url"],
                }

            cl = cluster_map[label]
            cl["embeddings"].append(face_records[idx]["embedding"])
            cl["photo_ids"].add(face_records[idx]["media_id"])

        # Sort by photo count
        sorted_clusters = sorted(
            cluster_map.values(),
            key=lambda c: len(c["photo_ids"]),
            reverse=True,
        )

        # Build output
        people = []
        for i, cl in enumerate(sorted_clusters):
            centroid = np.mean(cl["embeddings"], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            people.append(
                {
                    "person_index": i,
                    "label": f"Person {i + 1}",
                    "photo_ids": sorted(cl["photo_ids"]),
                    "photo_count": len(cl["photo_ids"]),
                    "face_count": len(cl["embeddings"]),
                    "centroid": centroid.tolist(),
                    "representative_url": cl["representative_url"],
                    "representative_photo_id": cl["representative_media_id"],
                }
            )

        logger.info(
            f"Clustering: {len(face_records)} faces → {len(people)} people "
            f"(eps={eps}, min_samples={min_s})"
        )
        return people


# ─── Module-level singleton ───────────────────────────────────────────────────
face_engine = FaceEngine()
