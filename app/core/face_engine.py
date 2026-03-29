# app/core/face_engine.py — InsightFace/ArcFace engine
#
# Detection strategy: exhaustive — tries every preprocessing combination
# until a face is found. Nothing is skipped.
#
# Clustering strategy: Agglomerative (cosine + average linkage) followed by
# a minimum-pairwise-distance post-merge pass that catches outlier embeddings
# the centroid-only approach misses.

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_FaceAnalysis = None


def _lazy_import():
    global _FaceAnalysis
    if _FaceAnalysis is None:
        from insightface.app import FaceAnalysis
        _FaceAnalysis = FaceAnalysis
    return _FaceAnalysis


class FaceEngine:

    def __init__(self):
        self._app         = None
        self._executor    = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face")
        self._lock        = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        async with self._lock:
            if self._initialized:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._load_models)
            self._initialized = True

    def _load_models(self):
        FA        = _lazy_import()
        self._app = FA(name=settings.INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info(f"InsightFace {settings.INSIGHTFACE_MODEL} ready")

    def shutdown(self):
        self._executor.shutdown(wait=False)

    # ── Public async API ──────────────────────────────────────────────────────

    async def extract_embeddings_from_bytes(
        self, image_bytes: bytes, url: str = ""
    ) -> List[dict]:
        if not self._initialized:
            await self.initialize()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._extract_sync, image_bytes, url
        )

    async def extract_probe_embedding(
        self, image_bytes: bytes
    ) -> Optional[List[float]]:
        if not self._initialized:
            await self.initialize()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._extract_probe_sync, image_bytes
        )

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes))

        try:
            from PIL import ExifTags
            exif = img._getexif() or {}
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if   val == 3: img = img.rotate(180, expand=True)
                    elif val == 6: img = img.rotate(270, expand=True)
                    elif val == 8: img = img.rotate(90,  expand=True)
                    break
        except Exception:
            pass

        img = img.convert("RGB")
        w, h = img.size
        md = settings.MAX_IMAGE_DIM
        if max(w, h) > md:
            scale = md / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_grayscale(bgr: np.ndarray) -> bool:
        b, g, r = cv2.split(bgr.astype(np.float32))
        return np.mean(np.abs(r - g)) < 8.0 and np.mean(np.abs(r - b)) < 8.0

    @staticmethod
    def _colorize(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = np.clip(gray * 1.08, 0, 255).astype(np.uint8)
        g = np.clip(gray * 0.92, 0, 255).astype(np.uint8)
        b = np.clip(gray * 0.78, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    @staticmethod
    def _clahe(bgr: np.ndarray) -> np.ndarray:
        lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _brighten(bgr: np.ndarray, alpha: float = 1.5, beta: float = 1.3) -> np.ndarray:
        hsv          = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * alpha * beta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _upscale(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        # Always upscale by 1.5x — even large photos benefit for small faces
        return cv2.resize(bgr,
                          (int(w * 1.5), int(h * 1.5)),
                          interpolation=cv2.INTER_CUBIC)

    # ── Raw detection wrappers ────────────────────────────────────────────────

    def _raw640(self, bgr: np.ndarray) -> list:
        return self._app.get(bgr) or []

    def _raw320(self, bgr: np.ndarray) -> list:
        self._app.det_model.input_size = (320, 320)
        try:
            return self._app.get(bgr) or []
        finally:
            self._app.det_model.input_size = (640, 640)

    # ── Exhaustive detection ──────────────────────────────────────────────────

    def _find_faces(self, bgr: np.ndarray) -> list:
        """
        Try every combination of preprocessing until faces are found.
        8 variants × 2 det_sizes × 2 thresholds = up to 32 attempts.
        Returns on the FIRST attempt that yields results.
        """
        nt = settings.DETECTION_THRESHOLD          # normal
        lt = max(nt - 0.10, 0.05)                  # low (lenient)

        up = self._upscale(bgr)

        variants = [
            bgr,
            up,
            self._clahe(bgr),
            self._clahe(up),
            self._brighten(bgr),
            self._brighten(up),
            self._clahe(self._brighten(bgr)),
            self._clahe(self._brighten(up)),
        ]

        def above(faces, thresh):
            return [f for f in faces if f.det_score >= thresh]

        for thresh in (nt, lt):
            # det_size=640 pass
            for v in variants:
                hits = above(self._raw640(v), thresh)
                if hits:
                    return hits
            # det_size=320 pass
            for v in variants:
                hits = above(self._raw320(v), thresh)
                if hits:
                    return hits

        return []

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _extract_sync(self, image_bytes: bytes, url: str) -> List[dict]:
        try:
            bgr = self._load_image(image_bytes)
        except Exception as e:
            logger.warning(f"Image decode failed ({url}): {e}")
            return []

        is_gray = self._is_grayscale(bgr)
        primary = self._colorize(bgr) if is_gray else bgr

        faces = self._find_faces(primary)
        if not faces and is_gray:
            faces = self._find_faces(bgr)  # last resort: raw grayscale

        if not faces:
            logger.debug(f"No face found: {url or 'image'}")
            return []

        results = []
        for face in faces:
            if face.normed_embedding is None:
                continue

            bbox = face.bbox.astype(int).tolist()
            pose = (face.pose.tolist()
                    if hasattr(face, "pose") and face.pose is not None
                    else [0.0, 0.0, 0.0])

            if is_gray:
                # Average embedding across colorized augments for stability
                aug_embs = [face.normed_embedding]
                for aug in [self._clahe(primary), self._brighten(primary, 1.1, 1.0)]:
                    af = self._find_faces(aug)
                    if af:
                        best = max(af, key=lambda f: float(f.det_score))
                        if best.normed_embedding is not None:
                            aug_embs.append(best.normed_embedding)
                avg  = np.mean(aug_embs, axis=0)
                norm = np.linalg.norm(avg)
                emb  = (avg / norm if norm > 0 else avg).tolist()
            else:
                emb = face.normed_embedding.tolist()

            results.append({
                "embedding":  emb,
                "bbox":       bbox,
                "det_score":  float(face.det_score),
                "pose":       pose,
            })

        logger.debug(f"{len(results)} face(s) in {url or 'image'} (gray={is_gray})")
        return results

    def _extract_probe_sync(self, image_bytes: bytes) -> Optional[List[float]]:
        try:
            bgr = self._load_image(image_bytes)
        except Exception as e:
            logger.warning(f"Probe decode failed: {e}")
            return None

        augments = [
            bgr,
            self._clahe(bgr),
            self._brighten(bgr, 1.15, 1.05),
            self._brighten(bgr, 0.85, 1.10),
            cv2.flip(bgr, 1),
        ]

        embeddings = []
        for aug in augments:
            faces = self._find_faces(aug)
            if faces:
                best = max(
                    faces,
                    key=lambda f: float(f.det_score)
                    * (f.bbox[2] - f.bbox[0])
                    * (f.bbox[3] - f.bbox[1]),
                )
                if best.normed_embedding is not None:
                    embeddings.append(best.normed_embedding)

        if not embeddings:
            logger.info("Probe: no face found")
            return None

        avg  = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        return (avg / norm if norm > 0 else avg).tolist()

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(
        self,
        probe: List[float],
        gallery: List[dict],
        threshold: Optional[float] = None,
    ) -> List[dict]:
        thresh    = threshold if threshold is not None else settings.MATCHING_THRESHOLD
        probe_arr = np.array(probe, dtype=np.float32)

        results = []
        for item in gallery:
            embeddings = item.get("face_embeddings") or []
            if not embeddings:
                continue
            best_dist = min(
                float(1.0 - np.dot(probe_arr, np.array(emb, dtype=np.float32)))
                for emb in embeddings
            )
            if best_dist <= thresh:
                score = int(max(0, (1.0 - best_dist / thresh) * 100))
                results.append({
                    "media_id":     item["id"],
                    "distance":     round(best_dist, 4),
                    "score":        score,
                    "url":          item.get("url"),
                    "file_name":    item.get("file_name"),
                    "file_type":    item.get("file_type"),
                    "mime_type":    item.get("mime_type"),
                    "storage_path": item.get("storage_path"),
                })

        results.sort(key=lambda x: x["distance"])
        return results

    # ── Clustering ────────────────────────────────────────────────────────────

    def cluster_faces(
        self,
        media_items: List[dict],
        epsilon: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> List[dict]:
        """
        Cluster face embeddings into person groups.

        Pipeline:
          1. Build L2-normalised face records
          2. Agglomerative clustering (cosine + average linkage)
          3. Post-merge: minimum pairwise distance between cluster pairs
             (catches outlier embeddings that centroid distance misses)
          4. Deduplicate single-face photos to one cluster
          5. Sort by photo count, build output
        """
        from sklearn.cluster import AgglomerativeClustering

        dist_thresh = epsilon if epsilon is not None else settings.CLUSTER_EPSILON

        # 1. Build face records (L2-normalised)
        face_records: List[dict] = []
        for item in media_items:
            for emb in (item.get("face_embeddings") or []):
                arr  = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                face_records.append({
                    "embedding": arr,
                    "media_id":  item["id"],
                    "url":       item.get("url", ""),
                })

        if not face_records:
            return []

        X = np.stack([r["embedding"] for r in face_records])

        # 2. Agglomerative clustering
        if len(face_records) == 1:
            labels = np.array([0])
        else:
            ac = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_thresh,
                metric="cosine",
                linkage="average",
            )
            labels = ac.fit_predict(X)

        def centroid(embs: List[np.ndarray]) -> np.ndarray:
            c    = np.mean(embs, axis=0)
            norm = np.linalg.norm(c)
            return c / norm if norm > 0 else c

        def min_pair_dist(ea: List[np.ndarray], eb: List[np.ndarray]) -> float:
            """Minimum cosine distance between any pair across two clusters."""
            A    = np.stack(ea)   # (Na, 512)
            B    = np.stack(eb)   # (Nb, 512)
            sims = A @ B.T        # cosine similarities (both L2-normalised)
            return float(1.0 - sims.max())

        # 3. Build cluster map
        cluster_map: dict = {}
        for idx, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = {
                    "embeddings": [],
                    "photo_ids":  set(),
                    "rep_url":    face_records[idx]["url"],
                    "rep_id":     face_records[idx]["media_id"],
                }
            cluster_map[label]["embeddings"].append(face_records[idx]["embedding"])
            cluster_map[label]["photo_ids"].add(face_records[idx]["media_id"])

        # 4. Post-merge using minimum pairwise distance
        # MERGE_THRESH = 0.67 means two faces are merged if ANY pair of embeddings
        # across the two clusters is within 0.67 cosine distance.
        # This is intentionally generous for the 10% that agglomerative misses
        # due to lighting / pose variation, while staying below ~0.70 where
        # genuinely different people start appearing.
        MERGE_THRESH = 0.67

        changed = True
        while changed:
            changed = False
            keys = list(cluster_map.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ka, kb = keys[i], keys[j]
                    if ka not in cluster_map or kb not in cluster_map:
                        continue
                    if min_pair_dist(cluster_map[ka]["embeddings"],
                                     cluster_map[kb]["embeddings"]) <= MERGE_THRESH:
                        cluster_map[ka]["embeddings"].extend(cluster_map[kb]["embeddings"])
                        cluster_map[ka]["photo_ids"].update(cluster_map[kb]["photo_ids"])
                        del cluster_map[kb]
                        changed = True
                        break
                if changed:
                    break

        # 5. Deduplicate: single-face photos → one cluster
        photo_face_count: dict = {}
        for r in face_records:
            photo_face_count[r["media_id"]] = photo_face_count.get(r["media_id"], 0) + 1

        photo_best: dict = {}
        for key, cl in cluster_map.items():
            c = centroid(cl["embeddings"])
            for r in face_records:
                if r["media_id"] not in cl["photo_ids"]:
                    continue
                dist = float(1.0 - np.dot(r["embedding"], c))
                prev = photo_best.get(r["media_id"])
                if prev is None or dist < prev[1]:
                    photo_best[r["media_id"]] = (key, dist)

        for media_id, (best_key, _) in photo_best.items():
            if photo_face_count.get(media_id, 1) > 1:
                continue
            for key in list(cluster_map.keys()):
                if key != best_key:
                    cluster_map[key]["photo_ids"].discard(media_id)

        cluster_map = {k: v for k, v in cluster_map.items() if v["photo_ids"]}

        # 6. Sort + build output
        sorted_clusters = sorted(
            cluster_map.values(),
            key=lambda c: len(c["photo_ids"]),
            reverse=True,
        )

        people = []
        for i, cl in enumerate(sorted_clusters):
            c = centroid(cl["embeddings"])

            best_k  = min(range(len(cl["embeddings"])),
                          key=lambda k: float(1.0 - np.dot(cl["embeddings"][k], c)))
            rep_emb = cl["embeddings"][best_k]
            rep_rec = next(
                (r for r in face_records
                 if r["media_id"] in cl["photo_ids"]
                 and np.allclose(r["embedding"], rep_emb, atol=1e-5)),
                None,
            )

            people.append({
                "person_index":            i,
                "label":                   f"Person {i + 1}",
                "photo_ids":               sorted(cl["photo_ids"]),
                "photo_count":             len(cl["photo_ids"]),
                "face_count":              len(cl["embeddings"]),
                "centroid":                c.tolist(),
                "representative_url":      rep_rec["url"]      if rep_rec else cl["rep_url"],
                "representative_photo_id": rep_rec["media_id"] if rep_rec else cl["rep_id"],
            })

        logger.info(
            f"Clustered {len(face_records)} faces → {len(people)} people "
            f"(dist_thresh={dist_thresh:.2f}, merge_thresh={MERGE_THRESH})"
        )
        return people


# Module-level singleton
face_engine = FaceEngine()
