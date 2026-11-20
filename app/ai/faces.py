import gc
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.ai.quality import decode_image_array
import app.database as db_module

logger = logging.getLogger("OnDemandFaces")


def _unit_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32).flatten()
    n = float(np.linalg.norm(arr))
    return (arr / n) if n > 1e-7 else arr


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(max(1, union))


class OnDemandFaceEmbedder:
    """
    Task 4: Lightweight, 100% local, thumbnail-only face detector and 256-D Face Identity
    Vector extractor. Loads optional ONNX/OpenCV models strictly on demand (0 MB RAM at rest)
    and frees all memory in `close_session()` / `__exit__()`.
    NEVER accesses physical USB drives — works 100% offline on `data/thumbnails/*.webp`.
    """

    VECTOR_DIM = 256
    _ONNX_PROJ: Optional[np.ndarray] = None

    def __init__(self):
        self.yunet_path = db_module.DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"
        self.sface_path = db_module.DATA_DIR / "models" / "face_recognition_sface_2021dec.onnx"
        self.mobilenet_path = db_module.DATA_DIR / "models" / "mobilenetv2.onnx"
        self._session = None
        self._yunet = None
        self._sface = None
        self._frontal_cascade = None
        self._profile_cascade = None
        self._eye_cascade = None

    @classmethod
    def _get_onnx_projection(cls) -> np.ndarray:
        if cls._ONNX_PROJ is None:
            rng = np.random.default_rng(20261120)
            mat = rng.standard_normal((1000, 32)).astype(np.float32)
            # Column-normalize for stable cosine geometry
            norms = np.linalg.norm(mat, axis=0, keepdims=True)
            cls._ONNX_PROJ = mat / np.maximum(norms, 1e-7)
        return cls._ONNX_PROJ

    def open_session(self) -> None:
        # Tier 1: Optional YuNet ONNX face detector (~230 KB)
        if self._yunet is None and self.yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(self.yunet_path.resolve()),
                    "",
                    (300, 300),
                    score_threshold=0.65,
                    nms_threshold=0.35,
                    top_k=20,
                )
            except Exception as exc:
                logger.debug("YuNet detector init skipped: %s", exc)
                self._yunet = None

        # Tier 2: Optional Haar cascades (if bundled in cv2.data.haarcascades or data/models)
        for attr, fname in (
            ("_frontal_cascade", "haarcascade_frontalface_default.xml"),
            ("_profile_cascade", "haarcascade_profileface.xml"),
            ("_eye_cascade", "haarcascade_eye.xml"),
        ):
            for base_dir in (Path(cv2.data.haarcascades), db_module.DATA_DIR / "models"):
                cand = base_dir / fname
                if cand.exists():
                    try:
                        clf = cv2.CascadeClassifier(str(cand))
                        if not clf.empty():
                            setattr(self, attr, clf)
                            break
                    except Exception:
                        pass

        # Optional SFace recognizer (~36 MB) or existing MobileNetV2 ONNX (~14 MB) on cropped face ROI
        if self._sface is None and self.sface_path.exists() and hasattr(cv2, "FaceRecognizerSF"):
            try:
                self._sface = cv2.FaceRecognizerSF.create(str(self.sface_path.resolve()), "")
            except Exception as exc:
                logger.debug("SFace init skipped: %s", exc)
                self._sface = None

        if (
            self._sface is None
            and self._session is None
            and self.mobilenet_path.exists()
            and self.mobilenet_path.stat().st_size > 1_000_000
        ):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self._session = ort.InferenceSession(
                    str(self.mobilenet_path.resolve()),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                logger.debug("MobileNetV2 face session init skipped: %s", exc)
                self._session = None

    def close_session(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None
        self._yunet = None
        self._sface = None
        self._frontal_cascade = None
        self._profile_cascade = None
        self._eye_cascade = None
        gc.collect()

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    @staticmethod
    def _skin_mask_ycrcb(img_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, (45, 133, 77), (255, 175, 127))
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, (0, 22, 45), (32, 230, 255))
        mask_hsv2 = cv2.inRange(hsv, (165, 22, 45), (179, 230, 255))
        return cv2.bitwise_and(mask_ycrcb, cv2.bitwise_or(mask_hsv, mask_hsv2))

    def detect_faces(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects faces in a thumbnail BGR array and returns `[(x, y, w, h, confidence), ...]`.
        Uses 3-tier local pipeline:
        1) YuNet ONNX (`cv2.FaceDetectorYN`) if model file present
        2) OpenCV Haar frontal + profile cascades validated with YCrCb skin tone if XMLs present
        3) Zero-file built-in YCrCb skin-region + ovality + facial-feature contrast detector
        """
        h, w = img_bgr.shape[:2]
        if h < 32 or w < 32:
            return []

        raw_boxes: List[Tuple[int, int, int, int, float]] = []

        # Tier 1: YuNet
        if self._yunet is not None:
            try:
                self._yunet.setInputSize((w, h))
                _, faces = self._yunet.detect(img_bgr)
                if faces is not None:
                    for f in faces:
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        score = float(f[14]) if len(f) > 14 else 0.9
                        fx = max(0, min(w - 1, fx))
                        fy = max(0, min(h - 1, fy))
                        fw = min(w - fx, max(1, fw))
                        fh = min(h - fy, max(1, fh))
                        if fw >= 12 and fh >= 12:
                            raw_boxes.append((fx, fy, fw, fh, round(min(1.0, score), 3)))
            except Exception as exc:
                logger.debug("YuNet detect warning: %s", exc)

        # Tier 2: Haar cascades with YCrCb skin validation (ONLY when YuNet is unavailable)
        if not raw_boxes and self._yunet is None and (self._frontal_cascade is not None or self._profile_cascade is not None):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            skin_full = self._skin_mask_ycrcb(img_bgr)
            for clf, base_conf in ((self._frontal_cascade, 0.90), (self._profile_cascade, 0.80)):
                if clf is None:
                    continue
                try:
                    found = clf.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(24, 24))
                    for fx, fy, fw, fh in found:
                        roi_skin = skin_full[fy : fy + fh, fx : fx + fw]
                        skin_ratio = float(np.count_nonzero(roi_skin)) / float(max(1, fw * fh))
                        if skin_ratio >= 0.18:
                            conf = base_conf
                            if self._eye_cascade is not None:
                                eyes = self._eye_cascade.detectMultiScale(
                                    gray[fy : fy + int(fh * 0.6), fx : fx + fw],
                                    scaleFactor=1.1,
                                    minNeighbors=3,
                                    minSize=(8, 8),
                                )
                                if len(eyes) >= 1:
                                    conf = min(0.98, conf + 0.06)
                            raw_boxes.append((int(fx), int(fy), int(fw), int(fh), round(conf, 3)))
                except Exception as exc:
                    logger.debug("Haar detect warning: %s", exc)

        # Tier 3: Zero-file built-in YCrCb skin-blob fallback ONLY when neither YuNet nor Haar cascades are loaded
        # (Prevents false-positive "faces" on tan rocks/roofs/drones/landscapes when YuNet legitimately finds 0 faces!)
        if not raw_boxes and self._yunet is None and self._frontal_cascade is None and self._profile_cascade is None:
            skin = self._skin_mask_ycrcb(img_bgr)
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            cleaned = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k_open)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_area = float(w * h)

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < 500 or area > img_area * 0.80:
                    continue
                fx, fy, fw, fh = cv2.boundingRect(cnt)
                if fw < 24 or fh < 24:
                    continue
                aspect = float(fw) / float(max(1, fh))
                if not (0.52 <= aspect <= 1.48):
                    continue
                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                solidity = area / max(1.0, hull_area)
                if solidity < 0.55:
                    continue

                # Verify internal facial structure (eyes / brows / mouth create distinct luminance contrast,
                # whereas flat beige walls or sand have near-zero internal contrast!)
                roi_gray = gray[fy : fy + fh, fx : fx + fw]
                upper_roi = roi_gray[int(fh * 0.15) : int(fh * 0.60), int(fw * 0.12) : int(fw * 0.88)]
                if upper_roi.size < 16:
                    continue
                inner_std = float(np.std(roi_gray))
                upper_std = float(np.std(upper_roi))
                if inner_std < 11.0 or upper_std < 9.0:
                    continue

                # Score combines oval solidity and facial feature contrast
                conf = round(min(0.92, 0.58 + 0.20 * min(1.0, solidity) + 0.14 * min(1.0, upper_std / 35.0)), 3)
                raw_boxes.append((int(fx), int(fy), int(fw), int(fh), conf))

        if not raw_boxes:
            return []

        # Deduplicate overlapping boxes with Non-Maximum Suppression (keep highest confidence / largest)
        raw_boxes.sort(key=lambda b: (b[4], b[2] * b[3]), reverse=True)
        kept: List[Tuple[int, int, int, int, float]] = []
        for cand in raw_boxes:
            if all(_iou(cand[:4], k[:4]) < 0.40 for k in kept):
                kept.append(cand)
            if len(kept) >= 8:
                break
        return kept

    def extract_face_vector(self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extracts a 256-D float32 L2-normalized Face Identity Vector strictly from the oval-masked
        face ROI (discarding background & clothing), using CLAHE lighting normalization + 4x4 HOG
        geometry + 4x4 facial structural contrast + CIE LAB face profile (+ optional ONNX face projection).
        """
        h, w = img_bgr.shape[:2]
        fx, fy, fw, fh = bbox
        fx = max(0, min(w - 1, int(fx)))
        fy = max(0, min(h - 1, int(fy)))
        fw = max(4, min(w - fx, int(fw)))
        fh = max(4, min(h - fy, int(fh)))

        crop = img_bgr[fy : fy + fh, fx : fx + fw]
        face_64 = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)

        # 1. Oval face mask — discards background & clothing around corners
        oval_mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.ellipse(oval_mask, (32, 32), (28, 30), 0, 0, 360, 255, -1)

        # 2. Local lighting & contrast normalization via CLAHE on CIE LAB L channel
        lab_64 = cv2.cvtColor(face_64, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_norm = clahe.apply(lab_64[:, :, 0])
        lab_64[:, :, 0] = l_norm
        norm_bgr_64 = cv2.cvtColor(lab_64, cv2.COLOR_LAB2BGR)
        norm_bgr_64 = cv2.bitwise_and(norm_bgr_64, norm_bgr_64, mask=oval_mask)
        gray_64 = cv2.bitwise_and(l_norm, l_norm, mask=oval_mask)

        # 3. Block 1 (128-D, weight 0.45): 4x4 spatial grid HOG (16 cells x 8 orientation bins)
        gx = cv2.Sobel(gray_64, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_64, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hog_parts: List[np.ndarray] = []
        for cy in range(4):
            for cx in range(4):
                m_cell = mag[cy * 16 : (cy + 1) * 16, cx * 16 : (cx + 1) * 16]
                a_cell = ang[cy * 16 : (cy + 1) * 16, cx * 16 : (cx + 1) * 16]
                hist, _ = np.histogram(a_cell, bins=8, range=(0.0, 360.0), weights=m_cell)
                hog_parts.append(np.sqrt(np.maximum(0.0, hist.astype(np.float32))))
        hog_vec = _unit_norm(np.concatenate(hog_parts)) * np.sqrt(0.45)

        # 4. Block 2 (64-D, weight 0.30): 4x4 spatial facial structure / local contrast descriptor
        struct_feats: List[float] = []
        g_float = gray_64.astype(np.float32) / 255.0
        for cy in range(4):
            for cx in range(4):
                patch = g_float[cy * 16 : (cy + 1) * 16, cx * 16 : (cx + 1) * 16]
                m_val = float(np.mean(patch))
                s_val = float(np.std(patch))
                h_diff = float(np.mean(patch[:, 8:]) - np.mean(patch[:, :8]))
                v_diff = float(np.mean(patch[8:, :]) - np.mean(patch[:8, :]))
                struct_feats.extend([m_val - 0.5, s_val, h_diff, v_diff])
        struct_vec = _unit_norm(np.array(struct_feats, dtype=np.float32)) * np.sqrt(0.30)

        # 5. Block 3 (64-D, weight 0.25): Face CIE LAB tone/chroma profile (+ optional ONNX face projection)
        # 4x4 LAB chroma/luminance (16 * 3 = 48-D) + 16-D horizontal symmetry/zone profile OR 32-D LAB + 32-D ONNX
        small_lab = cv2.resize(lab_64, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        small_lab = (small_lab - 128.0) / 128.0
        lab_48 = small_lab.flatten()  # 48 dims

        onnx_32: Optional[np.ndarray] = None
        if self._sface is not None:
            try:
                face_112 = cv2.resize(norm_bgr_64, (112, 112), interpolation=cv2.INTER_CUBIC)
                sf_feat = self._sface.feature(face_112).astype(np.float32).flatten()
                if sf_feat.size >= 32:
                    onnx_32 = _unit_norm(sf_feat[:32])
            except Exception as exc:
                logger.debug("SFace feature extraction skipped: %s", exc)
                onnx_32 = None
        elif self._session is not None:
            try:
                rgb_224 = cv2.cvtColor(
                    cv2.resize(norm_bgr_64, (224, 224), interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                tensor = np.expand_dims(np.transpose((rgb_224 - mean) / std, (2, 0, 1)), axis=0)
                in_name = self._session.get_inputs()[0].name
                out_name = self._session.get_outputs()[0].name
                logits = self._session.run([out_name], {in_name: tensor})[0][0].astype(np.float32)
                logits_n = _unit_norm(logits)
                onnx_32 = _unit_norm(np.dot(logits_n, self._get_onnx_projection()))
            except Exception as exc:
                logger.debug("ONNX face ROI projection skipped: %s", exc)
                onnx_32 = None

        if onnx_32 is not None:
            # 32-D LAB chroma (a, b channels on 4x4 grid = 32 dims) + 32-D ONNX face projection = 64 dims
            lab_ab_32 = _unit_norm(small_lab[:, :, 1:3].flatten()) * np.sqrt(0.50)
            b3_raw = np.concatenate([lab_ab_32, onnx_32 * np.sqrt(0.50)])
        else:
            # 48-D LAB (4x4x3) + 16-D vertical band profile (forehead/eyes/nose/mouth) = 64 dims
            band_feats: List[float] = []
            for by in range(4):
                band = lab_64[by * 16 : (by + 1) * 16, 12:52, :].astype(np.float32)
                band_feats.extend([
                    (float(np.mean(band[:, :, 0])) - 128.0) / 128.0,
                    (float(np.mean(band[:, :, 1])) - 128.0) / 128.0,
                    (float(np.mean(band[:, :, 2])) - 128.0) / 128.0,
                    float(np.std(band[:, :, 0])) / 64.0,
                ])
            b3_raw = np.concatenate([lab_48, np.array(band_feats, dtype=np.float32)])

        block3_vec = _unit_norm(b3_raw) * np.sqrt(0.25)

        final_vec = np.concatenate([hog_vec, struct_vec, block3_vec]).astype(np.float32)
        return _unit_norm(final_vec)

    def extract_faces_from_thumbnail(
        self,
        thumb_path: Path,
        allow_center_portrait_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extracts detected faces and their 256-D identity vectors strictly from a local thumbnail file.
        """
        img = decode_image_array(thumb_path, grayscale=False)
        if img is None:
            return []

        auto_opened = False
        if (
            self._session is None
            and self._sface is None
            and self._yunet is None
            and self._frontal_cascade is None
        ):
            self.open_session()
            auto_opened = True

        try:
            h, w = img.shape[:2]
            boxes = self.detect_faces(img)
            if not boxes and allow_center_portrait_fallback and h >= 32 and w >= 32:
                # Only used when user explicitly requests same-person search on a photo where detector found 0 boxes
                cw, ch = int(w * 0.55), int(h * 0.55)
                cx, cy = (w - cw) // 2, max(0, int(h * 0.12))
                boxes = [(cx, cy, cw, ch, 0.50)]

            out: List[Dict[str, Any]] = []
            for idx, (fx, fy, fw, fh, conf) in enumerate(boxes):
                vec = self.extract_face_vector(img, (fx, fy, fw, fh))
                out.append(
                    {
                        "face_index": idx,
                        "bbox_norm": [
                            round(fx / float(max(1, w)), 4),
                            round(fy / float(max(1, h)), 4),
                            round(fw / float(max(1, w)), 4),
                            round(fh / float(max(1, h)), 4),
                        ],
                        "confidence": float(conf),
                        "vector": vec,
                        "vector_bytes": vec.tobytes(),
                    }
                )
            return out
        finally:
            if auto_opened:
                self.close_session()

    def index_faces_for_items(
        self,
        db: Session,
        item_ids: List[str],
        thumbnails_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_cb: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Indexes faces for `item_ids` reading ONLY local thumbnails (100% offline safe!).
        Preserves any existing `person_name` on re-indexed items and releases all models in `finally:`.
        Supports cooperative stopping via `stop_cb`.
        """
        from app.models import MediaFace, MediaItem, MediaTag

        eff_thumbs = Path(thumbnails_dir) if thumbnails_dir is not None else db_module.THUMBNAILS_DIR
        self.open_session()
        processed = 0
        faces_found_total = 0
        items_with_faces = 0
        stopped = False
        remaining_ids: List[str] = []

        try:
            total = len(item_ids)
            for ch_start in range(0, total, 100):
                if stop_cb and stop_cb():
                    stopped = True
                    remaining_ids = item_ids[ch_start:]
                    break
                ch_ids = item_ids[ch_start : ch_start + 100]
                rows = db.execute(
                    select(MediaItem.id, MediaItem.thumbnail_relative_path).where(MediaItem.id.in_(ch_ids))
                ).all()

                existing_names: Dict[Tuple[str, int], str] = {}
                for mid_r, fidx_r, pname_r in db.execute(
                    select(MediaFace.media_id, MediaFace.face_index, MediaFace.person_name)
                    .where(MediaFace.media_id.in_(ch_ids), MediaFace.person_name.is_not(None))
                ).all():
                    existing_names[(str(mid_r), int(fidx_r or 0))] = str(pname_r)

                for idx_in_ch, (mid, thumb_rel) in enumerate(rows):
                    if stop_cb and stop_cb():
                        stopped = True
                        remaining_ids = item_ids[ch_start + idx_in_ch :]
                        break
                    processed += 1
                    if not thumb_rel:
                        continue
                    thumb_p = eff_thumbs / thumb_rel
                    if not thumb_p.exists() or not thumb_p.is_file() or thumb_p.stat().st_size == 0:
                        continue

                    faces = self.extract_faces_from_thumbnail(thumb_p)
                    db.execute(delete(MediaFace).where(MediaFace.media_id == mid))
                    if faces:
                        items_with_faces += 1
                        for f in faces:
                            f_idx = int(f["face_index"])
                            person_name = existing_names.get((str(mid), f_idx)) or (
                                existing_names.get((str(mid), 0)) if f_idx == 0 else None
                            )
                            db.add(
                                MediaFace(
                                    media_id=str(mid),
                                    face_index=f_idx,
                                    bbox_json=json.dumps(f["bbox_norm"]),
                                    confidence=f["confidence"],
                                    person_name=person_name,
                                    vector=f["vector_bytes"],
                                )
                            )
                            faces_found_total += 1
                        # Ensure searchable face tag exists without duplicating
                        has_face_tag = db.scalar(
                            select(MediaTag.id).where(
                                MediaTag.media_id == str(mid),
                                MediaTag.tag == "osoba / twarz",
                            ).limit(1)
                        )
                        if has_face_tag is None:
                            db.add(
                                MediaTag(
                                    media_id=str(mid),
                                    tag="osoba / twarz",
                                    confidence=round(faces[0]["confidence"], 3),
                                    source_model="face_cluster",
                                )
                            )
                    if progress_cb and (processed % 20 == 0 or processed == total):
                        progress_cb(processed, total)
                db.commit()
                if stopped:
                    break

            return {
                "processed": processed,
                "items_with_faces": items_with_faces,
                "faces_found": faces_found_total,
                "stopped": stopped,
                "remaining_ids": remaining_ids,
            }
        finally:
            self.close_session()

    @classmethod
    def rank_similar_faces(
        cls,
        query_vectors: List[np.ndarray],
        candidate_rows: List[Tuple[str, bytes]],
        min_score: float = 0.45,
        top_n: int = 48,
    ) -> List[Tuple[str, float]]:
        """
        Computes max cosine similarity between query face vectors and candidate face vectors.
        Returns sorted `[(media_id, similarity_0_to_1), ...]`.
        """
        if not query_vectors or not candidate_rows:
            return []
        q_mat = np.vstack([_unit_norm(v) for v in query_vectors])  # (Q, 256)
        dim_bytes = cls.VECTOR_DIM * 4

        best_by_mid: Dict[str, float] = {}
        valid_mids: List[str] = []
        cand_vecs: List[np.ndarray] = []
        for mid, blob in candidate_rows:
            if not blob or len(blob) != dim_bytes:
                continue
            valid_mids.append(str(mid))
            cand_vecs.append(np.frombuffer(blob, dtype=np.float32))

        if not cand_vecs:
            return []

        c_mat = np.vstack(cand_vecs)  # (N, 256)
        sim_matrix = np.dot(c_mat, q_mat.T)  # (N, Q)
        max_per_face = np.max(sim_matrix, axis=1)

        for mid, sim_val in zip(valid_mids, max_per_face):
            sv = float(sim_val)
            if sv >= min_score and (mid not in best_by_mid or sv > best_by_mid[mid]):
                best_by_mid[mid] = sv

        ranked = sorted(best_by_mid.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
