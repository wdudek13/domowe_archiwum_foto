import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("MediaQuality")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError as exc:
    logger.debug("pillow_heif not available: %s", exc)


class MediaQualityAnalyzer:
    """
    Evaluates media decoding integrity, perceptual subject sharpness, and multi-reason
    deletion-candidate signals (blur, extreme under/overexposure, uniform blank frames, tiny resolution).
    """

    DEFAULT_BLUR_THRESHOLD: float = 45.0
    MIN_CONTRAST_STD: float = 6.5
    DARK_MEAN_THRESHOLD: float = 15.0
    DARK_P95_THRESHOLD: float = 38.0
    BRIGHT_MEAN_THRESHOLD: float = 243.0
    BRIGHT_P5_THRESHOLD: float = 220.0
    MIN_USEFUL_LONG_EDGE: int = 160

    @classmethod
    def _compute_subject_sharpness(cls, gray: np.ndarray) -> float:
        """
        Computes perceptual subject sharpness on an 8x8 grid after denoising ISO grain.
        Uses the 90th percentile of patch sharpness so smooth sky/bokeh backgrounds
        never cause sharp photos to be misclassified as blurry.
        """
        h, w = gray.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            gray = cv2.resize(gray, (max(64, int(w * scale)), max(64, int(h * scale))), interpolation=cv2.INTER_AREA)
        elif max(h, w) < 256:
            gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_CUBIC)

        # 1. Remove high-ISO sensor noise (3x3 Gaussian sigma=0.7) so grain is not mistaken for sharpness
        denoised = cv2.GaussianBlur(gray, (3, 3), 0.7)

        # 2. Compute Laplacian (fine edges) and Sobel (coarse gradients)
        lap = np.abs(cv2.Laplacian(denoised, cv2.CV_32F, ksize=3))
        gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        # 3. Evaluate 8x8 patches and take the 90th percentile (sharpest subject area)
        gh, gw = denoised.shape[:2]
        step_y, step_x = max(8, gh // 8), max(8, gw // 8)
        patch_scores = []

        for y in range(0, gh - step_y + 1, step_y):
            for x in range(0, gw - step_x + 1, step_x):
                p_lap = lap[y : y + step_y, x : x + step_x]
                p_grad = grad[y : y + step_y, x : x + step_x]
                mean_grad = float(np.mean(p_grad))
                if mean_grad < 3.5:
                    continue
                p95_lap = float(np.percentile(p_lap, 95))
                crispness_ratio = float(np.mean(p_lap)) / (mean_grad + 1.0)
                patch_scores.append(p95_lap * crispness_ratio)

        if not patch_scores:
            return 10.0

        return round(float(np.percentile(patch_scores, 90)) * 2.2, 1)

    @classmethod
    def _decode_grayscale(cls, path: Path) -> Optional[np.ndarray]:
        ext = path.suffix.lower()
        if ext in (".heic", ".heif"):
            with Image.open(path) as pil_img:
                return np.array(pil_img.convert("L"))
        file_bytes = np.fromfile(str(path.resolve()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    @classmethod
    def _detect_luminance_defects(cls, gray: np.ndarray) -> List[str]:
        """Returns defect labels for extreme underexposure, overexposure, or uniform blank frames."""
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        p5_lum, p95_lum = [float(v) for v in np.percentile(gray, [5, 95])]
        defects: List[str] = []
        if mean_lum < cls.DARK_MEAN_THRESHOLD and p95_lum < cls.DARK_P95_THRESHOLD:
            defects.append("niedoswietlone")
        elif mean_lum > cls.BRIGHT_MEAN_THRESHOLD and p5_lum > cls.BRIGHT_P5_THRESHOLD:
            defects.append("przeswietlone")
        if std_lum < cls.MIN_CONTRAST_STD and not defects:
            defects.append("brak_detali")
        return defects

    @classmethod
    def analyze_image_detailed(
        cls,
        path: Path,
        threshold: float = DEFAULT_BLUR_THRESHOLD,
        orig_dims: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Inspects an image for file corruption, blur, and additional deletion-candidate defects
        (extreme darkness/overexposure, uniform blank frames, very low original resolution).
        """
        if not path.exists() or path.stat().st_size == 0:
            return {
                "is_corrupted": True,
                "is_blurry": False,
                "blur_score": None,
                "reasons": ["uszkodzone"],
                "quality_tags": ["do_usuniecia", "do_usuniecia:uszkodzone"],
            }

        try:
            with Image.open(path) as pil_check:
                pil_check.verify()

            gray = cls._decode_grayscale(path)
            if gray is None or gray.size == 0:
                return {
                    "is_corrupted": True,
                    "is_blurry": False,
                    "blur_score": None,
                    "reasons": ["uszkodzone"],
                    "quality_tags": ["do_usuniecia", "do_usuniecia:uszkodzone"],
                }

            score = cls._compute_subject_sharpness(gray)
            reasons: List[str] = []
            if score < threshold:
                reasons.append("rozmyte")
            reasons.extend(cls._detect_luminance_defects(gray))

            if orig_dims and orig_dims[0] and orig_dims[1]:
                if max(int(orig_dims[0]), int(orig_dims[1])) < cls.MIN_USEFUL_LONG_EDGE:
                    reasons.append("niska_rozdzielczosc")

            quality_tags: List[str] = []
            if reasons:
                quality_tags.append("do_usuniecia")
                for r in reasons:
                    quality_tags.append(f"do_usuniecia:{r}")

            return {
                "is_corrupted": False,
                "is_blurry": bool(reasons),
                "blur_score": score,
                "reasons": reasons,
                "quality_tags": quality_tags,
            }
        except Exception as exc:
            logger.debug("Image analysis failed for %s: %s", path, exc)
            return {
                "is_corrupted": True,
                "is_blurry": False,
                "blur_score": None,
                "reasons": ["uszkodzone"],
                "quality_tags": ["do_usuniecia", "do_usuniecia:uszkodzone"],
            }

    @classmethod
    def classify_quality_tags(
        cls,
        path: Path,
        corrupted: bool,
        blurry: bool,
        score: Optional[float],
        orig_dims: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> Tuple[bool, List[str]]:
        """Derives descriptive deletion-candidate tags explaining why a media item failed quality checks."""
        if corrupted:
            return False, ["do_usuniecia", "do_usuniecia:uszkodzone"]

        reasons: List[str] = []
        if blurry and (score is not None and score < cls.DEFAULT_BLUR_THRESHOLD):
            reasons.append("rozmyte")

        try:
            gray = cls._decode_grayscale(path)
            if gray is not None and gray.size > 0:
                reasons.extend(cls._detect_luminance_defects(gray))
        except Exception as exc:
            logger.debug("Detailed luminance inspection skipped for %s: %s", path, exc)

        if orig_dims and orig_dims[0] and orig_dims[1]:
            if max(int(orig_dims[0]), int(orig_dims[1])) < cls.MIN_USEFUL_LONG_EDGE:
                reasons.append("niska_rozdzielczosc")

        if blurry and not reasons:
            reasons.append("rozmyte")

        tags: List[str] = []
        if reasons:
            tags.append("do_usuniecia")
            for r in reasons:
                tags.append(f"do_usuniecia:{r}")
        return bool(reasons), tags

    @classmethod
    def analyze_image(cls, path: Path, threshold: float = DEFAULT_BLUR_THRESHOLD) -> Tuple[bool, bool, Optional[float]]:
        res = cls.analyze_image_detailed(path, threshold=threshold)
        return res["is_corrupted"], res["is_blurry"], res["blur_score"]

    @classmethod
    def analyze_video(cls, path: Path, threshold: float = DEFAULT_BLUR_THRESHOLD) -> Tuple[bool, bool, Optional[float]]:
        if not path.exists() or path.stat().st_size == 0:
            return True, False, None
        cap = cv2.VideoCapture(str(path.resolve()))
        if not cap.isOpened():
            return True, False, None
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                return True, False, None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cls._compute_subject_sharpness(gray)
            return False, bool(score < threshold), score
        finally:
            cap.release()
