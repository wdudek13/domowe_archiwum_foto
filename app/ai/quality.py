import logging
from pathlib import Path
from typing import Tuple, Optional
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
    """Evaluates media decoding integrity and perceptual subject sharpness via Laplacian/Sobel energy."""

    DEFAULT_BLUR_THRESHOLD: float = 45.0

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
    def analyze_image(cls, path: Path, threshold: float = DEFAULT_BLUR_THRESHOLD) -> Tuple[bool, bool, Optional[float]]:
        if not path.exists() or path.stat().st_size == 0:
            return True, False, None

        try:
            ext = path.suffix.lower()
            if ext in (".heic", ".heif"):
                with Image.open(path) as pil_img:
                    gray = np.array(pil_img.convert("L"))
            else:
                file_bytes = np.fromfile(str(path.resolve()), dtype=np.uint8)
                gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if gray is None or gray.size == 0:
                return True, False, None

            score = cls._compute_subject_sharpness(gray)
            return False, bool(score < threshold), score
        except Exception as exc:
            logger.debug("Image analysis failed for %s: %s", path, exc)
            return True, False, None

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
