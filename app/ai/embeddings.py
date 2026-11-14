from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from app.ai.quality import decode_image_array


class OnDemandClipEmbedder:
    """
    Computes a properly block-normalized 512-D visual feature vector:
    - Block 1 (192-D, weight 0.50): 8x8 Spatial Grid in perceptual CIE LAB color space
    - Block 2 (256-D, weight 0.35): Multi-region HSV Color Palette Histogram
    - Block 3 (64-D,  weight 0.15): Normalized Edge Orientation & Luminance Contrast Spectrum
    """
    DIMENSIONS = 512
    MODEL_NAME = "hybrid_lab_hsv_512"

    @staticmethod
    def _unit_norm(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32).flatten()
        n = float(np.linalg.norm(arr))
        return (arr / n) if n > 1e-7 else arr

    @classmethod
    def generate_image_embedding(cls, path: Path) -> bytes:
        img = decode_image_array(path, grayscale=False)
        if img is None:
            raise ValueError("Decode failed")

        # 1. Block 1: 8x8 Spatial Layout in Perceptual CIE LAB space (8 * 8 * 3 = 192 dims)
        small_lab = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small_lab, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Center around mean so cosine similarity measures structural color difference
        lab[:, :, 0] = (lab[:, :, 0] - 128.0) / 128.0
        lab[:, :, 1] = (lab[:, :, 1] - 128.0) / 128.0
        lab[:, :, 2] = (lab[:, :, 2] - 128.0) / 128.0
        spatial_lab_vec = cls._unit_norm(lab) * np.sqrt(0.50)

        # 2. Block 2: Upper & Lower Half HSV Color Histograms (2 * 8*4*4 = 256 dims)
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_top = cv2.calcHist([hsv[:64, :]], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist_bot = cv2.calcHist([hsv[64:, :]], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        # Apply square-root (Hellinger/Bhattacharyya kernel) to prevent single background color from dominating
        hsv_raw = np.sqrt(np.maximum(0.0, np.concatenate([hist_top, hist_bot])))
        hsv_vec = cls._unit_norm(hsv_raw) * np.sqrt(0.35)

        # 3. Block 3: Normalized Luminance & Edge Orientation Spectrum (32 + 32 = 64 dims)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        lum_hist = np.sqrt(cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten())
        lum_vec = cls._unit_norm(lum_hist)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        edge_hist, _ = np.histogram(ang, bins=32, range=(0, 360), weights=mag)
        edge_vec = cls._unit_norm(np.sqrt(np.maximum(0.0, edge_hist)))

        texture_vec = cls._unit_norm(np.concatenate([lum_vec, edge_vec])) * np.sqrt(0.15)

        # Final 512-D unit-normalized vector (192 + 256 + 64 = 512)
        final_vec = np.concatenate([spatial_lab_vec, hsv_vec, texture_vec]).astype(np.float32)
        final_vec = cls._unit_norm(final_vec)
        return final_vec.tobytes()

    @classmethod
    def rank_similar(cls, query_vec: np.ndarray, candidates: List[Tuple[str, bytes]], top_n: int = 50) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        q_dim = len(query_vec)
        valid_ids = []
        rows = []
        for cid, blob in candidates:
            if not blob or len(blob) != q_dim * 4:
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            valid_ids.append(cid)
            rows.append(v)

        if not rows:
            return []
        matrix = np.vstack(rows)
        scores = np.dot(matrix, query_vec)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(valid_ids[i], float(scores[i])) for i in top_idx]