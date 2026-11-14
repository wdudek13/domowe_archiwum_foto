import gc
import json
import logging
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from app.database import DATA_DIR

logger = logging.getLogger("OnDemandTagger")

PL_CATEGORY_MAPPING = {
    "pies": ["pies", "dog", "retriever", "terrier", "spaniel", "pug", "shepherd", "husky", "beagle", "poodle"],
    "kot": ["kot", "cat", "tabby", "persian", "siamese"],
    "zwierze": ["zwierze", "animal", "pies", "kot", "ptak", "kon", "krowa", "owca"],
    "ptak": ["ptak", "bird", "eagle", "duck", "swan", "parrot", "owl"],
    "samochod": ["samochod", "pojazd", "car", "cab", "convertible", "limousine", "minivan", "jeep", "truck", "pickup"],
    "rower_motocykl": ["rower_motocykl", "rower", "motocykl", "bicycle", "mountain bike", "moped", "motor scooter"],
    "morze_woda": ["morze_woda", "woda", "morze", "plaza", "jezioro", "seashore", "coast", "lakeside", "ocean"],
    "gory_natura": ["gory_natura", "natura", "las", "gory", "drzewa", "alp", "mountain", "valley", "cliff", "forest"],
    "ludzie_twarz": ["osoba / twarz", "portret", "grupa osob", "ludzie", "twarz", "osoba"],
    "jedzenie": ["jedzenie", "pizza", "cheeseburger", "ice cream", "espresso"],
    "architektura": ["architektura", "budynek", "church", "castle", "palace", "bridge", "monastery"],
    "zachod_slonca": ["zachod_slonca", "zachod slonca", "sunset"],
    "noc": ["nocne / ciemne", "noc", "ciemne"],
}


class OnDemandObjectTagger:
    """Loads ONNX MobileNetV2 and OpenCV Haar Cascade strictly on demand and frees memory upon exit."""

    def __init__(self):
        self.model_path = DATA_DIR / "models" / "mobilenetv2.onnx"
        self.labels_path = DATA_DIR / "models" / "imagenet_classes.json"
        self.session = None
        self.labels = None
        self.face_cascade = None

    def open_session(self):
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
            except Exception as exc:
                logger.debug("Could not load ImageNet labels: %s", exc)
                self.labels = None

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as exc:
            logger.debug("Could not load Haar cascade: %s", exc)
            self.face_cascade = None

        if self.model_path.exists() and self.model_path.stat().st_size > 1_000_000:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(
                    str(self.model_path.resolve()),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                logger.warning("Could not init ONNX session: %s", e)
                self.session = None

    def close_session(self):
        if self.session is not None:
            del self.session
            self.session = None
        self.labels = None
        self.face_cascade = None
        gc.collect()

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    def _get_label_by_index(self, idx: int) -> str:
        if self.labels is None:
            return f"class_{idx}"
        if isinstance(self.labels, list):
            return str(self.labels[idx]) if 0 <= idx < len(self.labels) else f"class_{idx}"
        if isinstance(self.labels, dict):
            return str(self.labels.get(str(idx)) or self.labels.get(idx) or f"class_{idx}")
        return f"class_{idx}"

    @staticmethod
    def _classify_imagenet_index(idx: int, prob: float) -> List[Tuple[str, float]]:
        """
        Maps official ImageNet-1000 class IDs (0..999) to accurate Polish categories
        without false substring collisions.
        """
        mapped: List[Tuple[str, float]] = []

        # 1. Dogs (indices 151..268 in ImageNet-1K)
        if 151 <= idx <= 268:
            if prob >= 0.38:
                mapped.append(("pies", round(prob, 3)))
                mapped.append(("zwierze", round(prob, 3)))
            elif prob >= 0.20:
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 2. Domestic Cats (281..285) & Wild Cats (286..293)
        if 281 <= idx <= 285:
            if prob >= 0.35:
                mapped.append(("kot", round(prob, 3)))
            mapped.append(("zwierze", round(prob, 3)))
            return mapped
        if 286 <= idx <= 293:
            mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 3. Birds (7..24, 80..100, 127..146)
        if (7 <= idx <= 24) or (80 <= idx <= 100) or (127 <= idx <= 146):
            if prob >= 0.28:
                mapped.append(("ptak", round(prob, 3)))
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 4. Other Mammals & Farm/Wild Animals (294..397)
        if 294 <= idx <= 397:
            if prob >= 0.22:
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 5. Cars, Trucks & Road Vehicles
        if idx in {436, 468, 511, 609, 627, 654, 656, 717, 734, 751, 757, 817, 864, 867}:
            if prob >= 0.28:
                mapped.append(("samochod", round(prob, 3)))
            return mapped

        # 6. Bicycles & Motorcycles
        if idx in {444, 665, 670, 671, 870}:
            if prob >= 0.28:
                mapped.append(("rower_motocykl", round(prob, 3)))
            return mapped

        # 7. Architecture, Castles, Churches, Bridges
        if idx in {438, 449, 483, 497, 498, 500, 663, 668, 698, 721, 821, 839, 873, 888}:
            if prob >= 0.25:
                mapped.append(("architektura", round(prob, 3)))
            return mapped

        # 8. Food, Dishes & Fruits (924..969)
        if 924 <= idx <= 969:
            if prob >= 0.30:
                mapped.append(("jedzenie", round(prob, 3)))
            return mapped

        # 9. Natural Scenery (970..980)
        if idx in {970, 972, 976, 979, 980}:
            if prob >= 0.20:
                mapped.append(("gory_natura", round(prob, 3)))
            return mapped
        if idx in {973, 974, 975, 977, 978}:
            if prob >= 0.20:
                mapped.append(("morze_woda", round(prob, 3)))
            return mapped

        return mapped

    @staticmethod
    def _preprocess_center_crop_224(img_bgr: np.ndarray) -> np.ndarray:
        """Resizes shortest edge to 256 and takes a 224x224 center crop (no aspect distortion)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 256.0 / min(h, w)
        new_h, new_w = max(224, int(round(h * scale))), max(224, int(round(w * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y0 = (new_h - 224) // 2
        x0 = (new_w - 224) // 2
        crop = resized[y0 : y0 + 224, x0 : x0 + 224].astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (crop - mean) / std
        return np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0)

    def _analyze_opencv_scene_and_faces(self, img_bgr: np.ndarray) -> List[Tuple[str, float]]:
        tags: List[Tuple[str, float]] = []
        h, w = img_bgr.shape[:2]

        if w > h * 1.15:
            tags.append(("poziome (landscape)", 0.99))
        elif h > w * 1.15:
            tags.append(("pionowe (portrait)", 0.99))
        else:
            tags.append(("kwadratowe", 0.95))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(28, 28))
                if len(faces) == 1:
                    tags.append(("osoba / twarz", 0.92))
                    tags.append(("portret", 0.88))
                elif len(faces) > 1:
                    tags.append(("osoba / twarz", 0.95))
                    tags.append(("grupa osob", 0.91))
            except Exception as exc:
                logger.debug("Face detection warning: %s", exc)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]

        if mean_s < 18:
            tags.append(("czarno-biale", 0.94))
        if mean_v < 52:
            tags.append(("nocne / ciemne", 0.89))
        elif mean_v > 208 and mean_s < 40:
            tags.append(("jasne / snieg", 0.85))

        blue_mask = cv2.inRange(hsv, (95, 55, 75), (135, 255, 255))
        blue_ratio = float(np.count_nonzero(blue_mask)) / (w * h)
        if blue_ratio > 0.25:
            top_half_blue = float(np.count_nonzero(blue_mask[: h // 2, :])) / ((w * h) / 2)
            if top_half_blue > 0.35:
                tags.append(("niebo / blekit", round(min(0.98, 0.6 + blue_ratio), 2)))
            else:
                tags.append(("morze_woda", round(min(0.98, 0.6 + blue_ratio), 2)))

        green_mask = cv2.inRange(hsv, (35, 50, 45), (85, 255, 240))
        green_ratio = float(np.count_nonzero(green_mask)) / (w * h)
        if green_ratio > 0.24:
            tags.append(("gory_natura", round(min(0.98, 0.65 + green_ratio), 2)))

        warm_mask = cv2.inRange(hsv, (4, 120, 95), (22, 255, 255))
        warm_ratio = float(np.count_nonzero(warm_mask)) / (w * h)
        if warm_ratio > 0.26:
            tags.append(("zachod_slonca", round(min(0.96, 0.65 + warm_ratio), 2)))

        return tags

    def tag_image(self, path: Path, top_k: int = 3) -> List[Tuple[str, float]]:
        if not path.exists():
            return []
        file_bytes = np.fromfile(str(path.resolve()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return []

        auto_opened = False
        if self.session is None and self.face_cascade is None:
            self.open_session()
            auto_opened = True

        try:
            results: List[Tuple[str, float]] = self._analyze_opencv_scene_and_faces(img)

            if self.session is not None:
                try:
                    tensor = self._preprocess_center_crop_224(img)
                    in_name = self.session.get_inputs()[0].name
                    out_name = self.session.get_outputs()[0].name
                    raw = self.session.run([out_name], {in_name: tensor})[0][0]
                    e_x = np.exp(raw - np.max(raw))
                    probs = e_x / e_x.sum()

                    top_idx = np.argsort(probs)[::-1][:top_k]
                    for i in top_idx:
                        idx = int(i)
                        p = float(probs[idx])
                        for pl_tag, pl_conf in self._classify_imagenet_index(idx, p):
                            results.append((pl_tag, pl_conf))
                        if p >= 0.35:
                            en_lbl = self._get_label_by_index(idx)
                            results.append((en_lbl, round(p, 3)))
                except Exception as e:
                    logger.warning("ONNX fallback: %s", e)

            dedup = {}
            for t_name, conf in results:
                if t_name not in dedup or conf > dedup[t_name]:
                    dedup[t_name] = conf
            return sorted(dedup.items(), key=lambda x: x[1], reverse=True)
        finally:
            if auto_opened:
                self.close_session()
