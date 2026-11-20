import gc
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.ai.quality import decode_image_array
import app.database as db_module

logger = logging.getLogger("OnDemandTagger")

_PL_TRANSLATE = str.maketrans(
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "acelnoszzacelnoszz",
)


def normalize_pl(text: str) -> str:
    """Lowercases text and folds Polish diacritics so e.g. 'róża' matches 'roza'."""
    return (text or "").strip().translate(_PL_TRANSLATE).lower()


def _label_matches_token(norm_lbl: str, token: str) -> bool:
    """Strict whole-word boundary check so 'flower' never matches 'cauliflower' and 'pot' never matches 'potter's wheel'."""
    if not token or not norm_lbl:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", norm_lbl) is not None


PL_CATEGORY_MAPPING: Dict[str, List[str]] = {
    "kwiaty / rosliny": [
        "daisy", "rose", "sunflower", "tulip", "orchid", "lily", "dahlia", "bouquet",
        "hibiscus", "dandelion", "poppy", "lavender", "lotus", "blossom", "petal",
        "yellow lady's slipper", "lady's slipper", "cardoon", "rosehip",
        "flowerpot", "vase", "rapeseed", "acorn", "plant", "flora", "flower",
        "kwiat", "kwiaty", "roslina", "rosliny", "bukiet", "roza", "róża", "tulipan",
        "tulipany", "storczyk", "stokrotka", "slonecznik", "słonecznik", "wazon",
        "doniczka", "ogrod", "ogród", "rzepak", "szklarnia", "greenhouse",
    ],
    "grzyby": [
        "grzyb", "grzyby", "borowik", "pieczarka", "mushroom", "coral fungus",
        "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete",
    ],
    "pies": [
        "pies", "psy", "piesek", "szczeniak", "dog", "retriever", "terrier", "spaniel",
        "pug", "shepherd", "husky", "beagle", "poodle", "labrador", "bulldog", "collie",
        "chihuahua", "dalmatian", "boxer", "rottweiler", "corgi", "samoyed", "malamute",
    ],
    "kot": [
        "kot", "koty", "kotek", "kociak", "cat", "tabby", "persian", "siamese", "egyptian cat",
        "tiger cat", "lynx",
    ],
    "zwierze": [
        "zwierze", "zwierzę", "zwierzeta", "animal", "pies", "kot", "ptak", "kon", "koń",
        "krowa", "owca", "horse", "cow", "sheep", "deer", "bear", "fox", "squirrel", "rabbit",
        "butterfly", "motyl", "ryba", "fish", "bee", "pszczola", "ladybug", "biedronka",
    ],
    "ptak": [
        "ptak", "ptaki", "bird", "eagle", "duck", "swan", "parrot", "owl", "flamingo",
        "pelican", "seagull", "pigeon", "stork", "bocian", "kaczka", "labedz", "sowa", "orzel",
    ],
    "samochod": [
        "samochod", "samochód", "auto", "auta", "pojazd", "car", "cab", "convertible",
        "limousine", "minivan", "jeep", "truck", "pickup", "sports car", "beach wagon",
        "racer", "ambulance", "fire engine", "school bus", "trolleybus", "autobus", "kabriolet",
    ],
    "rower_motocykl": [
        "rower_motocykl", "rower", "rowery", "motocykl", "motor", "skuter", "bicycle",
        "mountain bike", "tandem", "moped", "motor scooter", "unicycle",
    ],
    "morze_woda": [
        "morze_woda", "woda", "morze", "plaza", "plaża", "jezioro", "rzeka", "zatoka",
        "seashore", "coast", "lakeside", "ocean", "beach", "sandbar", "coral reef",
        "speedboat", "sailboat", "catamaran", "yawl", "canoe", "kajak", "statek", "jacht", "molo", "pier",
    ],
    "gory_natura": [
        "gory_natura", "gory", "góry", "natura", "przyroda", "las", "drzewa", "drzewo",
        "alp", "mountain", "valley", "cliff", "forest", "volcano", "promontory", "dolina", "szczyt",
    ],
    "ludzie_twarz": [
        "osoba / twarz", "portret", "grupa osob", "ludzie", "twarz", "osoba", "osoby",
        "czlowiek", "rodzina", "dziecko", "person", "face", "portrait", "people", "bride", "groom",
    ],
    "jedzenie": [
        "jedzenie", "posilek", "obiad", "deser", "ciasto", "tort", "kawa", "owoc", "owoce",
        "pizza", "cheeseburger", "ice cream", "espresso", "coffee mug", "cup", "strawberry",
        "apple", "orange", "lemon", "banana", "pineapple", "chocolate", "pretzel", "bagel",
        "bakery", "trifle", "guacamole", "carbonara", "soup", "lody", "truskawka",
    ],
    "architektura": [
        "architektura", "budynek", "dom", "miasto", "zamek", "palac", "pałac", "kosciol",
        "kościół", "most", "wieza", "wieża", "church", "castle", "palace", "bridge",
        "monastery", "cathedral", "viaduct", "suspension bridge", "fountain", "beacon",
        "bell cote", "dome", "mosque", "stupa", "boathouse", "barn", "lighthouse",
    ],
    "zachod_slonca": ["zachod_slonca", "zachod slonca", "zachód słońca", "sunset", "wschod slonca"],
    "noc": ["nocne / ciemne", "noc", "ciemne", "night", "wieczor", "gwiazdy"],
    "muzyka / instrumenty": [
        "muzyka", "instrument", "gitara", "fortepian", "pianino", "skrzypce", "perkusja",
        "akordeon", "guitar", "acoustic guitar", "electric guitar", "grand piano", "upright",
        "violin", "accordion", "drum", "flute", "saxophone", "trumpet", "harp",
    ],
    "uroczystosc / impreza": [
        "uroczystosc", "uroczystość", "impreza", "urodziny", "slub", "ślub", "wesele",
        "swieta", "święta", "boze narodzenie", "choinka", "balon", "swieca", "świeca",
        "balloon", "candle", "christmas stocking", "goblet", "wine bottle", "champagne", "groom",
    ],
    "pogoda / akcesoria": [
        "parasol", "parasolka", "umbrella", "okulary", "sunglasses", "plecak", "backpack",
        "walizka", "suitcase", "kapelusz", "hat", "sombrero", "cowboy hat", "tent", "namiot",
    ],
    "sport / rekreacja": [
        "sport", "narty", "pilka", "piłka", "tenis", "plywanie", "ski", "soccer ball",
        "basketball", "volleyball", "tennis ball", "baseball", "snowmobile", "bobsled", "surfboard",
    ],
}

IMAGENET_PL_SYNONYMS: Dict[str, str] = {
    # Flowers, plants, garden & fungi
    "daisy": "stokrotka",
    "yellow lady's slipper": "storczyk / obuwik",
    "pot": "doniczka",
    "flowerpot": "doniczka",
    "vase": "wazon",
    "rapeseed": "rzepak",
    "cardoon": "karczoch / oset",
    "hip": "dzika róża",
    "rosehip": "dzika róża",
    "corn": "kukurydza",
    "acorn": "żołądź",
    "buckeye": "kasztan",
    "mushroom": "grzyb",
    "bolete": "borowik / grzyb",
    "agaric": "grzyb / pieczarka",
    "coral fungus": "grzyb koralowy",
    "gyromitra": "piestrzenica / grzyb",
    "stinkhorn": "grzyb",
    "earthstar": "gwiazdosz / grzyb",
    "hen-of-the-woods": "żagwica / grzyb",
    "ear": "kłos",
    "greenhouse": "szklarnia",
    # Architecture & travel
    "castle": "zamek",
    "palace": "pałac",
    "church": "kościół",
    "monastery": "klasztor",
    "suspension bridge": "most wiszący",
    "steel arch bridge": "most łukowy",
    "viaduct": "wiadukt",
    "beacon": "latarnia morska",
    "fountain": "fontanna",
    "bell cote": "dzwonnica",
    "dome": "kopuła",
    "boathouse": "przystań",
    "pier": "molo",
    "barn": "stodoła",
    "lakeside": "jezioro",
    "seashore": "brzeg morza",
    "alp": "alpy / góry",
    "valley": "dolina",
    "cliff": "klif",
    "volcano": "wulkan",
    "sandbar": "mierzeja / plaża",
    "coral reef": "rafa koralowa",
    "promontory": "przylądek",
    "park bench": "ławka w parku",
    # Vehicles & transport
    "convertible": "kabriolet",
    "sports car": "samochód sportowy",
    "cab": "taksówka",
    "limousine": "limuzyna",
    "minivan": "minivan",
    "jeep": "terenówka / jeep",
    "pickup": "pickup",
    "beach wagon": "kombi",
    "mountain bike": "rower górski",
    "bicycle-built-for-two": "rower tandem",
    "moped": "motorower",
    "motor scooter": "skuter",
    "speedboat": "motorówka",
    "catamaran": "katamaran",
    "yawl": "jacht",
    "canoe": "kajak",
    "airliner": "samolot",
    "steam locomotive": "pociąg / lokomotywa",
    # Music, events & everyday objects
    "umbrella": "parasol",
    "acoustic guitar": "gitara akustyczna",
    "electric guitar": "gitara elektryczna",
    "grand piano": "fortepian",
    "upright": "pianino",
    "violin": "skrzypce",
    "accordion": "akordeon",
    "balloon": "balon",
    "candle": "świeca",
    "christmas stocking": "boże narodzenie",
    "jack-o'-lantern": "dynia",
    "teddy": "miś pluszowy",
    "coffee mug": "kubek",
    "teapot": "czajnik",
    "wine bottle": "butelka wina",
    "goblet": "kieliszek",
    "sunglasses": "okulary przeciwsłoneczne",
    "backpack": "plecak",
    "tent": "namiot",
    "ski": "narty",
    "soccer ball": "piłka nożna",
    "laptop": "laptop",
    "bookcase": "książki / regał",
    "dining table": "stół",
    "studio couch": "kanapa",
    # Food & fruit
    "pizza": "pizza",
    "cheeseburger": "burger",
    "ice cream": "lody",
    "espresso": "kawa / espresso",
    "strawberry": "truskawka",
    "orange": "pomarańcza",
    "lemon": "cytryna",
    "banana": "banan",
    "pineapple": "ananas",
    "pomegranate": "granat",
    "fig": "figa",
    "granny smith": "jabłko",
    "pretzel": "precel",
    "chocolate sauce": "czekolada",
    "trifle": "ciasto / deser",
    "bakery": "piekarnia / wypieki",
    # Animals
    "golden retriever": "golden retriever",
    "labrador retriever": "labrador",
    "german shepherd": "owczarek niemiecki",
    "tabby": "kot pręgowany",
    "persian cat": "kot perski",
    "siamese cat": "kot syjamski",
    "white stork": "bocian",
    "mallard": "kaczka",
    "bald eagle": "orzeł",
    "flamingo": "flaming",
    "goldfish": "złota rybka",
    "monarch": "motyl",
    "admiral": "motyl",
    "sulphur butterfly": "motyl",
    "ladybug": "biedronka",
    "bee": "pszczoła",
    "brown bear": "niedźwiedź",
    "red fox": "lis",
    "fox squirrel": "wiewiórka",
    "hare": "zając",
}

_FLOWER_PLANT_INDICES = {
    580, 738, 883, 946, 984, 985, 986, 989, 998,
}
_PURE_FLOWER_INDICES = {738, 883, 946, 984, 985, 986, 989}
_MUSHROOM_INDICES = {947, 991, 992, 993, 994, 995, 996, 997}
_MUSIC_INDICES = {401, 402, 420, 486, 513, 541, 546, 558, 566, 579, 594, 642, 683, 684, 776, 875, 881, 889}
_EVENT_INDICES = {417, 470, 496, 572, 607, 907, 982}
_WEATHER_ACC_INDICES = {414, 515, 808, 836, 837, 879}


class OnDemandObjectTagger:
    """
    Loads ONNX MobileNetV2 and OpenCV detectors strictly on demand (0 MB RAM at rest)
    and releases all sessions immediately upon close_session() / context exit.
    Supports two-level tagging (general category + full ImageNet-1000 detailed tags)
    as well as on-demand keyword scanning on local thumbnails.
    """

    def __init__(self):
        self.model_path = db_module.DATA_DIR / "models" / "mobilenetv2.onnx"
        self.labels_path = db_module.DATA_DIR / "models" / "imagenet_classes.json"
        self._session = None
        self.labels = None
        self.face_cascade = None

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, val):
        self._session = val

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
                self._session = ort.InferenceSession(
                    str(self.model_path.resolve()),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                logger.warning("Could not init ONNX session: %s", e)
                self._session = None

    def close_session(self):
        if self._session is not None:
            del self._session
            self._session = None
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
    def _format_detailed_imagenet_tag(en_label: str) -> str:
        """
        Formats a raw ImageNet-1000 class label into a clean searchable Polish/English detailed tag
        (e.g. 'daisy' -> 'stokrotka (daisy)', 'pot, flowerpot' -> 'doniczka (pot)').
        Always fits safely within MediaTag.tag (max 64 chars).
        """
        raw = (en_label or "").strip()
        if not raw or raw.startswith("class_"):
            return ""
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        primary = parts[0].lower() if parts else raw.lower()

        pl_match = IMAGENET_PL_SYNONYMS.get(primary)
        if not pl_match:
            for alt in parts[1:]:
                pl_match = IMAGENET_PL_SYNONYMS.get(alt.lower())
                if pl_match:
                    primary = alt.lower()
                    break

        if pl_match:
            if normalize_pl(pl_match) == normalize_pl(primary):
                out = pl_match
            else:
                out = f"{pl_match} ({primary})"
        else:
            out = primary
        return out[:60].strip()

    @staticmethod
    def _classify_imagenet_index(idx: int, prob: float) -> List[Tuple[str, float]]:
        """
        Maps official ImageNet-1000 class IDs (0..999) to accurate Polish categories
        without false substring collisions.
        """
        mapped: List[Tuple[str, float]] = []

        # 0. Flowers, Plants & Garden
        if idx in _FLOWER_PLANT_INDICES:
            if prob >= 0.15:
                mapped.append(("kwiaty / rosliny", round(prob, 3)))
                if idx in _PURE_FLOWER_INDICES:
                    mapped.append(("kwiaty", round(prob, 3)))
            return mapped

        if idx in _MUSHROOM_INDICES:
            if prob >= 0.18:
                mapped.append(("grzyby", round(prob, 3)))
            return mapped

        # 1. Dogs (indices 151..268 in ImageNet-1K)
        if 151 <= idx <= 268:
            if prob >= 0.30:
                mapped.append(("pies", round(prob, 3)))
                mapped.append(("zwierze", round(prob, 3)))
            elif prob >= 0.18:
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 2. Domestic Cats (281..285) & Wild Cats (286..293)
        if 281 <= idx <= 285:
            if prob >= 0.30:
                mapped.append(("kot", round(prob, 3)))
            mapped.append(("zwierze", round(prob, 3)))
            return mapped
        if 286 <= idx <= 293:
            if prob >= 0.20:
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 3. Birds (7..24, 80..100, 127..146)
        if (7 <= idx <= 24) or (80 <= idx <= 100) or (127 <= idx <= 146):
            if prob >= 0.25:
                mapped.append(("ptak", round(prob, 3)))
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 4. Other Mammals, Insects & Farm/Wild Animals (294..397)
        if 294 <= idx <= 397:
            if prob >= 0.22:
                mapped.append(("zwierze", round(prob, 3)))
            return mapped

        # 5. Cars, Trucks & Road Vehicles
        if idx in {436, 468, 511, 609, 627, 654, 656, 717, 734, 751, 757, 817, 864, 867}:
            if prob >= 0.25:
                mapped.append(("samochod", round(prob, 3)))
            return mapped

        # 6. Bicycles & Motorcycles
        if idx in {444, 665, 670, 671, 870}:
            if prob >= 0.25:
                mapped.append(("rower_motocykl", round(prob, 3)))
            return mapped

        # 7. Architecture, Castles, Churches, Bridges
        if idx in {438, 449, 483, 497, 498, 500, 562, 663, 668, 698, 721, 821, 839, 873, 888}:
            if prob >= 0.22:
                mapped.append(("architektura", round(prob, 3)))
            return mapped

        # 8. Food, Dishes & Fruits (924..969 except cardoon/mushroom handled above)
        if 924 <= idx <= 969:
            if prob >= 0.25:
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

        # 10. Music instruments, events & accessories
        if idx in _MUSIC_INDICES and prob >= 0.22:
            mapped.append(("muzyka / instrumenty", round(prob, 3)))
            return mapped
        if idx in _EVENT_INDICES and prob >= 0.22:
            mapped.append(("uroczystosc / impreza", round(prob, 3)))
            return mapped
        if idx in _WEATHER_ACC_INDICES and prob >= 0.22:
            mapped.append(("pogoda / akcesoria", round(prob, 3)))
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

    def _run_onnx_probs(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._session is None:
            return None
        tensor = self._preprocess_center_crop_224(img_bgr)
        in_name = self._session.get_inputs()[0].name
        out_name = self._session.get_outputs()[0].name
        raw = self._session.run([out_name], {in_name: tensor})[0][0]
        e_x = np.exp(raw - np.max(raw))
        return e_x / e_x.sum()

    def _analyze_opencv_scene_and_faces(self, img_bgr: np.ndarray) -> List[Tuple[str, float]]:
        tags: List[Tuple[str, float]] = []
        h, w = img_bgr.shape[:2]
        total_px = float(max(1, w * h))

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
        blue_ratio = float(np.count_nonzero(blue_mask)) / total_px
        if blue_ratio > 0.25:
            top_half_blue = float(np.count_nonzero(blue_mask[: h // 2, :])) / (total_px / 2.0)
            if top_half_blue > 0.35:
                tags.append(("niebo / blekit", round(min(0.98, 0.6 + blue_ratio), 2)))
            else:
                tags.append(("morze_woda", round(min(0.98, 0.6 + blue_ratio), 2)))

        green_mask = cv2.inRange(hsv, (35, 50, 45), (85, 255, 240))
        green_ratio = float(np.count_nonzero(green_mask)) / total_px
        if green_ratio > 0.24:
            tags.append(("gory_natura", round(min(0.98, 0.65 + green_ratio), 2)))

        # Task 2: Tightened HSV Flower heuristic — green foliage (>=12%) + compact vivid petal cluster(s)
        # Rejects extreme panoramas (PANO_*) and scattered drone roofs / sunlit dry fields!
        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        foliage_mask = cv2.inRange(hsv, (35, 50, 40), (85, 255, 240))
        foliage_ratio = float(np.count_nonzero(foliage_mask)) / total_px
        if foliage_ratio >= 0.12 and aspect_ratio <= 2.2:
            petal_red1 = cv2.inRange(hsv, (0, 145, 105), (10, 255, 255))
            petal_red2 = cv2.inRange(hsv, (158, 145, 105), (179, 255, 255))
            petal_yellow = cv2.inRange(hsv, (18, 175, 130), (32, 255, 255))
            petal_purple = cv2.inRange(hsv, (128, 140, 100), (154, 255, 255))
            petal_mask = cv2.bitwise_or(
                cv2.bitwise_or(petal_red1, petal_red2),
                cv2.bitwise_or(petal_yellow, petal_purple),
            )
            petal_ratio = float(np.count_nonzero(petal_mask)) / total_px
            if 0.05 <= petal_ratio <= 0.55:
                k_fl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                p_clean = cv2.morphologyEx(petal_mask, cv2.MORPH_OPEN, k_fl)
                p_contours, _ = cv2.findContours(p_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                has_blossom_cluster = False
                for cnt in p_contours:
                    c_area = float(cv2.contourArea(cnt))
                    c_ratio = c_area / total_px
                    if not (0.018 <= c_ratio <= 0.45):
                        continue
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    b_asp = float(bw) / float(max(1, bh))
                    if not (0.35 <= b_asp <= 2.8):
                        continue
                    hull_a = float(cv2.contourArea(cv2.convexHull(cnt)))
                    solidity = c_area / max(1.0, hull_a)
                    # Reject flat sky/horizon bands stuck along top edge
                    if solidity >= 0.48 and not (by <= 2 and bw >= int(w * 0.7)):
                        has_blossom_cluster = True
                        break
                if has_blossom_cluster:
                    fl_conf = round(min(0.92, 0.62 + min(petal_ratio, 0.25) + min(foliage_ratio, 0.15) * 0.4), 2)
                    tags.append(("kwiaty / rosliny", fl_conf))
                    tags.append(("kwiaty", fl_conf))

        warm_mask = cv2.inRange(hsv, (4, 120, 95), (22, 255, 255))
        warm_ratio = float(np.count_nonzero(warm_mask)) / total_px
        if warm_ratio > 0.26:
            tags.append(("zachod_slonca", round(min(0.96, 0.65 + warm_ratio), 2)))

        return tags

    def tag_image_with_sources(self, path: Path, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Task 2: Two-level tagging returning `(tag_name, confidence, source_model)`:
        - General categories & OpenCV scene tags -> `source_model="opencv+onnx"`
        - Specific ImageNet-1000 class tags (for ANY class with confidence >= 0.15) -> `source_model="onnx_detail"`
        Zero information loss across all 1000 ImageNet classes!
        """
        img = decode_image_array(path, grayscale=False)
        if img is None:
            return []

        auto_opened = False
        if self._session is None and self.face_cascade is None:
            self.open_session()
            auto_opened = True

        try:
            results: List[Tuple[str, float, str]] = [
                (t_name, conf, "opencv+onnx")
                for t_name, conf in self._analyze_opencv_scene_and_faces(img)
            ]

            if self._session is not None:
                try:
                    probs = self._run_onnx_probs(img)
                    if probs is not None:
                        top_idx = np.argsort(probs)[::-1][:top_k]
                        for i in top_idx:
                            idx = int(i)
                            p = float(probs[idx])
                            for pl_tag, pl_conf in self._classify_imagenet_index(idx, p):
                                results.append((pl_tag, pl_conf, "opencv+onnx"))
                            if p >= 0.15:
                                en_lbl = self._get_label_by_index(idx)
                                detail_tag = self._format_detailed_imagenet_tag(en_lbl)
                                if detail_tag:
                                    results.append((detail_tag, round(p, 3), "onnx_detail"))
                                # Also match category via PL_CATEGORY_MAPPING using strict word boundaries
                                norm_en = normalize_pl(en_lbl)
                                for pl_cat, words in PL_CATEGORY_MAPPING.items():
                                    if any(
                                        len(normalize_pl(w)) >= 4 and _label_matches_token(norm_en, normalize_pl(w))
                                        for w in words
                                    ):
                                        if p >= 0.20:
                                            results.append((pl_cat, round(p, 3), "opencv+onnx"))
                except Exception as e:
                    logger.warning("ONNX fallback: %s", e)

            dedup: Dict[str, Tuple[float, str]] = {}
            for t_name, conf, src in results:
                if not t_name:
                    continue
                if t_name not in dedup:
                    dedup[t_name] = (conf, src)
                else:
                    old_conf, old_src = dedup[t_name]
                    # Prefer general source tag when same label occurs in both
                    best_src = "opencv+onnx" if ("opencv+onnx" in (old_src, src)) else src
                    dedup[t_name] = (max(old_conf, conf), best_src)

            return sorted(
                [(k, v[0], v[1]) for k, v in dedup.items()],
                key=lambda x: x[1],
                reverse=True,
            )
        finally:
            if auto_opened:
                self.close_session()

    def tag_image(self, path: Path, top_k: int = 5) -> List[Tuple[str, float]]:
        """Backwards-compatible wrapper returning `(tag_name, confidence)` pairs."""
        return [(t_name, conf) for t_name, conf, _ in self.tag_image_with_sources(path, top_k=top_k)]

    def resolve_keyword_to_imagenet_indices(self, keyword: str) -> Tuple[Set[int], Set[str], Set[str]]:
        """
        Translates a Polish or English user search keyword into:
        - matching ImageNet-1000 class indices (0..999),
        - matching Polish general category keys from PL_CATEGORY_MAPPING,
        - normalized English/Polish search tokens.
        """
        q_norm = normalize_pl(keyword)
        if not q_norm:
            return set(), set(), set()

        matched_cats: Set[str] = set()
        tokens: Set[str] = {q_norm}
        # Add stemmed prefix for common Polish plurals (e.g. kwiaty -> kwiat, koty -> kot, psy -> pies)
        if len(q_norm) >= 4 and q_norm.endswith(("y", "i", "e", "a")):
            tokens.add(q_norm[:-1])

        for pl_cat, words in PL_CATEGORY_MAPPING.items():
            norm_cat = normalize_pl(pl_cat)
            norm_words = [normalize_pl(w) for w in words]
            if (
                q_norm in norm_cat
                or any(q_norm in nw or (len(nw) >= 3 and nw in q_norm) for nw in norm_words)
            ):
                matched_cats.add(pl_cat)
                for nw in norm_words:
                    if len(nw) >= 3:
                        tokens.add(nw)

        for en_k, pl_v in IMAGENET_PL_SYNONYMS.items():
            n_en = normalize_pl(en_k)
            n_pl = normalize_pl(pl_v)
            if q_norm in n_pl or q_norm in n_en or (len(n_pl) >= 3 and n_pl in q_norm):
                tokens.add(n_en)

        target_indices: Set[int] = set()
        if "kwiaty / rosliny" in matched_cats:
            target_indices.update(_PURE_FLOWER_INDICES)
        if "grzyby" in matched_cats:
            target_indices.update(_MUSHROOM_INDICES)
        if "pies" in matched_cats:
            target_indices.update(range(151, 269))
        if "kot" in matched_cats:
            target_indices.update(range(281, 286))
        if "ptak" in matched_cats:
            target_indices.update(range(7, 25))
            target_indices.update(range(80, 101))
            target_indices.update(range(127, 147))
        if "samochod" in matched_cats:
            target_indices.update({436, 468, 511, 609, 627, 654, 656, 717, 734, 751, 757, 817, 864, 867})
        if "rower_motocykl" in matched_cats:
            target_indices.update({444, 665, 670, 671, 870})
        if "architektura" in matched_cats:
            target_indices.update({438, 449, 483, 497, 498, 500, 562, 663, 668, 698, 721, 821, 839, 873, 888})
        if "jedzenie" in matched_cats:
            target_indices.update(range(924, 970))
        if "gory_natura" in matched_cats:
            target_indices.update({970, 972, 976, 979, 980})
        if "morze_woda" in matched_cats:
            target_indices.update({973, 974, 975, 977, 978})
        if "muzyka / instrumenty" in matched_cats:
            target_indices.update(_MUSIC_INDICES)
        if "uroczystosc / impreza" in matched_cats:
            target_indices.update(_EVENT_INDICES)
        if "pogoda / akcesoria" in matched_cats:
            target_indices.update(_WEATHER_ACC_INDICES)

        # Match directly against all 1000 loaded ImageNet labels using strict word boundaries
        # (prevents e.g. 'flower' -> 'cauliflower', 'pot' -> 'potter's wheel', 'ear' -> 'bear', 'hip' -> 'ship')
        if self.labels is not None:
            items_iter = (
                enumerate(self.labels)
                if isinstance(self.labels, list)
                else ((int(k), v) for k, v in self.labels.items())
            )
            for idx, lbl in items_iter:
                norm_lbl = normalize_pl(str(lbl))
                if any(len(tok) >= 3 and _label_matches_token(norm_lbl, tok) for tok in tokens):
                    target_indices.add(int(idx))

        return target_indices, matched_cats, tokens

    def scan_thumbnails_for_keyword(
        self,
        db: Session,
        keyword: str,
        limit_scan: int = 3000,
        thumbnails_dir: Optional[Path] = None,
        confidence_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Task 1: On-Demand ONNX + OpenCV HSV scanner operating strictly on local WebP thumbnails
        (`data/thumbnails/*.webp`), prioritizing untagged items, using Top-10 predictions with lowered
        threshold (`>= 0.06`) for the requested keyword, saving matches into MediaTag, and ALWAYS
        closing the ONNX session + calling gc.collect() in `finally:`.
        """
        from app.models import MediaItem, MediaTag

        clean_kw = (keyword or "").strip()
        if not clean_kw:
            return {"keyword": "", "scanned_count": 0, "matched_count": 0, "matched_ids": [], "session_released": True}

        eff_thumbs = Path(thumbnails_dir) if thumbnails_dir is not None else db_module.THUMBNAILS_DIR
        self.open_session()
        scanned_count = 0
        matched_ids: List[str] = []
        tags_added = 0

        try:
            target_indices, matched_cats, search_tokens = self.resolve_keyword_to_imagenet_indices(clean_kw)

            # Prioritize items without AI tags first, then remaining active non-backup items
            has_ai_tag = exists(
                select(MediaTag.id).where(
                    MediaTag.media_id == MediaItem.id,
                    MediaTag.source_model.in_(["opencv+onnx", "onnx_detail", "onnx_on_demand"]),
                )
            )
            candidates = list(
                db.execute(
                    select(MediaItem.id, MediaItem.thumbnail_relative_path)
                    .where(
                        MediaItem.is_trashed == False,
                        MediaItem.is_backup_copy == False,
                        MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
                    )
                    .order_by(has_ai_tag.asc(), MediaItem.effective_date.desc())
                    .limit(max(1, int(limit_scan)))
                ).all()
            )

            existing_tags_by_mid: Dict[str, Set[str]] = {}
            cand_ids = [str(r[0]) for r in candidates]
            for ch_start in range(0, len(cand_ids), 900):
                ch = cand_ids[ch_start : ch_start + 900]
                for mid_r, tag_r in db.execute(
                    select(MediaTag.media_id, MediaTag.tag).where(MediaTag.media_id.in_(ch))
                ).all():
                    existing_tags_by_mid.setdefault(str(mid_r), set()).add(normalize_pl(str(tag_r)))

            norm_kw = normalize_pl(clean_kw)

            for mid, thumb_rel in candidates:
                if not thumb_rel:
                    continue
                thumb_p = eff_thumbs / thumb_rel
                if not thumb_p.exists() or not thumb_p.is_file() or thumb_p.stat().st_size == 0:
                    continue

                img = decode_image_array(thumb_p, grayscale=False)
                if img is None:
                    continue

                scanned_count += 1
                new_tags_for_item: List[Tuple[str, float, str]] = []

                # 1. OpenCV scene / HSV rules (including flower HSV heuristic!)
                cv_tags = self._analyze_opencv_scene_and_faces(img)
                for cv_t, cv_c in cv_tags:
                    n_cv = normalize_pl(cv_t)
                    if (
                        cv_t in matched_cats
                        or norm_kw in n_cv
                        or any(len(tok) >= 3 and tok in n_cv for tok in search_tokens)
                    ):
                        new_tags_for_item.append((cv_t, cv_c, "opencv+onnx"))

                # 2. ONNX MobileNetV2 Top-10 predictions with lowered threshold for target keyword
                if self._session is not None and target_indices:
                    try:
                        probs = self._run_onnx_probs(img)
                        if probs is not None:
                            top10 = np.argsort(probs)[::-1][:10]
                            for idx_raw in top10:
                                idx = int(idx_raw)
                                p = float(probs[idx])
                                if p < confidence_threshold:
                                    continue
                                if idx in target_indices:
                                    en_lbl = self._get_label_by_index(idx)
                                    detail_tag = self._format_detailed_imagenet_tag(en_lbl)
                                    if detail_tag:
                                        new_tags_for_item.append((detail_tag, round(p, 3), "onnx_detail"))
                                    for cat_k in matched_cats:
                                        new_tags_for_item.append((cat_k, round(p, 3), "opencv+onnx"))
                                    for pl_tag, pl_conf in self._classify_imagenet_index(idx, p):
                                        new_tags_for_item.append((pl_tag, pl_conf, "opencv+onnx"))
                    except Exception as exc:
                        logger.debug("On-demand ONNX scan warning for %s: %s", mid, exc)

                if new_tags_for_item:
                    matched_ids.append(str(mid))
                    have_tags = existing_tags_by_mid.setdefault(str(mid), set())
                    dedup_new: Dict[str, Tuple[float, str]] = {}
                    for tn, tc, ts in new_tags_for_item:
                        if tn not in dedup_new or tc > dedup_new[tn][0]:
                            dedup_new[tn] = (tc, ts)
                    for tn, (tc, ts) in dedup_new.items():
                        ntn = normalize_pl(tn)
                        if ntn not in have_tags:
                            db.add(MediaTag(media_id=str(mid), tag=tn[:64], confidence=float(tc), source_model=ts))
                            have_tags.add(ntn)
                            tags_added += 1
                    if tags_added > 0 and len(matched_ids) % 25 == 0:
                        db.commit()

            if tags_added > 0:
                db.commit()

            result = {
                "keyword": clean_kw,
                "scanned_count": scanned_count,
                "matched_count": len(matched_ids),
                "tags_added": tags_added,
                "matched_ids": matched_ids,
                "session_released": False,
            }
        finally:
            self.close_session()

        result["session_released"] = self._session is None
        return result
