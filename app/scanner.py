import ctypes
from ctypes import wintypes
import datetime
import filecmp
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import uuid
import cv2
from PIL import ExifTags, Image, ImageOps
from app.database import THUMBNAILS_DIR
from app.models import MediaItem, WindowsVolume, MediaEmbedding
from app.ai.embeddings import OnDemandClipEmbedder

logger = logging.getLogger("Scanner")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError as exc:
    logger.debug("pillow_heif not available: %s", exc)

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".gif", ".avif"}
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".mts", ".m2ts", ".3gp", ".mpg", ".mpeg"}

DATE_PATTERNS = [
    re.compile(r"(?P<year>19\d{2}|20\d{2})[-_]?(?P<month>0[1-9]|1[0-2])[-_]?(?P<day>0[1-9]|[12]\d|3[01])[-_\sT](?P<hour>[01]\d|2[0-3])[-_]?(?P<min>[0-5]\d)[-_]?(?P<sec>[0-5]\d)"),
    re.compile(r"(?P<year>19\d{2}|20\d{2})[-_](?P<month>0[1-9]|1[0-2])[-_](?P<day>0[1-9]|[12]\d|3[01])"),
    re.compile(r"IMG[-_](?P<year>19\d{2}|20\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"),
    re.compile(r"VID[-_](?P<year>19\d{2}|20\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"),
]


class WindowsVolumeInspector:
    """Inspects Windows hardware volume serial numbers (VSN) and labels via Win32 kernel32 API."""

    @staticmethod
    def get_volume_info(path: Path) -> Tuple[str, str, str]:
        abs_path = path.resolve()
        drive = os.path.splitdrive(str(abs_path))[0]
        drive_root = (drive.rstrip("\\/") + "\\") if drive else (str(abs_path.anchor) if abs_path.anchor else "C:\\")

        try:
            kernel32 = ctypes.windll.kernel32
            volume_name = ctypes.create_unicode_buffer(1024)
            fs_name = ctypes.create_unicode_buffer(1024)
            serial_number = wintypes.DWORD()
            max_component_len = wintypes.DWORD()
            flags = wintypes.DWORD()

            res = kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(drive_root),
                volume_name,
                ctypes.sizeof(volume_name),
                ctypes.byref(serial_number),
                ctypes.byref(max_component_len),
                ctypes.byref(flags),
                fs_name,
                ctypes.sizeof(fs_name),
            )
            if res:
                vsn_int = serial_number.value
                vsn_hex = f"{(vsn_int >> 16) & 0xFFFF:04X}-{vsn_int & 0xFFFF:04X}"
                label = volume_name.value if volume_name.value else "UNNAMED_DRIVE"
                return vsn_hex, label, drive_root
        except Exception as exc:
            logger.debug("GetVolumeInformationW fallback for %s: %s", drive_root, exc)

        return f"VOL_{drive.rstrip(':') or 'LOCAL'}", "LOCAL_STORAGE", drive_root

    @classmethod
    def get_connected_vsn_map(cls) -> Dict[str, Tuple[str, str]]:
        """Returns mapping of {vsn_hex: (drive_root, volume_label)} for all currently mounted Windows drives."""
        mounted: Dict[str, Tuple[str, str]] = {}
        try:
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for i in range(26):
                if bitmask & (1 << i):
                    drive_letter = f"{chr(65 + i)}:\\"
                    if Path(drive_letter).exists():
                        vsn, label, root = cls.get_volume_info(Path(drive_letter))
                        mounted[vsn] = (root, label)
        except Exception as exc:
            logger.debug("Logical drive scan warning: %s", exc)
        return mounted


class FastHasher:
    """Computes full SHA-256 for images, fast partial 3x64KB SHA-256 for videos, and multi-layer 100% verification."""

    CHUNK_SIZE = 64 * 1024

    @classmethod
    def hash_image_full(cls, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(cls.CHUNK_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def hash_video_partial(cls, file_path: Path, file_size: int) -> str:
        if file_size <= cls.CHUNK_SIZE * 3:
            return cls.hash_image_full(file_path)
        hasher = hashlib.sha256()
        hasher.update(file_size.to_bytes(8, "big"))
        with open(file_path, "rb") as f:
            hasher.update(f.read(cls.CHUNK_SIZE))
            f.seek(file_size // 2 - cls.CHUNK_SIZE // 2)
            hasher.update(f.read(cls.CHUNK_SIZE))
            f.seek(file_size - cls.CHUNK_SIZE)
            hasher.update(f.read(cls.CHUNK_SIZE))
        return hasher.hexdigest()

    @classmethod
    def compute_multi_hashes(cls, file_path: Path) -> Tuple[str, str, str]:
        """Computes three independent full-stream cryptographic digests (SHA-256, BLAKE2b, MD5) in a single pass."""
        h_sha256 = hashlib.sha256()
        h_blake2b = hashlib.blake2b()
        h_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(256 * 1024):
                h_sha256.update(chunk)
                h_blake2b.update(chunk)
                h_md5.update(chunk)
        return h_sha256.hexdigest(), h_blake2b.hexdigest(), h_md5.hexdigest()

    @classmethod
    def verify_files_identical_100_percent(cls, path_a: Path, path_b: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Multi-layer independent verification guaranteeing 100% certainty that path_a and path_b
        are distinct physical files with identical byte-for-byte content and valid structure.
        """
        try:
            if not path_a.exists() or not path_b.exists():
                return False, "Jeden z plików fizycznie nie istnieje na dysku.", None
            if not path_a.is_file() or not path_b.is_file():
                return False, "Ścieżka nie wskazuje na prawidłowy plik regularny.", None

            res_a = path_a.resolve()
            res_b = path_b.resolve()
            if res_a == res_b:
                return False, "Oba rekordy wskazują na ten sam fizyczny plik (ochrona przed usunięciem jedynej kopii).", None

            try:
                if os.path.samefile(res_a, res_b):
                    return False, "Pliki mają ten sam deskryptor sprzętowy inode/hardlink.", None
            except OSError as exc:
                logger.debug("os.path.samefile check info: %s", exc)

            stat_a = res_a.stat()
            stat_b = res_b.stat()
            if stat_a.st_size <= 0 or stat_b.st_size <= 0:
                return False, "Plik ma zerowy rozmiar (0 bajtów).", None
            if stat_a.st_size != stat_b.st_size:
                return False, f"Różny rozmiar plików ({stat_a.st_size} B vs {stat_b.st_size} B).", None

            # Layer 1 & 2: Full SHA-256, Full BLAKE2b, and Full MD5 cryptographic verification
            sha_a, blake_a, md5_a = cls.compute_multi_hashes(res_a)
            sha_b, blake_b, md5_b = cls.compute_multi_hashes(res_b)
            if sha_a != sha_b:
                return False, "Niezgodność pełnej sumy SHA-256.", None
            if blake_a != blake_b:
                return False, "Niezgodność pełnej sumy BLAKE2b.", None
            if md5_a != md5_b:
                return False, "Niezgodność pełnej sumy MD5.", None

            # Layer 3: Direct byte-by-byte stream comparison (shallow=False forces full content comparison)
            if not filecmp.cmp(str(res_a), str(res_b), shallow=False):
                return False, "Bezpośrednie porównanie strumieni bajtowych (filecmp) wykryło różnicę.", None

            # Layer 4: Media structural decode integrity verification on retained file
            ext = res_a.suffix.lower()
            if ext in IMAGE_EXTS:
                try:
                    with Image.open(res_a) as img_a, Image.open(res_b) as img_b:
                        if img_a.size != img_b.size or img_a.mode != img_b.mode:
                            return False, "Niezgodność wymiarów lub przestrzeni barw obrazu.", None
                        img_a.verify()
                except Exception as img_exc:
                    logger.debug("Image structural decode warning: %s", img_exc)
                    return False, f"Błąd struktury obrazu podczas weryfikacji: {img_exc}", None

            return True, "Zweryfikowano w 100% (Rozmiar + SHA-256 + BLAKE2b + MD5 + Bajt-po-bajcie).", sha_a
        except Exception as exc:
            return False, f"Wyjątek weryfikacji bezpieczeństwa: {exc}", None

    hash_image = hash_image_full
    hash_video = hash_video_partial


class MetadataDateResolver:
    """Extracts and resolves media creation dates from EXIF, filename, folder names, and filesystem timestamps."""

    IGNORED_FOLDER_NAMES = {
        "", ".", "/", "\\", "backup", "temp", "tmp", "appdata", "local", "users",
        "drivea", "driveb", "drivec_offline", "root_folder", "domowearchiwumalbumy",
    }

    FOLDER_DAY_PATTERNS = [
        re.compile(r"(?<!\d)(?P<year>19[7-9]\d|20[0-3]\d)[-_.](?P<month>0[1-9]|1[0-2])[-_.](?P<day>0[1-9]|[12]\d|3[01])(?!\d)"),
        re.compile(r"(?<!\d)(?P<day>0[1-9]|[12]\d|3[01])[-_.](?P<month>0[1-9]|1[0-2])[-_.](?P<year>19[7-9]\d|20[0-3]\d)(?!\d)"),
        re.compile(r"(?<!\d)(?P<year>19[7-9]\d|20[0-3]\d)(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])(?!\d)"),
    ]
    FOLDER_MONTH_PATTERNS = [
        re.compile(r"(?<!\d)(?P<year>19[7-9]\d|20[0-3]\d)[-_.](?P<month>0[1-9]|1[0-2])(?!\d)"),
        re.compile(r"(?<!\d)(?P<month>0[1-9]|1[0-2])[-_.](?P<year>19[7-9]\d|20[0-3]\d)(?!\d)"),
    ]
    FOLDER_YEAR_PATTERN = re.compile(r"(?<!\d)(?P<year>19[7-9]\d|20[0-3]\d)(?!\d)")

    @staticmethod
    def _is_valid_date(dt: Optional[datetime.datetime], upper_bound: datetime.datetime) -> bool:
        return bool(dt and datetime.datetime(1975, 1, 1) <= dt <= upper_bound)

    @staticmethod
    def parse_exif(path: Path) -> Optional[datetime.datetime]:
        try:
            with Image.open(path) as img:
                exif = img._getexif()
                if not exif:
                    return None
                for tag_name in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                    for tag_id, name in ExifTags.TAGS.items():
                        if name == tag_name and tag_id in exif:
                            val = str(exif[tag_id]).strip("\x00").strip()
                            try:
                                return datetime.datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                            except ValueError as exc:
                                logger.debug("Unparseable EXIF date '%s': %s", val, exc)
        except Exception as exc:
            logger.debug("EXIF extraction skipped for %s: %s", path, exc)
            return None
        return None

    @classmethod
    def parse_filename(cls, name: str) -> Optional[datetime.datetime]:
        for pattern in DATE_PATTERNS:
            m = pattern.search(name)
            if m:
                d = m.groupdict()
                try:
                    return datetime.datetime(
                        int(d["year"]), int(d["month"]), int(d["day"]),
                        int(d.get("hour", 0)), int(d.get("min", 0)), int(d.get("sec", 0)),
                    )
                except ValueError:
                    continue
        return None

    @classmethod
    def parse_folder_date(
        cls,
        file_path: Path,
        rel_path: Optional[str] = None,
    ) -> Optional[Tuple[datetime.datetime, str, str]]:
        """Extracts a probable date (day, month, or year precision) from containing folder names."""
        if rel_path and str(rel_path).strip() not in ("", "."):
            raw_parts = list(Path(rel_path).parent.parts)
        else:
            raw_parts = [file_path.parent.name] if file_path.parent.name else []

        upper_year = datetime.datetime.utcnow().year + 1
        for part in reversed(raw_parts):
            clean = part.strip()
            if not clean or clean.lower() in cls.IGNORED_FOLDER_NAMES or clean.endswith(":"):
                continue
            # Skip hexadecimal hashes / UUID-like folder names
            if len(clean) >= 16 and re.fullmatch(r"[0-9a-fA-F_-]+", clean) and not re.search(r"[-_]", clean):
                continue

            for pat in cls.FOLDER_DAY_PATTERNS:
                m = pat.search(clean)
                if m:
                    yr = int(m.group("year"))
                    if 1975 <= yr <= upper_year:
                        try:
                            return datetime.datetime(yr, int(m.group("month")), int(m.group("day")), 0, 0, 0), "day", clean
                        except ValueError:
                            continue

            for pat in cls.FOLDER_MONTH_PATTERNS:
                m = pat.search(clean)
                if m:
                    yr = int(m.group("year"))
                    if 1975 <= yr <= upper_year:
                        try:
                            return datetime.datetime(yr, int(m.group("month")), 1, 0, 0, 0), "month", clean
                        except ValueError:
                            continue

            m_yr = cls.FOLDER_YEAR_PATTERN.search(clean)
            if m_yr:
                yr = int(m_yr.group("year"))
                if 1975 <= yr <= upper_year:
                    return datetime.datetime(yr, 1, 1, 0, 0, 0), "year", clean
        return None

    @staticmethod
    def dates_agree(dt_a: datetime.datetime, dt_b: datetime.datetime, date_only: bool = False) -> bool:
        if date_only:
            return abs((dt_a.date() - dt_b.date()).days) <= 1
        return abs((dt_a - dt_b).total_seconds()) <= 86400

    @staticmethod
    def matches_folder_date(dt: datetime.datetime, folder_dt: datetime.datetime, precision: str) -> bool:
        if precision == "day":
            return abs((dt.date() - folder_dt.date()).days) <= 1
        if precision == "month":
            return dt.year == folder_dt.year and dt.month == folder_dt.month
        return dt.year == folder_dt.year

    @classmethod
    def evaluate_date_sources(
        cls,
        exif: Optional[datetime.datetime],
        fn_date: Optional[datetime.datetime],
        ctime: datetime.datetime,
        mtime: datetime.datetime,
        folder_info: Optional[Tuple[datetime.datetime, str, str]] = None,
    ) -> Tuple[datetime.datetime, bool, Optional[str]]:
        """
        Determines the most probable effective date across EXIF, filename, folder name, and filesystem
        timestamps, flagging anomalies whenever available sources contradict each other or file timestamps
        were modified during copying. Missing EXIF alone is NOT treated as an anomaly.
        """
        now_upper = datetime.datetime.utcnow() + datetime.timedelta(days=1)
        valid_exif = exif if cls._is_valid_date(exif, now_upper) else None
        valid_fn = fn_date if cls._is_valid_date(fn_date, now_upper) else None
        valid_ctime = ctime if cls._is_valid_date(ctime, now_upper) else now_upper
        valid_mtime = mtime if cls._is_valid_date(mtime, now_upper) else now_upper
        fs_oldest = min(valid_ctime, valid_mtime)
        fs_newest = max(valid_ctime, valid_mtime)

        valid_folder: Optional[Tuple[datetime.datetime, str, str]] = None
        if folder_info and cls._is_valid_date(folder_info[0], now_upper):
            valid_folder = folder_info

        # Select the most probable date from available evidence
        if valid_exif and valid_fn:
            fn_date_only = valid_fn.hour == 0 and valid_fn.minute == 0 and valid_fn.second == 0
            if cls.dates_agree(valid_exif, valid_fn, date_only=fn_date_only):
                effective = valid_exif
            elif valid_folder:
                f_dt, f_prec, _ = valid_folder
                exif_ok = cls.matches_folder_date(valid_exif, f_dt, f_prec)
                fn_ok = cls.matches_folder_date(valid_fn, f_dt, f_prec)
                if exif_ok and not fn_ok:
                    effective = valid_exif
                elif fn_ok and not exif_ok:
                    effective = valid_fn
                else:
                    effective = min(valid_exif, valid_fn)
            else:
                exif_fs = cls.dates_agree(valid_exif, fs_oldest)
                fn_fs = cls.dates_agree(valid_fn, fs_oldest, date_only=fn_date_only)
                if exif_fs and not fn_fs:
                    effective = valid_exif
                elif fn_fs and not exif_fs:
                    effective = valid_fn
                else:
                    effective = min(valid_exif, valid_fn)
        elif valid_exif:
            if valid_folder and not cls.matches_folder_date(valid_exif, valid_folder[0], valid_folder[1]) and cls.matches_folder_date(fs_oldest, valid_folder[0], valid_folder[1]):
                effective = fs_oldest
            else:
                effective = valid_exif
        elif valid_fn:
            if valid_folder and not cls.matches_folder_date(valid_fn, valid_folder[0], valid_folder[1]) and cls.matches_folder_date(fs_oldest, valid_folder[0], valid_folder[1]):
                effective = fs_oldest
            else:
                effective = valid_fn
        elif valid_folder:
            f_dt, f_prec, _ = valid_folder
            if cls.matches_folder_date(fs_oldest, f_dt, f_prec):
                effective = fs_oldest
            else:
                effective = f_dt
        else:
            effective = fs_oldest

        # Flag genuine discrepancies across sources or copy-modified filesystem dates
        reasons: List[str] = []
        if valid_exif and valid_fn:
            fn_date_only = valid_fn.hour == 0 and valid_fn.minute == 0 and valid_fn.second == 0
            if not cls.dates_agree(valid_exif, valid_fn, date_only=fn_date_only):
                reasons.append(
                    f"Niezgodność dat: EXIF ({valid_exif.strftime('%Y-%m-%d')}) vs nazwa pliku ({valid_fn.strftime('%Y-%m-%d')}) (>24h)"
                )

        if valid_folder:
            f_dt, f_prec, f_seg = valid_folder
            if valid_exif and not cls.matches_folder_date(valid_exif, f_dt, f_prec):
                reasons.append(f"Niezgodność dat: EXIF ({valid_exif.strftime('%Y-%m-%d')}) z folderem '{f_seg}'")
            if valid_fn and not cls.matches_folder_date(valid_fn, f_dt, f_prec):
                reasons.append(f"Niezgodność dat: nazwa pliku ({valid_fn.strftime('%Y-%m-%d')}) z folderem '{f_seg}'")

        if valid_exif or valid_fn:
            ref_date_only = valid_exif is None and valid_fn is not None and valid_fn.hour == 0 and valid_fn.minute == 0 and valid_fn.second == 0
            if not cls.dates_agree(fs_oldest, effective, date_only=ref_date_only) or not cls.dates_agree(fs_newest, effective, date_only=ref_date_only):
                reasons.append(
                    f"Niezgodność dat: data pliku ({fs_newest.strftime('%Y-%m-%d')}) zmodyfikowana przy kopiowaniu (ustawiono {effective.strftime('%Y-%m-%d')})"
                )
        elif valid_folder:
            f_dt, f_prec, f_seg = valid_folder
            if not cls.matches_folder_date(fs_oldest, f_dt, f_prec) or not cls.matches_folder_date(fs_newest, f_dt, f_prec):
                reasons.append(
                    f"Niezgodność dat: data pliku ({fs_newest.strftime('%Y-%m-%d')}) z folderem '{f_seg}' po kopiowaniu (ustawiono {effective.strftime('%Y-%m-%d')})"
                )
        elif abs((valid_ctime - valid_mtime).total_seconds()) > 86400:
            reasons.append(
                f"Niezgodność dat: utworzenie pliku ({valid_ctime.strftime('%Y-%m-%d')}) zmodyfikowane przy kopiowaniu vs modyfikacja ({valid_mtime.strftime('%Y-%m-%d')})"
            )

        if reasons:
            return effective, True, "; ".join(reasons)[:255]
        return effective, False, None

    @classmethod
    def resolve_dates(cls, file_path: Path, is_image: bool, rel_path: Optional[str] = None):
        stat = file_path.stat()
        ctime = datetime.datetime.fromtimestamp(stat.st_ctime)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        exif = cls.parse_exif(file_path) if is_image else None
        fn_date = cls.parse_filename(file_path.name)
        folder_info = cls.parse_folder_date(file_path, rel_path=rel_path)

        effective, anomaly, reason = cls.evaluate_date_sources(exif, fn_date, ctime, mtime, folder_info=folder_info)
        return exif, fn_date, ctime, mtime, effective, anomaly, reason

    resolve = resolve_dates


class ThumbnailService:
    """Generates lightweight 300px WebP thumbnails on the local disk for offline browsing."""

    @classmethod
    def generate(cls, path: Path, media_id: str, is_img: bool) -> Tuple[str, Optional[int], Optional[int], Optional[float]]:
        dest = THUMBNAILS_DIR / f"{media_id}.webp"
        if is_img:
            with Image.open(path) as img:
                orig_w, orig_h = img.size
                try:
                    img.draft("RGB", (320, 320))
                except Exception as exc:
                    logger.debug("Image.draft not applicable: %s", exc)
                img = ImageOps.exif_transpose(img)
                img.thumbnail((300, 300), Image.Resampling.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(dest, "WEBP", quality=78, method=2)
            return dest.name, orig_w, orig_h, None
        else:
            cap = cv2.VideoCapture(str(path.resolve()))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur = float(frames / fps) if fps > 0 else 0.0

                cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(fps), max(0, int(frames - 1))))
                success, frame = cap.read()
                if not success or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = cap.read()

                if success and frame is not None:
                    scale = min(300 / max(1, frame.shape[1]), 300 / max(1, frame.shape[0]))
                    resized = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))))
                    cv2.imwrite(str(dest.resolve()), resized, [int(cv2.IMWRITE_WEBP_QUALITY), 78])
                else:
                    dest.touch()
                return dest.name, w, h, dur
            finally:
                cap.release()


class FileScannerService:
    """Indexes individual image and video files into MediaItem entities with thumbnails and 512-D vectors."""

    @staticmethod
    def is_in_backup_folder(rel_path: str) -> bool:
        parts = [p.lower() for p in Path(rel_path).parts if p not in ("", ".", "/", "\\")]
        return bool(parts and parts[0] == "backup")

    @classmethod
    def process_file(cls, path: Path, volume: WindowsVolume, rel_override: Optional[str] = None) -> Optional[MediaItem]:
        ext = path.suffix.lower()
        is_img = ext in IMAGE_EXTS
        is_vid = ext in VIDEO_EXTS
        if not (is_img or is_vid):
            return None

        try:
            stat = path.stat()
        except OSError:
            return None

        if stat.st_size == 0:
            return None

        if rel_override:
            rel_path = rel_override
        else:
            drive_root = Path(volume.last_known_drive_letter)
            try:
                rel_path = path.resolve().relative_to(drive_root.resolve()).as_posix()
            except ValueError:
                rel_path = path.name

        media_id = str(uuid.uuid4())
        file_hash = FastHasher.hash_image_full(path) if is_img else FastHasher.hash_video_partial(path, stat.st_size)
        exif, fn_dt, ctime, mtime, effective, anomaly, reason = MetadataDateResolver.resolve_dates(
            path, is_img, rel_path=rel_path
        )

        try:
            thumb_name, w, h, dur = ThumbnailService.generate(path, media_id, is_img)
            corrupted = False
        except Exception as exc:
            logger.debug("Thumbnail generation failed for %s: %s", path, exc)
            thumb_name = f"{media_id}.webp"
            (THUMBNAILS_DIR / thumb_name).touch()
            w, h, dur, corrupted = None, None, None, True

        is_backup = cls.is_in_backup_folder(rel_path)
        now = datetime.datetime.utcnow()
        item = MediaItem(
            id=media_id,
            volume_id=volume.id,
            relative_path=rel_path,
            file_name=path.name,
            file_size_bytes=stat.st_size,
            media_type="image" if is_img else "video",
            mime_type=f"{'image' if is_img else 'video'}/{ext.lstrip('.')}",
            file_hash=file_hash,
            hash_algorithm="sha256" if is_img else "partial_3x64k",
            is_primary_duplicate=not is_backup,
            is_backup_copy=is_backup,
            verified_sha256=file_hash if is_img else None,
            width=w,
            height=h,
            duration_seconds=dur,
            rotation_angle=0,
            exif_date=exif,
            filename_date=fn_dt,
            fs_ctime=ctime,
            fs_mtime=mtime,
            effective_date=effective,
            has_date_anomaly=anomaly,
            date_anomaly_reason=reason,
            is_corrupted=corrupted,
            is_blurry=False,
            blur_score=None,
            thumbnail_relative_path=thumb_name,
            group_id=None,
            is_trashed=False,
            created_at=now,
            updated_at=now,
        )

        # Compute 512-D visual embedding from cached thumbnail during indexing
        thumb_full = THUMBNAILS_DIR / thumb_name
        if not corrupted and thumb_full.exists() and thumb_full.stat().st_size > 0:
            try:
                blob = OnDemandClipEmbedder.generate_image_embedding(thumb_full)
                item.embedding = MediaEmbedding(
                    media_id=media_id, model_name=OnDemandClipEmbedder.MODEL_NAME, dimensions=512, vector=blob
                )
            except Exception as exc:
                logger.debug("Embedding generation skipped for %s: %s", media_id, exc)

        return item
