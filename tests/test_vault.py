import datetime
import hashlib
import io
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import uuid

import cv2
from fastapi.testclient import TestClient
from starlette.requests import Request
import numpy as np
from PIL import Image, JpegImagePlugin
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import sessionmaker, Session

from app.database import Base, ensure_schema_migrations, get_db, set_sqlite_pragma
from app.models import (
    WindowsVolume,
    MediaGroup,
    MediaItem,
    MediaTag,
    MediaEmbedding,
    BackgroundJob,
    OperationLogEntry,
    DeletedDuplicateHistory,
)
from app.backup import BackupManager
from app.operation_log import OperationLogger, ProcessManager
from app.scanner import (
    WindowsVolumeInspector,
    FastHasher,
    MetadataDateResolver,
    ThumbnailService,
    FileScannerService,
)
from app.storage_ops import StorageAutomationService
from app.ai.embeddings import OnDemandClipEmbedder
from app.ai.quality import MediaQualityAnalyzer
from app.ai.tagger import OnDemandObjectTagger
from app.cloud.google_drive import GoogleDriveManager
import main as main_module
from app.routers import actions as actions_module
from app.routers import media as media_module
from app.routers import semantic as semantic_module


def create_synthetic_jpeg(
    path: Path,
    width: int = 320,
    height: int = 240,
    sharp: bool = True,
    color=(40, 120, 200),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    if sharp:
        for y in range(0, height, 16):
            for x in range(0, width, 16):
                if ((x // 16) + (y // 16)) % 2 == 0:
                    img[y : y + 16, x : x + 16] = (240, 240, 240)
        cv2.rectangle(img, (20, 20), (width - 20, height - 20), (10, 250, 10), 2)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img.save(path, format="JPEG", quality=95)
    return path


class TestDomoweArchiwumFotoSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "test_vault.db"
        self.backup_dir = self.root / "backups"
        self.thumbs_dir = self.root / "thumbnails"
        self.logs_dir = self.root / "logs"
        self.reports_dir = self.root / "reports"
        self.log_file = self.logs_dir / "operations.log"
        self.drive_a = self.root / "DriveA"
        self.drive_b = self.root / "DriveB"
        self.drive_c_offline = self.root / "DriveC_Offline"

        for d in (self.backup_dir, self.thumbs_dir, self.logs_dir, self.reports_dir, self.drive_a, self.drive_b):
            d.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        from sqlalchemy import event
        def fast_test_pragma(dbapi_connection, connection_record):
            set_sqlite_pragma(dbapi_connection, connection_record)
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA synchronous = OFF;")
            cur.close()

        event.listen(self.engine, "connect", fast_test_pragma)
        Base.metadata.create_all(bind=self.engine)
        ensure_schema_migrations(self.db_path)

        self.TestingSessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=False
        )

        # Patch global paths and session factories so routers and background jobs use isolated test environment
        self.vsn_patch = patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={})
        self.patches = [
            patch("app.database.DB_PATH", self.db_path),
            patch("app.database.BACKUP_DIR", self.backup_dir),
            patch("app.database.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.database.REPORTS_DIR", self.reports_dir),
            patch("app.database.OPERATIONS_LOG_FILE", self.log_file),
            patch("app.database.SessionLocal", self.TestingSessionLocal),
            patch("app.scanner.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.media.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.semantic.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.actions.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.actions.DATA_DIR", self.root),
            patch("app.routers.actions.SessionLocal", self.TestingSessionLocal),
            self.vsn_patch,
        ]
        for p in self.patches:
            p.start()

        def override_get_db():
            db = self.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        main_module.app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(main_module.app)

    def tearDown(self):
        main_module.app.dependency_overrides.clear()
        for p in reversed(self.patches):
            p.stop()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _delete_permanently_two_phase(self, data: dict, headers=None, allow_originals: bool = True):
        """Executes Zero-Trust Two-Phase Commit: /actions/delete-preview -> token -> /actions/delete-permanently."""
        import re

        r_prev = self.client.post("/actions/delete-preview", data=data)
        m_tok = re.search(r"name='deletion_token' value='([^']+)'", r_prev.text)
        if not m_tok:
            return r_prev
        m_cnt = re.search(r"name='confirmed_count' value='([^']+)'", r_prev.text)
        post_data = {
            "scope": "selected",
            "deletion_token": m_tok.group(1),
            "confirmed_count": m_cnt.group(1) if m_cnt else "",
            "duplicates_only": str(data.get("duplicates_only", "false")),
        }
        if "skip_db_backup" in data:
            post_data["skip_db_backup"] = data["skip_db_backup"]
        if "run_sync" in data:
            post_data["run_sync"] = data["run_sync"]
        if allow_originals and "confirm_originals_text" in r_prev.text:
            post_data["allow_originals"] = "true"
            post_data["confirm_originals_text"] = "USUŃ"
        return self.client.post("/actions/delete-permanently", data=post_data, headers=headers or {})

    # =========================================================================
    # 1. HASHING, 100% MULTI-LAYER VERIFICATION, DATE RESOLUTION & SCANNER
    # =========================================================================
    def test_hasher_and_100_percent_verification_layers(self):
        f1 = create_synthetic_jpeg(self.drive_a / "orig.jpg", sharp=True)
        f2 = self.drive_b / "copy.jpg"
        f2.write_bytes(f1.read_bytes())

        # Identical distinct physical files must pass 100% verification
        ok, msg, sha = FastHasher.verify_files_identical_100_percent(f1, f2)
        self.assertTrue(ok, msg)
        self.assertEqual(len(sha), 64)

        # Same physical file path must be blocked (protects against deleting the only copy!)
        ok_same, msg_same, _ = FastHasher.verify_files_identical_100_percent(f1, f1)
        self.assertFalse(ok_same)
        self.assertIn("ten sam fizyczny plik", msg_same)

        # Simulated os.path.samefile returning True on distinct Path objects
        with patch("os.path.samefile", return_value=True):
            ok_hardlink, msg_hl, _ = FastHasher.verify_files_identical_100_percent(f1, f2)
            self.assertFalse(ok_hardlink)
            self.assertIn("inode/hardlink", msg_hl)

        # Simulated filecmp returning False
        with patch("filecmp.cmp", return_value=False):
            ok_cmp, msg_cmp, _ = FastHasher.verify_files_identical_100_percent(f1, f2)
            self.assertFalse(ok_cmp)
            self.assertIn("filecmp", msg_cmp)

        # Simulated BLAKE2b / MD5 mismatch
        with patch.object(FastHasher, "compute_multi_hashes", side_effect=[("sha1", "blake1", "md5_1"), ("sha1", "blake2", "md5_1")]):
            ok_b, msg_b, _ = FastHasher.verify_files_identical_100_percent(f1, f2)
            self.assertFalse(ok_b)
            self.assertIn("BLAKE2b", msg_b)

        with patch.object(FastHasher, "compute_multi_hashes", side_effect=[("sha1", "blake1", "md5_1"), ("sha1", "blake1", "md5_2")]):
            ok_m, msg_m, _ = FastHasher.verify_files_identical_100_percent(f1, f2)
            self.assertFalse(ok_m)
            self.assertIn("MD5", msg_m)

        # Missing file must fail
        ok_miss, _, _ = FastHasher.verify_files_identical_100_percent(f1, self.drive_b / "nonexistent.jpg")
        self.assertFalse(ok_miss)

        # Directory instead of file must fail
        ok_dir, _, _ = FastHasher.verify_files_identical_100_percent(f1, self.drive_b)
        self.assertFalse(ok_dir)

        # Zero-byte file must fail
        f_zero = self.drive_b / "zero.jpg"
        f_zero.touch()
        ok_zero, _, _ = FastHasher.verify_files_identical_100_percent(f1, f_zero)
        self.assertFalse(ok_zero)

        # Different size must fail
        f_diff_size = self.drive_b / "diff_size.jpg"
        f_diff_size.write_bytes(f1.read_bytes() + b"\x00")
        ok_size, _, _ = FastHasher.verify_files_identical_100_percent(f1, f_diff_size)
        self.assertFalse(ok_size)

        # Same size, mutated byte content must fail
        mutated = bytearray(f1.read_bytes())
        mutated[-10] ^= 0xFF
        f_mut = self.drive_b / "mutated.jpg"
        f_mut.write_bytes(bytes(mutated))
        ok_mut, _, _ = FastHasher.verify_files_identical_100_percent(f1, f_mut)
        self.assertFalse(ok_mut)

        # Corrupted non-image with .jpg extension but identical bytes must fail structural image decode check
        bad_img_a = self.drive_a / "bad1.jpg"
        bad_img_b = self.drive_b / "bad2.jpg"
        bad_img_a.write_bytes(b"NOT_A_JPEG_HEADER_AT_ALL" * 20)
        bad_img_b.write_bytes(b"NOT_A_JPEG_HEADER_AT_ALL" * 20)
        ok_bad_img, _, _ = FastHasher.verify_files_identical_100_percent(bad_img_a, bad_img_b)
        self.assertFalse(ok_bad_img)

    def test_video_partial_hash_small_and_large(self):
        small_vid = self.drive_a / "small.mp4"
        small_vid.write_bytes(b"V" * 10000)
        h_small = FastHasher.hash_video_partial(small_vid, 10000)
        self.assertEqual(h_small, FastHasher.hash_image_full(small_vid))

        large_vid = self.drive_a / "large.mp4"
        size_b = 512 * 1024
        large_vid.write_bytes(b"A" * size_b)
        h1 = FastHasher.hash_video_partial(large_vid, size_b)
        with open(large_vid, "r+b") as f:
            f.seek(size_b // 2)
            f.write(b"Z")
        h2 = FastHasher.hash_video_partial(large_vid, size_b)
        self.assertNotEqual(h1, h2)

    def test_date_resolver_and_scanner_service(self):
        img_file = create_synthetic_jpeg(self.drive_a / "IMG_20210510_123045.jpg")
        exif_dt, fn_dt, ctime, mtime, effective_dt, anomaly, reason = MetadataDateResolver.resolve_dates(img_file, is_image=True)
        self.assertIsNone(exif_dt)
        self.assertEqual(fn_dt, datetime.datetime(2021, 5, 10, 12, 30, 45))
        # Rule (b): Without EXIF, effective_date is ALWAYS mtime even if filename date differs (with filename anomaly!)
        self.assertEqual(effective_dt, mtime)
        self.assertTrue(anomaly)
        self.assertIn("nazwie pliku", reason)

        # Missing EXIF alone WITHOUT path/filename discrepancy must NOT trigger an anomaly
        plain_img = create_synthetic_jpeg(self.drive_a / "random_shot.jpg")
        _, _, _, plain_mtime, plain_eff, plain_anom, plain_reason = MetadataDateResolver.resolve_dates(plain_img, is_image=True)
        self.assertEqual(plain_eff, plain_mtime)
        self.assertFalse(plain_anom)
        self.assertIsNone(plain_reason)

        # Folder date discrepancy: without EXIF, effective is mtime AND path discrepancy is flagged
        folder_img = create_synthetic_jpeg(self.drive_a / "2019-08_Grecja" / "beach.jpg")
        _, _, _, f_mtime, f_eff, f_anom, f_rsn = MetadataDateResolver.resolve_dates(
            folder_img, is_image=True, rel_path="2019-08_Grecja/beach.jpg"
        )
        self.assertTrue(f_anom)
        self.assertEqual(f_eff, f_mtime)
        self.assertIn("ścieżce", f_rsn)
        self.assertIn("folder", f_rsn)

        # Rule (b): Day-level folder may differ in DAY as long as YEAR and MONTH match -> NO anomaly
        same_ym_eff, same_ym_anom, same_ym_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=None,
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2022, 7, 28, 14, 0),
            folder_info=(datetime.datetime(2022, 7, 15, 0, 0), "day", "2022-07-15"),
        )
        self.assertFalse(same_ym_anom)
        self.assertIsNone(same_ym_rsn)
        self.assertEqual(same_ym_eff, datetime.datetime(2022, 7, 28, 14, 0))

        # Difference between ctime and mtime is NOT an anomaly (shown in details only!), effective = mtime
        cp_eff, cp_anom, cp_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=None,
            ctime=datetime.datetime(2026, 6, 1, 10, 0),
            mtime=datetime.datetime(2021, 3, 20, 9, 0),
            folder_info=None,
        )
        self.assertFalse(cp_anom)
        self.assertIsNone(cp_rsn)
        self.assertEqual(cp_eff, datetime.datetime(2021, 3, 20, 9, 0))

        # Rule (a): EXIF is effective_date; disagreements with path and filename ARE flagged cleanly
        ex_eff, ex_anom, ex_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=datetime.datetime(2023, 5, 10, 12, 0),
            fn_date=datetime.datetime(2021, 5, 10, 12, 0),
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2026, 1, 1, 10, 0),
            folder_info=(datetime.datetime(2021, 5, 1, 0, 0), "month", "2021-05"),
        )
        self.assertEqual(ex_eff, datetime.datetime(2023, 5, 10, 12, 0))
        self.assertTrue(ex_anom)
        self.assertIn("ścieżce", ex_rsn)
        self.assertIn("nazwie pliku", ex_rsn)

        # Rule (a) when EXIF matches folder & filename (even if ctime/mtime are much newer) -> NO anomaly
        ex_ok_eff, ex_ok_anom, ex_ok_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=datetime.datetime(2023, 5, 10, 12, 0),
            fn_date=datetime.datetime(2023, 5, 10, 12, 0),
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2025, 2, 1, 10, 0),
            folder_info=(datetime.datetime(2023, 5, 1, 0, 0), "month", "2023-05"),
        )
        self.assertEqual(ex_ok_eff, datetime.datetime(2023, 5, 10, 12, 0))
        self.assertFalse(ex_ok_anom)
        self.assertIsNone(ex_ok_rsn)

        # Rule (b) without EXIF: effective is ALWAYS mtime even if filename/folder are older; both discrepancies reported
        eff_fn_fold, anom_fn_fold, rsn_fn_fold = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=datetime.datetime(2019, 5, 10, 12, 0),
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2025, 6, 5, 10, 0),
            folder_info=(datetime.datetime(2021, 5, 1, 0, 0), "month", "2021-05"),
        )
        self.assertTrue(anom_fn_fold)
        self.assertEqual(eff_fn_fold, datetime.datetime(2025, 6, 5, 10, 0))
        self.assertIn("ścieżce", rsn_fn_fold)
        self.assertIn("nazwie pliku", rsn_fn_fold)

        # Rule (b) when mtime agrees with filename and folder (even with newer ctime) -> NO anomaly
        eff_agree, anom_agree, _ = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=datetime.datetime(2022, 4, 5, 11, 0),
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2022, 4, 5, 11, 5),
            folder_info=MetadataDateResolver.parse_folder_date(self.drive_a / "2022-04-05" / "a.jpg", rel_path="abcdef0123456789abcdef/2022-04-05/a.jpg"),
        )
        self.assertFalse(anom_agree)
        self.assertEqual(eff_agree, datetime.datetime(2022, 4, 5, 11, 5))

        # Test EXIF parsing on JpegImageFile (EXIF matches plain filename -> NO anomaly; mismatched filename -> anomaly)
        with patch.object(JpegImagePlugin.JpegImageFile, "_getexif", return_value={36867: "2023:05:10 12:30:45"}):
            ex_dt, _, _, _, eff_dt, anom, rsn = MetadataDateResolver.resolve_dates(plain_img, is_image=True)
            self.assertEqual(ex_dt, datetime.datetime(2023, 5, 10, 12, 30, 45))
            self.assertEqual(eff_dt, ex_dt)
            self.assertFalse(anom)
            self.assertIsNone(rsn)
            # On IMG_20210510_123045.jpg, EXIF 2023 disagrees with filename 2021 -> effective=EXIF, anomaly=True
            _, _, _, _, eff_mm, anom_mm, rsn_mm = MetadataDateResolver.resolve_dates(img_file, is_image=True)
            self.assertEqual(eff_mm, datetime.datetime(2023, 5, 10, 12, 30, 45))
            self.assertTrue(anom_mm)
            self.assertIn("nazwie pliku", rsn_mm)

        with patch.object(JpegImagePlugin.JpegImageFile, "_getexif", return_value={36867: "bad-date"}):
            self.assertIsNone(MetadataDateResolver.parse_exif(img_file))

        # Invalid calendar date in filename should be skipped gracefully
        self.assertIsNone(MetadataDateResolver.parse_filename("IMG_20210231_120000.jpg"))
        self.assertIsNone(MetadataDateResolver.parse_filename("random_name.jpg"))

        # Test WindowsVolumeInspector & fallback when kernel32 fails
        vsn, label, root = WindowsVolumeInspector.get_volume_info(self.drive_a)
        self.assertTrue(len(vsn) > 0)
        with patch("ctypes.windll.kernel32.GetVolumeInformationW", return_value=0):
            vsn_fb, lbl_fb, _ = WindowsVolumeInspector.get_volume_info(self.drive_a)
            self.assertTrue(vsn_fb.startswith("VOL_"))
            self.assertEqual(lbl_fb, "LOCAL_STORAGE")

        # Test real get_connected_vsn_map with bitmask mocked to C:\ only (bit 2 = 4)
        self.vsn_patch.stop()
        try:
            with patch("ctypes.windll.kernel32.GetLogicalDrives", return_value=4):
                connected = WindowsVolumeInspector.get_connected_vsn_map()
                self.assertIsInstance(connected, dict)
        finally:
            self.vsn_patch.start()

        with self.TestingSessionLocal() as db:
            vol = WindowsVolume(volume_serial_number="VSN-A", volume_label="Dysk_A", last_known_drive_letter=str(self.drive_a))
            db.add(vol)
            db.commit()
            db.refresh(vol)

            item = FileScannerService.process_file(img_file, vol)
            self.assertIsNotNone(item)
            self.assertFalse(item.is_backup_copy)
            self.assertIsNotNone(item.embedding)

            # File inside daf_backup/ or backup/ folder must be flagged is_backup_copy=True
            bkp_sub = self.drive_a / "daf_backup" / "2021"
            bkp_sub.mkdir(parents=True, exist_ok=True)
            bkp_img = create_synthetic_jpeg(bkp_sub / "IMG_20210510_123045.jpg")
            bkp_item = FileScannerService.process_file(bkp_img, vol)
            self.assertTrue(bkp_item.is_backup_copy)
            self.assertFalse(bkp_item.is_primary_duplicate)

            bkp_legacy = self.drive_a / "backup" / "2021"
            bkp_legacy.mkdir(parents=True, exist_ok=True)
            bkp_legacy_img = create_synthetic_jpeg(bkp_legacy / "IMG_20210510_123046.jpg")
            bkp_legacy_item = FileScannerService.process_file(bkp_legacy_img, vol)
            self.assertTrue(bkp_legacy_item.is_backup_copy)

            # Corrupted image file handling in FileScannerService.process_file
            corrupt_jpg = self.drive_a / "corrupt_scan.jpg"
            corrupt_jpg.write_bytes(b"CORRUPTED_BYTES_NOT_IMAGE")
            corrupt_item = FileScannerService.process_file(corrupt_jpg, vol)
            self.assertIsNotNone(corrupt_item)
            self.assertTrue(corrupt_item.is_corrupted)

            # Unsupported, missing, and empty files return None
            txt_f = self.drive_a / "note.txt"
            txt_f.write_text("hello")
            self.assertIsNone(FileScannerService.process_file(txt_f, vol))
            self.assertIsNone(FileScannerService.process_file(self.drive_a / "missing.jpg", vol))
            empty_f = self.drive_a / "empty.jpg"
            empty_f.touch()
            self.assertIsNone(FileScannerService.process_file(empty_f, vol))

            # Synthetic video scanning + MediaQualityAnalyzer.analyze_video
            vid_path = self.drive_a / "VID_20220801_100000.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (160, 120))
            for _ in range(5):
                frame = np.full((120, 160, 3), 128, dtype=np.uint8)
                cv2.rectangle(frame, (10, 10), (100, 100), (255, 255, 255), 2)
                writer.write(frame)
            writer.release()
            if vid_path.exists() and vid_path.stat().st_size > 0:
                vid_item = FileScannerService.process_file(vid_path, vol)
                self.assertIsNotNone(vid_item)
                self.assertEqual(vid_item.media_type, "video")
                v_corrupt, v_blur, v_score = MediaQualityAnalyzer.analyze_video(vid_path)
                self.assertFalse(v_corrupt)
                self.assertIsNotNone(v_score)

            # Corrupted/missing video check
            v_c2, _, _ = MediaQualityAnalyzer.analyze_video(self.drive_a / "no_video.mp4")
            self.assertTrue(v_c2)
            bad_vid = self.drive_a / "bad.mp4"
            bad_vid.write_bytes(b"NOT_A_VIDEO")
            v_c3, _, _ = MediaQualityAnalyzer.analyze_video(bad_vid)
            self.assertTrue(v_c3)

    # =========================================================================
    # 2. BACKUP & 100% SAFE DEDUPLICATION + ALBUM 'WAŻNE' + PAIRWISE WORKFLOW
    # =========================================================================
    def test_backup_and_deduplication_and_important_album(self):
        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="VSN-TEST-A",
                volume_label="Dysk_A",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            vol_b = WindowsVolume(
                volume_serial_number="VSN-TEST-B",
                volume_label="Dysk_B",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
            )
            vol_c = WindowsVolume(
                volume_serial_number="VSN-TEST-C",
                volume_label="Dysk_Offline_C",
                last_known_drive_letter=str(self.drive_c_offline),
                is_online=False,
            )
            db.add_all([vol_a, vol_b, vol_c])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            db.refresh(vol_c)
            vol_a_id, vol_b_id, vol_c_id = vol_a.id, vol_b.id, vol_c.id

            # Create File 1: 1 copy on Drive A, 2 copies on Drive B (with mtime = 2023-06-15)
            p1_a = create_synthetic_jpeg(self.drive_a / "IMG_20230615_100000.jpg", color=(10, 80, 200))
            p1_b_orig = self.drive_b / "01_Wakacje" / "IMG_20230615_100000.jpg"
            p1_b_orig.parent.mkdir(parents=True, exist_ok=True)
            p1_b_orig.write_bytes(p1_a.read_bytes())
            p1_b_dup = self.drive_b / "99_Kopia" / "IMG_20230615_100000_dup.jpg"
            p1_b_dup.parent.mkdir(parents=True, exist_ok=True)
            p1_b_dup.write_bytes(p1_a.read_bytes())
            ts_2023_06_15 = datetime.datetime(2023, 6, 15, 10, 0, 0).timestamp()
            for p_ts in (p1_a, p1_b_orig, p1_b_dup):
                os.utime(p_ts, (ts_2023_06_15, ts_2023_06_15))

            # Create File 2: only 1 copy on Drive A (unbacked!, mtime = 2023-07-20)
            p2_a = create_synthetic_jpeg(self.drive_a / "IMG_20230720_150000.jpg", color=(200, 50, 30))
            ts_2023_07_20 = datetime.datetime(2023, 7, 20, 15, 0, 0).timestamp()
            os.utime(p2_a, (ts_2023_07_20, ts_2023_07_20))

            for p, v in [(p1_a, vol_a), (p1_b_orig, vol_b), (p1_b_dup, vol_b), (p2_a, vol_a)]:
                it = FileScannerService.process_file(p, v)
                db.add(it)
            db.commit()

            svc = StorageAutomationService(db)
            svc.flag_duplicate_clusters()

            # Check GET /actions/backup-panel
            r_bp = self.client.get(f"/actions/backup-panel?target_volume_id={vol_b_id}")
            self.assertEqual(r_bp.status_code, 200)

            # Check unbacked detection (Rule 1g)
            unbacked_before = svc.detect_files_without_backup()
            self.assertEqual(len(unbacked_before), 2)

            # Execute automated backup and 100%-verified deduplication via HTTP endpoint
            r_exec_bkp = self.client.post(
                "/actions/execute-backup",
                data={"preferred_target_volume_id": str(vol_b_id)},
            )
            self.assertEqual(r_exec_bkp.status_code, 200)

            # After backup execution, 0 files lack backup on another device; backup does NOT delete duplicates!
            db.expire_all()
            unbacked_after = svc.detect_files_without_backup()
            self.assertEqual(len(unbacked_after), 0)
            self.assertTrue(p1_b_dup.exists(), "Backup must not delete duplicates automatically")
            self.assertTrue(p1_b_orig.exists(), "Original on Drive B must remain untouched in its place")

            # Now remove redundant duplicate via /actions/redundant-duplicate-ids + /actions/delete-permanently
            r_dup_ids = self.client.get("/actions/redundant-duplicate-ids?duplicates_only=true")
            self.assertEqual(r_dup_ids.status_code, 200)
            dup_ids_list = r_dup_ids.json()["ids"]
            self.assertTrue(len(dup_ids_list) >= 1)
            r_del_dups = self._delete_permanently_two_phase(
                {
                    "scope": "selected",
                    "duplicates_only": "true",
                    "selected_ids_json": json.dumps(dup_ids_list),
                },
            )
            self.assertEqual(r_del_dups.status_code, 200)
            self.assertFalse(p1_b_dup.exists(), "Redundant duplicate on Drive B must be safely removed from Duplikaty binarne")
            self.assertTrue(p1_b_orig.exists(), "Original on Drive B must remain untouched")

            # Verify Album 'Ważne - Dysk_B' -> '2023' -> '06' -> '15' (rok/miesiac/dzien) hierarchy was created
            important_root = db.scalar(select(MediaGroup).where(MediaGroup.name == "Ważne - Dysk_B"))
            self.assertIsNotNone(important_root)
            imp_yr = db.scalar(select(MediaGroup).where(MediaGroup.name == "2023", MediaGroup.parent_id == important_root.id))
            self.assertIsNotNone(imp_yr)
            imp_mo = db.scalar(select(MediaGroup).where(MediaGroup.name == "06", MediaGroup.parent_id == imp_yr.id))
            self.assertIsNotNone(imp_mo)
            imp_day = db.scalar(select(MediaGroup).where(MediaGroup.name == "15", MediaGroup.parent_id == imp_mo.id))
            self.assertIsNotNone(imp_day)

            # Test insufficient disk space warning in execute_backup_and_deduplication
            p3_a = create_synthetic_jpeg(self.drive_a / "IMG_20230801_120000.jpg", color=(90, 90, 10))
            it3 = FileScannerService.process_file(p3_a, vol_a)
            db.add(it3)
            db.commit()
            with patch.object(StorageAutomationService, "check_volume_free_space", return_value=(False, 1024)):
                res_nospace = svc.execute_backup_and_deduplication(preferred_target_volume_id=vol_b_id)
                self.assertEqual(len(res_nospace["space_warnings"]), 1)

            # Test offline pair prompt when target volume is offline
            res_offline_pair = svc.execute_backup_and_deduplication(preferred_target_volume_id=vol_c_id)
            self.assertGreaterEqual(res_offline_pair["pending_offline_pairs_count"], 1)
            self.assertIsNotNone(res_offline_pair["next_pair_prompt"])

            # Test /actions/execute-backup when target is offline (sets job status to WAITING_FOR_DEVICE)
            r_wait = self.client.post(
                "/actions/execute-backup",
                data={"preferred_target_volume_id": str(vol_c_id)},
            )
            self.assertEqual(r_wait.status_code, 200)

            # Test verification_blocks when verify_files_identical_100_percent returns False during backup copy
            with patch.object(FastHasher, "verify_files_identical_100_percent", return_value=(False, "SimFail", None)):
                res_vf = svc.execute_backup_and_deduplication(preferred_target_volume_id=vol_b_id)
                self.assertGreaterEqual(res_vf["verification_blocks"], 1)

            # Test duplicate deletion when physical duplicate file is already missing on disk, and when 100% check blocks deletion
            (self.drive_b / "01_DupTest").mkdir(parents=True, exist_ok=True)
            p4_b1 = create_synthetic_jpeg(self.drive_b / "01_DupTest" / "IMG_20230901_100000.jpg", color=(12, 34, 56))
            (self.drive_b / "99_DupTest").mkdir(parents=True, exist_ok=True)
            p4_b2 = self.drive_b / "99_DupTest" / "IMG_20230901_100000_dup.jpg"
            p4_b2.write_bytes(p4_b1.read_bytes())
            (self.drive_a / "backup").mkdir(parents=True, exist_ok=True)
            p4_a_bkp = self.drive_a / "backup" / "IMG_20230901_100000.jpg"
            p4_a_bkp.write_bytes(p4_b1.read_bytes())
            for p_x, v_x in [(p4_b1, vol_b), (p4_b2, vol_b), (p4_a_bkp, vol_a)]:
                db.add(FileScannerService.process_file(p_x, v_x))
            db.commit()

            # Block duplicate deletion via 100% verification failure
            with patch.object(FastHasher, "verify_files_identical_100_percent", return_value=(False, "BlockDup", None)):
                svc.execute_backup_and_deduplication(preferred_target_volume_id=vol_a_id)

            # Unlink p4_b2 so missing dup_path cleanup branch is exercised
            if p4_b2.exists():
                p4_b2.unlink()
            svc.execute_backup_and_deduplication(preferred_target_volume_id=vol_a_id)

            # Test new_backup_drive_path registration in /actions/execute-backup
            extra_drv = self.root / "ExtraBackupDrive"
            extra_drv.mkdir(parents=True, exist_ok=True)
            with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("VSN-EXTRA-99", "Extra_Bkp", str(extra_drv))):
                self.client.post("/actions/execute-backup", data={"new_backup_drive_path": str(extra_drv)})

            # Test refresh_volumes_online_status when VSN is in connected_map
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"VSN-TEST-A": (str(self.drive_a), "Dysk_A")}):
                svc.refresh_volumes_online_status()

            # Test check_volume_free_space on invalid path
            ok_sp, free_sp = StorageAutomationService.check_volume_free_space("Z:\\NonExistentDriveRoot", 100)
            self.assertFalse(ok_sp)
            self.assertEqual(free_sp, 0)

    # =========================================================================
    # 3. ALBUMS: FAST DELETE, TRANSFER & DELETE SOURCE, 1-CLICK AUTO-ORGANIZE
    # =========================================================================
    def test_album_operations_and_restrictions(self):
        with self.TestingSessionLocal() as db:
            vol = WindowsVolume(volume_serial_number="V-ALB", volume_label="Pendrive", last_known_drive_letter=str(self.drive_a))
            db.add(vol)
            db.commit()
            db.refresh(vol)

            sub_dir = self.drive_a / "Grecja"
            sub_dir.mkdir(parents=True, exist_ok=True)
            img_p = create_synthetic_jpeg(sub_dir / "IMG_20240812_100000.jpg")
            it = FileScannerService.process_file(img_p, vol)
            db.add(it)
            db.commit()
            it_id = it.id

            # 1-Click Auto-Organization without volume_id and folder_prefix must be rejected (Requirement 2e)
            r_fail = self.client.post("/actions/auto-group", data={"mode": "date", "scope": "filtered"})
            self.assertEqual(r_fail.status_code, 400)
            self.assertIn("wyłącznie po wybraniu konkretnego urządzenia", r_fail.text)

            # With volume_id and folder_prefix, 1-Click Auto-Organization succeeds
            r_ok = self.client.post(
                "/actions/auto-group",
                data={"mode": "folder", "scope": "filtered", "volume_id": str(vol.id), "folder_prefix": "Grecja/"},
            )
            self.assertEqual(r_ok.status_code, 200)

            r_date = self.client.post(
                "/actions/auto-group",
                data={"mode": "date", "scope": "filtered", "volume_id": str(vol.id), "folder_prefix": "Grecja/"},
            )
            self.assertEqual(r_date.status_code, 200)

            # Assign to new album and existing album via /actions/assign-group (Requirement 2c)
            r_assign_new = self.client.post(
                "/actions/assign-group",
                data={"group_name": "Nowy_Album_Test", "scope": "selected", "selected_ids": [it_id]},
            )
            self.assertEqual(r_assign_new.status_code, 200)
            created_grp = db.scalar(select(MediaGroup).where(MediaGroup.name == "Nowy_Album_Test"))
            self.assertIsNotNone(created_grp)

            r_assign_ex = self.client.post(
                "/actions/assign-group",
                data={"existing_group_id": str(created_grp.id), "scope": "selected", "selected_ids": [it_id]},
            )
            self.assertEqual(r_assign_ex.status_code, 200)

            # Error cases in assign-group
            self.assertIn("nie istnieje", self.client.post("/actions/assign-group", data={"existing_group_id": "99999"}).text)
            self.assertIn("Wybierz istniejący", self.client.post("/actions/assign-group", data={}).text)

            # Create subalbum, rename, reparent
            self.client.post("/actions/group-manage", data={"action": "create_sub", "target_group_id": str(created_grp.id), "new_name": "Podalbum1"})
            self.client.post("/actions/group-manage", data={"action": "rename", "target_group_id": str(created_grp.id), "new_name": "Zmieniony_Album"})
            self.client.post("/actions/group-manage", data={"action": "reparent", "target_group_id": str(created_grp.id), "new_parent_id": "0"})
            self.assertIn("własnym rodzicem", self.client.post("/actions/group-manage", data={"action": "reparent", "target_group_id": str(created_grp.id), "new_parent_id": str(created_grp.id)}).text)

            # Create two albums A and B, assign photo to A, transfer to B with deleting A (Requirement 2g)
            alb_a = MediaGroup(name="Album_Stary")
            alb_b = MediaGroup(name="Album_Docelowy")
            db.add_all([alb_a, alb_b])
            db.commit()
            db.refresh(alb_a)
            db.refresh(alb_b)
            alb_a_id = alb_a.id
            alb_b_id = alb_b.id

            it_db = db.get(MediaItem, it_id)
            it_db.group_id = alb_a_id
            db.commit()

            r_transfer = self.client.post(
                "/actions/group-manage",
                data={
                    "action": "transfer_and_delete",
                    "target_group_id": str(alb_a_id),
                    "destination_group_id": str(alb_b_id),
                },
            )
            self.assertEqual(r_transfer.status_code, 200)
            db.expire_all()
            self.assertIsNone(db.get(MediaGroup, alb_a_id))
            self.assertEqual(db.get(MediaItem, it_id).group_id, alb_b_id)

            # Consolidate group onto Drive B
            r_cons = self.client.post(
                "/actions/consolidate",
                data={"group_id": str(alb_b_id), "target_volume_id": str(vol.id)},
            )
            self.assertEqual(r_cons.status_code, 200)

            # Fast delete of Album B without deleting photos (Requirement 2f)
            r_del = self.client.post(
                "/actions/group-manage",
                data={"action": "delete", "target_group_id": str(alb_b_id)},
            )
            self.assertEqual(r_del.status_code, 200)
            db.expire_all()
            self.assertIsNone(db.get(MediaGroup, alb_b_id))
            self.assertIsNone(db.get(MediaItem, it_id).group_id)

            # Additional edge cases in group-manage, delete_album_fast, transfer_album_and_delete_source
            svc = StorageAutomationService(db)
            with self.assertRaises(ValueError):
                svc.delete_album_fast(999999)
            with self.assertRaises(ValueError):
                svc.transfer_album_and_delete_source(1, 1)
            with self.assertRaises(ValueError):
                svc.transfer_album_and_delete_source(999998, 999999)

            # Transfer parent album into its own child subalbum (detaches child first)
            p_grp = MediaGroup(name="Rodzic")
            db.add(p_grp)
            db.commit()
            db.refresh(p_grp)
            c_grp = MediaGroup(name="Dziecko", parent_id=p_grp.id)
            db.add(c_grp)
            db.commit()
            db.refresh(c_grp)
            svc.transfer_album_and_delete_source(p_grp.id, c_grp.id)

            self.client.post("/actions/group-manage", data={"action": "unknown", "target_group_id": str(c_grp.id)})
            self.client.post("/actions/group-manage", data={"action": "transfer_and_delete", "target_group_id": str(c_grp.id)})
            self.client.post("/actions/group-manage", data={"action": "delete", "target_group_id": "999999"})
            self.client.post("/actions/consolidate", data={"group_id": str(c_grp.id), "target_volume_id": "999999"})

    # =========================================================================
    # 4. MULTIMEDIA: DATE CHANGE, CLEAR ANOMALY WITHOUT DATE CHANGE, ROTATION
    # =========================================================================
    def test_multimedia_date_anomaly_and_rotation_persistence(self):
        with self.TestingSessionLocal() as db:
            vol = WindowsVolume(volume_serial_number="V-MM", volume_label="Dysk_MM", last_known_drive_letter=str(self.drive_a))
            grp = MediaGroup(name="Album_Daty")
            db.add_all([vol, grp])
            db.commit()
            db.refresh(vol)
            db.refresh(grp)

            p1 = create_synthetic_jpeg(self.drive_a / "IMG_20200101_120000.jpg")
            p2 = create_synthetic_jpeg(self.drive_a / "IMG_20200102_120000.jpg")
            it1 = FileScannerService.process_file(p1, vol)
            it2 = FileScannerService.process_file(p2, vol)
            it1.group_id = grp.id
            it2.group_id = grp.id
            db.add_all([it1, it2])
            db.commit()
            it1_id, it2_id, grp_id = it1.id, it2.id, grp.id

            # Invalid date format check
            r_bad_dt = self.client.post("/actions/change-date", data={"new_date": "nie-data", "scope": "selected", "selected_ids": [it1_id]})
            self.assertEqual(r_bad_dt.status_code, 400)

            # 4a-1: Change date of it1 -> updates effective_date and clears has_date_anomaly
            r_chg = self.client.post(
                "/actions/change-date",
                data={"scope": "selected", "selected_ids": [it1_id], "new_date": "2022-05-15 18:30"},
            )
            self.assertEqual(r_chg.status_code, 200)
            db.expire_all()
            it1_reloaded = db.get(MediaItem, it1_id)
            self.assertEqual(it1_reloaded.effective_date, datetime.datetime(2022, 5, 15, 18, 30))
            self.assertFalse(it1_reloaded.has_date_anomaly)

            # Change date by album_id
            r_chg_alb = self.client.post(
                "/actions/change-date",
                data={"album_id": str(grp_id), "new_date": "2022-06-01"},
            )
            self.assertEqual(r_chg_alb.status_code, 200)

            # 4a-2: Clear anomaly of it2 on demand WITHOUT changing its date (by media_id and by album_id)
            it2_re = db.get(MediaItem, it2_id)
            it2_re.has_date_anomaly = True
            db.commit()
            orig_dt = it2_re.effective_date
            r_clr = self.client.post("/actions/clear-date-anomaly", data={"media_id": it2_id})
            self.assertEqual(r_clr.status_code, 200)
            db.expire_all()
            it2_reloaded = db.get(MediaItem, it2_id)
            self.assertEqual(it2_reloaded.effective_date, orig_dt)
            self.assertFalse(it2_reloaded.has_date_anomaly)

            self.client.post("/actions/clear-date-anomaly", data={"album_id": str(grp_id)})

            # 4b & 4c: Rotate photo and verify persistence in SQLite
            r_rot = self.client.post(
                "/actions/rotate-media",
                data={"media_id": it1_id, "delta": 90},
                headers={"Accept": "application/json"},
            )
            self.assertEqual(r_rot.status_code, 200)
            self.assertEqual(r_rot.json()["rotation_angle"], 90)
            db.expire_all()
            self.assertEqual(db.get(MediaItem, it1_id).rotation_angle, 90)

            # Rotate -90 back to 0 via HTML response
            r_rot2 = self.client.post("/actions/rotate-media", data={"media_id": it1_id, "delta": -90})
            self.assertEqual(r_rot2.status_code, 200)
            db.expire_all()
            self.assertEqual(db.get(MediaItem, it1_id).rotation_angle, 0)

            # Error cases for rotate-media
            self.assertEqual(self.client.post("/actions/rotate-media", data={}).status_code, 400)
            self.assertEqual(self.client.post("/actions/rotate-media", data={"media_id": "nonexistent"}).status_code, 404)

            # Batch rotation by selected list, album_id, and filtered scope
            r_rot_list = self.client.post(
                "/actions/rotate-media",
                data={"scope": "selected", "selected_ids": [it1_id, it2_id], "delta": 90},
                headers={"Accept": "application/json"},
            )
            self.assertEqual(r_rot_list.status_code, 200)
            self.assertEqual(r_rot_list.json()["count"], 2)

            r_rot_alb = self.client.post(
                "/actions/rotate-media",
                data={"album_id": str(grp_id), "delta": 90},
            )
            self.assertEqual(r_rot_alb.status_code, 200)
            self.assertIn("Obrócono 2 zdjęć", r_rot_alb.text)

            r_rot_flt = self.client.post(
                "/actions/rotate-media",
                data={"scope": "filtered", "group_id": str(grp_id), "delta": 90},
                headers={"Accept": "application/json"},
            )
            self.assertEqual(r_rot_flt.status_code, 200)
            db.expire_all()
            self.assertEqual(db.get(MediaItem, it1_id).rotation_angle, 270)
            self.assertEqual(db.get(MediaItem, it2_id).rotation_angle, 270)

            # Test recalculate-date-anomalies endpoint
            r_recalc = self.client.post("/actions/recalculate-date-anomalies")
            self.assertEqual(r_recalc.status_code, 200)
            self.assertIn("Przeliczono źródła dat", r_recalc.text)

    # =========================================================================
    # 5. ONLINE FILE MOVES & REMOVAL OF OBSOLETE TODO FEATURE
    # =========================================================================
    def test_online_file_moves_and_no_todo_endpoints(self):
        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="VSN-OFF-A",
                volume_label="Dysk_Online",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            vol_b = WindowsVolume(
                volume_serial_number="VSN-OFF-B",
                volume_label="Dysk_Online_B",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
            )
            vol_c = WindowsVolume(
                volume_serial_number="VSN-OFF-C",
                volume_label="Dysk_Offline_C",
                last_known_drive_letter=str(self.drive_c_offline),
                is_online=False,
            )
            db.add_all([vol_a, vol_b, vol_c])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            db.refresh(vol_c)
            vol_a_id, vol_b_id, vol_c_id = vol_a.id, vol_b.id, vol_c.id

            f_a = create_synthetic_jpeg(self.drive_a / "online_move_test.jpg")
            it_a = FileScannerService.process_file(f_a, vol_a)
            db.add(it_a)
            db.commit()
            it_a_id = it_a.id

            # Online move via target_volume_id succeeds physically
            r_direct = self.client.post(
                "/actions/move-files",
                data={"scope": "selected", "selected_ids": [it_a_id], "target_volume_id": str(vol_b_id), "target_subfolder": "DirectSub"},
            )
            self.assertEqual(r_direct.status_code, 200)
            self.assertTrue((self.drive_b / "DirectSub" / "online_move_test.jpg").exists())

            # Attempting move to an offline target volume skips without modifying DB location
            r_mv = self.client.post(
                "/actions/move-files",
                data={
                    "scope": "selected",
                    "selected_ids": [it_a_id],
                    "target_volume_id": str(vol_c_id),
                    "target_subfolder": "Planowane_2025",
                },
            )
            self.assertEqual(r_mv.status_code, 200)
            self.assertIn("Pominięto", r_mv.text)
            db.expire_all()
            it_stay = db.get(MediaItem, it_a_id)
            self.assertEqual(it_stay.volume_id, vol_b_id)

            # Test destination_path branches in /actions/move-files (empty, online, offline drive)
            self.assertIn("Wybierz dysk", self.client.post("/actions/move-files", data={"scope": "selected", "selected_ids": [it_a_id]}).text)
            with patch.object(
                WindowsVolumeInspector,
                "get_volume_info",
                return_value=("VSN-OFF-A", "Dysk_Online", str(self.drive_a)),
            ), patch.object(
                WindowsVolumeInspector,
                "get_connected_vsn_map",
                return_value={
                    "VSN-OFF-A": (str(self.drive_a), "Dysk_Online"),
                    "VSN-OFF-B": (str(self.drive_b), "Dysk_Online_B"),
                },
            ):
                r_dest_ok = self.client.post("/actions/move-files", data={"scope": "selected", "selected_ids": [it_a_id], "destination_path": str(self.drive_a / "NowyFolder")})
                self.assertEqual(r_dest_ok.status_code, 200)
                self.assertTrue((self.drive_a / "NowyFolder" / "online_move_test.jpg").exists())
            r_dest_off = self.client.post("/actions/move-files", data={"scope": "selected", "selected_ids": [it_a_id], "destination_path": "Z:\\OfflineUnregistered"})
            self.assertEqual(r_dest_off.status_code, 200)
            self.assertIn("Offline", r_dest_off.text)

            # Verify TODO UI and endpoints are completely removed
            r_home = self.client.get("/")
            self.assertEqual(r_home.status_code, 200)
            self.assertNotIn("Lista TODO (Offline)", r_home.text)
            self.assertEqual(self.client.get("/actions/todo-panel").status_code, 404)
            self.assertEqual(self.client.post("/actions/todo-execute").status_code, 404)
            self.assertEqual(self.client.post("/actions/todo-rollback").status_code, 404)
            self.assertEqual(self.client.post("/actions/prepare-offline-report").status_code, 404)
            self.assertEqual(self.client.get("/actions/download-offline-report").status_code, 404)

    # =========================================================================
    # 6. STATUS, BACKGROUND PROCESS CONTROL, GRANULAR AI, SCAN ENDPOINT & LOGS
    # =========================================================================
    def test_status_process_control_scan_and_granular_ai(self):
        # Test POST /actions/scan on self.drive_a
        create_synthetic_jpeg(self.drive_a / "scan_img1.jpg", width=400, height=250, color=(230, 140, 30))
        create_synthetic_jpeg(self.drive_a / "scan_img2.jpg", width=250, height=400, color=(40, 180, 40))

        r_scan_bad = self.client.post("/actions/scan", data={"path": str(self.root / "NonExistentDir")})
        self.assertEqual(r_scan_bad.status_code, 400)

        r_scan1 = self.client.post("/actions/scan", data={"path": str(self.drive_a), "force_rescan": "false"})
        self.assertEqual(r_scan1.status_code, 200)
        self.assertIn("Dodano nowych: 2", r_scan1.text)

        # Second scan without force_rescan -> already_in_db = 2, plus repair missing thumbnail
        for tf in self.thumbs_dir.glob("*.webp"):
            tf.unlink()
        r_scan2 = self.client.post("/actions/scan", data={"path": str(self.drive_a), "force_rescan": "false"})
        self.assertEqual(r_scan2.status_code, 200)
        self.assertIn("Odświeżono miniatur: 2", r_scan2.text)

        # Force rescan
        r_scan3 = self.client.post("/actions/scan", data={"path": str(self.drive_a), "force_rescan": "true"})
        self.assertEqual(r_scan3.status_code, 200)

        with self.TestingSessionLocal() as db:
            vol = db.scalar(select(WindowsVolume))
            items = list(db.scalars(select(MediaItem)).all())
            self.assertEqual(len(items), 2)
            it1_id, it2_id = items[0].id, items[1].id

            # Trigger granular AI analysis synchronously (Requirement 3b) with fast mocked Haar cascade
            with patch.object(OnDemandObjectTagger, "open_session", return_value=None):
                for mode in ("tags", "vectors", "quality", "all"):
                    resp = self.client.post(
                        "/actions/analyze-all",
                        data={
                            "analysis_type": mode,
                            "target_drive_scope": f"volume:{vol.id}",
                            "run_sync": "true",
                        },
                    )
                    self.assertEqual(resp.status_code, 200)

                # Test /actions/ai-status and /actions/ai-stop
                self.assertEqual(self.client.get("/actions/ai-status").status_code, 200)
                self.assertEqual(self.client.post("/actions/ai-stop").status_code, 200)
                self.assertEqual(self.client.get("/actions/ai-status").status_code, 200)

                # Create an unfinished stopped job and test stop, resume, cancel via /actions/process-control
                job_id = ProcessManager.create_job(
                    job_type="ai_analysis",
                    title="Test Wznowienia AI",
                    item_ids=[it1_id, it2_id],
                    params={"analysis_type": "all"},
                    initial_status="STOPPED",
                    db=db,
                )
                r_status = self.client.get("/actions/status-panel")
                self.assertEqual(r_status.status_code, 200)
                self.assertIn("Test Wznowienia AI", r_status.text)

                self.client.post("/actions/process-control", data={"action": "stop", "job_id": str(job_id)})
                r_resume = self.client.post(
                    "/actions/process-control",
                    data={"action": "resume", "job_id": str(job_id), "run_sync": "true"},
                )
                self.assertEqual(r_resume.status_code, 200)
                db.expire_all()
                self.assertEqual(db.get(BackgroundJob, job_id).status, "COMPLETED")

                # Test ProcessManager.request_stop(None) and start_background_thread
                ProcessManager.request_stop(None)
                j_bg = ProcessManager.create_job("custom", "BgThread", [], db=db)
                ProcessManager.start_background_thread(j_bg, lambda jid: ProcessManager.finish_job(jid))

                # Test cancel action and resume on backup_dedup / move_to_trash / delete_permanently jobs
                j_bkp = ProcessManager.create_job("backup_dedup", "BkpJob", [], initial_status="STOPPED", db=db)
                self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(j_bkp)})
                self.client.post("/actions/process-control", data={"action": "cancel", "job_id": str(j_bkp)})

                j_tr = ProcessManager.create_job("move_to_trash", "TrJob", [it1_id], initial_status="STOPPED", db=db)
                self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(j_tr)})

                j_del = ProcessManager.create_job("delete_permanently", "DelJob", [it2_id], initial_status="STOPPED", db=db)
                self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(j_del)})

                j_empty = ProcessManager.create_job("custom_done", "EmptyJob", [], initial_status="STOPPED", db=db)
                self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(j_empty)})
                self.client.post("/actions/process-control", data={"action": "stop", "job_id": "999999"})

                # Test AI_BATCH_STATE['running'] = True branch in /actions/ai-status and /actions/analyze-all
                actions_module.AI_BATCH_STATE["running"] = True
                actions_module.AI_BATCH_STATE["total"] = 10
                actions_module.AI_BATCH_STATE["processed"] = 5
                try:
                    self.assertIn("Przetwarzanie AI", self.client.get("/actions/ai-status").text)
                    self.assertIn("Przetwarzanie AI", self.client.post("/actions/analyze-all", data={"analysis_type": "all"}).text)
                finally:
                    actions_module.AI_BATCH_STATE["running"] = False

                # Test _run_background_ai_batch stop_requested midway & error handling
                j_stop_ai = ProcessManager.create_job("ai_analysis", "StopAI", [it1_id], db=db)
                actions_module.AI_BATCH_STATE["stop_requested"] = True
                actions_module._run_background_ai_batch(j_stop_ai, [it1_id], analysis_type="all")

                j_err_ai = ProcessManager.create_job("ai_analysis", "ErrAI", [it1_id], db=db)
                with patch.object(MediaQualityAnalyzer, "analyze_image", side_effect=RuntimeError("AI_Crash")):
                    actions_module._run_background_ai_batch(j_err_ai, [it1_id], analysis_type="quality")

                # Test execute_scan exception branch and stop_requested branch
                with patch.object(WindowsVolumeInspector, "get_volume_info", side_effect=RuntimeError("ScanCrash")):
                    self.assertEqual(actions_module.execute_scan(str(self.drive_a))["status"], "error")

                # Test empty selection responses across endpoints
                self.client.post("/actions/analyze-all", data={"scope": "selected"})
                self.client.post("/actions/assign-group", data={"group_name": "PustyAlbum", "scope": "selected"})
                self.client.post("/actions/auto-group", data={"mode": "date", "volume_id": str(vol.id), "folder_prefix": "BrakPlikow/"})
                self.client.post("/actions/change-date", data={"new_date": "2023-01-01", "scope": "selected"})
                self.client.post("/actions/clear-date-anomaly", data={"scope": "selected"})
                self.client.post("/actions/export-json-disk", data={"scope": "selected"})
                self.client.post("/actions/move-files", data={"scope": "selected"})
                self.client.post("/actions/move-to-trash", data={"scope": "selected"})
                self.client.post("/actions/unindex", data={"scope": "selected"})
                self.client.post("/actions/delete-permanently", data={"scope": "selected"})
                self.client.post("/actions/rename-volume", data={"volume_id": "999999", "new_label": ""})

                # Test stop_requested inside _execute_move_to_trash_items and _execute_delete_permanently_items
                j_st_tr = ProcessManager.create_job("move_to_trash", "StTr", [it1_id], db=db)
                ProcessManager.request_stop(j_st_tr)
                actions_module._execute_move_to_trash_items(db, [it1_id], job_id=j_st_tr)

                j_st_dl = ProcessManager.create_job("delete_permanently", "StDl", [it1_id], db=db)
                ProcessManager.request_stop(j_st_dl)
                actions_module._execute_delete_permanently_items(db, [it1_id], job_id=j_st_dl)

            # Verify persistent operation log on disk (Requirement 3d) and file fallback in OperationLogger
            self.assertTrue(self.log_file.exists())
            log_text = self.log_file.read_text(encoding="utf-8")
            self.assertIn("AI_ANALYSIS", log_text)

            db.query(OperationLogEntry).delete()
            db.commit()
            file_logs = OperationLogger.get_recent_logs(db, limit=10, log_file=self.log_file)
            self.assertGreater(len(file_logs), 0)

    # =========================================================================
    # 7. AI TAGGER, QUALITY ANALYZER, SEMANTIC SIMILARITY & MEDIA ENDPOINTS
    # =========================================================================
    def test_tagger_quality_semantic_and_media_endpoints(self):
        # Test all ImageNet class index mappings in OnDemandObjectTagger
        cases = [
            (200, 0.5, "pies"),
            (200, 0.25, "zwierze"),
            (282, 0.5, "kot"),
            (290, 0.5, "zwierze"),
            (15, 0.5, "ptak"),
            (350, 0.5, "zwierze"),
            (511, 0.5, "samochod"),
            (670, 0.5, "rower_motocykl"),
            (497, 0.5, "architektura"),
            (950, 0.5, "jedzenie"),
            (970, 0.5, "gory_natura"),
            (975, 0.5, "morze_woda"),
            (1, 0.5, None),
        ]
        for idx, prob, expected_tag in cases:
            res = OnDemandObjectTagger._classify_imagenet_index(idx, prob)
            if expected_tag:
                self.assertTrue(any(t == expected_tag for t, _ in res), f"Failed for idx={idx}")
            else:
                self.assertEqual(res, [])

        # Test OnDemandObjectTagger context manager & open_session with temp labels & mocked ort
        labels_tmp = self.root / "imagenet_classes.json"
        labels_tmp.write_text(json.dumps(["tench", "goldfish"]), encoding="utf-8")
        fake_onnx = self.root / "mobilenetv2.onnx"
        fake_onnx.write_bytes(b"X" * 1_100_000)

        with patch("onnxruntime.InferenceSession", return_value=MagicMock()):
            t_ctx = OnDemandObjectTagger()
            t_ctx.labels_path = labels_tmp
            t_ctx.model_path = fake_onnx
            with t_ctx as active_t:
                self.assertEqual(active_t._get_label_by_index(0), "tench")
                active_t.labels = {"0": "tench_dict"}
                self.assertEqual(active_t._get_label_by_index(0), "tench_dict")
                active_t.labels = None
                self.assertEqual(active_t._get_label_by_index(5), "class_5")

        # Test scene color heuristics (square, B&W, dark, snow, sunset, water)
        sq_dark = create_synthetic_jpeg(self.drive_a / "sq_dark.jpg", width=200, height=200, sharp=False, color=(15, 15, 15))
        snow_bright = create_synthetic_jpeg(self.drive_a / "snow.jpg", width=1200, height=900, sharp=False, color=(240, 240, 240))
        sunset_img = create_synthetic_jpeg(self.drive_a / "sunset.jpg", width=200, height=200, sharp=False, color=(20, 100, 240))
        # Water image: white top half, blue bottom half
        water_arr = np.full((200, 200, 3), 240, dtype=np.uint8)
        water_arr[100:, :] = (220, 110, 20)  # BGR blue in bottom half
        water_img = self.drive_a / "water.jpg"
        cv2.imwrite(str(water_img), water_arr)

        tagger = OnDemandObjectTagger()
        mock_ort_sess = MagicMock()
        mock_in = MagicMock()
        mock_in.name = "input"
        mock_out = MagicMock()
        mock_out.name = "output"
        mock_ort_sess.get_inputs.return_value = [mock_in]
        mock_ort_sess.get_outputs.return_value = [mock_out]
        logits = np.zeros((1, 1000), dtype=np.float32)
        logits[0, 200] = 10.0  # dog class
        mock_ort_sess.run.return_value = [logits]

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(10, 10, 50, 50), (80, 10, 50, 50)]
        tagger.session = mock_ort_sess
        tagger.face_cascade = mock_cascade
        tags_out = tagger.tag_image(sq_dark)
        self.assertTrue(any(t == "pies" for t, _ in tags_out))
        self.assertTrue(any(t == "grupa osob" for t, _ in tags_out))

        # Single face -> portret
        mock_cascade.detectMultiScale.return_value = [(10, 10, 50, 50)]
        tags_portrait = tagger.tag_image(water_img)
        self.assertTrue(any(t == "portret" for t, _ in tags_portrait))
        self.assertTrue(any(t == "morze_woda" for t, _ in tags_portrait))

        tagger.tag_image(snow_bright)
        tagger.tag_image(sunset_img)
        self.assertEqual(tagger.tag_image(self.drive_a / "missing_tag.jpg"), [])
        bad_tag_f = self.drive_a / "bad_tag.jpg"
        bad_tag_f.write_bytes(b"NOT_AN_IMAGE")
        self.assertEqual(tagger.tag_image(bad_tag_f), [])
        tagger.close_session()

        # MediaQualityAnalyzer edge sizes (>1024px and <256px and flat and corrupt)
        MediaQualityAnalyzer.analyze_image(snow_bright)
        small_img = create_synthetic_jpeg(self.drive_a / "tiny.jpg", width=100, height=100, sharp=False)
        MediaQualityAnalyzer.analyze_image(small_img)
        MediaQualityAnalyzer.analyze_image(self.drive_a / "missing_q.jpg")
        MediaQualityAnalyzer.analyze_image(bad_tag_f)

        # OnDemandClipEmbedder edge cases
        self.assertEqual(OnDemandClipEmbedder.rank_similar(np.zeros(512), []), [])
        self.assertEqual(OnDemandClipEmbedder.rank_similar(np.zeros(512), [("id1", b"short")]), [])

        with self.TestingSessionLocal() as db:
            vol = WindowsVolume(volume_serial_number="V-SEM", volume_label="Dysk_Sem", last_known_drive_letter=str(self.drive_a))
            grp = MediaGroup(name="Pieski")
            db.add_all([vol, grp])
            db.commit()
            db.refresh(vol)
            db.refresh(grp)

            p1 = create_synthetic_jpeg(self.drive_a / "dog_1.jpg", color=(50, 100, 150))
            p2 = create_synthetic_jpeg(self.drive_a / "dog_2.jpg", color=(55, 105, 155))
            # Create a .tiff image to test on-the-fly TIFF->JPEG conversion in /{media_id}/raw
            tiff_p = self.drive_a / "sample.tiff"
            Image.new("RGB", (120, 90), color=(100, 150, 200)).save(tiff_p, format="TIFF")
            mov_p = self.drive_a / "sample_clip.mov"
            mov_p.write_bytes(b"FAKE_MOV_CONTENT" * 100)

            it1 = FileScannerService.process_file(p1, vol)
            it2 = FileScannerService.process_file(p2, vol)
            it_tiff = FileScannerService.process_file(tiff_p, vol)
            it_mov = FileScannerService.process_file(mov_p, vol)
            it1.group_id = grp.id
            it2.effective_date = it1.effective_date - datetime.timedelta(hours=2)
            # Clear it2.embedding to test in-RAM vector fallback in semantic.py
            it2.embedding = None
            db.add_all([it1, it2, it_tiff, it_mov])
            db.flush()
            db.add(MediaTag(media_id=it1.id, tag="pies", confidence=0.9, source_model="test"))
            db.add(MediaTag(media_id=it2.id, tag="pies", confidence=0.85, source_model="test"))
            db.commit()
            it1_id, it2_id, tiff_id, mov_id, grp_id, vol_id = it1.id, it2.id, it_tiff.id, it_mov.id, grp.id, vol.id
            it1_thumb = it1.thumbnail_relative_path

            # Test all 6 similarity modes + invalid mode + missing media_id
            for mode in list(semantic_module.SIMILARITY_MODES.keys()) + ["invalid_mode"]:
                r_sim = self.client.get(f"/semantic/similar/{it1_id}?mode={mode}")
                self.assertEqual(r_sim.status_code, 200)
            self.assertEqual(self.client.get("/semantic/similar/nonexistent_id").status_code, 200)

            # Test semantic text search (normal and empty query)
            r_q = self.client.get("/semantic/search?q=pies")
            self.assertEqual(r_q.status_code, 200)
            self.assertIn("dog_1.jpg", r_q.text)
            self.assertEqual(self.client.get("/semantic/search?q=   ").status_code, 200)

            # Test media detail, raw, thumbnails (including lazy regeneration), filters
            self.assertEqual(self.client.get("/").status_code, 200)
            self.assertEqual(self.client.get("/media/sidebar").status_code, 200)
            for vm in ("grid", "compact", "table"):
                for sb in ("date_desc", "date_asc", "size_desc", "size_asc", "name_asc", "blur_asc"):
                    r_s = self.client.get(
                        f"/media/search?view_mode={vm}&sort_by={sb}&q=dog&media_type=image"
                        f"&volume_id={vol_id}&group_id={grp_id}&tag=pies&year={it1.effective_date.year}"
                    )
                    self.assertEqual(r_s.status_code, 200)

            # Test boolean filter flags on /media/search
            self.assertEqual(
                self.client.get(
                    "/media/search?anomaly_only=true&blurry_only=true&duplicates_only=true"
                    "&untagged_only=true&ungrouped_only=true&unbacked_only=true&trashed_only=true"
                ).status_code,
                200,
            )

            self.assertEqual(self.client.get(f"/media/{it1_id}/detail").status_code, 200)
            self.assertEqual(self.client.get("/media/nonexistent/detail").status_code, 404)
            self.assertEqual(self.client.get(f"/media/{it1_id}/raw").status_code, 200)
            self.assertEqual(self.client.get(f"/media/{tiff_id}/raw").status_code, 200)
            self.assertEqual(self.client.get("/media/nonexistent/raw").status_code, 404)

            # Test .mov video transcoding branch in /{media_id}/raw (both uncached ffmpeg call and cached)
            cached_mp4 = self.thumbs_dir / f"{mov_id}_stream.mp4"
            def fake_ffmpeg_run(*args, **kwargs):
                cached_mp4.write_bytes(b"MP4_STREAM_DATA" * 200)
            with patch("subprocess.run", side_effect=fake_ffmpeg_run):
                self.assertEqual(self.client.get(f"/media/{mov_id}/raw").status_code, 200)

            # Test offline raw fallback to WebP thumbnail and 404 when thumbnail is also missing
            it_off = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_id,
                relative_path="missing_on_disk.jpg",
                file_name="missing_on_disk.jpg",
                file_size_bytes=123,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="abc",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2023, 1, 1),
                fs_mtime=datetime.datetime(2023, 1, 1),
                effective_date=datetime.datetime(2023, 1, 1),
                thumbnail_relative_path=it1_thumb,
            )
            db.add(it_off)
            db.commit()
            it_off_id = it_off.id
            self.assertEqual(self.client.get(f"/media/{it_off_id}/raw").status_code, 200)

            # Delete cached thumbnail and request /media/thumbnails/... to trigger lazy regeneration
            (self.thumbs_dir / it1_thumb).unlink()
            self.assertEqual(self.client.get(f"/media/{it_off_id}/raw").status_code, 404)
            self.assertEqual(self.client.get(f"/media/thumbnails/{it1_thumb}").status_code, 200)
            self.assertEqual(self.client.get("/media/thumbnails/nonexistent.webp").status_code, 404)

            # Export JSON
            r_exp = self.client.post("/actions/export-json-disk", data={"scope": "filtered"})
            self.assertEqual(r_exp.status_code, 200)

            # Rename volume
            r_ren = self.client.post("/actions/rename-volume", data={"volume_id": str(vol_id), "new_label": "Nowa_Nazwa"})
            self.assertEqual(r_ren.status_code, 200)

            # Move group to trash -> restore group -> move selected to trash -> restore selected (with collision) -> empty trash
            self.client.post("/actions/move-to-trash", data={"trash_group_id": str(grp_id)})
            self.client.post("/actions/restore-from-trash", data={"restore_mode": "group", "restore_group_id": str(grp_id)})
            self.client.post("/actions/move-to-trash", data={"scope": "selected", "selected_ids": [it1_id]})
            # Create a file at p1 so restore triggers _restored_1 collision handling
            create_synthetic_jpeg(p1)
            self.client.post("/actions/restore-from-trash", data={"restore_mode": "selected", "selected_ids": [it1_id]})
            self.client.post("/actions/move-to-trash", data={"scope": "selected", "selected_ids": [it1_id]})
            self.client.post("/actions/empty-trash", data={})

            # Permanent delete (by group_id via Two-Phase Commit and direct bypass blocked) & unindex
            r_grp_direct = self.client.post("/actions/delete-permanently", data={"delete_group_id": str(grp_id)})
            self.assertIn("wymagany jest ważny token potwierdzenia", r_grp_direct.text)
            self._delete_permanently_two_phase({"delete_group_id": str(grp_id)})
            self._delete_permanently_two_phase({"scope": "selected", "selected_ids": [it2_id]})
            self.client.post("/actions/unindex", data={"scope": "selected", "selected_ids": [tiff_id]})
            self.client.post("/actions/unindex", data={"unindex_volume_id": str(vol_id)})

    # =========================================================================
    # 8. SQLITE BACKUP MANAGER, SCHEMA MIGRATIONS & GOOGLE DRIVE CLOUD SYNC
    # =========================================================================
    def test_sqlite_backup_migrations_and_google_drive_verification(self):
        # Test get_db generator
        gen = get_db()
        s_db = next(gen)
        self.assertIsNotNone(s_db)
        gen.close()

        # Test ensure_schema_migrations on non-existent DB and on legacy minimal tables
        ensure_schema_migrations(self.root / "does_not_exist.db")
        legacy_db = self.root / "legacy.db"
        conn = sqlite3.connect(legacy_db.as_posix())
        conn.execute("CREATE TABLE media_groups (id INTEGER PRIMARY KEY, name TEXT);")
        conn.execute("CREATE TABLE media_items (id TEXT PRIMARY KEY, file_name TEXT);")
        conn.commit()
        conn.close()
        ensure_schema_migrations(legacy_db)

        mgr = BackupManager(db_path=self.db_path, backup_dir=self.backup_dir, max_retained_backups=2)
        self.assertTrue(mgr.check_integrity())
        mgr.create_backup("t1")
        mgr.create_backup("t2")
        b3 = mgr.create_backup("t3")
        self.assertTrue(b3.exists())
        self.assertLessEqual(len(list(self.backup_dir.glob("media_vault_*.db*"))), 2)

        # Test create_backup on minimal legacy db without media_embeddings
        mgr_legacy = BackupManager(db_path=legacy_db, backup_dir=self.backup_dir)
        b_leg = mgr_legacy.create_backup("leg_noclip", exclude_embeddings=True)
        self.assertTrue(b_leg.exists())

        # Test restore_from_backup with corrupted .gz file
        broken_gz = self.root / "corrupted_backup.db.gz"
        broken_gz.write_bytes(b"INVALID_GZIP_STREAM")
        with self.assertRaises(RuntimeError) as ctx:
            mgr.restore_from_backup(broken_gz)
        self.assertIn("Błąd dekompresji", str(ctx.exception))

        # Test GoogleDriveManager authentication, folder creation, MD5 verification, and purge_local
        sample = create_synthetic_jpeg(self.drive_a / "gdrive_test.jpg")
        md5_hex = GoogleDriveManager.compute_local_md5(sample)
        self.assertEqual(len(md5_hex), 32)

        g_mgr = GoogleDriveManager()
        g_mgr.CREDENTIALS_FILE = self.root / "missing_creds.json"
        with self.assertRaises(FileNotFoundError):
            g_mgr.authenticate()

        # Test authenticate() with mocked Credentials and build
        g_mgr.CREDENTIALS_FILE = self.root / "creds.json"
        g_mgr.TOKEN_FILE = self.root / "token.json"
        g_mgr.CREDENTIALS_FILE.write_text("{}", encoding="utf-8")
        g_mgr.TOKEN_FILE.write_text("{}", encoding="utf-8")
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_svc = MagicMock()
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=mock_creds), patch(
            "googleapiclient.discovery.build", return_value=mock_svc
        ):
            g_mgr.authenticate()
            self.assertIsNotNone(g_mgr.service)

        mock_svc.files().list().execute.return_value = {"files": []}
        mock_svc.files().create().execute.return_value = {"id": "created_fld"}
        self.assertEqual(g_mgr.get_or_create_folder("NowyFolder"), "created_fld")

        mock_svc.files().list().execute.return_value = {"files": [{"id": "fld1", "name": "DomoweArchiwumFoto"}]}
        self.assertEqual(g_mgr.get_or_create_folder("DomoweArchiwumFoto"), "fld1")
        self.assertEqual(len(g_mgr.list_folders()), 1)
        self.assertEqual(len(g_mgr.list_files_in_folder("fld1")), 1)

        # Mock next_chunk() for upload_and_verify
        mock_req = MagicMock()
        mock_req.next_chunk.return_value = (
            None,
            {
                "id": "cloud123",
                "name": sample.name,
                "size": str(sample.stat().st_size),
                "md5Checksum": md5_hex,
                "webViewLink": "https://drive.google.com/file/d/cloud123",
            },
        )
        mock_svc.files().create.return_value = mock_req
        with patch("googleapiclient.http.MediaFileUpload", return_value=MagicMock()):
            up_res = g_mgr.upload_and_verify(sample, "fld1", purge_local_after_verify=False)
            self.assertTrue(up_res["verified"])

            # Checksum mismatch must raise RuntimeError and NOT delete local file
            mock_req.next_chunk.return_value = (
                None,
                {"id": "bad", "name": sample.name, "size": str(sample.stat().st_size), "md5Checksum": "0000"},
            )
            with self.assertRaises(RuntimeError):
                g_mgr.upload_and_verify(sample, "fld1", purge_local_after_verify=True)
            self.assertTrue(sample.exists())

            # Test purge_local_after_verify=True when verified
            sample_purge = create_synthetic_jpeg(self.drive_a / "gdrive_purge.jpg")
            purge_md5 = GoogleDriveManager.compute_local_md5(sample_purge)
            mock_req.next_chunk.return_value = (
                None,
                {
                    "id": "cloud_p",
                    "name": sample_purge.name,
                    "size": str(sample_purge.stat().st_size),
                    "md5Checksum": purge_md5,
                },
            )
            up_purge = g_mgr.upload_and_verify(sample_purge, "fld1", purge_local_after_verify=True)
            self.assertTrue(up_purge["purged_local"])
            self.assertFalse(sample_purge.exists())

        # Test /actions/gdrive-panel and /actions/gdrive-sync endpoints (both db_backup and media modes)
        with self.TestingSessionLocal() as db:
            vol = WindowsVolume(volume_serial_number="V-GD", volume_label="Dysk_GD", last_known_drive_letter=str(self.drive_a))
            db.add(vol)
            db.commit()
            db.refresh(vol)
            it_gd = FileScannerService.process_file(sample, vol)
            db.add(it_gd)
            db.commit()
            it_gd_id = it_gd.id

        mock_req.next_chunk.return_value = (
            None,
            {
                "id": "cloud123",
                "name": sample.name,
                "size": str(sample.stat().st_size),
                "md5Checksum": md5_hex,
                "webViewLink": "https://drive.google.com/file/d/cloud123",
            },
        )
        with patch("app.cloud.google_drive.GoogleDriveManager", return_value=g_mgr), patch(
            "googleapiclient.http.MediaFileUpload", return_value=MagicMock()
        ):
            self.assertEqual(self.client.get("/actions/gdrive-panel").status_code, 200)
            with patch.object(g_mgr, "upload_and_verify", return_value={"id": "c1", "name": "b.db", "size_bytes": 100, "md5": "abc", "verified": True, "purged_local": False}):
                r_sync_db = self.client.post("/actions/gdrive-sync", data={"mode": "db_backup", "folder_id": "fld1"})
                self.assertEqual(r_sync_db.status_code, 200)
                r_sync_media = self.client.post(
                    "/actions/gdrive-sync",
                    data={"mode": "media", "new_folder_name": "Chmura_Test", "scope": "selected", "selected_ids": [it_gd_id], "purge_local": "true"},
                )
                self.assertEqual(r_sync_media.status_code, 200)

        # Test gdrive-panel and gdrive-sync error handling when credentials fail
        with patch("app.cloud.google_drive.GoogleDriveManager", side_effect=RuntimeError("Brak OAuth")):
            self.assertEqual(self.client.get("/actions/gdrive-panel").status_code, 200)
            self.assertEqual(self.client.post("/actions/gdrive-sync", data={"mode": "db_backup"}).status_code, 200)

        # Test move-files by destination_path (both online and offline drive paths), group trash/restore, and TIFF raw conversion
        with self.TestingSessionLocal() as db:
            vol_m = WindowsVolume(volume_serial_number="V-MV-99", volume_label="Dysk_MV", last_known_drive_letter=str(self.drive_a), is_online=True)
            vol_z = WindowsVolume(volume_serial_number="V-MV-Z", volume_label="Dysk_Z_Off", last_known_drive_letter="Z:\\", is_online=False)
            db.add_all([vol_m, vol_z])
            db.commit()
            db.refresh(vol_m)
            db.refresh(vol_z)
            vol_m_id = vol_m.id

            p_mv = create_synthetic_jpeg(self.drive_a / "SubFolder" / "mv_item.jpg")
            it_mv = FileScannerService.process_file(p_mv, vol_m)
            # Also create a .tiff image for on-the-fly JPEG conversion in /media/{id}/raw
            tiff_path = self.drive_a / "SubFolder" / "sample_image.tiff"
            Image.new("RGBA", (64, 64), (100, 150, 200, 255)).save(tiff_path, format="TIFF")
            it_tiff = FileScannerService.process_file(tiff_path, vol_m)
            grp_parent = MediaGroup(name="Rodzic_Kosz")
            db.add_all([it_mv, it_tiff, grp_parent])
            db.commit()
            db.refresh(grp_parent)
            grp_child = MediaGroup(name="Dziecko_Kosz", parent_id=grp_parent.id)
            db.add(grp_child)
            db.commit()
            db.refresh(grp_child)
            it_mv.group_id = grp_child.id
            db.commit()
            mv_id, tiff_id, child_gid = it_mv.id, it_tiff.id, grp_child.id

        # Search with duplicates_only and volume_id (subfolders aggregation)
        self.assertEqual(self.client.get(f"/media/search?duplicates_only=true&volume_id={vol_m_id}").status_code, 200)

        # Raw TIFF on-the-fly conversion to JPEG
        r_tiff = self.client.get(f"/media/{tiff_id}/raw")
        self.assertEqual(r_tiff.status_code, 200)
        self.assertEqual(r_tiff.headers["content-type"], "image/jpeg")

        # Move group to trash and restore group hierarchy from trash
        self.assertEqual(self.client.post("/actions/move-to-trash", data={"trash_group_id": str(child_gid)}).status_code, 200)
        self.assertEqual(
            self.client.post("/actions/restore-from-trash", data={"restore_mode": "group", "restore_group_id": str(child_gid)}).status_code,
            200,
        )

        # Move by destination_path (online folder)
        dest_online = self.drive_b / "CustomTarget"
        with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("V-MV-NEW", "NowyDysk", str(self.drive_b))):
            r_mv_on = self.client.post(
                "/actions/move-files",
                data={"destination_path": str(dest_online), "scope": "selected", "selected_ids": [mv_id]},
            )
            self.assertEqual(r_mv_on.status_code, 200)

        # Move by destination_path (offline registered Z:\ drive)
        r_mv_off = self.client.post(
            "/actions/move-files",
            data={"destination_path": "Z:\\Archiwum\\2024", "scope": "selected", "selected_ids": [mv_id]},
        )
        self.assertEqual(r_mv_off.status_code, 200)
        # Move by destination_path (unregistered offline X:\ drive)
        r_mv_unreg = self.client.post(
            "/actions/move-files",
            data={"destination_path": "X:\\Nieznany", "scope": "selected", "selected_ids": [mv_id]},
        )
        self.assertEqual(r_mv_unreg.status_code, 200)

        # BackupManager edge cases (non-existent db_path, failed integrity check)
        bm_missing = BackupManager(db_path=self.root / "no_such_db.sqlite", backup_dir=self.backup_dir)
        self.assertTrue(bm_missing.check_integrity())
        self.assertEqual(bm_missing.create_backup("missing"), self.root / "no_such_db.sqlite")

        # GoogleDriveManager token refresh and OAuth flow branches, plus service=None auto-authenticate
        mock_expired = MagicMock()
        mock_expired.valid = False
        mock_expired.expired = True
        mock_expired.refresh_token = "ref_tok"
        mock_expired.to_json.return_value = "{}"
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=mock_expired), patch(
            "googleapiclient.discovery.build", return_value=mock_svc
        ):
            g_mgr.authenticate()
            mock_expired.refresh.assert_called_once()

        g_mgr.TOKEN_FILE.unlink()
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_expired
        with patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file", return_value=mock_flow), patch(
            "googleapiclient.discovery.build", return_value=mock_svc
        ):
            g_mgr.authenticate()

        # Test service=None lazy authenticate in list_folders, get_or_create_folder, list_files_in_folder, upload_and_verify
        def _fake_auth():
            g_mgr.service = mock_svc

        with patch.object(g_mgr, "authenticate", side_effect=_fake_auth):
            g_mgr.service = None
            g_mgr.list_folders()
            g_mgr.service = None
            g_mgr.get_or_create_folder("Test")
            g_mgr.service = None
            g_mgr.list_files_in_folder("fld1")
            g_mgr.service = None
            with self.assertRaises(FileNotFoundError):
                g_mgr.upload_and_verify(self.root / "nonexistent_upload.jpg", "fld1")

        # MediaQualityAnalyzer .heic and video empty frame branches
        heic_ok = self.root / "fake_valid.heic"
        Image.new("RGB", (64, 64), (80, 120, 160)).save(heic_ok, format="JPEG")
        c_h, b_h, s_h = MediaQualityAnalyzer.analyze_image(heic_ok)
        self.assertFalse(c_h)
        heic_bad = self.root / "fake_corrupt.heic"
        heic_bad.write_bytes(b"corrupt_heic_bytes")
        c_hb, _, _ = MediaQualityAnalyzer.analyze_image(heic_bad)
        self.assertTrue(c_hb)

        mock_vcap = MagicMock()
        mock_vcap.isOpened.return_value = True
        mock_vcap.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=mock_vcap):
            c_v, _, _ = MediaQualityAnalyzer.analyze_video(heic_ok)
            self.assertTrue(c_v)

        # OnDemandObjectTagger invalid labels JSON and invalid ONNX model handling
        bad_lbl = self.root / "bad_labels.json"
        bad_lbl.write_text("{invalid_json", encoding="utf-8")
        bad_onnx = self.root / "bad_model.onnx"
        bad_onnx.write_bytes(b"0" * 1_100_000)
        tagger_bad = OnDemandObjectTagger()
        tagger_bad.model_path = bad_onnx
        tagger_bad.labels_path = bad_lbl
        tagger_bad.open_session()
        self.assertIsNone(tagger_bad.session)
        tagger_bad.labels = 12345
        self.assertEqual(tagger_bad._get_label_by_index(5), "class_5")
        tagger_bad.close_session()

        # Test stopping _run_background_ai_batch at chunk start and mid-batch, and stopping execute_scan midway
        from app.routers.actions import _run_background_ai_batch, AI_BATCH_STATE, execute_scan
        j_ai1 = ProcessManager.create_job("ai_analysis", "AI Stop Chunk", [mv_id])
        with patch.object(ProcessManager, "is_stop_requested", return_value=True):
            _run_background_ai_batch(j_ai1, [mv_id], analysis_type="tags")

        j_ai2 = ProcessManager.create_job("ai_analysis", "AI Stop Mid", [mv_id])
        with patch.object(ProcessManager, "is_stop_requested", side_effect=[False, True]):
            _run_background_ai_batch(j_ai2, [mv_id], analysis_type="tags")
        AI_BATCH_STATE["stop_requested"] = False

        # Test _run_background_ai_batch with missing file, video tag, embedding error, and blurry item
        with self.TestingSessionLocal() as db:
            p_blur_ai = create_synthetic_jpeg(self.drive_a / "SubFolder" / "blur_ai.jpg", sharp=False)
            it_blur_ai = FileScannerService.process_file(p_blur_ai, vol_m)
            it_vid_ai = MediaItem(
                id="vid-ai-test-1",
                volume_id=vol_m_id,
                relative_path="SubFolder/vid_ai_unique.mp4",
                file_name="vid_ai.mp4",
                file_size_bytes=1024,
                media_type="video",
                mime_type="video/mp4",
                file_hash="h_vid_ai",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2023, 1, 1),
                fs_mtime=datetime.datetime(2023, 1, 1),
                effective_date=datetime.datetime(2023, 1, 1),
                thumbnail_relative_path=it_blur_ai.thumbnail_relative_path,
            )
            it_miss_ai = MediaItem(
                id="miss-ai-test-1",
                volume_id=vol_m_id,
                relative_path="SubFolder/no_such_file.jpg",
                file_name="no_such_file.jpg",
                file_size_bytes=1024,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="h_miss_ai",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2023, 1, 1),
                fs_mtime=datetime.datetime(2023, 1, 1),
                effective_date=datetime.datetime(2023, 1, 1),
                thumbnail_relative_path="no_such_thumb.webp",
            )
            db.add_all([it_blur_ai, it_vid_ai, it_miss_ai])
            db.commit()
            blur_ai_id = it_blur_ai.id

        j_ai3 = ProcessManager.create_job("ai_analysis", "AI Full Branches", [blur_ai_id, "vid-ai-test-1", "miss-ai-test-1"])
        with patch.object(OnDemandClipEmbedder, "generate_image_embedding", side_effect=RuntimeError("VecErr")):
            _run_background_ai_batch(j_ai3, [blur_ai_id, "vid-ai-test-1", "miss-ai-test-1"], analysis_type="all")

        # Test async thread launch in /actions/analyze-all and /actions/process-control (invalid json & async resume)
        with patch.object(ProcessManager, "start_background_thread") as mock_bg:
            self.client.post("/actions/analyze-all", data={"scope": "selected", "selected_ids": [mv_id], "analysis_type": "tags"})
            mock_bg.assert_called_once()
            with self.TestingSessionLocal() as db:
                jb_bad = BackgroundJob(job_type="other", title="BadJSON", status="STOPPED", remaining_ids_json="{bad", params_json="{bad")
                jb_async = BackgroundJob(job_type="ai_analysis", title="AsyncResume", status="STOPPED", remaining_ids_json=json.dumps([mv_id]), params_json="{}")
                db.add_all([jb_bad, jb_async])
                db.commit()
                db.refresh(jb_bad)
                db.refresh(jb_async)
                jb_bad_id, jb_async_id = jb_bad.id, jb_async.id
            self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(jb_bad_id)})
            self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(jb_async_id)})

        # Test _run_background_ai_batch exception handler
        j_ai_fail = ProcessManager.create_job("ai_analysis", "AI Fail Job", [mv_id])
        with patch("app.routers.actions.get_db", side_effect=RuntimeError("DBBatchFail")):
            _run_background_ai_batch(j_ai_fail, [mv_id], analysis_type="tags")

        # Test date_discrepancy stop request in _run_background_ai_batch
        j_ano_stop = ProcessManager.create_job("ai_analysis", "Date Anomaly Stop", [mv_id])
        AI_BATCH_STATE["stop_requested"] = True
        _run_background_ai_batch(j_ano_stop, [mv_id], analysis_type="date_discrepancy")
        AI_BATCH_STATE["stop_requested"] = False

        # Test execute_scan with empty path and disk_usage exception
        res_empty = execute_scan("   ")
        self.assertEqual(res_empty["status"], "error")
        with patch("shutil.disk_usage", side_effect=Exception("DiskUsageFail")):
            execute_scan(str(self.drive_a))

        # Test WindowsVolumeInspector exception fallbacks
        with patch("ctypes.windll.kernel32.GetVolumeInformationW", side_effect=Exception("Kernel32Err")):
            vsn_f, lbl_f, root_f = WindowsVolumeInspector.get_volume_info(self.drive_a)
            self.assertEqual(lbl_f, "LOCAL_STORAGE")
        with patch("ctypes.windll.kernel32.GetLogicalDrives", side_effect=Exception("DrivesErr")):
            self.assertEqual(WindowsVolumeInspector.get_connected_vsn_map(), {})

        # Test render_template fallback in media.py
        from app.routers.media import render_template as media_render_template, templates as media_templates
        fake_req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
        with patch.object(media_templates, "TemplateResponse", side_effect=[TypeError("OldStarlette"), "ok"]):
            self.assertEqual(media_render_template(fake_req, "partials/media_grid.html", {}), "ok")

        # Additional action branches: consolidate offline, rename-volume invalid, restore-from-trash all, empty-trash selected & trashed group, move-files existing volume, gdrive empty folders & sync error
        with self.TestingSessionLocal() as db:
            vol_z_obj = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "V-MV-Z"))
            vol_z_id = vol_z_obj.id
        self.client.post("/actions/consolidate", data={"group_id": str(child_gid), "target_volume_id": str(vol_z_id)})
        self.client.post("/actions/rename-volume", data={"volume_id": str(vol_m_id), "new_label": "   "})
        self.client.post("/actions/move-to-trash", data={"trash_group_id": str(child_gid)})
        self.client.post("/actions/restore-from-trash", data={"restore_mode": "all"})
        self.client.post("/actions/move-to-trash", data={"trash_group_id": str(child_gid)})
        self.client.post("/actions/empty-trash", data={"selected_ids": [mv_id]})
        self.client.post("/actions/empty-trash", data={})

        # Second move to already-registered V-MV-NEW volume
        with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("V-MV-NEW", "NowyDysk", str(self.drive_b))):
            self.client.post(
                "/actions/move-files",
                data={"destination_path": str(dest_online), "scope": "selected", "selected_ids": [tiff_id]},
            )

        # GDrive empty folders list and error fallback with folder_id
        with patch("app.cloud.google_drive.GoogleDriveManager", return_value=g_mgr):
            with patch.object(g_mgr, "list_folders", side_effect=[[], [{"id": "fld1", "name": "DomoweArchiwumFoto"}]]):
                self.client.get("/actions/gdrive-panel")
            with patch.object(g_mgr, "upload_and_verify", side_effect=RuntimeError("UploadBoom")):
                self.client.post("/actions/gdrive-sync", data={"mode": "db_backup", "folder_id": "fld1"})
            with patch.object(g_mgr, "upload_and_verify", return_value={"id": "c2", "name": "b.db", "size_bytes": 100, "md5": "abc", "verified": True, "purged_local": False}):
                self.client.post("/actions/gdrive-sync", data={"mode": "db_backup"})

        j_scan = ProcessManager.create_job(
            "scan",
            "Scan Stop",
            ["mv_item.jpg"],
            params={"path": str(self.drive_a), "force_rescan": False},
        )
        ProcessManager.request_stop(j_scan)
        with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("V-MV-99", "Dysk_MV", str(self.drive_a))):
            execute_scan(str(self.drive_a), force_rescan=True, job_id=j_scan)

        # Resume the stopped scan job via /actions/process-control
        with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("V-MV-99", "Dysk_MV", str(self.drive_a))):
            r_res_scan = self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(j_scan)})
            self.assertEqual(r_res_scan.status_code, 200)

        # Verify empty path guard in execute_scan and scan resume
        self.assertEqual(execute_scan("   ")["status"], "error")
        j_scan_empty = ProcessManager.create_job("scan", "Scan Empty Path", ["mv_item.jpg"], params={})
        self.client.post("/actions/process-control", data={"action": "resume", "job_id": str(j_scan_empty)})

        # Test execute_scan deduplicating duplicate relative_paths (e.g. Windows junctions / fallback collisions)
        from app.routers.actions import _commit_scan_batch, _compute_scan_relative_path
        dup_scan_dir = self.root / "DupScanDrive"
        (dup_scan_dir / "Dir1").mkdir(parents=True, exist_ok=True)
        (dup_scan_dir / "Dir2").mkdir(parents=True, exist_ok=True)
        create_synthetic_jpeg(dup_scan_dir / "Dir1" / "same_name.jpg", color=(10, 20, 30))
        create_synthetic_jpeg(dup_scan_dir / "Dir2" / "same_name.jpg", color=(40, 50, 60))
        with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("V-SCAN-DUP", "ScanDup", str(dup_scan_dir))), patch(
            "app.routers.actions._compute_scan_relative_path", return_value="virtual/colliding_rel.jpg"
        ):
            scan_dup_res = execute_scan(str(dup_scan_dir))
            self.assertEqual(scan_dup_res["status"], "ok")
            self.assertEqual(scan_dup_res["indexed"], 1)

        # Test _compute_scan_relative_path fallback chain
        p_outside = self.root / "OtherPlace" / "pic.jpg"
        self.assertEqual(
            _compute_scan_relative_path(p_outside, Path("C:/NonExistentRoot"), Path("C:/NonExistentRoot"), self.root),
            "OtherPlace/pic.jpg",
        )
        self.assertEqual(
            _compute_scan_relative_path(p_outside, Path("C:/NonExistentRoot"), Path("C:/NonExistentRoot"), Path("D:/AnotherRoot")),
            "pic.jpg",
        )

        # Test _commit_scan_batch conflict fallback when batch contains duplicate (volume_id, relative_path)
        with self.TestingSessionLocal() as db:
            vol_cb = WindowsVolume(volume_serial_number="V-CB-1", volume_label="CB_Vol", last_known_drive_letter=str(self.drive_a))
            db.add(vol_cb)
            db.commit()
            db.refresh(vol_cb)
            p_cb1 = create_synthetic_jpeg(self.drive_a / "cb1.jpg", color=(11, 22, 33))
            p_cb2 = create_synthetic_jpeg(self.drive_a / "cb2.jpg", color=(44, 55, 66))
            it_cb1_a = FileScannerService.process_file(p_cb1, vol_cb, rel_override="cb_dup.jpg")
            it_cb1_b = FileScannerService.process_file(p_cb1, vol_cb, rel_override="cb_dup.jpg")
            it_cb2 = FileScannerService.process_file(p_cb2, vol_cb, rel_override="cb_ok.jpg")
            added_cb = _commit_scan_batch(db, [it_cb1_a, it_cb1_b, it_cb2], vol_cb.id)
            self.assertEqual(added_cb, 2)

        # Test GoogleDriveManager.list_db_backups & download_and_verify_backup + BackupManager.restore_from_backup + /actions/gdrive-restore-db
        snapshot_path = BackupManager(db_path=self.db_path, backup_dir=self.root / "cloud_snaps").create_backup("cloud_snap", compress=False)
        snap_bytes = snapshot_path.read_bytes()
        snap_md5 = GoogleDriveManager.compute_local_md5(snapshot_path)

        mock_svc.files().list().execute.return_value = {
            "files": [
                {"id": "db_bkp_1", "name": "media_vault_cloud.db", "size": str(len(snap_bytes)), "md5Checksum": snap_md5, "createdTime": "2026-11-14T10:00:00Z"},
                {"id": "img_1", "name": "photo.jpg", "size": "123", "md5Checksum": "aaa"},
            ]
        }
        with patch.object(g_mgr, "authenticate", side_effect=_fake_auth):
            g_mgr.service = None
            db_bkps = g_mgr.list_db_backups(folder_id="fld1")
        self.assertEqual(len(db_bkps), 1)
        self.assertEqual(db_bkps[0]["id"], "db_bkp_1")

        # Mutate DB after snapshot so restore visibly brings data back
        with self.TestingSessionLocal() as db:
            db.add(WindowsVolume(volume_serial_number="V-AFTER-SNAP", volume_label="AfterSnap", last_known_drive_letter="W:\\"))
            db.commit()

        # Verify FileNotFoundError and corrupt DB error in restore_from_backup
        bm_restore = BackupManager(db_path=self.db_path, backup_dir=self.backup_dir)
        with self.assertRaises(FileNotFoundError):
            bm_restore.restore_from_backup(self.root / "missing_backup.db")

        # Check integrity failure when PRAGMA integrity_check != ['ok']
        with patch("sqlite3.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [("corrupt page",)]
            mock_conn.return_value.cursor.return_value = mock_cur
            with self.assertRaises(RuntimeError):
                bm_restore.check_integrity()

        bad_sqlite = self.root / "not_a_db.db"
        bad_sqlite.write_bytes(b"not sqlite content")
        with self.assertRaises(RuntimeError):
            bm_restore.restore_from_backup(bad_sqlite)

        empty_sqlite = self.root / "empty_no_tables.db"
        c_empty = sqlite3.connect(empty_sqlite.as_posix())
        c_empty.execute("CREATE TABLE other (id INTEGER);")
        c_empty.commit()
        c_empty.close()
        with self.assertRaises(RuntimeError):
            bm_restore.restore_from_backup(empty_sqlite)

        # Corrupted PRAGMA check on existing table
        corrupt_pragma_sqlite = self.root / "corrupt_pragma.db"
        c_corrupt = sqlite3.connect(corrupt_pragma_sqlite.as_posix())
        c_corrupt.execute("CREATE TABLE media_items (id TEXT);")
        c_corrupt.commit()
        c_corrupt.close()
        with patch("sqlite3.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [("corrupted structure",)]
            mock_conn.return_value.cursor.return_value = mock_cur
            with self.assertRaises(RuntimeError):
                bm_restore.restore_from_backup(corrupt_pragma_sqlite)

        # Mock MediaIoBaseDownload to write snapshot bytes cleanly
        class _FakeDownloader:
            def __init__(self, fh, _req):
                self.fh = fh

            def next_chunk(self):
                self.fh.write(snap_bytes)
                return None, True

        mock_svc.files().get().execute.return_value = {
            "id": "db_bkp_1",
            "name": "media_vault_cloud.db",
            "size": str(len(snap_bytes)),
            "md5Checksum": snap_md5,
            "createdTime": "2026-11-14T10:00:00Z",
        }
        dl_dest = self.root / "downloaded_restore.db"
        with patch("googleapiclient.http.MediaIoBaseDownload", _FakeDownloader):
            with patch.object(g_mgr, "authenticate", side_effect=_fake_auth):
                g_mgr.service = None
                dl_res = g_mgr.download_and_verify_backup("db_bkp_1", dl_dest)
            self.assertTrue(dl_res["verified"])

            # Checksum mismatch raises RuntimeError
            mock_svc.files().get().execute.return_value = {
                "id": "db_bkp_bad",
                "name": "corrupt.db",
                "size": str(len(snap_bytes)),
                "md5Checksum": "ffffffffffffffffffffffffffffffff",
            }
            with self.assertRaises(RuntimeError):
                g_mgr.download_and_verify_backup("db_bkp_bad", self.root / "bad_dl.db")

            # Restore mock metadata and invoke POST /actions/gdrive-restore-db
            mock_svc.files().get().execute.return_value = {
                "id": "db_bkp_1",
                "name": "media_vault_cloud.db",
                "size": str(len(snap_bytes)),
                "md5Checksum": snap_md5,
                "createdTime": "2026-11-14T10:00:00Z",
            }
            with patch("app.cloud.google_drive.GoogleDriveManager", return_value=g_mgr):
                r_rest = self.client.post("/actions/gdrive-restore-db", data={"file_id": "db_bkp_1", "folder_id": "fld1"})
                self.assertEqual(r_rest.status_code, 200)
                self.assertIn("Przywrócono bazę SQLite", r_rest.text)

        # Verify restored DB no longer contains V-AFTER-SNAP
        with self.TestingSessionLocal() as db:
            self.assertIsNone(db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "V-AFTER-SNAP")))

        # Test GDrive restore with .db.gz compressed file
        snap_bytes_gz = gzip.compress(snap_bytes)
        snap_gz_md5 = hashlib.md5(snap_bytes_gz).hexdigest()
        class _FakeGzDownloader:
            def __init__(self, fh, _req):
                self.fh = fh

            def next_chunk(self):
                self.fh.write(snap_bytes_gz)
                return None, True

        mock_svc.files().get().execute.return_value = {
            "id": "db_bkp_gz_1",
            "name": "media_vault_cloud_test.db.gz",
            "size": str(len(snap_bytes_gz)),
            "md5Checksum": snap_gz_md5,
            "createdTime": "2026-11-14T11:00:00Z",
        }
        with patch("googleapiclient.http.MediaIoBaseDownload", _FakeGzDownloader):
            with patch("app.cloud.google_drive.GoogleDriveManager", return_value=g_mgr):
                r_rest_gz = self.client.post("/actions/gdrive-restore-db", data={"file_id": "db_bkp_gz_1", "folder_id": "fld1"})
                self.assertEqual(r_rest_gz.status_code, 200)
                self.assertIn("Przywrócono bazę SQLite", r_rest_gz.text)

        # Error branch for /actions/gdrive-restore-db
        with patch("app.cloud.google_drive.GoogleDriveManager", side_effect=RuntimeError("CloudRestoreErr")):
            r_rest_err = self.client.post("/actions/gdrive-restore-db", data={"file_id": "db_bkp_1", "folder_id": "fld1"})
            self.assertEqual(r_rest_err.status_code, 200)
            self.assertIn("CloudRestoreErr", r_rest_err.text)

        # =========================================================================
        # REQ 1: Always keep 1 DB backup on Google Drive (prune older backups)
        # =========================================================================
        mock_svc.files().list().execute.return_value = {
            "files": [
                {"id": "db_latest", "name": "media_vault_latest.db"},
                {"id": "db_older_1", "name": "media_vault_older_1.db"},
                {"id": "db_older_2", "name": "media_vault_older_2.db"},
            ]
        }
        deleted_fids = []
        def _fake_delete(fileId):
            deleted_fids.append(fileId)
            class _FakeExec:
                def execute(self):
                    return {}
            return _FakeExec()

        mock_svc.files().delete.side_effect = _fake_delete

        # Prune keeping specific file ID
        deleted = g_mgr.prune_older_db_backups(folder_id="fld1", keep_file_id="db_latest")
        self.assertEqual(len(deleted), 2)
        self.assertIn("db_older_1", deleted_fids)
        self.assertIn("db_older_2", deleted_fids)

        # Prune keeping newest (first) when keep_file_id is None
        deleted_fids.clear()
        deleted2 = g_mgr.prune_older_db_backups(folder_id="fld1", keep_file_id=None)
        self.assertEqual(len(deleted2), 2)

        # Prune with deletion failure logs warning but does not raise
        def _failing_delete(fileId):
            class _FailExec:
                def execute(self):
                    raise RuntimeError("API quota delete error")
            return _FailExec()

        mock_svc.files().delete.side_effect = _failing_delete
        deleted_fail = g_mgr.prune_older_db_backups(folder_id="fld1", keep_file_id="db_latest")
        self.assertEqual(deleted_fail, [])

        # Restore normal delete side effect
        mock_svc.files().delete.side_effect = _fake_delete

        # Test prune_older_db_backups with service=None auto-authenticating
        with patch.object(g_mgr, "authenticate", side_effect=_fake_auth):
            g_mgr.service = None
            g_mgr.prune_older_db_backups(folder_id="fld1", keep_file_id="db_latest")

        # Test upload_and_verify with a .db file to trigger keep_only_one_db / is_db pruning
        db_upload_sample = self.root / "test_db_prune.db"
        db_upload_sample.write_bytes(b"dummy db content")
        mock_req.next_chunk.return_value = (
            None,
            {
                "id": "db_cloud_123",
                "name": db_upload_sample.name,
                "size": str(db_upload_sample.stat().st_size),
                "md5Checksum": GoogleDriveManager.compute_local_md5(db_upload_sample),
            },
        )
        with patch("googleapiclient.http.MediaFileUpload", return_value=MagicMock()), patch.object(
            g_mgr, "prune_older_db_backups", return_value=["old.db"]
        ) as mock_prune:
            up_db_res = g_mgr.upload_and_verify(db_upload_sample, "fld1", keep_only_one_db=True)
            self.assertTrue(up_db_res["verified"])
            self.assertEqual(up_db_res["pruned_backups"], ["old.db"])
            mock_prune.assert_called_once()

        # Test POST /actions/gdrive-sync in mode="db_backup" prunes older backups
        mock_upload_res = {
            "id": "db_new_upload",
            "name": "media_vault_gdrive_cloud_new.db",
            "size_bytes": 1024,
            "md5": "mocked_md5",
            "verified": True,
            "purged_local": False,
            "webViewLink": "https://drive.google.com/test_db",
        }
        mock_svc.files().list().execute.return_value = {
            "files": [
                {"id": "db_new_upload", "name": "media_vault_gdrive_cloud_new.db"},
                {"id": "db_prior", "name": "media_vault_prior.db"},
            ]
        }
        with patch("app.cloud.google_drive.GoogleDriveManager", return_value=g_mgr), patch.object(g_mgr, "upload_and_verify", return_value=mock_upload_res):
            r_sync_db = self.client.post("/actions/gdrive-sync", data={"mode": "db_backup", "folder_id": "fld1"})
            self.assertEqual(r_sync_db.status_code, 200)
            self.assertIn("Kopia zapasowa bazy SQLite", r_sync_db.text)
            self.assertIn("Usunięto starsze kopie bazy z Google Drive", r_sync_db.text)

            # Test POST /actions/gdrive-sync in mode="db_backup" with exclude_embeddings=True and compress=True
            r_sync_db_light = self.client.post(
                "/actions/gdrive-sync",
                data={"mode": "db_backup", "folder_id": "fld1", "exclude_embeddings": "true", "compress": "true"},
            )
            self.assertEqual(r_sync_db_light.status_code, 200)
            self.assertIn("lekki backup bez wektorów CLIP", r_sync_db_light.text)
            self.assertIn("skompresowany gzip", r_sync_db_light.text)

        # =========================================================================
        # REQ 2 & 3: DB backup on chosen medium & Restore from chosen medium
        # =========================================================================
        medium_dir = self.root / "medium_usb" / "daf_backup" / "db_backups"
        bm_medium = BackupManager(db_path=self.db_path, backup_dir=self.backup_dir)

        # 1. Direct BackupManager.create_backup to target_dir and list_backups_in_dir
        saved_on_medium = bm_medium.create_backup("test_medium", target_dir=medium_dir)
        self.assertTrue(saved_on_medium.exists())
        backups_list = BackupManager.list_backups_in_dir(medium_dir)
        self.assertGreaterEqual(len(backups_list), 1)
        self.assertEqual(backups_list[0]["name"], saved_on_medium.name)
        self.assertIn("size_mb", backups_list[0])

        # Test listing empty / missing dir returns []
        self.assertEqual(BackupManager.list_backups_in_dir(self.root / "nonexistent_dir"), [])

        # 2. Add an online WindowsVolume in DB for medium backup endpoint testing
        with self.TestingSessionLocal() as db:
            vol_medium = WindowsVolume(
                volume_serial_number="VSN-MED-USB1",
                volume_label="USB_BACKUP_DRIVE",
                file_system="NTFS",
                last_known_drive_letter=str(self.root / "medium_usb") + "\\",
                is_online=True,
                total_space_bytes=100 * 1024 * 1024 * 1024,
                free_space_bytes=50 * 1024 * 1024 * 1024,
            )
            db.add(vol_medium)
            db.commit()
            vol_medium_id = vol_medium.id

        # 3. POST /actions/backup-db-to-medium via volume_id
        r_bkp_med = self.client.post(
            "/actions/backup-db-to-medium",
            data={"target_volume_id": str(vol_medium_id), "folder_id": "fld1"}
        )
        self.assertEqual(r_bkp_med.status_code, 200)
        self.assertIn("Kopia zapasowa bazy SQLite", r_bkp_med.text)
        self.assertIn("została pomyślnie utworzona na nośniku", r_bkp_med.text)

        # 4. POST /actions/backup-db-to-medium via custom_target_path
        custom_out_dir = self.root / "custom_ext_drive"
        r_bkp_custom = self.client.post(
            "/actions/backup-db-to-medium",
            data={"custom_target_path": str(custom_out_dir), "folder_id": "fld1"}
        )
        self.assertEqual(r_bkp_custom.status_code, 200)
        self.assertIn("Kopia zapasowa bazy SQLite", r_bkp_custom.text)
        self.assertIn("została pomyślnie utworzona", r_bkp_custom.text)

        # Test custom_target_path with daf_backup in path
        daf_custom_dir = self.root / "ext_drive" / "daf_backup" / "db_backups"
        r_bkp_daf = self.client.post(
            "/actions/backup-db-to-medium",
            data={"custom_target_path": str(daf_custom_dir), "folder_id": "fld1"}
        )
        self.assertEqual(r_bkp_daf.status_code, 200)
        self.assertIn("Kopia zapasowa bazy SQLite", r_bkp_daf.text)
        self.assertIn("została pomyślnie utworzona", r_bkp_daf.text)

        # Test custom_target_path as root drive letter
        r_bkp_root = self.client.post(
            "/actions/backup-db-to-medium",
            data={"custom_target_path": str(self.root.anchor), "folder_id": "fld1"}
        )
        self.assertEqual(r_bkp_root.status_code, 200)

        # Test backup-db-to-medium with exclude_embeddings=True and compress=True
        r_bkp_med_light = self.client.post(
            "/actions/backup-db-to-medium",
            data={"target_volume_id": str(vol_medium_id), "folder_id": "fld1", "exclude_embeddings": "true", "compress": "true"},
        )
        self.assertEqual(r_bkp_med_light.status_code, 200)
        self.assertIn("lekki backup bez wektorów CLIP", r_bkp_med_light.text)
        self.assertIn("skompresowany gzip", r_bkp_med_light.text)

        # Test backup-db-to-medium with exclude_embeddings=False and compress=False
        r_bkp_med_raw = self.client.post(
            "/actions/backup-db-to-medium",
            data={"target_volume_id": str(vol_medium_id), "folder_id": "fld1", "exclude_embeddings": "false", "compress": "false"},
        )
        self.assertEqual(r_bkp_med_raw.status_code, 200)
        self.assertIn("pełny backup z wektorami CLIP", r_bkp_med_raw.text)

        # 5. POST /actions/backup-db-to-medium error branches (missing volume, offline volume, no inputs)
        r_bkp_err1 = self.client.post("/actions/backup-db-to-medium", data={"target_volume_id": "99999"})
        self.assertEqual(r_bkp_err1.status_code, 200)
        self.assertIn("Wybrany wolumen nie istnieje", r_bkp_err1.text)

        r_bkp_err2 = self.client.post("/actions/backup-db-to-medium", data={})
        self.assertEqual(r_bkp_err2.status_code, 200)
        self.assertIn("Wskaż podłączony nośnik", r_bkp_err2.text)

        with self.TestingSessionLocal() as db:
            vol_offline = WindowsVolume(
                volume_serial_number="VSN-MED-OFFLINE",
                volume_label="OFFLINE_DRIVE",
                file_system="FAT32",
                last_known_drive_letter="Z:\\NonExistentPath\\",
                is_online=False,
            )
            db.add(vol_offline)
            db.commit()
            vol_off_id = vol_offline.id

        r_bkp_err3 = self.client.post("/actions/backup-db-to-medium", data={"target_volume_id": str(vol_off_id)})
        self.assertEqual(r_bkp_err3.status_code, 200)
        self.assertIn("jest obecnie odłączony", r_bkp_err3.text)

        # POST /actions/backup-db-to-medium failure after job_id is created
        with patch.object(BackupManager, "create_backup", side_effect=RuntimeError("SimulatedDiskFullError")):
            r_bkp_err4 = self.client.post(
                "/actions/backup-db-to-medium",
                data={"target_volume_id": str(vol_medium_id), "folder_id": "fld1"}
            )
            self.assertEqual(r_bkp_err4.status_code, 200)
            self.assertIn("SimulatedDiskFullError", r_bkp_err4.text)

        # Direct create_backup with optimize_vacuum=True and exclude_embeddings=False and progress callback
        prog_calls = []
        def _test_cb(msg, d, t):
            prog_calls.append((msg, d, t))

        vac_snap = bm_medium.create_backup(
            "test_vacuum_true",
            target_dir=medium_dir,
            optimize_vacuum=True,
            exclude_embeddings=False,
            progress_callback=_test_cb,
        )
        self.assertTrue(vac_snap.exists())
        self.assertGreater(len(prog_calls), 0)

        # Direct create_backup with compress=False and progress callback
        raw_calls = []
        raw_snap = bm_medium.create_backup(
            "test_raw_prog",
            target_dir=medium_dir,
            compress=False,
            progress_callback=lambda msg, d, t: raw_calls.append((msg, d, t)),
        )
        self.assertTrue(raw_snap.exists())
        self.assertGreater(len(raw_calls), 0)

        # Test corrupt gzip restore
        bad_gz = self.root / "corrupt.db.gz"
        bad_gz.write_bytes(b"NOT_A_VALID_GZIP_FILE")
        with self.assertRaises(RuntimeError):
            bm_medium.restore_from_backup(bad_gz)

        # Test corrupt sqlite restore
        bad_db = self.root / "corrupt.db"
        bad_db.write_bytes(b"SQLite format 3\x00INVALID_DATA_PADDING_CORRUPT")
        with self.assertRaises(RuntimeError):
            bm_medium.restore_from_backup(bad_db)

        # Test /actions/backup-job-status endpoint
        # 1. Idle status (no active job)
        r_status_idle = self.client.get("/actions/backup-job-status")
        self.assertEqual(r_status_idle.status_code, 200)
        self.assertIn("backup-live-progress", r_status_idle.text)

        # 2. Running job status via ProcessManager
        dummy_jid = ProcessManager.create_job(
            job_type="DB_BACKUP_MEDIUM",
            title="Kopia testowa",
            item_ids=["snap"],
        )
        ProcessManager.update_progress(dummy_jid, processed_count=42, total_count=100, message="Testowy postęp 42%")
        r_status_run = self.client.get(f"/actions/backup-job-status?folder_id=fld1&medium_volume_id={vol_medium_id}")
        self.assertEqual(r_status_run.status_code, 200)
        self.assertIn("42%", r_status_run.text)
        self.assertIn("Testowy postęp 42%", r_status_run.text)

        # Cover ProcessManager.update_progress edge cases
        ProcessManager.update_progress(999999, processed_count=1) # non-existent job
        with self.TestingSessionLocal() as db:
            ProcessManager.update_progress(dummy_jid, processed_delta=1, remaining_ids=["rem1"], status="RUNNING", db=db)
            db.commit()

        # Cover get_active_backup_job fallback to DB
        ProcessManager._job_cache.clear()
        with self.TestingSessionLocal() as db:
            active_from_db = ProcessManager.get_active_backup_job(db)
            self.assertIsNotNone(active_from_db)
            self.assertEqual(active_from_db["id"], dummy_jid)

        ProcessManager.finish_job(dummy_jid, status="COMPLETED", message="Test zakończony")

        # After completion, fallback to DB when cache cleared returns None
        ProcessManager._job_cache.clear()
        with self.TestingSessionLocal() as db:
            self.assertIsNone(ProcessManager.get_active_backup_job(db))

        # Test backup error handling inside progress callback
        def _exploding_cb(msg, d, t):
            raise ValueError("ExplodingCallback")
        # Test get_active_backup_job with db=None (own_session)
        ProcessManager._job_cache.clear()
        self.assertIsNone(ProcessManager.get_active_backup_job())

        # Test get_active_backup_job exception branch
        with patch.object(Session, "scalar", side_effect=Exception("ScalarFail")):
            self.assertIsNone(ProcessManager.get_active_backup_job())

        # Test update_progress exception branch
        with patch.object(Session, "get", side_effect=Exception("GetFail")):
            ProcessManager.update_progress(dummy_jid, status="RUNNING")

        # Test OperationLogger edge cases (OSError file write, DB commit exception, get_recent_logs missing file)
        OperationLogger.log("LOG_TEST", "msg", log_file=Path("Z:\\NonExistent_ReadOnly_Path\\test.log"))
        with patch.object(Session, "commit", side_effect=Exception("CommitFail")):
            OperationLogger.log("LOG_TEST", "msg")

        with self.TestingSessionLocal() as empty_db:
            # Query with empty table to test missing file fallback
            with patch.object(Session, "scalars") as mock_sc:
                mock_sc.return_value.all.return_value = []
                logs_empty = OperationLogger.get_recent_logs(empty_db, log_file=self.root / "no_log_file.log")
                self.assertEqual(logs_empty, [])

        # Test create_backup when final_path already exists (overwrites)
        snap_safe = bm_medium.create_backup("test_safe_prog", target_dir=medium_dir, compress=False, progress_callback=_exploding_cb)
        self.assertTrue(snap_safe.exists())

        # 6. POST /actions/restore-db-from-medium
        # Make a snapshot of the current state, mutate DB, and restore from medium
        medium_snap = bm_medium.create_backup("medium_snap_to_restore", target_dir=medium_dir, compress=False)
        with self.TestingSessionLocal() as db:
            db.add(WindowsVolume(volume_serial_number="VSN-MUTATION-1", volume_label="MutationTemp", last_known_drive_letter="M:\\"))
            db.commit()

        # Restore from backup_file_path
        r_rest_med = self.client.post(
            "/actions/restore-db-from-medium",
            data={"backup_file_path": str(medium_snap)}
        )
        self.assertEqual(r_rest_med.status_code, 200)
        self.assertIn("Pomyślnie przywrócono bazę SQLite z nośnika lokalnego", r_rest_med.text)

        # Verify mutation was rolled back by restore
        with self.TestingSessionLocal() as db:
            self.assertIsNone(db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "VSN-MUTATION-1")))

        # Restore from target_volume_id and selected_backup_filename
        with self.TestingSessionLocal() as db:
            db.add(WindowsVolume(volume_serial_number="VSN-MUTATION-2", volume_label="MutationTemp2", last_known_drive_letter="N:\\"))
            db.commit()

        r_rest_med2 = self.client.post(
            "/actions/restore-db-from-medium",
            data={"target_volume_id": str(vol_medium_id), "selected_backup_filename": medium_snap.name}
        )
        self.assertEqual(r_rest_med2.status_code, 200)
        self.assertIn("Pomyślnie przywrócono bazę SQLite z nośnika lokalnego", r_rest_med2.text)

        with self.TestingSessionLocal() as db:
            self.assertIsNone(db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "VSN-MUTATION-2")))

        # Test direct create_backup with compress=True and exclude_embeddings=True, and list_backups_in_dir
        gz_snap = bm_medium.create_backup("medium_snap_gz", target_dir=medium_dir, compress=True, exclude_embeddings=True)
        self.assertTrue(gz_snap.name.endswith(".db.gz"))
        self.assertIn("_no_clip", gz_snap.name)
        med_backups_info = BackupManager.list_backups_in_dir(medium_dir)
        gz_info = next((b for b in med_backups_info if b["name"] == gz_snap.name), None)
        self.assertIsNotNone(gz_info)
        self.assertTrue(gz_info["is_compressed"])
        self.assertFalse(gz_info["has_clip"])

        # Mutate DB and restore from gz_snap
        with self.TestingSessionLocal() as db:
            db.add(WindowsVolume(volume_serial_number="VSN-MUTATION-GZ", volume_label="MutationTempGZ", last_known_drive_letter="P:\\"))
            db.commit()

        r_rest_gz_med = self.client.post(
            "/actions/restore-db-from-medium",
            data={"backup_file_path": str(gz_snap)}
        )
        self.assertEqual(r_rest_gz_med.status_code, 200)
        self.assertIn("Pomyślnie przywrócono bazę SQLite z nośnika lokalnego", r_rest_gz_med.text)
        with self.TestingSessionLocal() as db:
            self.assertIsNone(db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "VSN-MUTATION-GZ")))

        # Restore error branches (file not found, corrupted file, missing params)
        r_rest_err1 = self.client.post("/actions/restore-db-from-medium", data={})
        self.assertEqual(r_rest_err1.status_code, 200)
        self.assertIn("Wskaż plik kopii zapasowej", r_rest_err1.text)

        r_rest_err2 = self.client.post("/actions/restore-db-from-medium", data={"backup_file_path": str(self.root / "missing.db")})
        self.assertEqual(r_rest_err2.status_code, 200)
        self.assertIn("nie istnieje", r_rest_err2.text)

        # Restore error with invalid volume ID
        r_rest_bad_vol = self.client.post(
            "/actions/restore-db-from-medium",
            data={"target_volume_id": "99999", "selected_backup_filename": "dummy.db"}
        )
        self.assertEqual(r_rest_bad_vol.status_code, 200)
        self.assertIn("Wybrany wolumen nie istnieje", r_rest_bad_vol.text)

        # Restore unexpected exception error branch
        with patch.object(BackupManager, "restore_from_backup", side_effect=RuntimeError("MediumRestoreFail")):
            r_rest_fail = self.client.post(
                "/actions/restore-db-from-medium",
                data={"backup_file_path": str(medium_snap)}
            )
            self.assertEqual(r_rest_fail.status_code, 200)
            self.assertIn("MediumRestoreFail", r_rest_fail.text)

        # Test BackupManager list_backups_in_dir OSError on file stat
        mock_p = MagicMock()
        mock_p.name = "media_vault_test.db"
        mock_p.stat.side_effect = OSError("Disk read error")
        with patch.object(Path, "glob", return_value=[mock_p]):
            self.assertEqual(BackupManager.list_backups_in_dir(medium_dir), [])

        # Test BackupManager _rotate unlink OSError
        mock_rot = MagicMock()
        mock_rot.unlink.side_effect = OSError("Locked file")
        with patch.object(Path, "glob", return_value=[mock_rot, mock_rot, mock_rot]), patch("os.path.getmtime", return_value=12345):
            mgr_rotate = BackupManager(db_path=self.db_path, backup_dir=self.backup_dir, max_retained_backups=1)
            mgr_rotate._rotate()

        # Test restore_from_medium with SessionLocal error post-restore
        with patch("app.database.SessionLocal", side_effect=Exception("PostRestoreSessionErr")):
            r_rest_session_err = self.client.post(
                "/actions/restore-db-from-medium",
                data={"backup_file_path": str(medium_snap)}
            )
            self.assertEqual(r_rest_session_err.status_code, 200)

        # Test restore_from_medium with _get_volumes_and_medium_backups error in exception handler
        with patch("app.routers.actions._get_volumes_and_medium_backups", side_effect=Exception("VolListFail")):
            r_rest_err_branch = self.client.post("/actions/restore-db-from-medium", data={"backup_file_path": str(self.root / "missing.db")})
            self.assertEqual(r_rest_err_branch.status_code, 200)

        # Test gdrive-restore-db authenticate when service is None, and SessionLocal post-restore exception
        with patch.object(g_mgr, "authenticate", side_effect=_fake_auth), patch("app.database.SessionLocal", side_effect=Exception("GDrivePostRestoreErr")):
            g_mgr.service = None
            with patch("app.cloud.google_drive.GoogleDriveManager", return_value=g_mgr):
                r_rest_g_auth = self.client.post("/actions/gdrive-restore-db", data={"file_id": "db_bkp_1", "folder_id": "fld1"})
                self.assertEqual(r_rest_g_auth.status_code, 200)

        # Test _get_volumes_and_medium_backups fallback branches
        from app.routers.actions import _get_volumes_and_medium_backups
        with patch.object(BackupManager, "list_backups_in_dir", side_effect=Exception("DiskErr")):
            _get_volumes_and_medium_backups(None)

        # 7. GET /actions/gdrive-panel loads online volumes and medium backups
        r_panel_vols = self.client.get(f"/actions/gdrive-panel?medium_volume_id={vol_medium_id}")
        self.assertEqual(r_panel_vols.status_code, 200)
        self.assertIn("USB_BACKUP_DRIVE", r_panel_vols.text)
        self.assertIn("Kopia zapasowa bazy SQLite na wybranym nośniku", r_panel_vols.text)

        # Test HTML5 user guide (/przewodnik) and technical docs (/dokumentacja) routes
        r_guide = self.client.get("/przewodnik")
        self.assertEqual(r_guide.status_code, 200)
        self.assertIn("Przewodnik Użytkownika", r_guide.text)

        r_docs = self.client.get("/dokumentacja")
        self.assertEqual(r_docs.status_code, 200)
        self.assertIn("Domowe Archiwum Foto", r_docs.text)

    # =========================================================================
    # 9. OFFLINE DISK SPACE PERSISTENCE
    # =========================================================================
    def test_offline_space_persistence_and_local_text_report(self):
        with self.TestingSessionLocal() as db:
            svc = StorageAutomationService(db)
            vol_a = WindowsVolume(
                volume_serial_number="VSN-REP-A",
                volume_label="Archiwum_Offline_A",
                last_known_drive_letter="X:\\MissingDriveA\\",
                is_online=False,
                total_space_bytes=500 * 1024 * 1024 * 1024,
                free_space_bytes=120 * 1024 * 1024 * 1024,
            )
            vol_b = WindowsVolume(
                volume_serial_number="VSN-REP-B",
                volume_label="Backup_Online_B",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
            )
            vol_empty = WindowsVolume(
                volume_serial_number="VSN-REP-EMP",
                volume_label="Pendrive_Bez_Pojemnosci",
                last_known_drive_letter="Y:\\MissingEmpty\\",
                is_online=False,
                total_space_bytes=0,
                free_space_bytes=0,
            )
            db.add_all([vol_a, vol_b, vol_empty])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            db.refresh(vol_empty)

            # Verify refresh_volumes_online_status keeps cached offline capacity for disconnected volume
            # and updates live capacity for online volume_b
            svc.refresh_volumes_online_status()
            db.refresh(vol_a)
            db.refresh(vol_b)
            self.assertFalse(vol_a.is_online)
            self.assertEqual(vol_a.free_space_bytes, 120 * 1024 * 1024 * 1024)
            self.assertTrue(vol_b.is_online)
            self.assertGreater(vol_b.free_space_bytes, 0)

            # Check fallback_free_bytes works when drive is offline
            ok_off, fb_bytes = StorageAutomationService.check_volume_free_space(
                vol_a.last_known_drive_letter, 1024 * 1024, fallback_free_bytes=vol_a.free_space_bytes
            )
            self.assertTrue(ok_off)
            self.assertEqual(fb_bytes, 120 * 1024 * 1024 * 1024)

            # Create sample media items: unbacked item on offline vol_a + duplicate on vol_b + pending TODO move
            now_dt = datetime.datetime.utcnow()
            it_orig = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_a.id,
                relative_path="2022/wakacje_offline.jpg",
                file_name="wakacje_offline.jpg",
                file_size_bytes=5 * 1024 * 1024,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="hash_offline_1",
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                effective_date=datetime.datetime(2022, 7, 1, 12, 0),
                thumbnail_relative_path="t1.webp",
            )
            it_dup_src = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_b.id,
                relative_path="Stary_Folder/duplikat.jpg",
                file_name="duplikat.jpg",
                file_size_bytes=4 * 1024 * 1024,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="hash_dup_pair",
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                effective_date=datetime.datetime(2022, 8, 1, 12, 0),
                thumbnail_relative_path="t2.webp",
            )
            it_dup_extra = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_b.id,
                relative_path="Kopia/duplikat_kopia.jpg",
                file_name="duplikat_kopia.jpg",
                file_size_bytes=4 * 1024 * 1024,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="hash_dup_pair",
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=False,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                effective_date=datetime.datetime(2022, 8, 1, 12, 0),
                thumbnail_relative_path="t3.webp",
            )
            it_reloc_orig = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_a.id,
                relative_path="2021/relokacja.jpg",
                file_name="relokacja.jpg",
                file_size_bytes=2 * 1024 * 1024,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="hash_reloc_pair",
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                effective_date=datetime.datetime(2021, 9, 1, 12, 0),
                thumbnail_relative_path="t4.webp",
            )
            it_reloc_copy = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_b.id,
                relative_path="Luzne_Kopie/relokacja.jpg",
                file_name="relokacja.jpg",
                file_size_bytes=2 * 1024 * 1024,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="hash_reloc_pair",
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=False,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                effective_date=datetime.datetime(2021, 9, 1, 12, 0),
                thumbnail_relative_path="t5.webp",
            )
            db.add_all([it_orig, it_dup_src, it_dup_extra, it_reloc_orig, it_reloc_copy])
            db.commit()

            # Check sidebar and backup panel display cached offline free space
            r_sb = self.client.get("/media/sidebar")
            self.assertEqual(r_sb.status_code, 200)
            self.assertIn("Offline", r_sb.text)
            self.assertIn("120.0 GB", r_sb.text)

            r_bp = self.client.get("/actions/backup-panel")
            self.assertEqual(r_bp.status_code, 200)
            self.assertIn("120.00 GB", r_bp.text)

    # =========================================================================
    # 12. QUALITY DEFECT TAGS, ON-DEMAND DUP/UNBACKED, PAIR MODES & UNINDEX JOB
    # =========================================================================
    def test_advanced_quality_ai_unindex_backup_modes_and_ui_updates(self):
        # 1) DB Migration clears legacy missing-EXIF anomaly rows
        mig_db_path = self.root / "mig_exif_cleanup.db"
        conn = sqlite3.connect(mig_db_path.as_posix())
        conn.execute(
            "CREATE TABLE media_items (id TEXT PRIMARY KEY, file_name TEXT, "
            "has_date_anomaly INTEGER DEFAULT 0, date_anomaly_reason TEXT);"
        )
        conn.execute(
            "INSERT INTO media_items VALUES ('x1', 'a.jpg', 1, 'Brak metadanych EXIF DateTimeOriginal')"
        )
        conn.execute(
            "INSERT INTO media_items VALUES ('x2', 'b.jpg', 1, 'Niezgodność dat: folder wskazuje inny rok')"
        )
        conn.commit()
        conn.close()
        ensure_schema_migrations(mig_db_path)
        conn = sqlite3.connect(mig_db_path.as_posix())
        row_x1 = conn.execute("SELECT has_date_anomaly, date_anomaly_reason FROM media_items WHERE id='x1'").fetchone()
        row_x2 = conn.execute("SELECT has_date_anomaly, date_anomaly_reason FROM media_items WHERE id='x2'").fetchone()
        conn.close()
        self.assertEqual(row_x1[0], 0)
        self.assertIsNone(row_x1[1])
        self.assertEqual(row_x2[0], 1)

        # 2) Multi-criteria quality defect classification (dark, overexposed, blank, low-res, corrupt)
        q_dir = self.root / "quality_samples"
        q_dir.mkdir(parents=True, exist_ok=True)
        dark_p = q_dir / "dark.png"
        bright_p = q_dir / "bright.png"
        lowres_p = q_dir / "tiny.png"
        corrupt_p = q_dir / "broken.jpg"

        cv2.imwrite(str(dark_p), np.full((600, 600, 3), 4, dtype=np.uint8))
        cv2.imwrite(str(bright_p), np.full((600, 600, 3), 253, dtype=np.uint8))
        cv2.imwrite(str(lowres_p), np.full((120, 140, 3), 120, dtype=np.uint8))
        corrupt_p.write_bytes(b"NOT_AN_IMAGE_AT_ALL")

        d_res = MediaQualityAnalyzer.analyze_image_detailed(dark_p)
        b_res = MediaQualityAnalyzer.analyze_image_detailed(bright_p)
        l_res = MediaQualityAnalyzer.analyze_image_detailed(lowres_p, orig_dims=(120, 140))
        c_res = MediaQualityAnalyzer.analyze_image_detailed(corrupt_p)
        self.assertIn("do_usuniecia:niedoswietlone", d_res["quality_tags"])
        self.assertIn("do_usuniecia:przeswietlone", b_res["quality_tags"])
        self.assertIn("do_usuniecia:niska_rozdzielczosc", l_res["quality_tags"])
        self.assertIn("do_usuniecia:brak_detali", l_res["quality_tags"])
        self.assertTrue(c_res["is_corrupted"])
        self.assertIn("do_usuniecia:uszkodzone", c_res["quality_tags"])

        # classify_quality_tags direct branches (corrupted, low_res + fallback blur reason)
        _, ct1 = MediaQualityAnalyzer.classify_quality_tags(dark_p, True, False, None)
        self.assertIn("do_usuniecia:uszkodzone", ct1)
        sharp_p = q_dir / "sharp_small.png"
        checker = np.indices((50, 50)).sum(axis=0) % 2 * 255
        cv2.imwrite(str(sharp_p), checker.astype(np.uint8))
        _, ct2 = MediaQualityAnalyzer.classify_quality_tags(sharp_p, False, True, 50.0, orig_dims=(120, 140))
        self.assertIn("do_usuniecia:niska_rozdzielczosc", ct2)
        _, ct3 = MediaQualityAnalyzer.classify_quality_tags(sharp_p, False, True, 50.0, orig_dims=(500, 500))
        self.assertEqual(ct3, ["do_usuniecia", "do_usuniecia:rozmyte"])

        # 3) Prepare 3 volumes: A (2 identical + 1 collision), B (backup target), C (separate pair)
        import app.routers.actions as actions_module

        drv_a = self.root / "VolPairA"
        drv_b = self.root / "VolPairB"
        drv_c = self.root / "VolPairC"
        (drv_a / "2023").mkdir(parents=True, exist_ok=True)
        drv_b.mkdir(parents=True, exist_ok=True)
        drv_c.mkdir(parents=True, exist_ok=True)

        img_a1 = drv_a / "2023" / "2023-06-15_wakacje.jpg"
        create_synthetic_jpeg(img_a1, width=500, height=500, sharp=True, color=(40, 120, 200))
        (drv_a / "duplikaty").mkdir(parents=True, exist_ok=True)
        img_a_dup = drv_a / "duplikaty" / "kopia_wakacje.jpg"
        shutil.copy2(img_a1, img_a_dup)

        # Dark image placed on Drive A to verify AI batch quality tags and unanalyzed filter
        img_a_dark = drv_a / "2023" / "ciemne_foto.png"
        cv2.imwrite(str(img_a_dark), np.full((500, 500, 3), 3, dtype=np.uint8))

        # File on Drive C for multi-pair filtering
        img_c1 = drv_c / "rodzina_c.jpg"
        create_synthetic_jpeg(img_c1, width=500, height=500, sharp=True, color=(180, 60, 40))

        vsn_map = {
            drv_a.resolve(): ("AAAA-1111", "Dysk_A", str(drv_a) + "\\"),
            drv_b.resolve(): ("BBBB-2222", "Dysk_B", str(drv_b) + "\\"),
            drv_c.resolve(): ("CCCC-3333", "Dysk_C", str(drv_c) + "\\"),
        }

        def fake_vol_info(path_obj: Path):
            res_p = path_obj.resolve()
            for root_p, info in vsn_map.items():
                if res_p == root_p or str(res_p).startswith(str(root_p)):
                    return info
            return ("ZZZZ-0000", "Other", str(path_obj.resolve()) + "\\")

        connected_test_vsns = {
            "AAAA-1111": (str(drv_a) + "\\", "Dysk_A"),
            "BBBB-2222": (str(drv_b) + "\\", "Dysk_B"),
            "CCCC-3333": (str(drv_c) + "\\", "Dysk_C"),
        }
        with patch.object(WindowsVolumeInspector, "get_volume_info", side_effect=fake_vol_info), \
             patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=connected_test_vsns), \
             patch.object(StorageAutomationService, "refresh_volumes_online_status"):

            # Scan A and C: confirm scan does NOT auto-cluster duplicates (dup_total == 0)
            s_a = actions_module.execute_scan(str(drv_a))
            self.assertEqual(s_a["dup_total"], 0)
            actions_module.execute_scan(str(drv_c))

            with self.TestingSessionLocal() as db:
                vol_a = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "AAAA-1111"))
                vol_c = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "CCCC-3333"))
                vol_b = WindowsVolume(
                    volume_serial_number="BBBB-2222",
                    volume_label="Dysk_B",
                    last_known_drive_letter=str(drv_b) + "\\",
                    is_online=True,
                    total_space_bytes=100 * 1024**3,
                    free_space_bytes=50 * 1024**3,
                )
                vol_a.is_online = True
                vol_c.is_online = True
                vol_d_offline = WindowsVolume(
                    volume_serial_number="DDDD-4444",
                    volume_label="Dysk_D_Offline",
                    last_known_drive_letter="Z:\\",
                    is_online=False,
                    total_space_bytes=100 * 1024**3,
                    free_space_bytes=50 * 1024**3,
                )
                db.add_all([vol_b, vol_d_offline])
                db.commit()
                va_id, vb_id, vc_id = vol_a.id, vol_b.id, vol_c.id

                items_a = list(db.scalars(select(MediaItem).where(MediaItem.volume_id == va_id)).all())
                # All items must start with is_primary_duplicate == True before user clicks Analyze AI
                self.assertTrue(all(x.is_primary_duplicate for x in items_a))

                # Add a synthetic hash collision + an offline item to exercise exact collision splitting & offline skip
                fake_col_path = drv_a / "kolizja_inna.jpg"
                create_synthetic_jpeg(fake_col_path, width=500, height=500, sharp=True, color=(20, 200, 50))
                orig_a1 = next(x for x in items_a if x.file_name == "2023-06-15_wakacje.jpg")
                shared_hash = orig_a1.file_hash
                now_dt = datetime.datetime.now()
                col_item = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=va_id,
                    relative_path="kolizja_inna.jpg",
                    file_name="kolizja_inna.jpg",
                    file_size_bytes=fake_col_path.stat().st_size,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash=shared_hash,
                    hash_algorithm="sha256",
                    width=500,
                    height=500,
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    effective_date=now_dt,
                    thumbnail_relative_path="col.webp",
                )
                off_item = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=vol_d_offline.id,
                    relative_path="offline.jpg",
                    file_name="offline.jpg",
                    file_size_bytes=1234,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash=shared_hash,
                    hash_algorithm="sha256",
                    width=500,
                    height=500,
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    effective_date=now_dt,
                    thumbnail_relative_path="off.webp",
                )
                db.add_all([col_item, off_item])
                db.commit()
                a1_id = orig_a1.id
                col_id = col_item.id

            # 4) Verify Sidebar & Home UI updates (no auto-organize info box, no TODO in sidebar, AI missing section present)
            r_home = self.client.get("/")
            self.assertEqual(r_home.status_code, 200)
            self.assertIn("persistedSelectedIds", r_home.text)
            self.assertIn("duplicates_fast", r_home.text)
            self.assertIn("duplicates_exact", r_home.text)
            self.assertIn("Trwa wykonywanie operacji", r_home.text)

            r_sb = self.client.get("/media/sidebar")
            self.assertEqual(r_sb.status_code, 200)
            self.assertIn("Niezgodność dat", r_sb.text)
            self.assertIn("Zdjęcia bez danych Analizy AI", r_sb.text)
            self.assertNotIn("1-Klik Auto-Organizacja", r_sb.text)
            self.assertNotIn("Lista TODO Przeniesień", r_sb.text)

            # New missing-AI-data search filters
            self.assertEqual(self.client.get("/media/search?unvectored_only=true").status_code, 200)
            self.assertEqual(self.client.get("/media/search?unanalyzed_quality_only=true").status_code, 200)
            self.assertEqual(self.client.get("/media/search?untagged_only=true").status_code, 200)

            # File details modal displays all 5 date sources & auto-refresh hook
            r_det = self.client.get(f"/media/{a1_id}/detail")
            self.assertEqual(r_det.status_code, 200)
            self.assertIn("Data z nazwy pliku:", r_det.text)
            self.assertIn("Data z folderu:", r_det.text)
            self.assertIn("htmx.ajax(", r_det.text)

            # 5) On-demand Analyze AI: quality, duplicates_fast, duplicates_exact, unbacked
            r_q = self.client.post(
                "/actions/analyze-all",
                data={"analysis_type": "quality", "scope": "all_drives", "run_sync": "true"},
            )
            self.assertEqual(r_q.status_code, 200)

            with self.TestingSessionLocal() as db:
                dark_obj = db.scalar(select(MediaItem).where(MediaItem.file_name == "ciemne_foto.png"))
                self.assertTrue(dark_obj.is_blurry)
                q_tags = {
                    t.tag for t in db.scalars(
                        select(MediaTag).where(MediaTag.media_id == dark_obj.id, MediaTag.source_model == "quality_check")
                    ).all()
                }
                self.assertIn("do_usuniecia", q_tags)
                self.assertIn("do_usuniecia:niedoswietlone", q_tags)

            # Test quality defect sub-filters, defect counts in sidebar, and modal details
            r_sb_q = self.client.get("/media/sidebar")
            self.assertEqual(r_sb_q.status_code, 200)
            self.assertIn("Wady jakości (do usunięcia)", r_sb_q.text)
            self.assertIn("Niedoświetlone (ciemne)", r_sb_q.text)
            self.assertIn("do_usuniecia:niedoswietlone", r_sb_q.text)

            r_q_filter = self.client.get("/media/search?quality_defect=niedoswietlone")
            self.assertEqual(r_q_filter.status_code, 200)
            self.assertIn("ciemne_foto.png", r_q_filter.text)
            self.assertIn("CIEMNE", r_q_filter.text)

            r_q_tag = self.client.get("/media/search?tag=do_usuniecia:niedoswietlone")
            self.assertEqual(r_q_tag.status_code, 200)
            self.assertIn("ciemne_foto.png", r_q_tag.text)

            r_q_all = self.client.get("/media/search?quality_defect=all")
            self.assertEqual(r_q_all.status_code, 200)
            self.assertIn("ciemne_foto.png", r_q_all.text)

            with self.TestingSessionLocal() as db:
                dark_id = db.scalar(select(MediaItem.id).where(MediaItem.file_name == "ciemne_foto.png"))
            r_det_q = self.client.get(f"/media/{dark_id}/detail")
            self.assertEqual(r_det_q.status_code, 200)
            self.assertIn("Wykryte wady jakości", r_det_q.text)
            self.assertIn("Niedoświetlone (skrajnie ciemne)", r_det_q.text)

            # Exercise all defect branches in get_media_detail and search sort/filter
            with self.TestingSessionLocal() as db:
                multi_def = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=va_id,
                    relative_path="multi_def.jpg",
                    file_name="multi_def.jpg",
                    file_size_bytes=1000,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash="fakehash_multi_def",
                    hash_algorithm="sha256",
                    width=100,
                    height=100,
                    is_corrupted=True,
                    is_blurry=True,
                    blur_score=10.0,
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    effective_date=datetime.datetime(2023, 5, 1, 12, 0),
                    thumbnail_relative_path="multi.webp",
                )
                db.add(multi_def)
                db.flush()
                for qt in [
                    "do_usuniecia:uszkodzone",
                    "do_usuniecia:przeswietlone",
                    "do_usuniecia:brak_detali",
                    "do_usuniecia:niska_rozdzielczosc",
                    "do_usuniecia:rozmyte",
                    "do_usuniecia:rozmyte",
                ]:
                    db.add(MediaTag(media_id=multi_def.id, tag=qt, confidence=1.0, source_model="quality_check"))
                db.commit()
                multi_id = multi_def.id

            r_det_multi = self.client.get(f"/media/{multi_id}/detail")
            self.assertEqual(r_det_multi.status_code, 200)
            self.assertIn("Plik uszkodzony", r_det_multi.text)
            self.assertIn("Prześwietlone", r_det_multi.text)
            self.assertIn("Brak detali", r_det_multi.text)
            self.assertIn("Niska rozdzielczość", r_det_multi.text)

            # Test corrupted fallback and blurry fallback without tags
            with self.TestingSessionLocal() as db:
                corr_fallback = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=va_id,
                    relative_path="corr_fb.jpg",
                    file_name="corr_fb.jpg",
                    file_size_bytes=500,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash="fakehash_corrfb",
                    hash_algorithm="sha256",
                    is_corrupted=True,
                    is_blurry=False,
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    effective_date=now_dt,
                    thumbnail_relative_path="corrfb.webp",
                )
                blur_fallback = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=va_id,
                    relative_path="blur_fb.jpg",
                    file_name="blur_fb.jpg",
                    file_size_bytes=600,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash="fakehash_blurfb",
                    hash_algorithm="sha256",
                    is_corrupted=False,
                    is_blurry=True,
                    blur_score=15.0,
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    effective_date=now_dt,
                    thumbnail_relative_path="blurfb.webp",
                )
                db.add_all([corr_fallback, blur_fallback])
                db.commit()
                cfb_id, bfb_id = corr_fallback.id, blur_fallback.id

            self.assertEqual(self.client.get(f"/media/{cfb_id}/detail").status_code, 200)
            self.assertEqual(self.client.get(f"/media/{bfb_id}/detail").status_code, 200)

            # Additional search filters and sort options coverage
            self.assertEqual(self.client.get("/media/search?year=2023").status_code, 200)
            self.assertEqual(self.client.get("/media/search?year=invalid").status_code, 200)
            self.assertEqual(self.client.get("/media/search?quality_defect=niedoswietlone&tag=ciemne").status_code, 200)
            self.assertEqual(self.client.get("/media/search?sort_by=size_asc").status_code, 200)
            self.assertEqual(self.client.get("/media/search?sort_by=size_desc").status_code, 200)
            self.assertEqual(self.client.get("/media/search?sort_by=name_asc").status_code, 200)
            self.assertEqual(self.client.get("/media/search?sort_by=blur_asc").status_code, 200)

            # Verify that files in daf_backup or with is_backup_copy=True are NOT displayed in media search (Req 2)
            with self.TestingSessionLocal() as db:
                daf_bkp_item = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=va_id,
                    relative_path="daf_backup/2023/2023-01/daf_vault_copy.jpg",
                    file_name="daf_vault_copy.jpg",
                    file_size_bytes=100,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash="fake_daf_bkp_hash",
                    hash_algorithm="sha256",
                    is_primary_duplicate=False,
                    is_backup_copy=True,
                    verified_sha256="fake_daf_bkp_hash",
                    effective_date=datetime.datetime(2023, 1, 1),
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    thumbnail_relative_path="daf_bkp.webp",
                )
                db.add(daf_bkp_item)
                db.commit()

            r_search_daf = self.client.get("/media/search?q=daf_vault_copy")
            self.assertEqual(r_search_daf.status_code, 200)
            self.assertNotIn("daf_vault_copy.jpg", r_search_daf.text)

            r_df = self.client.post(
                "/actions/analyze-all",
                data={"analysis_type": "duplicates_fast", "scope": "all_drives", "run_sync": "true"},
            )
            self.assertIn("ORYGINAŁ, BACKUP zielony/pomarańczowy, DUPLIKAT", r_df.text)

            r_de = self.client.post(
                "/actions/analyze-all",
                data={"analysis_type": "duplicates_exact", "scope": "all_drives", "run_sync": "true"},
            )
            self.assertIn("Dokładna analiza duplikatów 100%", r_de.text)

            with self.TestingSessionLocal() as db:
                col_reloaded = db.get(MediaItem, col_id)
                self.assertEqual(col_reloaded.hash_algorithm, "sha256_verified")
                self.assertTrue(col_reloaded.is_primary_duplicate)

                # Verify offline items are IGNORED cleanly by quality AI analysis (no corrupted marks/thumbnails)
                off_obj = db.scalar(select(MediaItem).where(MediaItem.file_name == "offline.jpg"))
                self.assertIsNotNone(off_obj)
                off_thumb = self.thumbs_dir / off_obj.thumbnail_relative_path
                off_thumb.write_bytes(b"VALID_CACHED_PREVIEW_BYTES_" * 10)
                r_q_off = self.client.post(
                    "/actions/analyze-all",
                    data={
                        "analysis_type": "quality",
                        "scope": "selected",
                        "selected_ids": [off_obj.id],
                        "run_sync": "true",
                    },
                )
                self.assertEqual(r_q_off.status_code, 200)
                self.assertIn("Pominięto offline", r_q_off.text)
                db.expire_all()
                off_reloaded = db.get(MediaItem, off_obj.id)
                self.assertIsNone(off_reloaded.blur_score)
                self.assertFalse(off_reloaded.is_corrupted)

                # 0-byte corrupted thumbnail file for offline media must be cleaned up and return 404 (not empty bytes)
                off_thumb.write_bytes(b"")
                r_th_empty = self.client.get(f"/media/thumbnails/{off_obj.thumbnail_relative_path}")
                self.assertEqual(r_th_empty.status_code, 404)
                self.assertFalse(off_thumb.exists())

                # Delete synthetic collision & offline items before backup tests
                db.delete(col_reloaded)
                db.delete(off_reloaded)
                db.commit()

            r_ub = self.client.post(
                "/actions/analyze-all",
                data={"analysis_type": "unbacked", "scope": "all_drives", "run_sync": "true"},
            )
            self.assertIn("Analiza kopii zapasowych", r_ub.text)

            # On-demand date discrepancy analysis via 'Analizuj AI' (EXIF cleared, conflict picks oldest + flags)
            self.assertIn("date_discrepancy", r_home.text)
            with self.TestingSessionLocal() as db:
                a1_obj = db.get(MediaItem, a1_id)
                a1_obj.exif_date = datetime.datetime(2023, 6, 15, 10, 0)
                a1_obj.has_date_anomaly = True
                a1_obj.date_anomaly_reason = "Stara niezgodnosc"
                db.commit()
            r_dd = self.client.post(
                "/actions/analyze-all",
                data={"analysis_type": "date_discrepancy", "scope": "all_drives", "run_sync": "true"},
            )
            self.assertEqual(r_dd.status_code, 200)
            self.assertIn("Analiza niezgodności dat", r_dd.text)
            with self.TestingSessionLocal() as db:
                a1_re = db.get(MediaItem, a1_id)
                self.assertFalse(a1_re.has_date_anomaly)
                self.assertEqual(a1_re.effective_date, datetime.datetime(2023, 6, 15, 10, 0))
                self.assertIsNotNone(
                    db.scalar(select(BackgroundJob).where(BackgroundJob.title.like("%Analiza niezgodności dat%")))
                )
            # Also cover stop branch during date_discrepancy batch
            with patch.object(ProcessManager, "is_stop_requested", return_value=True):
                j_dd_stop = ProcessManager.create_job("ai_analysis", "DD Stop", [a1_id])
                actions_module._run_background_ai_batch(j_dd_stop, [a1_id], analysis_type="date_discrepancy")

            # 6) Backup modal has NO duplicates UI/list, supports target volume selection, minimize_transfers, and YYYY/MM/DD
            self.assertIsNone(StorageAutomationService.parse_pair_keys(None))
            self.assertEqual(StorageAutomationService.parse_pair_keys([]), set())
            self.assertEqual(StorageAutomationService.parse_pair_keys([f"{vb_id}:{va_id}"]), {(min(va_id, vb_id), max(va_id, vb_id))})

            pair_ab = f"{min(va_id, vb_id)}:{max(va_id, vb_id)}"
            r_bp = self.client.get("/actions/backup-panel")
            self.assertEqual(r_bp.status_code, 200)
            self.assertIn("filterUnbackedBySelectedPairs", r_bp.text)
            self.assertIn("Brakujące backupy", r_bp.text)
            self.assertIn("Backupy zielone", r_bp.text)
            self.assertIn("Backupy pomarańczowe", r_bp.text)
            self.assertIn("Zminimalizuj przeniesienia", r_bp.text)
            self.assertIn("Zrób backup", r_bp.text)
            self.assertNotIn("Lista duplikatów z wybranych urządzeń", r_bp.text)
            self.assertNotIn("duplicate-file-row", r_bp.text)

            # Execute backup ONLY on pair {A, B}: backup created on B in YYYY/MM/DD, A's duplicate still preserved, C untouched
            r_exec_bkp = self.client.post(
                "/actions/execute-backup",
                data={
                    "preferred_target_volume_id": str(vb_id),
                    "target_vols_submitted": "1",
                    "target_volume_ids": [str(vb_id)],
                    "pairs_submitted": "1",
                    "selected_pairs": [pair_ab],
                },
            )
            self.assertEqual(r_exec_bkp.status_code, 200)
            self.assertTrue(img_a_dup.exists())
            self.assertTrue(any((drv_b / "daf_backup" / "2023" / "06" / "15").rglob("2023-06-15_wakacje.jpg")))
            with self.TestingSessionLocal() as db:
                svc_chk = StorageAutomationService(db)
                unbacked_after_ab = svc_chk.detect_files_without_backup(recluster=True)
                self.assertTrue(any(u.volume_id == vc_id for u in unbacked_after_ab))

            # Now test Duplikaty binarne view + cross-page selection + safe duplicate removal of redundant 3rd copy
            r_dup_view = self.client.get("/media/search?duplicates_only=true")
            self.assertEqual(r_dup_view.status_code, 200)
            self.assertNotIn("Automatyczny Backup i 100% Bezpieczne Usunięcie Duplikatów", r_dup_view.text)
            self.assertIn("Zaznacz tylko nadmiarowe kopie", r_dup_view.text)
            self.assertIn("ORYGINAŁ", r_dup_view.text)
            self.assertIn("DUPLIKAT", r_dup_view.text)

            r_red = self.client.get("/actions/redundant-duplicate-ids?duplicates_only=true")
            red_ids = r_red.json()["ids"]
            self.assertTrue(len(red_ids) >= 1)
            r_del_red = self._delete_permanently_two_phase(
                {
                    "scope": "selected",
                    "duplicates_only": "true",
                    "selected_ids_json": json.dumps(red_ids),
                },
            )
            self.assertEqual(r_del_red.status_code, 200)
            self.assertFalse(img_a_dup.exists())
            self.assertTrue(img_a1.exists())

            # 7) Fast chunked Unindex + BackgroundJob + /actions/unindex-status polling paths
            actions_module.UNINDEX_BATCH_STATE["running"] = True
            actions_module.UNINDEX_BATCH_STATE["processed"] = 5
            actions_module.UNINDEX_BATCH_STATE["total"] = 20
            try:
                self.assertIn("Wyindeksowanie dysku z bazy w toku", self.client.get("/actions/unindex-status").text)
            finally:
                actions_module.UNINDEX_BATCH_STATE["running"] = False

            self.assertIn("Usunięto z indeksu bazy", self.client.get("/actions/unindex-status").text)

            r_un_c = self.client.post(
                "/actions/unindex",
                data={"unindex_volume_id": str(vc_id), "run_sync": "true"},
            )
            self.assertEqual(r_un_c.status_code, 200)
            self.assertIn("Usunięto z indeksu bazy", r_un_c.text)
            with self.TestingSessionLocal() as db:
                self.assertIsNone(db.get(WindowsVolume, vc_id))
                job_un = db.scalar(select(BackgroundJob).where(BackgroundJob.job_type == "unindex"))
                self.assertIsNotNone(job_un)
                self.assertEqual(job_un.status, "COMPLETED")
                # Original file on Disk C remains untouched on disk
                self.assertTrue(img_c1.exists())

        # 8) Disconnected hardware VSN must be detected as OFFLINE even if another drive took its drive letter
        with self.TestingSessionLocal() as db:
            vol_reused = WindowsVolume(
                volume_serial_number="1111-2222",
                volume_label="Odlaczony_USB",
                last_known_drive_letter="C:\\",
                is_online=True,
            )
            db.add(vol_reused)
            db.commit()
            vr_id = vol_reused.id
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"9999-8888": ("C:\\", "Inny_Dysk")}):
                StorageAutomationService(db).refresh_volumes_online_status()
            db.expire_all()
            self.assertFalse(db.get(WindowsVolume, vr_id).is_online)

    # ------------------------------------------------------------------
    # 14. TRUNCATED JPEGs, TINY VALID WEBP THUMBNAILS & AI THUMB REPAIR
    # ------------------------------------------------------------------
    def test_truncated_jpeg_resilience_small_webp_and_preview_repair(self):
        # 1) Create a slightly truncated camera JPEG (missing final 10 bytes / EOI marker)
        full_jpg = self.drive_a / "full_src.jpg"
        trunc_jpg = self.drive_a / "trunc_camera.jpg"
        create_synthetic_jpeg(full_jpg, width=320, height=240, sharp=True)
        raw_bytes = full_jpg.read_bytes()
        trunc_jpg.write_bytes(raw_bytes[:-10])

        # ThumbnailService must decode truncated JPEG cleanly and write valid WebP
        t_id = str(uuid.uuid4())
        t_name, tw, th, _ = ThumbnailService.generate(trunc_jpg, t_id, is_img=True)
        self.assertEqual((tw, th), (320, 240))
        self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / t_name))

        # OpenCV fallback in ThumbnailService when PIL raises while cv2.imread succeeds
        t_cv_id = str(uuid.uuid4())
        with patch("app.scanner.Image.open", side_effect=OSError("pil blocked")):
            t_cv_name, cw, ch, _ = ThumbnailService.generate(full_jpg, t_cv_id, is_img=True)
            self.assertEqual((cw, ch), (320, 240))
            self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / t_cv_name))

        # Quality, tagger, and vector embedder must all decode truncated JPEG without flagging corrupted
        q_res = MediaQualityAnalyzer.analyze_image_detailed(trunc_jpg)
        self.assertFalse(q_res["is_corrupted"])
        self.assertIsNotNone(q_res["blur_score"])
        vec = OnDemandClipEmbedder.generate_image_embedding(trunc_jpg)
        self.assertEqual(len(vec), 512 * 4)
        tagger = OnDemandObjectTagger()
        tagger.model_path = self.root / "no_model.onnx"
        tags = tagger.tag_image(trunc_jpg)
        self.assertIsInstance(tags, list)

        # Genuinely corrupted bytes must still be marked corrupted
        garbage_jpg = self.drive_a / "garbage.jpg"
        garbage_jpg.write_bytes(b"\x00\x00\x10\x27GARBAGE_NOT_JPEG")
        self.assertTrue(MediaQualityAnalyzer.analyze_image_detailed(garbage_jpg)["is_corrupted"])

        # Embedded frame extraction (e.g. Olympus/camera file with corrupted prefix but valid embedded JPEG)
        embedded_raw = b"\x00\x00\x10\x27" * 50 + full_jpg.read_bytes()
        embedded_file = self.drive_a / "embedded_prefix.jpg"
        embedded_file.write_bytes(embedded_raw)
        emb_id = str(uuid.uuid4())
        emb_name, ew, eh, _ = ThumbnailService.generate(embedded_file, emb_id, is_img=True)
        self.assertEqual((ew, eh), (320, 240))
        self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / emb_name))

        # Test decode_image_array with embedded frame fallback (both color and grayscale)
        from app.ai.quality import decode_image_array
        arr_gray = decode_image_array(embedded_file, grayscale=True)
        self.assertIsNotNone(arr_gray)
        arr_col = decode_image_array(embedded_file, grayscale=False)
        self.assertIsNotNone(arr_col)

        # Totally unreadable image/video files must generate a valid placeholder thumbnail (so 404 is avoided)
        unreadable_img = self.drive_a / "total_junk.jpg"
        unreadable_img.write_bytes(b"TOTAL_JUNK_DATA_NO_JPEG")
        uj_id = str(uuid.uuid4())
        uj_name, uw, uh, _ = ThumbnailService.generate(unreadable_img, uj_id, is_img=True)
        self.assertIsNone(uw)
        self.assertIsNone(uh)
        self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / uj_name))

        unreadable_vid = self.drive_a / "total_junk.mov"
        unreadable_vid.write_bytes(b"TOTAL_JUNK_VIDEO_NO_MOOV")
        uv_id = str(uuid.uuid4())
        uv_name, vw, vh, vd = ThumbnailService.generate(unreadable_vid, uv_id, is_img=False)
        self.assertIsNone(vw)
        self.assertIsNone(vh)
        self.assertIsNone(vd)
        self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / uv_name))

        # Embedded frame inside a broken video (e.g. Motion-JPEG MOV with missing moov atom)
        broken_vid = self.drive_a / "broken_mjpeg.mov"
        broken_vid.write_bytes(b"\x00\x00\x00\x14pnot" + full_jpg.read_bytes())
        bv_id = str(uuid.uuid4())
        bv_name, bw, bh, _ = ThumbnailService.generate(broken_vid, bv_id, is_img=False)
        self.assertEqual((bw, bh), (320, 240))
        self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / bv_name))

        # MediaQualityAnalyzer.analyze_video on unreadable video returns corrupted=True, blurry=True, blur_score=0.0
        v_corr, v_blur, v_sc = MediaQualityAnalyzer.analyze_video(unreadable_vid)
        self.assertTrue(v_corr)
        self.assertTrue(v_blur)
        self.assertEqual(v_sc, 0.0)

        # 2) Valid tiny WebP (<100 bytes, e.g. 44B-90B icon) on OFFLINE volume must NOT be unlinked
        tiny_img = Image.new("RGB", (6, 6), (20, 80, 140))
        tiny_id = str(uuid.uuid4())
        tiny_thumb = self.thumbs_dir / f"{tiny_id}.webp"
        tiny_img.save(tiny_thumb, "WEBP", quality=50)
        self.assertLess(tiny_thumb.stat().st_size, 100)
        self.assertTrue(ThumbnailService.is_valid_thumbnail(tiny_thumb))

        with self.TestingSessionLocal() as db:
            vol_on = WindowsVolume(
                volume_serial_number="TRNC-1111",
                volume_label="OnlineTrunc",
                last_known_drive_letter=str(self.drive_a) + "\\",
                is_online=True,
            )
            vol_off = WindowsVolume(
                volume_serial_number="OFFL-2222",
                volume_label="OfflineTiny",
                last_known_drive_letter="Z:\\",
                is_online=False,
            )
            db.add_all([vol_on, vol_off])
            db.flush()

            # Stale corrupted truncated JPEG with MISSING thumbnail (simulates pre-fix scan state)
            stale_id = str(uuid.uuid4())
            stale_item = MediaItem(
                id=stale_id,
                volume_id=vol_on.id,
                relative_path="trunc_camera.jpg",
                file_name="trunc_camera.jpg",
                file_size_bytes=trunc_jpg.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="trunc_hash_1",
                hash_algorithm="sha256",
                is_corrupted=True,
                blur_score=None,
                fs_ctime=datetime.datetime(2024, 5, 1),
                fs_mtime=datetime.datetime(2024, 5, 1),
                effective_date=datetime.datetime(2024, 5, 1),
                thumbnail_relative_path=f"{stale_id}.webp",
            )
            tiny_item = MediaItem(
                id=tiny_id,
                volume_id=vol_off.id,
                relative_path="tiny_off.webp",
                file_name="tiny_off.webp",
                file_size_bytes=200,
                media_type="image",
                mime_type="image/webp",
                file_hash="tiny_hash_2",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2024, 5, 2),
                fs_mtime=datetime.datetime(2024, 5, 2),
                effective_date=datetime.datetime(2024, 5, 2),
                thumbnail_relative_path=tiny_thumb.name,
            )
            db.add_all([stale_item, tiny_item])
            db.commit()

        vsn_map = {"TRNC-1111": (str(self.drive_a) + "\\", "OnlineTrunc")}
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            # Offline valid tiny (<100B) WebP must return 200 and survive intact
            r_tiny = self.client.get(f"/media/thumbnails/{tiny_thumb.name}")
            self.assertEqual(r_tiny.status_code, 200)
            self.assertTrue(tiny_thumb.exists())

            # /raw for flagged truncated JPEG must re-encode cleanly (200 image/jpeg), including OpenCV fallback
            r_raw = self.client.get(f"/media/{stale_id}/raw")
            self.assertEqual(r_raw.status_code, 200)
            self.assertEqual(r_raw.headers.get("content-type"), "image/jpeg")

            with patch("PIL.Image.open", side_effect=OSError("pil raw err")):
                r_raw_cv = self.client.get(f"/media/{stale_id}/raw")
                self.assertEqual(r_raw_cv.status_code, 200)
                self.assertEqual(r_raw_cv.headers.get("content-type"), "image/jpeg")

            # Also verify PIL fallback in decode_image_array when OpenCV imdecode + imread return None
            with patch("app.ai.quality.cv2.imdecode", return_value=None), patch("app.ai.quality.cv2.imread", return_value=None):
                q_pil = MediaQualityAnalyzer.analyze_image_detailed(full_jpg)
                self.assertFalse(q_pil["is_corrupted"])

            # Test error branches and RGBA/L modes for full scanner and quality coverage
            with patch.object(MediaQualityAnalyzer, "_decode_grayscale", return_value=None):
                q_dec_none = MediaQualityAnalyzer.analyze_image_detailed(full_jpg)
                self.assertTrue(q_dec_none["is_corrupted"])
                self.assertEqual(q_dec_none["blur_score"], 0.0)

            with patch.object(MediaQualityAnalyzer, "_decode_grayscale", side_effect=RuntimeError("lum err")):
                _, q_tags_err = MediaQualityAnalyzer.classify_quality_tags(full_jpg, False, True, 20.0)
                self.assertIn("do_usuniecia:rozmyte", q_tags_err)

            with patch("app.scanner.ThumbnailService.extract_embedded_frame", side_effect=RuntimeError("emb err")):
                self.assertIsNone(decode_image_array(garbage_jpg))

            # Test placeholder failure when dest path is invalid
            self.assertFalse(ThumbnailService.generate_placeholder(Path("invalid/\x00/dest.webp")))
            self.assertIsNone(ThumbnailService.extract_embedded_frame(Path("invalid/\x00/src.jpg")))

            # Test embedded frame with L mode in ThumbnailService.generate (both img and vid)
            l_img = Image.new("L", (120, 120), 128)
            l_buf = io.BytesIO()
            l_img.save(l_buf, format="JPEG")
            l_file = self.drive_a / "embedded_l_mode.jpg"
            l_file.write_bytes(b"\x00" * 80 + l_buf.getvalue())
            l_id = str(uuid.uuid4())
            ThumbnailService.generate(l_file, l_id, is_img=True)
            self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / f"{l_id}.webp"))

            l_vid = self.drive_a / "embedded_l_vid.mov"
            l_vid.write_bytes(b"\x00" * 80 + l_buf.getvalue())
            l_vid_id = str(uuid.uuid4())
            ThumbnailService.generate(l_vid, l_vid_id, is_img=False)
            self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / f"{l_vid_id}.webp"))

            # Semantic fallback to src_p when thumbnail is invalid and embedding exception
            from app.routers import semantic as semantic_mod
            shutil.copy(full_jpg, self.drive_a / "sem_unique_src.jpg")
            with self.TestingSessionLocal() as db_sem:
                sem_vol = WindowsVolume(volume_serial_number="SEM-1", volume_label="SemVol", last_known_drive_letter=str(self.drive_a) + "\\", is_online=True)
                db_sem.add(sem_vol)
                db_sem.flush()
                sem_item = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=sem_vol.id,
                    relative_path="sem_unique_src.jpg",
                    file_name="sem_unique_src.jpg",
                    file_size_bytes=full_jpg.stat().st_size,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash="sem_hash_1",
                    hash_algorithm="sha256",
                    effective_date=datetime.datetime(2024, 1, 1),
                    fs_ctime=datetime.datetime(2024, 1, 1),
                    fs_mtime=datetime.datetime(2024, 1, 1),
                    thumbnail_relative_path="missing_sem_thumb.webp",
                )
                db_sem.add(sem_item)
                db_sem.commit()
                db_sem.refresh(sem_item)
                vec_fb = semantic_mod._compute_vector_in_ram(sem_item)
                self.assertEqual(len(vec_fb), 512)
                with patch.object(OnDemandClipEmbedder, "generate_image_embedding", side_effect=RuntimeError("vec err")):
                    vec_zero = semantic_mod._compute_vector_in_ram(sem_item)
                    self.assertEqual(len(vec_zero), 512)

            # Run AI batch ('all') -> heals missing thumbnail AND clears false is_corrupted flag
            r_ai = self.client.post(
                "/actions/analyze-all",
                data={"analysis_type": "all", "scope": "all_drives", "unanalyzed_quality_only": "true", "run_sync": "true"},
            )
            self.assertEqual(r_ai.status_code, 200)
            self.assertTrue(ThumbnailService.is_valid_thumbnail(self.thumbs_dir / f"{stale_id}.webp"))
            with self.TestingSessionLocal() as db:
                healed = db.get(MediaItem, stale_id)
                self.assertFalse(healed.is_corrupted)
                self.assertIsNotNone(healed.blur_score)
                self.assertEqual((healed.width, healed.height), (320, 240))

    def test_pairwise_schedule_and_backup_dedup_filtering(self):
        """
        Tests the pairwise schedule (A <-> B) logic, multi-pair eligibility,
        pair-scoped execution planning, and live filtering of unbacked files and duplicate lists.
        """
        drive_c = self.root / "DriveC"
        drive_c.mkdir(parents=True, exist_ok=True)

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="VSN-PW-A",
                volume_label="DriveA",
                last_known_drive_letter=str(self.drive_a) + "\\",
                is_online=True,
                free_space_bytes=50000000,
                total_space_bytes=100000000,
            )
            vol_b = WindowsVolume(
                volume_serial_number="VSN-PW-B",
                volume_label="DriveB",
                last_known_drive_letter=str(self.drive_b) + "\\",
                is_online=True,
                free_space_bytes=50000000,
                total_space_bytes=100000000,
            )
            vol_c = WindowsVolume(
                volume_serial_number="VSN-PW-C",
                volume_label="DriveC",
                last_known_drive_letter=str(drive_c) + "\\",
                is_online=True,
                free_space_bytes=50000000,
                total_space_bytes=100000000,
            )
            db.add_all([vol_a, vol_b, vol_c])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            db.refresh(vol_c)
            va_id, vb_id, vc_id = vol_a.id, vol_b.id, vol_c.id

        img_a = Image.new("RGB", (100, 100), (255, 0, 0))
        path_a = self.drive_a / "unbacked_on_a.jpg"
        img_a.save(path_a)
        hash_a = hashlib.sha256(path_a.read_bytes()).hexdigest()

        img_dup = Image.new("RGB", (100, 100), (0, 255, 0))
        path_dup1 = self.drive_a / "same_dup1.jpg"
        path_dup2 = self.drive_a / "same_dup2.jpg"
        img_dup.save(path_dup1)
        img_dup.save(path_dup2)
        hash_dup = hashlib.sha256(path_dup1.read_bytes()).hexdigest()

        img_cross = Image.new("RGB", (100, 100), (0, 0, 255))
        path_cb = self.drive_b / "cross_dup.jpg"
        path_cc = drive_c / "cross_dup.jpg"
        img_cross.save(path_cb)
        img_cross.save(path_cc)
        shutil.copy(path_cc, drive_c / "cross_dup_extra.jpg")
        hash_cross = hashlib.sha256(path_cb.read_bytes()).hexdigest()

        now_dt = datetime.datetime.now()
        with self.TestingSessionLocal() as db:
            item_a = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=va_id,
                relative_path="unbacked_on_a.jpg",
                file_name="unbacked_on_a.jpg",
                file_size_bytes=path_a.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash=hash_a,
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                verified_sha256=hash_a,
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="a.webp",
            )
            item_dup1 = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=va_id,
                relative_path="same_dup1.jpg",
                file_name="same_dup1.jpg",
                file_size_bytes=path_dup1.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash=hash_dup,
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                verified_sha256=hash_dup,
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="dup1.webp",
            )
            item_dup2 = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=va_id,
                relative_path="same_dup2.jpg",
                file_name="same_dup2.jpg",
                file_size_bytes=path_dup2.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash=hash_dup,
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=False,
                verified_sha256=hash_dup,
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="dup2.webp",
            )
            item_cb = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vb_id,
                relative_path="cross_dup.jpg",
                file_name="cross_dup.jpg",
                file_size_bytes=path_cb.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash=hash_cross,
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                verified_sha256=hash_cross,
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="cb.webp",
            )
            item_cc = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vc_id,
                relative_path="cross_dup.jpg",
                file_name="cross_dup.jpg",
                file_size_bytes=path_cc.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash=hash_cross,
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=False,
                verified_sha256=hash_cross,
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="cc.webp",
            )
            item_cc2 = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vc_id,
                relative_path="cross_dup_extra.jpg",
                file_name="cross_dup_extra.jpg",
                file_size_bytes=path_cc.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash=hash_cross,
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=False,
                verified_sha256=hash_cross,
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="cc2.webp",
            )
            db.add_all([item_a, item_dup1, item_dup2, item_cb, item_cc, item_cc2])
            db.commit()

        # Step 1: StorageAutomationService plan verification
        with self.TestingSessionLocal() as db:
            svc = StorageAutomationService(db)
            plan = svc.build_backup_and_dedup_plan()
            self.assertGreaterEqual(len(plan["ordered_pairs"]), 3)
            pair_keys = [p["pair_key"] for p in plan["ordered_pairs"]]
            self.assertIn((min(va_id, vb_id), max(va_id, vb_id)), pair_keys)
            self.assertIn((min(va_id, vc_id), max(va_id, vc_id)), pair_keys)
            self.assertIn((min(vb_id, vc_id), max(vb_id, vc_id)), pair_keys)

            # Check pair metrics (per-device + zero double counting + reconciliation)
            p_ac_key_str = f"{min(va_id, vc_id)}:{max(va_id, vc_id)}"
            imp_sum = sum(p["metrics"]["important_to_assign_count"] for p in plan["ordered_pairs"])
            self.assertEqual(imp_sum, len(plan["important_candidates"]))
            self.assertEqual(plan["reconciliation"]["important_pairs_sum"], plan["reconciliation"]["important_plan_total"])
            self.assertEqual(plan["reconciliation"]["new_copies_pairs_sum"], plan["reconciliation"]["new_copies_plan_total"])
            for p_item in plan["ordered_pairs"]:
                self.assertEqual(
                    p_item["a_metrics"]["new_copies_count"] + p_item["b_metrics"]["new_copies_count"],
                    p_item["metrics"]["new_copies_count"],
                )
                self.assertEqual(
                    p_item["a_metrics"]["important_to_assign_count"] + p_item["b_metrics"]["important_to_assign_count"],
                    p_item["metrics"]["important_to_assign_count"],
                )

            # Step 2: Build plan restricted to Pair (va, vc) targeting vc_id
            plan_ac = svc.build_backup_and_dedup_plan(
                selected_pair_keys=[p_ac_key_str],
                target_volume_ids=[vc_id],
            )
            p_ac_entry = next(p for p in plan_ac["ordered_pairs"] if p["pair_key_str"] == p_ac_key_str)
            self.assertGreaterEqual(p_ac_entry["metrics"]["important_to_assign_count"], 1)
            copy_tasks_ac = p_ac_entry["tasks"]["backups_to_copy"]
            self.assertTrue(any(t["orig_item_id"] == item_a.id and t["dst_vol_id"] == vc_id for t in copy_tasks_ac))

        # Step 3: Test GET /actions/backup-panel with selected_pairs (per-device pair lines + no giant preview tables)
        r_all = self.client.get("/actions/backup-panel")
        self.assertEqual(r_all.status_code, 200)
        self.assertIn("backup-pair-counts-data", r_all.text)
        self.assertIn("Bilans zgodności", r_all.text)
        self.assertIn("↳ Na ", r_all.text)
        self.assertNotIn("Nowe kopie do utworzenia w <code>daf_backup/</code>", r_all.text)
        self.assertNotIn("Kopie lokalne do przeniesienia do kartoteki", r_all.text)
        self.assertNotIn("Lista oryginałów dodawanych do albumu „Ważne…”", r_all.text)

        r_filtered = self.client.get(f"/actions/backup-panel?target_volume_id={vc_id}&selected_pairs={p_ac_key_str}")
        self.assertEqual(r_filtered.status_code, 200)
        self.assertIn("Wybrane operacje do wykonania", r_filtered.text)

        # Step 3b: Test POST /actions/backup-panel-tables with selected pairs and empty pairs
        r_tables = self.client.post(
            "/actions/backup-panel-tables",
            data={"selected_pairs": [p_ac_key_str], "pairs_submitted": "true", "preferred_target_volume_id": str(vc_id)},
        )
        self.assertEqual(r_tables.status_code, 200)
        self.assertIn("Wybrane operacje do wykonania", r_tables.text)
        self.assertIn("backup-pair-counts-data", r_tables.text)

        r_tables_empty = self.client.post(
            "/actions/backup-panel-tables",
            data={"pairs_submitted": "true"},
        )
        self.assertEqual(r_tables_empty.status_code, 200)

        # Directly test _balanced_sample_unbacked, _balanced_sample_duplicates, and _balanced_sample_important
        from app.routers.actions import _balanced_sample_unbacked, _balanced_sample_duplicates, _balanced_sample_important
        fake_unbacked = [
            MediaItem(id=f"u_{i}", volume_id=1 if i < 20 else 2, relative_path=f"f_{i}.jpg")
            for i in range(50)
        ]
        sampled_u = _balanced_sample_unbacked(fake_unbacked, max_total=10)
        self.assertEqual(len(sampled_u), 10)
        self.assertEqual(_balanced_sample_unbacked([], max_total=10), [])

        fake_dups = [
            {"id": f"d_{i}", "pair_key_str": "1:2" if i < 15 else "1:3"}
            for i in range(40)
        ]
        sampled_d = _balanced_sample_duplicates(fake_dups, max_total=10)
        self.assertEqual(len(sampled_d), 10)
        self.assertEqual(_balanced_sample_duplicates([], max_total=10), [])

        fake_important = [
            {"id": f"imp_{i}", "vol_id": 1 if i < 15 else 2}
            for i in range(40)
        ]
        sampled_imp = _balanced_sample_important(fake_important, max_total=10)
        self.assertEqual(len(sampled_imp), 10)
        self.assertEqual(_balanced_sample_important([], max_total=10), [])
        self.assertEqual(len(sampled_d), 10)
        self.assertEqual(_balanced_sample_duplicates([], max_total=10), [])

        # Step 4: Execute backup restricted to pair (va, vc) targeting vc_id
        r_exec = self.client.post(
            "/actions/execute-backup",
            data={
                "selected_pairs": [p_ac_key_str],
                "pairs_submitted": "true",
                "target_vols_submitted": "1",
                "target_volume_ids": [str(vc_id)],
            },
        )
        self.assertEqual(r_exec.status_code, 200)
        self.assertIn("Raport wykonania Backupu", r_exec.text)

        # Verify physical backup on Vol C was created in daf_backup/
        backup_files_c = list((drive_c / "daf_backup").glob("**/*"))
        self.assertTrue(len(backup_files_c) > 0)

        # Step 6: Test execute backup with a new external backup drive path
        drive_d = self.root / "DriveD"
        drive_d.mkdir(parents=True, exist_ok=True)
        r_new_drive = self.client.post(
            "/actions/execute-backup",
            data={
                "new_backup_drive_path": str(drive_d) + "\\",
                "operation_mode": "backup_only",
                "pairs_submitted": "true",
            },
        )
        self.assertEqual(r_new_drive.status_code, 200)

        # Step 7: Test cluster where all copies have is_backup_copy=True and relocation candidate
        with self.TestingSessionLocal() as db:
            all_bkp1 = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=va_id,
                relative_path="all_bkp1.jpg",
                file_name="all_bkp1.jpg",
                file_size_bytes=100,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="fake_all_bkp_hash",
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=True,
                verified_sha256="fake_all_bkp_hash",
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="ab1.webp",
            )
            reloc_orig = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=va_id,
                relative_path="reloc_orig.jpg",
                file_name="reloc_orig.jpg",
                file_size_bytes=100,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="fake_reloc_hash",
                hash_algorithm="sha256",
                is_primary_duplicate=True,
                is_backup_copy=False,
                verified_sha256="fake_reloc_hash",
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="ro.webp",
            )
            reloc_copy = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vb_id,
                relative_path="reloc_copy.jpg",
                file_name="reloc_copy.jpg",
                file_size_bytes=100,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="fake_reloc_hash",
                hash_algorithm="sha256",
                is_primary_duplicate=False,
                is_backup_copy=False,
                verified_sha256="fake_reloc_hash",
                width=100,
                height=100,
                effective_date=now_dt,
                fs_ctime=now_dt,
                fs_mtime=now_dt,
                thumbnail_relative_path="rc.webp",
            )
            db.add_all([all_bkp1, reloc_orig, reloc_copy])
            db.commit()

            svc = StorageAutomationService(db)
            plan_reloc = svc.build_backup_and_dedup_plan()
            p_ab_key = (min(va_id, vb_id), max(va_id, vb_id))
            p_ab = next(p for p in plan_reloc["ordered_pairs"] if p["pair_key"] == p_ab_key)
            self.assertGreaterEqual(p_ab["metrics"]["backups_to_relocate_count"], 1)

            # Test backup filename collision (while dest_path.exists()) in YYYY/MM/DD structure
            sub_d = item_a.effective_date.strftime("%Y/%m/%d")
            col_target = drive_c / "daf_backup" / sub_d / item_a.file_name
            col_target.parent.mkdir(parents=True, exist_ok=True)
            col_target.write_bytes(b"existing_file")

            res_bkp_only = svc.execute_backup_and_deduplication(
                selected_pair_keys=[p_ac_key_str],
                target_volume_ids=[vc_id],
            )
            self.assertIn("backups_created", res_bkp_only)

    # =========================================================================
    # 15. CLUSTER ROLES (GREEN/ORANGE BACKUP), SAFE DUP REMOVAL, BACKUP TARGETS & MOVE BACKUPS
    # =========================================================================
    def test_new_duplicate_tags_backup_target_selection_and_move_backups(self):
        drive_c = self.root / "DriveC_Dedicated"
        drive_c.mkdir(parents=True, exist_ok=True)

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="VSN-NEW-A",
                volume_label="Dysk_Zrodlo_A",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=10 * 1024**3,
            )
            vol_b = WindowsVolume(
                volume_serial_number="VSN-NEW-B",
                volume_label="Dysk_B",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=15 * 1024**3,
            )
            vol_c = WindowsVolume(
                volume_serial_number="VSN-NEW-C",
                volume_label="Dysk_Dedykowany_Backup_C",
                last_known_drive_letter=str(drive_c),
                is_online=True,
                total_space_bytes=500 * 1024**3,
                free_space_bytes=450 * 1024**3,
            )
            db.add_all([vol_a, vol_b, vol_c])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            db.refresh(vol_c)
            va_id, vb_id, vc_id = vol_a.id, vol_b.id, vol_c.id

            # Cluster 1 (1 item): Missing backup -> ORYGINAL
            p_missing = create_synthetic_jpeg(self.drive_a / "only_one.jpg", color=(10, 20, 30))

            # Cluster 2 (2 items, same device A): ORYGINAL + pomaranczowy BACKUP
            p_orange_1 = create_synthetic_jpeg(self.drive_a / "orange_1.jpg", color=(40, 50, 60))
            p_orange_2 = self.drive_a / "orange_2.jpg"
            p_orange_2.write_bytes(p_orange_1.read_bytes())

            # Cluster 3 (2 items, cross device A and B outside daf_backup): ORYGINAL + zielony BACKUP
            p_green_1 = create_synthetic_jpeg(self.drive_a / "green_1.jpg", color=(70, 80, 90))
            p_green_2 = self.drive_b / "loose_bkp" / "green_2.jpg"
            p_green_2.parent.mkdir(parents=True, exist_ok=True)
            p_green_2.write_bytes(p_green_1.read_bytes())

            # Cluster 4 (3 items on A: A1, A2, A3): ORYGINAL + pomaranczowy BACKUP + czerwony DUPLIKAT
            p_trip_1 = create_synthetic_jpeg(self.drive_a / "trip_1.jpg", color=(110, 120, 130))
            p_trip_2 = self.drive_a / "trip_2.jpg"
            p_trip_3 = self.drive_a / "trip_3.jpg"
            p_trip_2.write_bytes(p_trip_1.read_bytes())
            p_trip_3.write_bytes(p_trip_1.read_bytes())

            it_missing = FileScannerService.process_file(p_missing, vol_a)
            it_or1 = FileScannerService.process_file(p_orange_1, vol_a)
            it_or2 = FileScannerService.process_file(p_orange_2, vol_a)
            it_gr1 = FileScannerService.process_file(p_green_1, vol_a)
            it_gr2 = FileScannerService.process_file(p_green_2, vol_b)
            it_tr1 = FileScannerService.process_file(p_trip_1, vol_a)
            it_tr2 = FileScannerService.process_file(p_trip_2, vol_a)
            it_tr3 = FileScannerService.process_file(p_trip_3, vol_a)
            db.add_all([it_missing, it_or1, it_or2, it_gr1, it_gr2, it_tr1, it_tr2, it_tr3])
            db.commit()

            svc = StorageAutomationService(db)
            svc.flag_duplicate_clusters()
            db.expire_all()

            # Verify cluster 1..4 roles
            self.assertEqual(db.get(MediaItem, it_missing.id).duplicate_role, "original")
            self.assertEqual(db.get(MediaItem, it_or1.id).duplicate_role, "original")
            self.assertEqual(db.get(MediaItem, it_or2.id).duplicate_role, "backup_orange")
            self.assertEqual(db.get(MediaItem, it_gr1.id).duplicate_role, "original")
            self.assertEqual(db.get(MediaItem, it_gr2.id).duplicate_role, "backup_green")
            self.assertEqual(db.get(MediaItem, it_tr1.id).duplicate_role, "original")
            self.assertEqual(db.get(MediaItem, it_tr2.id).duplicate_role, "backup_orange")
            self.assertEqual(db.get(MediaItem, it_tr3.id).duplicate_role, "duplicate")

            # Stats sum invariant: missing(1) + green(1) + orange(2) == total_unique(4)
            # Mock disk_usage per drive path so Drive C's 450 GB capacity beats Drive B's 15 GB cleanly
            def _mock_usage(path_arg):
                s = str(path_arg)
                if "DriveC_Dedicated" in s:
                    return shutil._ntuple_diskusage(500 * 1024**3, 50 * 1024**3, 450 * 1024**3)
                if "DriveB" in s:
                    return shutil._ntuple_diskusage(50 * 1024**3, 35 * 1024**3, 15 * 1024**3)
                return shutil._ntuple_diskusage(50 * 1024**3, 40 * 1024**3, 10 * 1024**3)

            with patch("app.storage_ops.shutil.disk_usage", side_effect=_mock_usage):
                plan_default = svc.build_backup_and_dedup_plan()
                st = plan_default["stats"]
                self.assertEqual(st["missing_backups"] + st["green_backups"] + st["orange_backups"], st["total_unique"])
                self.assertEqual(st["total_unique"], 4)
                # Default target drive automatically picks empty/dedicated largest Drive C!
                self.assertEqual(plan_default["default_target_volume_ids"], [vc_id])

                # When minimize_transfers=False, green BACKUP on B (non-target) is scheduled for remote move onto Drive C!
                plan_consol = svc.build_backup_and_dedup_plan(minimize_transfers=False)
                self.assertTrue(any(r["backup_item_id"] == it_gr2.id and r["dst_vol_id"] == vc_id for r in plan_consol["remote_move_rows"]))

            # Test Refresh Duplicate Tags endpoint
            r_ref = self.client.post("/actions/refresh-duplicate-tags")
            self.assertEqual(r_ref.status_code, 200)
            self.assertIn("ORYGINAŁ, BACKUP (zielony/pomarańczowy), DUPLIKAT", r_ref.text)

            # Safety Gate: Trying to delete Original or sole Backup (or when duplicates_only=True) must be BLOCKED
            r_blk = self.client.post(
                "/actions/move-to-trash",
                data={"scope": "selected", "duplicates_only": "true", "selected_ids": [it_or2.id]},
            )
            self.assertEqual(r_blk.status_code, 200)
            self.assertIn("Zablokowano", r_blk.text)
            self.assertTrue(p_orange_2.exists())

            # Test Safe Duplicate Removal on 3rd copy (to Kosz: logical-only, physical file untouched!) when Volume A is online
            r_ok_tr = self.client.post(
                "/actions/move-to-trash",
                data={"scope": "selected", "duplicates_only": "true", "selected_ids": [it_tr3.id]},
            )
            self.assertEqual(r_ok_tr.status_code, 200)
            db.expire_all()
            self.assertTrue(p_trip_3.exists(), "Logical Kosz must NOT delete or move physical file on disk")
            self.assertTrue(db.get(MediaItem, it_tr3.id).is_trashed)

            # Simulate an offline volume holding a duplicate -> must prompt user to connect the offline disk
            vol_a_db = db.get(WindowsVolume, va_id)
            p_trip_3_offline = self.drive_a / "trip_4_off.jpg"
            p_trip_3_offline.write_bytes(p_trip_1.read_bytes())
            it_tr4 = FileScannerService.process_file(p_trip_3_offline, vol_a_db)
            db.add(it_tr4)
            db.commit()
            svc.flag_duplicate_clusters()
            vol_a_db.is_online = False
            vol_a_db.last_known_drive_letter = "Z:\\OfflineMissing\\"
            db.commit()

            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"VSN-NEW-B": (str(self.drive_b), "Dysk_B"), "VSN-NEW-C": (str(drive_c), "Dysk_Dedykowany_Backup_C")}):
                r_off_dup = self._delete_permanently_two_phase(
                    {"scope": "selected", "duplicates_only": "true", "selected_ids": [it_tr4.id]},
                )
                self.assertEqual(r_off_dup.status_code, 200)
                self.assertIn("Podłącz kolejny dysk", r_off_dup.text)
                self.assertIn("Kontynuuj na podłączonym dysku", r_off_dup.text)

            # Restore Volume A online & run backup onto Drive C (minimize_transfers=True -> green_2 relocated on B, new copies go to C)
            vol_a_db.is_online = True
            vol_a_db.last_known_drive_letter = str(self.drive_a)
            db.commit()

            r_exec_c = self.client.post(
                "/actions/execute-backup",
                data={
                    "target_vols_submitted": "1",
                    "target_volume_ids": [str(vc_id)],
                    "minimize_submitted": "1",
                    "minimize_transfers": "true",
                },
            )
            self.assertEqual(r_exec_c.status_code, 200)
            self.assertTrue(any((drive_c / "daf_backup").rglob("only_one.jpg")))
            self.assertTrue(any((self.drive_b / "daf_backup").rglob("green_2.jpg")))

            # Test duplicate verification fail block & collision rehash during execute_safe_duplicate_removal
            db.expire_all()
            with patch.object(FastHasher, "verify_files_identical_100_percent", return_value=(False, "SimDupMismatch", None)):
                res_vf_dup = svc.execute_safe_duplicate_removal(
                    [it_tr4.id], delete_mode="trash", force_duplicate_verification=True,
                    trash_dir=self.root / "kosz", thumbnails_dir=self.thumbs_dir,
                )
                self.assertEqual(res_vf_dup["blocked_verify"], 1, res_vf_dup)

            # Reset it_tr4 hash and safely delete it while Drive A is online
            it_tr4_re = db.get(MediaItem, it_tr4.id)
            it_tr4_re.file_hash = db.get(MediaItem, it_tr1.id).file_hash
            db.commit()
            svc.flag_duplicate_clusters()
            res_ok_perm = svc.execute_safe_duplicate_removal(
                [it_tr4.id], delete_mode="permanent", force_duplicate_verification=True,
                trash_dir=self.root / "kosz", thumbnails_dir=self.thumbs_dir,
            )
            self.assertEqual(res_ok_perm["processed_count"], 1)

            # Test execute_safe_duplicate_removal with empty item_ids
            res_empty_rem = svc.execute_safe_duplicate_removal(["nonexistent-id"])
            self.assertEqual(res_empty_rem["processed_count"], 0)

            # Test POST /actions/backup-panel recalculation with minimize_transfers=False and recalc_pairs=true
            r_recalc = self.client.post(
                "/actions/backup-panel",
                data={
                    "target_vols_submitted": "1",
                    "target_volume_ids": [str(vc_id)],
                    "minimize_submitted": "1",
                    "recalc_pairs": "true",
                },
            )
            self.assertEqual(r_recalc.status_code, 200)

            # Test New Feature 2c: Move Backups Between Devices (Move green_2 from B -> C, blocked if target holds Original!)
            r_mv_panel = self.client.post(
                "/actions/move-backups-panel",
                data={"source_volume_id": str(vb_id), "target_volume_id": str(vc_id)},
            )
            self.assertEqual(r_mv_panel.status_code, 200)
            self.assertIn("green_2.jpg", r_mv_panel.text)

            # Trying to move backups from B -> A (which holds green_1 Original!) MUST block green_2 due to same-device rule!
            r_mv_blocked = self.client.post(
                "/actions/move-backups-panel",
                data={"source_volume_id": str(vb_id), "target_volume_id": str(va_id)},
            )
            self.assertEqual(r_mv_blocked.status_code, 200)
            self.assertIn("Pominięto (zasada różnych dysków)", r_mv_blocked.text)

            # Edge cases for execute_move_backups: same drive, empty movable, offline drive
            self.assertEqual(svc.execute_move_backups(vb_id, vb_id)["status"], "error")
            self.assertEqual(svc.execute_move_backups(vb_id, va_id)["status"], "empty")
            vol_c_db = db.get(WindowsVolume, vc_id)
            vol_c_db.is_online = False
            vol_c_db.last_known_drive_letter = "Z:\\OfflineC\\"
            db.commit()
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"VSN-NEW-A": (str(self.drive_a), "Dysk_Zrodlo_A"), "VSN-NEW-B": (str(self.drive_b), "Dysk_B")}):
                self.assertEqual(svc.execute_move_backups(vb_id, vc_id)["status"], "offline")
            vol_c_db.is_online = True
            vol_c_db.last_known_drive_letter = str(drive_c)
            db.commit()

            # Executing move B -> C succeeds with 100% verification!
            r_mv_exec = self.client.post(
                "/actions/execute-move-backups",
                data={"source_volume_id": str(vb_id), "target_volume_id": str(vc_id)},
            )
            self.assertEqual(r_mv_exec.status_code, 200)
            self.assertIn("Przeniesiono backupów", r_mv_exec.text)
            self.assertTrue(any((drive_c / "daf_backup").rglob("green_2.jpg")))
            # Immutable daf_backup/ policy: existing file in source daf_backup/ is NEVER deleted
            self.assertTrue(any((self.drive_b / "daf_backup").rglob("green_2.jpg")))

            # Now test remote backup consolidation (minimize_transfers=False) via /actions/execute-backup
            p_rem_orig = create_synthetic_jpeg(self.drive_a / "remote_consol.jpg", color=(150, 60, 20))
            p_rem_bkp = self.drive_b / "daf_backup" / "2024" / "01" / "01" / "remote_consol.jpg"
            p_rem_bkp.parent.mkdir(parents=True, exist_ok=True)
            p_rem_bkp.write_bytes(p_rem_orig.read_bytes())
            it_rem_o = FileScannerService.process_file(p_rem_orig, vol_a_db)
            it_rem_b = FileScannerService.process_file(p_rem_bkp, db.get(WindowsVolume, vb_id))
            db.add_all([it_rem_o, it_rem_b])
            db.commit()

            r_exec_consol = self.client.post(
                "/actions/execute-backup",
                data={
                    "target_vols_submitted": "1",
                    "target_volume_ids": [str(vc_id)],
                    "minimize_submitted": "1",
                },
            )
            self.assertEqual(r_exec_consol.status_code, 200)
            self.assertTrue(any((drive_c / "daf_backup").rglob("remote_consol.jpg")))
            # Immutable daf_backup/ policy: existing source file inside daf_backup/ is NEVER deleted
            self.assertTrue(p_rem_bkp.exists())

            # Also verify duplicate view in all 3 modes (table, compact, grid) + detail modal
            self.assertEqual(self.client.get("/media/search?duplicates_only=true&view_mode=table").status_code, 200)
            self.assertEqual(self.client.get("/media/search?duplicates_only=true&view_mode=compact").status_code, 200)
            self.assertEqual(self.client.get(f"/media/{it_tr1.id}/detail").status_code, 200)

    def test_followup_features_green_original_device_stats_active_pairs_and_async_dup_delete(self):
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        from app.routers import actions as actions_router

        drive_c = self.root / "DriveC"
        drive_c.mkdir(parents=True, exist_ok=True)
        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="VSN-FU-A",
                volume_label="DiskA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            vol_b = WindowsVolume(
                volume_serial_number="VSN-FU-B",
                volume_label="DiskB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=30 * 1024**3,
            )
            vol_c = WindowsVolume(
                volume_serial_number="VSN-FU-C",
                volume_label="DiskC",
                last_known_drive_letter=str(drive_c),
                is_online=True,
                total_space_bytes=500 * 1024**3,
                free_space_bytes=450 * 1024**3,
            )
            db.add_all([vol_a, vol_b, vol_c])
            db.commit()
            for v in (vol_a, vol_b, vol_c):
                db.refresh(v)

            # Photo 1: Original on A + Vault copy in B/daf_backup/ + Redundant duplicate on A -> Green ORYGINAL!
            p1_o = create_synthetic_jpeg(self.drive_a / "vault_backed.jpg", color=(25, 180, 90))
            p1_v = self.drive_b / "daf_backup" / "2024" / "05" / "10" / "vault_backed.jpg"
            p1_v.parent.mkdir(parents=True, exist_ok=True)
            p1_v.write_bytes(p1_o.read_bytes())
            p1_d = self.drive_a / "vault_backed_extra.jpg"
            p1_d.write_bytes(p1_o.read_bytes())

            # Photo 2: Original on A + Redundant same-disk copy (no daf_backup/) -> Blue ORYGINAL + Orange BACKUP!
            p2_o = create_synthetic_jpeg(self.drive_a / "no_vault.jpg", color=(210, 40, 40))
            p2_b = self.drive_a / "no_vault_copy.jpg"
            p2_b.write_bytes(p2_o.read_bytes())

            # Photo 3: Original on A + Green copy outside daf_backup/ on B (for per-vol reloc_green counter)
            p3_o = create_synthetic_jpeg(self.drive_a / "outside_bkp.jpg", color=(60, 70, 200))
            p3_b = self.drive_b / "outside_bkp_copy.jpg"
            p3_b.write_bytes(p3_o.read_bytes())

            items = [
                FileScannerService.process_file(p1_o, vol_a),
                FileScannerService.process_file(p1_v, vol_b),
                FileScannerService.process_file(p1_d, vol_a),
                FileScannerService.process_file(p2_o, vol_a),
                FileScannerService.process_file(p2_b, vol_a),
                FileScannerService.process_file(p3_o, vol_a),
                FileScannerService.process_file(p3_b, vol_b),
            ]
            db.add_all(items)
            db.commit()

            svc = StorageAutomationService(db)
            svc.flag_duplicate_clusters()

            # Req 1: daf_backup/ hidden from gallery, but ORYGINAL turns green for p1_o and stays blue for p2_o
            for vmode in ("grid", "table", "compact"):
                r_dup = self.client.get(f"/media/search?duplicates_only=true&view_mode={vmode}")
                self.assertEqual(r_dup.status_code, 200)
                self.assertNotIn("daf_backup/2024/05/10/vault_backed.jpg", r_dup.text)
                self.assertIn("bg-emerald-600", r_dup.text)
                self.assertIn("bg-sky-700", r_dup.text)
                self.assertIn("DiskB", r_dup.text)
            r_det = self.client.get(f"/media/{items[0].id}/detail")
            self.assertEqual(r_det.status_code, 200)
            self.assertIn("ORYGINAŁ ✓", r_det.text)

            # Req 3: Per-volume backup counts (valid in daf_backup/, reloc_green, reloc_orange)
            plan = svc.build_backup_and_dedup_plan(target_volume_ids=[vol_c.id], minimize_transfers=True)
            vsum_a = plan["volume_space_summary"][vol_a.id]
            vsum_b = plan["volume_space_summary"][vol_b.id]
            self.assertEqual(vsum_a["bkp_reloc_orange_count"], 1)
            self.assertEqual(vsum_b["bkp_valid_in_vault_count"], 1)
            self.assertEqual(vsum_b["bkp_reloc_green_count"], 1)

            # Req 4 & 5: Backup panel shows only scheduled active pairs, shows per-drive backup breakdown,
            # and omits the "Przygotuj procedurę Offline" button. When A<->C has work, B<->C has 0 work -> hidden!
            def _mock_disk_usage(path_str):
                s = str(path_str)
                if "DriveC" in s:
                    return SimpleNamespace(total=500 * 1024**3, used=50 * 1024**3, free=450 * 1024**3)
                if "DriveB" in s:
                    return SimpleNamespace(total=100 * 1024**3, used=70 * 1024**3, free=30 * 1024**3)
                return SimpleNamespace(total=100 * 1024**3, used=60 * 1024**3, free=40 * 1024**3)

            with patch("app.storage_ops.shutil.disk_usage", side_effect=_mock_disk_usage), \
                 patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={
                     "VSN-FU-A": (str(self.drive_a), "DiskA"),
                     "VSN-FU-B": (str(self.drive_b), "DiskB"),
                     "VSN-FU-C": (str(drive_c), "DiskC"),
                 }):
                r_bp = self.client.get("/actions/backup-panel")
                self.assertEqual(r_bp.status_code, 200)
                self.assertIn("Prawidłowe (w <code>daf_backup/</code>):", r_bp.text)
                self.assertIn("zielone:", r_bp.text)
                self.assertIn("pomarańczowe:", r_bp.text)
                self.assertNotIn("Przygotuj procedurę Offline i zapisz raport (.txt)", r_bp.text)
                # DiskB <-> DiskC has 0 ops and 0 important albums -> must NOT appear in Harmonogram!
                self.assertNotIn("DiskB", [p["vol_a"].volume_label for p in plan["scheduled_pairs"] if p["vol_b"].id == vol_c.id])
                self.assertNotIn(f'value="{vol_b.id}:{vol_c.id}"', r_bp.text)

                # Req 2 (Bug): Async permanent duplicate deletion returns live polling progress bar & finishes cleanly
                it_dup_extra = db.get(MediaItem, items[2].id)
                self.assertEqual(it_dup_extra.duplicate_role, "duplicate")
                r_async_del = self._delete_permanently_two_phase(
                    {"scope": "selected", "selected_ids": [it_dup_extra.id], "duplicates_only": "true"},
                    headers={"HX-Request": "true"},
                )
                self.assertEqual(r_async_del.status_code, 200)
                self.assertTrue(
                    ("/actions/dup-delete-status" in r_async_del.text) or ("Trwale usunięto z dysku" in r_async_del.text)
                )
                for _ in range(100):
                    if not actions_router.DUP_DELETE_STATE["running"]:
                        break
                    time.sleep(0.05)
                r_poll = self.client.get("/actions/dup-delete-status")
                self.assertEqual(r_poll.status_code, 200)
                self.assertIn("Trwale usunięto z dysku", r_poll.text)
                self.assertFalse(p1_d.exists())
                self.assertTrue(p1_o.exists())
                self.assertTrue(p1_v.exists())

                # Test stop endpoint works without error
                r_stop = self.client.post("/actions/dup-delete-stop")
                self.assertEqual(r_stop.status_code, 200)

                # Verify huge (>1.2MB / >33k IDs) selected_ids_json does NOT hit 400 Bad Request or SQLite 32766 limit
                # and verify live status panel includes Etap banner when DUP_DELETE_STATE is running
                p1_d2 = self.drive_a / "vault_backed_extra2.jpg"
                p1_d2.write_bytes(p1_o.read_bytes())
                it_extra2 = FileScannerService.process_file(p1_d2, vol_a)
                db.add(it_extra2)
                db.commit()
                svc.flag_duplicate_clusters()
                huge_ids = [it_extra2.id] + [f"00000000-0000-0000-0000-{i:012d}" for i in range(36000)]
                r_huge = self._delete_permanently_two_phase(
                    {
                        "scope": "selected",
                        "selected_ids_json": json.dumps(huge_ids),
                        "duplicates_only": "true",
                    },
                    headers={"HX-Request": "true"},
                )
                self.assertEqual(r_huge.status_code, 200)
                r_st_panel = self.client.get("/actions/status-panel")
                self.assertEqual(r_st_panel.status_code, 200)
                self.assertIn("status-panel-root", r_st_panel.text)
                for _ in range(100):
                    if not actions_router.DUP_DELETE_STATE["running"]:
                        break
                    time.sleep(0.05)
                self.assertFalse(p1_d2.exists())

                # Verify prompt priorities, per-drive delete counts, and skip_db_backup=1 on continuation:
                # Create DiskD (offline, 0 duplicates, only reference copies) + DiskE (offline, 2 duplicates)
                drive_d_off = self.root / "DriveD_OffRef"
                drive_e_off = self.root / "DriveE_OffDups"
                drive_d_off.mkdir(parents=True, exist_ok=True)
                drive_e_off.mkdir(parents=True, exist_ok=True)
                vol_d = WindowsVolume(
                    volume_serial_number="DD00-0001",
                    volume_label="DiskD_RefOnly",
                    last_known_drive_letter=str(drive_d_off),
                    is_online=False,
                )
                vol_e = WindowsVolume(
                    volume_serial_number="EE00-0002",
                    volume_label="DiskE_HasDups",
                    last_known_drive_letter=str(drive_e_off),
                    is_online=False,
                )
                db.add_all([vol_d, vol_e])
                db.commit()
                db.refresh(vol_d)
                db.refresh(vol_e)

                # Online duplicate on DiskA whose ONLY surviving reference copies sit on offline DiskD (0 dups on D!)
                # + 2 duplicates on offline DiskE referencing DiskA/DiskB
                p_ref_only = create_synthetic_jpeg(drive_d_off / "ref_only.jpg", color=(12, 34, 56))
                p_bkp_d = drive_d_off / "daf_backup" / "2024" / "06" / "01" / "ref_only.jpg"
                p_bkp_d.parent.mkdir(parents=True, exist_ok=True)
                p_bkp_d.write_bytes(p_ref_only.read_bytes())
                p_dup_a_from_d = self.drive_a / "ref_only_dup.jpg"
                p_dup_a_from_d.write_bytes(p_ref_only.read_bytes())

                p_e_orig = create_synthetic_jpeg(self.drive_a / "e_src.jpg", color=(90, 80, 70))
                p_e_orig_a2 = self.drive_a / "e_src_copy2.jpg"
                p_e_orig_a3 = self.drive_a / "e_src_copy3.jpg"
                p_e_orig_a2.write_bytes(p_e_orig.read_bytes())
                p_e_orig_a3.write_bytes(p_e_orig.read_bytes())
                p_e_bkp = self.drive_b / "daf_backup" / "2024" / "06" / "01" / "e_src.jpg"
                p_e_bkp.parent.mkdir(parents=True, exist_ok=True)
                p_e_bkp.write_bytes(p_e_orig.read_bytes())
                p_e_dup1 = drive_e_off / "e_dup1.jpg"
                p_e_dup2 = drive_e_off / "e_dup2.jpg"
                p_e_dup1.write_bytes(p_e_orig.read_bytes())
                p_e_dup2.write_bytes(p_e_orig.read_bytes())

                extra_items = [
                    FileScannerService.process_file(p_ref_only, vol_d),
                    FileScannerService.process_file(p_bkp_d, vol_d),
                    FileScannerService.process_file(p_dup_a_from_d, vol_a),
                    FileScannerService.process_file(p_e_orig, vol_a),
                    FileScannerService.process_file(p_e_bkp, vol_b),
                    FileScannerService.process_file(p_e_dup1, vol_e),
                    FileScannerService.process_file(p_e_dup2, vol_e),
                    FileScannerService.process_file(p_e_orig_a2, vol_a),
                    FileScannerService.process_file(p_e_orig_a3, vol_a),
                ]
                db.add_all(extra_items)
                db.commit()
                svc.flag_duplicate_clusters()

                # Mark DiskD offline AFTER scanning (A, B, C online; D and E offline)
                dup_target_ids = [extra_items[2].id, extra_items[5].id, extra_items[6].id]
                r_off_plan = self._delete_permanently_two_phase(
                    {
                        "scope": "selected",
                        "selected_ids": dup_target_ids,
                        "duplicates_only": "true",
                    },
                    headers={"HX-Request": "true"},
                )
                self.assertEqual(r_off_plan.status_code, 200)
                for _ in range(100):
                    if not actions_router.DUP_DELETE_STATE["running"]:
                        break
                    time.sleep(0.05)
                r_off_poll = self.client.get("/actions/dup-delete-status")
                self.assertEqual(r_off_poll.status_code, 200)
                # Online duplicate on DiskA (even though DiskD refs are offline!) must be verified & deleted right away!
                # Prompt must ask ONLY for DiskE_HasDups (2 duplicates), NEVER DiskD_RefOnly or online DiskA!
                self.assertFalse(p_dup_a_from_d.exists())
                self.assertIn("DiskE_HasDups", r_off_poll.text)
                self.assertNotIn("DiskD_RefOnly", r_off_poll.text)
                self.assertNotIn("<u>DiskA</u>", r_off_poll.text)
                self.assertIn("do usunięcia na tym dysku: <strong>2</strong> duplikatów", r_off_poll.text)
                self.assertIn("▶️ Kontynuuj na podłączonym dysku (2 szt.)", r_off_poll.text)
                self.assertIn("&quot;skip_db_backup&quot;: &quot;true&quot;", r_off_poll.text)
                self.assertIn("&quot;continue_job_id&quot;:", r_off_poll.text)

                latest_job = db.scalars(select(BackgroundJob).order_by(BackgroundJob.id.desc())).first()
                self.assertIsNotNone(latest_job)

                # Clicking continue WITHOUT deletion_token MUST BE BLOCKED (C-02);
                # clicking the rendered button WITH server-issued continuation token must NOT call BackupManager.create_backup
                # and must reuse the existing BackgroundJob ID.
                r_cont_no_tok = self.client.post(
                    "/actions/delete-permanently",
                    headers={"HX-Request": "true"},
                    data={
                        "scope": "selected",
                        "selected_ids": [extra_items[5].id, extra_items[6].id],
                        "duplicates_only": "true",
                        "skip_db_backup": "true",
                        "continue_job_id": str(latest_job.id),
                    },
                )
                self.assertIn("wymagany jest ważny token potwierdzenia", r_cont_no_tok.text)

                import html as _html_mod
                import re as _re_mod
                m_cont_vals = _re_mod.search(r'hx-vals="([^"]+)"', r_off_poll.text)
                self.assertIsNotNone(m_cont_vals)
                cont_btn_payload = json.loads(_html_mod.unescape(m_cont_vals.group(1)))
                self.assertIn("deletion_token", cont_btn_payload)

                with patch("app.routers.actions.BackupManager.create_backup") as mock_bkp:
                    r_cont = self.client.post(
                        "/actions/delete-permanently",
                        headers={"HX-Request": "true"},
                        data=cont_btn_payload,
                    )
                    self.assertEqual(r_cont.status_code, 200)
                    for _ in range(100):
                        if not actions_router.DUP_DELETE_STATE["running"]:
                            break
                        time.sleep(0.05)
                    mock_bkp.assert_not_called()
                    self.assertEqual(actions_router.DUP_DELETE_STATE["job_id"], latest_job.id)

                # Zadanie 2 Przyczyna B + Zadanie 4: Protected offline Original passed to safe removal is blocked by safety,
                # and resuming a duplicate job via /actions/process-control uses safe verification (never raw delete!).
                res_prot = svc.execute_safe_duplicate_removal([extra_items[0].id], delete_mode="permanent", force_duplicate_verification=True)
                self.assertEqual(res_prot["blocked_safety"], 1)
                self.assertEqual(len(res_prot["offline_items"]), 0)

                safe_job = BackgroundJob(
                    job_type="delete_permanently",
                    title="Safe Dup Job Resume",
                    status="STOPPED",
                    processed_count=0,
                    total_count=1,
                    params_json=json.dumps({"is_duplicate_removal": True, "delete_mode": "permanent"}),
                    remaining_ids_json=json.dumps([extra_items[0].id]),
                )
                db.add(safe_job)
                db.commit()
                db.refresh(safe_job)
                r_res = self.client.post(
                    "/actions/process-control",
                    data={"action": "resume", "job_id": str(safe_job.id), "run_sync": "true"},
                )
                self.assertEqual(r_res.status_code, 200)
                self.assertIsNotNone(db.get(MediaItem, extra_items[0].id))  # Original MUST survive!

                # Edge case: unverified online video duplicate whose Original + Backup are both on OFFLINE DiskD_RefOnly (del_cnt=0).
                # Must NOT raise AttributeError on None next_offline_prompt or leave job stuck in WAITING_FOR_DEVICE.
                import uuid as _uuid
                vid_payload = b"ftypisom_edge_video_payload_" * 40
                p_vid_dup = self.drive_a / "v_dup.mp4"
                p_vid_dup.write_bytes(vid_payload)
                now_vid_dt = datetime.datetime(2024, 6, 1, 12, 0, 0)
                vid_specs = [
                    (vol_d.id, "v_orig.mp4", "v_orig.mp4", "original", False),
                    (vol_d.id, "daf_backup/2024/06/01/v_orig.mp4", "v_orig.mp4", "backup_same_dev", True),
                    (vol_a.id, "v_dup.mp4", "v_dup.mp4", "duplicate", False),
                ]
                vid_items = [
                    MediaItem(
                        id=str(_uuid.uuid4()),
                        volume_id=v_id,
                        relative_path=rel_p,
                        file_name=fname,
                        file_size_bytes=len(vid_payload),
                        file_hash="hash_edge_vid_001",
                        hash_algorithm="sha256_fast_sample",
                        verified_sha256=None,
                        media_type="video",
                        mime_type="video/mp4",
                        fs_ctime=now_vid_dt,
                        fs_mtime=now_vid_dt,
                        effective_date=now_vid_dt,
                        thumbnail_relative_path="dummy_vid_thumb.webp",
                        duplicate_role=d_role,
                        is_backup_copy=is_bkp,
                    )
                    for v_id, rel_p, fname, d_role, is_bkp in vid_specs
                ]
                db.add_all(vid_items)
                db.commit()
                res_vid_edge = svc.execute_safe_duplicate_removal(
                    [vid_items[2].id], delete_mode="permanent", force_duplicate_verification=True
                )
                self.assertEqual(res_vid_edge["blocked_verify"], 1)
                self.assertEqual(res_vid_edge["offline_items"], [])
                self.assertEqual(res_vid_edge["offline_drives"], [])
                self.assertIsNone(res_vid_edge["next_offline_prompt"])
                r_vid_edge = actions_router._build_safe_removal_html_response(
                    "/actions/delete-permanently", "permanent", res_vid_edge, safe_job.id, db
                )
                self.assertEqual(r_vid_edge.status_code, 200)
                db.refresh(safe_job)
                self.assertEqual(safe_job.status, "COMPLETED")
                self.assertTrue(p_vid_dup.exists())

                # Reconcile already-unlinked duplicate on ONLINE drive (interrupted run recovery)
                # + verify concurrent GET /actions/status-panel polling works cleanly during duplicate deletion
                p_rec_o = create_synthetic_jpeg(self.drive_a / "rec_orig.jpg", color=(33, 66, 99))
                p_rec_b = self.drive_b / "daf_backup" / "2024" / "07" / "01" / "rec_orig.jpg"
                p_rec_b.parent.mkdir(parents=True, exist_ok=True)
                p_rec_b.write_bytes(p_rec_o.read_bytes())
                p_rec_ghost = self.drive_a / "rec_ghost_dup.jpg"
                p_rec_live = self.drive_a / "rec_live_dup.jpg"
                p_rec_ghost.write_bytes(p_rec_o.read_bytes())
                p_rec_live.write_bytes(p_rec_o.read_bytes())
                rec_items = [
                    FileScannerService.process_file(p_rec_o, vol_a),
                    FileScannerService.process_file(p_rec_b, vol_b),
                    FileScannerService.process_file(p_rec_ghost, vol_a),
                    FileScannerService.process_file(p_rec_live, vol_a),
                ]
                db.add_all(rec_items)
                db.commit()
                svc.flag_duplicate_clusters()

                # Simulate crash after physical unlink of ghost duplicate before DB commit
                p_rec_ghost.unlink()
                poll_statuses: list[int] = []
                actions_router.DUP_DELETE_STATE["running"] = True
                try:
                    def _poll_during_delete(_done: int, _tot: int, _msg: str, _drv: str) -> None:
                        r_poll_mid = self.client.get("/actions/status-panel")
                        poll_statuses.append(r_poll_mid.status_code)

                    with patch.object(
                        StorageAutomationService,
                        "refresh_volumes_online_status",
                        wraps=svc.refresh_volumes_online_status,
                    ) as spy_refresh:
                        res_rec = svc.execute_safe_duplicate_removal(
                            [rec_items[2].id, rec_items[3].id],
                            delete_mode="permanent",
                            force_duplicate_verification=True,
                            job_id=safe_job.id,
                            trash_dir=self.root / "kosz",
                            thumbnails_dir=self.thumbs_dir,
                            progress_cb=_poll_during_delete,
                        )
                        # Called once at start of execute_safe_duplicate_removal, NEVER inside status-panel polls
                        self.assertEqual(spy_refresh.call_count, 1)
                finally:
                    actions_router.DUP_DELETE_STATE["running"] = False

                self.assertTrue(poll_statuses)
                self.assertTrue(all(sc == 200 for sc in poll_statuses))
                self.assertEqual(res_rec["missing_on_disk"], 1)
                self.assertEqual(res_rec["processed_count"], 2)
                self.assertIsNone(db.get(MediaItem, rec_items[2].id))
                self.assertIsNone(db.get(MediaItem, rec_items[3].id))
                self.assertIsNotNone(db.get(MediaItem, rec_items[0].id))
                self.assertIsNotNone(db.get(MediaItem, rec_items[1].id))

                # C-03: ReadOnly Windows attribute duplicate MUST BE PROTECTED (never strip Read-Only or delete!)
                import stat as _stat
                p_ro_dup = self.drive_a / "rec_ro_dup.jpg"
                p_ro_dup.write_bytes(p_rec_o.read_bytes())
                it_ro_dup = FileScannerService.process_file(p_ro_dup, vol_a)
                db.add(it_ro_dup)
                db.commit()
                svc.flag_duplicate_clusters()
                os.chmod(p_ro_dup, _stat.S_IREAD)
                try:
                    res_ro = svc.execute_safe_duplicate_removal(
                        [it_ro_dup.id],
                        delete_mode="permanent",
                        force_duplicate_verification=True,
                        trash_dir=self.root / "kosz",
                        thumbnails_dir=self.thumbs_dir,
                    )
                    self.assertEqual(res_ro["physical_count"], 0)
                    self.assertEqual(res_ro["processed_count"], 0)
                    self.assertEqual(res_ro["blocked_safety"], 1)
                    self.assertTrue(p_ro_dup.exists(), "Read-Only file must stay intact on disk!")
                    self.assertFalse(bool(os.stat(p_ro_dup).st_mode & _stat.S_IWRITE), "Read-Only attribute must NOT be stripped!")
                    self.assertIsNotNone(db.get(MediaItem, it_ro_dup.id))
                finally:
                    os.chmod(p_ro_dup, _stat.S_IWRITE)

                # Deleted duplicate paths must be preserved in deleted_duplicate_history and searchable without restoring duplicates
                hist_rows = list(db.scalars(select(DeletedDuplicateHistory).where(DeletedDuplicateHistory.file_hash == rec_items[0].file_hash)).all())
                hist_names = {h.file_name for h in hist_rows}
                self.assertIn("rec_live_dup.jpg", hist_names)
                self.assertIn("rec_ghost_dup.jpg", hist_names)

                for vm in ("grid", "table"):
                    r_hist_q = self.client.get(f"/media/search?q=rec_live_dup.jpg&view_mode={vm}")
                    self.assertEqual(r_hist_q.status_code, 200)
                    self.assertIn("rec_orig.jpg", r_hist_q.text)
                    self.assertIn("Dawny duplikat", r_hist_q.text)

                r_hist_det = self.client.get(f"/media/{rec_items[0].id}/detail")
                self.assertEqual(r_hist_det.status_code, 200)
                self.assertIn("Historia usuniętych duplikatów powiązanych z tym plikiem", r_hist_det.text)
                self.assertIn("rec_live_dup.jpg", r_hist_det.text)

                # Verify historical recovery from a pre-deletion SQLite backup file
                p_hist_bkp_dup = self.drive_a / "rec_from_backup_hist_dup.jpg"
                p_hist_bkp_dup.write_bytes(p_rec_o.read_bytes())
                it_hist_bkp_dup = FileScannerService.process_file(p_hist_bkp_dup, vol_a)
                db.add(it_hist_bkp_dup)
                db.commit()
                snap_bkp_path = self.root / "pre_delete_snap.db"
                src_conn = sqlite3.connect(self.db_path.as_posix())
                dst_conn = sqlite3.connect(snap_bkp_path.as_posix())
                src_conn.backup(dst_conn)
                dst_conn.close()
                src_conn.close()
                db.delete(it_hist_bkp_dup)
                db.commit()
                recovered_cnt = svc.recover_deleted_duplicate_history_from_backup(snap_bkp_path)
                self.assertGreaterEqual(recovered_cnt, 1)
                r_rec_q = self.client.get("/media/search?q=rec_from_backup_hist_dup.jpg")
                self.assertEqual(r_rec_q.status_code, 200)
                self.assertIn("rec_orig.jpg", r_rec_q.text)

                # Reconcile sidebar counts with Backup panel stats + redundant_clusters_only / backups_to_move_only filters
                svc.flag_duplicate_clusters()
                plan_rec = svc.build_backup_and_dedup_plan()
                p_stats = plan_rec["stats"]
                r_sb_reconcile = self.client.get("/media/sidebar")
                self.assertEqual(r_sb_reconcile.status_code, 200)
                self.assertIn("Backupy do przeniesienia", r_sb_reconcile.text)
                self.assertIn(f">{p_stats['total_unique']}<", r_sb_reconcile.text)
                self.assertIn(f">{p_stats['missing_backups']}<", r_sb_reconcile.text)
                self.assertIn(f">{p_stats['orange_backups']}<", r_sb_reconcile.text)
                # Ensure green copy outside daf_backup (p3_b) counts under backups_to_move_only, not as lacking cross-device backup
                r_unb = self.client.get("/media/search?unbacked_only=true")
                self.assertEqual(r_unb.status_code, 200)
                self.assertNotIn("outside_bkp.jpg", r_unb.text)
                r_bkp_mv = self.client.get("/media/search?duplicates_only=true&backups_to_move_only=true")
                self.assertEqual(r_bkp_mv.status_code, 200)
                self.assertIn("outside_bkp.jpg", r_bkp_mv.text)
                self.assertIn("Otwórz Panel Backupu zdjęć/filmów", r_bkp_mv.text)

                # redundant_clusters_only=true shows ONLY clusters with duplicate_role=='duplicate' (excludes 2-copy ORYGINAL+BACKUP clusters)
                r_red_only = self.client.get("/media/search?duplicates_only=true&redundant_clusters_only=true")
                self.assertEqual(r_red_only.status_code, 200)
                self.assertNotIn("outside_bkp.jpg", r_red_only.text)
                self.assertIn(
                    f"Nadmiarowych kopii do usunięcia: <span class=\"text-rose-300 font-mono\">{p_stats['redundant_duplicates']}</span>",
                    r_red_only.text,
                )

                # Offline-ref video with partial_3x64k is eligible for metadata SHA-256 verification instead of blocking as unverified
                p_v3_on = self.drive_a / "vid_p3_dup.mp4"
                p_v3_payload = b"ftypisom_partial3x64k_" + (b"\x11\x22\x33\x44" * 1024)
                p_v3_on.write_bytes(p_v3_payload)
                p_v3_hash = FastHasher.hash_video_partial(p_v3_on, len(p_v3_payload))
                v3_specs = [
                    (vol_d.id, "v3_orig.mp4", "original", False),
                    (vol_d.id, "daf_backup/2024/08/01/v3_orig.mp4", "backup_same_dev", True),
                    (vol_a.id, "vid_p3_dup.mp4", "duplicate", False),
                ]
                v3_items = [
                    MediaItem(
                        id=str(_uuid.uuid4()),
                        volume_id=vid_i,
                        relative_path=rp,
                        file_name=Path(rp).name,
                        file_size_bytes=len(p_v3_payload),
                        file_hash=p_v3_hash,
                        hash_algorithm="partial_3x64k",
                        verified_sha256=None,
                        media_type="video",
                        mime_type="video/mp4",
                        fs_ctime=now_vid_dt,
                        fs_mtime=now_vid_dt,
                        effective_date=now_vid_dt,
                        thumbnail_relative_path="dummy_v3_thumb.webp",
                        duplicate_role=dr,
                        is_backup_copy=ibkp,
                    )
                    for vid_i, rp, dr, ibkp in v3_specs
                ]
                db.add_all(v3_items)
                db.commit()
                res_v3 = svc.execute_safe_duplicate_removal(
                    [v3_items[2].id], delete_mode="permanent", force_duplicate_verification=True
                )
                # V-07 protection: offline video duplicate with partial_3x64k and verified_sha256=None MUST be blocked!
                self.assertEqual(res_v3["physical_count"], 0)
                self.assertEqual(res_v3["blocked_verify"], 1)
                self.assertTrue(p_v3_on.exists())

                # Verify once all 3rd+ duplicates are removed, Duplikaty binarne shows 0 while 2-copy clusters remain under Backupy do przeniesienia
                for rem_dup in db.scalars(select(MediaItem).where(MediaItem.is_trashed == False, MediaItem.duplicate_role == "duplicate")).all():
                    db.delete(rem_dup)
                db.commit()
                svc.flag_duplicate_clusters()
                r_red_zero = self.client.get("/media/search?duplicates_only=true&redundant_clusters_only=true")
                self.assertEqual(r_red_zero.status_code, 200)
                self.assertIn("Brak plików spełniających kryteria.", r_red_zero.text)
                self.assertIn("Nadmiarowych kopii do usunięcia: <span class=\"text-rose-300 font-mono\">0</span>", r_red_zero.text)

    def test_maintenance_vacuum_and_orphan_thumbnails(self):
        with self.TestingSessionLocal() as db:
            vol = WindowsVolume(
                volume_serial_number="VSN-MNT-1",
                volume_label="MaintDisk",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            db.add(vol)
            db.commit()
            db.refresh(vol)

            p_keep = create_synthetic_jpeg(self.drive_a / "keep.jpg")
            p_trash = create_synthetic_jpeg(self.drive_a / "trashed.jpg")
            it_keep = FileScannerService.process_file(p_keep, vol)
            it_trash = FileScannerService.process_file(p_trash, vol)
            it_trash.is_trashed = True
            db.add_all([it_keep, it_trash])
            db.commit()

            # Create an orphan thumbnail in self.thumbs_dir that does NOT belong to any DB item
            orphan_thumb = self.thumbs_dir / "orphan_leftover_123.webp"
            orphan_thumb.write_bytes(b"RIFF1234WEBPorphan_data_padding" * 10)
            valid_thumb = self.thumbs_dir / it_keep.thumbnail_relative_path
            trashed_thumb = self.thumbs_dir / it_trash.thumbnail_relative_path
            self.assertTrue(orphan_thumb.exists())
            self.assertTrue(valid_thumb.exists())
            self.assertTrue(trashed_thumb.exists())

            # Status panel shows both maintenance tools
            r_sp = self.client.get("/actions/status-panel")
            self.assertEqual(r_sp.status_code, 200)
            self.assertIn("Kompaktuj bazę (VACUUM)", r_sp.text)
            self.assertIn("Usuń osierocone miniatury", r_sp.text)

            # Cleanup orphan thumbnails deletes ONLY orphan_thumb while keeping active and trashed thumbnails
            r_clean = self.client.post("/actions/cleanup-orphan-thumbs")
            self.assertEqual(r_clean.status_code, 200)
            self.assertIn("Wyczyszczono osierocone miniatury", r_clean.text)
            self.assertFalse(orphan_thumb.exists())
            self.assertTrue(valid_thumb.exists())
            self.assertTrue(trashed_thumb.exists())

            # Manual VACUUM compacts SQLite cleanly and logs DB_VACUUM
            r_vac = self.client.post("/actions/vacuum-db")
            self.assertEqual(r_vac.status_code, 200)
            self.assertIn("Zakończono kompaktowanie bazy (VACUUM)", r_vac.text)

    def test_vsn_guard_consistency_organize_physical_albums_filters_and_offline_video(self):
        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="AA11-2233",
                volume_label="OrigDiskA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=60 * 1024**3,
            )
            vol_b = WindowsVolume(
                volume_serial_number="BB44-5566",
                volume_label="BackupDiskB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=200 * 1024**3,
                free_space_bytes=150 * 1024**3,
            )
            db.add_all([vol_a, vol_b])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            vol_a_id, vol_b_id = vol_a.id, vol_b.id

            svc = StorageAutomationService(db)

            # 1) Item 2: Live VSN mismatch guard & drive-letter reassignment
            # Different connected VSN occupies E:\ -> must block (return None)
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"9999-8888": ("E:\\", "WrongUSB")}):
                vol_e_swap = WindowsVolume(
                    volume_serial_number="1234-ABCD",
                    volume_label="ExpectedUSB",
                    last_known_drive_letter="E:\\",
                    is_online=True,
                )
                self.assertIsNone(svc.verify_and_resolve_online_volume_root(vol_e_swap))
                self.assertFalse(vol_e_swap.is_online)

            # Matching VSN mounted under a new drive letter G:\ -> resolves G:\ and updates vol_e_swap
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"1234-ABCD": ("G:\\", "ExpectedUSB")}):
                resolved = svc.verify_and_resolve_online_volume_root(vol_e_swap)
                self.assertEqual(resolved, Path("G:\\"))
                self.assertEqual(vol_e_swap.last_known_drive_letter, "G:\\")
                self.assertTrue(vol_e_swap.is_online)

            # 2) Item 1 & Item 6: Non-destructive force_rescan preserves AI tags, quality, albums; daf_albums/ skipped;
            #    verify_db_disk_consistency detects missing physical files & cleanup_missing_db_items removes them cleanly
            trip_dir = self.drive_a / "Wakacje_2022"
            trip_dir.mkdir(parents=True, exist_ok=True)
            p_keep = create_synthetic_jpeg(trip_dir / "IMG_20220710_120000.jpg", color=(30, 120, 210))
            p_missing_later = create_synthetic_jpeg(trip_dir / "IMG_20220710_120500.jpg", color=(210, 60, 40))

            grp = MediaGroup(name="Grecja_Test")
            db.add(grp)
            db.commit()
            db.refresh(grp)
            grp_id = grp.id

            it_keep = FileScannerService.process_file(p_keep, vol_a)
            it_missing = FileScannerService.process_file(p_missing_later, vol_a)
            it_keep_id, it_missing_id = it_keep.id, it_missing.id
            it_keep_hash = it_keep.file_hash
            it_keep.group_id = grp_id
            it_keep.blur_score = 123.4
            it_keep.rotation_angle = 90
            it_keep.tags.append(MediaTag(tag="plaża", confidence=0.95, source_model="test_ai"))
            it_keep.has_date_anomaly = True
            it_keep.date_anomaly_reason = "Niezgodność dat: utworzenie pliku (2024-01-01) zmodyfikowane przy kopiowaniu vs modyfikacja (2022-07-10)"
            db.add_all([it_keep, it_missing])
            db.commit()

            # Force rescan via execute_scan must NOT wipe AI tags, blur_score, rotation_angle, or album
            with patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("AA11-2233", "OrigDiskA", str(self.drive_a))):
                # Delete one file before scanning to verify missing_on_disk detection
                p_missing_later.unlink()
                r_rescan = self.client.post("/actions/scan", data={"path": str(self.drive_a), "force_rescan": "true"})
                self.assertEqual(r_rescan.status_code, 200)
                self.assertIn("Brakujących plików na dysku", r_rescan.text)

            db.expire_all()
            it_keep_after = db.get(MediaItem, it_keep_id)
            self.assertEqual(it_keep_after.group_id, grp_id)
            self.assertEqual(it_keep_after.rotation_angle, 90)
            self.assertEqual(it_keep_after.blur_score, 123.4)
            self.assertTrue(any(t.tag == "plaża" for t in it_keep_after.tags))

            # Explicit DB-disk consistency report & cleanup
            with patch.object(
                WindowsVolumeInspector,
                "get_connected_vsn_map",
                return_value={"AA11-2233": (str(self.drive_a), "OrigDiskA"), "BB44-5566": (str(self.drive_b), "BackupDiskB")},
            ):
                r_cons = self.client.post("/actions/verify-db-disk-consistency")
                self.assertEqual(r_cons.status_code, 200)
                self.assertIn("IMG_20220710_120500.jpg", r_cons.text)
                self.assertIn("Decyzja: Usuń z bazy 1 brakujących rekordów", r_cons.text)

                r_cln = self.client.post("/actions/cleanup-missing-db-items")
                self.assertEqual(r_cln.status_code, 200)
                self.assertIn("Usunięto z bazy <strong>1</strong>", r_cln.text)
                db.expire_all()
                self.assertIsNone(db.get(MediaItem, it_missing_id))

            # 3) Item 6: Physical album export -> daf_albums/<AlbumName>/ & skipped by scanner
            with patch.object(
                WindowsVolumeInspector,
                "get_connected_vsn_map",
                return_value={"AA11-2233": (str(self.drive_a), "OrigDiskA"), "BB44-5566": (str(self.drive_b), "BackupDiskB")},
            ), patch.object(WindowsVolumeInspector, "get_volume_info", return_value=("BB44-5566", "BackupDiskB", str(self.drive_b))):
                r_exp = self.client.post(
                    "/actions/export-physical-album",
                    data={"album_id": str(grp_id), "target_volume_id": str(vol_b_id)},
                )
                self.assertEqual(r_exp.status_code, 200)
                exported_f = self.drive_b / "daf_albums" / "Grecja_Test" / "IMG_20220710_120000.jpg"
                self.assertTrue(exported_f.exists())
                # FileScannerService & /actions/scan must completely ignore daf_albums/
                self.assertIsNone(FileScannerService.process_file(exported_f, db.get(WindowsVolume, vol_b_id)))
                r_scan_b = self.client.post("/actions/scan", data={"path": str(self.drive_b), "force_rescan": "false"})
                self.assertEqual(r_scan_b.status_code, 200)
                self.assertEqual(
                    db.scalar(select(func.count(MediaItem.id)).where(MediaItem.relative_path.ilike("%daf_albums%"))),
                    0,
                )

            # 4) Item 5: Organize ungrouped Originals -> daf_oryginal/YYYY/MM/DD/ preserving original_relative_path
            ungrouped_dir = self.drive_a / "StaryKatalog_2021"
            ungrouped_dir.mkdir(parents=True, exist_ok=True)
            p_ungrp = create_synthetic_jpeg(ungrouped_dir / "IMG_20210815_100000.jpg", color=(80, 180, 90))
            it_ungrp = FileScannerService.process_file(p_ungrp, db.get(WindowsVolume, vol_a_id))
            it_ungrp_id = it_ungrp.id
            it_ungrp.has_date_anomaly = True
            it_ungrp.date_anomaly_reason = "Niezgodność dat: utworzenie pliku (2023-01-01) z folderem '2020_Wyjazd'"
            db.add(it_ungrp)
            db.commit()

            with patch.object(
                WindowsVolumeInspector,
                "get_connected_vsn_map",
                return_value={"AA11-2233": (str(self.drive_a), "OrigDiskA"), "BB44-5566": (str(self.drive_b), "BackupDiskB")},
            ):
                # H-07: Without volume_id, organize-originals must prompt user to select a specific volume first
                r_org_novol = self.client.post("/actions/organize-originals")
                self.assertEqual(r_org_novol.status_code, 200)
                self.assertIn("Wybierz najpierw konkretny dysk", r_org_novol.text)

                r_org = self.client.post("/actions/organize-originals", data={"volume_id": str(vol_a_id)})
                self.assertEqual(r_org.status_code, 200)
                self.assertIn("Uporządkowano oryginałów", r_org.text)

            db.expire_all()
            it_ungrp_after = db.get(MediaItem, it_ungrp_id)
            self.assertTrue(it_ungrp_after.relative_path.startswith("daf_oryginal/"))
            self.assertEqual(it_ungrp_after.original_relative_path, "OrigDiskA/StaryKatalog_2021/IMG_20210815_100000.jpg")
            self.assertTrue((self.drive_a / it_ungrp_after.relative_path).exists())
            # Must still be findable by original folder name & prefix!
            r_find_orig_folder = self.client.get("/media/search?q=StaryKatalog_2021")
            self.assertIn("IMG_20210815_100000.jpg", r_find_orig_folder.text)
            r_find_folder_pref = self.client.get(f"/media/search?volume_id={vol_a_id}&folder_prefix=StaryKatalog_2021/")
            self.assertIn("IMG_20210815_100000.jpg", r_find_folder_pref.text)

            # 5) Item 4: Date anomaly type filter (folder vs copy_ctime)
            it_keep_anom = db.get(MediaItem, it_keep_id)
            it_keep_anom.has_date_anomaly = True
            it_keep_anom.date_anomaly_reason = "Niezgodność dat: utworzenie pliku (2024-01-01) zmodyfikowane przy kopiowaniu vs modyfikacja (2022-07-10)"
            db.commit()

            r_anom_all = self.client.get("/media/search?anomaly_only=true")
            self.assertEqual(r_anom_all.status_code, 200)
            self.assertIn("IMG_20210815_100000.jpg", r_anom_all.text)
            self.assertIn("IMG_20220710_120000.jpg", r_anom_all.text)
            r_anom_folder = self.client.get("/media/search?anomaly_only=true&anomaly_type=folder")
            self.assertIn("IMG_20210815_100000.jpg", r_anom_folder.text)
            self.assertNotIn("IMG_20220710_120000.jpg", r_anom_folder.text)
            r_anom_copy = self.client.get("/media/search?anomaly_only=true&anomaly_type=copy_ctime")
            self.assertIn("IMG_20220710_120000.jpg", r_anom_copy.text)
            self.assertNotIn("IMG_20210815_100000.jpg", r_anom_copy.text)

            # 6) Item 3 & Item 10: Backup visibility filter (default hides all backups, including inside albums!)
            #    and Move Backups checkboxes (in_vault, green_outside, orange)
            bkp_vault_f = self.drive_b / "daf_backup" / "2022" / "07" / "10" / "IMG_20220710_120000.jpg"
            bkp_vault_f.parent.mkdir(parents=True, exist_ok=True)
            bkp_vault_f.write_bytes(p_keep.read_bytes())
            it_bkp_vault = FileScannerService.process_file(bkp_vault_f, db.get(WindowsVolume, vol_b_id))
            # Also an orange backup on DiskA for it_ungrp
            orange_f = self.drive_a / "orange_copy.jpg"
            orange_f.write_bytes((self.drive_a / it_ungrp_after.relative_path).read_bytes())
            it_orange = FileScannerService.process_file(orange_f, db.get(WindowsVolume, vol_a_id))
            it_orange_day = it_orange.effective_date.strftime("%Y/%m/%d")
            db.add_all([it_bkp_vault, it_orange])
            db.commit()
            svc.flag_duplicate_clusters()

            # Default album & gallery hide backups (neither daf_backup nor orange_copy visible)
            r_def_gal = self.client.get("/media/search")
            self.assertNotIn("orange_copy.jpg", r_def_gal.text)
            self.assertNotIn("daf_backup/2022/07/10/IMG_20220710_120000.jpg", r_def_gal.text)
            r_def_alb = self.client.get(f"/media/search?group_id={grp_id}")
            self.assertIn("IMG_20220710_120000.jpg", r_def_alb.text)
            self.assertNotIn("daf_backup/2022/07/10/IMG_20220710_120000.jpg", r_def_alb.text)
            # Enabling backup_filter=only_daf_backup in album shows daf_backup copy linked to that album!
            r_alb_bkp = self.client.get(f"/media/search?group_id={grp_id}&backup_filter=only_daf_backup")
            self.assertIn("daf_backup/2022/07/10/IMG_20220710_120000.jpg", r_alb_bkp.text)
            # Enabling backup_filter=only_orange in gallery shows orange backup
            r_gal_or = self.client.get("/media/search?backup_filter=only_orange")
            self.assertIn("orange_copy.jpg", r_gal_or.text)

            # Item 10: Move Backups modal shows per-type breakdown & moves orange backup on DiskA -> DiskB/daf_backup/YYYY/MM/DD
            with patch.object(
                WindowsVolumeInspector,
                "get_connected_vsn_map",
                return_value={"AA11-2233": (str(self.drive_a), "OrigDiskA"), "BB44-5566": (str(self.drive_b), "BackupDiskB")},
            ):
                r_mbp = self.client.post(
                    "/actions/move-backups-panel",
                    data={
                        "source_volume_id": str(vol_a_id),
                        "target_volume_id": str(vol_b_id),
                        "types_submitted": "1",
                        "include_orange_same_disk": "true",
                    },
                )
                self.assertEqual(r_mbp.status_code, 200)
                self.assertIn("orange_copy.jpg", r_mbp.text)
                self.assertIn(f"daf_backup/{it_orange_day}/orange_copy.jpg", r_mbp.text)

                r_exec_mb = self.client.post(
                    "/actions/execute-move-backups",
                    data={
                        "source_volume_id": str(vol_a_id),
                        "target_volume_id": str(vol_b_id),
                        "types_submitted": "1",
                        "include_orange_same_disk": "true",
                    },
                )
                self.assertEqual(r_exec_mb.status_code, 200)
                self.assertIn("Przeniesiono backupów z OrigDiskA -&gt; BackupDiskB", r_exec_mb.text)
                self.assertTrue((self.drive_b / "daf_backup" / it_orange_day / "orange_copy.jpg").exists())

            # 7) Item 8: Verification query after duplicate removal
            hist_rec = DeletedDuplicateHistory(
                deleted_item_id="old-dup-uuid-1",
                original_item_id=it_keep_id,
                file_hash=it_keep_hash,
                volume_id=vol_a_id,
                volume_label="OrigDiskA",
                drive_letter=str(self.drive_a),
                relative_path="old_dup/IMG_20220710_120000_copy.jpg",
                file_name="IMG_20220710_120000_copy.jpg",
            )
            db.add(hist_rec)
            db.commit()
            r_ver_dups = self.client.post("/actions/verify-deleted-duplicates")
            self.assertEqual(r_ver_dups.status_code, 200)
            self.assertIn("Liczba usuniętych duplikatów: 1", r_ver_dups.text)
            self.assertIn("1 z 1 (100.0%)", r_ver_dups.text)

            # 8) Item 9: Offline video detail preview renders cached .webp thumbnail instead of video playback error
            off_vid_id = str(uuid.uuid4())
            off_vid = MediaItem(
                id=off_vid_id,
                volume_id=vol_a_id,
                relative_path="DJI_20260829155141_0016_D.MP4",
                file_name="DJI_20260829155141_0016_D.MP4",
                file_size_bytes=5000,
                media_type="video",
                mime_type="video/mp4",
                file_hash="dji_vid_hash_016",
                hash_algorithm="partial_3x64k",
                fs_ctime=datetime.datetime(2026, 8, 29, 15, 51, 41),
                fs_mtime=datetime.datetime(2026, 8, 29, 15, 51, 41),
                effective_date=datetime.datetime(2026, 8, 29, 15, 51, 41),
                thumbnail_relative_path="dji_016.webp",
            )
            (self.thumbs_dir / "dji_016.webp").write_bytes(b"RIFF1234WEBPdummy_thumb_bytes")
            db.add(off_vid)
            vol_a_db = db.get(WindowsVolume, vol_a_id)
            vol_a_db.is_online = False
            db.commit()

            r_vid_det = self.client.get(f"/media/{off_vid_id}/detail")
            self.assertEqual(r_vid_det.status_code, 200)
            self.assertIn("/media/thumbnails/dji_016.webp", r_vid_det.text)
            self.assertIn("Nośnik offline", r_vid_det.text)
            self.assertNotIn("<video src=", r_vid_det.text)

            # 9) Item 7: Logical-only Kosz on regular photo keeps file physically on disk
            p_keep_path = self.drive_a / db.get(MediaItem, it_keep_id).relative_path
            self.assertTrue(p_keep_path.exists())
            r_tr_reg = self.client.post(
                "/actions/move-to-trash",
                data={"scope": "selected", "selected_ids": [it_keep_id]},
            )
            self.assertEqual(r_tr_reg.status_code, 200)
            db.expire_all()
            self.assertTrue(db.get(MediaItem, it_keep_id).is_trashed)
            self.assertTrue(p_keep_path.exists())

            # 10) Item 0: Mid-batch partial duplicate removal across online/offline drives must NOT
            #     call flag_duplicate_clusters() while offline drives remain pending
            vol_a_db.is_online = True
            vol_b_db = db.get(WindowsVolume, vol_b_id)
            vol_b_db.is_online = False
            vol_b_db.last_known_drive_letter = str(self.root / "OfflineMissingB_Item0")
            db.commit()
            with patch.object(
                WindowsVolumeInspector,
                "get_connected_vsn_map",
                return_value={"AA11-2233": (str(self.drive_a), "OrigDiskA")},
            ), patch.object(StorageAutomationService, "flag_duplicate_clusters") as spy_flag:
                p_i0 = create_synthetic_jpeg(self.drive_a / "i0_orig.jpg", color=(15, 25, 35))
                p_i0_bkp = self.drive_a / "daf_backup" / "2024" / "01" / "01" / "i0_orig.jpg"
                p_i0_bkp.parent.mkdir(parents=True, exist_ok=True)
                p_i0_bkp.write_bytes(p_i0.read_bytes())
                p_i0_on_dup = self.drive_a / "i0_on_dup.jpg"
                p_i0_on_dup.write_bytes(p_i0.read_bytes())
                it_i0_1 = FileScannerService.process_file(p_i0, db.get(WindowsVolume, vol_a_id))
                it_i0_2 = FileScannerService.process_file(p_i0_bkp, db.get(WindowsVolume, vol_a_id))
                it_i0_3 = FileScannerService.process_file(p_i0_on_dup, db.get(WindowsVolume, vol_a_id))
                it_i0_4 = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=vol_b_id,
                    relative_path="i0_off_dup.jpg",
                    file_name="i0_off_dup.jpg",
                    file_size_bytes=it_i0_1.file_size_bytes,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash=it_i0_1.file_hash,
                    hash_algorithm="sha256",
                    fs_ctime=datetime.datetime(2024, 1, 1),
                    fs_mtime=datetime.datetime(2024, 1, 1),
                    effective_date=datetime.datetime(2024, 1, 1),
                    thumbnail_relative_path="i0_off.webp",
                    duplicate_role="duplicate",
                    is_primary_duplicate=False,
                )
                it_i0_1.duplicate_role = "original"
                it_i0_2.duplicate_role = "backup_orange"
                it_i0_3.duplicate_role = "duplicate"
                it_i0_3.is_primary_duplicate = False
                db.add_all([it_i0_1, it_i0_2, it_i0_3, it_i0_4])
                db.commit()
                res_i0 = svc.execute_safe_duplicate_removal(
                    [it_i0_3.id, it_i0_4.id],
                    delete_mode="permanent",
                    force_duplicate_verification=True,
                )
                self.assertEqual(res_i0["processed_count"], 1)
                self.assertEqual(len(res_i0["offline_items"]), 1)
                self.assertEqual(res_i0["offline_ids"], [it_i0_4.id])
                spy_flag.assert_not_called()

    def test_quality_defect_clear_offline_drive_queues_sidebar_reset_and_semantic_backup_filter(self):
        with self.TestingSessionLocal() as db:
            vol_on = WindowsVolume(
                volume_serial_number="ON11-0001",
                volume_label="OnlineMain",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=50 * 1024**3,
            )
            vol_off1 = WindowsVolume(
                volume_serial_number="OF11-0002",
                volume_label="OffDriveAlpha",
                last_known_drive_letter=str(self.root / "OfflineMissingAlpha"),
                is_online=False,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=10 * 1024**3,
            )
            vol_off2 = WindowsVolume(
                volume_serial_number="OF22-0003",
                volume_label="OffDriveBeta",
                last_known_drive_letter=str(self.root / "OfflineMissingBeta"),
                is_online=False,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=10 * 1024**3,
            )
            db.add_all([vol_on, vol_off1, vol_off2])
            db.commit()
            db.refresh(vol_on)
            db.refresh(vol_off1)
            db.refresh(vol_off2)

            vsn_online_only = {"ON11-0001": (str(self.drive_a), "OnlineMain")}

            # 1) Item 1: Clear quality defects (single, filtered defect sub-view, selected batch)
            p_corr = create_synthetic_jpeg(self.drive_a / "corr_fix.jpg", color=(80, 90, 100))
            p_blur = create_synthetic_jpeg(self.drive_a / "blur_fix.jpg", color=(100, 110, 120))
            p_dark = create_synthetic_jpeg(self.drive_a / "dark_fix.jpg", color=(20, 25, 30))
            it_corr = FileScannerService.process_file(p_corr, vol_on)
            it_corr.is_corrupted = True
            it_corr.blur_score = None
            it_corr.duplicate_role = "original"
            it_blur = FileScannerService.process_file(p_blur, vol_on)
            it_blur.is_blurry = True
            it_blur.blur_score = 18.5
            it_blur.duplicate_role = "original"
            it_dark = FileScannerService.process_file(p_dark, vol_on)
            it_dark.is_blurry = False
            it_dark.blur_score = 78.0
            it_dark.duplicate_role = "original"
            db.add_all([it_corr, it_blur, it_dark])
            db.commit()

            db.add_all([
                MediaTag(media_id=it_corr.id, tag="do_usuniecia", confidence=1.0, source_model="quality_check"),
                MediaTag(media_id=it_corr.id, tag="do_usuniecia:uszkodzone", confidence=1.0, source_model="quality_check"),
                MediaTag(media_id=it_blur.id, tag="do_usuniecia", confidence=1.0, source_model="quality_check"),
                MediaTag(media_id=it_blur.id, tag="do_usuniecia:rozmyte", confidence=1.0, source_model="quality_check"),
                MediaTag(media_id=it_dark.id, tag="do_usuniecia", confidence=1.0, source_model="quality_check"),
                MediaTag(media_id=it_dark.id, tag="do_usuniecia:niedoswietlone", confidence=1.0, source_model="quality_check"),
                MediaTag(media_id=it_dark.id, tag="krajobraz", confidence=0.95, source_model="clip"),
            ])
            db.commit()

            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_online_only):
                r_detail = self.client.get(f"/media/{it_corr.id}/detail")
                self.assertEqual(r_detail.status_code, 200)
                self.assertIn("Oznacz jako bez wady / nieuszkodzone", r_detail.text)

                r_grid_def = self.client.get("/media/search?quality_defect=uszkodzone")
                self.assertEqual(r_grid_def.status_code, 200)
                self.assertIn("corr_fix.jpg", r_grid_def.text)
                self.assertIn("/actions/clear-quality-defect", r_grid_def.text)
                self.assertIn("✓ Bez wady", r_grid_def.text)

                r_tbl_def = self.client.get("/media/search?view_mode=table&quality_defect=uszkodzone")
                self.assertEqual(r_tbl_def.status_code, 200)
                self.assertIn("✓ Bez wady", r_tbl_def.text)

                # Clear single item by media_id
                r_cls1 = self.client.post("/actions/clear-quality-defect", data={"media_id": it_corr.id})
                self.assertEqual(r_cls1.status_code, 200)
                self.assertEqual(r_cls1.headers.get("HX-Trigger"), "refresh-all")
                db.expire_all()
                it_corr_db = db.get(MediaItem, it_corr.id)
                self.assertFalse(it_corr_db.is_corrupted)
                self.assertFalse(it_corr_db.is_blurry)
                self.assertEqual(it_corr_db.blur_score, 100.0)
                self.assertEqual(
                    list(db.scalars(select(MediaTag.tag).where(MediaTag.media_id == it_corr.id)).all()),
                    [],
                )

                # Must NOT show up as unanalyzed quality after being cleared
                r_unan = self.client.get("/media/search?unanalyzed_quality_only=true")
                self.assertEqual(r_unan.status_code, 200)
                self.assertNotIn("corr_fix.jpg", r_unan.text)

                # Clear filtered view (only niedoswietlone), preserving blur_score >= 45 and non-defect tags
                r_cls_dark = self.client.post(
                    "/actions/clear-quality-defect",
                    data={"scope": "filtered", "quality_defect": "niedoswietlone"},
                )
                self.assertEqual(r_cls_dark.status_code, 200)
                db.expire_all()
                it_dark_db = db.get(MediaItem, it_dark.id)
                self.assertEqual(it_dark_db.blur_score, 78.0)
                self.assertEqual(
                    list(db.scalars(select(MediaTag.tag).where(MediaTag.media_id == it_dark.id)).all()),
                    ["krajobraz"],
                )
                # Blurry item was NOT in niedoswietlone and remains intact
                self.assertTrue(db.get(MediaItem, it_blur.id).is_blurry)

                # Clear selected batch
                r_cls_blur = self.client.post(
                    "/actions/clear-quality-defect",
                    data={"scope": "selected", "selected_ids": [it_blur.id]},
                )
                self.assertEqual(r_cls_blur.status_code, 200)
                db.expire_all()
                it_blur_db = db.get(MediaItem, it_blur.id)
                self.assertFalse(it_blur_db.is_blurry)
                self.assertEqual(it_blur_db.blur_score, 100.0)
                self.assertEqual(
                    list(db.scalars(select(MediaTag.tag).where(MediaTag.media_id == it_blur.id)).all()),
                    [],
                )

            # 2) Item 2: Ordered offline drive queues (page 1 vs entire multi-page view)
            # Clear prior test items so page 1 / page 2 counts are exact
            db.execute(delete(MediaTag))
            db.execute(delete(MediaItem))
            db.commit()

            base_dt = datetime.datetime(2025, 5, 1, 12, 0, 0)
            queue_items = []
            # Page 1 (top 10 newest): 6 on OffDriveAlpha, 2 on OffDriveBeta, 2 on OnlineMain
            specs_p1 = (
                [vol_off1.id] * 6
                + [vol_off2.id] * 2
                + [vol_on.id] * 2
            )
            # Page 2 (next 5 older): 5 on OffDriveBeta -> full view totals Beta=7, Alpha=6
            specs_p2 = [vol_off2.id] * 5
            for idx, vid_q in enumerate(specs_p1 + specs_p2):
                dt_i = base_dt - datetime.timedelta(minutes=idx)
                fname_i = f"queue_{idx:02d}.jpg"
                queue_items.append(
                    MediaItem(
                        id=str(uuid.uuid4()),
                        volume_id=vid_q,
                        relative_path=fname_i,
                        file_name=fname_i,
                        file_size_bytes=1024,
                        media_type="image",
                        mime_type="image/jpeg",
                        file_hash=f"qh_{idx:04d}" + ("0" * 57),
                        hash_algorithm="sha256",
                        fs_ctime=dt_i,
                        fs_mtime=dt_i,
                        effective_date=dt_i,
                        thumbnail_relative_path=f"q_{idx}.webp",
                        duplicate_role="original",
                        is_primary_duplicate=True,
                        is_backup_copy=False,
                        blur_score=95.0,
                    )
                )
            db.add_all(queue_items)
            db.commit()

            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_online_only):
                r_q_page1 = self.client.get("/media/search?page=1&page_size=10")
                self.assertEqual(r_q_page1.status_code, 200)
                txt_p1 = r_q_page1.text
                self.assertIn("Dyski offline wymagane do podglądu oryginałów w kolejności", txt_p1)
                self.assertIn("Pierwsza zakładka / bieżąca strona", txt_p1)
                self.assertIn("Cały aktywny widok / filtr", txt_p1)
                p1_sec, view_sec = txt_p1.split("Cały aktywny widok / filtr", 1)
                # On page 1: OffDriveAlpha (6) must precede OffDriveBeta (2)
                self.assertIn("6 szt.", p1_sec)
                self.assertIn("2 szt.", p1_sec)
                self.assertLess(p1_sec.index("OffDriveAlpha"), p1_sec.index("OffDriveBeta"))
                # On full view: OffDriveBeta (7) must precede OffDriveAlpha (6)
                self.assertIn("7 szt.", view_sec)
                self.assertIn("6 szt.", view_sec)
                self.assertLess(view_sec.index("OffDriveBeta"), view_sec.index("OffDriveAlpha"))

            # 3) Item 3 & 4: Unanalyzed quality visibility + sidebar backup exclusion + semantic search hiding backups
            p_dji_orig = create_synthetic_jpeg(self.drive_a / "DJI_20250101_001.jpg", color=(40, 80, 120))
            p_dji_vault = self.drive_a / "daf_backup" / "2025" / "01" / "01" / "DJI_20250101_vault.jpg"
            p_dji_vault.parent.mkdir(parents=True, exist_ok=True)
            p_dji_vault.write_bytes(p_dji_orig.read_bytes())
            p_dji_green = create_synthetic_jpeg(self.drive_a / "DJI_20250101_green.jpg", color=(45, 85, 125))
            p_dji_orange = create_synthetic_jpeg(self.drive_a / "DJI_20250101_orange.jpg", color=(50, 90, 130))

            it_dji_orig = FileScannerService.process_file(p_dji_orig, vol_on)
            it_dji_orig.duplicate_role = "original"
            it_dji_orig.blur_score = 88.0

            it_dji_vault = FileScannerService.process_file(p_dji_vault, vol_on)
            it_dji_vault.duplicate_role = "backup_green"
            it_dji_vault.is_backup_copy = True
            it_dji_vault.blur_score = None
            it_dji_vault.has_date_anomaly = True

            it_dji_green = FileScannerService.process_file(p_dji_green, vol_on)
            it_dji_green.duplicate_role = "backup_green"
            it_dji_green.is_backup_copy = False
            it_dji_green.blur_score = None
            it_dji_green.has_date_anomaly = True

            it_dji_orange = FileScannerService.process_file(p_dji_orange, vol_on)
            it_dji_orange.duplicate_role = "backup_orange"
            it_dji_orange.is_backup_copy = False
            it_dji_orange.blur_score = None
            it_dji_orange.has_date_anomaly = True

            it_unan_recycle = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_on.id,
                relative_path="$RECYCLE.BIN/S-1-5-21/$RZUR758.jpg",
                file_name="$RZUR758.jpg",
                file_size_bytes=512,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="unan_" + ("a" * 59),
                hash_algorithm="sha256",
                fs_ctime=base_dt,
                fs_mtime=base_dt,
                effective_date=base_dt,
                thumbnail_relative_path="rzur.webp",
                duplicate_role="original",
                is_primary_duplicate=True,
                is_backup_copy=False,
                blur_score=None,
                is_corrupted=False,
            )
            db.add_all([it_dji_orig, it_dji_vault, it_dji_green, it_dji_orange, it_unan_recycle])
            db.commit()

            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_online_only):
                # Sidebar count for Wszystkie multimedia = 15 queue_items + DJI_001 + $RZUR758 = 17 (NOT 19 with outside backups!)
                r_sb_cons = self.client.get("/media/sidebar")
                self.assertEqual(r_sb_cons.status_code, 200)
                self.assertIn(">17<", r_sb_cons.text)
                self.assertIn(">1<", r_sb_cons.text)  # unanalyzed_quality count == 1 ($RZUR758.jpg), excluding backup_green/orange

                r_unan_only = self.client.get("/media/search?unanalyzed_quality_only=true&q=")
                self.assertEqual(r_unan_only.status_code, 200)
                self.assertIn("$RZUR758.jpg", r_unan_only.text)
                self.assertNotIn("DJI_20250101_green.jpg", r_unan_only.text)

                # Semantic search MUST hide daf_backup, backup_green, and backup_orange by default
                r_sem_hide = self.client.get("/semantic/search?q=DJI_20250101")
                self.assertEqual(r_sem_hide.status_code, 200)
                self.assertIn("DJI_20250101_001.jpg", r_sem_hide.text)
                self.assertIn("DJI_20250101", r_sem_hide.text)
                self.assertNotIn("DJI_20250101_vault.jpg", r_sem_hide.text)
                self.assertNotIn("DJI_20250101_green.jpg", r_sem_hide.text)
                self.assertNotIn("DJI_20250101_orange.jpg", r_sem_hide.text)

                # When backup_filter=all_with_backups is enabled, semantic search displays backups too
                r_sem_all = self.client.get("/semantic/search?q=DJI_20250101&backup_filter=all_with_backups")
                self.assertEqual(r_sem_all.status_code, 200)
                self.assertIn("DJI_20250101_001.jpg", r_sem_all.text)
                self.assertIn("DJI_20250101_vault.jpg", r_sem_all.text)
                self.assertIn("DJI_20250101_green.jpg", r_sem_all.text)
                self.assertIn("DJI_20250101_orange.jpg", r_sem_all.text)

                # Home HTML includes reset of 'q', stopPropagation on #main-search-input, online_only, and Odśwież button without keyup auto-search
                r_home_html = self.client.get("/")
                self.assertEqual(r_home_html.status_code, 200)
                self.assertIn("['q', 'volume_id'", r_home_html.text)
                self.assertIn("onchange=\"event.stopPropagation()\"", r_home_html.text)
                self.assertIn('id="search-refresh-btn"', r_home_html.text)
                self.assertIn("🔄 Odśwież", r_home_html.text)
                self.assertNotIn("keyup changed delay:250ms", r_home_html.text)
                self.assertIn("name=\"online_only\"", r_home_html.text)

                # Filter: online_only=true keeps only online drives and retains return-to-all button
                r_online_only = self.client.get("/media/search?online_only=true")
                self.assertEqual(r_online_only.status_code, 200)
                self.assertIn("queue_08.jpg", r_online_only.text)
                self.assertIn("DJI_20250101_001.jpg", r_online_only.text)
                self.assertNotIn("queue_00.jpg", r_online_only.text)
                self.assertNotIn("queue_10.jpg", r_online_only.text)
                self.assertIn("Wszystkie urządzenia", r_online_only.text)
                self.assertIn("Tylko dostępne online", r_online_only.text)

                # Selecting a single offline (or online) volume still renders return-to-all button
                r_vol_sel = self.client.get(f"/media/search?volume_id={vol_off1.id}")
                self.assertEqual(r_vol_sel.status_code, 200)
                self.assertIn("Wszystkie urządzenia", r_vol_sel.text)
                self.assertIn("queue_00.jpg", r_vol_sel.text)
                self.assertNotIn("queue_08.jpg", r_vol_sel.text)

                r_vol_on_sel = self.client.get(f"/media/search?volume_id={vol_on.id}")
                self.assertEqual(r_vol_on_sel.status_code, 200)
                self.assertIn("Wszystkie urządzenia", r_vol_on_sel.text)

    def test_strict_target_writes_only_protection_and_gui_polish(self):
        drive_c = self.root / "DriveC_Protected"
        drive_c.mkdir(parents=True, exist_ok=True)
        with self.TestingSessionLocal() as db:
            vol_bkp1 = WindowsVolume(
                volume_serial_number="BK01-1111",
                volume_label="Backup-1",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=80 * 1024**3,
            )
            vol_foto1 = WindowsVolume(
                volume_serial_number="FT01-2222",
                volume_label="Foto-1",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            vol_old3 = WindowsVolume(
                volume_serial_number="OD03-3333",
                volume_label="OldDisk-3",
                last_known_drive_letter=str(drive_c),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            db.add_all([vol_bkp1, vol_foto1, vol_old3])
            db.commit()
            for v in (vol_bkp1, vol_foto1, vol_old3):
                db.refresh(v)

            vsn_all3 = {
                "BK01-1111": (str(self.drive_a), "Backup-1"),
                "FT01-2222": (str(self.drive_b), "Foto-1"),
                "OD03-3333": (str(drive_c), "OldDisk-3"),
            }

            # 1) Unbacked original on OldDisk-3 -> should get new backup on Backup-1/Foto-1 (never OldDisk-3)
            p_old_unb = create_synthetic_jpeg(drive_c / "old_unbacked.jpg", color=(10, 20, 30))
            it_old_unb = FileScannerService.process_file(p_old_unb, vol_old3)

            # 2) Orange same-device duplicate pair on OldDisk-3 -> new cross backup on Backup-1/Foto-1, NO 'Ważne' on OldDisk-3
            p_old_imp1 = create_synthetic_jpeg(drive_c / "old_imp.jpg", color=(30, 40, 50))
            p_old_imp2 = drive_c / "old_imp_copy.jpg"
            p_old_imp2.write_bytes(p_old_imp1.read_bytes())
            it_old_imp1 = FileScannerService.process_file(p_old_imp1, vol_old3)
            it_old_imp2 = FileScannerService.process_file(p_old_imp2, vol_old3)

            # 3) Orange same-device duplicate pair on Foto-1 (SELECTED!) -> 'Ważne - Foto-1' SHOULD be created & assigned
            p_foto_imp1 = create_synthetic_jpeg(self.drive_b / "foto_imp.jpg", color=(60, 70, 80))
            p_foto_imp2 = self.drive_b / "foto_imp_copy.jpg"
            p_foto_imp2.write_bytes(p_foto_imp1.read_bytes())
            it_foto_imp1 = FileScannerService.process_file(p_foto_imp1, vol_foto1)
            it_foto_imp2 = FileScannerService.process_file(p_foto_imp2, vol_foto1)

            # 4) Green outside backup on OldDisk-3 for an original on Foto-1 -> must NOT be moved locally on OldDisk-3
            p_foto_g_orig = create_synthetic_jpeg(self.drive_b / "foto_with_old_green.jpg", color=(90, 100, 110))
            p_old_g_bkp = drive_c / "old_green_outside.jpg"
            p_old_g_bkp.write_bytes(p_foto_g_orig.read_bytes())
            it_foto_g_orig = FileScannerService.process_file(p_foto_g_orig, vol_foto1)
            it_old_g_bkp = FileScannerService.process_file(p_old_g_bkp, vol_old3)

            # 5) Green outside backup on Backup-1 (SELECTED!) for an original on Foto-1 -> SHOULD move into Backup-1/daf_backup/
            p_foto_b1_orig = create_synthetic_jpeg(self.drive_b / "foto_with_bkp1_green.jpg", color=(120, 130, 140))
            p_bkp1_g_bkp = self.drive_a / "bkp1_green_outside.jpg"
            p_bkp1_g_bkp.write_bytes(p_foto_b1_orig.read_bytes())
            it_foto_b1_orig = FileScannerService.process_file(p_foto_b1_orig, vol_foto1)
            it_bkp1_g_bkp = FileScannerService.process_file(p_bkp1_g_bkp, vol_bkp1)

            db.add_all([
                it_old_unb, it_old_imp1, it_old_imp2,
                it_foto_imp1, it_foto_imp2,
                it_foto_g_orig, it_old_g_bkp,
                it_foto_b1_orig, it_bkp1_g_bkp,
            ])
            db.commit()

            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_all3):
                # Check Backup panel UI shows the new strict protection mode toggle & counts
                r_panel = self.client.post(
                    "/actions/backup-panel",
                    data={
                        "target_vols_submitted": "true",
                        "target_volume_ids": [str(vol_bkp1.id), str(vol_foto1.id)],
                        "minimize_submitted": "true",
                        "minimize_transfers": "true",
                        "strict_target_submitted": "true",
                        "strict_target_writes_only": "true",
                    },
                )
                self.assertEqual(r_panel.status_code, 200)
                self.assertIn("strict_target_writes_only", r_panel.text)
                self.assertIn("Tryb ochrony dysków aktywny", r_panel.text)

                # Execute backup via HTTP POST with only Backup-1 & Foto-1 allowed for writes
                r_exec = self.client.post(
                    "/actions/execute-backup",
                    data={
                        "target_vols_submitted": "true",
                        "target_volume_ids": [str(vol_bkp1.id), str(vol_foto1.id)],
                        "minimize_submitted": "true",
                        "minimize_transfers": "true",
                        "strict_target_submitted": "true",
                        "strict_target_writes_only": "true",
                    },
                )
                self.assertEqual(r_exec.status_code, 200)
                self.assertIn("Tryb ochrony: zapisy wyłącznie na wybranych dyskach docelowych", r_exec.text)

                db.expire_all()
                # OldDisk-3 filesystem MUST have zero daf_backup/ directory created and untouched source files
                self.assertFalse((drive_c / "daf_backup").exists())
                self.assertTrue((drive_c / "old_unbacked.jpg").exists())
                self.assertTrue((drive_c / "old_imp_copy.jpg").exists())
                self.assertTrue((drive_c / "old_green_outside.jpg").exists())
                # OldDisk-3 item must NOT be assigned to 'Ważne - OldDisk-3'
                self.assertIsNone(db.get(MediaItem, it_old_imp1.id).group_id)

                # Selected targets MUST have received new backups, local relocation on Backup-1, and 'Ważne - Foto-1'
                it_foto_imp1_db = db.get(MediaItem, it_foto_imp1.id)
                self.assertIsNotNone(it_foto_imp1_db.group_id)
                self.assertFalse((self.drive_a / "bkp1_green_outside.jpg").exists())
                self.assertTrue(db.get(MediaItem, it_bkp1_g_bkp.id).is_backup_copy)

                # All new daf_backup files written by this run must exist only on Backup-1 or Foto-1
                new_bkp_vols = set(
                    db.scalars(select(MediaItem.volume_id).where(MediaItem.is_backup_copy == True)).all()
                )
                self.assertTrue(new_bkp_vols.issubset({vol_bkp1.id, vol_foto1.id}))

                # GUI polish classes present across /, /media/sidebar, /media/search
                r_home_g = self.client.get("/")
                self.assertEqual(r_home_g.status_code, 200)
                self.assertIn("daf-modal-card", r_home_g.text)
                self.assertIn("daf-modal-header", r_home_g.text)
                self.assertIn("daf-section-toggle", r_home_g.text)

    def test_backup_offline_pair_prompt_status_and_resume(self):
        with self.TestingSessionLocal() as db:
            vol_src = WindowsVolume(
                volume_serial_number="AA01-1111",
                volume_label="SrcOfflineA",
                last_known_drive_letter=str(self.drive_a),
                is_online=False,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            vol_dst = WindowsVolume(
                volume_serial_number="BB02-2222",
                volume_label="DstOfflineB",
                last_known_drive_letter=str(self.drive_b),
                is_online=False,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=80 * 1024**3,
            )
            db.add_all([vol_src, vol_dst])
            db.commit()
            db.refresh(vol_src)
            db.refresh(vol_dst)

            p1 = create_synthetic_jpeg(self.drive_a / "off_pair_1.jpg", color=(15, 35, 55))
            p2 = create_synthetic_jpeg(self.drive_a / "off_pair_2.jpg", color=(65, 85, 105))
            it1 = FileScannerService.process_file(p1, vol_src)
            it2 = FileScannerService.process_file(p2, vol_src)
            db.add_all([it1, it2])
            db.commit()

            # 1) Start backup while BOTH drives are offline (only unrelated SysC connected)
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={"CC00-0001": ("C:\\", "SysC")}):
                r_start = self.client.post(
                    "/actions/execute-backup",
                    data={
                        "target_vols_submitted": "true",
                        "target_volume_ids": [str(vol_dst.id)],
                        "minimize_submitted": "true",
                        "minimize_transfers": "true",
                    },
                )
                self.assertEqual(r_start.status_code, 200)
                self.assertIn("Podłączyłem wskazaną parę dysków — Kontynuuj backup", r_start.text)
                self.assertIn("SrcOfflineA", r_start.text)
                self.assertIn("DstOfflineB", r_start.text)

                job = db.scalars(
                    select(BackgroundJob)
                    .where(BackgroundJob.job_type == "backup_exec")
                    .order_by(BackgroundJob.id.desc())
                    .limit(1)
                ).first()
                self.assertIsNotNone(job)
                self.assertEqual(job.status, "WAITING_FOR_DEVICE")
                self.assertEqual(job.total_count, 2)
                self.assertEqual(job.processed_count, 0)
                self.assertIn("SrcOfflineA", job.last_message)
                self.assertIn("DstOfflineB", job.last_message)

                # 2) Clicking Resume while drives are STILL offline must keep WAITING_FOR_DEVICE (NOT COMPLETED 0/0)
                r_resume_still_off = self.client.post(
                    "/actions/process-control",
                    data={"action": "resume", "job_id": str(job.id)},
                )
                self.assertEqual(r_resume_still_off.status_code, 200)
                db.expire_all()
                job_still = db.get(BackgroundJob, job.id)
                self.assertEqual(job_still.status, "WAITING_FOR_DEVICE")
                self.assertEqual(job_still.total_count, 2)
                self.assertIn("SrcOfflineA", job_still.last_message)
                self.assertIn("DstOfflineB", job_still.last_message)

            # 3) Connect both drives and click Resume in Status panel -> must actually copy backups & mark COMPLETED 2/2
            vsn_both = {
                "AA01-1111": (str(self.drive_a), "SrcOfflineA"),
                "BB02-2222": (str(self.drive_b), "DstOfflineB"),
            }
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
                r_resume_online = self.client.post(
                    "/actions/process-control",
                    data={"action": "resume", "job_id": str(job.id)},
                )
                self.assertEqual(r_resume_online.status_code, 200)
                db.expire_all()
                job_done = db.get(BackgroundJob, job.id)
                self.assertEqual(job_done.status, "COMPLETED")
                self.assertEqual(job_done.processed_count, 2)
                self.assertEqual(job_done.total_count, 2)

                bkp_items = list(
                    db.scalars(
                        select(MediaItem).where(MediaItem.is_backup_copy == True, MediaItem.volume_id == vol_dst.id)
                    ).all()
                )
                self.assertEqual(len(bkp_items), 2)

    def test_scanner_and_backup_skip_recycle_bin_and_avoid_requeueing_attempted_pairs(self):
        """
        1) Scanner skips $RECYCLE.BIN and System Volume Information folders.
        2) Existing $RECYCLE.BIN/$I* rows in DB are excluded from backup planning (no infinite verification block loops).
        3) Multi-pair backup_exec jobs remember attempted_pair_keys across resumes so a pair already processed
           online (e.g. with a missing physical source file) is NOT requested again when resuming later pairs.
        """
        from app.scanner import FileScannerService

        self.assertTrue(FileScannerService.is_in_excluded_album_export_folder("$RECYCLE.BIN/$IQDWKV0.JPG"))
        self.assertTrue(FileScannerService.is_in_excluded_album_export_folder("System Volume Information/IndexerVolumeGuid"))

        recycle_dir = self.drive_a / "$RECYCLE.BIN"
        recycle_dir.mkdir(parents=True, exist_ok=True)
        (recycle_dir / "$IQDWKV0.JPG").write_bytes(b"\x02\x00\x00\x00" + b"\x00" * 120)
        sys_dir = self.drive_a / "System Volume Information"
        sys_dir.mkdir(parents=True, exist_ok=True)
        (sys_dir / "meta.jpg").write_bytes(b"\x01\x02\x03")
        create_synthetic_jpeg(self.drive_a / "valid_photo.jpg", color=(40, 80, 120))

        with patch.object(
            WindowsVolumeInspector,
            "get_volume_info",
            return_value=("AA01-0001", "ScanVolA", str(self.drive_a)),
        ), patch.object(
            WindowsVolumeInspector,
            "get_connected_vsn_map",
            return_value={"AA01-0001": (str(self.drive_a), "ScanVolA")},
        ):
            r_scan = self.client.post("/actions/scan", data={"path": str(self.drive_a)})
            self.assertEqual(r_scan.status_code, 200)

        with self.TestingSessionLocal() as db:
            items = list(db.scalars(select(MediaItem).where(MediaItem.is_trashed == False)).all())
            names = [it.file_name for it in items]
            self.assertIn("valid_photo.jpg", names)
            self.assertNotIn("$IQDWKV0.JPG", names)
            self.assertNotIn("meta.jpg", names)

            # Prepare 3 drives: Vol1 (missing physical file + pre-existing $RECYCLE.BIN/$I row), Vol2 (valid unbacked), BackupVol
            vol1 = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "AA01-0001"))
            vol1.total_space_bytes = 50 * 1024**3
            vol1.free_space_bytes = 40 * 1024**3
            # Delete valid_photo.jpg physically after scan to simulate missing source file on Vol1
            (self.drive_a / "valid_photo.jpg").unlink()

            # Also insert an existing $RECYCLE.BIN/$I* row that might have been scanned by an older version
            recycle_item = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol1.id,
                relative_path="$RECYCLE.BIN/$IQDWKV0.JPG",
                file_name="$IQDWKV0.JPG",
                file_size_bytes=544,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="recycle_header_hash_1",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2025, 1, 1),
                fs_mtime=datetime.datetime(2025, 1, 1),
                effective_date=datetime.datetime(2025, 1, 1),
                thumbnail_relative_path="rec_thumb.webp",
                is_corrupted=True,
                duplicate_role="original",
            )
            db.add(recycle_item)

            drive_c = self.root / "drive_c"
            drive_c.mkdir(parents=True, exist_ok=True)
            vol2 = WindowsVolume(
                volume_serial_number="CC02-0002",
                volume_label="ScanVolC",
                last_known_drive_letter=str(drive_c),
                is_online=False,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            vol_bkp = WindowsVolume(
                volume_serial_number="BB03-0003",
                volume_label="BackupDest",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=100 * 1024**3,
                free_space_bytes=90 * 1024**3,
            )
            db.add_all([vol2, vol_bkp])
            db.commit()
            db.refresh(vol2)
            db.refresh(vol_bkp)

            p_c = create_synthetic_jpeg(drive_c / "c_photo.jpg", color=(90, 20, 40))
            it_c = FileScannerService.process_file(p_c, vol2)
            db.add(it_c)
            db.commit()

            svc = StorageAutomationService(db)
            plan = svc.build_backup_and_dedup_plan(target_volume_ids=[vol_bkp.id])
            self.assertEqual(plan["skipped_corrupted_or_recycle"], 1)
            planned_names = [t["file_name"] for p in plan["scheduled_pairs"] for t in p["tasks"]["backups_to_copy"]]
            self.assertNotIn("$IQDWKV0.JPG", planned_names)
            self.assertIn("valid_photo.jpg", planned_names)
            self.assertIn("c_photo.jpg", planned_names)

            # Run 1: Vol1 + BackupDest online, Vol2 offline -> checks Vol1<->BackupDest (missing_source_files=1), waits ONLY for Vol2<->BackupDest
            vsn_run1 = {
                "AA01-0001": (str(self.drive_a), "ScanVolA"),
                "BB03-0003": (str(self.drive_b), "BackupDest"),
            }
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_run1):
                r1 = self.client.post(
                    "/actions/execute-backup",
                    data={
                        "target_vols_submitted": "true",
                        "target_volume_ids": [str(vol_bkp.id)],
                        "minimize_submitted": "true",
                        "minimize_transfers": "true",
                    },
                )
                self.assertEqual(r1.status_code, 200)
                self.assertIn("Brakujące pliki źródłowe na dysku", r1.text)
                self.assertIn("ScanVolC", r1.text)
                job = db.scalars(
                    select(BackgroundJob).where(BackgroundJob.job_type == "backup_exec").order_by(BackgroundJob.id.desc()).limit(1)
                ).first()
                self.assertEqual(job.status, "WAITING_FOR_DEVICE")
                params_1 = json.loads(job.params_json)
                pair_v1_bkp = f"{min(vol1.id, vol_bkp.id)}:{max(vol1.id, vol_bkp.id)}"
                self.assertIn(pair_v1_bkp, params_1.get("attempted_pair_keys", []))

            # Run 2: Swap drives -> disconnect Vol1, connect Vol2 + BackupDest -> resume must NOT ask for ScanVolA again!
            vsn_run2 = {
                "CC02-0002": (str(drive_c), "ScanVolC"),
                "BB03-0003": (str(self.drive_b), "BackupDest"),
            }
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_run2):
                r2 = self.client.post(
                    "/actions/process-control",
                    data={"action": "resume", "job_id": str(job.id)},
                )
                self.assertEqual(r2.status_code, 200)
                db.expire_all()
                job_after = db.get(BackgroundJob, job.id)
                self.assertEqual(job_after.status, "COMPLETED")
                self.assertNotIn("ScanVolA", job_after.last_message)
                bkp_c = list(
                    db.scalars(
                        select(MediaItem).where(MediaItem.is_backup_copy == True, MediaItem.volume_id == vol_bkp.id)
                    ).all()
                )
                self.assertEqual(len(bkp_c), 1)
                self.assertEqual(bkp_c[0].file_name, "c_photo.jpg")

    def test_auto_verify_after_duplicate_removal_and_no_standalone_button(self):
        """
        Permanent duplicate removal must automatically run verify_deleted_duplicates_integrity()
        upon completion and include the verification report in the response and job message,
        while the separate standalone verification button is removed from maintenance tools.
        """
        import time
        from app.scanner import FileScannerService
        from app.routers import actions as actions_mod

        r_sp = self.client.get("/actions/status-panel")
        self.assertEqual(r_sp.status_code, 200)
        self.assertNotIn("Weryfikuj usunięte duplikaty (100% Oryginałów)", r_sp.text)

        p_orig = create_synthetic_jpeg(self.drive_a / "auto_ver_orig.jpg", color=(25, 110, 175))
        p_bkp = self.drive_b / "daf_backup" / "2025" / "02" / "02" / "auto_ver_orig.jpg"
        p_bkp.parent.mkdir(parents=True, exist_ok=True)
        p_bkp.write_bytes(p_orig.read_bytes())
        p_dup = self.drive_a / "auto_ver_dup.jpg"
        p_dup.write_bytes(p_orig.read_bytes())

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="AA11-1111",
                volume_label="AutoVerA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            vol_b = WindowsVolume(
                volume_serial_number="BB22-2222",
                volume_label="AutoVerB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            db.add_all([vol_a, vol_b])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)

            it_orig = FileScannerService.process_file(p_orig, vol_a)
            it_bkp = FileScannerService.process_file(p_bkp, vol_b)
            it_dup = FileScannerService.process_file(p_dup, vol_a)
            it_orig.duplicate_role = "original"
            it_orig.is_primary_duplicate = True
            it_bkp.is_backup_copy = True
            it_bkp.duplicate_role = "backup_green"
            it_bkp.is_primary_duplicate = False
            it_dup.duplicate_role = "duplicate"
            it_dup.is_primary_duplicate = False
            db.add_all([it_orig, it_bkp, it_dup])
            db.commit()
            dup_id = it_dup.id

        vsn_both = {
            "AA11-1111": (str(self.drive_a), "AutoVerA"),
            "BB22-2222": (str(self.drive_b), "AutoVerB"),
        }
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
            r_del = self._delete_permanently_two_phase(
                {
                    "scope": "selected",
                    "selected_ids": [dup_id],
                    "duplicates_only": "true",
                    "skip_db_backup": "true",
                },
            )
            self.assertEqual(r_del.status_code, 200)
            self.assertIn("Automatyczna weryfikacja 100% Oryginałów po usunięciu", r_del.text)
            self.assertIn("Liczba usuniętych duplikatów: 1", r_del.text)
            self.assertIn("1 z 1 (100.0%)", r_del.text)

        with self.TestingSessionLocal() as db:
            job = db.scalars(
                select(BackgroundJob)
                .where(BackgroundJob.job_type == "delete_permanently")
                .order_by(BackgroundJob.id.desc())
                .limit(1)
            ).first()
            self.assertIsNotNone(job)
            self.assertEqual(job.status, "COMPLETED")
            self.assertIn("1 z 1 (100.0%)", job.last_message)
            self.assertFalse(p_dup.exists())

    def test_important_album_action_move_path_preservation_and_offline_backup_preview_fallback(self):
        """
        1) POST /actions/assign-important-albums assigns Originals with >=2 non-backup copies on the same disk
           to 'Ważne - <device>/YYYY/MM/DD' purely in DB (works when drives are offline).
        2) organize_ungrouped_originals moves without copying, and both organize_ungrouped_originals and
           schedule_or_execute_moves preserve original_relative_path across multiple moves (searchable via /media/search).
        3) When an Original's disk is offline but a backup copy is online, /media/{id}/raw, /media/search
           (lightbox/slideshow), and /media/{id}/detail serve full media read-only from the online backup
           without touching the backup when operations target the Original.
        """
        from app.scanner import FileScannerService

        p_imp1 = create_synthetic_jpeg(self.drive_a / "wakacje_stare" / "imp_photo.jpg", color=(33, 66, 99))
        p_imp2 = self.drive_a / "kopia_lokalna" / "imp_photo_copy.jpg"
        p_imp2.parent.mkdir(parents=True, exist_ok=True)
        p_imp2.write_bytes(p_imp1.read_bytes())

        p_move = create_synthetic_jpeg(self.drive_a / "archiwum_2022" / "move_me.jpg", color=(120, 45, 75))
        p_off = create_synthetic_jpeg(self.drive_a / "offline_src" / "off_photo.jpg", color=(190, 140, 30))
        p_off_bkp = self.drive_b / "daf_backup" / "2025" / "03" / "03" / "off_photo.jpg"
        p_off_bkp.parent.mkdir(parents=True, exist_ok=True)
        p_off_bkp.write_bytes(p_off.read_bytes())

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="AA44-1111",
                volume_label="FotoOffA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            vol_b = WindowsVolume(
                volume_serial_number="BB44-2222",
                volume_label="BackupOnB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
                total_space_bytes=50 * 1024**3,
                free_space_bytes=40 * 1024**3,
            )
            db.add_all([vol_a, vol_b])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)

            it_imp1 = FileScannerService.process_file(p_imp1, vol_a)
            it_imp2 = FileScannerService.process_file(p_imp2, vol_a)
            it_move = FileScannerService.process_file(p_move, vol_a)
            it_off = FileScannerService.process_file(p_off, vol_a)
            it_off_bkp = FileScannerService.process_file(p_off_bkp, vol_b)

            it_imp1.duplicate_role = "original"
            it_imp1.is_primary_duplicate = True
            it_imp2.duplicate_role = "duplicate"
            it_imp2.is_primary_duplicate = False
            it_move.duplicate_role = "original"
            it_move.is_primary_duplicate = True
            it_off.duplicate_role = "original"
            it_off.is_primary_duplicate = True
            it_off_bkp.is_backup_copy = True
            it_off_bkp.duplicate_role = "backup_green"
            it_off_bkp.is_primary_duplicate = False

            # Also add an offline video with an online backup copy
            p_vid_bkp = self.drive_b / "daf_backup" / "2025" / "03" / "04" / "clip_off.mp4"
            p_vid_bkp.parent.mkdir(parents=True, exist_ok=True)
            p_vid_bkp.write_bytes(b"FAKE_MP4_STREAM_PAYLOAD_1234567890")
            it_vid_off = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_a.id,
                relative_path="offline_src/clip_off.mp4",
                file_name="clip_off.mp4",
                file_size_bytes=34,
                media_type="video",
                mime_type="video/mp4",
                file_hash="vid_fallback_hash_44",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2025, 3, 4, 12, 0, 0),
                fs_mtime=datetime.datetime(2025, 3, 4, 12, 0, 0),
                effective_date=datetime.datetime(2025, 3, 4, 12, 0, 0),
                thumbnail_relative_path="vid_fb_thumb.webp",
                duplicate_role="original",
                is_primary_duplicate=True,
            )
            it_vid_bkp = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_b.id,
                relative_path="daf_backup/2025/03/04/clip_off.mp4",
                file_name="clip_off.mp4",
                file_size_bytes=34,
                media_type="video",
                mime_type="video/mp4",
                file_hash="vid_fallback_hash_44",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2025, 3, 4, 12, 0, 0),
                fs_mtime=datetime.datetime(2025, 3, 4, 12, 0, 0),
                effective_date=datetime.datetime(2025, 3, 4, 12, 0, 0),
                thumbnail_relative_path="vid_fb_thumb.webp",
                is_backup_copy=True,
                duplicate_role="backup_green",
                is_primary_duplicate=False,
            )
            db.add_all([it_imp1, it_imp2, it_move, it_off, it_off_bkp, it_vid_off, it_vid_bkp])
            db.commit()

            vol_a_id, vol_b_id = vol_a.id, vol_b.id
            imp1_id, move_id = it_imp1.id, it_move.id
            off_id, off_bkp_id = it_off.id, it_off_bkp.id
            vid_off_id = it_vid_off.id

            # 1) Assign 'Ważne' album purely in DB even while FotoOffA is offline
            vol_a.is_online = False
            db.commit()
            r_imp = self.client.post("/actions/assign-important-albums")
            self.assertEqual(r_imp.status_code, 200)
            self.assertIn("1 oryginałów", r_imp.text)
            db.expire_all()
            imp1_db = db.get(MediaItem, imp1_id)
            self.assertIsNotNone(imp1_db.group_id)
            move_db = db.get(MediaItem, move_id)
            self.assertIsNone(move_db.group_id)

            # 2) Organize ungrouped originals moves without copying, then schedule_or_execute_moves preserves BOTH paths
            vol_a_db = db.get(WindowsVolume, vol_a_id)
            vol_a_db.is_online = True
            db.commit()
            vsn_both = {
                "AA44-1111": (str(self.drive_a), "FotoOffA"),
                "BB44-2222": (str(self.drive_b), "BackupOnB"),
            }
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
                svc = StorageAutomationService(db)
                org_res = svc.organize_ungrouped_originals(volume_id=vol_a_id)
                self.assertGreaterEqual(org_res["moved_count"], 1)
                self.assertFalse(p_move.exists())  # Moved, NOT copied!
                db.expire_all()
                move_db = db.get(MediaItem, move_id)
                self.assertTrue(move_db.relative_path.startswith("daf_oryginal/"))
                self.assertIn("FotoOffA/archiwum_2022/move_me.jpg", move_db.original_relative_path)
                self.assertTrue((self.drive_a / move_db.relative_path).exists())

                # Move again to another subfolder via schedule_or_execute_moves -> original path must remain preserved!
                mv_res = svc.schedule_or_execute_moves([move_db], vol_a_id, target_subfolder="wybrane_final")
                self.assertEqual(mv_res["moved_physically"], 1)
                db.expire_all()
                move_db = db.get(MediaItem, move_id)
                self.assertTrue(move_db.relative_path.startswith("wybrane_final/"))
                self.assertIn("FotoOffA/archiwum_2022/move_me.jpg", move_db.original_relative_path)
                self.assertIn("daf_oryginal/", move_db.original_relative_path)

            # Search by original path finds the twice-moved original
            r_search_orig = self.client.get("/media/search", params={"q": "archiwum_2022"})
            self.assertEqual(r_search_orig.status_code, 200)
            self.assertIn("move_me.jpg", r_search_orig.text)

            # 3) Disconnect FotoOffA -> BackupOnB stays online -> /raw, /search, and /detail use read-only backup fallback
            vol_a_db = db.get(WindowsVolume, vol_a_id)
            vol_a_db.is_online = False
            db.commit()
            vsn_only_b = {"BB44-2222": (str(self.drive_b), "BackupOnB")}
            with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_only_b):
                r_raw = self.client.get(f"/media/{off_id}/raw")
                self.assertEqual(r_raw.status_code, 200)
                self.assertEqual(r_raw.content, p_off_bkp.read_bytes())
                self.assertIn("BackupOnB", r_raw.headers.get("X-DAF-Preview-Fallback", ""))

                r_raw_vid = self.client.get(f"/media/{vid_off_id}/raw")
                self.assertEqual(r_raw_vid.status_code, 200)
                self.assertEqual(r_raw_vid.content, b"FAKE_MP4_STREAM_PAYLOAD_1234567890")
                self.assertIn("BackupOnB", r_raw_vid.headers.get("X-DAF-Preview-Fallback", ""))

                r_grid = self.client.get("/media/search", params={"q": "off_photo.jpg"})
                self.assertEqual(r_grid.status_code, 200)
                self.assertIn('data-fallback-label="BackupOnB', r_grid.text)

                r_vid_det = self.client.get(f"/media/{vid_off_id}/detail")
                self.assertEqual(r_vid_det.status_code, 200)
                self.assertIn(f'<video src="/media/{vid_off_id}/raw"', r_vid_det.text)
                self.assertIn("Podgląd z kopii online", r_vid_det.text)

                # Mutating operation (rotate) on Original affects ONLY Original, never touches backup copy
                r_rot = self.client.post("/actions/rotate-media", data={"media_id": off_id, "delta": "90"})
                self.assertEqual(r_rot.status_code, 200)
                db.expire_all()
                self.assertEqual(db.get(MediaItem, off_id).rotation_angle, 90)
                self.assertEqual(db.get(MediaItem, off_bkp_id).rotation_angle, 0)
                self.assertTrue(p_off_bkp.exists())

            # 4) Verify legacy media_groups UNIQUE(name) migration drops the constraint while preserving rows
            from app import database as db_module
            legacy_db = self.root / "legacy_unique_groups.db"
            lconn = sqlite3.connect(legacy_db.as_posix())
            lconn.execute(
                """
                CREATE TABLE media_groups (
                    id INTEGER NOT NULL PRIMARY KEY,
                    name VARCHAR(128) NOT NULL,
                    description TEXT,
                    target_volume_id INTEGER,
                    created_at DATETIME NOT NULL,
                    UNIQUE (name)
                )
                """
            )
            lconn.execute(
                "INSERT INTO media_groups (id, name, created_at) VALUES (1, '2024', '2025-01-01 00:00:00')"
            )
            lconn.commit()
            lconn.close()

            db_module.ensure_schema_migrations(legacy_db)

            lconn2 = sqlite3.connect(legacy_db.as_posix())
            ddl_after = lconn2.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='media_groups'"
            ).fetchone()[0]
            self.assertNotIn("UNIQUE", ddl_after.upper())
            # Inserting a second sub-album named '2024' under a different parent must now succeed
            lconn2.execute(
                "INSERT INTO media_groups (name, parent_id, is_trashed, created_at) VALUES ('2024', 1, 0, '2025-01-02 00:00:00')"
            )
            cnt_2024 = lconn2.execute("SELECT COUNT(*) FROM media_groups WHERE name='2024'").fetchone()[0]
            lconn2.close()
            self.assertEqual(cnt_2024, 2)

    def test_delete_permanently_respects_unbacked_filter_vsn_check_and_blocks_unfiltered_wipe(self):
        """
        Regression test for critical safety fixes:
        1) scope=filtered with unbacked_only=true deletes ONLY unbacked items, NEVER backed-up originals.
        2) _execute_delete_permanently_items verifies VSN and NEVER deletes files on a swapped drive letter.
        3) scope=filtered with no active filters is refused to prevent accidental full-archive wipe.
        """
        from app.scanner import FileScannerService

        p_backed_orig = create_synthetic_jpeg(self.drive_a / "family_safe.jpg", color=(10, 90, 200))
        p_backed_bkp = self.drive_b / "daf_backup" / "2025" / "05" / "05" / "family_safe.jpg"
        p_backed_bkp.parent.mkdir(parents=True, exist_ok=True)
        p_backed_bkp.write_bytes(p_backed_orig.read_bytes())

        p_unbacked = create_synthetic_jpeg(self.drive_a / "unbacked_only.jpg", color=(200, 40, 20))
        # File on drive_a with same relative path as an offline volume claiming drive_a's letter
        p_swapped = create_synthetic_jpeg(self.drive_a / "shared_rel.jpg", color=(70, 180, 90))

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="AA99-1111",
                volume_label="SafeDriveA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            vol_b = WindowsVolume(
                volume_serial_number="BB99-2222",
                volume_label="SafeBackupB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
            )
            vol_off_same_letter = WindowsVolume(
                volume_serial_number="CC99-3333",
                volume_label="OfflineSameLetter",
                last_known_drive_letter=str(self.drive_a),  # Same letter as SafeDriveA, different VSN!
                is_online=False,
            )
            db.add_all([vol_a, vol_b, vol_off_same_letter])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)
            db.refresh(vol_off_same_letter)

            it_orig = FileScannerService.process_file(p_backed_orig, vol_a)
            it_bkp = FileScannerService.process_file(p_backed_bkp, vol_b)
            it_unb = FileScannerService.process_file(p_unbacked, vol_a)
            it_orig.duplicate_role = "original"
            it_bkp.is_backup_copy = True
            it_bkp.duplicate_role = "backup_green"
            it_bkp.is_primary_duplicate = False
            it_unb.duplicate_role = "original"

            it_off_fake = MediaItem(
                id=str(uuid.uuid4()),
                volume_id=vol_off_same_letter.id,
                relative_path="shared_rel.jpg",
                file_name="shared_rel.jpg",
                file_size_bytes=p_swapped.stat().st_size,
                media_type="image",
                mime_type="image/jpeg",
                file_hash="swapped_vsn_unique_hash",
                hash_algorithm="sha256",
                fs_ctime=datetime.datetime(2025, 5, 5),
                fs_mtime=datetime.datetime(2025, 5, 5),
                effective_date=datetime.datetime(2025, 5, 5),
                thumbnail_relative_path="swap_thumb.webp",
                duplicate_role="original",
            )
            db.add_all([it_orig, it_bkp, it_unb, it_off_fake])
            db.commit()
            orig_id, unb_id, off_fake_id = it_orig.id, it_unb.id, it_off_fake.id

        vsn_map = {
            "AA99-1111": (str(self.drive_a), "SafeDriveA"),
            "BB99-2222": (str(self.drive_b), "SafeBackupB"),
        }
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            # 1) Unfiltered scope=filtered must be BLOCKED
            r_blk = self.client.post(
                "/actions/delete-permanently",
                data={"scope": "filtered", "skip_db_backup": "true"},
            )
            self.assertEqual(r_blk.status_code, 200)
            self.assertIn("Zablokowano trwałe usuwanie", r_blk.text)
            self.assertTrue(p_backed_orig.exists())
            self.assertTrue(p_unbacked.exists())

            # 2) scope=filtered with unbacked_only=true deletes ONLY unbacked items, NEVER backed-up family_safe.jpg
            r_unb = self._delete_permanently_two_phase(
                {"scope": "filtered", "unbacked_only": "true", "skip_db_backup": "true"},
            )
            self.assertEqual(r_unb.status_code, 200)
            self.assertTrue(p_backed_orig.exists())
            self.assertFalse(p_unbacked.exists())
            # 3) VSN mismatch protection: OfflineSameLetter pointed at drive_a/shared_rel.jpg, MUST NOT delete drive_a/shared_rel.jpg!
            self.assertTrue(p_swapped.exists())

        with self.TestingSessionLocal() as db:
            self.assertIsNotNone(db.get(MediaItem, orig_id))
            self.assertIsNone(db.get(MediaItem, unb_id))
            # V-03 protection: offline volume DB record MUST remain intact during permanent delete
            self.assertIsNotNone(db.get(MediaItem, off_fake_id))

    def test_daf_backup_immutability_shared_thumb_protection_and_delete_preview(self):
        """
        1) _safe_unlink and _safe_move hard-block deleting or overwriting existing files in daf_backup/.
        2) POST /actions/delete-permanently refuses to delete daf_backup/ items and preserves thumbnails shared with backups.
        3) POST /actions/delete-preview returns exact file count and breakdown by drive label (without drive letters).
        """
        from app.scanner import FileScannerService
        from app.storage_ops import _safe_unlink, _safe_move, _safe_copy_new_to_daf_backup

        p_orig = create_synthetic_jpeg(self.drive_a / "orig_shared_thumb.jpg", color=(80, 120, 160))
        p_bkp = self.drive_b / "daf_backup" / "2025" / "06" / "06" / "orig_shared_thumb.jpg"
        p_bkp.parent.mkdir(parents=True, exist_ok=True)
        p_bkp.write_bytes(p_orig.read_bytes())

        # 1) Hard guard on daf_backup/
        with self.assertRaises(PermissionError):
            _safe_unlink(p_bkp)
        with self.assertRaises(PermissionError):
            _safe_move(p_bkp, self.drive_a / "stolen.jpg")
        with self.assertRaises(PermissionError):
            _safe_copy_new_to_daf_backup(p_orig, p_bkp)  # existing file in daf_backup/ cannot be overwritten
        self.assertTrue(p_bkp.exists())

        # Writing a brand-new file into daf_backup/ is allowed and 100% verified
        p_new_bkp = self.drive_b / "daf_backup" / "2025" / "06" / "06" / "brand_new_backup.jpg"
        ok_new, _, sha_new = _safe_copy_new_to_daf_backup(p_orig, p_new_bkp)
        self.assertTrue(ok_new)
        self.assertTrue(p_new_bkp.exists())
        self.assertIsNotNone(sha_new)

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="AA77-1111",
                volume_label="LabelOrigA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            vol_b = WindowsVolume(
                volume_serial_number="BB77-2222",
                volume_label="LabelBackupB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
            )
            db.add_all([vol_a, vol_b])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)

            it_orig = FileScannerService.process_file(p_orig, vol_a)
            it_bkp = FileScannerService.process_file(p_bkp, vol_b)
            it_orig.duplicate_role = "original"
            it_bkp.is_backup_copy = True
            it_bkp.duplicate_role = "backup_green"
            it_bkp.is_primary_duplicate = False
            # Share thumbnail filename between original and backup copy
            it_bkp.thumbnail_relative_path = it_orig.thumbnail_relative_path
            db.add_all([it_orig, it_bkp])
            db.commit()
            orig_id, bkp_id = it_orig.id, it_bkp.id
            shared_thumb_path = self.thumbs_dir / it_orig.thumbnail_relative_path
            self.assertTrue(shared_thumb_path.exists())

        vsn_map = {
            "AA77-1111": (str(self.drive_a), "LabelOrigA"),
            "BB77-2222": (str(self.drive_b), "LabelBackupB"),
        }
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            # 2) Pre-delete preview shows drive label breakdown & blocks daf_backup/ item automatically
            r_prev = self.client.post(
                "/actions/delete-preview",
                data={"scope": "selected", "selected_ids": [orig_id, bkp_id]},
            )
            self.assertEqual(r_prev.status_code, 200)
            self.assertIn("Podsumowanie przed trwałym usunięciem (1 plików)", r_prev.text)
            self.assertIn("LabelOrigA", r_prev.text)
            self.assertIn("Ochrona daf_backup/", r_prev.text)

            # 3) Two-phase delete-permanently on [orig_id, bkp_id] MUST NOT delete daf_backup/ item OR shared thumbnail!
            r_del = self._delete_permanently_two_phase(
                {"scope": "selected", "selected_ids": [orig_id, bkp_id], "skip_db_backup": "true"},
            )
            self.assertEqual(r_del.status_code, 200)
            self.assertTrue(p_bkp.exists())
            self.assertTrue(shared_thumb_path.exists())

        with self.TestingSessionLocal() as db:
            self.assertIsNone(db.get(MediaItem, orig_id))
            self.assertIsNotNone(db.get(MediaItem, bkp_id))

    def test_zero_trust_security_audit_v01_to_v14_and_two_phase_delete(self):
        """
        Verifies Zero-Trust File Safety fixes from the security audit:
        - V-03 & Sec 6: delete-permanently and empty-trash NEVER delete DB records for OFFLINE drives.
        - V-04: Weak filters (media_type=image alone) are rejected for scope=filtered deletion.
        - V-06: backup/ folder is protected identically to daf_backup/.
        - V-09: BackupManager._rotate preserves oversized rescue snapshots (>20% larger).
        - Sec 7: Two-Phase Commit deletion_token + Original Protection Gate (requires 'USUŃ').
        """
        import re
        from app.backup import BackupManager
        from app.scanner import FileScannerService
        from app.storage_ops import _safe_unlink

        # V-06 + NEW-BUG-03: Top-level backup/ folder is protected, while nested user subfolder DCIM/backup/ is NOT!
        p_legacy_bkp = self.drive_b / "backup" / "2024" / "legacy_bkp.jpg"
        p_legacy_bkp.parent.mkdir(parents=True, exist_ok=True)
        create_synthetic_jpeg(p_legacy_bkp, color=(12, 34, 56))
        with self.assertRaises(PermissionError):
            _safe_unlink(p_legacy_bkp, volume_root=self.drive_b)
        self.assertTrue(p_legacy_bkp.exists())

        p_nested_user_bkp = self.drive_a / "DCIM" / "backup" / "user_photo.jpg"
        p_nested_user_bkp.parent.mkdir(parents=True, exist_ok=True)
        create_synthetic_jpeg(p_nested_user_bkp, color=(56, 34, 12))
        _safe_unlink(p_nested_user_bkp, volume_root=self.drive_a)
        self.assertFalse(p_nested_user_bkp.exists())

        # V-09: Oversized rescue snapshot preserved across rotation
        rot_dir = self.root / "rot_test_backups"
        rot_dir.mkdir(parents=True, exist_ok=True)
        big_snap = rot_dir / "media_vault_pre_permanent_disk_delete_20261119_194555.db.gz"
        big_snap.write_bytes(b"X" * 5000)
        os.utime(big_snap, (1000, 1000))
        for i in range(4):
            sm = rot_dir / f"media_vault_small_{i}.db.gz"
            sm.write_bytes(b"Y" * 1000)
            os.utime(sm, (2000 + i * 10, 2000 + i * 10))
        bm = BackupManager(backup_dir=rot_dir, max_retained_backups=2)
        bm._rotate()
        self.assertTrue(big_snap.exists(), "Oversized rescue snapshot must never be auto-rotated out!")

        # V-03, V-04, Sec 6, Sec 7: offline DB preservation + Two-Phase Commit deletion_token
        p_on = create_synthetic_jpeg(self.drive_a / "zt_online.jpg", color=(90, 10, 10))
        p_off_file = create_synthetic_jpeg(self.drive_b / "zt_offline.jpg", color=(10, 90, 10))
        with self.TestingSessionLocal() as db:
            vol_on = WindowsVolume(
                volume_serial_number="AA88-1111",
                volume_label="ZTrustOnline",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            vol_off = WindowsVolume(
                volume_serial_number="BB88-2222",
                volume_label="ZTrustOffline",
                last_known_drive_letter=str(self.drive_b),
                is_online=False,
            )
            db.add_all([vol_on, vol_off])
            db.commit()
            db.refresh(vol_on)
            db.refresh(vol_off)

            it_on = FileScannerService.process_file(p_on, vol_on)
            it_off = FileScannerService.process_file(p_off_file, vol_off)
            it_on.duplicate_role = "original"
            it_off.duplicate_role = "original"
            db.add_all([it_on, it_off])
            db.commit()
            on_id, off_id = it_on.id, it_off.id

        vsn_only_a = {"AA88-1111": (str(self.drive_a), "ZTrustOnline")}
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_only_a):
            # V-04: media_type=image alone MUST NOT bypass scope=filtered guard in delete-preview
            r_weak = self.client.post(
                "/actions/delete-preview",
                data={"scope": "filtered", "media_type": "image"},
            )
            self.assertIn("Zablokowano trwałe usuwanie", r_weak.text)

            # NEW-BUG-01: Direct POST /actions/delete-permanently without deletion_token MUST BE BLOCKED!
            r_direct_bypass = self.client.post(
                "/actions/delete-permanently",
                data={"scope": "selected", "selected_ids": [on_id], "skip_db_backup": "true"},
            )
            self.assertIn("wymagany jest ważny token potwierdzenia", r_direct_bypass.text)
            self.assertTrue(p_on.exists())

            # V-03: Deleting an OFFLINE item via valid Two-Phase Commit MUST NOT remove its DB record!
            r_off_del = self._delete_permanently_two_phase(
                {"scope": "selected", "selected_ids": [off_id], "skip_db_backup": "true"},
            )
            self.assertEqual(r_off_del.status_code, 200)
            with self.TestingSessionLocal() as db:
                self.assertIsNotNone(db.get(MediaItem, off_id), "Offline drive DB record must NOT be deleted!")
            self.assertTrue(p_off_file.exists())

            # Sec 6: Logical trash + empty-trash on OFFLINE item preserves DB record; on ONLINE deletes physical file
            r_tr_off = self.client.post("/actions/move-to-trash", data={"scope": "selected", "selected_ids": [off_id]})
            self.assertEqual(r_tr_off.status_code, 200)
            r_emp = self.client.post("/actions/empty-trash", data={"selected_ids": [off_id]})
            self.assertEqual(r_emp.status_code, 200)
            self.assertIn("Pominięto dyski OFFLINE", r_emp.text)
            with self.TestingSessionLocal() as db:
                self.assertIsNotNone(db.get(MediaItem, off_id))

            # Sec 7 + NEW-BUG-01: Two-Phase Commit delete-preview -> token + Original Gate ('USUŃ')
            r_prev = self.client.post("/actions/delete-preview", data={"scope": "selected", "selected_ids": [on_id]})
            self.assertEqual(r_prev.status_code, 200)
            m_tok = re.search(r"name='deletion_token' value='([^']+)'", r_prev.text)
            self.assertIsNotNone(m_tok)
            tok = m_tok.group(1)

            # Wrong/missing 'USUŃ' text -> blocked AND token MUST NOT be destroyed!
            r_no_gate = self.client.post(
                "/actions/delete-permanently",
                data={"deletion_token": tok, "confirmed_count": "1", "allow_originals": "true", "confirm_originals_text": "TYPO", "skip_db_backup": "true"},
            )
            self.assertIn("Zablokowano usunięcie Oryginałów", r_no_gate.text)
            self.assertTrue(p_on.exists())

            # Retrying with the EXACT SAME token + 'USUŃ' MUST succeed (NEW-BUG-01 fix)!
            r_ok = self.client.post(
                "/actions/delete-permanently",
                data={
                    "deletion_token": tok,
                    "confirmed_count": "1",
                    "allow_originals": "true",
                    "confirm_originals_text": "USUŃ",
                    "skip_db_backup": "true",
                },
            )
            self.assertEqual(r_ok.status_code, 200)
            self.assertFalse(p_on.exists())

            # Replaying the SAME token a second time AFTER success MUST fail as expired/consumed!
            r_replay = self.client.post(
                "/actions/delete-permanently",
                data={
                    "deletion_token": tok,
                    "confirmed_count": "1",
                    "allow_originals": "true",
                    "confirm_originals_text": "USUŃ",
                    "skip_db_backup": "true",
                },
            )
            self.assertIn("Token potwierdzenia wygasł", r_replay.text)

    def test_code_review_regression_fixes_new_bug_01_to_03_and_brak_01_to_04(self):
        """
        End-to-end verification of all 7 fixes from the code review audit:
        - NEW-BUG-01 & NEW-BUG-02: Offline continuation token + accurate duplicates_only=false & 'plików' label.
        - NEW-BUG-03: is_path_in_daf_backup matches root backup/ only, never nested DCIM/backup/.
        - BRAK-01 (V-01): Unified filter resolution & unfiltered scope=filtered guard across move-files/move-to-trash/assign-group/rotate-media/gdrive-sync.
        - BRAK-02 (V-13): Global ProcessManager storage mutex blocks concurrent disk/DB mutations cleanly.
        - BRAK-03 (V-02): Online backup_green outside daf_backup/ in a 2-copy cluster is blocked even with force_duplicate_verification=False.
        - BRAK-04 (V-14): >999 IDs chunked cleanly in clear-quality-defect, restore-from-trash, and rotate-media.
        """
        import html as html_mod
        import re
        from app.routers import actions as actions_router
        from app.scanner import FileScannerService
        from app.storage_ops import is_path_in_daf_backup

        # NEW-BUG-03 direct path assertions
        self.assertTrue(is_path_in_daf_backup(Path("E:\\backup\\2024\\a.jpg")))
        self.assertTrue(is_path_in_daf_backup(Path("E:\\daf_backup\\2024\\a.jpg")))
        self.assertFalse(is_path_in_daf_backup(Path("E:\\DCIM\\WhatsApp\\backup\\IMG_01.jpg")))

        # Setup Drive A (online) and Drive B (online, later offline)
        p_orig_2c = create_synthetic_jpeg(self.drive_a / "pair_orig.jpg", color=(11, 22, 33))
        p_bkp_out = self.drive_b / "outside_copy" / "pair_bkp_green.jpg"
        p_bkp_out.parent.mkdir(parents=True, exist_ok=True)
        p_bkp_out.write_bytes(p_orig_2c.read_bytes())

        p_unb_only = create_synthetic_jpeg(self.drive_a / "solo_unbacked.jpg", color=(200, 120, 40))
        p_off_orig = create_synthetic_jpeg(self.drive_b / "offline_solo_orig.jpg", color=(40, 120, 200))

        with self.TestingSessionLocal() as db:
            vol_a = WindowsVolume(
                volume_serial_number="AA44-1111",
                volume_label="CRDriveA",
                last_known_drive_letter=str(self.drive_a),
                is_online=True,
            )
            vol_b = WindowsVolume(
                volume_serial_number="BB44-2222",
                volume_label="CRDriveB",
                last_known_drive_letter=str(self.drive_b),
                is_online=True,
            )
            db.add_all([vol_a, vol_b])
            db.commit()
            db.refresh(vol_a)
            db.refresh(vol_b)

            it_orig_2c = FileScannerService.process_file(p_orig_2c, vol_a)
            it_bkp_out = FileScannerService.process_file(p_bkp_out, vol_b)
            it_unb_only = FileScannerService.process_file(p_unb_only, vol_a)
            it_off_orig = FileScannerService.process_file(p_off_orig, vol_b)
            db.add_all([it_orig_2c, it_bkp_out, it_unb_only, it_off_orig])
            db.commit()

            svc = StorageAutomationService(db)
            svc.flag_duplicate_clusters()
            db.expire_all()

            orig_2c_id = it_orig_2c.id
            bkp_out_id = it_bkp_out.id
            unb_id = it_unb_only.id
            off_orig_id = it_off_orig.id
            vol_a_id, vol_b_id = vol_a.id, vol_b.id

            bkp_row = db.get(MediaItem, bkp_out_id)
            self.assertEqual(bkp_row.duplicate_role, "backup_green")
            self.assertFalse(bkp_row.is_backup_copy)

        vsn_both = {
            "AA44-1111": (str(self.drive_a), "CRDriveA"),
            "BB44-2222": (str(self.drive_b), "CRDriveB"),
        }
        vsn_only_a = {"AA44-1111": (str(self.drive_a), "CRDriveA")}

        # BRAK-03: Online backup_green outside daf_backup/ in a 2-copy cluster MUST be blocked even with force_duplicate_verification=False!
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
            with self.TestingSessionLocal() as db:
                res_bkp_blk = StorageAutomationService(db).execute_safe_duplicate_removal(
                    [bkp_out_id],
                    delete_mode="permanent",
                    force_duplicate_verification=False,
                    trash_dir=self.root / "kosz",
                    thumbnails_dir=self.thumbs_dir,
                )
                self.assertEqual(res_bkp_blk["blocked_safety"], 1)
                self.assertEqual(res_bkp_blk["processed_count"], 0)
            self.assertTrue(p_bkp_out.exists(), "2-copy online backup_green outside daf_backup/ must NOT be deleted!")

        # NEW-BUG-01 & NEW-BUG-02: Non-duplicate offline Original -> continuation button uses duplicates_only=false, 'plików', and valid deletion_token!
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_only_a):
            with self.TestingSessionLocal() as db:
                vb = db.get(WindowsVolume, vol_b_id)
                vb.is_online = False
                db.commit()
                res_off_nondup = StorageAutomationService(db).execute_safe_duplicate_removal(
                    [off_orig_id],
                    delete_mode="permanent",
                    force_duplicate_verification=False,
                    trash_dir=self.root / "kosz",
                    thumbnails_dir=self.thumbs_dir,
                )
                self.assertEqual(len(res_off_nondup["offline_ids"]), 1)
                j_cont = ProcessManager.create_job("delete_permanently", "ContNonDup", [off_orig_id], db=db)
                resp_cont = actions_router._build_safe_removal_html_response(
                    "/actions/delete-permanently",
                    "permanent",
                    res_off_nondup,
                    j_cont,
                    db,
                    force_duplicate_verification=False,
                )
                html_txt = resp_cont.body.decode("utf-8")
                self.assertIn("<strong>1</strong> plików", html_txt)
                self.assertNotIn("duplikatów", html_txt)
                m_vals = re.search(r'hx-vals="([^"]+)"', html_txt)
                self.assertIsNotNone(m_vals)
                cont_vals = json.loads(html_mod.unescape(m_vals.group(1)))
                self.assertEqual(cont_vals["duplicates_only"], "false")
                self.assertIn("deletion_token", cont_vals)

        # Reconnect Drive B and click the exact continuation button -> must delete p_off_orig cleanly (no infinite loop!)
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
            r_exec_cont = self.client.post("/actions/delete-permanently", data=cont_vals)
            self.assertEqual(r_exec_cont.status_code, 200)
            self.assertFalse(p_off_orig.exists(), "Continuation token for non-duplicate Original must delete once drive is online!")

        # BRAK-01: Unfiltered scope=filtered blocked on move-files, move-to-trash, gdrive-sync; unbacked_only respected!
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
            r_mv_blk = self.client.post("/actions/move-files", data={"scope": "filtered", "target_volume_id": str(vol_b_id)})
            self.assertIn("Zablokowano przenoszenie plików", r_mv_blk.text)

            r_tr_blk = self.client.post("/actions/move-to-trash", data={"scope": "filtered"})
            self.assertIn("Zablokowano przenoszenie do Kosza", r_tr_blk.text)

            r_gd_blk = self.client.post("/actions/gdrive-sync", data={"mode": "media", "scope": "filtered"})
            self.assertIn("Zablokowano synchronizację Google Drive", r_gd_blk.text)

            # Empty scope=selected MUST NOT fall back to entire database!
            r_empty_sel = self.client.post("/actions/move-to-trash", data={"scope": "selected"})
            self.assertIn("Brak wybranych elementów", r_empty_sel.text)

            # assign-group with scope=filtered + unbacked_only=true assigns ONLY solo_unbacked.jpg, NEVER pair_orig.jpg!
            r_ag = self.client.post(
                "/actions/assign-group",
                data={"group_name": "OnlyUnbackedAlb", "scope": "filtered", "unbacked_only": "true"},
            )
            self.assertIn("Przypisano 1 plików", r_ag.text)
            with self.TestingSessionLocal() as db:
                self.assertIsNotNone(db.get(MediaItem, unb_id).group_id)
                self.assertIsNone(db.get(MediaItem, orig_2c_id).group_id)

            # rotate-media with scope=filtered + unbacked_only=true rotates ONLY solo_unbacked.jpg!
            r_rot = self.client.post(
                "/actions/rotate-media",
                data={"scope": "filtered", "unbacked_only": "true", "delta": "90"},
            )
            self.assertEqual(r_rot.status_code, 200)
            with self.TestingSessionLocal() as db:
                self.assertEqual(db.get(MediaItem, unb_id).rotation_angle, 90)
                self.assertEqual(db.get(MediaItem, orig_2c_id).rotation_angle, 0)

            # move-files with scope=filtered + unbacked_only=true moves ONLY solo_unbacked.jpg to Drive B/MovedUnb!
            r_mv_unb = self.client.post(
                "/actions/move-files",
                data={
                    "scope": "filtered",
                    "unbacked_only": "true",
                    "target_volume_id": str(vol_b_id),
                    "target_subfolder": "MovedUnb",
                },
            )
            self.assertIn("Przeniesiono fizycznie: 1 plików", r_mv_unb.text)
            self.assertTrue(p_orig_2c.exists())
            self.assertFalse(p_unb_only.exists())
            self.assertTrue((self.drive_b / "MovedUnb" / "solo_unbacked.jpg").exists())

        # BRAK-02 (V-13): Global storage mutex blocks concurrent mutating endpoints
        ok_lk, _ = ProcessManager.try_acquire_storage_lock("Symulowana Operacja Testowa")
        self.assertTrue(ok_lk)
        try:
            r_lk_org = self.client.post("/actions/organize-originals", data={})
            self.assertIn("Operacja zablokowana", r_lk_org.text)
            self.assertIn("Symulowana Operacja Testowa", r_lk_org.text)

            r_lk_vac = self.client.post("/actions/vacuum-db", data={})
            self.assertIn("Operacja zablokowana", r_lk_vac.text)

            r_lk_emp = self.client.post("/actions/empty-trash", data={})
            self.assertIn("Operacja zablokowana", r_lk_emp.text)

            r_lk_scan = actions_router.execute_scan(str(self.drive_a))
            self.assertEqual(r_lk_scan["status"], "error")
            self.assertIn("Operacja zablokowana", r_lk_scan["message"])
        finally:
            ProcessManager.release_storage_lock()

        # Once released, organize-originals succeeds normally when a specific volume_id is supplied
        r_lk_free = self.client.post("/actions/organize-originals", data={"volume_id": str(vol_a_id)})
        self.assertEqual(r_lk_free.status_code, 200)
        self.assertNotIn("Operacja zablokowana", r_lk_free.text)

        # BRAK-04 (V-14): >999 IDs chunking across clear-quality-defect, restore-from-trash, rotate-media
        big_id_list = [unb_id] + [f"fake-id-v14-{i:04d}" for i in range(1050)]
        r_v14_qd = self.client.post(
            "/actions/clear-quality-defect",
            data={"scope": "selected", "selected_ids_json": json.dumps(big_id_list)},
        )
        self.assertEqual(r_v14_qd.status_code, 200)
        r_v14_rot = self.client.post(
            "/actions/rotate-media",
            data={"scope": "selected", "selected_ids_json": json.dumps(big_id_list), "delta": "90"},
        )
        self.assertEqual(r_v14_rot.status_code, 200)
        r_v14_res = self.client.post(
            "/actions/restore-from-trash",
            data={"restore_mode": "selected", "selected_ids": big_id_list},
        )
        self.assertEqual(r_v14_res.status_code, 200)

    def test_on_demand_onnx_flowers_manual_tags_album_search_and_offline_face_recognition(self):
        """
        Task 5 verification covering all 4 new 100%-local, thumbnail-only, On-Demand features:
        1) On-Demand ONNX keyword scan (`POST /actions/on-demand-keyword-scan`) finds untagged thumbnails,
           persists tags to SQLite, and releases the ONNX session (`tagger._session is None`).
        2) Two-level ImageNet dictionary + Flower HSV heuristic + manual tags (`POST /actions/add-tags`,
           `POST /actions/remove-tag`) preserved across AI re-runs (`POST /actions/analyze-all`).
        3) Search by album name/description including subalbums (`Wakacje Mazury / Lipiec` matched by `q=Mazury`).
        4) Same-person face recognition strictly on local thumbnails for OFFLINE drives
           (`/semantic/similar/{id}?mode=same_person`, `analysis_type=faces`, `POST /actions/name-person`).
        """
        import cv2
        import numpy as np
        from app.ai.faces import OnDemandFaceEmbedder
        from app.ai.tagger import OnDemandObjectTagger
        from app.models import MediaFace

        def _write_synthetic_flower_thumb(dest: Path) -> Path:
            # Green foliage background + vivid red/yellow petals in center
            arr = np.zeros((300, 300, 3), dtype=np.uint8)
            arr[:, :] = (35, 145, 30)  # BGR green foliage
            cv2.circle(arr, (150, 150), 68, (40, 25, 235), -1)  # vivid red blossom (~16% area)
            cv2.circle(arr, (150, 150), 22, (20, 220, 245), -1)  # yellow center
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            img.save(dest, "WEBP", quality=85)
            return dest

        def _write_synthetic_face_thumb(dest: Path, bg_bgr, skin_bgr=(138, 178, 226), eye_dx=22, mouth_w=24) -> Path:
            arr = np.zeros((300, 300, 3), dtype=np.uint8)
            arr[:, :] = bg_bgr
            # Oval skin face
            cv2.ellipse(arr, (150, 145), (56, 72), 0, 0, 360, skin_bgr, -1)
            # Eyebrows
            cv2.line(arr, (150 - eye_dx - 12, 112), (150 - eye_dx + 12, 110), (45, 45, 55), 4)
            cv2.line(arr, (150 + eye_dx - 12, 110), (150 + eye_dx + 12, 112), (45, 45, 55), 4)
            # Eyes (sclera + dark iris)
            cv2.ellipse(arr, (150 - eye_dx, 128), (11, 6), 0, 0, 360, (240, 240, 240), -1)
            cv2.ellipse(arr, (150 + eye_dx, 128), (11, 6), 0, 0, 360, (240, 240, 240), -1)
            cv2.circle(arr, (150 - eye_dx, 128), 5, (30, 35, 45), -1)
            cv2.circle(arr, (150 + eye_dx, 128), 5, (30, 35, 45), -1)
            # Nose bridge & shadow
            cv2.line(arr, (150, 130), (150, 160), (95, 130, 175), 3)
            cv2.ellipse(arr, (150, 162), (9, 5), 0, 0, 180, (90, 120, 165), 2)
            # Mouth
            cv2.ellipse(arr, (150, 182), (mouth_w, 8), 0, 0, 180, (70, 75, 170), 4)
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            img.save(dest, "WEBP", quality=90)
            return dest

        def _write_plain_blue_thumb(dest: Path) -> Path:
            arr = np.zeros((300, 300, 3), dtype=np.uint8)
            arr[:, :] = (210, 140, 40)  # BGR sky blue
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            img.save(dest, "WEBP", quality=85)
            return dest

        now_dt = datetime.datetime(2025, 7, 15, 12, 0, 0)
        with self.TestingSessionLocal() as db:
            # OFFLINE drive — physical files NEVER exist on Z:\, ONLY local thumbnails exist in self.thumbs_dir!
            vol_off = WindowsVolume(
                volume_serial_number="OFF9-9999",
                volume_label="OfflineArchiwum",
                last_known_drive_letter="Z:\\",
                is_online=False,
            )
            db.add(vol_off)
            db.commit()
            db.refresh(vol_off)

            # Task 3: Album hierarchy `Wakacje Mazury` -> subalbum `Lipiec`
            grp_parent = MediaGroup(name="Wakacje Mazury", description="Wyjazd nad jeziora")
            db.add(grp_parent)
            db.commit()
            db.refresh(grp_parent)
            grp_child = MediaGroup(name="Lipiec", parent_id=grp_parent.id)
            db.add(grp_child)
            db.commit()
            db.refresh(grp_child)

            def _add_offline_item(fname: str, thumb_fn, group_id=None):
                mid = str(uuid.uuid4())
                t_rel = f"{mid}.webp"
                thumb_fn(self.thumbs_dir / t_rel)
                it = MediaItem(
                    id=mid,
                    volume_id=vol_off.id,
                    relative_path=f"OfflineFolder/{fname}",
                    file_name=fname,
                    file_size_bytes=204800,
                    media_type="image",
                    mime_type="image/jpeg",
                    file_hash=f"hash_{mid.replace('-', '')}",
                    hash_algorithm="sha256",
                    fs_ctime=now_dt,
                    fs_mtime=now_dt,
                    effective_date=now_dt,
                    thumbnail_relative_path=t_rel,
                    group_id=group_id,
                    duplicate_role="original",
                )
                db.add(it)
                db.commit()
                return mid

            flower_id = _add_offline_item("ogrod_roza.jpg", _write_synthetic_flower_thumb)
            mazury_id = _add_offline_item("jezioro_01.jpg", _write_plain_blue_thumb, group_id=grp_child.id)
            # Same person (Ania) on two completely different backgrounds!
            ania1_id = _add_offline_item("portret_ania_plaza.jpg", lambda p: _write_synthetic_face_thumb(p, bg_bgr=(220, 160, 60)))
            ania2_id = _add_offline_item("portret_ania_las.jpg", lambda p: _write_synthetic_face_thumb(p, bg_bgr=(35, 110, 40)))

        # ---------------------------------------------------------------------
        # 1) Task 3: Album name search finds photos in subalbum (`Wakacje Mazury / Lipiec` via `q=Mazury`)
        # ---------------------------------------------------------------------
        r_alb = self.client.get("/media/search", params={"q": "Mazury"})
        self.assertEqual(r_alb.status_code, 200)
        self.assertIn("jezioro_01.jpg", r_alb.text)
        self.assertNotIn("ogrod_roza.jpg", r_alb.text)
        # Also verify sidebar renders album search input
        r_sb = self.client.get("/media/sidebar")
        self.assertIn("Szukaj albumu...", r_sb.text)

        # ---------------------------------------------------------------------
        # 2) Task 1: On-Demand Keyword Scan on untagged local thumbnails + session release check
        # ---------------------------------------------------------------------
        # Before scan, searching "kwiaty" finds 0 items and offers the on-demand AI button
        r_kw_before = self.client.get("/media/search", params={"q": "kwiaty"})
        self.assertNotIn("ogrod_roza.jpg", r_kw_before.text)
        self.assertIn("/actions/on-demand-keyword-scan", r_kw_before.text)

        # Verify direct scan_thumbnails_for_keyword releases ONNX session (`tagger._session is None`)
        tagger_check = OnDemandObjectTagger()
        with self.TestingSessionLocal() as db:
            scan_res = tagger_check.scan_thumbnails_for_keyword(db, "kwiaty", thumbnails_dir=self.thumbs_dir)
        self.assertIsNone(tagger_check._session, "ONNX session MUST be released immediately after scan!")
        self.assertTrue(scan_res["session_released"])
        self.assertIn(flower_id, scan_res["matched_ids"])

        # Also test HTTP endpoint POST /actions/on-demand-keyword-scan
        r_od = self.client.post("/actions/on-demand-keyword-scan", data={"q": "kwiaty"})
        self.assertEqual(r_od.status_code, 200)
        self.assertIn("Przeszukano miniatury AI", r_od.text)

        # Now searching "kwiaty" finds ogrod_roza.jpg instantly from SQLite!
        r_kw_after = self.client.get("/media/search", params={"q": "kwiaty"})
        self.assertIn("ogrod_roza.jpg", r_kw_after.text)

        # ---------------------------------------------------------------------
        # 3) Task 2: Two-level ImageNet tagging (category + detailed tag) + manual tags preserved across AI re-run + remove-tag
        # ---------------------------------------------------------------------
        # Verify two-level tagging produces BOTH category (`kwiaty / rosliny`) AND detailed tag (`stokrotka (daisy)`)
        daisy_probs = np.zeros(1000, dtype=np.float32)
        daisy_probs[985] = 0.85  # ImageNet 985 = daisy
        tagger_2lvl = OnDemandObjectTagger()
        tagger_2lvl.labels = {"985": "daisy"}
        tagger_2lvl._session = MagicMock()
        with patch.object(tagger_2lvl, "_run_onnx_probs", return_value=daisy_probs):
            tags_with_src = tagger_2lvl.tag_image_with_sources(self.thumbs_dir / f"{flower_id}.webp")
        tag_src_map = {t: src for t, _c, src in tags_with_src}
        self.assertIn("kwiaty / rosliny", tag_src_map)
        self.assertEqual(tag_src_map["kwiaty / rosliny"], "opencv+onnx")
        self.assertIn("stokrotka (daisy)", tag_src_map)
        self.assertEqual(tag_src_map["stokrotka (daisy)"], "onnx_detail")
        tagger_2lvl.close_session()
        self.assertIsNone(tagger_2lvl._session)

        # Add manual tags (`róże babci, komunia`) via POST /actions/add-tags
        r_add_m = self.client.post(
            "/actions/add-tags",
            data={"media_id": flower_id, "tags": "róże babci, komunia"},
        )
        self.assertEqual(r_add_m.status_code, 200)
        with self.TestingSessionLocal() as db:
            m_tags = list(
                db.scalars(
                    select(MediaTag).where(MediaTag.media_id == flower_id, MediaTag.source_model == "manual")
                ).all()
            )
            self.assertEqual({t.tag for t in m_tags}, {"róże babci", "komunia"})

        # Searchable immediately by manual tag
        r_find_manual = self.client.get("/media/search", params={"q": "komunia"})
        self.assertIn("ogrod_roza.jpg", r_find_manual.text)

        # Re-running AI tagging (`POST /actions/analyze-all` with `analysis_type=tags`) MUST PRESERVE manual tags!
        r_reai = self.client.post(
            "/actions/analyze-all",
            data={
                "analysis_type": "tags",
                "scope": "selected",
                "selected_ids": [flower_id],
                "run_sync": "true",
            },
        )
        self.assertEqual(r_reai.status_code, 200)
        with self.TestingSessionLocal() as db:
            m_tags_after = list(
                db.scalars(
                    select(MediaTag).where(MediaTag.media_id == flower_id, MediaTag.source_model == "manual")
                ).all()
            )
            self.assertEqual(
                {t.tag for t in m_tags_after},
                {"róże babci", "komunia"},
                "Manual tags MUST NOT be deleted when AI tagging is re-run!",
            )
            komunia_row = next(t for t in m_tags_after if t.tag == "komunia")
            komunia_tid = komunia_row.id

        # Removing a single tag via POST /actions/remove-tag removes ONLY that tag
        r_rm_tag = self.client.post(
            "/actions/remove-tag",
            data={"media_id": flower_id, "tag_id": str(komunia_tid)},
        )
        self.assertEqual(r_rm_tag.status_code, 200)
        with self.TestingSessionLocal() as db:
            rem_manual = {
                t.tag
                for t in db.scalars(
                    select(MediaTag).where(MediaTag.media_id == flower_id, MediaTag.source_model == "manual")
                ).all()
            }
            self.assertEqual(rem_manual, {"róże babci"})

        # ---------------------------------------------------------------------
        # 4) Task 4: Offline Same-Person Face Indexing, Similarity Search & Person Naming
        # ---------------------------------------------------------------------
        # Verify Item #1 fix: when YuNet IS active and returns 0 faces on a tan/brown landscape/drone thumbnail,
        # Tier-3 skin-blob fallback MUST NOT fire (eliminates false positives on drone/landscape photos!)
        fe_yn_guard = OnDemandFaceEmbedder()
        fake_yn_zero = MagicMock()
        fake_yn_zero.detect.return_value = (1, None)
        fe_yn_guard._yunet = fake_yn_zero
        tan_drone_arr = cv2.imread(str(self.thumbs_dir / f"{ania1_id}.webp"))
        self.assertEqual(fe_yn_guard.detect_faces(tan_drone_arr), [])
        fe_yn_guard.close_session()

        # Run face indexing via Analyze AI (`analysis_type=faces`) on OFFLINE drive synthetic cartoon thumbnails
        # (using Tier-3 fallback mode with YuNet disabled for synthetic OpenCV cartoon drawings)
        orig_open_sess = OnDemandFaceEmbedder.open_session
        def _open_without_yunet(self_fe):
            self_fe.yunet_path = self.root / "no_yunet_for_synthetic.onnx"
            orig_open_sess(self_fe)

        with patch.object(OnDemandFaceEmbedder, "open_session", _open_without_yunet):
            r_face_idx = self.client.post(
                "/actions/analyze-all",
                data={
                    "analysis_type": "faces",
                    "target_drive_scope": "all_drives",
                    "run_sync": "true",
                },
            )
        self.assertEqual(r_face_idx.status_code, 200)
        self.assertIn("Zaindeksowano twarze na miniaturach", r_face_idx.text)

        with self.TestingSessionLocal() as db:
            faces_a1 = list(db.scalars(select(MediaFace).where(MediaFace.media_id == ania1_id)).all())
            faces_a2 = list(db.scalars(select(MediaFace).where(MediaFace.media_id == ania2_id)).all())
            self.assertGreaterEqual(len(faces_a1), 1, "Face must be detected on offline thumbnail ania1!")
            self.assertGreaterEqual(len(faces_a2), 1, "Face must be detected on offline thumbnail ania2!")
            self.assertEqual(len(faces_a1[0].vector), OnDemandFaceEmbedder.VECTOR_DIM * 4)

        # Find same person across completely different backgrounds strictly from offline thumbnails!
        r_same = self.client.get(f"/semantic/similar/{ania1_id}", params={"mode": "same_person"})
        self.assertEqual(r_same.status_code, 200)
        self.assertIn("portret_ania_plaza.jpg", r_same.text)
        self.assertIn("portret_ania_las.jpg", r_same.text)
        self.assertNotIn("jezioro_01.jpg", r_same.text)
        self.assertIn("Zmień osoby", r_same.text)

        # Name this person ("Ania") via POST /actions/name-person -> searchable immediately via q=Ania!
        r_name = self.client.post(
            "/actions/name-person",
            data={"person_name": "Ania", "selected_ids": [ania1_id, ania2_id]},
        )
        self.assertEqual(r_name.status_code, 200)
        self.assertIn("Ania", r_name.text)

        r_find_ania = self.client.get("/media/search", params={"q": "Ania"})
        self.assertEqual(r_find_ania.status_code, 200)
        self.assertIn("portret_ania_plaza.jpg", r_find_ania.text)
        self.assertIn("portret_ania_las.jpg", r_find_ania.text)
        self.assertNotIn("jezioro_01.jpg", r_find_ania.text)

        # ---------------------------------------------------------------------
        # 5) Code Review Fixes Verification ([FACES-01], [FACES-02], [FACES-03], [CACHE-01])
        # ---------------------------------------------------------------------
        # [FACES-01]: self._sface.feature() MUST be used in extract_face_vector() when loaded
        fe_sf = OnDemandFaceEmbedder()
        fake_sface = MagicMock()
        fake_sface.feature.return_value = np.linspace(0.1, 1.0, 128, dtype=np.float32).reshape(1, 128)
        fe_sf._sface = fake_sface
        fe_sf._session = None
        dummy_bgr = cv2.imread(str(self.thumbs_dir / f"{ania1_id}.webp"))
        sf_vec = fe_sf.extract_face_vector(dummy_bgr, (80, 60, 140, 160))
        fake_sface.feature.assert_called_once()
        self.assertEqual(sf_vec.shape, (OnDemandFaceEmbedder.VECTOR_DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(sf_vec)), 1.0, places=4)
        fe_sf.close_session()
        self.assertIsNone(fe_sf._sface)

        # [FACES-02] & [FACES-03]: Per-face `face_index` filtering on group photos + preserving names for face_index > 0
        with self.TestingSessionLocal() as db:
            grp_photo_id = _add_offline_item("grupa_ania_marek.jpg", _write_plain_blue_thumb)
            person_a_id = _add_offline_item("solo_osoba_a.jpg", _write_plain_blue_thumb)
            person_b_id = _add_offline_item("solo_osoba_b.jpg", _write_plain_blue_thumb)

            vec_a = np.zeros(OnDemandFaceEmbedder.VECTOR_DIM, dtype=np.float32)
            vec_a[:64] = 1.0 / np.sqrt(64.0)
            vec_b = np.zeros(OnDemandFaceEmbedder.VECTOR_DIM, dtype=np.float32)
            vec_b[128:192] = 1.0 / np.sqrt(64.0)

            db.add_all(
                [
                    MediaFace(
                        media_id=grp_photo_id,
                        face_index=0,
                        bbox_json="[0.1, 0.2, 0.3, 0.4]",
                        confidence=0.95,
                        person_name="OsobaA",
                        vector=vec_a.tobytes(),
                    ),
                    MediaFace(
                        media_id=grp_photo_id,
                        face_index=1,
                        bbox_json="[0.5, 0.2, 0.3, 0.4]",
                        confidence=0.92,
                        person_name="OsobaB",
                        vector=vec_b.tobytes(),
                    ),
                    MediaFace(
                        media_id=person_a_id,
                        face_index=0,
                        bbox_json="[0.2, 0.2, 0.4, 0.4]",
                        confidence=0.96,
                        vector=vec_a.tobytes(),
                    ),
                    MediaFace(
                        media_id=person_b_id,
                        face_index=0,
                        bbox_json="[0.2, 0.2, 0.4, 0.4]",
                        confidence=0.96,
                        vector=vec_b.tobytes(),
                    ),
                ]
            )
            db.commit()

        # Detail modal renders clickable per-face links with distinct face_index=0 and face_index=1
        r_det_grp = self.client.get(f"/media/{grp_photo_id}/detail")
        self.assertEqual(r_det_grp.status_code, 200)
        self.assertIn(f"/semantic/similar/{grp_photo_id}?mode=same_person&amp;face_index=0", r_det_grp.text)
        self.assertIn(f"/semantic/similar/{grp_photo_id}?mode=same_person&amp;face_index=1", r_det_grp.text)

        # Clicking face_index=0 matches ONLY Person A; clicking face_index=1 matches ONLY Person B
        r_f0 = self.client.get(f"/semantic/similar/{grp_photo_id}", params={"mode": "same_person", "face_index": 0})
        self.assertEqual(r_f0.status_code, 200)
        self.assertIn("solo_osoba_a.jpg", r_f0.text)
        self.assertNotIn("solo_osoba_b.jpg", r_f0.text)

        r_f1 = self.client.get(f"/semantic/similar/{grp_photo_id}", params={"mode": "same_person", "face_index": 1})
        self.assertEqual(r_f1.status_code, 200)
        self.assertIn("solo_osoba_b.jpg", r_f1.text)
        self.assertNotIn("solo_osoba_a.jpg", r_f1.text)

        # Naming face_index=1 on group photo updates ONLY face_index=1 (preserves face_index=0 name!)
        r_name_f1 = self.client.post(
            "/actions/name-person",
            data={
                "person_name": "Marek",
                "target_media_id": grp_photo_id,
                "face_index": "1",
                "selected_ids": [grp_photo_id, person_b_id],
            },
        )
        self.assertEqual(r_name_f1.status_code, 200)
        with self.TestingSessionLocal() as db:
            grp_faces_named = {
                f.face_index: f.person_name
                for f in db.scalars(select(MediaFace).where(MediaFace.media_id == grp_photo_id)).all()
            }
            self.assertEqual(grp_faces_named, {0: "OsobaA", 1: "Marek"})

        # [FACES-03]: Re-indexing via index_faces_for_items preserves person_name for BOTH face_index=0 AND face_index=1!
        fe_reindex = OnDemandFaceEmbedder()
        fake_extracted_2 = [
            {"face_index": 0, "bbox_norm": [0.1, 0.2, 0.3, 0.4], "confidence": 0.94, "vector": vec_a, "vector_bytes": vec_a.tobytes()},
            {"face_index": 1, "bbox_norm": [0.5, 0.2, 0.3, 0.4], "confidence": 0.91, "vector": vec_b, "vector_bytes": vec_b.tobytes()},
        ]
        with self.TestingSessionLocal() as db:
            with patch.object(fe_reindex, "extract_faces_from_thumbnail", return_value=fake_extracted_2):
                fe_reindex.index_faces_for_items(db, [grp_photo_id], thumbnails_dir=self.thumbs_dir)
            grp_faces_after = {
                f.face_index: f.person_name
                for f in db.scalars(select(MediaFace).where(MediaFace.media_id == grp_photo_id)).all()
            }
            self.assertEqual(grp_faces_after, {0: "OsobaA", 1: "Marek"})

        # [CACHE-01]: Deleting an older (non-max ID) tag MUST immediately invalidate _SIDEBAR_CACHE
        r_add_cache = self.client.post(
            "/actions/add-tags",
            data={"media_id": flower_id, "tags": "stary_tag_cache, nowszy_tag_cache"},
        )
        self.assertEqual(r_add_cache.status_code, 200)
        r_sb_before_rm = self.client.get("/media/sidebar")
        self.assertIn("stary_tag_cache", r_sb_before_rm.text)
        self.assertIn("nowszy_tag_cache", r_sb_before_rm.text)

        with self.TestingSessionLocal() as db:
            stary_row = db.scalar(
                select(MediaTag).where(MediaTag.media_id == flower_id, MediaTag.tag == "stary_tag_cache")
            )
            nowszy_row = db.scalar(
                select(MediaTag).where(MediaTag.media_id == flower_id, MediaTag.tag == "nowszy_tag_cache")
            )
            self.assertIsNotNone(stary_row)
            self.assertIsNotNone(nowszy_row)
            self.assertLess(stary_row.id, nowszy_row.id)
            stary_tid = stary_row.id

        r_rm_older = self.client.post(
            "/actions/remove-tag",
            data={"media_id": flower_id, "tag_id": str(stary_tid)},
        )
        self.assertEqual(r_rm_older.status_code, 200)
        # max(MediaTag.id) did NOT change, yet sidebar MUST refresh immediately via count(MediaTag.id)!
        r_sb_after_rm = self.client.get("/media/sidebar")
        self.assertNotIn("stary_tag_cache", r_sb_after_rm.text)
        self.assertIn("nowszy_tag_cache", r_sb_after_rm.text)

        # ---------------------------------------------------------------------
        # 6) UX & Precision Polish Verification (Requests 1-6)
        # ---------------------------------------------------------------------
        # Req 1: No "Obróć album +90°" button in active album bar
        r_alb_view = self.client.get("/media/search", params={"group_id": str(grp_parent.id)})
        self.assertEqual(r_alb_view.status_code, 200)
        self.assertNotIn("Obróć album +90°", r_alb_view.text)
        # Req 5: Inline "Zmień osoby" available directly in media_grid
        self.assertIn("Zmień osoby (zaznaczone):", r_alb_view.text)

        # Req 2 & Req 4: Sidebar album tree collapsed under root + Osoby i Twarze section (Ania, Marek, Twarz #1, Twarz #2)
        self.assertIn('data-album-children class="hidden"', r_sb_after_rm.text)
        self.assertIn("toggleAlbumBranch", r_sb_after_rm.text)
        self.assertIn("Osoby i Twarze", r_sb_after_rm.text)
        self.assertIn("Ania", r_sb_after_rm.text)
        self.assertIn("Marek", r_sb_after_rm.text)
        self.assertIn("Twarz #1", r_sb_after_rm.text)
        self.assertIn("Twarz #2", r_sb_after_rm.text)

        # Req 4: Main page renders collapsible top panel & independent scrolling grid container
        r_home_ui = self.client.get("/")
        self.assertEqual(r_home_ui.status_code, 200)
        self.assertIn("top-controls-panel", r_home_ui.text)
        self.assertIn("toggleTopControlsPanel", r_home_ui.text)
        self.assertIn("media-grid-scroll", r_home_ui.text)

        # Filtering by sidebar face/person tags works cleanly
        r_tw2 = self.client.get("/media/search", params={"tag": "twarz:2"})
        self.assertIn("grupa_ania_marek.jpg", r_tw2.text)
        self.assertNotIn("ogrod_roza.jpg", r_tw2.text)
        r_os_ania = self.client.get("/media/search", params={"tag": "osoba:Ania"})
        self.assertIn("portret_ania_plaza.jpg", r_os_ania.text)
        self.assertNotIn("ogrod_roza.jpg", r_os_ania.text)

        # Req 2 & Req 3: Wider + resizable detail modal, full non-200px path, native hx-get same_person
        self.assertIn("resize: horizontal", r_det_grp.text)
        self.assertIn("detail-modal-box", r_det_grp.text)
        self.assertNotIn("max-w-[200px]", r_det_grp.text)
        self.assertIn(f'hx-get="/semantic/similar/{grp_photo_id}?mode=same_person"', r_det_grp.text)
        self.assertIn("Zmień osoby:", r_det_grp.text)

        # Req 5: Comma-separated multi-person "Zmień osoby" ("Ania, Ala") assigns photo to BOTH separate people!
        r_multi_p = self.client.post(
            "/actions/name-person",
            data={"person_name": "Ania, Ala", "person_action": "replace", "selected_ids": [ania1_id]},
        )
        self.assertEqual(r_multi_p.status_code, 200)
        r_sb_multi = self.client.get("/media/sidebar")
        self.assertIn("Ania", r_sb_multi.text)
        self.assertIn("Ala", r_sb_multi.text)
        self.assertIn("portret_ania_plaza.jpg", self.client.get("/media/search", params={"tag": "osoba:Ania"}).text)
        self.assertIn("portret_ania_plaza.jpg", self.client.get("/media/search", params={"tag": "osoba:Ala"}).text)

        # Replacing with just "Ania" removes "Ala" cleanly; clearing removes all persons
        r_rep_one = self.client.post(
            "/actions/name-person",
            data={"person_name": "Ania", "person_action": "replace", "selected_ids": [ania1_id]},
        )
        self.assertEqual(r_rep_one.status_code, 200)
        self.assertNotIn("portret_ania_plaza.jpg", self.client.get("/media/search", params={"tag": "osoba:Ala"}).text)
        self.assertIn("portret_ania_plaza.jpg", self.client.get("/media/search", params={"tag": "osoba:Ania"}).text)

        # Removing osoba:Ania via /actions/remove-tag also clears MediaFace.person_name
        r_rm_p_tag = self.client.post(
            "/actions/remove-tag",
            data={"media_id": ania1_id, "tag": "osoba:Ania"},
        )
        self.assertEqual(r_rm_p_tag.status_code, 200)
        self.assertNotIn("portret_ania_plaza.jpg", self.client.get("/media/search", params={"tag": "osoba:Ania"}).text)

        # person_action=clear removes all person assignments from ania2_id
        r_clr_p = self.client.post(
            "/actions/name-person",
            data={"person_action": "clear", "selected_ids": [ania2_id]},
        )
        self.assertEqual(r_clr_p.status_code, 200)
        self.assertNotIn("portret_ania_las.jpg", self.client.get("/media/search", params={"tag": "osoba:Ania"}).text)

        # Req 4: Similarity results (even non-same_person mode) allow changing persons / adding tags & keep_view=1 refreshes sidebar only
        r_sim_hyb = self.client.get(f"/semantic/similar/{ania1_id}", params={"mode": "hybrid"})
        self.assertEqual(r_sim_hyb.status_code, 200)
        self.assertIn("Zmień osoby", r_sim_hyb.text)
        self.assertIn("Dodaj tag do zaznaczonych", r_sim_hyb.text)
        r_keep_tag = self.client.post(
            "/actions/add-tags",
            data={"scope": "selected", "selected_ids": [ania1_id], "tags": "sesja_plaza", "keep_view": "1"},
        )
        self.assertEqual(r_keep_tag.status_code, 200)
        self.assertEqual(r_keep_tag.headers.get("HX-Trigger"), "refresh-sidebar")

        # Req 5: Strict word-boundary & flower precision (tulipan MUST NOT match cauliflower/potter's wheel/bear/ship/earthstar!)
        tg_prec = OnDemandObjectTagger()
        tg_prec.labels = {
            "294": "brown bear, bruin",
            "724": "pirate, pirate ship",
            "738": "pot, flowerpot",
            "739": "potter's wheel",
            "849": "teapot",
            "938": "cauliflower",
            "985": "daisy",
            "995": "earthstar",
        }
        t_idxs, t_cats, _ = tg_prec.resolve_keyword_to_imagenet_indices("tulipan")
        self.assertIn(985, t_idxs)
        self.assertNotIn(938, t_idxs, "cauliflower must NEVER match tulipan/flower!")
        self.assertNotIn(739, t_idxs, "potter's wheel must NEVER match tulipan/pot!")
        self.assertNotIn(849, t_idxs, "teapot must NEVER match tulipan/pot!")
        self.assertNotIn(294, t_idxs, "brown bear must NEVER match tulipan/ear!")
        self.assertNotIn(724, t_idxs, "pirate ship must NEVER match tulipan/hip!")
        self.assertNotIn(995, t_idxs, "earthstar mushroom must NEVER match tulipan!")

        # Search q=tulipan MUST NOT return items tagged with bear / cauliflower / potter's wheel
        with self.TestingSessionLocal() as db:
            noise_id = _add_offline_item("niedzwiedz_kalafior.jpg", _write_plain_blue_thumb)
            db.add_all([
                MediaTag(media_id=noise_id, tag="brown bear", confidence=0.9, source_model="onnx_detail"),
                MediaTag(media_id=noise_id, tag="cauliflower", confidence=0.9, source_model="onnx_detail"),
                MediaTag(media_id=noise_id, tag="potter's wheel", confidence=0.9, source_model="onnx_detail"),
            ])
            db.commit()
        r_tulip_q = self.client.get("/media/search", params={"q": "tulipan"})
        self.assertEqual(r_tulip_q.status_code, 200)
        self.assertNotIn("niedzwiedz_kalafior.jpg", r_tulip_q.text)
        self.assertIn("ogrod_roza.jpg", r_tulip_q.text)


if __name__ == "__main__":
    unittest.main()
