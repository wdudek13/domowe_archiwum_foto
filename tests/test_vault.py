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
from sqlalchemy import create_engine, select
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
        self.assertEqual(effective_dt, fn_dt)
        self.assertTrue(anomaly)
        self.assertIn("Niezgodność dat", reason)

        # Missing EXIF alone WITHOUT source discrepancy must NOT trigger an anomaly
        plain_img = create_synthetic_jpeg(self.drive_a / "random_shot.jpg")
        _, _, _, _, _, plain_anom, plain_reason = MetadataDateResolver.resolve_dates(plain_img, is_image=True)
        self.assertFalse(plain_anom)
        self.assertIsNone(plain_reason)

        # Folder date discrepancy after file copy: folder 2019-08 vs filesystem timestamps in 2026
        folder_img = create_synthetic_jpeg(self.drive_a / "2019-08_Grecja" / "beach.jpg")
        _, _, _, _, f_eff, f_anom, f_rsn = MetadataDateResolver.resolve_dates(
            folder_img, is_image=True, rel_path="2019-08_Grecja/beach.jpg"
        )
        self.assertTrue(f_anom)
        self.assertEqual((f_eff.year, f_eff.month), (2019, 8))
        self.assertIn("folderem", f_rsn)

        # Rule (b): Day-level folder may differ in DAY as long as YEAR and MONTH match -> NO anomaly
        same_ym_eff, same_ym_anom, same_ym_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=None,
            ctime=datetime.datetime(2022, 7, 28, 14, 0),
            mtime=datetime.datetime(2022, 7, 28, 14, 0),
            folder_info=(datetime.datetime(2022, 7, 15, 0, 0), "day", "2022-07-15"),
        )
        self.assertFalse(same_ym_anom)
        self.assertIsNone(same_ym_rsn)
        self.assertEqual(same_ym_eff, datetime.datetime(2022, 7, 28, 14, 0))

        # Rule (c): Copy-modified ctime with older mtime matching folder -> picks mtime and flags anomaly
        day_eff, day_anom, _ = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=None,
            ctime=datetime.datetime(2026, 5, 1, 10, 0),
            mtime=datetime.datetime(2022, 7, 15, 14, 0),
            folder_info=(datetime.datetime(2022, 7, 15, 0, 0), "day", "2022-07-15"),
        )
        self.assertTrue(day_anom)
        self.assertEqual(day_eff, datetime.datetime(2022, 7, 15, 14, 0))

        # Copy discrepancy without folder/EXIF/filename when ctime is >24h newer than mtime
        cp_eff, cp_anom, cp_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=None,
            ctime=datetime.datetime(2026, 6, 1, 10, 0),
            mtime=datetime.datetime(2021, 3, 20, 9, 0),
            folder_info=None,
        )
        self.assertTrue(cp_anom)
        self.assertEqual(cp_eff, datetime.datetime(2021, 3, 20, 9, 0))
        self.assertIn("kopiowaniu", cp_rsn)

        # Rule (a): Whenever EXIF exists, return EXIF with NO anomaly even if other sources differ
        ex_only_eff, ex_only_anom, ex_only_rsn = MetadataDateResolver.evaluate_date_sources(
            exif=datetime.datetime(2023, 5, 10, 12, 0),
            fn_date=datetime.datetime(2021, 5, 10, 12, 0),
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2026, 1, 1, 10, 0),
            folder_info=(datetime.datetime(2021, 5, 1, 0, 0), "month", "2021-05"),
        )
        self.assertFalse(ex_only_anom)
        self.assertIsNone(ex_only_rsn)
        self.assertEqual(ex_only_eff, datetime.datetime(2023, 5, 10, 12, 0))

        # Rule (c) with no EXIF: filename older than folder -> picks oldest (2019-05-10)
        eff_fn_fold, anom_fn_fold, _ = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=datetime.datetime(2019, 5, 10, 12, 0),
            ctime=datetime.datetime(2026, 1, 1, 10, 0),
            mtime=datetime.datetime(2026, 1, 1, 10, 0),
            folder_info=(datetime.datetime(2021, 5, 1, 0, 0), "month", "2021-05"),
        )
        self.assertTrue(anom_fn_fold)
        self.assertEqual(eff_fn_fold.year, 2019)

        # Rule (b) all agree -> returns ctime with NO anomaly
        eff_agree, anom_agree, _ = MetadataDateResolver.evaluate_date_sources(
            exif=None,
            fn_date=datetime.datetime(2022, 4, 5, 11, 0),
            ctime=datetime.datetime(2022, 4, 5, 11, 5),
            mtime=datetime.datetime(2022, 4, 5, 11, 5),
            folder_info=MetadataDateResolver.parse_folder_date(self.drive_a / "2022-04-05" / "a.jpg", rel_path="abcdef0123456789abcdef/2022-04-05/a.jpg"),
        )
        self.assertFalse(anom_agree)
        self.assertEqual(eff_agree, datetime.datetime(2022, 4, 5, 11, 5))

        # Rule (c) folder month older than filename -> picks folder (2020-06) and flags discrepancy
        self.assertEqual(
            MetadataDateResolver.evaluate_date_sources(
                exif=None,
                fn_date=datetime.datetime(2026, 1, 2, 10, 0),
                ctime=datetime.datetime(2025, 6, 5, 10, 0),
                mtime=datetime.datetime(2025, 6, 5, 10, 0),
                folder_info=(datetime.datetime(2020, 6, 1, 0, 0), "month", "2020-06"),
            )[0].year,
            2020,
        )

        # Test EXIF parsing on JpegImageFile (rule a: EXIF always takes priority without anomaly)
        with patch.object(JpegImagePlugin.JpegImageFile, "_getexif", return_value={36867: "2023:05:10 12:30:45"}):
            ex_dt, _, _, _, eff_dt, anom, rsn = MetadataDateResolver.resolve_dates(img_file, is_image=True)
            self.assertEqual(ex_dt, datetime.datetime(2023, 5, 10, 12, 30, 45))
            self.assertEqual(eff_dt, ex_dt)
            self.assertFalse(anom)
            self.assertIsNone(rsn)

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

            # Create File 1: 1 copy on Drive A, 2 copies on Drive B
            p1_a = create_synthetic_jpeg(self.drive_a / "IMG_20230615_100000.jpg", color=(10, 80, 200))
            p1_b_orig = self.drive_b / "01_Wakacje" / "IMG_20230615_100000.jpg"
            p1_b_orig.parent.mkdir(parents=True, exist_ok=True)
            p1_b_orig.write_bytes(p1_a.read_bytes())
            p1_b_dup = self.drive_b / "99_Kopia" / "IMG_20230615_100000_dup.jpg"
            p1_b_dup.parent.mkdir(parents=True, exist_ok=True)
            p1_b_dup.write_bytes(p1_a.read_bytes())

            # Create File 2: only 1 copy on Drive A (unbacked!)
            p2_a = create_synthetic_jpeg(self.drive_a / "IMG_20230720_150000.jpg", color=(200, 50, 30))

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
            r_del_dups = self.client.post(
                "/actions/delete-permanently",
                data={
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

            # Permanent delete (by group_id and selected_ids) & unindex (by selected_ids and volume_id)
            self.client.post("/actions/delete-permanently", data={"delete_group_id": str(grp_id)})
            self.client.post("/actions/delete-permanently", data={"scope": "selected", "selected_ids": [it2_id]})
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

        with patch.object(WindowsVolumeInspector, "get_volume_info", side_effect=fake_vol_info), \
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
            r_del_red = self.client.post(
                "/actions/delete-permanently",
                data={
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

            # Check pair metrics (Kopie lokalne / Kopie zdalne / Nowe kopie / Do albumu Ważne)
            p_ac_key_str = f"{min(va_id, vc_id)}:{max(va_id, vc_id)}"
            p_ab = next(p for p in plan["ordered_pairs"] if p["pair_key"] == (min(va_id, vb_id), max(va_id, vb_id)))
            self.assertGreaterEqual(p_ab["metrics"]["important_to_assign_count"], 1)

            # Step 2: Build plan restricted to Pair (va, vc) targeting vc_id
            plan_ac = svc.build_backup_and_dedup_plan(
                selected_pair_keys=[p_ac_key_str],
                target_volume_ids=[vc_id],
            )
            p_ac_entry = next(p for p in plan_ac["ordered_pairs"] if p["pair_key_str"] == p_ac_key_str)
            self.assertGreaterEqual(p_ac_entry["metrics"]["important_to_assign_count"], 1)
            copy_tasks_ac = p_ac_entry["tasks"]["backups_to_copy"]
            self.assertTrue(any(t["orig_item_id"] == item_a.id and t["dst_vol_id"] == vc_id for t in copy_tasks_ac))

        # Step 3: Test GET /actions/backup-panel with selected_pairs
        r_all = self.client.get("/actions/backup-panel")
        self.assertEqual(r_all.status_code, 200)
        self.assertIn("backup-pair-counts-data", r_all.text)
        self.assertIn("Lista oryginałów dodawanych do albumu „Ważne…”", r_all.text)

        r_filtered = self.client.get(f"/actions/backup-panel?target_volume_id={vc_id}&selected_pairs={p_ac_key_str}")
        self.assertEqual(r_filtered.status_code, 200)
        self.assertIn("unbacked_on_a.jpg", r_filtered.text)
        self.assertIn("same_dup1.jpg", r_filtered.text)

        # Step 3b: Test POST /actions/backup-panel-tables with selected pairs and empty pairs
        r_tables = self.client.post(
            "/actions/backup-panel-tables",
            data={"selected_pairs": [p_ac_key_str], "pairs_submitted": "true", "preferred_target_volume_id": str(vc_id)},
        )
        self.assertEqual(r_tables.status_code, 200)
        self.assertIn("unbacked_on_a.jpg", r_tables.text)
        self.assertIn("same_dup1.jpg", r_tables.text)
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

            # Test Safe Duplicate Removal on 3rd copy (to Kosz) when Volume A is online, and Offline prompt when Volume is offline
            r_ok_tr = self.client.post(
                "/actions/move-to-trash",
                data={"scope": "selected", "duplicates_only": "true", "selected_ids": [it_tr3.id]},
            )
            self.assertEqual(r_ok_tr.status_code, 200)
            self.assertFalse(p_trip_3.exists())

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
                r_off_dup = self.client.post(
                    "/actions/delete-permanently",
                    data={"scope": "selected", "duplicates_only": "true", "selected_ids": [it_tr4.id]},
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
            self.assertFalse(any((self.drive_b / "daf_backup").rglob("green_2.jpg")))

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
            self.assertFalse(p_rem_bkp.exists())

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
                r_async_del = self.client.post(
                    "/actions/delete-permanently",
                    headers={"HX-Request": "true"},
                    data={"scope": "selected", "selected_ids": [it_dup_extra.id], "duplicates_only": "true"},
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
                r_huge = self.client.post(
                    "/actions/delete-permanently",
                    headers={"HX-Request": "true"},
                    data={
                        "scope": "selected",
                        "selected_ids_json": json.dumps(huge_ids),
                        "duplicates_only": "true",
                    },
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


if __name__ == "__main__":
    unittest.main()
