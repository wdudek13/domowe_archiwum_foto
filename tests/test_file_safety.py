import datetime
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import uuid

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import FormData

from app.backup import BackupManager
from app.database import Base, ensure_schema_migrations, get_db, set_sqlite_pragma
from app.file_safety import (
    FileSafetyGuard,
    QUARANTINE_DIR_NAME,
    get_audit_log_path,
    quarantine_file,
    safe_resolve_volume_path,
)
from app.models import BackgroundJob, MediaGroup, MediaItem, MediaTag, WindowsVolume
from app.operation_log import ProcessManager
from app.routers import actions as actions_router
from app.scanner import FileScannerService, WindowsVolumeInspector
from app.storage_ops import StorageAutomationService, _safe_move, _safe_unlink
import main as main_module


def _make_jpeg(path: Path, color=(120, 80, 40), size=(240, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, "JPEG", quality=90)
    return path


class _FilterFormFieldExtractor(HTMLParser):
    """Parses app/templates/index.html and collects all form field names inside <form id='filter-form'>."""

    def __init__(self):
        super().__init__()
        self.in_filter_form = False
        self.field_names = set()

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "form" and attr_map.get("id") == "filter-form":
            self.in_filter_form = True
            return
        if self.in_filter_form and tag in ("input", "select", "textarea"):
            name = attr_map.get("name")
            if name:
                self.field_names.add(name)

    def handle_endtag(self, tag):
        if tag == "form" and self.in_filter_form:
            self.in_filter_form = False


class TestZeroTrustFileSafetyMasterSpec(unittest.TestCase):
    """
    Comprehensive test suite verifying every finding (C-01..C-05, H-01..H-10, M-01..M-08, L-01..L-02)
    from the Consolidated Master Security Specification.
    Runs 100% inside isolated TemporaryDirectory with mocked VSN maps; NEVER touches real user disks.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "safety_test.db"
        self.backup_dir = self.root / "backups"
        self.thumbs_dir = self.root / "thumbnails"
        self.logs_dir = self.root / "logs"
        self.reports_dir = self.root / "reports"
        self.log_file = self.logs_dir / "operations.log"
        self.drive_a = self.root / "DriveA"
        self.drive_b = self.root / "DriveB"
        self.drive_c = self.root / "DriveC"

        for d in (
            self.backup_dir,
            self.thumbs_dir,
            self.logs_dir,
            self.reports_dir,
            self.drive_a,
            self.drive_b,
            self.drive_c,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
            future=True,
        )

        def _fast_pragma(dbapi_conn, _rec):
            set_sqlite_pragma(dbapi_conn, _rec)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA synchronous = OFF;")
            cur.close()

        event.listen(self.engine, "connect", _fast_pragma)
        Base.metadata.create_all(bind=self.engine)
        ensure_schema_migrations(self.db_path)

        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=False
        )

        self.patches = [
            patch("app.database.DB_PATH", self.db_path),
            patch("app.database.BACKUP_DIR", self.backup_dir),
            patch("app.database.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.database.LOGS_DIR", self.logs_dir),
            patch("app.database.REPORTS_DIR", self.reports_dir),
            patch("app.database.OPERATIONS_LOG_FILE", self.log_file),
            patch("app.database.SessionLocal", self.SessionLocal),
            patch("app.scanner.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.media.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.semantic.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.actions.THUMBNAILS_DIR", self.thumbs_dir),
            patch("app.routers.actions.DATA_DIR", self.root),
            patch("app.routers.actions.SessionLocal", self.SessionLocal),
            patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value={}),
        ]
        for p in self.patches:
            p.start()

        def _override_get_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        main_module.app.dependency_overrides[get_db] = _override_get_db
        self.client = TestClient(main_module.app)

    def tearDown(self):
        main_module.app.dependency_overrides.clear()
        actions_router.DELETION_TOKENS.clear()
        # Ensure any Read-Only files created in temp dir are writable before cleanup
        for p in self.root.rglob("*"):
            try:
                if p.is_file() and not (os.stat(p).st_mode & stat.S_IWRITE):
                    os.chmod(p, stat.S_IWRITE)
            except OSError:
                pass
        for p in reversed(self.patches):
            p.stop()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _create_volumes(self, db):
        va = WindowsVolume(
            volume_serial_number="AA11-1111",
            volume_label="SafeDiskA",
            last_known_drive_letter=str(self.drive_a),
            is_online=True,
        )
        vb = WindowsVolume(
            volume_serial_number="BB22-2222",
            volume_label="SafeDiskB",
            last_known_drive_letter=str(self.drive_b),
            is_online=True,
        )
        db.add_all([va, vb])
        db.commit()
        db.refresh(va)
        db.refresh(vb)
        return va, vb

    # =========================================================================
    # 1. C-05 & L-02: FILTER PARITY (INCLUDING trashed_only)
    # =========================================================================
    def test_c05_l02_filter_form_parity_and_trashed_only_isolation(self):
        """
        Parses app/templates/index.html #filter-form and verifies:
        1) 100% of filter fields (including trashed_only) are extracted by _extract_filter_params.
        2) GET /media/search total_matching == len(_resolve_bulk_ids_from_form) == len(resolve_target_items)
           for every filter including trashed_only + volume_id!
        3) Delete-preview with trashed_only=true NEVER includes active (non-trashed) files.
        """
        index_html = (Path(__file__).resolve().parent.parent / "app" / "templates" / "index.html").read_text(
            encoding="utf-8"
        )
        parser = _FilterFormFieldExtractor()
        parser.feed(index_html)

        non_filter_display_fields = {"page", "sort_by", "view_mode"}
        expected_filter_fields = parser.field_names - non_filter_display_fields
        self.assertIn("trashed_only", expected_filter_fields)
        self.assertIn("unbacked_only", expected_filter_fields)
        self.assertIn("quality_defect", expected_filter_fields)

        extracted_keys = set(actions_router._extract_filter_params(FormData()).keys())
        missing = expected_filter_fields - extracted_keys
        self.assertEqual(missing, set(), f"Filter drift detected! Fields missing from _extract_filter_params: {missing}")

        with self.SessionLocal() as db:
            va, vb = self._create_volumes(db)
            grp = MediaGroup(name="ParityAlbum")
            db.add(grp)
            db.commit()
            db.refresh(grp)

            p_active = _make_jpeg(self.drive_a / "active_family.jpg", color=(10, 120, 220))
            p_bkp = self.drive_b / "daf_backup" / "2025" / "01" / "01" / "active_family.jpg"
            p_bkp.parent.mkdir(parents=True, exist_ok=True)
            p_bkp.write_bytes(p_active.read_bytes())
            p_trashed = _make_jpeg(self.drive_a / "in_trash_item.jpg", color=(220, 20, 20))
            p_unbacked = _make_jpeg(self.drive_a / "solo_unbacked.jpg", color=(40, 200, 60))

            it_active = FileScannerService.process_file(p_active, va)
            it_active.group_id = grp.id
            it_active.has_date_anomaly = True
            it_active.date_anomaly_reason = "Niezgodność dat: folder"
            it_active.is_blurry = True
            it_active.blur_score = 12.0

            it_bkp = FileScannerService.process_file(p_bkp, vb)
            it_bkp.is_backup_copy = True
            it_bkp.duplicate_role = "backup_green"

            it_trashed = FileScannerService.process_file(p_trashed, va)
            it_trashed.is_trashed = True

            it_unbacked = FileScannerService.process_file(p_unbacked, va)
            it_unbacked.is_corrupted = True

            db.add_all([it_active, it_bkp, it_trashed, it_unbacked])
            db.commit()
            db.add(MediaTag(media_id=it_unbacked.id, tag="do_usuniecia:uszkodzone", confidence=1.0, source_model="test"))
            db.commit()

            va_id = va.id
            active_id, trashed_id, unbacked_id = it_active.id, it_trashed.id, it_unbacked.id

        vsn_map = {
            "AA11-1111": (str(self.drive_a), "SafeDiskA"),
            "BB22-2222": (str(self.drive_b), "SafeDiskB"),
        }
        filter_cases = [
            {"trashed_only": "true"},
            {"trashed_only": "true", "volume_id": str(va_id)},
            {"unbacked_only": "true"},
            {"anomaly_only": "true"},
            {"blurry_only": "true"},
            {"ungrouped_only": "true"},
            {"untagged_only": "true"},
            {"quality_defect": "uszkodzone"},
            {"volume_id": str(va_id)},
            {"q": "active_family"},
            {"group_id": str(grp.id)},
            {"online_only": "true"},
        ]

        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            for params in filter_cases:
                r_get = self.client.get("/media/search", params=params)
                self.assertEqual(r_get.status_code, 200)
                m_tot = re.search(r'id="view-total-matching"[^>]*value="(\d+)"', r_get.text)
                self.assertIsNotNone(m_tot, f"Missing #view-total-matching for params={params}")
                get_count = int(m_tot.group(1))

                form_pairs = [("scope", "filtered")] + list(params.items())
                with self.SessionLocal() as db:
                    post_ids, fp = actions_router._resolve_bulk_ids_from_form(db, FormData(form_pairs))
                    orm_items = actions_router.resolve_target_items(
                        db,
                        scope="filtered",
                        selected_ids=[],
                        q=fp["q"],
                        media_type=fp["media_type"],
                        volume_id=fp["volume_id"],
                        folder_prefix=fp["folder_prefix"],
                        group_id=fp["group_id"],
                        tag=fp["tag"],
                        year=fp["year"],
                        anomaly_only=fp["anomaly_only"],
                        blurry_only=fp["blurry_only"],
                        duplicates_only=fp["duplicates_only"],
                        untagged_only=fp["untagged_only"],
                        ungrouped_only=fp["ungrouped_only"],
                        unbacked_only=fp["unbacked_only"],
                        unvectored_only=fp["unvectored_only"],
                        unanalyzed_quality_only=fp["unanalyzed_quality_only"],
                        quality_defect=fp["quality_defect"],
                        online_only=fp["online_only"],
                        trashed_only=fp["trashed_only"],
                    )
                self.assertEqual(
                    len(post_ids),
                    get_count,
                    f"GET vs POST count drift for params={params}: GET={get_count}, POST={len(post_ids)}",
                )
                self.assertEqual(len(orm_items), get_count)

            # C-05 exploit test: Trash view + volume_id + scope=filtered MUST target ONLY trashed_id!
            r_prev_trash = self.client.post(
                "/actions/delete-preview",
                data={"scope": "filtered", "trashed_only": "true", "volume_id": str(va_id)},
            )
            self.assertEqual(r_prev_trash.status_code, 200)
            self.assertIn("Podsumowanie przed trwałym usunięciem (1 plików)", r_prev_trash.text)
            m_tok = re.search(r"name='deletion_token' value='([^']+)'", r_prev_trash.text)
            self.assertIsNotNone(m_tok)
            tok_ids = actions_router.DELETION_TOKENS[m_tok.group(1)]["ids"]
            self.assertEqual(tok_ids, [trashed_id])
            self.assertNotIn(active_id, tok_ids)
            self.assertNotIn(unbacked_id, tok_ids)

    # =========================================================================
    # 2. C-01 + INFO-01: delete_group_id MUST GO THROUGH PREVIEW + TOKEN
    # =========================================================================
    def test_c01_delete_group_id_requires_two_phase_preview_and_originals_gate(self):
        with self.SessionLocal() as db:
            va, _ = self._create_volumes(db)
            grp = MediaGroup(name="Wakacje2010")
            db.add(grp)
            db.commit()
            db.refresh(grp)
            p1 = _make_jpeg(self.drive_a / "wakacje1.jpg", color=(20, 40, 200))
            it1 = FileScannerService.process_file(p1, va)
            it1.group_id = grp.id
            it1.duplicate_role = "original"
            db.add(it1)
            db.commit()
            grp_id, it1_id = grp.id, it1.id

        vsn_map = {"AA11-1111": (str(self.drive_a), "SafeDiskA")}
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            # Direct POST /actions/delete-permanently with delete_group_id MUST BE BLOCKED!
            r_bypass = self.client.post("/actions/delete-permanently", data={"delete_group_id": str(grp_id)})
            self.assertEqual(r_bypass.status_code, 200)
            self.assertIn("wymagany jest ważny token potwierdzenia", r_bypass.text)
            self.assertTrue(p1.exists())
            with self.SessionLocal() as db:
                self.assertIsNotNone(db.get(MediaGroup, grp_id))
                self.assertIsNotNone(db.get(MediaItem, it1_id))

            # Proper flow via /actions/delete-preview -> token + USUŃ -> quarantines file & removes group
            r_prev = self.client.post("/actions/delete-preview", data={"delete_group_id": str(grp_id)})
            self.assertEqual(r_prev.status_code, 200)
            self.assertIn("Wakacje2010", r_prev.text)
            self.assertIn("Jedyne egzemplarze:", r_prev.text)
            self.assertIn("SafeDiskA/wakacje1.jpg", r_prev.text)
            m_tok = re.search(r"name='deletion_token' value='([^']+)'", r_prev.text)
            self.assertIsNotNone(m_tok)
            tok = m_tok.group(1)

            # Without 'USUŃ' -> blocked
            r_no_word = self.client.post(
                "/actions/delete-permanently",
                data={"deletion_token": tok, "confirmed_count": "1"},
            )
            self.assertIn("Zablokowano usunięcie Oryginałów", r_no_word.text)
            self.assertTrue(p1.exists())

            # With 'USUŃ' -> quarantined to daf_kwarantanna/ (recoverable!) and removed from DB
            r_ok = self.client.post(
                "/actions/delete-permanently",
                data={
                    "deletion_token": tok,
                    "confirmed_count": "1",
                    "allow_originals": "true",
                    "confirm_originals_text": "USUŃ",
                },
            )
            self.assertEqual(r_ok.status_code, 200)
            self.assertFalse(p1.exists())
            q_files = list((self.drive_a / QUARANTINE_DIR_NAME).rglob("wakacje1.jpg"))
            self.assertEqual(len(q_files), 1, "Album deletion must move file to daf_kwarantanna/, not hard-unlink!")
            with self.SessionLocal() as db:
                self.assertIsNone(db.get(MediaGroup, grp_id))
                self.assertIsNone(db.get(MediaItem, it1_id))

            # BUG-01 (SZLIF-1): Album with 1 online + 1 OFFLINE item MUST NOT delete MediaGroup from DB!
            with self.SessionLocal() as db:
                va_row = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "AA11-1111"))
                vb_row = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == "BB22-2222"))
                grp_mix = MediaGroup(name="MixedOfflineAlbum")
                db.add(grp_mix)
                db.commit()
                db.refresh(grp_mix)
                p_on_m = _make_jpeg(self.drive_a / "mix_online.jpg", color=(55, 66, 77))
                p_off_m = _make_jpeg(self.drive_b / "mix_offline.jpg", color=(77, 66, 55))
                it_on_m = FileScannerService.process_file(p_on_m, va_row)
                it_off_m = FileScannerService.process_file(p_off_m, vb_row)
                it_on_m.group_id = grp_mix.id
                it_off_m.group_id = grp_mix.id
                it_on_m.duplicate_role = "original"
                it_off_m.duplicate_role = "original"
                db.add_all([it_on_m, it_off_m])
                db.commit()
                mix_grp_id, on_m_id, off_m_id = grp_mix.id, it_on_m.id, it_off_m.id

            # Only SafeDiskA is online; SafeDiskB is offline
            r_prev_mix = self.client.post("/actions/delete-preview", data={"delete_group_id": str(mix_grp_id)})
            m_tok_mix = re.search(r"name='deletion_token' value='([^']+)'", r_prev_mix.text)
            self.assertIsNotNone(m_tok_mix)
            r_exec_mix = self.client.post(
                "/actions/delete-permanently",
                data={
                    "deletion_token": m_tok_mix.group(1),
                    "confirmed_count": "2",
                    "allow_originals": "true",
                    "confirm_originals_text": "USUŃ",
                },
            )
            self.assertEqual(r_exec_mix.status_code, 200)
            with self.SessionLocal() as db:
                self.assertIsNone(db.get(MediaItem, on_m_id))
                self.assertIsNotNone(db.get(MediaItem, off_m_id), "Offline item must stay in DB!")
                self.assertIsNotNone(
                    db.get(MediaGroup, mix_grp_id),
                    "BUG-01: Album MediaGroup must NOT be deleted while offline items still belong to it!",
                )

    # =========================================================================
    # 3. C-02 + INFO-02 + M-04: RESUME BLOCKED WITHOUT PREVIEW; skip_db_backup GUARD
    # =========================================================================
    def test_c02_m04_process_control_blocks_delete_resume_and_skip_backup_forgery(self):
        with self.SessionLocal() as db:
            va, _ = self._create_volumes(db)
            p1 = _make_jpeg(self.drive_a / "dont_resume_me.jpg", color=(180, 90, 20))
            it1 = FileScannerService.process_file(p1, va)
            it1.duplicate_role = "original"
            db.add(it1)
            db.commit()
            it1_id = it1.id

            job = BackgroundJob(
                job_type="delete_permanently",
                title="Stopped Incident Job",
                status="STOPPED",
                processed_count=0,
                total_count=1,
                params_json=json.dumps({"is_duplicate_removal": False, "delete_mode": "permanent"}),
                remaining_ids_json=json.dumps([it1_id]),
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            job_id = job.id

        vsn_map = {"AA11-1111": (str(self.drive_a), "SafeDiskA")}
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            # 1) Direct resume via /actions/process-control MUST BE BLOCKED!
            r_res = self.client.post(
                "/actions/process-control",
                data={"action": "resume", "job_id": str(job_id), "run_sync": "true"},
            )
            self.assertEqual(r_res.status_code, 200)
            self.assertIn("Zablokowano bezpośrednie wznowienie zadania trwałego usuwania", r_res.text)
            self.assertIn("resume_job_id", r_res.text)
            self.assertTrue(p1.exists(), "File must NOT be deleted when resume is attempted!")
            with self.SessionLocal() as db:
                self.assertEqual(db.get(BackgroundJob, job_id).status, "STOPPED")
                self.assertIsNotNone(db.get(MediaItem, it1_id))

            # 2) Status panel renders preview button for delete_permanently jobs (not direct resume)
            r_st = self.client.get("/actions/status-panel")
            self.assertEqual(r_st.status_code, 200)
            self.assertIn("Podgląd przed wznowieniem", r_st.text)

            # 3) Direct continue_job_id bypass without deletion_token MUST BE BLOCKED!
            r_cont_bypass = self.client.post(
                "/actions/delete-permanently",
                data={"continue_job_id": str(job_id), "duplicates_only": "true", "selected_ids": [it1_id]},
            )
            self.assertIn("wymagany jest ważny token potwierdzenia", r_cont_bypass.text)
            self.assertTrue(p1.exists())

            # 4) M-04: Forged skip_db_backup=true on non-continuation preview token MUST STILL create DB backup!
            r_prev = self.client.post("/actions/delete-preview", data={"resume_job_id": str(job_id)})
            m_tok = re.search(r"name='deletion_token' value='([^']+)'", r_prev.text)
            self.assertIsNotNone(m_tok)
            with patch("app.routers.actions.BackupManager.create_backup") as mock_bkp:
                r_exec = self.client.post(
                    "/actions/delete-permanently",
                    data={
                        "deletion_token": m_tok.group(1),
                        "confirmed_count": "1",
                        "allow_originals": "true",
                        "confirm_originals_text": "USUŃ",
                        "skip_db_backup": "true",
                    },
                )
                self.assertEqual(r_exec.status_code, 200)
                mock_bkp.assert_called_once_with("pre_permanent_disk_delete")

    # =========================================================================
    # 4. H-01 & M-02: view_total_matching MISMATCH & MAX_ORIGINALS_DELETE_BATCH
    # =========================================================================
    def test_h01_m02_view_total_mismatch_and_originals_batch_limit(self):
        with self.SessionLocal() as db:
            va, _ = self._create_volumes(db)
            ids = []
            for i in range(3):
                p = _make_jpeg(self.drive_a / f"batch_{i}.jpg", color=(10 + i * 30, 50, 90))
                it = FileScannerService.process_file(p, va)
                it.duplicate_role = "original"
                db.add(it)
                ids.append(it.id)
            db.commit()
            va_id = va.id

        vsn_map = {"AA11-1111": (str(self.drive_a), "SafeDiskA")}
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_map):
            # H-01 + BUG-02: Both view_total_matching < actual (1 vs 3) AND view_total_matching > actual (10 vs 3) MUST BE BLOCKED!
            for bad_expected in ("1", "10"):
                for url, extra in [
                    ("/actions/delete-preview", {}),
                    ("/actions/move-to-trash", {}),
                    ("/actions/unindex", {}),
                    ("/actions/move-files", {"target_volume_id": str(va_id), "target_subfolder": "Sub"}),
                ]:
                    r_mm = self.client.post(
                        url,
                        data={"scope": "filtered", "volume_id": str(va_id), "view_total_matching": bad_expected, **extra},
                    )
                    self.assertEqual(r_mm.status_code, 200)
                    self.assertIn("rozbieżność filtra", r_mm.text, f"Mismatch ({bad_expected} vs 3) not blocked on {url}")

            # Originals batch limit exceeded -> BLOCKED!
            with patch.object(actions_router, "MAX_ORIGINALS_DELETE_BATCH", 2):
                r_cap = self.client.post(
                    "/actions/delete-preview",
                    data={"scope": "filtered", "volume_id": str(va_id), "view_total_matching": "3"},
                )
                self.assertEqual(r_cap.status_code, 200)
                self.assertIn("przekroczono limit usuwania Oryginałów", r_cap.text)
                self.assertNotIn("deletion_token", r_cap.text)

    # =========================================================================
    # 5. C-03, C-04, H-03, H-06, L-01, M-06: FileSafetyGuard CORE INVARIANTS
    # =========================================================================
    def test_c03_c04_h03_h06_l01_m06_file_safety_guard_invariants(self):
        # H-06: Path traversal & drive escape blocked
        for bad_rel in ("", ".", "../escape.jpg", "sub/../../escape.jpg", "/etc/passwd", "\\Windows\\win.ini", "C:\\secret.jpg", "D:rel.jpg"):
            with self.assertRaises(ValueError, msg=f"Path {bad_rel!r} should have been rejected"):
                safe_resolve_volume_path(self.drive_a, bad_rel)
        good_p = safe_resolve_volume_path(self.drive_a, "DCIM/2026/ok.jpg")
        self.assertTrue(good_p.is_relative_to(self.drive_a.resolve()))

        # C-03: Quarantine moves file atomically to daf_kwarantanna/YYYYMMDD/..., writes manifest & audit JSONL
        p_q = _make_jpeg(self.drive_a / "Family" / "photo1.jpg", color=(100, 150, 200))
        orig_bytes = p_q.read_bytes()
        q_dst = quarantine_file(p_q, self.drive_a, reason="unit_test", file_hash="abc123", vsn="AA11-1111")
        self.assertFalse(p_q.exists())
        self.assertTrue(q_dst.exists())
        self.assertEqual(q_dst.read_bytes(), orig_bytes)
        self.assertIn(QUARANTINE_DIR_NAME, q_dst.parts)

        manifest_p = self.drive_a / QUARANTINE_DIR_NAME / "manifest.jsonl"
        self.assertTrue(manifest_p.exists())
        manifest_rec = json.loads(manifest_p.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(manifest_rec["relative_path"], "Family/photo1.jpg")
        self.assertEqual(manifest_rec["reason"], "unit_test")

        # L-01: Audit JSONL written inside isolated test logs dir
        audit_p = get_audit_log_path()
        self.assertTrue(audit_p.exists())
        audit_rec = json.loads(audit_p.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(audit_rec["op"], "quarantine")
        self.assertEqual(audit_rec["status"], "OK")

        # M-06: Scanner excludes daf_kwarantanna/
        self.assertTrue(FileScannerService.is_in_excluded_album_export_folder("daf_kwarantanna/20261120/Family/photo1.jpg"))
        self.assertTrue(FileScannerService.is_in_excluded_album_export_folder("$RECYCLE.BIN/S-1-5/a.jpg"))

        # C-03: Read-Only file MUST NEVER have Read-Only stripped or be deleted/quarantined/moved!
        p_ro = _make_jpeg(self.drive_a / "readonly_priceless.jpg", color=(250, 200, 10))
        os.chmod(p_ro, stat.S_IREAD)
        try:
            with self.assertRaises(PermissionError):
                _safe_unlink(p_ro, volume_root=self.drive_a)
            with self.assertRaises(PermissionError):
                quarantine_file(p_ro, self.drive_a)
            with self.assertRaises(PermissionError):
                _safe_move(p_ro, self.drive_a / "moved_ro.jpg", src_volume_root=self.drive_a, dst_volume_root=self.drive_a)
            self.assertTrue(p_ro.exists())
            self.assertFalse(bool(os.stat(p_ro).st_mode & stat.S_IWRITE), "Read-Only bit must remain intact!")
        finally:
            os.chmod(p_ro, stat.S_IWRITE)

        # C-04: Hard unlink MUST be refused without live 100%-verified online second copy
        p_last = _make_jpeg(self.drive_a / "only_copy.jpg", color=(77, 88, 99))
        p_diff = _make_jpeg(self.drive_b / "different.jpg", color=(11, 22, 33))
        with self.assertRaises(PermissionError):
            FileSafetyGuard.quarantine_or_unlink(p_last, volume_root=self.drive_a, allow_hard_unlink=True)
        with self.assertRaises(PermissionError):
            FileSafetyGuard.quarantine_or_unlink(
                p_last, volume_root=self.drive_a, verified_copy_path=p_diff, allow_hard_unlink=True
            )
        self.assertTrue(p_last.exists())

        # H-03: _safe_move MUST NEVER overwrite an existing destination file!
        p_src = _make_jpeg(self.drive_a / "mv_src.jpg", color=(10, 10, 10))
        p_dst = _make_jpeg(self.drive_a / "mv_dst_existing.jpg", color=(240, 240, 240))
        dst_orig_bytes = p_dst.read_bytes()
        with self.assertRaises(FileExistsError):
            _safe_move(p_src, p_dst, src_volume_root=self.drive_a, dst_volume_root=self.drive_a)
        self.assertTrue(p_src.exists())
        self.assertEqual(p_dst.read_bytes(), dst_orig_bytes, "Existing destination file must be byte-identical!")

    # =========================================================================
    # 6. H-04, H-05, H-07, H-09, H-10: ATOMICITY, STRICT VSN, GDRIVE, OFFLINE DUP
    # =========================================================================
    def test_h04_h05_h07_h09_h10_consistency_vsn_gdrive_and_offline_dup_quarantine(self):
        with self.SessionLocal() as db:
            va, vb = self._create_volumes(db)
            p1 = _make_jpeg(self.drive_a / "atomic_1.jpg", color=(30, 60, 90))
            p2 = _make_jpeg(self.drive_a / "atomic_2.jpg", color=(90, 60, 30))
            it1 = FileScannerService.process_file(p1, va)
            it2 = FileScannerService.process_file(p2, va)
            db.add_all([it1, it2])
            db.commit()
            va_id, vb_id, it1_id, it2_id = va.id, vb.id, it1.id, it2.id

        # H-05: Default strict_vsn=True MUST reject hardware VSN / drive root when connected_map is empty!
        with self.SessionLocal() as db:
            svc = StorageAutomationService(db)
            va_row = db.get(WindowsVolume, va_id)
            self.assertIsNone(svc.verify_and_resolve_online_volume_root(va_row))

        vsn_both = {
            "AA11-1111": (str(self.drive_a), "SafeDiskA"),
            "BB22-2222": (str(self.drive_b), "SafeDiskB"),
        }
        # H-04: Per-file commit in schedule_or_execute_moves — crash on 2nd file keeps 1st file committed in DB!
        real_safe_move = _safe_move
        call_counter = {"n": 0}

        def _flaky_move(*args, **kwargs):
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                raise OSError("Simulated I/O disconnect on 2nd file")
            return real_safe_move(*args, **kwargs)

        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both), \
             patch("app.storage_ops._safe_move", side_effect=_flaky_move):
            with self.SessionLocal() as db:
                svc = StorageAutomationService(db)
                items_to_mv = [db.get(MediaItem, it1_id), db.get(MediaItem, it2_id)]
                with self.assertRaises(OSError):
                    svc.schedule_or_execute_moves(items_to_mv, target_volume_id=vb_id, target_subfolder="MovedBatch")

        with self.SessionLocal() as db:
            it1_after = db.get(MediaItem, it1_id)
            it2_after = db.get(MediaItem, it2_id)
            self.assertEqual(it1_after.volume_id, vb_id, "1st physically moved file must already be committed in DB!")
            self.assertEqual(it1_after.relative_path, "MovedBatch/atomic_1.jpg")
            self.assertEqual(it2_after.volume_id, va_id, "2nd failed file must still point to original DriveA in DB!")

        # H-07: organize-originals requires explicit volume_id
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both):
            r_no_vol = self.client.post("/actions/organize-originals")
            self.assertIn("Wybierz najpierw konkretny dysk", r_no_vol.text)

        # H-09: GDrive purge_local MUST NOT delete non-original (`backup_orange`) if NO other cross-drive local copy exists!
        with self.SessionLocal() as db:
            va = db.get(WindowsVolume, va_id)
            p_orange = _make_jpeg(self.drive_a / "solo_orange.jpg", color=(255, 140, 0))
            it_orange = FileScannerService.process_file(p_orange, va)
            it_orange.duplicate_role = "backup_orange"
            it_orange.is_backup_copy = False
            db.add(it_orange)
            db.commit()
            orange_id = it_orange.id

        mock_gd = MagicMock()
        mock_gd.get_or_create_folder.return_value = "fld_gd"
        mock_gd.list_folders.return_value = []
        mock_gd.list_files_in_folder.return_value = []
        mock_gd.upload_and_verify.return_value = {
            "id": "gd1",
            "name": "solo_orange.jpg",
            "size_bytes": p_orange.stat().st_size,
            "md5": "abc",
            "verified": True,
            "purged_local": False,
        }
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_both), \
             patch("app.cloud.google_drive.GoogleDriveManager", return_value=mock_gd):
            r_gd = self.client.post(
                "/actions/gdrive-sync",
                data={"mode": "media", "scope": "selected", "selected_ids": [orange_id], "purge_local": "true"},
            )
            self.assertEqual(r_gd.status_code, 200)
            # Upload called with purge_local_after_verify=False and solo file stays on disk!
            mock_gd.upload_and_verify.assert_called_once()
            _, kw = mock_gd.upload_and_verify.call_args
            self.assertFalse(kw.get("purge_local_after_verify", True))
            self.assertTrue(p_orange.exists(), "Solo non-original file without another local copy MUST NOT be purged!")

        # H-10: Online duplicate with ALL surviving copies OFFLINE must be moved to daf_kwarantanna/ (never hard-unlinked!)
        with self.SessionLocal() as db:
            va = db.get(WindowsVolume, va_id)
            vb = db.get(WindowsVolume, vb_id)
            p_orig_off = _make_jpeg(self.drive_b / "off_orig.jpg", color=(66, 133, 244))
            p_bkp_off = self.drive_b / "daf_backup" / "2025" / "05" / "05" / "off_orig.jpg"
            p_bkp_off.parent.mkdir(parents=True, exist_ok=True)
            p_bkp_off.write_bytes(p_orig_off.read_bytes())
            p_dup_on = self.drive_a / "on_dup_offline_refs.jpg"
            p_dup_on.write_bytes(p_orig_off.read_bytes())

            it_o = FileScannerService.process_file(p_orig_off, vb)
            it_b = FileScannerService.process_file(p_bkp_off, vb)
            it_d = FileScannerService.process_file(p_dup_on, va)
            it_o.duplicate_role = "original"
            it_b.duplicate_role = "backup_green"
            it_b.is_backup_copy = True
            it_d.duplicate_role = "duplicate"
            it_d.is_primary_duplicate = False
            db.add_all([it_o, it_b, it_d])
            db.commit()
            dup_on_id = it_d.id

        vsn_only_a = {"AA11-1111": (str(self.drive_a), "SafeDiskA")}
        with patch.object(WindowsVolumeInspector, "get_connected_vsn_map", return_value=vsn_only_a):
            with self.SessionLocal() as db:
                res = StorageAutomationService(db).execute_safe_duplicate_removal(
                    [dup_on_id],
                    delete_mode="permanent",
                    force_duplicate_verification=True,
                    trash_dir=self.root / "kosz",
                    thumbnails_dir=self.thumbs_dir,
                )
                self.assertEqual(res["physical_count"], 1)
            self.assertFalse(p_dup_on.exists())
            q_copies = list((self.drive_a / QUARANTINE_DIR_NAME).rglob("on_dup_offline_refs.jpg"))
            self.assertEqual(len(q_copies), 1, "Offline-ref duplicate must be preserved in daf_kwarantanna/!")

    # =========================================================================
    # 7. M-01 CSRF GUARD, M-05 BACKUP ROTATION POOL, STATIC SAFETY AUDIT
    # =========================================================================
    def test_m01_csrf_m05_backup_rotation_and_static_code_audit(self):
        # M-01: Foreign Origin / Referer / Origin: null MUST return HTTP 403
        for hdrs in [
            {"Origin": "https://evil.example.com"},
            {"Referer": "https://attacker.invalid/csrf.html"},
            {"Origin": "null"},
        ]:
            r_csrf = self.client.post("/actions/delete-preview", data={"scope": "selected"}, headers=hdrs)
            self.assertEqual(r_csrf.status_code, 403, f"CSRF header {hdrs} must return 403!")

        r_local = self.client.post(
            "/actions/delete-preview",
            data={"scope": "selected"},
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        self.assertEqual(r_local.status_code, 200)

        # M-05: Dedicated retention pool protects >= 5 most recent pre_permanent_disk_delete snapshots
        rot_dir = self.root / "m05_rot_backups"
        rot_dir.mkdir(parents=True, exist_ok=True)
        pre_snaps = []
        for i in range(6):
            p = rot_dir / f"media_vault_pre_permanent_disk_delete_20261119_10000{i}.db.gz"
            p.write_bytes(b"SAME_SIZE_PAYLOAD" * 50)
            os.utime(p, (1000 + i * 10, 1000 + i * 10))
            pre_snaps.append(p)
        for i in range(25):
            p = rot_dir / f"media_vault_auto_20261120_1200{i:02d}.db.gz"
            p.write_bytes(b"SAME_SIZE_PAYLOAD" * 50)
            os.utime(p, (5000 + i * 10, 5000 + i * 10))

        bm = BackupManager(backup_dir=rot_dir, max_retained_backups=5)
        bm._rotate()

        # The 5 newest pre_permanent_disk_delete snapshots MUST survive despite 25 newer auto backups!
        for survived in pre_snaps[1:]:
            self.assertTrue(survived.exists(), f"Protected pre_permanent_disk_delete snapshot was rotated out: {survived.name}")
        self.assertFalse(pre_snaps[0].exists(), "6th oldest pre_permanent_disk_delete snapshot may rotate out")

        # Static AST audit: No actual shutil.move() or os.chmod() calls in core app modules,
        # and no direct .unlink() calls in storage_ops.py (all routed through app/file_safety.py).
        import ast

        repo_root = Path(__file__).resolve().parent.parent
        for rel_file in ("app/storage_ops.py", "app/routers/actions.py", "app/cloud/google_drive.py", "app/file_safety.py"):
            tree = ast.parse((repo_root / rel_file).read_text(encoding="utf-8"), filename=rel_file)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
                    self.assertFalse(
                        owner == "shutil" and attr == "move",
                        f"Raw shutil.move() call found in {rel_file}:{node.lineno}",
                    )
                    self.assertFalse(
                        owner == "os" and attr == "chmod",
                        f"os.chmod() call found in {rel_file}:{node.lineno}",
                    )
                    if rel_file == "app/storage_ops.py":
                        self.assertNotEqual(
                            attr,
                            "unlink",
                            f"Direct .unlink() call found in {rel_file}:{node.lineno} (must route via FileSafetyGuard)",
                        )


if __name__ == "__main__":
    unittest.main()
