import datetime
import gzip
import logging
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Dict, List, Optional
import app.database as db_module

logger = logging.getLogger("BackupManager")


class BackupManager:
    """Manages online transactional SQLite backups with integrity verification and rotation."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        max_retained_backups: int = 20,
        max_retained: Optional[int] = None,
    ):
        self.db_path = Path(db_path if db_path is not None else db_module.DB_PATH)
        self.backup_dir = Path(backup_dir if backup_dir is not None else db_module.BACKUP_DIR)
        self.max_retained_backups = max_retained if max_retained is not None else max_retained_backups
        self.max_retained = self.max_retained_backups

    def check_integrity(self, quick: bool = True) -> bool:
        if not self.db_path.exists():
            return True
        conn = sqlite3.connect(self.db_path.as_posix())
        try:
            cursor = conn.cursor()
            sql = "PRAGMA quick_check;" if quick else "PRAGMA integrity_check;"
            cursor.execute(sql)
            results: List[str] = [row[0] for row in cursor.fetchall()]
            if results != ["ok"]:
                raise RuntimeError(f"Integrity check failed: {results}")
            return True
        finally:
            conn.close()

    def create_backup(
        self,
        tag: str = "auto",
        target_dir: Optional[Path] = None,
        compress: bool = True,
        exclude_embeddings: bool = False,
        optimize_vacuum: Optional[bool] = None,
        progress_callback: Optional[Any] = None,
    ) -> Path:
        if not self.db_path.exists():
            return self.db_path

        dest_dir = Path(target_dir) if target_dir is not None else self.backup_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        self.check_integrity(quick=True)

        # Only vacuum when embeddings were deleted (to release space), unless explicitly specified
        should_vacuum = optimize_vacuum if optimize_vacuum is not None else exclude_embeddings

        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        actual_tag = f"{tag}_no_clip" if exclude_embeddings else tag
        ext = ".db.gz" if compress else ".db"
        final_file_name = f"media_vault_{actual_tag}_{timestamp}{ext}"
        final_path = dest_dir / final_file_name
        target_tmp_path = dest_dir / f"{final_file_name}.tmp"

        # Always stage locally on fast NVMe/SSD to avoid slow random writes/locks on external USB drives
        staging_dir = self.backup_dir / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        local_stage_db = staging_dir / f"stage_{actual_tag}_{timestamp}.db"

        def _report(msg: str, done: int, total: int = 100):
            if progress_callback:
                try:
                    progress_callback(msg, done, total)
                except Exception as exc:
                    logger.debug("progress_callback ignored error: %s", exc)

        try:
            _report("[Etap 1/3] Kopiowanie bazy SQLite do szybkiego bufora SSD...", 10, 100)

            # Fast transactional SQLite backup buffer
            source_conn = sqlite3.connect(self.db_path.as_posix())
            dest_conn = sqlite3.connect(local_stage_db.as_posix())
            try:
                def _sqlite_prog(status, remaining, total):
                    if total > 0:
                        pct = int(10 + (total - remaining) * 35 / total)
                        _report(f"[Etap 1/3] Kopiowanie stron bazy na SSD ({total - remaining}/{total})...", pct, 100)
                source_conn.backup(dest_conn, pages=10000, progress=_sqlite_prog, sleep=0)
            finally:
                dest_conn.close()
                source_conn.close()

            # Exclude CLIP embeddings if requested, or compact with VACUUM
            if exclude_embeddings:
                _report("[Etap 2/3] Usuwanie wektorów CLIP i kompaktowanie na SSD...", 48, 100)
                cleanup_conn = sqlite3.connect(local_stage_db.as_posix(), isolation_level=None)
                try:
                    try:
                        cleanup_conn.execute("DELETE FROM media_embeddings;")
                    except sqlite3.OperationalError:
                        pass  # media_embeddings might not exist in minimal or legacy schemas
                    cleanup_conn.execute("VACUUM;")
                finally:
                    cleanup_conn.close()
            elif should_vacuum:
                _report("[Etap 2/3] Kompaktowanie bazy (VACUUM)...", 48, 100)
                cleanup_conn = sqlite3.connect(local_stage_db.as_posix(), isolation_level=None)
                try:
                    cleanup_conn.execute("VACUUM;")
                finally:
                    cleanup_conn.close()

            # Stream to target destination (compressed with gzip or raw copy)
            stage_size = local_stage_db.stat().st_size
            if compress:
                _report("[Etap 2/3] Kompresja gzip i zapis na nośniku docelowym...", 55, 100)
                bytes_read = 0
                with open(local_stage_db, "rb") as f_in, gzip.open(target_tmp_path, "wb", compresslevel=4) as f_out:
                    while True:
                        chunk = f_in.read(256 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        bytes_read += len(chunk)
                        if stage_size > 0:
                            pct = int(55 + (bytes_read * 40 / stage_size))
                            mb_done = round(bytes_read / (1024 * 1024), 1)
                            mb_tot = round(stage_size / (1024 * 1024), 1)
                            _report(f"[Etap 2/3] Kompresja gzip i zapis na nośnik ({mb_done}/{mb_tot} MB)...", pct, 100)
            else:
                _report("[Etap 2/3] Zapis kopii na nośniku docelowym...", 55, 100)
                bytes_copied = 0
                with open(local_stage_db, "rb") as f_in, open(target_tmp_path, "wb") as f_out:
                    while True:
                        chunk = f_in.read(512 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        bytes_copied += len(chunk)
                        if stage_size > 0:
                            pct = int(55 + (bytes_copied * 40 / stage_size))
                            mb_done = round(bytes_copied / (1024 * 1024), 1)
                            mb_tot = round(stage_size / (1024 * 1024), 1)
                            _report(f"[Etap 2/3] Zapis na nośnik docelowy ({mb_done}/{mb_tot} MB)...", pct, 100)

            # Atomic rename to final path so partial files are never visible
            if target_tmp_path.exists():
                if final_path.exists():
                    try:
                        final_path.unlink()
                    except OSError:
                        pass
                target_tmp_path.rename(final_path)

            _report("[Etap 3/3] Zakończono! Plik kopii pomyślnie zapisany na nośniku docelowym.", 100, 100)

        finally:
            if local_stage_db.exists():
                try:
                    local_stage_db.unlink()
                except OSError as exc:
                    logger.debug("Failed removing local staged db: %s", exc)
            if target_tmp_path.exists():
                try:
                    target_tmp_path.unlink()
                except OSError:
                    pass

        if target_dir is None:
            self._rotate()
        return final_path

    def restore_from_backup(self, source_backup_path: Path) -> Path:
        """
        Verifies a downloaded or local SQLite backup file (.db or .db.gz) and atomically
        restores it into the active database using sqlite3.Connection.backup().
        Creates a safety pre_restore backup before overwriting existing data.
        """
        source_path = Path(source_backup_path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Plik kopii zapasowej nie istnieje: {source_path}")

        is_gz = source_path.name.lower().endswith(".gz")
        temp_decompressed: Optional[Path] = None
        working_source: Path = source_path

        if is_gz:
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            temp_decompressed = self.backup_dir / f"temp_decomp_{timestamp}_{source_path.stem}.db"
            temp_decompressed.parent.mkdir(parents=True, exist_ok=True)
            try:
                with gzip.open(source_path, "rb") as f_in, open(temp_decompressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out, length=64 * 1024)
                working_source = temp_decompressed
            except Exception as exc:
                if temp_decompressed.exists():
                    try:
                        temp_decompressed.unlink()
                    except OSError:
                        pass
                raise RuntimeError(f"Błąd dekompresji pliku kopii zapasowej: {exc}") from exc

        try:
            src_check = sqlite3.connect(working_source.as_posix())
            try:
                cur = src_check.cursor()
                cur.execute("PRAGMA integrity_check;")
                integrity = [row[0] for row in cur.fetchall()]
                if integrity != ["ok"]:
                    raise RuntimeError("Plik kopii zapasowej jest uszkodzony lub nie jest prawidłową bazą Domowego Archiwum Foto.")
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='media_items';")
                if not cur.fetchone():
                    raise RuntimeError("Plik kopii zapasowej jest uszkodzony lub nie jest prawidłową bazą Domowego Archiwum Foto.")
            except sqlite3.Error as exc:
                raise RuntimeError(
                    f"Plik kopii zapasowej jest uszkodzony lub nie jest prawidłową bazą Domowego Archiwum Foto: {exc}"
                ) from exc
            finally:
                src_check.close()

            if self.db_path.exists():
                self.create_backup("pre_restore", compress=False)

            if self.db_path.resolve() == db_module.DB_PATH.resolve():
                db_module.engine.dispose()

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            source_conn = sqlite3.connect(working_source.as_posix())
            dest_conn = sqlite3.connect(self.db_path.as_posix(), timeout=30.0)
            try:
                with dest_conn:
                    source_conn.backup(dest_conn, pages=2000, sleep=0.001)
            finally:
                dest_conn.close()
                source_conn.close()

            db_module.ensure_schema_migrations(self.db_path)
            return self.db_path
        finally:
            if temp_decompressed and temp_decompressed.exists():
                try:
                    temp_decompressed.unlink()
                except OSError as exc:
                    logger.debug("Failed cleaning decompressed restore temp: %s", exc)

    def _rotate(self) -> None:
        backups = sorted(
            list(self.backup_dir.glob("media_vault_*.db*")),
            key=os.path.getmtime,
            reverse=True,
        )
        if len(backups) <= self.max_retained_backups:
            return
        try:
            raw_sz = backups[0].stat().st_size
            newest_size = raw_sz if isinstance(raw_sz, int) else 0
        except OSError:
            newest_size = 0
        for old_backup in backups[self.max_retained_backups:]:
            try:
                # V-09 protection: never automatically rotate out a snapshot that is >20% larger
                # than the newest DB backup (protects pre-incident rescue snapshots).
                old_sz = old_backup.stat().st_size
                if newest_size > 0 and isinstance(old_sz, int) and old_sz > int(newest_size * 1.20):
                    logger.warning(
                        "Preserving oversized rescue backup snapshot from rotation: %s", old_backup.name
                    )
                    continue
                old_backup.unlink()
            except OSError as exc:
                logger.debug("Could not remove rotated backup %s: %s", old_backup, exc)

    @staticmethod
    def list_backups_in_dir(directory: Path) -> List[Dict[str, Any]]:
        """Lists SQLite database backup files (*.db, *.sqlite, *.db.gz) in the specified directory."""
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        files: List[Dict[str, Any]] = []
        for ext in ("*.db", "*.sqlite", "*.sqlite3", "*.db.gz", "*.sqlite.gz", "*.sqlite3.gz"):
            for p in dir_path.glob(ext):
                try:
                    st = p.stat()
                    name_l = p.name.lower()
                    has_clip = not ("_no_clip" in name_l)
                    is_gz = name_l.endswith(".gz")
                    files.append({
                        "name": p.name,
                        "path": str(p.resolve()),
                        "size_bytes": st.st_size,
                        "size_mb": round(st.st_size / (1024 * 1024), 2),
                        "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "is_compressed": is_gz,
                        "has_clip": has_clip,
                    })
                except OSError:
                    continue
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return files
