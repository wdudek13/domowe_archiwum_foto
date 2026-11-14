import datetime
import logging
import os
from pathlib import Path
import sqlite3
from typing import List, Optional
import app.database as db_module

logger = logging.getLogger("BackupManager")


class BackupManager:
    """Manages online transactional SQLite backups with integrity verification and rotation."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        max_retained_backups: int = 5,
        max_retained: Optional[int] = None,
    ):
        self.db_path = Path(db_path if db_path is not None else db_module.DB_PATH)
        self.backup_dir = Path(backup_dir if backup_dir is not None else db_module.BACKUP_DIR)
        self.max_retained_backups = max_retained if max_retained is not None else max_retained_backups
        self.max_retained = self.max_retained_backups

    def check_integrity(self) -> bool:
        if not self.db_path.exists():
            return True
        conn = sqlite3.connect(self.db_path.as_posix())
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            results: List[str] = [row[0] for row in cursor.fetchall()]
            if results != ["ok"]:
                raise RuntimeError(f"Integrity check failed: {results}")
            return True
        finally:
            conn.close()

    def create_backup(self, tag: str = "auto") -> Path:
        if not self.db_path.exists():
            return self.db_path

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.check_integrity()
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest_path = self.backup_dir / f"media_vault_{tag}_{timestamp}.db"

        source_conn = sqlite3.connect(self.db_path.as_posix())
        dest_conn = sqlite3.connect(dest_path.as_posix())
        try:
            with dest_conn:
                source_conn.backup(dest_conn, pages=250, sleep=0.005)
        finally:
            dest_conn.close()
            source_conn.close()

        self._rotate()
        return dest_path

    def restore_from_backup(self, source_backup_path: Path) -> Path:
        """
        Verifies a downloaded or local SQLite backup file and atomically restores it
        into the active database using sqlite3.Connection.backup().
        Creates a safety pre_restore backup before overwriting existing data.
        """
        source_path = Path(source_backup_path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Plik kopii zapasowej nie istnieje: {source_path}")

        src_check = sqlite3.connect(source_path.as_posix())
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
            self.create_backup("pre_restore")

        if self.db_path.resolve() == db_module.DB_PATH.resolve():
            db_module.engine.dispose()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        source_conn = sqlite3.connect(source_path.as_posix())
        dest_conn = sqlite3.connect(self.db_path.as_posix(), timeout=30.0)
        try:
            with dest_conn:
                source_conn.backup(dest_conn, pages=250, sleep=0.005)
        finally:
            dest_conn.close()
            source_conn.close()

        db_module.ensure_schema_migrations(self.db_path)
        return self.db_path

    def _rotate(self) -> None:
        backups = sorted(
            list(self.backup_dir.glob("media_vault_*.db")),
            key=os.path.getmtime,
            reverse=True,
        )
        if len(backups) > self.max_retained_backups:
            for old_backup in backups[self.max_retained_backups:]:
                try:
                    old_backup.unlink()
                except OSError as exc:
                    logger.debug("Could not remove rotated backup %s: %s", old_backup, exc)
