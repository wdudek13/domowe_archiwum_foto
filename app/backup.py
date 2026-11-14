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
