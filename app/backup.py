import datetime
import logging
import os
from pathlib import Path
import sqlite3
from typing import List
from app.database import DB_PATH, BACKUP_DIR

logger = logging.getLogger("BackupManager")


class BackupManager:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        backup_dir: Path = BACKUP_DIR,
        max_retained_backups: int = 5,
        max_retained: int = None,
    ):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
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
            for b in backups[self.max_retained_backups:]:
                try:
                    b.unlink()
                except OSError:
                    pass
