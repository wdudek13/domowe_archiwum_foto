from pathlib import Path
from typing import Generator
import logging
import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("Database")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
BACKUP_DIR = DATA_DIR / "backups"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
LOGS_DIR = DATA_DIR / "logs"
REPORTS_DIR = DATA_DIR / "reports"
OPERATIONS_LOG_FILE = LOGS_DIR / "operations.log"

for directory in (DB_DIR, BACKUP_DIR, THUMBNAILS_DIR, LOGS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "media_vault.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30.0},
    echo=False,
    future=True,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = FULL;")
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("PRAGMA busy_timeout = 30000;")
    cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA cache_size = -64000;")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def ensure_schema_migrations(target_db_path: Path = DB_PATH) -> None:
    """Safely adds new columns and FTS5 index to existing SQLite tables without data loss."""
    if not target_db_path.exists():
        return
    conn = sqlite3.connect(target_db_path.as_posix(), timeout=10.0)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(windows_volumes);")
        v_cols = {row[1] for row in cur.fetchall()}
        if v_cols:
            if "total_space_bytes" not in v_cols:
                cur.execute("ALTER TABLE windows_volumes ADD COLUMN total_space_bytes BIGINT DEFAULT 0;")
            if "free_space_bytes" not in v_cols:
                cur.execute("ALTER TABLE windows_volumes ADD COLUMN free_space_bytes BIGINT DEFAULT 0;")

        cur.execute("PRAGMA table_info(media_groups);")
        g_cols = {row[1] for row in cur.fetchall()}
        if g_cols:
            if "parent_id" not in g_cols:
                cur.execute(
                    "ALTER TABLE media_groups ADD COLUMN parent_id INTEGER REFERENCES media_groups(id) ON DELETE CASCADE;"
                )
            if "target_volume_id" not in g_cols:
                cur.execute("ALTER TABLE media_groups ADD COLUMN target_volume_id INTEGER;")
            if "is_trashed" not in g_cols:
                cur.execute("ALTER TABLE media_groups ADD COLUMN is_trashed BOOLEAN DEFAULT 0 NOT NULL;")

        cur.execute("PRAGMA table_info(media_items);")
        m_cols = {row[1] for row in cur.fetchall()}
        if m_cols:
            if "updated_at" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN updated_at DATETIME;")
            if "is_trashed" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN is_trashed BOOLEAN DEFAULT 0 NOT NULL;")
            if "rotation_angle" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN rotation_angle INTEGER DEFAULT 0 NOT NULL;")
            if "is_backup_copy" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN is_backup_copy BOOLEAN DEFAULT 0 NOT NULL;")
            if "verified_sha256" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN verified_sha256 VARCHAR(64);")

        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS media_tags_fts USING fts5(media_id UNINDEXED, tag, source_model UNINDEXED);"
        )
        # Clear legacy false-positive date anomalies caused solely by missing EXIF metadata
        if m_cols:
            cur.execute(
                """
                UPDATE media_items
                SET has_date_anomaly = 0, date_anomaly_reason = NULL
                WHERE date_anomaly_reason LIKE 'Brak metadanych EXIF%'
                   OR date_anomaly_reason LIKE 'Brak EXIF%';
                """
            )
        conn.commit()
    except sqlite3.Error as exc:
        logger.warning("Schema migration warning: %s", exc)
    finally:
        conn.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
