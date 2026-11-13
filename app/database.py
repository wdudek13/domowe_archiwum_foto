from pathlib import Path
from typing import Generator
import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
BACKUP_DIR = DATA_DIR / "backups"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"

for d in (DB_DIR, BACKUP_DIR, THUMBNAILS_DIR):
    d.mkdir(parents=True, exist_ok=True)

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


def ensure_schema_migrations() -> None:
    """Safely adds new columns to existing SQLite tables without data loss."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH.as_posix())
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(media_groups);")
        cols = {row[1] for row in cur.fetchall()}
        if cols and "parent_id" not in cols:
            cur.execute("ALTER TABLE media_groups ADD COLUMN parent_id INTEGER REFERENCES media_groups(id) ON DELETE CASCADE;")
            conn.commit()
    finally:
        conn.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
