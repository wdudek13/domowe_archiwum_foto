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

            # Drop legacy table-level UNIQUE(name) constraint so hierarchical sub-albums
            # (e.g. YYYY / MM / DD under different parents) can share names safely.
            mg_ddl_row = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='media_groups';"
            ).fetchone()
            mg_ddl = (mg_ddl_row[0] if mg_ddl_row and mg_ddl_row[0] else "")
            if "UNIQUE" in mg_ddl.upper():
                cur.execute("PRAGMA foreign_keys = OFF;")
                cur.execute(
                    """
                    CREATE TABLE media_groups_migrated (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        name VARCHAR(128) NOT NULL,
                        description TEXT,
                        parent_id INTEGER REFERENCES media_groups(id) ON DELETE CASCADE,
                        target_volume_id INTEGER REFERENCES windows_volumes(id) ON DELETE SET NULL,
                        is_trashed BOOLEAN DEFAULT 0 NOT NULL,
                        created_at DATETIME NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO media_groups_migrated
                        (id, name, description, parent_id, target_volume_id, is_trashed, created_at)
                    SELECT
                        id, name, description, parent_id, target_volume_id,
                        COALESCE(is_trashed, 0), created_at
                    FROM media_groups;
                    """
                )
                cur.execute("DROP TABLE media_groups;")
                cur.execute("ALTER TABLE media_groups_migrated RENAME TO media_groups;")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_media_groups_parent_id ON media_groups (parent_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_media_groups_is_trashed ON media_groups (is_trashed);")
                cur.execute("PRAGMA foreign_keys = ON;")

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
            if "duplicate_role" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN duplicate_role VARCHAR(32) DEFAULT 'original' NOT NULL;")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_media_items_duplicate_role ON media_items (duplicate_role);")
                needed_cols = {"file_hash", "volume_id", "is_primary_duplicate", "is_backup_copy", "effective_date", "relative_path", "is_trashed"}
                if needed_cols.issubset(m_cols):
                    cur.execute(
                        """
                        WITH vol_counts AS (
                            SELECT
                                id, file_hash, volume_id, is_backup_copy, effective_date, relative_path,
                                SUM(CASE WHEN is_backup_copy = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY file_hash, volume_id) AS same_vol_cnt
                            FROM media_items
                            WHERE is_trashed = 0
                        ),
                        orig_ranked AS (
                            SELECT
                                id, file_hash, volume_id, is_backup_copy, effective_date, relative_path,
                                ROW_NUMBER() OVER (
                                    PARTITION BY file_hash
                                    ORDER BY is_backup_copy ASC, same_vol_cnt DESC, effective_date ASC, relative_path ASC, id ASC
                                ) AS orig_rn,
                                FIRST_VALUE(volume_id) OVER (
                                    PARTITION BY file_hash
                                    ORDER BY is_backup_copy ASC, same_vol_cnt DESC, effective_date ASC, relative_path ASC, id ASC
                                ) AS orig_vol_id
                            FROM vol_counts
                        ),
                        role_ranked AS (
                            SELECT
                                id, volume_id, orig_vol_id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY file_hash
                                    ORDER BY
                                        (CASE WHEN orig_rn = 1 THEN 0 WHEN volume_id != orig_vol_id THEN 1 ELSE 2 END) ASC,
                                        is_backup_copy DESC,
                                        effective_date ASC,
                                        relative_path ASC,
                                        id ASC
                                ) AS final_rn
                            FROM orig_ranked
                        ),
                        computed AS (
                            SELECT
                                id,
                                CASE WHEN final_rn = 1 THEN 1 ELSE 0 END AS new_primary,
                                CASE
                                    WHEN final_rn = 1 THEN 'original'
                                    WHEN final_rn = 2 AND volume_id != orig_vol_id THEN 'backup_green'
                                    WHEN final_rn = 2 THEN 'backup_orange'
                                    ELSE 'duplicate'
                                END AS new_role
                            FROM role_ranked
                        )
                        UPDATE media_items
                        SET
                            is_primary_duplicate = computed.new_primary,
                            duplicate_role = computed.new_role
                        FROM computed
                        WHERE media_items.id = computed.id;
                        """
                    )
            if "verified_sha256" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN verified_sha256 VARCHAR(64);")
            if "original_relative_path" not in m_cols:
                cur.execute("ALTER TABLE media_items ADD COLUMN original_relative_path VARCHAR(1024);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_orig_rel_path ON media_items (original_relative_path);"
            )

        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS media_tags_fts USING fts5(media_id UNINDEXED, tag, source_model UNINDEXED);"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_duplicate_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deleted_item_id VARCHAR(36),
                original_item_id VARCHAR(36),
                file_hash VARCHAR(64) NOT NULL,
                volume_id INTEGER,
                volume_label VARCHAR(128) DEFAULT '' NOT NULL,
                drive_letter VARCHAR(16) DEFAULT '' NOT NULL,
                relative_path TEXT NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                deleted_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
            );
            """
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_del_dup_hist_unique ON deleted_duplicate_history (file_hash, volume_id, relative_path);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ix_deleted_duplicate_history_file_hash ON deleted_duplicate_history (file_hash);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ix_deleted_duplicate_history_file_name ON deleted_duplicate_history (file_name);"
        )
        if m_cols and {"is_trashed", "duplicate_role", "is_backup_copy", "file_hash"}.issubset(m_cols):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_role_bkp_trashed ON media_items (is_trashed, duplicate_role, is_backup_copy, file_hash);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_browse_date ON media_items (is_trashed, is_backup_copy, effective_date DESC, file_size_bytes);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_updated_at ON media_items (updated_at);"
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_media_sidebar_aggs ON media_items (
                    is_trashed, is_backup_copy, volume_id, group_id, effective_date,
                    has_date_anomaly, is_blurry, is_corrupted, blur_score
                );
                """
            )
            cur.execute(
                """
                UPDATE media_items SET is_backup_copy = 1
                WHERE is_backup_copy = 0
                  AND (LOWER(relative_path) LIKE 'daf_backup/%' OR LOWER(relative_path) LIKE 'daf_backup\\%'
                       OR LOWER(relative_path) LIKE 'backup/%' OR LOWER(relative_path) LIKE 'backup\\%');
                """
            )
        cur.execute("PRAGMA table_info(media_tags);")
        t_cols = {row[1] for row in cur.fetchall()}
        if {"source_model", "tag", "media_id"}.issubset(t_cols):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_tags_src_tag_mid ON media_tags (source_model, tag, media_id);"
            )
        # Clear legacy false-positive date anomalies caused solely by missing EXIF or copy timestamps on EXIF files
        if m_cols:
            cur.execute(
                """
                UPDATE media_items
                SET has_date_anomaly = 0, date_anomaly_reason = NULL
                WHERE date_anomaly_reason LIKE 'Brak metadanych EXIF%'
                   OR date_anomaly_reason LIKE 'Brak EXIF%';
                """
            )
            if "exif_date" in m_cols:
                cur.execute(
                    """
                    UPDATE media_items
                    SET effective_date = exif_date, has_date_anomaly = 0, date_anomaly_reason = NULL
                    WHERE exif_date IS NOT NULL
                      AND has_date_anomaly = 1
                      AND (
                          date_anomaly_reason LIKE 'Niezgodność dat: EXIF%'
                          OR date_anomaly_reason LIKE 'Niezgodność dat: data pliku%'
                      );
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
