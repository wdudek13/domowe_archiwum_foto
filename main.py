from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sqlite3
import sys
import traceback

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db, THUMBNAILS_DIR, DB_PATH
from app.routers import media, actions, semantic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HydraCore")


def run_safe_migrations():
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH.as_posix(), timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(media_groups);")
        g_cols = {r[1] for r in cur.fetchall()}
        if g_cols and "parent_id" not in g_cols:
            cur.execute("ALTER TABLE media_groups ADD COLUMN parent_id INTEGER;")
        if g_cols and "target_volume_id" not in g_cols:
            cur.execute("ALTER TABLE media_groups ADD COLUMN target_volume_id INTEGER;")
        if g_cols and "is_trashed" not in g_cols:
            cur.execute("ALTER TABLE media_groups ADD COLUMN is_trashed BOOLEAN DEFAULT 0 NOT NULL;")

        cur.execute("PRAGMA table_info(media_items);")
        m_cols = {r[1] for r in cur.fetchall()}
        if m_cols and "updated_at" not in m_cols:
            cur.execute("ALTER TABLE media_items ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP;")
        if m_cols and "is_trashed" not in m_cols:
            cur.execute("ALTER TABLE media_items ADD COLUMN is_trashed BOOLEAN DEFAULT 0 NOT NULL;")
        conn.commit()
    except Exception as e:
        logger.warning("Migration warning: %s", e)
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Uruchamianie silnika Hydra Media Vault...")
    Base.metadata.create_all(bind=engine)
    run_safe_migrations()
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Gotowe! Serwer nasluchuje.")
    yield


app = FastAPI(title="Hydra Media Vault", lifespan=lifespan, debug=True)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err = traceback.format_exc()
    logger.error("Blad na %s:\n%s", request.url.path, err)
    return HTMLResponse(
        status_code=200,
        content=f"<pre style='color:#f87171; background:#0b0f19; padding:1.5rem; border:1px solid #ef4444; border-radius:8px; font-size:12px; overflow:auto;'>{err}</pre>",
    )


app.include_router(media.router)
app.include_router(actions.router)
app.include_router(semantic.router)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    ctx = {}
    ctx.update(media.build_sidebar_context(db))
    ctx.update(media.build_search_context(request, db))
    return media.render_template(request, "index.html", ctx)