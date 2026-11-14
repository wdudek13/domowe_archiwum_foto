from contextlib import asynccontextmanager
import logging
import sys
import traceback

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from app.backup import BackupManager
from app.database import BASE_DIR, engine, Base, get_db, THUMBNAILS_DIR, ensure_schema_migrations
from app.operation_log import OperationLogger
from app.routers import media, actions, semantic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DomoweArchiwumFoto")


def run_safe_migrations() -> None:
    ensure_schema_migrations()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Uruchamianie silnika Domowe Archiwum Foto...")
    Base.metadata.create_all(bind=engine)
    run_safe_migrations()
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    OperationLogger.log("APP_STARTUP", "Uruchomiono serwer Domowe Archiwum Foto.", status="INFO")
    logger.info("Gotowe! Serwer Domowe Archiwum Foto nasłuchuje.")
    yield
    try:
        BackupManager().create_backup("shutdown")
        OperationLogger.log("APP_SHUTDOWN", "Zatrzymano serwer i utworzono kopię zapasową SQLite.", status="INFO")
    except Exception as exc:
        logger.warning("Shutdown backup warning: %s", exc)


app = FastAPI(title="Domowe Archiwum Foto", lifespan=lifespan, debug=True)


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


@app.get("/przewodnik", response_class=HTMLResponse)
def user_guide_page():
    guide_path = BASE_DIR / "przewodnik_uzytkownika.html"
    if not guide_path.exists():
        raise HTTPException(status_code=404, detail="Brak pliku przewodnik_uzytkownika.html")
    return FileResponse(guide_path, media_type="text/html; charset=utf-8")


@app.get("/dokumentacja", response_class=HTMLResponse)
def technical_docs_page():
    docs_path = BASE_DIR / "dokumentacja.html"
    if not docs_path.exists():
        raise HTTPException(status_code=404, detail="Brak pliku dokumentacja.html")
    return FileResponse(docs_path, media_type="text/html; charset=utf-8")
