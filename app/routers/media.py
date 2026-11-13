from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, asc, select, or_, func, extract
from sqlalchemy.orm import Session, joinedload

from app.database import get_db, THUMBNAILS_DIR
from app.models import MediaItem, WindowsVolume, MediaGroup, MediaTag, MediaEmbedding
from app.storage_ops import StorageAutomationService

router = APIRouter(prefix="/media", tags=["Media"])
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def to_int(val: Optional[str]) -> Optional[int]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def to_bool(val: Optional[str]) -> bool:
    if not val:
        return False
    return str(val).lower() in ("true", "1", "yes", "on")


def render_template(request: Request, name: str, context: dict = None):
    ctx = context or {}
    ctx["request"] = request
    try:
        return templates.TemplateResponse(request=request, name=name, context=ctx)
    except TypeError:
        return templates.TemplateResponse(name, ctx)


def apply_media_filters(
    stmt,
    db: Session,
    q: Optional[str] = None,
    media_type: str = "all",
    volume_id: Optional[int] = None,
    folder_prefix: Optional[str] = None,
    group_id: Optional[int] = None,
    tag: Optional[str] = None,
    year: Optional[int] = None,
    anomaly_only: bool = False,
    blurry_only: bool = False,
    duplicates_only: bool = False,
    untagged_only: bool = False,
    ungrouped_only: bool = False,
    trashed_only: bool = False,
):
    if trashed_only:
        stmt = stmt.where(MediaItem.is_trashed == True)
    else:
        stmt = stmt.where(MediaItem.is_trashed == False)

    if media_type in ("image", "video"):
        stmt = stmt.where(MediaItem.media_type == media_type)
    if volume_id:
        stmt = stmt.where(MediaItem.volume_id == volume_id)
    if folder_prefix and folder_prefix.strip():
        stmt = stmt.where(MediaItem.relative_path.ilike(f"{folder_prefix.strip()}%"))
    if group_id:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(group_id)
        stmt = stmt.where(MediaItem.group_id.in_(subtree_ids))
    if ungrouped_only:
        stmt = stmt.where(MediaItem.group_id.is_(None))
    if tag and tag.strip():
        stmt = stmt.join(MediaItem.tags).where(MediaTag.tag == tag.strip())
    if untagged_only:
        stmt = stmt.outerjoin(MediaItem.tags).where(MediaTag.id.is_(None))
    if year:
        stmt = stmt.where(extract("year", MediaItem.effective_date) == year)
    if anomaly_only:
        stmt = stmt.where(MediaItem.has_date_anomaly == True)
    if blurry_only:
        stmt = stmt.where(MediaItem.is_blurry == True)
    if duplicates_only:
        dup_hashes_subq = (
            select(MediaItem.file_hash)
            .where(MediaItem.is_trashed == False)
            .group_by(MediaItem.file_hash)
            .having(func.count(MediaItem.id) > 1)
        )
        stmt = stmt.where(MediaItem.file_hash.in_(dup_hashes_subq))
    if q and q.strip():
        clean_q = q.strip()
        stmt = stmt.outerjoin(MediaItem.tags).where(
            or_(
                MediaItem.file_name.ilike(f"%{clean_q}%"),
                MediaItem.relative_path.ilike(f"%{clean_q}%"),
                MediaTag.tag.ilike(f"%{clean_q}%"),
            )
        )
    return stmt


def build_sidebar_context(db: Session) -> dict:
    volumes = db.scalars(select(WindowsVolume)).all()
    vol_space = {}
    for v in volumes:
        v.is_online = Path(v.last_known_drive_letter).exists()
        if v.is_online:
            try:
                u = shutil.disk_usage(v.last_known_drive_letter)
                vol_space[v.id] = {
                    "free_gb": u.free / (1024 ** 3),
                    "total_gb": u.total / (1024 ** 3),
                    "used_pct": int((u.used / u.total) * 100) if u.total else 0,
                }
            except Exception:
                vol_space[v.id] = None
        else:
            vol_space[v.id] = None

    vol_counts = dict(
        db.execute(
            select(MediaItem.volume_id, func.count(MediaItem.id))
            .where(MediaItem.is_trashed == False)
            .group_by(MediaItem.volume_id)
        ).all()
    )

    groups = db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False).order_by(MediaGroup.name.asc())).all()
    trashed_groups = db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == True).order_by(MediaGroup.name.asc())).all()

    direct_counts = dict(
        db.execute(
            select(MediaItem.group_id, func.count(MediaItem.id))
            .where(MediaItem.is_trashed == False)
            .group_by(MediaItem.group_id)
        ).all()
    )
    children_map = defaultdict(list)
    for g in groups:
        children_map[g.parent_id].append(g)

    def build_tree(parent_id=None, depth=0):
        if depth > 8:
            return []
        nodes = []
        for g in children_map.get(parent_id, []):
            sub_nodes = build_tree(g.id, depth + 1)
            own = direct_counts.get(g.id, 0)
            total = own + sum(n["total_count"] for n in sub_nodes)
            nodes.append({"group": g, "depth": depth, "own_count": own, "total_count": total, "children": sub_nodes})
        return nodes

    top_tags = db.execute(
        select(MediaTag.tag, func.count(func.distinct(MediaTag.media_id)).label("cnt"))
        .join(MediaItem, MediaItem.id == MediaTag.media_id)
        .where(MediaItem.is_trashed == False)
        .group_by(MediaTag.tag)
        .order_by(desc("cnt"))
        .limit(35)
    ).all()

    years = db.execute(
        select(extract("year", MediaItem.effective_date).label("yr"), func.count(MediaItem.id))
        .where(MediaItem.is_trashed == False)
        .group_by("yr")
        .order_by(desc("yr"))
    ).all()

    total_all = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == False)) or 0
    total_anomalies = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == False, MediaItem.has_date_anomaly == True)) or 0
    total_dups = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == False, MediaItem.is_primary_duplicate == False)) or 0
    total_blurry = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == False, MediaItem.is_blurry == True)) or 0
    total_trashed = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == True)) or 0
    total_vectors = db.scalar(select(func.count(MediaEmbedding.media_id))) or 0

    return {
        "volumes": volumes,
        "vol_space": vol_space,
        "vol_counts": vol_counts,
        "group_tree": build_tree(None, 0),
        "all_groups": groups,
        "trashed_groups": trashed_groups,
        "top_tags": top_tags,
        "years": [(int(y), c) for y, c in years if y is not None],
        "total_all": total_all,
        "total_anomalies": total_anomalies,
        "total_dups": total_dups,
        "total_blurry": total_blurry,
        "total_trashed": total_trashed,
        "total_vectors": total_vectors,
    }


def build_search_context(request: Request, db: Session) -> dict:
    qp = request.query_params
    q = qp.get("q")
    media_type = qp.get("media_type", "all")
    vol_id_int = to_int(qp.get("volume_id"))
    folder_prefix = qp.get("folder_prefix")
    grp_id_int = to_int(qp.get("group_id"))
    tag = qp.get("tag")
    yr_int = to_int(qp.get("year"))
    anomaly_only = to_bool(qp.get("anomaly_only"))
    blurry_only = to_bool(qp.get("blurry_only"))
    duplicates_only = to_bool(qp.get("duplicates_only"))
    untagged_only = to_bool(qp.get("untagged_only"))
    ungrouped_only = to_bool(qp.get("ungrouped_only"))
    trashed_only = to_bool(qp.get("trashed_only"))
    sort_by = qp.get("sort_by", "date_desc")
    view_mode = qp.get("view_mode", "grid")
    page_int = max(1, to_int(qp.get("page")) or 1)
    page_size_int = min(500, max(10, to_int(qp.get("page_size")) or 120))

    base_stmt = select(MediaItem)
    filtered_stmt = apply_media_filters(
        base_stmt, db, q, media_type, vol_id_int, folder_prefix, grp_id_int, tag, yr_int,
        anomaly_only, blurry_only, duplicates_only, untagged_only, ungrouped_only, trashed_only
    ).distinct()

    count_subq = filtered_stmt.with_only_columns(MediaItem.id, MediaItem.file_size_bytes).subquery()
    set_stats = db.execute(select(func.count(count_subq.c.id), func.sum(count_subq.c.file_size_bytes))).one()
    total_matching = set_stats[0] or 0
    total_size_bytes = set_stats[1] or 0

    stmt = filtered_stmt.options(
        joinedload(MediaItem.volume),
        joinedload(MediaItem.group),
        joinedload(MediaItem.tags),
        joinedload(MediaItem.embedding),
    )

    if duplicates_only:
        stmt = stmt.order_by(asc(MediaItem.file_hash), desc(MediaItem.is_primary_duplicate), asc(MediaItem.effective_date))
    elif sort_by == "date_asc":
        stmt = stmt.order_by(asc(MediaItem.effective_date))
    elif sort_by == "size_desc":
        stmt = stmt.order_by(desc(MediaItem.file_size_bytes))
    elif sort_by == "size_asc":
        stmt = stmt.order_by(asc(MediaItem.file_size_bytes))
    elif sort_by == "name_asc":
        stmt = stmt.order_by(asc(MediaItem.file_name))
    elif sort_by == "blur_asc":
        stmt = stmt.order_by(asc(MediaItem.blur_score))
    else:
        stmt = stmt.order_by(desc(MediaItem.effective_date))

    offset = (page_int - 1) * page_size_int
    items = db.scalars(stmt.offset(offset).limit(page_size_int)).unique().all()
    total_pages = max(1, (total_matching + page_size_int - 1) // page_size_int)

    grouped = defaultdict(list)
    for it in items:
        if duplicates_only:
            cluster_key = f"🔁 Klaster duplikatu (SHA-256: {it.file_hash[:12]}...)"
            grouped[cluster_key].append(it)
        else:
            day_str = it.effective_date.strftime("%Y-%m-%d")
            grouped[day_str].append(it)

    active_group = db.get(MediaGroup, grp_id_int) if grp_id_int else None
    all_groups = db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False).order_by(MediaGroup.name.asc())).all()
    trashed_groups = db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == True).order_by(MediaGroup.name.asc())).all()
    all_volumes = db.scalars(select(WindowsVolume)).all()

    subfolders = []
    if vol_id_int and not trashed_only:
        paths = db.scalars(
            select(MediaItem.relative_path).where(MediaItem.volume_id == vol_id_int, MediaItem.is_trashed == False)
        ).all()
        folder_counts = defaultdict(int)
        for p in paths:
            parent_str = Path(p).parent.as_posix()
            if parent_str != ".":
                top_part = parent_str.split("/")[0]
                folder_counts[top_part] += 1
        subfolders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:25]

    return {
        "timeline": dict(grouped),
        "items_flat": items,
        "total_matching": total_matching,
        "total_size_mb": total_size_bytes / (1024 * 1024),
        "page": page_int,
        "page_size": page_size_int,
        "total_pages": total_pages,
        "view_mode": view_mode,
        "sort_by": sort_by,
        "active_group": active_group,
        "active_tag": tag,
        "active_volume_id": vol_id_int,
        "active_folder": folder_prefix,
        "trashed_only": trashed_only,
        "duplicates_only": duplicates_only,
        "trashed_groups": trashed_groups,
        "subfolders": subfolders,
        "all_groups": all_groups,
        "all_volumes": all_volumes,
        "similarity_scores": {},
    }


@router.get("/sidebar")
def get_sidebar_data(request: Request, db: Session = Depends(get_db)):
    return render_template(request, "partials/sidebar.html", build_sidebar_context(db))


@router.get("/search")
def search_media(request: Request, db: Session = Depends(get_db)):
    return render_template(request, "partials/media_grid.html", build_search_context(request, db))


@router.get("/thumbnails/{filename}")
def serve_thumbnail(filename: str, db: Session = Depends(get_db)):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    p = THUMBNAILS_DIR / filename
    if not p.exists() or p.stat().st_size < 100:
        media_id = filename.rsplit(".", 1)[0]
        it = db.scalars(select(MediaItem).options(joinedload(MediaItem.volume)).where(MediaItem.id == media_id)).unique().first()
        if it and it.volume:
            src_p = Path(it.volume.last_known_drive_letter) / it.relative_path
            if src_p.exists():
                try:
                    from app.scanner import ThumbnailService, IMAGE_EXTS
                    _, w, h, _ = ThumbnailService.generate(src_p, it.id, src_p.suffix.lower() in IMAGE_EXTS)
                    if w and h:
                        it.width = w
                        it.height = h
                        it.is_corrupted = False
                        db.commit()
                except Exception:
                    pass

    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="image/webp", headers={"Cache-Control": "no-cache"})


@router.get("/{media_id}/raw")
def serve_raw_media(media_id: str, db: Session = Depends(get_db)):
    import io
    import subprocess
    from fastapi.responses import StreamingResponse
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    it = db.scalars(select(MediaItem).options(joinedload(MediaItem.volume)).where(MediaItem.id == media_id)).unique().first()
    if not it:
        raise HTTPException(status_code=404)

    if it.volume:
        full_path = Path(it.volume.last_known_drive_letter) / it.relative_path
        if full_path.exists() and full_path.is_file():
            ext = full_path.suffix.lower()

            # 1. Konwersja obrazow HEIC/HEIF/TIFF w locie do JPEG
            if ext in (".heic", ".heif", ".tiff", ".tif"):
                try:
                    with Image.open(full_path) as img:
                        img = ImageOps.exif_transpose(img)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=90)
                        buf.seek(0)
                        return StreamingResponse(buf, media_type="image/jpeg")
                except Exception:
                    pass

            # 2. Transkodowanie wideo .MOV / .MKV / .AVI / .WMV do H.264 MP4 (obraz + dzwiek w przegladarce)
            if ext in (".mov", ".mkv", ".avi", ".wmv", ".mts", ".m2ts", ".flv", ".3gp"):
                cached_mp4 = THUMBNAILS_DIR / f"{it.id}_stream.mp4"
                if not cached_mp4.exists() or cached_mp4.stat().st_size < 1000:
                    try:
                        import imageio_ffmpeg
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                        cmd = [
                            ffmpeg_exe, "-y", "-i", str(full_path.resolve()),
                            "-vf", "scale='min(1280,iw)':-2",
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "128k",
                            "-movflags", "+faststart",
                            str(cached_mp4.resolve()),
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
                    except Exception:
                        pass
                if cached_mp4.exists() and cached_mp4.stat().st_size > 1000:
                    return FileResponse(cached_mp4, media_type="video/mp4", filename=f"{full_path.stem}.mp4")

            return FileResponse(full_path, media_type=it.mime_type, filename=it.file_name)

    thumb_path = THUMBNAILS_DIR / it.thumbnail_relative_path
    if thumb_path.exists():
        return FileResponse(thumb_path, media_type="image/webp")
    raise HTTPException(status_code=404, detail="File offline and no thumbnail cached")


@router.get("/{media_id}/detail")
def get_media_detail(request: Request, media_id: str, db: Session = Depends(get_db)):
    stmt = (
        select(MediaItem)
        .options(
            joinedload(MediaItem.volume),
            joinedload(MediaItem.group),
            joinedload(MediaItem.tags),
            joinedload(MediaItem.embedding),
        )
        .where(MediaItem.id == media_id)
    )
    it = db.scalars(stmt).unique().first()
    if not it:
        raise HTTPException(status_code=404)

    all_groups = db.scalars(select(MediaGroup).order_by(MediaGroup.name.asc())).all()
    duplicate_copies = list(
        db.scalars(
            select(MediaItem)
            .options(joinedload(MediaItem.volume), joinedload(MediaItem.group))
            .where(MediaItem.file_hash == it.file_hash, MediaItem.is_trashed == False)
            .order_by(desc(MediaItem.is_primary_duplicate), asc(MediaItem.effective_date))
        ).unique().all()
    )
    return render_template(
        request,
        "partials/modal_detail.html",
        {"item": it, "all_groups": all_groups, "duplicate_copies": duplicate_copies},
    )