from collections import defaultdict
import io
import logging
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, asc, select, or_, func, extract, and_
from sqlalchemy.orm import Session, joinedload, aliased, load_only

from app.database import get_db, THUMBNAILS_DIR
from app.models import MediaItem, WindowsVolume, MediaGroup, MediaTag, MediaEmbedding, DeletedDuplicateHistory
from app.operation_log import ProcessManager
from app.scanner import MetadataDateResolver
from app.storage_ops import StorageAutomationService

logger = logging.getLogger("MediaRouter")
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


def render_template(request: Request, name: str, context: Optional[dict] = None):
    ctx = context or {}
    ctx["request"] = request
    try:
        return templates.TemplateResponse(request=request, name=name, context=ctx)
    except TypeError:
        return templates.TemplateResponse(name, ctx)


def get_unbacked_primary_ids_subquery():
    """Returns a SQL subquery of primary MediaItem IDs that have no copy on a different device."""
    backed_hashes = select(MediaItem.file_hash).where(
        MediaItem.is_trashed == False,
        MediaItem.duplicate_role == "backup_green",
    )
    return (
        select(MediaItem.id)
        .where(
            MediaItem.is_trashed == False,
            MediaItem.is_primary_duplicate == True,
            ~MediaItem.file_hash.in_(backed_hashes),
        )
    )


def compute_backup_and_dup_sidebar_stats(db: Session) -> dict:
    """Computes sidebar counts reconciled 1-to-1 with the Backup & Duplicates modal summary."""
    rows = db.execute(
        select(MediaItem.duplicate_role, MediaItem.is_backup_copy, func.count(MediaItem.id))
        .where(MediaItem.is_trashed == False)
        .group_by(MediaItem.duplicate_role, MediaItem.is_backup_copy)
    ).all()
    counts = defaultdict(int)
    for role, is_bkp, cnt in rows:
        counts[(role or "original", bool(is_bkp))] += int(cnt or 0)

    total_unique = counts[("original", False)] + counts[("original", True)]
    green_in_vault = counts[("backup_green", True)]
    green_outside_vault = counts[("backup_green", False)]
    green_backups = green_in_vault + green_outside_vault
    orange_backups = counts[("backup_orange", False)] + counts[("backup_orange", True)]
    missing_backups = max(0, total_unique - green_backups - orange_backups)
    backups_to_relocate = green_outside_vault + orange_backups
    redundant_duplicates = counts[("duplicate", False)] + counts[("duplicate", True)]
    total_all_gallery = sum(cnt for (_role, is_bkp), cnt in counts.items() if not is_bkp)
    total_dups_gallery = green_outside_vault + counts[("backup_orange", False)] + counts[("duplicate", False)]
    return {
        "total_unique": total_unique,
        "missing_backups": missing_backups,
        "green_backups": green_backups,
        "green_in_vault": green_in_vault,
        "green_outside_vault": green_outside_vault,
        "orange_backups": orange_backups,
        "backups_to_relocate": backups_to_relocate,
        "unbacked_cross_device": missing_backups + orange_backups,
        "redundant_duplicates": redundant_duplicates,
        "total_all_gallery": total_all_gallery,
        "total_dups_gallery": total_dups_gallery,
    }


def _build_cross_vault_backup_map(db: Session, items: List[MediaItem]) -> dict:
    """
    For each item, inspects all non-trashed daf_backup/ copies sharing its file_hash on a DIFFERENT volume.
    Returns {item.id: 'Label (Letter), ...'} so ORYGINAL badges turn green when safely backed up cross-drive.
    """
    if not items:
        return {}
    hashes = list({it.file_hash for it in items if it.file_hash})
    if not hashes:
        return {}
    vault_rows = db.execute(
        select(
            MediaItem.file_hash,
            MediaItem.volume_id,
            WindowsVolume.volume_label,
            WindowsVolume.last_known_drive_letter,
        )
        .join(WindowsVolume, WindowsVolume.id == MediaItem.volume_id)
        .where(
            MediaItem.file_hash.in_(hashes),
            MediaItem.is_trashed == False,
            MediaItem.is_backup_copy == True,
        )
    ).all()
    by_hash = defaultdict(list)
    for fh, vid, vlabel, vletter in vault_rows:
        by_hash[fh].append((vid, f"{vlabel} ({vletter})"))
    result = {}
    for it in items:
        other_vols = sorted({lbl for vid, lbl in by_hash.get(it.file_hash, []) if vid != it.volume_id})
        if other_vols:
            result[it.id] = ", ".join(other_vols)
    return result


def _build_deleted_dup_history_map(db: Session, items: List[MediaItem], q: Optional[str] = None) -> tuple[dict, dict]:
    """Returns (counts_by_hash, query_matches_by_hash) for DeletedDuplicateHistory entries matching displayed items."""
    if not items:
        return {}, {}
    hashes = list({it.file_hash for it in items if it.file_hash})
    if not hashes:
        return {}, {}
    counts_by_hash = dict(
        db.execute(
            select(DeletedDuplicateHistory.file_hash, func.count(DeletedDuplicateHistory.id))
            .where(DeletedDuplicateHistory.file_hash.in_(hashes))
            .group_by(DeletedDuplicateHistory.file_hash)
        ).all()
    )
    matches_by_hash = defaultdict(list)
    if q and q.strip():
        clean_q = q.strip()
        like_pattern = f"%{clean_q}%"
        rows = db.scalars(
            select(DeletedDuplicateHistory).where(
                DeletedDuplicateHistory.file_hash.in_(hashes),
                or_(
                    DeletedDuplicateHistory.file_name.ilike(like_pattern),
                    DeletedDuplicateHistory.relative_path.ilike(like_pattern),
                    DeletedDuplicateHistory.volume_label.ilike(like_pattern),
                ),
            )
        ).all()
        for row in rows:
            lst = matches_by_hash[row.file_hash]
            if len(lst) < 3:
                lst.append(f"{row.volume_label} ({row.drive_letter}{row.relative_path})")
    return counts_by_hash, dict(matches_by_hash)


def _anomaly_type_clause(anomaly_type: Optional[str]):
    at = (anomaly_type or "all").strip().lower()
    if at == "folder":
        return MediaItem.date_anomaly_reason.ilike("%folder%")
    if at == "filename":
        return MediaItem.date_anomaly_reason.ilike("%nazwa pliku%")
    if at == "copy_ctime":
        return or_(
            MediaItem.date_anomaly_reason.ilike("%zmodyfikowane przy kopiowaniu%"),
            MediaItem.date_anomaly_reason.ilike("%mtime%"),
        )
    if at == "exif":
        return MediaItem.date_anomaly_reason.ilike("%EXIF%")
    if at == "other":
        return or_(
            MediaItem.date_anomaly_reason.is_(None),
            and_(
                ~MediaItem.date_anomaly_reason.ilike("%folder%"),
                ~MediaItem.date_anomaly_reason.ilike("%nazwa pliku%"),
                ~MediaItem.date_anomaly_reason.ilike("%zmodyfikowane przy kopiowaniu%"),
                ~MediaItem.date_anomaly_reason.ilike("%EXIF%"),
            ),
        )
    return None


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
    unbacked_only: bool = False,
    unvectored_only: bool = False,
    unanalyzed_quality_only: bool = False,
    quality_defect: Optional[str] = None,
    redundant_clusters_only: bool = False,
    backups_to_move_only: bool = False,
    backup_filter: str = "none",
    backup_types: Optional[List[str]] = None,
    backup_mode: str = "hide",
    anomaly_type: str = "all",
):
    if trashed_only:
        stmt = stmt.where(MediaItem.is_trashed == True)
    else:
        stmt = stmt.where(MediaItem.is_trashed == False)

    # Backup visibility filtering (default hides ALL backup types across gallery & album views)
    eff_bkp_filter = (backup_filter or "none").strip()
    eff_bkp_types = [t.strip() for t in (backup_types or []) if t and t.strip()]
    eff_bkp_mode = (backup_mode or "hide").strip()

    if eff_bkp_filter == "all_with_backups":
        eff_bkp_mode = "include"
        eff_bkp_types = ["daf_backup", "green", "orange"]
    elif eff_bkp_filter == "only_backups":
        eff_bkp_mode = "only"
        eff_bkp_types = ["daf_backup", "green", "orange"]
    elif eff_bkp_filter == "only_daf_backup":
        eff_bkp_mode = "only"
        eff_bkp_types = ["daf_backup"]
    elif eff_bkp_filter == "only_green":
        eff_bkp_mode = "only"
        eff_bkp_types = ["green"]
    elif eff_bkp_filter == "only_orange":
        eff_bkp_mode = "only"
        eff_bkp_types = ["orange"]
    elif eff_bkp_types and eff_bkp_mode == "hide":
        eff_bkp_mode = "include"

    type_clauses = []
    if "daf_backup" in eff_bkp_types:
        type_clauses.append(MediaItem.is_backup_copy == True)
    if "green" in eff_bkp_types:
        type_clauses.append(and_(MediaItem.is_backup_copy == False, MediaItem.duplicate_role == "backup_green"))
    if "orange" in eff_bkp_types:
        type_clauses.append(and_(MediaItem.is_backup_copy == False, MediaItem.duplicate_role == "backup_orange"))

    is_not_any_backup = and_(
        MediaItem.is_backup_copy == False,
        MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
    )

    if eff_bkp_mode == "only" and type_clauses:
        stmt = stmt.where(or_(*type_clauses))
    elif eff_bkp_mode == "only":
        stmt = stmt.where(
            or_(MediaItem.is_backup_copy == True, MediaItem.duplicate_role.in_(["backup_green", "backup_orange"]))
        )
    elif eff_bkp_mode == "include" and type_clauses:
        stmt = stmt.where(or_(is_not_any_backup, *type_clauses))
    elif duplicates_only or redundant_clusters_only or backups_to_move_only:
        stmt = stmt.where(MediaItem.is_backup_copy == False)
    else:
        stmt = stmt.where(is_not_any_backup)

    if media_type in ("image", "video"):
        stmt = stmt.where(MediaItem.media_type == media_type)
    if volume_id:
        stmt = stmt.where(MediaItem.volume_id == volume_id)
    if folder_prefix and folder_prefix.strip():
        fp_clean = folder_prefix.strip()
        stmt = stmt.where(
            or_(
                MediaItem.relative_path.ilike(f"{fp_clean}%"),
                MediaItem.original_relative_path.ilike(f"%/{fp_clean}%"),
                MediaItem.original_relative_path.ilike(f"{fp_clean}%"),
            )
        )
    if group_id:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(group_id)
        if eff_bkp_mode in ("include", "only") and type_clauses:
            album_hashes_subq = select(MediaItem.file_hash).where(
                MediaItem.group_id.in_(subtree_ids),
                MediaItem.is_trashed == False,
            )
            if eff_bkp_mode == "only":
                stmt = stmt.where(MediaItem.file_hash.in_(album_hashes_subq))
            else:
                stmt = stmt.where(or_(MediaItem.group_id.in_(subtree_ids), MediaItem.file_hash.in_(album_hashes_subq)))
        else:
            stmt = stmt.where(MediaItem.group_id.in_(subtree_ids))
    if ungrouped_only:
        stmt = stmt.where(MediaItem.group_id.is_(None))
    if quality_defect and quality_defect.strip():
        qdef = quality_defect.strip().lower()
        if qdef in ("all", "any"):
            stmt = stmt.join(MediaItem.tags).where(MediaTag.tag.ilike("do_usuniecia%"))
        else:
            defect_tag = qdef if qdef.startswith("do_usuniecia:") else f"do_usuniecia:{qdef}"
            stmt = stmt.join(MediaItem.tags).where(MediaTag.tag == defect_tag)
    if tag and tag.strip():
        if not quality_defect or not quality_defect.strip():
            stmt = stmt.join(MediaItem.tags).where(MediaTag.tag == tag.strip())
        else:
            tag_alias = aliased(MediaTag)
            stmt = stmt.join(tag_alias, tag_alias.media_id == MediaItem.id).where(tag_alias.tag == tag.strip())
    if untagged_only:
        stmt = stmt.where(~MediaItem.id.in_(select(MediaTag.media_id).distinct()))
    if unvectored_only:
        stmt = stmt.where(~MediaItem.id.in_(select(MediaEmbedding.media_id)))
    if unanalyzed_quality_only:
        stmt = stmt.where(MediaItem.blur_score.is_(None), MediaItem.is_corrupted == False)
    if year:
        stmt = stmt.where(extract("year", MediaItem.effective_date) == year)
    if anomaly_only:
        stmt = stmt.where(MediaItem.has_date_anomaly == True)
        at_clause = _anomaly_type_clause(anomaly_type)
        if at_clause is not None:
            stmt = stmt.where(at_clause)
    if blurry_only:
        stmt = stmt.where(or_(MediaItem.is_blurry == True, MediaItem.is_corrupted == True))
    if duplicates_only:
        dup_hashes_subq = (
            select(MediaItem.file_hash)
            .where(
                MediaItem.is_trashed == False,
                MediaItem.is_backup_copy == False,
            )
            .group_by(MediaItem.file_hash)
            .having(func.count(MediaItem.id) > 1)
        )
        stmt = stmt.where(MediaItem.file_hash.in_(dup_hashes_subq))
    if redundant_clusters_only:
        redundant_hashes_subq = select(MediaItem.file_hash).where(
            MediaItem.is_trashed == False,
            MediaItem.is_backup_copy == False,
            MediaItem.duplicate_role == "duplicate",
        )
        stmt = stmt.where(MediaItem.file_hash.in_(redundant_hashes_subq))
    if backups_to_move_only:
        bkp_move_hashes = select(MediaItem.file_hash).where(
            MediaItem.is_trashed == False,
            MediaItem.is_backup_copy == False,
            MediaItem.duplicate_role.in_(["backup_green", "backup_orange"]),
        )
        stmt = stmt.where(MediaItem.file_hash.in_(bkp_move_hashes))
    if unbacked_only:
        stmt = stmt.where(MediaItem.id.in_(get_unbacked_primary_ids_subquery()))
    if q and q.strip():
        clean_q = q.strip()
        del_hist_hashes = select(DeletedDuplicateHistory.file_hash).where(
            or_(
                DeletedDuplicateHistory.file_name.ilike(f"%{clean_q}%"),
                DeletedDuplicateHistory.relative_path.ilike(f"%{clean_q}%"),
                DeletedDuplicateHistory.volume_label.ilike(f"%{clean_q}%"),
            )
        )
        stmt = stmt.outerjoin(MediaItem.tags).where(
            or_(
                MediaItem.file_name.ilike(f"%{clean_q}%"),
                MediaItem.relative_path.ilike(f"%{clean_q}%"),
                MediaItem.original_relative_path.ilike(f"%{clean_q}%"),
                MediaTag.tag.ilike(f"%{clean_q}%"),
                MediaItem.file_hash.in_(del_hist_hashes),
            )
        )
    return stmt


def build_flat_album_choices(groups: List[MediaGroup]) -> List[dict]:
    """Builds a list of albums with their full hierarchical path for select dropdowns."""
    by_id = {g.id: g for g in groups}
    result = []
    for g in groups:
        parts = []
        curr = g
        depth = 0
        while curr and depth < 10:
            parts.append(curr.name)
            curr = by_id.get(curr.parent_id) if curr.parent_id else None
            depth += 1
        full_label = " / ".join(reversed(parts))
        result.append({"id": g.id, "name": g.name, "full_label": full_label, "parent_id": g.parent_id})
    result.sort(key=lambda x: x["full_label"].lower())
    return result


_SIDEBAR_CACHE: dict = {"stamp": None, "data": None}


def _compute_sidebar_static_aggs(db: Session) -> dict:
    """Computes heavy sidebar counts and tag breakdowns using fast covering-index queries."""
    vol_counts = dict(
        db.execute(
            select(MediaItem.volume_id, func.count(MediaItem.id))
            .where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False)
            .group_by(MediaItem.volume_id)
        ).all()
    )
    direct_counts = dict(
        db.execute(
            select(MediaItem.group_id, func.count(MediaItem.id))
            .where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False)
            .group_by(MediaItem.group_id)
        ).all()
    )
    years = db.execute(
        select(extract("year", MediaItem.effective_date).label("yr"), func.count(MediaItem.id))
        .where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False)
        .group_by("yr")
        .order_by(desc("yr"))
    ).all()

    bkp_stats = compute_backup_and_dup_sidebar_stats(db)
    total_all = bkp_stats["total_all_gallery"]
    total_anomalies = (
        db.scalar(
            select(func.count(MediaItem.id)).where(
                MediaItem.is_trashed == False,
                MediaItem.is_backup_copy == False,
                MediaItem.has_date_anomaly == True,
            )
        )
        or 0
    )
    total_dups = bkp_stats["total_dups_gallery"]
    total_redundant_dups = bkp_stats["redundant_duplicates"]
    total_blurry = (
        db.scalar(
            select(func.count(MediaItem.id)).where(
                MediaItem.is_trashed == False,
                MediaItem.is_backup_copy == False,
                or_(MediaItem.is_blurry == True, MediaItem.is_corrupted == True),
            )
        )
        or 0
    )
    total_trashed = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == True)) or 0
    total_unanalyzed_quality = (
        db.scalar(
            select(func.count(MediaItem.id)).where(
                MediaItem.is_trashed == False,
                MediaItem.is_backup_copy == False,
                MediaItem.blur_score.is_(None),
                MediaItem.is_corrupted == False,
            )
        )
        or 0
    )

    excl_ids = list(
        db.scalars(
            select(MediaItem.id).where(
                or_(MediaItem.is_trashed == True, MediaItem.is_backup_copy == True)
            )
        ).all()
    )
    if len(excl_ids) <= 500:
        tag_top_stmt = select(MediaTag.tag, func.count(func.distinct(MediaTag.media_id)).label("cnt")).where(
            MediaTag.source_model != "quality_check"
        )
        defect_stmt = select(MediaTag.tag, func.count(func.distinct(MediaTag.media_id))).where(
            MediaTag.source_model == "quality_check"
        )
        vec_stmt = select(func.count(MediaEmbedding.media_id))
        tagged_stmt = select(func.count(func.distinct(MediaTag.media_id)))
        if excl_ids:
            tag_top_stmt = tag_top_stmt.where(~MediaTag.media_id.in_(excl_ids))
            defect_stmt = defect_stmt.where(~MediaTag.media_id.in_(excl_ids))
            vec_stmt = vec_stmt.where(~MediaEmbedding.media_id.in_(excl_ids))
            tagged_stmt = tagged_stmt.where(~MediaTag.media_id.in_(excl_ids))
        top_tags = [
            (t, c)
            for t, c in db.execute(tag_top_stmt.group_by(MediaTag.tag).order_by(desc("cnt")).limit(35)).all()
        ]
        defect_counts_raw = dict(db.execute(defect_stmt.group_by(MediaTag.tag)).all())
        total_vectors = db.scalar(vec_stmt) or 0
        tagged_active_count = db.scalar(tagged_stmt) or 0
        total_untagged = max(0, total_all - tagged_active_count)
        total_unvectored = max(0, total_all - total_vectors)
    else:
        top_tags = [
            (t, c)
            for t, c in db.execute(
                select(MediaTag.tag, func.count(func.distinct(MediaTag.media_id)).label("cnt"))
                .join(MediaItem, MediaItem.id == MediaTag.media_id)
                .where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False, MediaTag.source_model != "quality_check")
                .group_by(MediaTag.tag)
                .order_by(desc("cnt"))
                .limit(35)
            ).all()
        ]
        defect_counts_raw = dict(
            db.execute(
                select(MediaTag.tag, func.count(func.distinct(MediaTag.media_id)))
                .join(MediaItem, MediaItem.id == MediaTag.media_id)
                .where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False, MediaTag.source_model == "quality_check")
                .group_by(MediaTag.tag)
            ).all()
        )
        total_vectors = (
            db.scalar(
                select(func.count(MediaEmbedding.media_id))
                .join(MediaItem, MediaItem.id == MediaEmbedding.media_id)
                .where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False)
            )
            or 0
        )
        total_untagged = (
            db.scalar(
                select(func.count(MediaItem.id)).where(
                    MediaItem.is_trashed == False,
                    MediaItem.is_backup_copy == False,
                    ~MediaItem.id.in_(select(MediaTag.media_id).distinct()),
                )
            )
            or 0
        )
        total_unvectored = (
            db.scalar(
                select(func.count(MediaItem.id)).where(
                    MediaItem.is_trashed == False,
                    MediaItem.is_backup_copy == False,
                    ~MediaItem.id.in_(select(MediaEmbedding.media_id)),
                )
            )
            or 0
        )

    total_defects_all = defect_counts_raw.get("do_usuniecia", 0) or total_blurry
    defect_counts = {
        "all": total_defects_all,
        "corrupted": defect_counts_raw.get("do_usuniecia:uszkodzone", 0),
        "blurry": defect_counts_raw.get("do_usuniecia:rozmyte", 0),
        "dark": defect_counts_raw.get("do_usuniecia:niedoswietlone", 0),
        "bright": defect_counts_raw.get("do_usuniecia:przeswietlone", 0),
        "detail": defect_counts_raw.get("do_usuniecia:brak_detali", 0),
        "lowres": defect_counts_raw.get("do_usuniecia:niska_rozdzielczosc", 0),
    }

    return {
        "vol_counts": vol_counts,
        "direct_counts": direct_counts,
        "top_tags": top_tags,
        "years": [(int(y), c) for y, c in years if y is not None],
        "total_all": total_all,
        "total_anomalies": total_anomalies,
        "total_dups": total_dups,
        "total_redundant_dups": total_redundant_dups,
        "total_backups_to_relocate": bkp_stats["backups_to_relocate"],
        "bkp_stats": bkp_stats,
        "total_blurry": total_blurry,
        "total_defects_all": total_defects_all,
        "defect_counts": defect_counts,
        "total_trashed": total_trashed,
        "total_vectors": total_vectors,
        "total_untagged": total_untagged,
        "total_unvectored": total_unvectored,
        "total_unanalyzed_quality": total_unanalyzed_quality,
        "total_unbacked": bkp_stats["unbacked_cross_device"],
    }


def build_sidebar_context(db: Session) -> dict:
    svc = StorageAutomationService(db)
    volumes = svc.refresh_volumes_online_status()
    vol_space = svc.get_volume_space_map(volumes)

    groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False).order_by(MediaGroup.name.asc())).all())
    trashed_groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == True).order_by(MediaGroup.name.asc())).all())

    bind_key = str(db.get_bind().url) if db.get_bind() is not None else ""
    stamp = (
        bind_key,
        db.scalar(select(func.count(MediaItem.id))) or 0,
        str(db.scalar(select(func.max(MediaItem.updated_at))) or ""),
        db.scalar(select(func.max(MediaTag.id))) or 0,
        db.scalar(select(func.count(MediaEmbedding.media_id))) or 0,
        db.scalar(select(func.count(MediaGroup.id))) or 0,
        db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_primary_duplicate == False)) or 0,
    )
    if _SIDEBAR_CACHE["stamp"] != stamp or _SIDEBAR_CACHE["data"] is None:
        _SIDEBAR_CACHE["data"] = _compute_sidebar_static_aggs(db)
        _SIDEBAR_CACHE["stamp"] = stamp
    aggs = _SIDEBAR_CACHE["data"]

    direct_counts = aggs["direct_counts"]
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

    jobs_summary = ProcessManager.get_jobs_summary(db)

    ctx = {
        "volumes": volumes,
        "vol_space": vol_space,
        "group_tree": build_tree(None, 0),
        "all_groups": groups,
        "flat_album_choices": build_flat_album_choices(groups),
        "trashed_groups": trashed_groups,
        "jobs_summary": jobs_summary,
    }
    ctx.update(aggs)
    return ctx


def build_search_context(request: Request, db: Session) -> dict:
    StorageAutomationService(db).refresh_volumes_online_status()
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
    redundant_clusters_only = to_bool(qp.get("redundant_clusters_only"))
    backups_to_move_only = to_bool(qp.get("backups_to_move_only"))
    untagged_only = to_bool(qp.get("untagged_only"))
    ungrouped_only = to_bool(qp.get("ungrouped_only"))
    trashed_only = to_bool(qp.get("trashed_only"))
    unbacked_only = to_bool(qp.get("unbacked_only"))
    unvectored_only = to_bool(qp.get("unvectored_only"))
    unanalyzed_quality_only = to_bool(qp.get("unanalyzed_quality_only"))
    quality_defect = qp.get("quality_defect")
    backup_filter = qp.get("backup_filter", "none")
    backup_types = qp.getlist("backup_types") if hasattr(qp, "getlist") else []
    backup_mode = qp.get("backup_mode", "hide")
    anomaly_type = qp.get("anomaly_type", "all")
    sort_by = qp.get("sort_by", "date_desc")
    view_mode = qp.get("view_mode", "grid")
    page_int = max(1, to_int(qp.get("page")) or 1)
    page_size_int = min(500, max(10, to_int(qp.get("page_size")) or 120))

    base_stmt = select(MediaItem)
    filtered_stmt = apply_media_filters(
        base_stmt, db, q, media_type, vol_id_int, folder_prefix, grp_id_int, tag, yr_int,
        anomaly_only, blurry_only, duplicates_only, untagged_only, ungrouped_only, trashed_only, unbacked_only,
        unvectored_only, unanalyzed_quality_only, quality_defect=quality_defect,
        redundant_clusters_only=redundant_clusters_only, backups_to_move_only=backups_to_move_only,
        backup_filter=backup_filter, backup_types=backup_types, backup_mode=backup_mode,
        anomaly_type=anomaly_type,
    )
    joins_tags = bool((q and q.strip()) or (tag and tag.strip()) or (quality_defect and quality_defect.strip()))
    if joins_tags:
        filtered_stmt = filtered_stmt.distinct()
        count_subq = filtered_stmt.with_only_columns(MediaItem.id, MediaItem.file_size_bytes).subquery()
        set_stats = db.execute(select(func.count(count_subq.c.id), func.sum(count_subq.c.file_size_bytes))).one()
        redundant_dup_count_filtered = 0
        if duplicates_only:
            redundant_dup_count_filtered = (
                db.scalar(
                    select(func.count(count_subq.c.id))
                    .join(MediaItem, MediaItem.id == count_subq.c.id)
                    .where(MediaItem.duplicate_role == "duplicate")
                )
                or 0
            )
    else:
        set_stats = db.execute(
            filtered_stmt.with_only_columns(func.count(MediaItem.id), func.sum(MediaItem.file_size_bytes)).order_by(None)
        ).one()
        redundant_dup_count_filtered = 0
        if duplicates_only:
            redundant_dup_count_filtered = (
                db.scalar(
                    filtered_stmt.where(MediaItem.duplicate_role == "duplicate")
                    .with_only_columns(func.count(MediaItem.id))
                    .order_by(None)
                )
                or 0
            )
    total_matching = set_stats[0] or 0
    total_size_bytes = set_stats[1] or 0

    stmt = filtered_stmt.options(
        joinedload(MediaItem.volume),
        joinedload(MediaItem.group),
        joinedload(MediaItem.tags),
        joinedload(MediaItem.embedding).load_only(MediaEmbedding.media_id, MediaEmbedding.model_name, MediaEmbedding.dimensions),
    )

    if duplicates_only:
        stmt = stmt.order_by(asc(MediaItem.file_hash), desc(MediaItem.is_primary_duplicate), asc(MediaItem.duplicate_role), asc(MediaItem.effective_date))
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
    items = list(db.scalars(stmt.offset(offset).limit(page_size_int)).unique().all())
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
    active_volume = db.get(WindowsVolume, vol_id_int) if vol_id_int else None
    all_groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False).order_by(MediaGroup.name.asc())).all())
    trashed_groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == True).order_by(MediaGroup.name.asc())).all())
    all_volumes = list(db.scalars(select(WindowsVolume)).all())

    subfolders = []
    if vol_id_int and not trashed_only:
        paths = db.execute(
            select(MediaItem.relative_path, MediaItem.original_relative_path).where(
                MediaItem.volume_id == vol_id_int,
                MediaItem.is_trashed == False,
                MediaItem.is_backup_copy == False,
            )
        ).all()
        folder_counts = defaultdict(int)
        for p, orig_p in paths:
            eff_p = p
            if orig_p and "/" in orig_p:
                eff_p = orig_p.split("/", 1)[1]
            parent_str = Path(eff_p).parent.as_posix()
            if parent_str != ".":
                top_part = parent_str.split("/")[0]
                if top_part.lower() in ("daf_backup", "backup"):
                    continue
                folder_counts[top_part] += 1
            else:
                folder_counts["."] += 1
        subfolders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:25]

    anomaly_breakdown = {}
    if anomaly_only:
        anom_base = and_(
            MediaItem.is_trashed == False,
            MediaItem.has_date_anomaly == True,
            MediaItem.is_backup_copy == False,
            MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
        )
        anomaly_breakdown = {
            "all": db.scalar(select(func.count(MediaItem.id)).where(anom_base)) or 0,
            "folder": db.scalar(select(func.count(MediaItem.id)).where(anom_base, _anomaly_type_clause("folder"))) or 0,
            "filename": db.scalar(select(func.count(MediaItem.id)).where(anom_base, _anomaly_type_clause("filename"))) or 0,
            "copy_ctime": db.scalar(select(func.count(MediaItem.id)).where(anom_base, _anomaly_type_clause("copy_ctime"))) or 0,
            "exif": db.scalar(select(func.count(MediaItem.id)).where(anom_base, _anomaly_type_clause("exif"))) or 0,
            "other": db.scalar(select(func.count(MediaItem.id)).where(anom_base, _anomaly_type_clause("other"))) or 0,
        }

    cross_vault_backup_by_item_id = _build_cross_vault_backup_map(db, items)
    deleted_dup_counts_by_hash, deleted_dup_matches_by_hash = _build_deleted_dup_history_map(db, items, q)

    return {
        "timeline": dict(grouped),
        "items_flat": items,
        "total_matching": total_matching,
        "total_redundant_dups": redundant_dup_count_filtered,
        "cross_vault_backup_by_item_id": cross_vault_backup_by_item_id,
        "deleted_dup_counts_by_hash": deleted_dup_counts_by_hash,
        "deleted_dup_matches_by_hash": deleted_dup_matches_by_hash,
        "total_size_mb": total_size_bytes / (1024 * 1024),
        "page": page_int,
        "page_size": page_size_int,
        "total_pages": total_pages,
        "view_mode": view_mode,
        "sort_by": sort_by,
        "active_group": active_group,
        "active_volume": active_volume,
        "active_tag": tag,
        "active_quality_defect": quality_defect,
        "active_volume_id": vol_id_int,
        "active_folder": folder_prefix,
        "trashed_only": trashed_only,
        "duplicates_only": duplicates_only,
        "redundant_clusters_only": redundant_clusters_only,
        "backups_to_move_only": backups_to_move_only,
        "backup_filter": backup_filter,
        "backup_types": list(backup_types or []),
        "backup_mode": backup_mode,
        "anomaly_only": anomaly_only,
        "anomaly_type": anomaly_type,
        "anomaly_breakdown": anomaly_breakdown,
        "unbacked_only": unbacked_only,
        "untagged_only": untagged_only,
        "unvectored_only": unvectored_only,
        "unanalyzed_quality_only": unanalyzed_quality_only,
        "trashed_groups": trashed_groups,
        "subfolders": subfolders,
        "all_groups": all_groups,
        "flat_album_choices": build_flat_album_choices(all_groups),
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
    except ImportError as exc:
        logger.debug("pillow_heif not available: %s", exc)

    from app.scanner import ThumbnailService, IMAGE_EXTS

    p = THUMBNAILS_DIR / filename
    if not ThumbnailService.is_valid_thumbnail(p):
        media_id = filename.rsplit(".", 1)[0]
        it = db.scalars(select(MediaItem).options(joinedload(MediaItem.volume)).where(MediaItem.id == media_id)).unique().first()
        if it and it.volume:
            StorageAutomationService(db).refresh_volumes_online_status()
            db.refresh(it.volume)
            src_p = Path(it.volume.last_known_drive_letter) / it.relative_path
            if it.volume.is_online and src_p.exists():
                try:
                    _, w, h, _ = ThumbnailService.generate(src_p, it.id, src_p.suffix.lower() in IMAGE_EXTS)
                    if w and h:
                        it.width = w
                        it.height = h
                        it.is_corrupted = False
                        db.commit()
                except Exception as exc:
                    logger.debug("Lazy thumbnail regeneration failed: %s", exc)

    if not ThumbnailService.is_valid_thumbnail(p):
        if p.exists():
            try:
                p.unlink()
            except OSError as rm_exc:
                logger.debug("Could not remove empty thumbnail %s: %s", p, rm_exc)
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="image/webp", headers={"Cache-Control": "no-cache"})


@router.get("/{media_id}/raw")
def serve_raw_media(media_id: str, db: Session = Depends(get_db)):
    from PIL import Image, ImageFile, ImageOps
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError as exc:
        logger.debug("pillow_heif not available: %s", exc)

    it = db.scalars(select(MediaItem).options(joinedload(MediaItem.volume)).where(MediaItem.id == media_id)).unique().first()
    if not it:
        raise HTTPException(status_code=404)

    if it.volume:
        full_path = Path(it.volume.last_known_drive_letter) / it.relative_path
        if full_path.exists() and full_path.is_file():
            ext = full_path.suffix.lower()

            # 1. On-the-fly conversion of HEIC/HEIF/TIFF or previously flagged truncated/damaged images to clean JPEG
            if ext in (".heic", ".heif", ".tiff", ".tif") or (it.is_corrupted and it.media_type == "image"):
                try:
                    with Image.open(full_path) as img:
                        img = ImageOps.exif_transpose(img)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=90)
                        buf.seek(0)
                        return StreamingResponse(buf, media_type="image/jpeg")
                except Exception as exc:
                    logger.debug("On-the-fly JPEG conversion failed: %s", exc)
                    if it.is_corrupted and it.media_type == "image":
                        from app.ai.quality import decode_image_array
                        import cv2
                        arr = decode_image_array(full_path, grayscale=False)
                        if arr is not None:
                            ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            if ok:
                                return StreamingResponse(io.BytesIO(enc.tobytes()), media_type="image/jpeg")

            # 2. On-the-fly transcoding of .MOV / .MKV / .AVI / .WMV to browser-playable H.264 MP4
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
                    except Exception as exc:
                        logger.debug("Video transcoding skipped: %s", exc)
                if cached_mp4.exists() and cached_mp4.stat().st_size > 1000:
                    return FileResponse(cached_mp4, media_type="video/mp4", filename=f"{full_path.stem}.mp4")

            return FileResponse(full_path, media_type=it.mime_type, filename=it.file_name)

    from app.scanner import ThumbnailService
    thumb_path = THUMBNAILS_DIR / it.thumbnail_relative_path
    if ThumbnailService.is_valid_thumbnail(thumb_path):
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

    all_groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False).order_by(MediaGroup.name.asc())).all())
    duplicate_copies = list(
        db.scalars(
            select(MediaItem)
            .options(joinedload(MediaItem.volume), joinedload(MediaItem.group))
            .where(MediaItem.file_hash == it.file_hash, MediaItem.is_trashed == False)
            .order_by(desc(MediaItem.is_primary_duplicate), asc(MediaItem.duplicate_role), asc(MediaItem.is_backup_copy), asc(MediaItem.effective_date))
        ).unique().all()
    )
    folder_info = MetadataDateResolver.parse_folder_date(Path(it.relative_path), rel_path=it.relative_path)
    folder_date_str = None
    if folder_info:
        f_dt, f_prec, f_seg = folder_info
        folder_date_str = f"{f_dt.strftime('%Y-%m-%d')} ({f_prec}, katalog: '{f_seg}')"

    quality_tags = [t for t in it.tags if t.source_model == "quality_check" or t.tag.startswith("do_usuniecia")]
    semantic_tags = [t for t in it.tags if t.source_model != "quality_check" and not t.tag.startswith("do_usuniecia")]

    defect_labels = []
    seen_tags = set()
    for qt in quality_tags:
        if qt.tag in seen_tags:
            continue
        seen_tags.add(qt.tag)
        if qt.tag == "do_usuniecia:uszkodzone":
            defect_labels.append({"tag": qt.tag, "name": "Plik uszkodzony", "desc": "Błąd nagłówka / dekompresji / uszkodzone bajty", "badge_class": "bg-rose-950 text-rose-300 border-rose-700"})
        elif qt.tag == "do_usuniecia:rozmyte":
            blur_val = f"{it.blur_score:.1f}" if it.blur_score is not None else "0.0"
            defect_labels.append({"tag": qt.tag, "name": f"Rozmyte / brak ostrości (wynik: {blur_val})", "desc": f"Niski wskaźnik ostrości Laplaciana ({blur_val} < 45.0)", "badge_class": "bg-amber-950 text-amber-300 border-amber-700"})
        elif qt.tag == "do_usuniecia:niedoswietlone":
            defect_labels.append({"tag": qt.tag, "name": "Niedoświetlone (skrajnie ciemne)", "desc": "Średnia jasność < 15.0 lub 95. percentyl < 38.0", "badge_class": "bg-slate-900 text-slate-300 border-slate-700"})
        elif qt.tag == "do_usuniecia:przeswietlone":
            defect_labels.append({"tag": qt.tag, "name": "Prześwietlone (skrajnie jasne)", "desc": "Średnia jasność > 243.0 lub 5. percentyl > 220.0", "badge_class": "bg-yellow-950 text-yellow-300 border-yellow-700"})
        elif qt.tag == "do_usuniecia:brak_detali":
            defect_labels.append({"tag": qt.tag, "name": "Brak detali (płaski / pusty kadr)", "desc": "Odchylenie standardowe luminancji < 6.5 (jednolite tło)", "badge_class": "bg-zinc-900 text-zinc-300 border-zinc-700"})
        elif qt.tag == "do_usuniecia:niska_rozdzielczosc":
            dims_str = f"{it.width}x{it.height}" if it.width and it.height else "< 160px"
            defect_labels.append({"tag": qt.tag, "name": f"Niska rozdzielczość ({dims_str})", "desc": "Dłuższa krawędź < 160 px", "badge_class": "bg-purple-950 text-purple-300 border-purple-700"})

    if it.is_corrupted and not any(d["tag"] == "do_usuniecia:uszkodzone" for d in defect_labels):
        defect_labels.append({"tag": "do_usuniecia:uszkodzone", "name": "Plik uszkodzony", "desc": "Błąd odczytu pliku źródłowego", "badge_class": "bg-rose-950 text-rose-300 border-rose-700"})
    if it.is_blurry and not any(d["tag"] == "do_usuniecia:rozmyte" for d in defect_labels):
        blur_val = f"{it.blur_score:.1f}" if it.blur_score is not None else "0.0"
        defect_labels.append({"tag": "do_usuniecia:rozmyte", "name": f"Rozmyte / brak ostrości (wynik: {blur_val})", "desc": "Niski wskaźnik ostrości Laplaciana", "badge_class": "bg-amber-950 text-amber-300 border-amber-700"})

    cross_vault_backup_by_item_id = _build_cross_vault_backup_map(db, [it] + duplicate_copies)
    deleted_duplicate_paths = list(
        db.scalars(
            select(DeletedDuplicateHistory)
            .where(DeletedDuplicateHistory.file_hash == it.file_hash)
            .order_by(DeletedDuplicateHistory.volume_label.asc(), DeletedDuplicateHistory.relative_path.asc())
        ).all()
    )

    return render_template(
        request,
        "partials/modal_detail.html",
        {
            "item": it,
            "all_groups": all_groups,
            "flat_album_choices": build_flat_album_choices(all_groups),
            "duplicate_copies": duplicate_copies,
            "deleted_duplicate_paths": deleted_duplicate_paths,
            "cross_vault_backup_by_item_id": cross_vault_backup_by_item_id,
            "folder_date_str": folder_date_str,
            "quality_tags": quality_tags,
            "semantic_tags": semantic_tags,
            "defect_labels": defect_labels,
        },
    )
