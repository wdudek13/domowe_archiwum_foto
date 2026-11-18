from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import select, delete, update, text, or_
from sqlalchemy.orm import Session

from app.backup import BackupManager
import app.database as db_module
from app.database import get_db, SessionLocal, THUMBNAILS_DIR, DATA_DIR
from app.models import (
    WindowsVolume,
    MediaItem,
    MediaGroup,
    MediaTag,
    MediaEmbedding,
    BackgroundJob,
)
from app.operation_log import OperationLogger, ProcessManager
from app.scanner import WindowsVolumeInspector, FileScannerService, MetadataDateResolver, ThumbnailService, IMAGE_EXTS, VIDEO_EXTS
from app.storage_ops import StorageAutomationService
from app.ai.quality import MediaQualityAnalyzer
from app.ai.tagger import OnDemandObjectTagger
from app.ai.embeddings import OnDemandClipEmbedder
from app.routers.media import apply_media_filters, to_int, to_bool, render_template

logger = logging.getLogger("ActionsRouter")
router = APIRouter(prefix="/actions", tags=["Actions"])


def _merge_selected_ids(selected_ids: List[str], selected_ids_json: Optional[str]) -> List[str]:
    merged: List[str] = list(selected_ids or [])
    if selected_ids_json and selected_ids_json.strip():
        try:
            parsed = json.loads(selected_ids_json)
            if isinstance(parsed, list):
                seen = set(merged)
                for sid in parsed:
                    s = str(sid).strip()
                    if s and s not in seen:
                        merged.append(s)
                        seen.add(s)
        except Exception as exc:
            logger.debug("Invalid selected_ids_json ignored: %s", exc)
    return merged


async def _read_large_form(request: Request):
    """Parses form bodies up to 64 MB / 250k fields so huge cross-page duplicate selections never 400."""
    return await request.form(max_fields=250000, max_part_size=64 * 1024 * 1024)


def _form_str(form: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    val = form.get(key)
    if val is None:
        return default
    return str(val)


def _form_bool(form: Any, key: str, default: bool = False) -> bool:
    val = form.get(key)
    if val is None:
        return default
    return to_bool(str(val))


def _fetch_items_by_ids_chunked(db: Session, ids: List[str], extra_where=None) -> List[MediaItem]:
    items: List[MediaItem] = []
    for idx in range(0, len(ids), 5000):
        chunk = ids[idx : idx + 5000]
        stmt = select(MediaItem).where(MediaItem.id.in_(chunk))
        if extra_where is not None:
            stmt = stmt.where(extra_where)
        items.extend(db.scalars(stmt).all())
    return items


def _has_duplicate_role_in_ids(db: Session, ids: List[str]) -> bool:
    for idx in range(0, len(ids), 5000):
        chunk = ids[idx : idx + 5000]
        found = db.scalar(
            select(MediaItem.id).where(MediaItem.id.in_(chunk), MediaItem.duplicate_role == "duplicate").limit(1)
        )
        if found is not None:
            return True
    return False


def _resolve_bulk_ids_fast(
    db: Session,
    scope: str,
    selected_ids: List[str],
    q: Optional[str] = None,
    media_type: str = "all",
    volume_id: Optional[str] = None,
    folder_prefix: Optional[str] = None,
    group_id: Optional[str] = None,
    tag: Optional[str] = None,
    year: Optional[str] = None,
    anomaly_only: bool = False,
    blurry_only: bool = False,
    duplicates_only: bool = False,
    untagged_only: bool = False,
    ungrouped_only: bool = False,
    unbacked_only: bool = False,
    unvectored_only: bool = False,
    unanalyzed_quality_only: bool = False,
    quality_defect: Optional[str] = None,
    selected_ids_json: Optional[str] = None,
    redundant_clusters_only: bool = False,
    backups_to_move_only: bool = False,
    backup_filter: str = "none",
    backup_types: Optional[List[str]] = None,
    backup_mode: str = "hide",
    anomaly_type: str = "all",
) -> List[str]:
    """Fast ID-only resolution that avoids loading tens of thousands of ORM rows on HTTP POST."""
    eff_ids = _merge_selected_ids(selected_ids, selected_ids_json)
    if scope == "selected" and eff_ids:
        return eff_ids
    if scope == "all_drives":
        stmt = select(MediaItem.id).where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False)
    elif scope.startswith("volume:"):
        vid = to_int(scope.split(":", 1)[1])
        stmt = select(MediaItem.id).where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False, MediaItem.volume_id == vid)
    else:
        stmt = apply_media_filters(
            select(MediaItem.id),
            db,
            q,
            media_type,
            to_int(volume_id),
            folder_prefix,
            to_int(group_id),
            tag,
            to_int(year),
            anomaly_only,
            blurry_only,
            duplicates_only,
            untagged_only,
            ungrouped_only,
            False,
            unbacked_only,
            unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
            quality_defect=quality_defect,
            redundant_clusters_only=redundant_clusters_only,
            backups_to_move_only=backups_to_move_only,
            backup_filter=backup_filter,
            backup_types=backup_types,
            backup_mode=backup_mode,
            anomaly_type=anomaly_type,
        ).distinct()
    return [str(x) for x in db.scalars(stmt).all()]


def resolve_target_items(
    db: Session,
    scope: str,
    selected_ids: List[str],
    q: Optional[str] = None,
    media_type: str = "all",
    volume_id: Optional[str] = None,
    folder_prefix: Optional[str] = None,
    group_id: Optional[str] = None,
    tag: Optional[str] = None,
    year: Optional[str] = None,
    anomaly_only: bool = False,
    blurry_only: bool = False,
    duplicates_only: bool = False,
    untagged_only: bool = False,
    ungrouped_only: bool = False,
    unbacked_only: bool = False,
    unvectored_only: bool = False,
    unanalyzed_quality_only: bool = False,
    quality_defect: Optional[str] = None,
    limit: Optional[int] = None,
    selected_ids_json: Optional[str] = None,
    redundant_clusters_only: bool = False,
    backups_to_move_only: bool = False,
    backup_filter: str = "none",
    backup_types: Optional[List[str]] = None,
    backup_mode: str = "hide",
    anomaly_type: str = "all",
) -> List[MediaItem]:
    eff_ids = _merge_selected_ids(selected_ids, selected_ids_json)
    if scope == "selected" and eff_ids:
        if limit:
            eff_ids = eff_ids[:limit]
        return _fetch_items_by_ids_chunked(db, eff_ids)
    elif scope == "all_drives":
        stmt = select(MediaItem).where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False)
    elif scope.startswith("volume:"):
        vid = to_int(scope.split(":", 1)[1])
        stmt = select(MediaItem).where(MediaItem.is_trashed == False, MediaItem.is_backup_copy == False, MediaItem.volume_id == vid)
    else:
        stmt = apply_media_filters(
            select(MediaItem),
            db,
            q,
            media_type,
            to_int(volume_id),
            folder_prefix,
            to_int(group_id),
            tag,
            to_int(year),
            anomaly_only,
            blurry_only,
            duplicates_only,
            untagged_only,
            ungrouped_only,
            False,
            unbacked_only,
            unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
            quality_defect=quality_defect,
            redundant_clusters_only=redundant_clusters_only,
            backups_to_move_only=backups_to_move_only,
            backup_filter=backup_filter,
            backup_types=backup_types,
            backup_mode=backup_mode,
            anomaly_type=anomaly_type,
        ).distinct()
    if limit:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).unique().all())


def _compute_scan_relative_path(full: Path, abs_root: Path, root_resolved: Path, abs_target: Path) -> str:
    """Computes a stable relative path without collapsing Windows junctions into duplicate target paths."""
    abs_full = Path(os.path.abspath(full))
    try:
        return abs_full.relative_to(abs_root).as_posix()
    except ValueError:
        try:
            return full.resolve().relative_to(root_resolved).as_posix()
        except ValueError:
            try:
                return abs_full.relative_to(abs_target).as_posix()
            except ValueError:
                return full.name


def _commit_scan_batch(db: Session, batch_items: List[MediaItem], vol_id: int) -> int:
    """Commits a batch of scanned MediaItems, falling back to row-by-row conflict-safe insert on error."""
    if not batch_items:
        return 0
    try:
        db.add_all(batch_items)
        db.commit()
        return len(batch_items)
    except Exception as exc:
        logger.warning("Scan batch commit fallback due to conflict/error (%s); inserting row-by-row.", exc)
        db.rollback()
        added = 0
        for item in batch_items:
            exists_id = db.scalar(
                select(MediaItem.id).where(
                    MediaItem.volume_id == vol_id,
                    MediaItem.relative_path == item.relative_path,
                )
            )
            if exists_id:
                continue
            try:
                db.add(item)
                db.commit()
                added += 1
            except Exception as row_exc:
                logger.debug("Skipping conflicting scan item %s: %s", item.relative_path, row_exc)
                db.rollback()
        return added


def execute_scan(path_str: str, force_rescan: bool = False, job_id: Optional[int] = None) -> dict:
    clean = path_str.strip(' \t\n\r"\'')
    if not clean:
        return {"status": "error", "message": "Podaj poprawną ścieżkę do skanowania."}
    if len(clean) == 2 and clean[1] == ":":
        clean += "\\"
    target = Path(clean).resolve()

    if not target.exists():
        return {"status": "error", "message": f"Ścieżka nie istnieje: {clean}"}

    try:
        vsn, label, root = WindowsVolumeInspector.get_volume_info(target)
        root_path = Path(root).resolve()
        abs_root = Path(os.path.abspath(root))
        abs_target = Path(os.path.abspath(target))

        with SessionLocal() as db:
            vol = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == vsn))
            try:
                usage = shutil.disk_usage(root)
                total_b, free_b = int(usage.total), int(usage.free)
            except Exception as exc:
                logger.debug("disk_usage skipped during scan for %s: %s", root, exc)
                total_b, free_b = 0, 0

            if not vol:
                vol = WindowsVolume(
                    volume_serial_number=vsn,
                    volume_label=label,
                    last_known_drive_letter=root,
                    is_online=True,
                    total_space_bytes=total_b,
                    free_space_bytes=free_b,
                )
                db.add(vol)
                db.commit()
                db.refresh(vol)
            else:
                vol.last_known_drive_letter = root
                vol.is_online = True
                if total_b > 0 or free_b > 0:
                    vol.total_space_bytes = total_b
                    vol.free_space_bytes = free_b
                db.commit()

            existing_rows = db.execute(
                select(MediaItem.relative_path, MediaItem.id, MediaItem.thumbnail_relative_path, MediaItem.is_trashed, MediaItem.is_backup_copy)
                .where(MediaItem.volume_id == vol.id)
            ).all()
            existing_map = {r[0]: (r[1], r[2], bool(r[3]), bool(r[4])) for r in existing_rows}
            existing_map_lower = {r[0].lower(): (r[0], r[1], r[2], bool(r[3]), bool(r[4])) for r in existing_rows}

            candidates_to_add = []
            existing_to_refresh: List[Tuple[Path, str]] = []
            queued_rels_lower: set[str] = set()
            seen_rels_lower: set[str] = set()
            found_total = 0
            already_in_db = 0
            repaired_thumbs = 0
            updated_existing = 0

            # Exclude daf_albums/ and daf_backup/ when walking top-level folders
            scan_target_low = target.name.lower()
            skip_dir_names = {"daf_albums"}
            if scan_target_low not in ("daf_backup", "backup"):
                skip_dir_names.add("daf_backup")

            for r, dirs, files in os.walk(target):
                dirs[:] = [d for d in dirs if d.lower() not in skip_dir_names]
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                        continue

                    full = Path(r) / f
                    found_total += 1
                    rel = _compute_scan_relative_path(full, abs_root, root_path, abs_target)
                    rel_low = rel.lower()
                    seen_rels_lower.add(rel_low)

                    if rel_low in queued_rels_lower:
                        continue

                    in_existing = rel in existing_map or rel_low in existing_map_lower
                    if in_existing:
                        already_in_db += 1
                        ex_info = existing_map.get(rel) or existing_map_lower[rel_low][1:]
                        item_id, thumb_name = ex_info[0], ex_info[1]
                        thumb_file = THUMBNAILS_DIR / thumb_name
                        if not thumb_file.exists() or thumb_file.stat().st_size == 0:
                            try:
                                ThumbnailService.generate(full, item_id, ext in IMAGE_EXTS)
                                repaired_thumbs += 1
                            except Exception as exc:
                                logger.debug("Thumbnail repair failed for %s: %s", full, exc)
                        if force_rescan:
                            # Non-destructive rescan: preserve AI tags, embeddings, quality scores, albums, rotation
                            queued_rels_lower.add(rel_low)
                            existing_to_refresh.append((full, item_id))
                    else:
                        queued_rels_lower.add(rel_low)
                        candidates_to_add.append((full, rel))

            if existing_to_refresh:
                for full_p, ex_id in existing_to_refresh:
                    ex_item = db.get(MediaItem, ex_id)
                    if not ex_item:
                        continue
                    try:
                        st = full_p.stat()
                        is_img_f = full_p.suffix.lower() in IMAGE_EXTS
                        exif_dt, fn_dt, ct, mt, eff, anom, reason = MetadataDateResolver.resolve_dates(
                            full_p, is_img_f, rel_path=ex_item.relative_path
                        )
                        ex_item.file_size_bytes = st.st_size
                        ex_item.exif_date = exif_dt
                        ex_item.filename_date = fn_dt
                        ex_item.fs_ctime = ct
                        ex_item.fs_mtime = mt
                        ex_item.effective_date = eff
                        ex_item.has_date_anomaly = anom
                        ex_item.date_anomaly_reason = reason
                        if not ex_item.embedding:
                            t_full = THUMBNAILS_DIR / ex_item.thumbnail_relative_path
                            if t_full.exists() and t_full.stat().st_size > 0:
                                blob = OnDemandClipEmbedder.generate_image_embedding(t_full)
                                ex_item.embedding = MediaEmbedding(
                                    media_id=ex_item.id, model_name=OnDemandClipEmbedder.MODEL_NAME, dimensions=512, vector=blob
                                )
                        updated_existing += 1
                    except Exception as exc:
                        logger.debug("Non-destructive rescan refresh skipped for %s: %s", full_p, exc)
                db.commit()

            # Check DB records under the scanned path prefix that are missing physically on disk
            scan_prefix_rel = ""
            if abs_target != abs_root:
                try:
                    scan_prefix_rel = abs_target.relative_to(abs_root).as_posix().rstrip("/").lower()
                except ValueError:
                    scan_prefix_rel = ""

            missing_on_disk = 0
            for rel_k, (ex_id, _th, ex_trashed, ex_bkp) in existing_map.items():
                if ex_trashed or ex_bkp:
                    continue
                rel_k_low = rel_k.lower()
                if rel_k_low.startswith("daf_backup/") or rel_k_low.startswith("backup/") or rel_k_low.startswith("daf_albums/"):
                    continue
                if scan_prefix_rel and not (rel_k_low.startswith(scan_prefix_rel + "/") or rel_k_low == scan_prefix_rel):
                    continue
                if rel_k_low not in seen_rels_lower:
                    missing_on_disk += 1

            if job_id is None and candidates_to_add:
                job_id = ProcessManager.create_job(
                    job_type="scan",
                    title=f"Skanowanie dysku {label} ({clean})",
                    item_ids=[c[1] for c in candidates_to_add],
                    params={"path": clean, "force_rescan": force_rescan},
                    db=db,
                )

            def _worker(args):
                f_path, r_path = args
                return FileScannerService.process_file(f_path, vol, rel_override=r_path)

            indexed = 0
            stopped_early = False
            pending_batch: List[MediaItem] = []
            if candidates_to_add:
                with ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 2))) as pool:
                    for idx, item in enumerate(pool.map(_worker, candidates_to_add)):
                        if job_id and ProcessManager.is_stop_requested(job_id):
                            stopped_early = True
                            if pending_batch:
                                indexed += _commit_scan_batch(db, pending_batch, vol.id)
                                pending_batch.clear()
                            remaining_paths = [c[1] for c in candidates_to_add[idx:]]
                            ProcessManager.update_progress(
                                job_id,
                                remaining_ids=remaining_paths,
                                status="STOPPED",
                                message=f"Wstrzymano skanowanie po zindeksowaniu {indexed} plików.",
                                db=db,
                            )
                            db.commit()
                            break
                        if item:
                            pending_batch.append(item)
                            if len(pending_batch) >= 50:
                                added_now = _commit_scan_batch(db, pending_batch, vol.id)
                                indexed += added_now
                                pending_batch.clear()
                                if job_id:
                                    ProcessManager.update_progress(job_id, processed_delta=added_now, db=db)
                                    db.commit()
                if pending_batch:
                    indexed += _commit_scan_batch(db, pending_batch, vol.id)
                    pending_batch.clear()

            # Duplicate clustering and unbacked analysis are strictly on-demand
            # (triggered from 'Analizuj AI' or 'Backup i Deduplikacja').
            dup_total = 0

            if job_id and not stopped_early:
                ProcessManager.finish_job(
                    job_id,
                    status="COMPLETED",
                    message=f"Zindeksowano {indexed} nowych plików z wolumenu {label} ({vsn}).",
                    db=db,
                )
                db.commit()
            elif not job_id:
                OperationLogger.log(
                    "SCAN",
                    f"Skanowanie {label} ({vsn}): znaleziono {found_total}, nowych: {indexed}, w bazie: {already_in_db}, brak na dysku: {missing_on_disk}",
                    status="OK" if missing_on_disk == 0 else "WARN",
                    db=db,
                )

            return {
                "status": "ok",
                "found_total": found_total,
                "indexed": indexed,
                "already_in_db": already_in_db,
                "updated_existing": updated_existing,
                "missing_on_disk": missing_on_disk,
                "repaired_thumbs": repaired_thumbs,
                "dup_total": dup_total,
                "volume": label,
                "vsn": vsn,
            }

    except Exception as e:
        err = traceback.format_exc()
        logger.error("Scan error:\n%s", err)
        if job_id:
            ProcessManager.finish_job(job_id, status="FAILED", message=str(e))
        return {"status": "error", "message": f"Błąd skanera: {str(e)}"}


@router.post("/scan")
def trigger_scan(path: str = Form(...), force_rescan: bool = Form(False)):
    res = execute_scan(path, force_rescan=force_rescan)
    if res["status"] == "error":
        return HTMLResponse(
            f"<span class='text-rose-400 text-xs font-semibold'>{res['message']}</span>",
            status_code=400,
        )

    msg = (
        f"Wolumen {res['volume']} ({res['vsn']}): znaleziono {res['found_total']} plików | "
        f"Dodano nowych: {res['indexed']} | Już w bazie (zachowane dane AI/albumy): {res['already_in_db']}"
    )
    if res.get("updated_existing", 0) > 0:
        msg += f" | Odświeżono metadane: {res['updated_existing']}"
    if res["repaired_thumbs"] > 0:
        msg += f" | Odświeżono miniatur: {res['repaired_thumbs']}"
    extra_btn = ""
    if res.get("missing_on_disk", 0) > 0:
        msg += f" | ⚠️ Brakujących plików na dysku (w bazie): {res['missing_on_disk']}"
        extra_btn = (
            " <button type='button' onclick='openStatusModal()' "
            "class='ml-2 underline text-amber-300 font-bold'>Sprawdź spójność / Usuń brakujące &rarr;</button>"
        )

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>{extra_btn}",
        headers={"HX-Trigger": "refresh-all"},
    )


AI_BATCH_STATE = {
    "running": False,
    "stop_requested": False,
    "job_id": None,
    "analysis_type": "all",
    "processed": 0,
    "total": 0,
    "tagged": 0,
    "vectors": 0,
    "blurry": 0,
    "skipped_offline": 0,
    "result_summary": "",
}


def _run_storage_analysis_in_batch(job_id: int, item_ids: List[str], analysis_type: str) -> str:
    """Executes on-demand duplicate clustering (fast/exact) or unbacked files scan inside the AI batch worker."""
    with SessionLocal() as db:
        svc = StorageAutomationService(db)
        if analysis_type == "duplicates_fast":
            dup_total = svc.flag_duplicate_clusters()
            AI_BATCH_STATE["processed"] = len(item_ids)
            msg = (
                f"Zaktualizowano tagi duplikatów i backupów (ORYGINAŁ, BACKUP zielony/pomarańczowy, DUPLIKAT) — "
                f"oznaczono {dup_total} kopii w indeksie."
            )
            OperationLogger.log("DUP_SCAN_FAST", msg, status="OK", db=db)
            return msg

        if analysis_type == "duplicates_exact":
            processed_local = {"n": 0}

            def _progress(_fname: str):
                processed_local["n"] += 1
                AI_BATCH_STATE["processed"] = processed_local["n"]
                if processed_local["n"] % 10 == 0:
                    ProcessManager.update_progress(
                        job_id,
                        processed_delta=10,
                        message=f"Weryfikacja 100% plików na dyskach: {processed_local['n']} porównań...",
                    )

            res = svc.run_exact_duplicate_analysis(target_ids=item_ids, progress_cb=_progress)
            AI_BATCH_STATE["processed"] = len(item_ids)
            return (
                f"Dokładna analiza duplikatów 100%: klastry={res['clusters']}, "
                f"potwierdzone={res['verified_identical']}, kolizje={res['mismatches_separated']}, "
                f"pominięte offline={res['offline_skipped']}, duplikatów={res['dup_total']}."
            )

        if analysis_type == "date_discrepancy":
            svc.refresh_volumes_online_status()
            volumes_map = {v.id: v for v in db.scalars(select(WindowsVolume)).all()}
            checked = 0
            anomalies = 0
            updated = 0
            for start in range(0, len(item_ids), 250):
                if AI_BATCH_STATE["stop_requested"] or ProcessManager.is_stop_requested(job_id):
                    ProcessManager.update_progress(
                        job_id,
                        remaining_ids=item_ids[start:],
                        status="STOPPED",
                        message=f"Zatrzymano analizę niezgodności dat na {checked}/{len(item_ids)}.",
                        db=db,
                    )
                    return f"Zatrzymano analizę niezgodności dat na {checked}/{len(item_ids)}."

                chunk_ids = item_ids[start : start + 250]
                chunk_items = list(db.scalars(select(MediaItem).where(MediaItem.id.in_(chunk_ids))).all())
                for it in chunk_items:
                    checked += 1
                    vol = volumes_map.get(it.volume_id)
                    src_p = (Path(vol.last_known_drive_letter) / it.relative_path) if vol else None
                    online_ok = bool(vol and vol.is_online and src_p and src_p.exists() and src_p.is_file())

                    exif_dt = it.exif_date
                    ctime = it.fs_ctime
                    mtime = it.fs_mtime
                    if online_ok and src_p is not None:
                        try:
                            st = src_p.stat()
                            ctime = datetime.datetime.fromtimestamp(st.st_ctime)
                            mtime = datetime.datetime.fromtimestamp(st.st_mtime)
                            it.fs_ctime = ctime
                            it.fs_mtime = mtime
                        except OSError as exc:
                            logger.debug("Stat refresh skipped for %s: %s", src_p, exc)
                        if exif_dt is None and it.media_type == "image":
                            parsed_exif = MetadataDateResolver.parse_exif(src_p)
                            if parsed_exif is not None:
                                exif_dt = parsed_exif
                                it.exif_date = parsed_exif

                    fn_dt = MetadataDateResolver.parse_filename(it.file_name) or it.filename_date
                    it.filename_date = fn_dt
                    folder_info = MetadataDateResolver.parse_folder_date(Path(it.relative_path), rel_path=it.relative_path)
                    new_eff, new_anom, new_reason = MetadataDateResolver.evaluate_date_sources(
                        exif_dt, fn_dt, ctime, mtime, folder_info=folder_info
                    )
                    if (
                        it.effective_date != new_eff
                        or bool(it.has_date_anomaly) != bool(new_anom)
                        or it.date_anomaly_reason != new_reason
                    ):
                        updated += 1
                    it.effective_date = new_eff
                    it.has_date_anomaly = new_anom
                    it.date_anomaly_reason = new_reason
                    if new_anom:
                        anomalies += 1

                db.commit()
                AI_BATCH_STATE["processed"] = checked
                ProcessManager.update_progress(
                    job_id,
                    processed_delta=len(chunk_items),
                    remaining_ids=item_ids[start + len(chunk_ids) :],
                    message=f"Sprawdzanie dat: {checked}/{len(item_ids)} (niezgodności: {anomalies})...",
                    db=db,
                )

            msg = (
                f"Analiza niezgodności dat: sprawdzono {checked} plików | "
                f"Wykryto niezgodności: {anomalies} | Zaktualizowano: {updated}."
            )
            OperationLogger.log("DATE_DISCREPANCY_SCAN", msg, status="OK", db=db)
            return msg

        unbacked_list = svc.detect_files_without_backup(recluster=True)
        AI_BATCH_STATE["processed"] = len(item_ids)
        msg = (
            f"Analiza kopii zapasowych: wykryto {len(unbacked_list)} oryginałów "
            f"bez backupu na innym urządzeniu fizycznym."
        )
        OperationLogger.log("UNBACKED_SCAN", msg, status="OK", db=db)
        return msg


def _run_background_ai_batch(job_id: int, item_ids: List[str], analysis_type: str = "all"):
    AI_BATCH_STATE["running"] = True
    AI_BATCH_STATE["stop_requested"] = False
    AI_BATCH_STATE["job_id"] = job_id
    AI_BATCH_STATE["analysis_type"] = analysis_type
    AI_BATCH_STATE["processed"] = 0
    AI_BATCH_STATE["total"] = len(item_ids)
    AI_BATCH_STATE["tagged"] = 0
    AI_BATCH_STATE["vectors"] = 0
    AI_BATCH_STATE["blurry"] = 0
    AI_BATCH_STATE["skipped_offline"] = 0
    AI_BATCH_STATE["result_summary"] = ""

    run_tags = analysis_type in ("all", "tags")
    run_vectors = analysis_type in ("all", "vectors")
    run_quality = analysis_type in ("all", "quality")
    is_storage_analysis = analysis_type in ("duplicates_fast", "duplicates_exact", "unbacked", "date_discrepancy")

    stopped_midway = False
    tagger = OnDemandObjectTagger() if run_tags else None
    try:
        if is_storage_analysis:
            if AI_BATCH_STATE["stop_requested"] or ProcessManager.is_stop_requested(job_id):
                stopped_midway = True
                ProcessManager.update_progress(
                    job_id,
                    remaining_ids=item_ids,
                    status="STOPPED",
                    message=f"Zatrzymano analizę ({analysis_type}).",
                )
            else:
                AI_BATCH_STATE["result_summary"] = _run_storage_analysis_in_batch(job_id, item_ids, analysis_type)
            return

        if tagger is not None:
            tagger.open_session()

        with SessionLocal() as db:
            StorageAutomationService(db).refresh_volumes_online_status()

        chunk_size = 50
        for start in range(0, len(item_ids), chunk_size):
            if AI_BATCH_STATE["stop_requested"] or ProcessManager.is_stop_requested(job_id):
                stopped_midway = True
                remaining = item_ids[start:]
                ProcessManager.update_progress(
                    job_id,
                    remaining_ids=remaining,
                    status="STOPPED",
                    message=f"Zatrzymano analizę AI ({analysis_type}) na {AI_BATCH_STATE['processed']}/{AI_BATCH_STATE['total']}.",
                )
                break

            batch_ids = item_ids[start : start + chunk_size]

            with SessionLocal() as db:
                rows = db.execute(
                    select(
                        MediaItem.id,
                        MediaItem.media_type,
                        MediaItem.thumbnail_relative_path,
                        MediaItem.relative_path,
                        WindowsVolume.last_known_drive_letter,
                        MediaItem.width,
                        MediaItem.height,
                        WindowsVolume.is_online,
                    )
                    .join(WindowsVolume, WindowsVolume.id == MediaItem.volume_id)
                    .where(MediaItem.id.in_(batch_ids))
                ).all()

            computed_batch = []
            for idx_in_batch, (mid, mtype, thumb_rel, rel_p, drive_let, orig_w, orig_h, vol_online) in enumerate(rows):
                if AI_BATCH_STATE["stop_requested"] or ProcessManager.is_stop_requested(job_id):
                    stopped_midway = True
                    remaining = item_ids[start + idx_in_batch :]
                    ProcessManager.update_progress(
                        job_id,
                        remaining_ids=remaining,
                        status="STOPPED",
                        message=f"Zatrzymano analizę AI ({analysis_type}) na {AI_BATCH_STATE['processed']}/{AI_BATCH_STATE['total']}.",
                    )
                    break

                thumb_p = THUMBNAILS_DIR / thumb_rel
                src_p = Path(drive_let) / rel_p
                src_accessible = bool(vol_online and src_p.exists() and src_p.is_file())
                thumb_ok = ThumbnailService.is_valid_thumbnail(thumb_p)

                # Quality / sharpness / defect inspection strictly requires direct disk access to the original file;
                # ignore offline files so cached thumbnails are never mistaken for corrupted originals.
                if analysis_type == "quality" and not src_accessible:
                    AI_BATCH_STATE["skipped_offline"] += 1
                    AI_BATCH_STATE["processed"] += 1
                    continue

                repaired_dims: Optional[Tuple[Optional[int], Optional[int]]] = None
                if not thumb_ok and src_accessible:
                    try:
                        _, rw, rh, _ = ThumbnailService.generate(src_p, mid, mtype == "image")
                        if rw and rh:
                            orig_w, orig_h = rw, rh
                            repaired_dims = (rw, rh)
                        thumb_ok = ThumbnailService.is_valid_thumbnail(thumb_p)
                    except Exception as thumb_exc:
                        logger.debug("AI batch thumbnail repair skipped for %s: %s", mid, thumb_exc)

                readable_path = thumb_p if thumb_ok else (src_p if src_accessible else None)
                if not readable_path and not (run_quality and src_accessible):
                    if not src_accessible:
                        AI_BATCH_STATE["skipped_offline"] += 1
                    AI_BATCH_STATE["processed"] += 1
                    continue

                tags = []
                if run_tags and tagger is not None and readable_path is not None:
                    tags = tagger.tag_image(readable_path)
                    if mtype == "video":
                        tags.append(("wideo", 1.0))

                vec_blob = None
                if run_vectors and readable_path is not None:
                    try:
                        vec_blob = OnDemandClipEmbedder.generate_image_embedding(readable_path)
                    except Exception as exc:
                        logger.debug("Vector embedding skipped for %s: %s", mid, exc)

                corrupted, blurry, score = False, False, None
                quality_tags: List[str] = []
                quality_done = False
                if run_quality and src_accessible:
                    quality_done = True
                    if mtype == "video":
                        corrupted, blurry, score = MediaQualityAnalyzer.analyze_video(src_p)
                        _, quality_tags = MediaQualityAnalyzer.classify_quality_tags(
                            src_p, corrupted, blurry, score
                        )
                    else:
                        detailed = MediaQualityAnalyzer.analyze_image_detailed(src_p, orig_dims=(orig_w, orig_h))
                        corrupted = detailed["is_corrupted"]
                        blurry = detailed["is_blurry"]
                        score = detailed["blur_score"]
                        quality_tags = detailed["quality_tags"]

                computed_batch.append((mid, tags, vec_blob, quality_done, corrupted, blurry, score, quality_tags, repaired_dims))
                AI_BATCH_STATE["processed"] += 1

            if computed_batch:
                with SessionLocal() as db:
                    for mid, tags, vec_blob, quality_done, corrupted, blurry, score, quality_tags, repaired_dims in computed_batch:
                        if repaired_dims is not None:
                            it_dim = db.get(MediaItem, mid)
                            if it_dim:
                                it_dim.width, it_dim.height = repaired_dims
                                it_dim.is_corrupted = False

                        if run_tags:
                            db.execute(delete(MediaTag).where(MediaTag.media_id == mid))
                            for t_name, conf in tags:
                                db.add(MediaTag(media_id=mid, tag=t_name, confidence=conf, source_model="opencv+onnx"))
                            if tags:
                                AI_BATCH_STATE["tagged"] += 1

                        if run_vectors and vec_blob is not None:
                            db.merge(
                                MediaEmbedding(
                                    media_id=mid,
                                    model_name=OnDemandClipEmbedder.MODEL_NAME,
                                    dimensions=512,
                                    vector=vec_blob,
                                )
                            )
                            AI_BATCH_STATE["vectors"] += 1

                        if run_quality and quality_done:
                            it_obj = db.get(MediaItem, mid)
                            if it_obj:
                                it_obj.is_corrupted = corrupted
                                it_obj.is_blurry = blurry
                                it_obj.blur_score = score
                                if blurry or corrupted:
                                    AI_BATCH_STATE["blurry"] += 1
                            db.execute(
                                delete(MediaTag).where(
                                    MediaTag.media_id == mid,
                                    MediaTag.source_model == "quality_check",
                                )
                            )
                            for qt in quality_tags:
                                db.add(MediaTag(media_id=mid, tag=qt, confidence=1.0, source_model="quality_check"))
                    db.commit()

                ProcessManager.update_progress(
                    job_id,
                    processed_delta=len(computed_batch),
                    remaining_ids=item_ids[start + len(computed_batch) :],
                    message=f"Przetworzono {AI_BATCH_STATE['processed']}/{AI_BATCH_STATE['total']} (Tagi: {AI_BATCH_STATE['tagged']}, Wektory: {AI_BATCH_STATE['vectors']}, Rozmyte/Wadliwe: {AI_BATCH_STATE['blurry']})",
                )
    except Exception as exc:
        logger.error("AI batch failure: %s", exc)
        ProcessManager.finish_job(job_id, status="FAILED", message=f"Błąd analizy AI: {exc}")
        stopped_midway = True
    finally:
        if tagger is not None:
            tagger.close_session()
            del tagger
        import gc
        gc.collect()
        AI_BATCH_STATE["running"] = False
        if not stopped_midway:
            offline_note = (
                f" | Pominięto offline (brak dostępu do pliku): {AI_BATCH_STATE['skipped_offline']}"
                if AI_BATCH_STATE.get("skipped_offline", 0) > 0
                else ""
            )
            final_msg = AI_BATCH_STATE.get("result_summary") or (
                f"Zakończono analizę AI ({analysis_type}): {AI_BATCH_STATE['processed']}/{AI_BATCH_STATE['total']} | "
                f"Otagowano: {AI_BATCH_STATE['tagged']} | Wektory: {AI_BATCH_STATE['vectors']} | Rozmyte/Wadliwe: {AI_BATCH_STATE['blurry']}{offline_note}"
            )
            ProcessManager.finish_job(
                job_id,
                status="COMPLETED",
                message=final_msg,
            )


@router.get("/ai-status")
def get_ai_batch_status():
    st = AI_BATCH_STATE
    if st["running"]:
        pct = int((st["processed"] / st["total"]) * 100) if st["total"] else 0
        return HTMLResponse(
            f"<div hx-get='/actions/ai-status' hx-trigger='every 1.5s' hx-swap='outerHTML' class='flex items-center space-x-3 text-xs'>"
            f"<span class='text-purple-300 font-semibold animate-pulse'>⚡ Przetwarzanie AI ({st['analysis_type']}) w tle: {st['processed']} / {st['total']} ({pct}%)</span>"
            f"<span class='text-slate-400 font-mono'>[Tagi: {st['tagged']} | Wektory: {st['vectors']} | Wady/Blur: {st['blurry']}]</span>"
            f"<button hx-post='/actions/ai-stop' hx-target='#global-notification' class='bg-rose-800 hover:bg-rose-700 text-white px-2 py-0.5 rounded text-[10px]'>⏹️ Zatrzymaj</button>"
            f"</div>"
        )
    if st["stop_requested"]:
        return HTMLResponse(
            f"<span class='text-amber-400 text-xs font-semibold'>"
            f"⏹️ Proces AI zatrzymany na {st['processed']} / {st['total']}. Możesz go wznowić w panelu 'Status i Procesy'."
            f"</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    if st.get("result_summary"):
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>✓ {st['result_summary']}</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    offline_note = (
        f" | Pominięto offline (brak dostępu do pliku): {st['skipped_offline']}"
        if st.get("skipped_offline", 0) > 0
        else ""
    )
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>"
        f"✓ Analiza AI ({st['analysis_type']}) zakończona! Przetworzono: {st['processed']} / {st['total']} | Otagowano: {st['tagged']} | Wektory 512-D: {st['vectors']} | Rozmyte/Wadliwe: {st['blurry']}{offline_note}"
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/ai-stop")
def stop_ai_batch():
    AI_BATCH_STATE["stop_requested"] = True
    ProcessManager.request_stop(AI_BATCH_STATE.get("job_id"))
    return HTMLResponse("<span class='text-amber-400 text-xs font-semibold'>⏹️ Zatrzymywanie procesu w tle...</span>")


@router.post("/analyze-all")
def trigger_full_ai_analysis(
    analysis_type: str = Form("all"),
    target_drive_scope: str = Form("current"),
    scope: str = Form("filtered"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    run_sync: bool = Form(False),
    db: Session = Depends(get_db),
):
    if AI_BATCH_STATE["running"]:
        return get_ai_batch_status()

    StorageAutomationService(db).refresh_volumes_online_status()
    effective_scope = target_drive_scope if target_drive_scope in ("all_drives",) or target_drive_scope.startswith("volume:") else scope
    items = resolve_target_items(
        db,
        effective_scope,
        selected_ids,
        q,
        media_type,
        volume_id,
        folder_prefix,
        group_id,
        tag,
        year,
        anomaly_only,
        blurry_only,
        duplicates_only,
        untagged_only=untagged_only,
        unvectored_only=unvectored_only,
        unanalyzed_quality_only=unanalyzed_quality_only,
        limit=None,
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików w wybranym zakresie do analizy AI.</span>")

    type_labels = {
        "all": "Pełna analiza AI (Tagi + Wektory 512-D + Ostrość i wady)",
        "tags": "Tagowanie obiektów i scen AI",
        "vectors": "Generowanie wektorów podobieństwa 512-D",
        "quality": "Analiza ostrości, ekspozycji i uszkodzonych plików",
        "duplicates_fast": "Szybka analiza duplikatów (baza SHA-256 / hash wideo)",
        "duplicates_exact": "Dokładna analiza duplikatów 100% (weryfikacja binarna na dyskach)",
        "unbacked": "Analiza plików bez kopii zapasowej na innym urządzeniu",
        "date_discrepancy": "Analiza niezgodności dat (EXIF / utworzenie pliku / nazwa / folder)",
    }
    item_ids = [it.id for it in items]
    job_id = ProcessManager.create_job(
        job_type="ai_analysis",
        title=f"{type_labels.get(analysis_type, 'Analiza AI')} ({len(item_ids)} szt.)",
        item_ids=item_ids,
        params={"analysis_type": analysis_type, "scope": effective_scope},
        db=db,
    )

    if run_sync:
        _run_background_ai_batch(job_id, item_ids, analysis_type=analysis_type)
        return get_ai_batch_status()

    ProcessManager.start_background_thread(job_id, _run_background_ai_batch, item_ids, analysis_type=analysis_type)
    return get_ai_batch_status()


def _get_maintenance_stats(db: Session) -> Dict[str, float]:
    """Returns current SQLite file size and reclaimable freelist MB without heavy disk walks."""
    db_path = db_module.DB_PATH
    db_size_mb = (db_path.stat().st_size / (1024 * 1024)) if db_path.exists() else 0.0
    freelist_mb = 0.0
    try:
        freelist_cnt = db.execute(text("PRAGMA freelist_count;")).scalar() or 0
        page_sz = db.execute(text("PRAGMA page_size;")).scalar() or 4096
        freelist_mb = (int(freelist_cnt) * int(page_sz)) / (1024 * 1024)
    except Exception as exc:
        logger.debug("freelist_count query failed: %s", exc)
    return {
        "db_size_mb": db_size_mb,
        "freelist_mb": freelist_mb,
    }


@router.get("/status-panel")
def get_status_and_logs_panel(request: Request, db: Session = Depends(get_db)):
    summary = ProcessManager.get_jobs_summary(db)
    logs = OperationLogger.get_recent_logs(db, limit=100)
    if DUP_DELETE_STATE["running"] or AI_BATCH_STATE["running"] or summary["has_running"]:
        volumes = list(db.scalars(select(WindowsVolume)).all())
    else:
        volumes = StorageAutomationService(db).refresh_volumes_online_status()
    maint_stats = _get_maintenance_stats(db)
    return render_template(
        request,
        "partials/status_panel.html",
        {
            "jobs_summary": summary,
            "logs": logs,
            "volumes": volumes,
            "ai_state": AI_BATCH_STATE,
            "dup_state": DUP_DELETE_STATE,
            "maint_stats": maint_stats,
        },
    )


@router.post("/vacuum-db")
def vacuum_sqlite_database(db: Session = Depends(get_db)):
    """Rebuilds the SQLite database file on disk (VACUUM) to physically reclaim freed pages."""
    db_path = db_module.DB_PATH
    if not db_path.exists():
        return HTMLResponse("<span class='text-rose-400 font-semibold'>Brak pliku bazy SQLite na dysku.</span>")

    before_bytes = db_path.stat().st_size
    wal_path = Path(str(db_path) + "-wal")
    if wal_path.exists():
        before_bytes += wal_path.stat().st_size

    # Release this request's SQLAlchemy transaction before exclusive VACUUM rebuild
    db.close()

    conn = sqlite3.connect(db_path.as_posix(), timeout=30.0, isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.execute("VACUUM;")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    except sqlite3.Error as exc:
        return HTMLResponse(
            f"<div class='p-2.5 rounded-lg bg-rose-950/70 border border-rose-700 text-rose-200'>"
            f"❌ Błąd kompaktowania bazy (VACUUM): {exc} (upewnij się, że inne ciężkie zapisy zostały zakończone)."
            f"</div>"
        )
    finally:
        conn.close()

    after_bytes = db_path.stat().st_size
    if wal_path.exists():
        after_bytes += wal_path.stat().st_size
    before_mb = before_bytes / (1024 * 1024)
    after_mb = after_bytes / (1024 * 1024)
    saved_mb = max(0.0, before_mb - after_mb)

    OperationLogger.log(
        "DB_VACUUM",
        f"Kompaktowanie bazy SQLite (VACUUM): {before_mb:.2f} MB -> {after_mb:.2f} MB (odzyskano {saved_mb:.2f} MB).",
        status="OK",
    )
    return HTMLResponse(
        f"<div class='p-2.5 rounded-lg bg-emerald-950/70 border border-emerald-600 text-emerald-100'>"
        f"🗜️ <strong>Zakończono kompaktowanie bazy (VACUUM):</strong> "
        f"<span class='font-mono'>{before_mb:.2f} MB &rarr; {after_mb:.2f} MB</span> "
        f"(odzyskano na dysku: <strong class='text-emerald-300 font-mono'>{saved_mb:.2f} MB</strong>)."
        f"</div>"
    )


@router.post("/cleanup-orphan-thumbs")
def cleanup_orphan_thumbnails(db: Session = Depends(get_db)):
    """Deletes thumbnail files in THUMBNAILS_DIR that are no longer referenced by any MediaItem (including Kosz)."""
    eff_thumbs_dir = THUMBNAILS_DIR
    if not eff_thumbs_dir.exists():
        return HTMLResponse("<span class='text-slate-400'>Katalog miniatur jeszcze nie istnieje (0 plików).</span>")

    known_thumbs = {
        str(p).replace("\\", "/").lstrip("/")
        for p in db.scalars(select(MediaItem.thumbnail_relative_path)).all()
        if p
    }
    scanned_cnt = 0
    deleted_cnt = 0
    freed_bytes = 0

    for fp in eff_thumbs_dir.rglob("*"):
        if not fp.is_file():
            continue
        scanned_cnt += 1
        rel = fp.relative_to(eff_thumbs_dir).as_posix()
        if rel not in known_thumbs:
            try:
                sz = fp.stat().st_size
                fp.unlink()
                deleted_cnt += 1
                freed_bytes += sz
            except OSError as exc:
                logger.debug("Orphan thumbnail unlink failed for %s: %s", fp, exc)

    freed_mb = freed_bytes / (1024 * 1024)
    OperationLogger.log(
        "THUMB_CLEANUP",
        f"Czyszczenie osieroconych miniatur: usunięto {deleted_cnt} z {scanned_cnt} plików (odzyskano {freed_mb:.2f} MB).",
        status="OK",
        db=db,
    )
    return HTMLResponse(
        f"<div class='p-2.5 rounded-lg bg-emerald-950/70 border border-emerald-600 text-emerald-100'>"
        f"🖼️ <strong>Wyczyszczono osierocone miniatury:</strong> usunięto <strong class='font-mono'>{deleted_cnt}</strong> "
        f"spośród <span class='font-mono'>{scanned_cnt}</span> plików w <code>data/thumbnails/</code> "
        f"(odzyskano: <strong class='text-emerald-300 font-mono'>{freed_mb:.2f} MB</strong>)."
        f"</div>"
    )


@router.post("/process-control")
def control_background_process(
    request: Request,
    action: str = Form(...),
    job_id: int = Form(...),
    run_sync: bool = Form(False),
    db: Session = Depends(get_db),
):
    job = db.get(BackgroundJob, job_id)
    if not job:
        return HTMLResponse("<span class='text-rose-400 text-xs'>Zadanie nie istnieje.</span>")

    if action == "stop":
        if job.job_type == "ai_analysis":
            AI_BATCH_STATE["stop_requested"] = True
        ProcessManager.request_stop(job_id)
        job.status = "STOPPED"
        job.last_message = "Zatrzymano na żądanie użytkownika."
        db.commit()
    elif action == "cancel":
        ProcessManager.request_stop(job_id)
        db.delete(job)
        db.commit()
        OperationLogger.log("PROCESS_CANCEL", f"Usunięto niedokończone zadanie #{job_id} ({job.title}).", status="WARN", db=db)
    elif action == "resume":
        try:
            rem_ids = json.loads(job.remaining_ids_json or "[]")
            params = json.loads(job.params_json or "{}")
        except Exception:
            rem_ids = []
            params = {}

        if not rem_ids and job.job_type != "backup_dedup":
            job.status = "COMPLETED"
            db.commit()
        else:
            ProcessManager.clear_stop(job.id)
            AI_BATCH_STATE["stop_requested"] = False
            job.status = "RUNNING"
            job.last_message = "Wznowiono wykonywanie procesu..."
            db.commit()
            if job.job_type == "ai_analysis":
                atype = params.get("analysis_type", "all")
                if run_sync:
                    _run_background_ai_batch(job.id, rem_ids, analysis_type=atype)
                else:
                    ProcessManager.start_background_thread(job.id, _run_background_ai_batch, rem_ids, analysis_type=atype)
            elif job.job_type in ("delete_permanently", "move_to_trash"):
                del_mode = "trash" if job.job_type == "move_to_trash" else "permanent"
                act_url = "/actions/move-to-trash" if del_mode == "trash" else "/actions/delete-permanently"
                is_dup_job = bool(params.get("is_duplicate_removal") or _has_duplicate_role_in_ids(db, rem_ids))
                if is_dup_job:
                    if run_sync:
                        res = StorageAutomationService(db).execute_safe_duplicate_removal(
                            item_ids=rem_ids,
                            delete_mode=del_mode,
                            force_duplicate_verification=True,
                            job_id=job.id,
                            trash_dir=DATA_DIR / "kosz",
                            thumbnails_dir=THUMBNAILS_DIR,
                        )
                        _build_safe_removal_html_response(act_url, del_mode, res, job.id, db)
                    else:
                        _start_duplicate_removal_async(
                            db=db,
                            item_ids=rem_ids,
                            delete_mode=del_mode,
                            force_duplicate_verification=True,
                            action_url=act_url,
                            skip_db_backup=True,
                            existing_job_id=job.id,
                        )
                elif job.job_type == "delete_permanently":
                    _execute_delete_permanently_items(db, rem_ids, job_id=job.id)
                else:
                    _execute_move_to_trash_items(db, rem_ids, job_id=job.id)
            elif job.job_type == "backup_dedup":
                svc = StorageAutomationService(db)
                svc.execute_backup_and_deduplication(job_id=job.id)
                ProcessManager.finish_job(job.id, status="COMPLETED", message="Backup i deduplikacja wznowione i wykonane.", db=db)
                db.commit()
            elif job.job_type == "scan":
                scan_path = str(params.get("path") or "").strip()
                if not scan_path:
                    ProcessManager.finish_job(
                        job.id,
                        status="FAILED",
                        message="Brak zapisanej ścieżki skanowania w parametrach zadania.",
                        db=db,
                    )
                    db.commit()
                else:
                    f_rescan = bool(params.get("force_rescan", False))
                    execute_scan(scan_path, force_rescan=f_rescan, job_id=job.id)
                    db.expire_all()

    resp = get_status_and_logs_panel(request, db)
    resp.headers["HX-Trigger"] = "refresh-all"
    return resp


def _balanced_sample_unbacked(items: List[MediaItem], max_total: int = 300) -> List[MediaItem]:
    """Provides a fair representation of unbacked files across all active volumes up to max_total."""
    if len(items) <= max_total:
        return items
    by_vol: Dict[int, List[MediaItem]] = defaultdict(list)
    for it in items:
        by_vol[it.volume_id].append(it)
    num_vols = len(by_vol)
    if num_vols == 0:
        return []
    share_per_vol = max(1, max_total // num_vols)
    sampled: List[MediaItem] = []
    remaining: List[MediaItem] = []
    for vid, v_items in by_vol.items():
        sampled.extend(v_items[:share_per_vol])
        remaining.extend(v_items[share_per_vol:])
    if len(sampled) < max_total and remaining:
        sampled.extend(remaining[: max_total - len(sampled)])
    return sampled[:max_total]


def _balanced_sample_duplicates(rows: List[Dict[str, Any]], max_total: int = 300) -> List[Dict[str, Any]]:
    """Provides a fair representation of duplicate rows across all active pairs up to max_total."""
    if len(rows) <= max_total:
        return rows
    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pk = r.get("pair_key_str", "other")
        by_key[pk].append(r)
    num_keys = len(by_key)
    if num_keys == 0:
        return []
    share = max(1, max_total // num_keys)
    sampled: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for k, k_rows in by_key.items():
        sampled.extend(k_rows[:share])
        remaining.extend(k_rows[share:])
    if len(sampled) < max_total and remaining:
        sampled.extend(remaining[: max_total - len(sampled)])
    return sampled[:max_total]


def _balanced_sample_important(rows: List[Dict[str, Any]], max_total: int = 300) -> List[Dict[str, Any]]:
    """Provides a fair representation of important candidate rows across all active volumes up to max_total."""
    if len(rows) <= max_total:
        return rows
    by_vol: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        vid = int(r.get("vol_id") or 0)
        by_vol[vid].append(r)
    num_vols = len(by_vol)
    if num_vols == 0:
        return []
    share = max(1, max_total // num_vols)
    sampled: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for vid, v_rows in by_vol.items():
        sampled.extend(v_rows[:share])
        remaining.extend(v_rows[share:])
    if len(sampled) < max_total and remaining:
        sampled.extend(remaining[: max_total - len(sampled)])
    return sampled[:max_total]


@router.get("/redundant-duplicate-ids")
def get_redundant_duplicate_ids(
    q: Optional[str] = None,
    media_type: str = "all",
    volume_id: Optional[str] = None,
    folder_prefix: Optional[str] = None,
    group_id: Optional[str] = None,
    tag: Optional[str] = None,
    year: Optional[str] = None,
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
    backup_types: List[str] = Query(default=[]),
    backup_mode: str = "hide",
    anomaly_type: str = "all",
    db: Session = Depends(get_db),
):
    """Returns IDs of all redundant copies (duplicate_role == 'duplicate', i.e. 3rd+ copy) across ALL pages for the current filter."""
    StorageAutomationService(db).flag_duplicate_clusters()
    stmt = apply_media_filters(
        select(MediaItem.id),
        db,
        q,
        media_type,
        to_int(volume_id),
        folder_prefix,
        to_int(group_id),
        tag,
        to_int(year),
        anomaly_only,
        blurry_only,
        duplicates_only,
        untagged_only,
        ungrouped_only,
        trashed_only,
        unbacked_only,
        unvectored_only=unvectored_only,
        unanalyzed_quality_only=unanalyzed_quality_only,
        quality_defect=quality_defect,
        redundant_clusters_only=redundant_clusters_only,
        backups_to_move_only=backups_to_move_only,
        backup_filter=backup_filter,
        backup_types=backup_types,
        backup_mode=backup_mode,
        anomaly_type=anomaly_type,
    ).where(MediaItem.duplicate_role == "duplicate").distinct()
    ids = [str(x) for x in db.scalars(stmt).all()]
    return JSONResponse({"ids": ids, "count": len(ids)})


@router.post("/refresh-duplicate-tags")
def refresh_duplicate_and_backup_tags(db: Session = Depends(get_db)):
    """Recomputes ORYGINAŁ / BACKUP (zielony/pomarańczowy) / DUPLIKAT tags across all clusters in SQLite."""
    svc = StorageAutomationService(db)
    dup_total = svc.flag_duplicate_clusters()
    msg = (
        f"✓ Zaktualizowano tagi klastrów: ORYGINAŁ, BACKUP (zielony/pomarańczowy), DUPLIKAT "
        f"(oznaczonych kopii poza oryginałami: {dup_total})."
    )
    OperationLogger.log("DUP_TAGS_REFRESH", msg, status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-emerald-300 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


def _balanced_sample_rows(rows: List[Dict[str, Any]], key_field: str = "pair_key_str", max_total: int = 300) -> List[Dict[str, Any]]:
    if len(rows) <= max_total:
        return rows
    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[str(r.get(key_field, "other"))].append(r)
    if not by_key:
        return []
    share = max(1, max_total // len(by_key))
    sampled: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for k_rows in by_key.values():
        sampled.extend(k_rows[:share])
        remaining.extend(k_rows[share:])
    if len(sampled) < max_total and remaining:
        sampled.extend(remaining[: max_total - len(sampled)])
    return sampled[:max_total]


def _prepare_backup_panel_context(
    db: Session,
    pref_vid: Optional[int] = None,
    selected_pairs: Optional[List[str]] = None,
    operation_mode: str = "backup_only",
    target_volume_ids: Optional[List[int]] = None,
    minimize_transfers: bool = True,
) -> dict:
    svc = StorageAutomationService(db)
    plan = svc.build_backup_and_dedup_plan(
        preferred_target_volume_id=pref_vid,
        selected_pair_keys=selected_pairs,
        target_volume_ids=target_volume_ids,
        minimize_transfers=minimize_transfers,
    )
    all_pair_key_strs = [p["pair_key_str"] for p in plan["ordered_pairs"]]
    effective_pair_keys = (
        [k for k in selected_pairs if k and ":" in k]
        if selected_pairs is not None
        else all_pair_key_strs
    )
    all_unbacked = plan.get("unbacked_items", [])
    all_local_relocs = plan.get("local_reloc_rows", [])
    all_remote_moves = plan.get("remote_move_rows", [])
    all_important_rows = plan.get("important_rows", [])

    unbacked_by_vol = plan.get("unbacked_count_by_vol", {})
    unbacked_by_pair = plan.get("unbacked_count_by_pair", {})
    local_reloc_by_pair = plan.get("local_reloc_by_pair", {})
    remote_move_by_pair = plan.get("remote_move_by_pair", {})
    important_by_vol = plan.get("important_count_by_vol", {})
    important_by_pair = plan.get("important_by_pair", {})

    if selected_pairs is not None:
        selected_set = set(effective_pair_keys)
        active_vols = {
            int(v) for p in selected_set if ":" in p for v in (p.split(":")[0], p.split(":")[1])
        }
        filtered_unbacked = [
            uf for uf in all_unbacked
            if (plan["unbacked_pair_keys_str"].get(uf.id) in selected_set) or (not plan["unbacked_pair_keys_str"].get(uf.id) and uf.volume_id in active_vols)
        ]
        filtered_local_relocs = [lr for lr in all_local_relocs if lr.get("pair_key_str") in selected_set]
        filtered_remote_moves = [rm for rm in all_remote_moves if rm.get("pair_key_str") in selected_set]
        filtered_important_rows = [
            ir for ir in all_important_rows
            if any(pk in selected_set for pk in ir.get("pair_keys", [])) or (not ir.get("pair_keys") and ir.get("vol_id") in active_vols)
        ]
        filtered_unbacked_count = sum(cnt for pk, cnt in unbacked_by_pair.items() if pk in selected_set)
        filtered_local_relocs_count = sum(cnt for pk, cnt in local_reloc_by_pair.items() if pk in selected_set)
        filtered_remote_moves_count = sum(cnt for pk, cnt in remote_move_by_pair.items() if pk in selected_set)
        filtered_important_count = sum(cnt for pk, cnt in important_by_pair.items() if pk in selected_set)
    else:
        filtered_unbacked = all_unbacked
        filtered_local_relocs = all_local_relocs
        filtered_remote_moves = all_remote_moves
        filtered_important_rows = all_important_rows
        filtered_unbacked_count = len(all_unbacked)
        filtered_local_relocs_count = len(all_local_relocs)
        filtered_remote_moves_count = len(all_remote_moves)
        filtered_important_count = len(all_important_rows)

    pair_counts = {
        "unbacked_by_vol": {str(k): v for k, v in unbacked_by_vol.items()},
        "unbacked_by_pair": unbacked_by_pair,
        "local_reloc_by_pair": local_reloc_by_pair,
        "remote_move_by_pair": remote_move_by_pair,
        "important_by_vol": {str(k): v for k, v in important_by_vol.items()},
        "important_by_pair": important_by_pair,
    }

    volumes_list = list(plan["volumes"].values())
    vol_space = svc.get_volume_space_map(volumes_list)
    latest_report_exists = (db_module.REPORTS_DIR / "raport_procedura_offline_ostatni.txt").exists()

    return {
        "plan": plan,
        "stats": plan["stats"],
        "space_problems": plan["space_problems"],
        "effective_target_volume_ids": plan["effective_target_volume_ids"],
        "minimize_transfers": plan["minimize_transfers"],
        "unbacked_files": _balanced_sample_unbacked(filtered_unbacked, 300),
        "unbacked_total": filtered_unbacked_count,
        "unbacked_all_total": len(all_unbacked),
        "local_reloc_files": _balanced_sample_rows(filtered_local_relocs, "pair_key_str", 300),
        "local_reloc_total": filtered_local_relocs_count,
        "local_reloc_all_total": len(all_local_relocs),
        "remote_move_files": _balanced_sample_rows(filtered_remote_moves, "pair_key_str", 300),
        "remote_move_total": filtered_remote_moves_count,
        "remote_move_all_total": len(all_remote_moves),
        "important_files": _balanced_sample_important(filtered_important_rows, 300),
        "important_total": filtered_important_count,
        "important_all_total": len(all_important_rows),
        "pair_counts": json.dumps(pair_counts),
        "volumes": volumes_list,
        "vol_space": vol_space,
        "latest_report_exists": latest_report_exists,
        "selected_operation_mode": "backup_only",
        "selected_pair_keys": effective_pair_keys,
    }


def _render_backup_panel_response(
    request: Request,
    db: Session,
    pref_vid: Optional[int] = None,
    selected_pairs: Optional[List[str]] = None,
    operation_mode: str = "backup_only",
    target_volume_ids: Optional[List[int]] = None,
    minimize_transfers: bool = True,
    execution_result: Optional[dict] = None,
    offline_report: Optional[dict] = None,
    trigger_refresh: bool = False,
):
    """Builds and renders the Backup modal partial with target volume & pair filtering state."""
    ctx = _prepare_backup_panel_context(
        db,
        pref_vid=pref_vid,
        selected_pairs=selected_pairs,
        operation_mode=operation_mode,
        target_volume_ids=target_volume_ids,
        minimize_transfers=minimize_transfers,
    )
    ctx["execution_result"] = execution_result
    ctx["offline_report"] = offline_report
    resp = render_template(request, "partials/backup_panel.html", ctx)
    if trigger_refresh:
        resp.headers["HX-Trigger"] = "refresh-all"
    return resp


def _render_backup_tables_response(
    request: Request,
    db: Session,
    pref_vid: Optional[int] = None,
    selected_pairs: Optional[List[str]] = None,
    target_volume_ids: Optional[List[int]] = None,
    minimize_transfers: bool = True,
):
    """Renders only the preview tables container for live pair-checkbox switching."""
    ctx = _prepare_backup_panel_context(
        db,
        pref_vid=pref_vid,
        selected_pairs=selected_pairs,
        target_volume_ids=target_volume_ids,
        minimize_transfers=minimize_transfers,
    )
    return render_template(request, "partials/backup_preview_tables.html", ctx)


def _parse_backup_form_controls(
    target_volume_ids: List[str],
    target_vols_submitted: bool,
    minimize_transfers: Optional[str],
    minimize_submitted: bool,
) -> Tuple[Optional[List[int]], bool]:
    parsed_targets: Optional[List[int]] = None
    if target_vols_submitted:
        parsed_targets = [int(x) for x in target_volume_ids if str(x).isdigit()]
    eff_minimize = True
    if minimize_submitted:
        eff_minimize = to_bool(minimize_transfers)
    elif minimize_transfers is not None:
        eff_minimize = to_bool(minimize_transfers)
    return parsed_targets, eff_minimize


@router.post("/backup-panel-tables")
def get_backup_panel_tables(
    request: Request,
    selected_pairs: List[str] = Form(default=[]),
    pairs_submitted: bool = Form(False),
    preferred_target_volume_id: Optional[str] = Form(None),
    target_volume_ids: List[str] = Form(default=[]),
    target_vols_submitted: bool = Form(False),
    minimize_transfers: Optional[str] = Form(None),
    minimize_submitted: bool = Form(False),
    db: Session = Depends(get_db),
):
    pref_vid = to_int(preferred_target_volume_id)
    effective_pairs = selected_pairs if pairs_submitted else (selected_pairs or None)
    parsed_targets, eff_minimize = _parse_backup_form_controls(
        target_volume_ids, target_vols_submitted, minimize_transfers, minimize_submitted
    )
    return _render_backup_tables_response(
        request,
        db,
        pref_vid=pref_vid,
        selected_pairs=effective_pairs,
        target_volume_ids=parsed_targets,
        minimize_transfers=eff_minimize,
    )


@router.get("/backup-panel")
def get_backup_panel(
    request: Request,
    target_volume_id: Optional[str] = None,
    selected_pairs: Optional[List[str]] = Query(None),
    db: Session = Depends(get_db),
):
    pref_vid = to_int(target_volume_id)
    return _render_backup_panel_response(
        request,
        db,
        pref_vid=pref_vid,
        selected_pairs=selected_pairs,
    )


@router.post("/backup-panel")
def recalculate_backup_panel(
    request: Request,
    target_volume_ids: List[str] = Form(default=[]),
    target_vols_submitted: bool = Form(False),
    minimize_transfers: Optional[str] = Form(None),
    minimize_submitted: bool = Form(False),
    selected_pairs: List[str] = Form(default=[]),
    pairs_submitted: bool = Form(False),
    recalc_pairs: bool = Form(False),
    db: Session = Depends(get_db),
):
    parsed_targets, eff_minimize = _parse_backup_form_controls(
        target_volume_ids, target_vols_submitted, minimize_transfers, minimize_submitted
    )
    effective_pairs = None if recalc_pairs else (selected_pairs if pairs_submitted else (selected_pairs or None))
    return _render_backup_panel_response(
        request,
        db,
        selected_pairs=effective_pairs,
        target_volume_ids=parsed_targets,
        minimize_transfers=eff_minimize,
    )


@router.post("/execute-backup")
def execute_backup_action(
    request: Request,
    preferred_target_volume_id: Optional[str] = Form(None),
    new_backup_drive_path: Optional[str] = Form(None),
    operation_mode: str = Form("backup_only"),
    selected_pairs: List[str] = Form(default=[]),
    pairs_submitted: bool = Form(False),
    target_volume_ids: List[str] = Form(default=[]),
    target_vols_submitted: bool = Form(False),
    minimize_transfers: Optional[str] = Form(None),
    minimize_submitted: bool = Form(False),
    db: Session = Depends(get_db),
):
    pref_vid = to_int(preferred_target_volume_id)
    parsed_targets, eff_minimize = _parse_backup_form_controls(
        target_volume_ids, target_vols_submitted, minimize_transfers, minimize_submitted
    )
    if new_backup_drive_path and new_backup_drive_path.strip():
        clean_p = new_backup_drive_path.strip(' \t\n\r"\'')
        if len(clean_p) == 2 and clean_p[1] == ":":
            clean_p += "\\"
        p_obj = Path(clean_p)
        if p_obj.exists():
            vsn, label, root = WindowsVolumeInspector.get_volume_info(p_obj)
            vol = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == vsn))
            if not vol:
                vol = WindowsVolume(
                    volume_serial_number=vsn,
                    volume_label=label,
                    last_known_drive_letter=root,
                    is_online=True,
                )
                db.add(vol)
                db.commit()
                db.refresh(vol)
            pref_vid = vol.id
            if parsed_targets is not None and vol.id not in parsed_targets:
                parsed_targets.append(vol.id)

    svc = StorageAutomationService(db)
    effective_pairs = selected_pairs if pairs_submitted else (selected_pairs or None)
    job_id = ProcessManager.create_job(
        job_type="backup_exec",
        title="Automatyczny Backup zdjęć i filmów do daf_backup/",
        item_ids=[],
        params={
            "preferred_target_volume_id": pref_vid,
            "target_volume_ids": parsed_targets,
            "minimize_transfers": eff_minimize,
            "selected_pairs": effective_pairs,
        },
        db=db,
    )
    exec_result = svc.execute_backup_and_deduplication(
        preferred_target_volume_id=pref_vid,
        job_id=job_id,
        operation_mode="backup_only",
        selected_pair_keys=effective_pairs,
        target_volume_ids=parsed_targets,
        minimize_transfers=eff_minimize,
    )
    if exec_result["pending_offline_pairs_count"] > 0:
        ProcessManager.update_progress(
            job_id,
            status="WAITING_FOR_DEVICE",
            message=f"Oczekiwanie na podłączenie kolejnej pary dysków (pozostało par: {exec_result['pending_offline_pairs_count']}).",
            db=db,
        )
    else:
        ProcessManager.finish_job(job_id, status="COMPLETED", message=exec_result["summary"], db=db)
    db.commit()

    return _render_backup_panel_response(
        request,
        db,
        pref_vid=pref_vid,
        selected_pairs=effective_pairs,
        operation_mode="backup_only",
        target_volume_ids=parsed_targets,
        minimize_transfers=eff_minimize,
        execution_result=exec_result,
        trigger_refresh=True,
    )


@router.get("/move-backups-panel")
@router.post("/move-backups-panel")
def get_move_backups_panel(
    request: Request,
    source_volume_id: Optional[str] = Form(None),
    target_volume_id: Optional[str] = Form(None),
    src_vid: Optional[str] = Query(None),
    dst_vid: Optional[str] = Query(None),
    include_in_vault: Optional[str] = Form(None),
    include_green_outside: Optional[str] = Form(None),
    include_orange_same_disk: Optional[str] = Form(None),
    types_submitted: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Displays and recalculates backup transfers between a user-selected pair of devices."""
    svc = StorageAutomationService(db)
    s_id = to_int(source_volume_id) or to_int(src_vid)
    t_id = to_int(target_volume_id) or to_int(dst_vid)
    inc_vault = to_bool(include_in_vault) if types_submitted else True
    inc_green = to_bool(include_green_outside) if types_submitted else True
    inc_orange = to_bool(include_orange_same_disk) if types_submitted else True
    analysis = svc.analyze_move_backups(
        source_volume_id=s_id,
        target_volume_id=t_id,
        include_in_vault=inc_vault,
        include_green_outside=inc_green,
        include_orange_same_disk=inc_orange,
    )
    return render_template(
        request,
        "partials/move_backups_panel.html",
        {"analysis": analysis, "execution_result": None},
    )


@router.post("/execute-move-backups")
def execute_move_backups_action(
    request: Request,
    source_volume_id: Optional[str] = Form(None),
    target_volume_id: Optional[str] = Form(None),
    include_in_vault: Optional[str] = Form(None),
    include_green_outside: Optional[str] = Form(None),
    include_orange_same_disk: Optional[str] = Form(None),
    types_submitted: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Moves verified backups between two online devices while keeping Original and Backup separated."""
    svc = StorageAutomationService(db)
    s_id = to_int(source_volume_id)
    t_id = to_int(target_volume_id)
    inc_vault = to_bool(include_in_vault) if types_submitted else True
    inc_green = to_bool(include_green_outside) if types_submitted else True
    inc_orange = to_bool(include_orange_same_disk) if types_submitted else True
    exec_res = svc.execute_move_backups(
        s_id or 0,
        t_id or 0,
        include_in_vault=inc_vault,
        include_green_outside=inc_green,
        include_orange_same_disk=inc_orange,
    )
    analysis = svc.analyze_move_backups(
        source_volume_id=s_id,
        target_volume_id=t_id,
        include_in_vault=inc_vault,
        include_green_outside=inc_green,
        include_orange_same_disk=inc_orange,
    )
    resp = render_template(
        request,
        "partials/move_backups_panel.html",
        {"analysis": analysis, "execution_result": exec_res},
    )
    if exec_res.get("moved_count", 0) > 0:
        resp.headers["HX-Trigger"] = "refresh-all"
    return resp


@router.post("/assign-group")
def assign_group(
    group_name: Optional[str] = Form(None),
    existing_group_id: Optional[str] = Form(None),
    parent_id: Optional[str] = Form(None),
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
        untagged_only=untagged_only, unvectored_only=unvectored_only,
        unanalyzed_quality_only=unanalyzed_quality_only,
    )

    ex_id = to_int(existing_group_id)
    par_id = to_int(parent_id)
    if par_id is not None and par_id <= 0:
        par_id = None

    if ex_id:
        grp = db.get(MediaGroup, ex_id)
        if not grp:
            return HTMLResponse("<span class='text-rose-400 text-xs'>Wybrany album nie istnieje.</span>")
    elif group_name and group_name.strip():
        clean_name = group_name.strip()
        grp = db.scalar(select(MediaGroup).where(MediaGroup.name == clean_name, MediaGroup.parent_id == par_id))
        if not grp:
            grp = MediaGroup(name=clean_name, parent_id=par_id)
            db.add(grp)
            db.commit()
            db.refresh(grp)
    else:
        return HTMLResponse("<span class='text-rose-400 text-xs'>Wybierz istniejący album z listy lub podaj nazwę nowego albumu.</span>")

    if not items:
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>Utworzono pusty album '{grp.name}'.</span>",
            headers={"HX-Trigger": "refresh-all"},
        )

    for it in items:
        it.group_id = grp.id
    db.commit()

    OperationLogger.log(
        "ALBUM_ASSIGN",
        f"Przypisano {len(items)} plików do albumu '{grp.name}' (ID: {grp.id}).",
        status="OK",
        db=db,
    )
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>Przypisano {len(items)} plików do albumu '{grp.name}'.</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/auto-group")
def auto_group_endpoint(
    mode: str = Form(...),
    scope: str = Form("filtered"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Requirement 2e: 1-Click Auto-Organization is restricted strictly to a selected device (volume_id)
    and directory (folder_prefix).
    """
    vid_int = to_int(volume_id)
    clean_folder = (folder_prefix or "").strip()
    if not vid_int or not clean_folder:
        return HTMLResponse(
            "<span class='text-amber-400 text-xs font-semibold'>"
            "⚠️ 1-Klik Auto-Organizacja jest dostępna wyłącznie po wybraniu konkretnego urządzenia (dysku) oraz katalogu."
            "</span>",
            status_code=400,
        )

    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
        untagged_only=untagged_only, unvectored_only=unvectored_only,
        unanalyzed_quality_only=unanalyzed_quality_only,
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików w wybranym katalogu urządzenia.</span>")

    svc = StorageAutomationService(db)
    if mode == "date":
        cnt = svc.auto_group_items_by_date(items)
        msg = f"Zorganizowano {cnt} plików z wybranego katalogu w drzewo albumów Rok -> Miesiąc."
    else:
        cnt = svc.auto_group_items_by_folder(items)
        msg = f"Zorganizowano {cnt} plików w drzewo albumów na podstawie wybranego katalogu z dysku."

    OperationLogger.log("AUTO_ORGANIZE", msg, status="OK", details={"volume_id": vid_int, "folder": clean_folder, "mode": mode}, db=db)
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/group-manage")
def manage_group(
    action: str = Form(...),
    target_group_id: int = Form(...),
    new_name: Optional[str] = Form(None),
    new_parent_id: Optional[str] = Form(None),
    destination_group_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    grp = db.get(MediaGroup, target_group_id)
    if not grp:
        return HTMLResponse("<span class='text-rose-400 text-xs'>Album nie istnieje.</span>")

    par_id = to_int(new_parent_id)
    dest_gid = to_int(destination_group_id)
    svc = StorageAutomationService(db)

    if action == "create_sub" and new_name and new_name.strip():
        sub = MediaGroup(name=new_name.strip(), parent_id=grp.id)
        db.add(sub)
        db.commit()
        msg = f"Utworzono podalbum '{sub.name}' wewnątrz '{grp.name}'."
        OperationLogger.log("ALBUM_CREATE_SUB", msg, status="OK", db=db)
    elif action == "rename" and new_name and new_name.strip():
        old_name = grp.name
        grp.name = new_name.strip()
        db.commit()
        msg = f"Zmieniono nazwę albumu z '{old_name}' na '{grp.name}'."
        OperationLogger.log("ALBUM_RENAME", msg, status="OK", db=db)
    elif action == "reparent":
        if par_id == grp.id:
            return HTMLResponse("<span class='text-rose-400 text-xs'>Album nie może być swoim własnym rodzicem.</span>")
        grp.parent_id = par_id if par_id and par_id > 0 else None
        db.commit()
        msg = f"Przeniesiono album '{grp.name}' w drzewie."
        OperationLogger.log("ALBUM_REPARENT", msg, status="OK", db=db)
    elif action == "delete":
        name = svc.delete_album_fast(target_group_id)
        msg = f"Usunięto album '{name}' (zdjęcia wróciły do swobodnej puli)."
    elif action in ("transfer_and_delete", "merge_into"):
        if not dest_gid:
            return HTMLResponse("<span class='text-rose-400 text-xs'>Wybierz docelowy album.</span>")
        try:
            moved_cnt, src_name, dst_name = svc.transfer_album_and_delete_source(target_group_id, dest_gid)
            msg = f"Przerzucono {moved_cnt} zdjęć z albumu '{src_name}' do '{dst_name}' i usunięto album '{src_name}'."
        except ValueError as exc:
            return HTMLResponse(f"<span class='text-rose-400 text-xs'>{exc}</span>")
    else:
        msg = "Nieznana operacja na albumie."

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/change-date")
def change_media_date(
    new_date: str = Form(...),
    album_id: Optional[str] = Form(None),
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Requirement 4a (Multimedia): Changes effective_date for an album or selected/filtered photos,
    and automatically removes them from the date anomaly list (has_date_anomaly = False).
    """
    clean_dt_str = new_date.strip().replace("T", " ")
    parsed_dt: Optional[datetime.datetime] = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed_dt = datetime.datetime.strptime(clean_dt_str, fmt)
            break
        except ValueError:
            continue

    if not parsed_dt:
        return HTMLResponse(
            "<span class='text-rose-400 text-xs'>Nieprawidłowy format daty (użyj RRRR-MM-DD lub RRRR-MM-DD GG:MM).</span>",
            status_code=400,
        )

    aid = to_int(album_id)
    if aid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(aid)
        items = list(db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == False)).all())
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
        )

    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych zdjęć do zmiany daty.</span>")

    for it in items:
        it.effective_date = parsed_dt
        it.has_date_anomaly = False
        it.date_anomaly_reason = None

    db.commit()
    msg = f"Zmieniono datę na {parsed_dt.strftime('%Y-%m-%d %H:%M')} dla {len(items)} zdjęć (usunięto z listy niezgodności dat)."
    OperationLogger.log("DATE_CHANGE", msg, status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>📅 {msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/clear-date-anomaly")
def clear_date_anomaly_without_changing_date(
    media_id: Optional[str] = Form(None),
    album_id: Optional[str] = Form(None),
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Requirement 4a (Multimedia): Removes photos from the invalid date (anomaly) list
    on demand without modifying their effective_date.
    """
    if media_id:
        items = list(db.scalars(select(MediaItem).where(MediaItem.id == media_id)).all())
    elif to_int(album_id):
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(to_int(album_id))
        items = list(db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == False)).all())
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
        )

    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak zdjęć do zatwierdzenia daty.</span>")

    cleared = 0
    for it in items:
        if it.has_date_anomaly or it.date_anomaly_reason:
            cleared += 1
        it.has_date_anomaly = False
        it.date_anomaly_reason = None

    db.commit()
    msg = f"Usunięto z listy niezgodności dat bez zmiany daty: {len(items)} zdjęć."
    OperationLogger.log("DATE_ANOMALY_CLEAR", msg, status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>✓ {msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/recalculate-date-anomalies")
def recalculate_date_anomalies_action(db: Session = Depends(get_db)):
    """Re-evaluates date discrepancies across folder name, filename, EXIF, and copy-modified timestamps."""
    svc = StorageAutomationService(db)
    res = svc.recalculate_date_anomalies()
    msg = (
        f"📅 Przeliczono źródła dat dla {res['evaluated']} plików: "
        f"wykryto rozbieżności / modyfikacje kopiowania: {res['anomalies_found']} | "
        f"automatycznie ustawiono prawdopodobną datę: {res['dates_updated']}."
    )
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/rotate-media")
def rotate_media_action(
    request: Request,
    media_id: Optional[str] = Form(None),
    album_id: Optional[str] = Form(None),
    delta: int = Form(90),
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Rotates a single photo, a selected list of photos, a filtered set, or an entire album hierarchy
    and persists rotation_angle (0, 90, 180, 270) in SQLite.
    """
    aid = to_int(album_id)
    if media_id:
        items = list(db.scalars(select(MediaItem).where(MediaItem.id == media_id)).all())
        if not items:
            return JSONResponse({"status": "error", "message": "Nie znaleziono zdjęcia."}, status_code=404)
    elif aid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(aid)
        items = list(
            db.scalars(
                select(MediaItem).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == False)
            ).all()
        )
    elif scope == "selected":
        if not selected_ids:
            return JSONResponse({"status": "error", "message": "Brak ID zdjęcia."}, status_code=400)
        items = list(db.scalars(select(MediaItem).where(MediaItem.id.in_(selected_ids))).all())
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
        )

    if not items:
        return JSONResponse({"status": "error", "message": "Brak zdjęć w wybranym zakresie do obrócenia."}, status_code=404)

    new_angles = {}
    for it in items:
        curr = it.rotation_angle or 0
        it.rotation_angle = (curr + delta) % 360
        new_angles[it.id] = it.rotation_angle

    db.commit()
    OperationLogger.log(
        "MEDIA_ROTATE",
        f"Obrócono zdjęcia ({len(items)} szt.) o {delta}° -> {new_angles}",
        status="OK",
        db=db,
    )

    accept = request.headers.get("accept", "")
    if "application/json" in accept or request.headers.get("x-requested-with") == "XMLHttpRequest":
        first_id = items[0].id
        return JSONResponse(
            {
                "status": "ok",
                "count": len(items),
                "media_id": first_id,
                "rotation_angle": new_angles[first_id],
                "angles": new_angles,
            }
        )

    if len(items) == 1:
        msg = f"↻ Obrócono zdjęcie i zapisano kąt ({items[0].rotation_angle}°) w bazie."
    else:
        msg = f"↻ Obrócono {len(items)} zdjęć o {delta}° i zapisano kąty w bazie."

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/consolidate")
def consolidate_group_endpoint(
    group_id: int = Form(...),
    target_volume_id: int = Form(...),
    db: Session = Depends(get_db),
):
    svc = StorageAutomationService(db)
    try:
        res = svc.consolidate_group(group_id, target_volume_id)
        msg = f"Skonsolidowano album: przeniesiono fizycznie {res['moved_physically']} plików"
        if res.get("skipped_offline", 0) > 0:
            msg += f" (pominięto {res['skipped_offline']} plików na odłączonych dyskach Offline)"
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>{msg}.</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    except Exception as e:
        return HTMLResponse(f"<span class='text-rose-400 text-xs font-semibold'>Błąd konsolidacji: {e}</span>")


def _filter_db_backups(cloud_files: list) -> list:
    return [
        f for f in cloud_files
        if any(str(f.get("name", "")).lower().endswith(ext) for ext in (
            ".db", ".sqlite", ".sqlite3", ".db.gz", ".sqlite.gz", ".sqlite3.gz"
        ))
    ]


def _load_gdrive_fallback(folder_id: Optional[str]) -> tuple:
    from app.cloud.google_drive import GoogleDriveManager

    folders = []
    cloud_files = []
    try:
        gdrive = GoogleDriveManager()
        folders = gdrive.list_folders()
        if folder_id:
            cloud_files = gdrive.list_files_in_folder(folder_id)
    except Exception as inner_exc:
        logger.debug("GDrive fallback listing failed: %s", inner_exc)
    return folders, cloud_files, _filter_db_backups(cloud_files)


def _get_volumes_and_medium_backups(
    db: Optional[Session] = None, selected_vol_id: Optional[int] = None
) -> Tuple[List[WindowsVolume], Optional[WindowsVolume], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import app.database as db_module

    online_vols: List[WindowsVolume] = []
    active_vol: Optional[WindowsVolume] = None
    if db is not None:
        try:
            volumes = db.scalars(select(WindowsVolume)).all()
            for v in volumes:
                p = Path(v.last_known_drive_letter)
                v.is_online = p.exists()
                if v.is_online:
                    online_vols.append(v)
            if selected_vol_id:
                active_vol = next((v for v in online_vols if v.id == selected_vol_id), None)
            if not active_vol and online_vols:
                active_vol = online_vols[0]
        except Exception as exc:
            logger.debug("Failed querying volumes: %s", exc)

    medium_backups: List[Dict[str, Any]] = []
    if active_vol:
        try:
            target_dir = Path(active_vol.last_known_drive_letter) / "daf_backup" / "db_backups"
            medium_backups = BackupManager.list_backups_in_dir(target_dir)
        except Exception as exc:
            logger.debug("Failed listing medium backups: %s", exc)

    local_backups: List[Dict[str, Any]] = []
    try:
        local_backups = BackupManager.list_backups_in_dir(db_module.BACKUP_DIR)
    except Exception as exc:
        logger.debug("Failed listing local backups: %s", exc)

    return online_vols, active_vol, medium_backups, local_backups


@router.get("/gdrive-panel")
def get_gdrive_panel(
    request: Request,
    folder_id: Optional[str] = None,
    medium_volume_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    from app.cloud.google_drive import GoogleDriveManager

    online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(
        db, selected_vol_id=medium_volume_id
    )

    active_backup_job = ProcessManager.get_active_backup_job(db)
    maint_stats = _get_maintenance_stats(db)

    try:
        gdrive = GoogleDriveManager()
        folders = gdrive.list_folders()
        if not folders:
            default_id = gdrive.get_or_create_folder("DomoweArchiwumFoto")
            folders = gdrive.list_folders()
        else:
            default_id = folders[0]["id"]

        active_fid = folder_id if (folder_id and folder_id.strip()) else default_id
        cloud_files = gdrive.list_files_in_folder(active_fid)

        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": active_fid,
                "cloud_files": cloud_files,
                "db_backups": _filter_db_backups(cloud_files),
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "active_backup_job": active_backup_job,
                "maint_stats": maint_stats,
                "error_msg": None,
                "status_report": False,
            },
        )
    except Exception as e:
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": [],
                "active_folder_id": "",
                "cloud_files": [],
                "db_backups": [],
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "active_backup_job": active_backup_job,
                "maint_stats": maint_stats,
                "error_msg": str(e),
                "status_report": False,
            },
        )


@router.get("/backup-job-status")
def get_backup_job_status(
    request: Request,
    folder_id: Optional[str] = None,
    medium_volume_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    job = ProcessManager.get_active_backup_job(db)
    if job and job.get("status") == "RUNNING":
        pct = max(job.get("processed_count", 0), 5)
        title = job.get("title", "Kopia bazy SQLite")
        msg = job.get("last_message", "")
        return HTMLResponse(
            f"""<div id="backup-live-progress" class="p-4 bg-indigo-950/90 border border-indigo-500 text-indigo-100 rounded-xl space-y-2.5 animate-pulse"
                 hx-get="/actions/backup-job-status?folder_id={folder_id or ''}&medium_volume_id={medium_volume_id or ''}"
                 hx-trigger="every 1s"
                 hx-target="#backup-live-progress"
                 hx-swap="outerHTML">
                <div class="flex items-center justify-between text-xs font-semibold text-indigo-300">
                    <span class="flex items-center space-x-2">
                        <svg class="animate-spin h-4 w-4 text-indigo-400" viewBox="0 0 24 24" fill="none">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                        </svg>
                        <span>⚙️ {title}</span>
                    </span>
                    <span class="font-mono font-bold text-indigo-200">
                        {job.get('processed_count', 0)}%
                    </span>
                </div>
                <div class="w-full bg-slate-900 rounded-full h-2.5 border border-indigo-700/60 overflow-hidden">
                    <div class="bg-indigo-500 h-2.5 rounded-full transition-all duration-300"
                         style="width: {pct}%;"></div>
                </div>
                <div class="text-[11px] text-slate-300 font-mono">
                    {msg}
                </div>
            </div>"""
        )
    return HTMLResponse('<div id="backup-live-progress"></div>')


@router.post("/gdrive-sync")
def sync_to_google_drive(
    request: Request,
    mode: str = Form("media"),
    exclude_embeddings: bool = Form(False),
    compress: bool = Form(True),
    folder_id: Optional[str] = Form(None),
    new_folder_name: Optional[str] = Form(None),
    purge_local: bool = Form(False),
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    from app.cloud.google_drive import GoogleDriveManager

    online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(db)
    job_id: Optional[int] = None
    try:
        gdrive = GoogleDriveManager()
        if new_folder_name and new_folder_name.strip():
            target_fid = gdrive.get_or_create_folder(new_folder_name.strip())
        elif folder_id and folder_id.strip():
            target_fid = folder_id.strip()
        else:
            target_fid = gdrive.get_or_create_folder("DomoweArchiwumFoto")

        uploaded_results = []
        job_id = None
        if mode == "db_backup":
            mode_desc = "lekki backup bez wektorów CLIP" if exclude_embeddings else "pełny backup z wektorami CLIP"
            comp_desc = "skompresowany gzip" if compress else "nieskompresowany .db"
            job_id = ProcessManager.create_job(
                job_type="GDRIVE_DB_BACKUP",
                title=f"Kopia bazy SQLite w Google Drive ({mode_desc}, {comp_desc})",
                item_ids=["db_gdrive_upload"],
                params={"folder_id": target_fid, "compress": compress, "exclude_embeddings": exclude_embeddings},
                db=db,
            )
            db.commit()

            def _prog(msg: str, done: int, total: int):
                pct = int(done * 0.6)
                if job_id:
                    ProcessManager.update_progress(job_id=job_id, processed_count=pct, total_count=100, message=msg)

            t_start = datetime.datetime.now()
            mgr = BackupManager()
            b_path = mgr.create_backup("gdrive_cloud", compress=compress, exclude_embeddings=exclude_embeddings, progress_callback=_prog)

            if job_id:
                ProcessManager.update_progress(job_id=job_id, processed_count=65, total_count=100, message="Wysyłanie kopii do chmury Google Drive...")
            res = gdrive.upload_and_verify(b_path, target_fid, purge_local_after_verify=False, keep_only_one_db=True)
            uploaded_results.append(res)

            if job_id:
                ProcessManager.update_progress(job_id=job_id, processed_count=90, total_count=100, message="Porządkowanie starszych kopii na Google Drive...")
            deleted_old = gdrive.prune_older_db_backups(target_fid, keep_file_id=res.get("id"))
            duration_sec = round((datetime.datetime.now() - t_start).total_seconds(), 1)
            size_mb = round(b_path.stat().st_size / (1024 * 1024), 2)
            mode_desc_str = " (lekki backup bez wektorów CLIP)" if exclude_embeddings else " (pełny backup z wektorami CLIP)"
            comp_desc_str = " [skompresowany gzip]" if compress else ""
            summary = f"Kopia zapasowa bazy SQLite{mode_desc_str}{comp_desc_str} ({b_path.name}, {size_mb} MB w {duration_sec} s) została zweryfikowana i zapisana w chmurze."
            if deleted_old:
                summary += f" Usunięto starsze kopie bazy z Google Drive ({len(deleted_old)} szt.: {', '.join(deleted_old)}), zachowując wyłącznie 1 najnowszą."

            if job_id:
                ProcessManager.finish_job(job_id=job_id, status="COMPLETED", message=summary)
        else:
            items = resolve_target_items(
                db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
                group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
                untagged_only=untagged_only, unvectored_only=unvectored_only,
                unanalyzed_quality_only=unanalyzed_quality_only, limit=100,
            )
            if purge_local:
                BackupManager().create_backup("pre_gdrive_purge")

            for it in items:
                vol = db.get(WindowsVolume, it.volume_id)
                if not vol:
                    continue
                local_p = Path(vol.last_known_drive_letter) / it.relative_path
                if not local_p.exists():
                    continue
                res = gdrive.upload_and_verify(local_p, target_fid, purge_local_after_verify=purge_local)
                uploaded_results.append(res)
                if res["purged_local"]:
                    db.delete(it)
            db.commit()
            summary = (
                f"Przesłano i zweryfikowano sumą MD5: {len(uploaded_results)} plików. "
                + ("Pliki lokalne zostały bezpiecznie usunięte." if purge_local else "Kopie lokalne pozostały na dysku.")
            )

        OperationLogger.log("GDRIVE_SYNC", summary, status="OK", db=db)
        folders = gdrive.list_folders()
        cloud_files = gdrive.list_files_in_folder(target_fid)

        resp = render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": target_fid,
                "cloud_files": cloud_files,
                "db_backups": _filter_db_backups(cloud_files),
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": None,
                "status_report": True,
                "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status_summary": summary,
                "uploaded_results": uploaded_results,
            },
        )
        resp.headers["HX-Trigger"] = "refresh-all"
        return resp

    except Exception as e:
        if job_id:
            ProcessManager.finish_job(job_id=job_id, status="FAILED", message=f"Błąd tworzenia kopii bazy w chmurze: {e}")
        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
                "db_backups": db_backups,
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": str(e),
                "status_report": False,
            },
        )


@router.post("/gdrive-restore-db")
def restore_db_from_google_drive(
    request: Request,
    file_id: str = Form(...),
    folder_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    from app.cloud.google_drive import GoogleDriveManager
    import app.database as db_module

    online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(db)
    temp_file: Optional[Path] = None
    try:
        gdrive = GoogleDriveManager()
        safe_fid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in file_id)
        remote_ext = ".db"
        try:
            if not gdrive.service:
                gdrive.authenticate()
            f_meta = gdrive.service.files().get(fileId=file_id, fields="name").execute()
            if str(f_meta.get("name", "")).lower().endswith(".gz"):
                remote_ext = ".db.gz"
        except Exception:
            pass
        temp_file = db_module.BACKUP_DIR / f"cloud_restore_{safe_fid}{remote_ext}"
        dl_info = gdrive.download_and_verify_backup(file_id, temp_file)

        # Release active session before overwriting database pages via SQLite backup API
        db.close()

        BackupManager().restore_from_backup(temp_file)

        summary = (
            f"Przywrócono bazę SQLite z kopii na Google Drive: {dl_info['name']} "
            f"(MD5 zweryfikowane: {dl_info['md5'][:12]}..., {dl_info['size_bytes']} B). "
            f"Poprzednia baza została zabezpieczona w data/backups/."
        )
        OperationLogger.log("GDRIVE_DB_RESTORE", summary, status="OK")

        folders = gdrive.list_folders()
        active_fid = folder_id.strip() if (folder_id and folder_id.strip()) else (folders[0]["id"] if folders else "")
        cloud_files = gdrive.list_files_in_folder(active_fid) if active_fid else []

        # Fresh volumes query after restore
        try:
            from app.database import SessionLocal
            with SessionLocal() as fresh_db:
                online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(fresh_db)
        except Exception:
            pass

        resp = render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": active_fid,
                "cloud_files": cloud_files,
                "db_backups": _filter_db_backups(cloud_files),
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": None,
                "status_report": True,
                "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status_summary": summary,
                "uploaded_results": [],
            },
        )
        resp.headers["HX-Trigger"] = "refresh-all"
        return resp
    except Exception as e:
        logger.error("Google Drive DB restore failed: %s", e)
        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
                "db_backups": db_backups,
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": str(e),
                "status_report": False,
            },
        )
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except OSError as exc:
                logger.debug("Could not clean temporary cloud restore file %s: %s", temp_file, exc)


@router.post("/backup-db-to-medium")
def backup_database_to_medium(
    request: Request,
    target_volume_id: Optional[int] = Form(None),
    custom_target_path: Optional[str] = Form(None),
    exclude_embeddings: bool = Form(False),
    compress: bool = Form(True),
    folder_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    job_id: Optional[int] = None
    target_label = "nośnik"
    try:
        target_dir: Optional[Path] = None
        vol: Optional[WindowsVolume] = None
        if target_volume_id:
            vol = db.get(WindowsVolume, target_volume_id)
            if not vol:
                raise ValueError("Wybrany wolumen nie istnieje w bazie.")
            drive_p = Path(vol.last_known_drive_letter)
            if not drive_p.exists():
                raise RuntimeError(f"Nośnik {vol.volume_label} ({vol.last_known_drive_letter}) jest obecnie odłączony (offline).")
            target_dir = drive_p / "daf_backup" / "db_backups"
            target_label = f"{vol.volume_label} ({vol.last_known_drive_letter})"
        elif custom_target_path and custom_target_path.strip():
            raw_path = Path(custom_target_path.strip())
            if str(raw_path.resolve()) == str(raw_path.anchor) or not raw_path.name:
                target_dir = raw_path / "daf_backup" / "db_backups"
            elif "daf_backup" in raw_path.parts:
                target_dir = raw_path
            else:
                target_dir = raw_path / "daf_backup" / "db_backups"
            target_label = str(target_dir)
        else:
            raise ValueError("Wskaż podłączony nośnik lub wprowadź prawidłową ścieżkę do katalogu.")

        target_dir.mkdir(parents=True, exist_ok=True)

        mode_desc = " (lekki backup bez wektorów CLIP)" if exclude_embeddings else " (pełny backup z wektorami CLIP)"
        comp_desc = " [skompresowany gzip]" if compress else ""

        job_id = ProcessManager.create_job(
            job_type="DB_BACKUP_MEDIUM",
            title=f"Kopia bazy SQLite na {target_label}{mode_desc}{comp_desc}",
            item_ids=["db_snapshot"],
            params={
                "target_dir": str(target_dir),
                "compress": compress,
                "exclude_embeddings": exclude_embeddings,
            },
            db=db,
        )
        db.commit()

        def _prog(msg: str, done: int, total: int):
            if job_id:
                ProcessManager.update_progress(
                    job_id=job_id,
                    processed_count=done,
                    total_count=total,
                    message=msg,
                )

        t_start = datetime.datetime.now()
        bm = BackupManager()
        saved_file = bm.create_backup(
            tag="medium",
            target_dir=target_dir,
            compress=compress,
            exclude_embeddings=exclude_embeddings,
            progress_callback=_prog,
        )
        duration_sec = round((datetime.datetime.now() - t_start).total_seconds(), 1)
        size_mb = round(saved_file.stat().st_size / (1024 * 1024), 2)
        summary = (
            f"Kopia zapasowa bazy SQLite{mode_desc}{comp_desc} została pomyślnie utworzona na nośniku {target_label}: "
            f"{saved_file.name} ({size_mb} MB w {duration_sec} s) w kartotece {target_dir}."
        )

        if job_id:
            ProcessManager.finish_job(job_id=job_id, status="COMPLETED", message=summary)
        OperationLogger.log("DB_BACKUP_MEDIUM", summary, status="OK", db=db)

        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(
            db, selected_vol_id=vol.id if vol else target_volume_id
        )

        resp = render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or (folders[0]["id"] if folders else ""),
                "cloud_files": cloud_files,
                "db_backups": db_backups,
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": None,
                "status_report": True,
                "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status_summary": summary,
                "uploaded_results": [],
            },
        )
        resp.headers["HX-Trigger"] = "refresh-all"
        return resp

    except Exception as e:
        if job_id:
            ProcessManager.finish_job(job_id=job_id, status="FAILED", message=f"Błąd tworzenia kopii bazy: {e}")
        OperationLogger.log("DB_BACKUP_MEDIUM", f"Błąd tworzenia kopii bazy: {e}", status="ERROR", db=db)
        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(
            db, selected_vol_id=target_volume_id
        )
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
                "db_backups": db_backups,
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": str(e),
                "status_report": False,
            },
        )


@router.post("/restore-db-from-medium")
def restore_database_from_medium(
    request: Request,
    backup_file_path: Optional[str] = Form(None),
    target_volume_id: Optional[int] = Form(None),
    selected_backup_filename: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    try:
        source_path: Optional[Path] = None
        if backup_file_path and backup_file_path.strip():
            source_path = Path(backup_file_path.strip())
        elif target_volume_id and selected_backup_filename and selected_backup_filename.strip():
            vol = db.get(WindowsVolume, target_volume_id)
            if not vol:
                raise ValueError("Wybrany wolumen nie istnieje.")
            source_path = Path(vol.last_known_drive_letter) / "daf_backup" / "db_backups" / selected_backup_filename.strip()
        else:
            raise ValueError("Wskaż plik kopii zapasowej do przywrócenia.")

        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Plik kopii zapasowej nie istnieje: {source_path}")

        # Close active DB session before overwriting database
        db.close()

        bm = BackupManager()
        bm.restore_from_backup(source_path)

        size_mb = round(source_path.stat().st_size / (1024 * 1024), 2)
        summary = (
            f"Pomyślnie przywrócono bazę SQLite z nośnika lokalnego: {source_path.name} "
            f"({size_mb} MB). Bieżący stan sprzed przywrócenia zabezpieczono w data/backups/."
        )
        OperationLogger.log("DB_RESTORE_MEDIUM", summary, status="OK")

        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        online_vols, active_vol, medium_backups, local_backups = [], None, [], []
        try:
            from app.database import SessionLocal
            with SessionLocal() as fresh_db:
                online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(
                    fresh_db, selected_vol_id=target_volume_id
                )
        except Exception:
            pass

        resp = render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
                "db_backups": db_backups,
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": None,
                "status_report": True,
                "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status_summary": summary,
                "uploaded_results": [],
            },
        )
        resp.headers["HX-Trigger"] = "refresh-all"
        return resp

    except Exception as e:
        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        online_vols, active_vol, medium_backups, local_backups = [], None, [], []
        try:
            online_vols, active_vol, medium_backups, local_backups = _get_volumes_and_medium_backups(
                db, selected_vol_id=target_volume_id
            )
        except Exception:
            pass
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
                "db_backups": db_backups,
                "online_volumes": online_vols,
                "active_medium_vol": active_vol,
                "medium_backups": medium_backups,
                "local_backups": local_backups,
                "error_msg": str(e),
                "status_report": False,
            },
        )


@router.post("/export-json-disk")
def export_json_to_disk(
    scope: str = Form("filtered"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
        untagged_only=untagged_only, unvectored_only=unvectored_only,
        unanalyzed_quality_only=unanalyzed_quality_only,
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików do eksportu.</span>")

    export_dir = DATA_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = export_dir / f"domowe_archiwum_export_{ts}.json"

    payload = []
    for it in items:
        vol = db.get(WindowsVolume, it.volume_id)
        grp = db.get(MediaGroup, it.group_id) if it.group_id else None
        payload.append(
            {
                "id": it.id,
                "file_name": it.file_name,
                "relative_path": it.relative_path,
                "volume_label": vol.volume_label if vol else None,
                "volume_vsn": vol.volume_serial_number if vol else None,
                "drive_letter": vol.last_known_drive_letter if vol else None,
                "album": grp.name if grp else None,
                "effective_date": it.effective_date.isoformat(),
                "exif_date": it.exif_date.isoformat() if it.exif_date else None,
                "file_size_bytes": it.file_size_bytes,
                "dimensions": {"width": it.width, "height": it.height},
                "rotation_angle": it.rotation_angle,
                "file_hash": it.file_hash,
                "is_duplicate": not it.is_primary_duplicate,
                "is_backup_copy": it.is_backup_copy,
                "is_blurry": it.is_blurry,
                "blur_score": it.blur_score,
            }
        )

    dest_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OperationLogger.log("EXPORT_JSON", f"Zapisano eksport JSON ({len(payload)} poz.) do {dest_file.name}", status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>"
        f"✓ Zapisano {len(payload)} rekordów na dysku: <code class='bg-slate-900 px-1.5 py-0.5 rounded text-white'>{dest_file}</code>"
        f"</span>"
    )


@router.post("/move-files")
def move_physical_files_action(
    destination_path: Optional[str] = Form(None),
    target_volume_id: Optional[str] = Form(None),
    target_subfolder: Optional[str] = Form(""),
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    q: Optional[str] = Form(None),
    media_type: str = Form("all"),
    volume_id: Optional[str] = Form(None),
    folder_prefix: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    anomaly_only: bool = Form(False),
    blurry_only: bool = Form(False),
    duplicates_only: bool = Form(False),
    untagged_only: bool = Form(False),
    unvectored_only: bool = Form(False),
    unanalyzed_quality_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Moves selected or filtered files physically between online devices and folders."""
    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
        untagged_only=untagged_only, unvectored_only=unvectored_only,
        unanalyzed_quality_only=unanalyzed_quality_only,
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików do przeniesienia.</span>")

    svc = StorageAutomationService(db)
    tv_id = to_int(target_volume_id)
    sub_folder = (target_subfolder or "").strip()

    try:
        if tv_id:
            res = svc.schedule_or_execute_moves(items, target_volume_id=tv_id, target_subfolder=sub_folder)
            msg = f"🚚 Przeniesiono fizycznie: {res['moved_physically']} plików"
            if res["skipped_offline"] > 0:
                msg += f" | Pominięto (dysk Offline): {res['skipped_offline']} plików"
            return HTMLResponse(
                f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
                headers={"HX-Trigger": "refresh-all"},
            )

        if not destination_path or not destination_path.strip():
            return HTMLResponse("<span class='text-rose-400 text-xs'>Wybierz dysk docelowy lub podaj ścieżkę.</span>")

        clean_dest = destination_path.strip(' \t\n\r"\'')
        if len(clean_dest) == 2 and clean_dest[1] == ":":
            clean_dest += "\\"
        dest_dir = Path(clean_dest)
        drive_part = os.path.splitdrive(str(dest_dir))[0]
        drive_root = (drive_part.rstrip("\\/") + "\\") if drive_part else "C:\\"

        if not Path(drive_root).exists():
            return HTMLResponse(
                f"<span class='text-rose-400 text-xs'>Dysk docelowy {drive_root} jest odłączony (Offline). Podłącz oba dyski, aby przenieść pliki.</span>"
            )

        dest_dir_resolved = dest_dir.resolve()
        dest_dir_resolved.mkdir(parents=True, exist_ok=True)
        vsn, label, root = WindowsVolumeInspector.get_volume_info(dest_dir_resolved)
        root_path = Path(root).resolve()

        target_vol = db.scalar(select(WindowsVolume).where(WindowsVolume.volume_serial_number == vsn))
        if not target_vol:
            target_vol = WindowsVolume(
                volume_serial_number=vsn,
                volume_label=label,
                last_known_drive_letter=root,
                is_online=True,
            )
            db.add(target_vol)
            db.commit()
            db.refresh(target_vol)
        else:
            target_vol.last_known_drive_letter = root
            target_vol.is_online = True
            db.commit()

        try:
            rel_sub = dest_dir_resolved.relative_to(root_path).as_posix()
            if rel_sub == ".":
                rel_sub = ""
        except ValueError:
            rel_sub = ""

        res = svc.schedule_or_execute_moves(items, target_volume_id=target_vol.id, target_subfolder=rel_sub)
        msg = f"🚚 Przeniesiono fizycznie {res['moved_physically']} plików do {dest_dir_resolved}"
        if res["skipped_offline"] > 0:
            msg += f" | Pominięto (dysk źródłowy Offline): {res['skipped_offline']}"
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    except Exception as e:
        return HTMLResponse(f"<span class='text-rose-400 text-xs font-semibold'>Błąd przenoszenia: {e}</span>")


def _execute_move_to_trash_items(db: Session, item_ids: List[str], job_id: Optional[int] = None) -> Tuple[int, int]:
    """Sets logical KOSZ attribute (is_trashed=True) in SQLite without touching or moving physical files on disk."""
    items = _fetch_items_by_ids_chunked(db, item_ids)
    processed = 0

    for idx, it in enumerate(items):
        if job_id and ProcessManager.is_stop_requested(job_id):
            ProcessManager.update_progress(
                job_id,
                remaining_ids=[x.id for x in items[idx:]],
                status="STOPPED",
                message=f"Zatrzymano przenoszenie do kosza po {processed}/{len(items)}.",
                db=db,
            )
            db.commit()
            return processed, 0

        it.is_trashed = True
        processed += 1

    if job_id:
        ProcessManager.finish_job(
            job_id,
            status="COMPLETED",
            message=f"Oznaczono jako KOSZ w bazie (bez ruszania plików na dysku): {processed} pozycji.",
            db=db,
        )
    db.commit()
    return processed, 0


def _build_safe_removal_html_response(
    action_url: str,
    delete_mode: str,
    res: Dict[str, Any],
    job_id: int,
    db: Session,
) -> HTMLResponse:
    import html as html_mod

    processed = res["processed_count"]
    physical = res["physical_count"]
    blocked_safety = res["blocked_safety"]
    blocked_verify = res["blocked_verify"]
    offline_items = res["offline_items"]
    offline_ids = res["offline_ids"]
    nxt = res["next_offline_prompt"]

    if offline_items and nxt:
        nxt_del = int(nxt.get("delete_count", nxt.get("count", 0)) or 0)
        nxt_ref = int(nxt.get("verify_only_count", 0) or 0)
        if nxt_del > 0:
            nxt_job_msg = (
                f"Przetworzono online: {processed} | Podłącz dysk {nxt['vol_label']} ({nxt['drive_letter']}) "
                f"— do usunięcia na tym dysku: {nxt_del} duplikatów (łącznie pozostało: {len(offline_ids)})."
            )
        else:
            nxt_job_msg = (
                f"Przetworzono online: {processed} | Podłącz dysk referencyjny {nxt['vol_label']} ({nxt['drive_letter']}) "
                f"(0 duplikatów na dysku, {nxt_ref} plików do weryfikacji 100%)."
            )
        ProcessManager.update_progress(
            job_id,
            remaining_ids=offline_ids,
            status="WAITING_FOR_DEVICE",
            message=nxt_job_msg,
            db=db,
        )
    else:
        verb = "Przeniesiono do Kosza" if delete_mode == "trash" else "Trwale usunięto z dysku"
        ProcessManager.finish_job(
            job_id,
            status="COMPLETED",
            message=f"{verb}: {processed} (fizycznie: {physical}, zablokowano bezpieczeństwem: {blocked_safety + blocked_verify}).",
            db=db,
        )
    db.commit()

    parts: List[str] = []
    if delete_mode == "trash":
        parts.append(
            f"<span class='text-amber-300 font-semibold'>🗑️ Oznaczono jako KOSZ (baza, pliki fizyczne nietknięte): {processed}"
        )
    else:
        parts.append(
            f"<span class='text-rose-300 font-semibold'>🔥 Trwale usunięto z dysku: {physical} (z bazy: {processed})"
        )
    if res["processed_drives"]:
        parts.append(f" | Dyski: {', '.join(res['processed_drives'])}")
    if blocked_safety > 0:
        parts.append(f" | 🛡️ Zablokowano (brak Oryginału + Backupu): {blocked_safety}")
    if blocked_verify > 0:
        parts.append(f" | ⚠️ Zablokowano (błąd weryfikacji 100%): {blocked_verify}")
    parts.append("</span>")

    if nxt and offline_ids:
        vals_payload = html_mod.escape(
            json.dumps(
                {
                    "scope": "selected",
                    "duplicates_only": "true",
                    "skip_db_backup": "true",
                    "continue_job_id": str(job_id),
                    "selected_ids_json": json.dumps(offline_ids),
                }
            )
        )
        nxt_del = int(nxt.get("delete_count", nxt.get("count", 0)) or 0)
        other_drives_summary = ", ".join(
            f"{html_mod.escape(d['vol_label'])} ({html_mod.escape(d['drive_letter'])}): {int(d.get('delete_count', d.get('count', 0)) or 0)} szt."
            for d in res["offline_drives"]
        )
        parts.append(
            f"<div class='mt-1.5 p-2 rounded-lg bg-amber-950/90 border border-amber-500 text-amber-100 flex flex-wrap items-center justify-between gap-2 text-[11px]'>"
            f"<span>🔌 <strong>Podłącz kolejny dysk ({len(res['offline_drives'])} w kolejce)</strong>: "
            f"<u>{html_mod.escape(nxt['vol_label'])}</u> ({html_mod.escape(nxt['drive_letter'])}, VSN: {html_mod.escape(nxt['vsn'])}) "
            f"— do usunięcia na tym dysku: <strong>{nxt_del}</strong> duplikatów "
            f"(łącznie na odłączonych dyskach: <strong>{len(offline_ids)}</strong> szt."
            f"{f' [{other_drives_summary}]' if other_drives_summary else ''}).</span>"
            f"<button type='button' hx-post='{action_url}' hx-vals=\"{vals_payload}\" hx-target='#global-notification' "
            f"class='bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 rounded font-semibold shadow cursor-pointer'>"
            f"▶️ Kontynuuj na podłączonym dysku ({nxt_del} szt.)"
            f"</button></div>"
        )

    return HTMLResponse("".join(parts), headers={"HX-Trigger": "refresh-all"})


DUP_DELETE_STATE: Dict[str, Any] = {
    "running": False,
    "stop_requested": False,
    "job_id": None,
    "delete_mode": "permanent",
    "stage": "backup",
    "bkp_pct": 0,
    "processed": 0,
    "total": 0,
    "current_drive": "",
    "phase": "",
    "final_html": "",
    "error_msg": "",
}


def _run_duplicate_removal_worker(
    job_id: int,
    item_ids: List[str],
    delete_mode: str,
    force_duplicate_verification: bool,
    action_url: str,
    skip_db_backup: bool = False,
) -> None:
    import html as html_mod

    try:
        if not skip_db_backup:
            DUP_DELETE_STATE["stage"] = "backup"
            DUP_DELETE_STATE["bkp_pct"] = 5
            DUP_DELETE_STATE["phase"] = "Etap 1/2: Tworzenie bezpiecznej kopii bazy SQLite przed usunięciem..."
            ProcessManager.update_cache_only(job_id, message=DUP_DELETE_STATE["phase"])

            def _on_bkp_progress(msg: str, done: int, total: int) -> None:
                pct = int((done * 100) / max(1, total))
                DUP_DELETE_STATE["bkp_pct"] = min(100, max(0, pct))
                full_msg = f"Etap 1/2 (Kopia bazy): {msg}"
                DUP_DELETE_STATE["phase"] = full_msg
                # RAM-only update: NEVER write to SQLite during sqlite3.Connection.backup()
                ProcessManager.update_cache_only(job_id, message=f"{full_msg} ({pct}%)")

            BackupManager().create_backup(
                "pre_trash_move" if delete_mode == "trash" else "pre_permanent_disk_delete",
                progress_callback=_on_bkp_progress,
            )

        DUP_DELETE_STATE["stage"] = "delete"
        DUP_DELETE_STATE["bkp_pct"] = 100
        DUP_DELETE_STATE["phase"] = f"Wczytywanie {len(item_ids)} plików i weryfikacja klastrów..."
        ProcessManager.update_cache_only(job_id, message=DUP_DELETE_STATE["phase"])

        with SessionLocal() as db:
            def _on_progress(processed: int, total: int, message: str, drive_text: str) -> None:
                DUP_DELETE_STATE["processed"] = processed
                DUP_DELETE_STATE["total"] = total
                DUP_DELETE_STATE["phase"] = f"Etap 2/2: {message}"
                DUP_DELETE_STATE["current_drive"] = drive_text

            res = StorageAutomationService(db).execute_safe_duplicate_removal(
                item_ids=item_ids,
                delete_mode=delete_mode,
                force_duplicate_verification=force_duplicate_verification,
                job_id=job_id,
                trash_dir=DATA_DIR / "kosz",
                thumbnails_dir=THUMBNAILS_DIR,
                progress_cb=_on_progress,
            )
            DUP_DELETE_STATE["processed"] = len(item_ids)
            resp = _build_safe_removal_html_response(action_url, delete_mode, res, job_id, db)
            body_html = resp.body.decode("utf-8")
            if res.get("stopped_by_user"):
                ProcessManager.finish_job(
                    job_id,
                    status="STOPPED",
                    message=f"Zatrzymano przez użytkownika (przetworzono: {res['processed_count']}).",
                    db=db,
                )
                db.commit()
                body_html = (
                    "<span class='text-amber-300 font-semibold'>⏹️ Zatrzymano usuwanie duplikatów: </span>"
                    + body_html
                )
            DUP_DELETE_STATE["final_html"] = body_html
    except Exception as exc:
        logger.exception("Duplicate removal worker error: %s", exc)
        DUP_DELETE_STATE["error_msg"] = str(exc)
        ProcessManager.finish_job(job_id, status="FAILED", message=f"Błąd usuwania: {exc}")
        DUP_DELETE_STATE["final_html"] = (
            "<span class='text-rose-400 text-xs font-semibold'>❌ Błąd podczas usuwania duplikatów: "
            f"{html_mod.escape(str(exc))}</span>"
        )
    finally:
        DUP_DELETE_STATE["running"] = False


def _start_duplicate_removal_async(
    db: Session,
    item_ids: List[str],
    delete_mode: str,
    force_duplicate_verification: bool,
    action_url: str,
    skip_db_backup: bool = False,
    existing_job_id: Optional[int] = None,
) -> HTMLResponse:
    if DUP_DELETE_STATE["running"]:
        return get_dup_delete_status()

    title_prefix = "Przenoszenie duplikatów do Kosza" if delete_mode == "trash" else "Trwałe usuwanie duplikatów z dysku"
    job_type = "move_to_trash" if delete_mode == "trash" else "delete_permanently"
    job_id = None
    if existing_job_id:
        existing_job = db.get(BackgroundJob, existing_job_id)
        if existing_job:
            existing_job.status = "RUNNING"
            existing_job.processed_count = 0
            existing_job.total_count = len(item_ids)
            existing_job.remaining_ids_json = json.dumps(item_ids, ensure_ascii=False)
            existing_job.last_message = f"Kontynuacja usuwania duplikatów ({len(item_ids)} poz.)..."
            existing_job.updated_at = datetime.datetime.now(datetime.UTC)
            db.commit()
            job_id = existing_job.id
    if job_id is None:
        job_id = ProcessManager.create_job(
            job_type=job_type,
            title=f"{title_prefix} ({len(item_ids)} poz.)",
            item_ids=item_ids,
            params={"is_duplicate_removal": True, "delete_mode": delete_mode},
            db=db,
        )
    DUP_DELETE_STATE.update(
        {
            "running": True,
            "stop_requested": False,
            "job_id": job_id,
            "delete_mode": delete_mode,
            "stage": "delete" if skip_db_backup else "backup",
            "bkp_pct": 100 if skip_db_backup else 0,
            "processed": 0,
            "total": len(item_ids),
            "current_drive": "",
            "phase": "Kontynuacja na kolejnym dysku..." if skip_db_backup else "Etap 1/2: Przygotowanie kopii zapasowej bazy SQLite...",
            "final_html": "",
            "error_msg": "",
        }
    )
    ProcessManager.start_background_thread(
        job_id,
        _run_duplicate_removal_worker,
        item_ids,
        delete_mode,
        force_duplicate_verification,
        action_url,
        skip_db_backup,
    )
    return get_dup_delete_status()


@router.get("/dup-delete-status")
def get_dup_delete_status():
    import html as html_mod

    if DUP_DELETE_STATE["running"]:
        stage = DUP_DELETE_STATE.get("stage", "delete")
        processed = int(DUP_DELETE_STATE["processed"] or 0)
        total = max(1, int(DUP_DELETE_STATE["total"] or 1))
        if stage == "backup":
            pct = min(100, max(0, int(DUP_DELETE_STATE.get("bkp_pct") or 0)))
            headline = f"🛡️ Etap 1/2: Kopia bazy SQLite ({pct}%) — do usunięcia: {total} szt."
            bar_cls = "bg-amber-500"
            title_cls = "text-amber-300"
        else:
            pct = min(100, max(0, int((processed / total) * 100)))
            is_trash = DUP_DELETE_STATE.get("delete_mode") == "trash"
            icon = "🗑️" if is_trash else "🔥"
            verb = "Przenoszenie duplikatów do Kosza" if is_trash else "Trwałe usuwanie duplikatów"
            headline = f"{icon} Etap 2/2: {verb} ({processed} / {total}, {pct}%)"
            bar_cls = "bg-amber-500" if is_trash else "bg-rose-500"
            title_cls = "text-amber-300" if is_trash else "text-rose-300"
        phase = html_mod.escape(str(DUP_DELETE_STATE.get("phase") or "Trwa bezpieczne usuwanie duplikatów..."))
        return HTMLResponse(
            f"<div id='dup-delete-progress-box' hx-get='/actions/dup-delete-status' hx-trigger='every 1s' hx-swap='outerHTML' "
            f"class='w-full max-w-2xl bg-slate-900/95 border border-rose-700/70 rounded-lg px-3 py-2 text-xs space-y-1.5 shadow'>"
            f"<div class='flex items-center justify-between gap-2'>"
            f"<span class='font-bold {title_cls} animate-pulse'>{headline}</span>"
            f"<div class='flex items-center gap-2'>"
            f"<button type='button' onclick='openStatusModal()' "
            f"class='bg-slate-800 hover:bg-slate-700 text-sky-300 px-2 py-0.5 rounded border border-borderBg text-[10px] font-semibold'>⚙️ Status i Log</button>"
            f"<button type='button' hx-post='/actions/dup-delete-stop' hx-target='#global-notification' "
            f"class='bg-rose-800 hover:bg-rose-700 text-white px-2 py-0.5 rounded text-[10px] font-semibold'>⏹️ Stop</button>"
            f"</div></div>"
            f"<div class='w-full bg-slate-800 rounded-full h-1.5 overflow-hidden'>"
            f"<div class='{bar_cls} h-1.5 transition-all' style='width: {pct}%'></div></div>"
            f"<div class='text-[10px] text-slate-300 truncate'>{phase}</div>"
            f"</div>"
        )

    if DUP_DELETE_STATE.get("final_html"):
        return HTMLResponse(DUP_DELETE_STATE["final_html"], headers={"HX-Trigger": "refresh-all"})
    return HTMLResponse("<span class='text-slate-400 text-xs'>Brak aktywnego usuwania duplikatów.</span>")


@router.post("/dup-delete-stop")
def stop_dup_delete():
    DUP_DELETE_STATE["stop_requested"] = True
    DUP_DELETE_STATE["phase"] = "Zatrzymywanie operacji po bieżącym pliku..."
    jid = DUP_DELETE_STATE.get("job_id")
    if jid:
        ProcessManager.request_stop(jid)
    return get_dup_delete_status()


@router.post("/move-to-trash")
async def move_items_to_local_trash(
    request: Request,
    db: Session = Depends(get_db),
):
    form = await _read_large_form(request)
    scope = _form_str(form, "scope", "selected") or "selected"
    selected_ids = [str(x) for x in form.getlist("selected_ids")]
    selected_ids_json = _form_str(form, "selected_ids_json")
    t_gid = to_int(_form_str(form, "trash_group_id"))
    q = _form_str(form, "q")
    media_type = _form_str(form, "media_type", "all") or "all"
    volume_id = _form_str(form, "volume_id")
    folder_prefix = _form_str(form, "folder_prefix")
    group_id = _form_str(form, "group_id")
    tag = _form_str(form, "tag")
    year = _form_str(form, "year")
    anomaly_only = _form_bool(form, "anomaly_only")
    blurry_only = _form_bool(form, "blurry_only")
    duplicates_only = _form_bool(form, "duplicates_only")
    redundant_clusters_only = _form_bool(form, "redundant_clusters_only")
    backups_to_move_only = _form_bool(form, "backups_to_move_only")
    untagged_only = _form_bool(form, "untagged_only")
    unvectored_only = _form_bool(form, "unvectored_only")
    unanalyzed_quality_only = _form_bool(form, "unanalyzed_quality_only")
    backup_filter = _form_str(form, "backup_filter", "none") or "none"
    backup_types = [str(x) for x in form.getlist("backup_types")]
    backup_mode = _form_str(form, "backup_mode", "hide") or "hide"
    anomaly_type = _form_str(form, "anomaly_type", "all") or "all"
    run_sync = _form_bool(form, "run_sync")
    skip_db_backup = _form_bool(form, "skip_db_backup")
    continue_job_id = to_int(_form_str(form, "continue_job_id"))
    is_htmx = request.headers.get("HX-Request") == "true"

    if t_gid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(t_gid)
        item_ids = [
            str(x)
            for x in db.scalars(
                select(MediaItem.id).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == False)
            ).all()
        ]
        for gid in subtree_ids:
            g_obj = db.get(MediaGroup, gid)
            if g_obj:
                g_obj.is_trashed = True
    else:
        item_ids = _resolve_bulk_ids_fast(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
            selected_ids_json=selected_ids_json,
            redundant_clusters_only=redundant_clusters_only,
            backups_to_move_only=backups_to_move_only,
            backup_filter=backup_filter,
            backup_types=backup_types,
            backup_mode=backup_mode,
            anomaly_type=anomaly_type,
        )

    if not item_ids and not t_gid:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do przeniesienia do kosza.</span>")

    has_dup_items = bool(duplicates_only or (item_ids and _has_duplicate_role_in_ids(db, item_ids)))

    if has_dup_items and not t_gid and is_htmx and not run_sync:
        return _start_duplicate_removal_async(
            db=db,
            item_ids=item_ids,
            delete_mode="trash",
            force_duplicate_verification=bool(duplicates_only),
            action_url="/actions/move-to-trash",
            skip_db_backup=skip_db_backup,
            existing_job_id=continue_job_id,
        )

    if not skip_db_backup:
        BackupManager().create_backup("pre_trash_move")
    job_id = ProcessManager.create_job(
        job_type="move_to_trash",
        title=f"Przenoszenie do Kosza ({len(item_ids)} poz.)",
        item_ids=item_ids,
        db=db,
    )
    if has_dup_items and not t_gid:
        res = StorageAutomationService(db).execute_safe_duplicate_removal(
            item_ids=item_ids,
            delete_mode="trash",
            force_duplicate_verification=bool(duplicates_only),
            job_id=job_id,
            trash_dir=DATA_DIR / "kosz",
            thumbnails_dir=THUMBNAILS_DIR,
        )
        return _build_safe_removal_html_response("/actions/move-to-trash", "trash", res, job_id, db)

    processed, moved_files = _execute_move_to_trash_items(db, item_ids, job_id=job_id)

    return HTMLResponse(
        f"<span class='text-amber-300 text-xs font-semibold'>"
        f"🗑️ Oznaczono jako KOSZ w bazie (atrybut KOSZ, pliki fizyczne na dysku nietknięte): {processed} pozycji. Możesz przywrócić z widoku Kosza."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/restore-from-trash")
def restore_from_trash(
    restore_mode: str = Form("selected"),
    restore_group_id: Optional[str] = Form(None),
    selected_ids: List[str] = Form(default=[]),
    db: Session = Depends(get_db),
):
    trash_dir = DATA_DIR / "kosz"
    r_gid = to_int(restore_group_id)

    if restore_mode == "group" and r_gid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(r_gid)
        for gid in subtree_ids:
            g_obj = db.get(MediaGroup, gid)
            if g_obj:
                g_obj.is_trashed = False
                curr_p = g_obj.parent_id
                while curr_p:
                    p_obj = db.get(MediaGroup, curr_p)
                    if p_obj:
                        p_obj.is_trashed = False
                        curr_p = p_obj.parent_id
                    else:
                        break
        items = list(db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == True)).all())
    elif restore_mode == "selected" and selected_ids:
        items = list(db.scalars(select(MediaItem).where(MediaItem.id.in_(selected_ids), MediaItem.is_trashed == True)).all())
    else:
        items = list(db.scalars(select(MediaItem).where(MediaItem.is_trashed == True)).all())

    restored_files = 0
    restored_logical = 0

    for it in items:
        vol = db.get(WindowsVolume, it.volume_id)
        trash_file = trash_dir / f"{it.id}_{it.file_name}"

        if vol and Path(vol.last_known_drive_letter).exists() and trash_file.exists():
            orig_full_path = Path(vol.last_known_drive_letter) / it.relative_path
            orig_parent_dir = orig_full_path.parent
            orig_parent_dir.mkdir(parents=True, exist_ok=True)

            target_restore_path = orig_full_path
            c = 1
            while target_restore_path.exists():
                target_restore_path = orig_parent_dir / f"{orig_full_path.stem}_restored_{c}{orig_full_path.suffix}"
                c += 1
            shutil.move(str(trash_file), str(target_restore_path))
            it.file_name = target_restore_path.name
            it.relative_path = target_restore_path.resolve().relative_to(Path(vol.last_known_drive_letter).resolve()).as_posix()
            restored_files += 1

        it.is_trashed = False
        restored_logical += 1
        if it.group_id:
            g_obj = db.get(MediaGroup, it.group_id)
            if g_obj:
                g_obj.is_trashed = False

    db.commit()
    msg = f"♻️ Przywrócono z Kosza: {restored_logical} pozycji (fizycznie na dysk: {restored_files})."
    OperationLogger.log("TRASH_RESTORE", msg, status="OK", db=db)

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/empty-trash")
def empty_trash_permanently(
    selected_ids: List[str] = Form(default=[]),
    db: Session = Depends(get_db),
):
    trash_dir = DATA_DIR / "kosz"
    stmt = select(MediaItem).where(MediaItem.is_trashed == True)
    if selected_ids:
        stmt = stmt.where(MediaItem.id.in_(selected_ids))
    items = list(db.scalars(stmt).all())

    deleted_cnt = 0
    for it in items:
        trash_file = trash_dir / f"{it.id}_{it.file_name}"
        if trash_file.exists():
            try:
                trash_file.unlink()
            except OSError as exc:
                logger.debug("Could not unlink trash file %s: %s", trash_file, exc)
        thumb_p = THUMBNAILS_DIR / it.thumbnail_relative_path
        if thumb_p.exists():
            try:
                thumb_p.unlink()
            except OSError as exc:
                logger.debug("Could not unlink thumbnail %s: %s", thumb_p, exc)
        db.delete(it)
        deleted_cnt += 1

    if not selected_ids:
        for tg in db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == True)).all():
            db.delete(tg)

    db.commit()
    OperationLogger.log("TRASH_EMPTY", f"Trwale opróżniono Kosz ({deleted_cnt} plików).", status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-rose-400 text-xs font-semibold'>🔥 Trwale usunięto z Kosza: {deleted_cnt} plików.</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/rename-volume")
def rename_volume_action(
    volume_id: int = Form(...),
    new_label: str = Form(...),
    db: Session = Depends(get_db),
):
    vol = db.get(WindowsVolume, volume_id)
    if not vol or not new_label.strip():
        return HTMLResponse("<span class='text-rose-400 text-xs'>Nieprawidłowa nazwa dysku.</span>")

    old_label = vol.volume_label
    vol.volume_label = new_label.strip()[:64]
    db.commit()
    OperationLogger.log("VOLUME_RENAME", f"Zmieniono nazwę dysku z '{old_label}' na '{vol.volume_label}'.", status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>✓ Zmieniono nazwę dysku na: {vol.volume_label}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


UNINDEX_BATCH_STATE = {
    "running": False,
    "job_id": None,
    "processed": 0,
    "total": 0,
    "removed_vol_label": None,
}


def _run_unindex_worker(
    job_id: int,
    item_rows: List[Tuple[str, str]],
    u_vid: Optional[int],
):
    UNINDEX_BATCH_STATE["running"] = True
    UNINDEX_BATCH_STATE["job_id"] = job_id
    UNINDEX_BATCH_STATE["processed"] = 0
    UNINDEX_BATCH_STATE["total"] = len(item_rows)
    removed_cnt = 0
    try:
        chunk_size = 500
        for start in range(0, len(item_rows), chunk_size):
            batch = item_rows[start : start + chunk_size]
            batch_ids = [r[0] for r in batch]
            for _, thumb_rel in batch:
                if thumb_rel:
                    thumb_p = THUMBNAILS_DIR / thumb_rel
                    if thumb_p.exists():
                        try:
                            thumb_p.unlink()
                        except OSError as exc:
                            logger.debug("Unindex thumbnail cleanup warning: %s", exc)

            with SessionLocal() as db:
                db.execute(delete(MediaTag).where(MediaTag.media_id.in_(batch_ids)))
                db.execute(delete(MediaEmbedding).where(MediaEmbedding.media_id.in_(batch_ids)))
                db.execute(delete(MediaItem).where(MediaItem.id.in_(batch_ids)))
                db.commit()

            removed_cnt += len(batch_ids)
            UNINDEX_BATCH_STATE["processed"] = removed_cnt
            ProcessManager.update_progress(
                job_id,
                processed_delta=len(batch_ids),
                remaining_ids=[r[0] for r in item_rows[start + chunk_size :]],
                message=f"Wyindeksowano {removed_cnt} / {len(item_rows)} pozycji...",
            )

        with SessionLocal() as db:
            if u_vid:
                vol = db.get(WindowsVolume, u_vid)
                if vol:
                    db.delete(vol)
            OperationLogger.log(
                "UNINDEX",
                f"Wyindeksowano z bazy {removed_cnt} pozycji (bez usuwania plików).",
                status="OK",
                db=db,
            )
            ProcessManager.finish_job(
                job_id,
                status="COMPLETED",
                message=f"Usunięto z indeksu bazy {removed_cnt} pozycji.",
                db=db,
            )
            db.commit()
    except Exception as exc:
        logger.error("Unindex worker failure: %s", exc)
        ProcessManager.finish_job(job_id, status="FAILED", message=f"Błąd wyindeksowania: {exc}")
    finally:
        UNINDEX_BATCH_STATE["running"] = False


@router.get("/unindex-status")
def get_unindex_status():
    st = UNINDEX_BATCH_STATE
    if st["running"]:
        pct = int((st["processed"] / st["total"]) * 100) if st["total"] else 0
        return HTMLResponse(
            f"<div hx-get='/actions/unindex-status' hx-trigger='every 1s' hx-swap='outerHTML' class='flex items-center space-x-2 text-xs'>"
            f"<span class='text-sky-300 font-semibold animate-pulse'>⏳ Wyindeksowanie dysku z bazy w toku: {st['processed']} / {st['total']} ({pct}%)...</span>"
            f"</div>"
        )
    return HTMLResponse(
        f"<span class='text-sky-300 text-xs font-semibold'>"
        f"✖️ Usunięto z indeksu bazy: {st['processed']} pozycji (oryginalne pliki na dysku pozostały nienaruszone)."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/unindex")
async def unindex_items_from_db(
    request: Request,
    db: Session = Depends(get_db),
):
    form = await _read_large_form(request)
    scope = _form_str(form, "scope", "selected") or "selected"
    selected_ids = [str(x) for x in form.getlist("selected_ids")]
    selected_ids_json = _form_str(form, "selected_ids_json")
    u_vid = to_int(_form_str(form, "unindex_volume_id"))
    q = _form_str(form, "q")
    media_type = _form_str(form, "media_type", "all") or "all"
    volume_id = _form_str(form, "volume_id")
    folder_prefix = _form_str(form, "folder_prefix")
    group_id = _form_str(form, "group_id")
    tag = _form_str(form, "tag")
    year = _form_str(form, "year")
    anomaly_only = _form_bool(form, "anomaly_only")
    blurry_only = _form_bool(form, "blurry_only")
    duplicates_only = _form_bool(form, "duplicates_only")
    untagged_only = _form_bool(form, "untagged_only")
    unvectored_only = _form_bool(form, "unvectored_only")
    unanalyzed_quality_only = _form_bool(form, "unanalyzed_quality_only")
    run_sync = _form_bool(form, "run_sync")

    if u_vid:
        rows = db.execute(
            select(MediaItem.id, MediaItem.thumbnail_relative_path).where(MediaItem.volume_id == u_vid)
        ).all()
        item_rows = [(str(r[0]), str(r[1] or "")) for r in rows]
        vol = db.get(WindowsVolume, u_vid)
        vol_label = vol.volume_label if vol else f"#{u_vid}"
        title = f"Wyindeksowanie dysku {vol_label} ({len(item_rows)} poz.)"
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
            selected_ids_json=selected_ids_json,
        )
        item_rows = [(it.id, it.thumbnail_relative_path or "") for it in items]
        title = f"Wyindeksowanie z bazy ({len(item_rows)} poz.)"

    if not item_rows and not u_vid:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do wyindeksowania z bazy.</span>")

    job_id = ProcessManager.create_job(
        job_type="unindex",
        title=title,
        item_ids=[r[0] for r in item_rows],
        params={"unindex_volume_id": u_vid},
        db=db,
    )

    if len(item_rows) > 300 and not run_sync:
        ProcessManager.start_background_thread(job_id, _run_unindex_worker, item_rows, u_vid)
        return get_unindex_status()

    _run_unindex_worker(job_id, item_rows, u_vid)
    return HTMLResponse(
        f"<span class='text-sky-300 text-xs font-semibold'>"
        f"✖️ Usunięto z indeksu bazy: {len(item_rows)} pozycji (oryginalne pliki na dysku pozostały nienaruszone)."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


def _execute_delete_permanently_items(db: Session, item_ids: List[str], job_id: Optional[int] = None) -> Tuple[int, int]:
    trash_dir = DATA_DIR / "kosz"
    items = _fetch_items_by_ids_chunked(db, item_ids)
    deleted_disk = 0
    removed_db = 0

    for idx, it in enumerate(items):
        if job_id and ProcessManager.is_stop_requested(job_id):
            ProcessManager.update_progress(
                job_id,
                remaining_ids=[x.id for x in items[idx:]],
                status="STOPPED",
                message=f"Zatrzymano trwałe usuwanie po {removed_db}/{len(items)}.",
                db=db,
            )
            db.commit()
            return deleted_disk, removed_db

        vol = db.get(WindowsVolume, it.volume_id)
        if vol:
            src_p = Path(vol.last_known_drive_letter) / it.relative_path
            if src_p.exists():
                try:
                    src_p.unlink()
                    deleted_disk += 1
                except OSError as exc:
                    logger.debug("Permanent delete unlink warning: %s", exc)

        trash_p = trash_dir / f"{it.id}_{it.file_name}"
        if trash_p.exists():
            try:
                trash_p.unlink()
                deleted_disk += 1
            except OSError as exc:
                logger.debug("Permanent delete trash unlink warning: %s", exc)

        thumb_p = THUMBNAILS_DIR / it.thumbnail_relative_path
        if thumb_p.exists():
            try:
                thumb_p.unlink()
            except OSError as exc:
                logger.debug("Permanent delete thumb unlink warning: %s", exc)

        db.delete(it)
        removed_db += 1

    if job_id:
        ProcessManager.finish_job(
            job_id,
            status="COMPLETED",
            message=f"Trwale usunięto z dysku: {deleted_disk} plików (z bazy: {removed_db}).",
            db=db,
        )
    db.commit()
    return deleted_disk, removed_db


@router.post("/delete-permanently")
async def delete_files_permanently_from_disk(
    request: Request,
    db: Session = Depends(get_db),
):
    form = await _read_large_form(request)
    scope = _form_str(form, "scope", "selected") or "selected"
    selected_ids = [str(x) for x in form.getlist("selected_ids")]
    selected_ids_json = _form_str(form, "selected_ids_json")
    d_gid = to_int(_form_str(form, "delete_group_id"))
    q = _form_str(form, "q")
    media_type = _form_str(form, "media_type", "all") or "all"
    volume_id = _form_str(form, "volume_id")
    folder_prefix = _form_str(form, "folder_prefix")
    group_id = _form_str(form, "group_id")
    tag = _form_str(form, "tag")
    year = _form_str(form, "year")
    anomaly_only = _form_bool(form, "anomaly_only")
    blurry_only = _form_bool(form, "blurry_only")
    duplicates_only = _form_bool(form, "duplicates_only")
    redundant_clusters_only = _form_bool(form, "redundant_clusters_only")
    backups_to_move_only = _form_bool(form, "backups_to_move_only")
    untagged_only = _form_bool(form, "untagged_only")
    unvectored_only = _form_bool(form, "unvectored_only")
    unanalyzed_quality_only = _form_bool(form, "unanalyzed_quality_only")
    backup_filter = _form_str(form, "backup_filter", "none") or "none"
    backup_types = [str(x) for x in form.getlist("backup_types")]
    backup_mode = _form_str(form, "backup_mode", "hide") or "hide"
    anomaly_type = _form_str(form, "anomaly_type", "all") or "all"
    run_sync = _form_bool(form, "run_sync")
    skip_db_backup = _form_bool(form, "skip_db_backup")
    continue_job_id = to_int(_form_str(form, "continue_job_id"))
    is_htmx = request.headers.get("HX-Request") == "true"

    grp_to_delete = None
    if d_gid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(d_gid)
        item_ids = [
            str(x)
            for x in db.scalars(select(MediaItem.id).where(MediaItem.group_id.in_(subtree_ids))).all()
        ]
        grp_to_delete = db.get(MediaGroup, d_gid)
    else:
        item_ids = _resolve_bulk_ids_fast(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
            selected_ids_json=selected_ids_json,
            redundant_clusters_only=redundant_clusters_only,
            backups_to_move_only=backups_to_move_only,
            backup_filter=backup_filter,
            backup_types=backup_types,
            backup_mode=backup_mode,
            anomaly_type=anomaly_type,
        )

    if not item_ids and not grp_to_delete:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do trwałego usunięcia.</span>")

    has_dup_items = bool(duplicates_only or (item_ids and _has_duplicate_role_in_ids(db, item_ids)))

    if has_dup_items and not d_gid and is_htmx and not run_sync:
        return _start_duplicate_removal_async(
            db=db,
            item_ids=item_ids,
            delete_mode="permanent",
            force_duplicate_verification=bool(duplicates_only),
            action_url="/actions/delete-permanently",
            skip_db_backup=skip_db_backup,
            existing_job_id=continue_job_id,
        )

    if not skip_db_backup:
        BackupManager().create_backup("pre_permanent_disk_delete")
    job_id = ProcessManager.create_job(
        job_type="delete_permanently",
        title=f"Trwałe usuwanie plików z dysku ({len(item_ids)} poz.)",
        item_ids=item_ids,
        db=db,
    )
    if has_dup_items and not d_gid:
        res = StorageAutomationService(db).execute_safe_duplicate_removal(
            item_ids=item_ids,
            delete_mode="permanent",
            force_duplicate_verification=bool(duplicates_only),
            job_id=job_id,
            trash_dir=DATA_DIR / "kosz",
            thumbnails_dir=THUMBNAILS_DIR,
        )
        return _build_safe_removal_html_response("/actions/delete-permanently", "permanent", res, job_id, db)

    deleted_disk, removed_db = _execute_delete_permanently_items(db, item_ids, job_id=job_id)

    if grp_to_delete and not ProcessManager.is_stop_requested(job_id):
        db.delete(grp_to_delete)
        db.commit()

    return HTMLResponse(
        f"<span class='text-rose-400 text-xs font-semibold'>"
        f"🔥 Trwale usunięto fizycznie z dysku: {deleted_disk} plików (usunięto z bazy: {removed_db})."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/verify-db-disk-consistency")
def verify_db_disk_consistency_action(
    volume_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Checks all online, VSN-verified drives against SQLite records without altering existing AI/album metadata."""
    import html as html_mod

    svc = StorageAutomationService(db)
    res = svc.verify_db_disk_consistency(volume_id=to_int(volume_id))
    missing = res["missing_items"]
    online_vols = res["checked_online_volumes"]
    offline_vols = res["offline_volumes"]

    vols_html = "".join(
        f"<div class='px-2.5 py-1.5 rounded bg-slate-950/70 border border-borderBg text-[11px] flex justify-between'>"
        f"<span>🟢 <strong>{html_mod.escape(v['label'])}</strong> ({html_mod.escape(v['drive_letter'])}, VSN: {html_mod.escape(v['vsn'])})</span>"
        f"<span class='font-mono'>OK: <strong class='text-emerald-400'>{v['valid_count']}</strong> | Brak: <strong class='text-rose-400'>{v['missing_count']}</strong></span>"
        f"</div>"
        for v in online_vols
    )
    if offline_vols:
        vols_html += "".join(
            f"<div class='px-2.5 py-1.5 rounded bg-slate-950/40 border border-amber-800/50 text-[11px] flex justify-between text-amber-300'>"
            f"<span>🔴 OFFLINE (pominięto weryfikację plików): {html_mod.escape(v['label'])} ({html_mod.escape(v['drive_letter'])})</span>"
            f"<span class='font-mono'>W bazie: {v['db_count']} szt.</span>"
            f"</div>"
            for v in offline_vols
        )

    preview_rows = ""
    if missing:
        rows_str = "".join(
            f"<tr class='border-b border-borderBg/40'>"
            f"<td class='py-1 px-2 text-slate-200 truncate max-w-[160px]'>{html_mod.escape(m['file_name'])}</td>"
            f"<td class='py-1 px-2 text-amber-300'>{html_mod.escape(m['volume_label'])}</td>"
            f"<td class='py-1 px-2 text-slate-400 truncate max-w-[220px]'>{html_mod.escape(m['relative_path'])}</td>"
            f"<td class='py-1 px-2 font-mono'>{'✅ Ma kopię (' + str(m['other_copies_count']) + ')' if m['other_copies_count'] > 0 else '⚠️ Jedyna kopia w bazie'}</td>"
            f"</tr>"
            for m in missing[:200]
        )
        preview_rows = (
            "<div class='max-h-48 overflow-y-auto border border-borderBg rounded-lg mt-2'>"
            "<table class='w-full text-left text-[10px] font-mono'>"
            "<thead><tr class='bg-slate-950 text-slate-400 border-b border-borderBg'>"
            "<th class='py-1 px-2'>Plik</th><th class='py-1 px-2'>Dysk</th><th class='py-1 px-2'>Ścieżka</th><th class='py-1 px-2'>Kopie w bazie</th>"
            "</tr></thead><tbody>" + rows_str + "</tbody></table></div>"
        )

    cleanup_btn = ""
    if res["missing_count"] > 0:
        cleanup_btn = (
            f"<button type='button' "
            f"hx-post='/actions/cleanup-missing-db-items' "
            f"hx-confirm='Czy na pewno usunąć z bazy {res['missing_count']} rekordów plików, których fizycznie NIE MA na podłączonych dyskach? Pozostałe pliki i dane AI pozostaną nietknięte.' "
            f"hx-target='#maint-result-status' "
            f"class='mt-2 bg-rose-700 hover:bg-rose-600 text-white px-3.5 py-2 rounded-lg font-bold shadow'>"
            f"🗑️ Decyzja: Usuń z bazy {res['missing_count']} brakujących rekordów</button>"
        )

    return HTMLResponse(
        f"<div class='p-3 rounded-xl bg-slate-950/90 border border-sky-700/70 space-y-2 text-xs'>"
        f"<div class='font-bold text-sky-300'>🔍 Raport spójności Baza ↔ Dyski (dane AI i albumy zachowane)</div>"
        f"<div class='grid grid-cols-2 md:grid-cols-4 gap-2 font-mono text-[11px]'>"
        f"<div class='p-2 rounded bg-slate-900 border border-borderBg'>Zweryfikowano na dyskach: <strong class='text-emerald-400'>{res['valid_online_count']}</strong></div>"
        f"<div class='p-2 rounded bg-slate-900 border border-borderBg'>Brakuje fizycznie: <strong class='text-rose-400'>{res['missing_count']}</strong></div>"
        f"<div class='p-2 rounded bg-slate-900 border border-borderBg'>Brakujących z kopią w bazie: <strong class='text-teal-300'>{res['missing_with_backup_count']}</strong></div>"
        f"<div class='p-2 rounded bg-slate-900 border border-borderBg'>Różny rozmiar na dysku: <strong class='text-amber-300'>{res['size_mismatch_count']}</strong></div>"
        f"</div>"
        f"<div class='space-y-1'>{vols_html}</div>"
        f"{preview_rows}{cleanup_btn}"
        f"</div>"
    )


@router.post("/cleanup-missing-db-items")
def cleanup_missing_db_items_action(
    volume_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Deletes SQLite records for files verified missing on VSN-verified online volumes."""
    res = StorageAutomationService(db).cleanup_missing_db_items(volume_id=to_int(volume_id))
    return HTMLResponse(
        f"<div class='p-2.5 rounded-lg bg-emerald-950/70 border border-emerald-600 text-emerald-100'>"
        f"✅ Usunięto z bazy <strong>{res['removed_count']}</strong> rekordów brakujących plików "
        f"(pominięto istniejące lub na dyskach offline: {res['skipped_exists_or_offline']})."
        f"</div>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/organize-originals")
def organize_originals_action(
    volume_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Moves ungrouped Originals into daf_oryginal/YYYY/MM/DD on the same disk while preserving original_relative_path."""
    res = StorageAutomationService(db).organize_ungrouped_originals(volume_id=to_int(volume_id))
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>🗂️ {res['message']}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/export-physical-album")
async def export_physical_album_action(
    request: Request,
    db: Session = Depends(get_db),
):
    """Copies album/filtered/selected photos to daf_albums/<AlbumName>/ on a chosen disk without indexing copies."""
    form = await _read_large_form(request)
    album_id = to_int(_form_str(form, "album_id"))
    target_volume_id = to_int(_form_str(form, "target_volume_id"))
    custom_name = _form_str(form, "custom_album_name")

    if not target_volume_id:
        return HTMLResponse("<span class='text-rose-400 text-xs font-semibold'>Wybierz urządzenie docelowe dla fizycznego eksportu albumu.</span>")

    if album_id:
        svc = StorageAutomationService(db)
        grp = db.get(MediaGroup, album_id)
        subtree_ids = svc.get_group_subtree_ids(album_id)
        items = list(
            db.scalars(
                select(MediaItem).where(
                    MediaItem.group_id.in_(subtree_ids),
                    MediaItem.is_trashed == False,
                    MediaItem.is_backup_copy == False,
                )
            ).all()
        )
        eff_name = (custom_name or "").strip() or (grp.name if grp else f"Album_{album_id}")
    else:
        scope = _form_str(form, "scope", "selected") or "selected"
        selected_ids = [str(x) for x in form.getlist("selected_ids")]
        selected_ids_json = _form_str(form, "selected_ids_json")
        q = _form_str(form, "q")
        media_type = _form_str(form, "media_type", "all") or "all"
        volume_id = _form_str(form, "volume_id")
        folder_prefix = _form_str(form, "folder_prefix")
        group_id = _form_str(form, "group_id")
        tag = _form_str(form, "tag")
        year = _form_str(form, "year")
        anomaly_only = _form_bool(form, "anomaly_only")
        blurry_only = _form_bool(form, "blurry_only")
        duplicates_only = _form_bool(form, "duplicates_only")
        untagged_only = _form_bool(form, "untagged_only")
        unvectored_only = _form_bool(form, "unvectored_only")
        unanalyzed_quality_only = _form_bool(form, "unanalyzed_quality_only")
        backup_filter = _form_str(form, "backup_filter", "none") or "none"
        backup_types = [str(x) for x in form.getlist("backup_types")]
        backup_mode = _form_str(form, "backup_mode", "hide") or "hide"
        anomaly_type = _form_str(form, "anomaly_type", "all") or "all"
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
            selected_ids_json=selected_ids_json,
            backup_filter=backup_filter, backup_types=backup_types, backup_mode=backup_mode,
            anomaly_type=anomaly_type,
        )
        grp_cur = db.get(MediaGroup, to_int(group_id)) if to_int(group_id) else None
        eff_name = (custom_name or "").strip() or (grp_cur.name if grp_cur else "Wybrany_Album")

    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs font-semibold'>Brak plików do wyeksportowania do daf_albums/.</span>")

    res = StorageAutomationService(db).export_physical_album(items, target_volume_id, eff_name)
    color_cls = "text-emerald-400" if res["status"] == "ok" else "text-amber-300"
    return HTMLResponse(f"<span class='{color_cls} text-xs font-semibold'>📤 {res['message']}</span>")


@router.post("/verify-deleted-duplicates")
def verify_deleted_duplicates_action(db: Session = Depends(get_db)):
    """Displays post-deduplication SQL verification proving every deleted duplicate has an active Original."""
    import html as html_mod

    res = StorageAutomationService(db).verify_deleted_duplicates_integrity()
    ok_cls = "bg-emerald-950/70 border-emerald-600 text-emerald-100" if res["missing_orig_count"] == 0 else "bg-amber-950/70 border-amber-600 text-amber-100"
    snap_note = f"<div class='text-[10px] text-slate-400 font-mono'>Kopia referencyjna: {html_mod.escape(res['snapshot_name'])}</div>" if res.get("snapshot_name") else ""
    return HTMLResponse(
        f"<div class='p-3 rounded-xl border {ok_cls} text-xs space-y-1'>"
        f"<div class='font-bold'>✅ Weryfikacja spójności po usunięciu duplikatów:</div>"
        f"<div>{html_mod.escape(res['report_text'])}</div>"
        f"{snap_note}"
        f"</div>"
    )


@router.post("/clear-quality-defect")
async def clear_quality_defect_action(request: Request, db: Session = Depends(get_db)):
    """Marks selected/filtered items (or a single media_id) as defect-free without trashing files."""
    form = await _read_large_form(request)
    media_id = _form_str(form, "media_id")
    if media_id and media_id.strip():
        item_ids = [media_id.strip()]
    else:
        item_ids = _resolve_bulk_ids_fast(
            db,
            scope=_form_str(form, "scope", "selected") or "selected",
            selected_ids=[str(x) for x in form.getlist("selected_ids")],
            q=_form_str(form, "q"),
            media_type=_form_str(form, "media_type", "all") or "all",
            volume_id=_form_str(form, "volume_id"),
            folder_prefix=_form_str(form, "folder_prefix"),
            group_id=_form_str(form, "group_id"),
            tag=_form_str(form, "tag"),
            year=_form_str(form, "year"),
            anomaly_only=_form_bool(form, "anomaly_only"),
            blurry_only=_form_bool(form, "blurry_only"),
            duplicates_only=_form_bool(form, "duplicates_only"),
            untagged_only=_form_bool(form, "untagged_only"),
            ungrouped_only=_form_bool(form, "ungrouped_only"),
            unbacked_only=_form_bool(form, "unbacked_only"),
            unvectored_only=_form_bool(form, "unvectored_only"),
            unanalyzed_quality_only=_form_bool(form, "unanalyzed_quality_only"),
            quality_defect=_form_str(form, "quality_defect"),
            selected_ids_json=_form_str(form, "selected_ids_json"),
            redundant_clusters_only=_form_bool(form, "redundant_clusters_only"),
            backups_to_move_only=_form_bool(form, "backups_to_move_only"),
            backup_filter=_form_str(form, "backup_filter", "none") or "none",
            backup_types=[str(x) for x in form.getlist("backup_types")],
            backup_mode=_form_str(form, "backup_mode", "hide") or "hide",
            anomaly_type=_form_str(form, "anomaly_type", "all") or "all",
        )

    if not item_ids:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików do oznaczenia jako nieuszkodzone / bez wady.</span>")

    for idx in range(0, len(item_ids), 5000):
        chunk = item_ids[idx : idx + 5000]
        db.execute(
            update(MediaItem)
            .where(MediaItem.id.in_(chunk))
            .values(is_corrupted=False, is_blurry=False)
        )
        db.execute(
            update(MediaItem)
            .where(
                MediaItem.id.in_(chunk),
                or_(MediaItem.blur_score.is_(None), MediaItem.blur_score < 45.0),
            )
            .values(blur_score=100.0)
        )
        db.execute(
            delete(MediaTag).where(
                MediaTag.media_id.in_(chunk),
                MediaTag.tag.ilike("do_usuniecia%"),
            )
        )

    db.commit()
    msg = f"Oznaczono jako bez wady / nieuszkodzone: {len(item_ids)} zdjęć."
    OperationLogger.log("QUALITY_DEFECT_CLEAR", msg, status="OK", db=db)
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>✓ {msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )

