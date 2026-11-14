from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import traceback
from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import select, delete, update
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
    PendingFileMove,
    BackgroundJob,
)
from app.operation_log import OperationLogger, ProcessManager
from app.scanner import WindowsVolumeInspector, FileScannerService, MetadataDateResolver, ThumbnailService, IMAGE_EXTS, VIDEO_EXTS
from app.storage_ops import StorageAutomationService
from app.ai.quality import MediaQualityAnalyzer
from app.ai.tagger import OnDemandObjectTagger
from app.ai.embeddings import OnDemandClipEmbedder
from app.routers.media import apply_media_filters, to_int, render_template

logger = logging.getLogger("ActionsRouter")
router = APIRouter(prefix="/actions", tags=["Actions"])


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
    limit: Optional[int] = None,
) -> List[MediaItem]:
    if scope == "selected" and selected_ids:
        stmt = select(MediaItem).where(MediaItem.id.in_(selected_ids))
    elif scope == "all_drives":
        stmt = select(MediaItem).where(MediaItem.is_trashed == False)
    elif scope.startswith("volume:"):
        vid = to_int(scope.split(":", 1)[1])
        stmt = select(MediaItem).where(MediaItem.is_trashed == False, MediaItem.volume_id == vid)
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
                select(MediaItem.relative_path, MediaItem.id, MediaItem.thumbnail_relative_path)
                .where(MediaItem.volume_id == vol.id)
            ).all()
            existing_map = {r[0]: (r[1], r[2]) for r in existing_rows}
            existing_map_lower = {r[0].lower(): (r[0], r[1], r[2]) for r in existing_rows}

            candidates_to_add = []
            queued_rels_lower: set[str] = set()
            ids_to_delete: List[str] = []
            found_total = 0
            already_in_db = 0
            repaired_thumbs = 0

            for r, _, files in os.walk(target):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                        continue

                    full = Path(r) / f
                    found_total += 1
                    rel = _compute_scan_relative_path(full, abs_root, root_path, abs_target)
                    rel_low = rel.lower()

                    if rel_low in queued_rels_lower:
                        continue

                    in_existing = rel in existing_map or rel_low in existing_map_lower
                    if in_existing and not force_rescan:
                        already_in_db += 1
                        item_id, thumb_name = existing_map.get(rel) or existing_map_lower[rel_low][1:]
                        thumb_file = THUMBNAILS_DIR / thumb_name
                        if not thumb_file.exists() or thumb_file.stat().st_size == 0:
                            try:
                                ThumbnailService.generate(full, item_id, ext in IMAGE_EXTS)
                                repaired_thumbs += 1
                            except Exception as exc:
                                logger.debug("Thumbnail repair failed for %s: %s", full, exc)
                    else:
                        if in_existing and force_rescan:
                            del_id = existing_map[rel][0] if rel in existing_map else existing_map_lower[rel_low][1]
                            ids_to_delete.append(del_id)
                        queued_rels_lower.add(rel_low)
                        candidates_to_add.append((full, rel))

            if ids_to_delete:
                for i in range(0, len(ids_to_delete), 500):
                    db.execute(delete(MediaItem).where(MediaItem.id.in_(ids_to_delete[i : i + 500])))
                db.commit()

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
                    f"Skanowanie {label} ({vsn}): znaleziono {found_total}, nowych: 0, już w bazie: {already_in_db}",
                    status="OK",
                    db=db,
                )

            return {
                "status": "ok",
                "found_total": found_total,
                "indexed": indexed,
                "already_in_db": already_in_db,
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
        f"Dodano nowych: {res['indexed']} | Już w bazie: {res['already_in_db']}"
    )
    if res["repaired_thumbs"] > 0:
        msg += f" | Odświeżono miniatur: {res['repaired_thumbs']}"

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
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
            msg = f"Szybka analiza duplikatów (baza SHA-256 / hash): oznaczono {dup_total} duplikatów w indeksie."
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


@router.get("/status-panel")
def get_status_and_logs_panel(request: Request, db: Session = Depends(get_db)):
    summary = ProcessManager.get_jobs_summary(db)
    logs = OperationLogger.get_recent_logs(db, limit=100)
    volumes = StorageAutomationService(db).refresh_volumes_online_status()
    return render_template(
        request,
        "partials/status_panel.html",
        {
            "jobs_summary": summary,
            "logs": logs,
            "volumes": volumes,
            "ai_state": AI_BATCH_STATE,
        },
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

        if not rem_ids and job.job_type not in ("backup_dedup", "todo_sync"):
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
            elif job.job_type == "delete_permanently":
                _execute_delete_permanently_items(db, rem_ids, job_id=job.id)
            elif job.job_type == "move_to_trash":
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


def _render_backup_panel_response(
    request: Request,
    db: Session,
    pref_vid: Optional[int] = None,
    selected_pairs: Optional[List[str]] = None,
    operation_mode: str = "both",
    execution_result: Optional[dict] = None,
    offline_report: Optional[dict] = None,
    trigger_refresh: bool = False,
):
    """Builds and renders the Backup & Deduplication modal partial with pair filtering state."""
    svc = StorageAutomationService(db)
    plan = svc.build_backup_and_dedup_plan(preferred_target_volume_id=pref_vid)
    norm_mode = operation_mode if operation_mode in ("both", "backup_only", "dedup_only") else "both"
    all_pair_key_strs = [p["pair_key_str"] for p in plan["ordered_pairs"]]
    effective_pair_keys = (
        [k for k in selected_pairs if k and ":" in k]
        if selected_pairs is not None
        else all_pair_key_strs
    )
    all_unbacked = svc.detect_files_without_backup(recluster=True)
    pair_map = plan.get("unbacked_pair_map", {})
    all_duplicate_rows = plan.get("duplicate_rows", [])
    if selected_pairs is not None:
        selected_set = set(effective_pair_keys)
        filtered_unbacked_count = sum(1 for uf in all_unbacked if pair_map.get(uf.id, "") in selected_set)
        filtered_dups_count = sum(1 for dr in all_duplicate_rows if dr.get("pair_key_str", "") in selected_set)
    else:
        filtered_unbacked_count = len(all_unbacked)
        filtered_dups_count = len(all_duplicate_rows)

    volumes_list = list(plan["volumes"].values())
    vol_space = svc.get_volume_space_map(volumes_list)
    latest_report_exists = (db_module.REPORTS_DIR / "raport_procedura_offline_ostatni.txt").exists()

    resp = render_template(
        request,
        "partials/backup_panel.html",
        {
            "plan": plan,
            "unbacked_files": all_unbacked[:300],
            "unbacked_total": filtered_unbacked_count,
            "unbacked_all_total": len(all_unbacked),
            "duplicate_files": all_duplicate_rows[:300],
            "duplicates_total": filtered_dups_count,
            "duplicates_all_total": len(all_duplicate_rows),
            "volumes": volumes_list,
            "vol_space": vol_space,
            "execution_result": execution_result,
            "offline_report": offline_report,
            "latest_report_exists": latest_report_exists,
            "selected_operation_mode": norm_mode,
            "selected_pair_keys": effective_pair_keys,
        },
    )
    if trigger_refresh:
        resp.headers["HX-Trigger"] = "refresh-all"
    return resp


@router.get("/backup-panel")
def get_backup_panel(request: Request, target_volume_id: Optional[str] = None, db: Session = Depends(get_db)):
    pref_vid = to_int(target_volume_id)
    return _render_backup_panel_response(request, db, pref_vid=pref_vid)


@router.post("/prepare-offline-report")
def prepare_offline_report_action(
    request: Request,
    preferred_target_volume_id: Optional[str] = Form(None),
    operation_mode: str = Form("both"),
    selected_pairs: List[str] = Form(default=[]),
    pairs_submitted: bool = Form(False),
    target_panel: str = Form("backup"),
    db: Session = Depends(get_db),
):
    """Prepares the offline backup/dedup/move procedure and saves a local text report before physical execution."""
    svc = StorageAutomationService(db)
    pref_vid = to_int(preferred_target_volume_id)
    effective_pairs = selected_pairs if pairs_submitted else (selected_pairs or None)
    report_res = svc.generate_offline_procedure_report(
        preferred_target_volume_id=pref_vid,
        operation_mode=operation_mode,
        selected_pair_keys=effective_pairs,
    )

    if target_panel == "todo":
        return render_template(
            request,
            "partials/todo_panel.html",
            {
                "summary": report_res["pending_summary"],
                "exec_report": None,
                "offline_report": report_res,
            },
        )

    return _render_backup_panel_response(
        request,
        db,
        pref_vid=pref_vid,
        selected_pairs=effective_pairs,
        operation_mode=operation_mode,
        offline_report=report_res,
    )


@router.get("/download-offline-report")
def download_offline_report_file(db: Session = Depends(get_db)):
    """Downloads the latest offline backup/dedup/move procedure text report."""
    latest_file = db_module.REPORTS_DIR / "raport_procedura_offline_ostatni.txt"
    if not latest_file.exists():
        StorageAutomationService(db).generate_offline_procedure_report()
    return FileResponse(
        latest_file,
        media_type="text/plain; charset=utf-8",
        filename="raport_procedura_offline.txt",
    )


@router.post("/execute-backup")
def execute_backup_action(
    request: Request,
    preferred_target_volume_id: Optional[str] = Form(None),
    new_backup_drive_path: Optional[str] = Form(None),
    operation_mode: str = Form("both"),
    selected_pairs: List[str] = Form(default=[]),
    pairs_submitted: bool = Form(False),
    db: Session = Depends(get_db),
):
    pref_vid = to_int(preferred_target_volume_id)
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

    svc = StorageAutomationService(db)
    effective_pairs = selected_pairs if pairs_submitted else (selected_pairs or None)
    job_id = ProcessManager.create_job(
        job_type="backup_dedup",
        title="Automatyczny Backup na inne urządzenie i 100% bezpieczna deduplikacja",
        item_ids=[],
        params={
            "preferred_target_volume_id": pref_vid,
            "operation_mode": operation_mode,
            "selected_pairs": effective_pairs,
        },
        db=db,
    )
    exec_result = svc.execute_backup_and_deduplication(
        preferred_target_volume_id=pref_vid,
        job_id=job_id,
        operation_mode=operation_mode,
        selected_pair_keys=effective_pairs,
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
        operation_mode=operation_mode,
        execution_result=exec_result,
        trigger_refresh=True,
    )


@router.get("/todo-panel")
def get_offline_todo_panel(request: Request, db: Session = Depends(get_db)):
    svc = StorageAutomationService(db)
    summary = svc.get_pending_moves_summary()
    return render_template(
        request,
        "partials/todo_panel.html",
        {
            "summary": summary,
            "exec_report": None,
            "offline_report": None,
        },
    )


@router.post("/todo-execute")
def execute_offline_todo_moves(request: Request, db: Session = Depends(get_db)):
    svc = StorageAutomationService(db)
    report = svc.execute_pending_moves_pairwise()
    summary = svc.get_pending_moves_summary()
    resp = render_template(
        request,
        "partials/todo_panel.html",
        {
            "summary": summary,
            "exec_report": report,
            "offline_report": None,
        },
    )
    resp.headers["HX-Trigger"] = "refresh-all"
    return resp


@router.post("/todo-rollback")
def rollback_offline_todo_moves(
    request: Request,
    todo_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    svc = StorageAutomationService(db)
    tid = to_int(todo_id)
    reverted = svc.rollback_pending_moves(todo_ids=[tid] if tid else None)
    summary = svc.get_pending_moves_summary()
    report = {
        "status": "rolled_back",
        "message": f"Zrezygnowano z przenoszenia. Przywrócono poprzednie lokalizacje w bazie dla {reverted} plików i usunięto zadania z listy TODO.",
    }
    resp = render_template(
        request,
        "partials/todo_panel.html",
        {
            "summary": summary,
            "exec_report": report,
            "offline_report": None,
        },
    )
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
        if res["queued_offline"] > 0:
            msg += f", dodano do listy TODO (offline): {res['queued_offline']} plików"
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>{msg}.</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    except Exception as e:
        return HTMLResponse(f"<span class='text-rose-400 text-xs font-semibold'>Błąd konsolidacji: {e}</span>")


def _filter_db_backups(cloud_files: list) -> list:
    return [
        f for f in cloud_files
        if str(f.get("name", "")).lower().endswith((".db", ".sqlite", ".sqlite3"))
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


@router.get("/gdrive-panel")
def get_gdrive_panel(
    request: Request,
    folder_id: Optional[str] = None,
):
    from app.cloud.google_drive import GoogleDriveManager

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
                "error_msg": str(e),
                "status_report": False,
            },
        )


@router.post("/gdrive-sync")
def sync_to_google_drive(
    request: Request,
    mode: str = Form("media"),
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

    try:
        gdrive = GoogleDriveManager()
        if new_folder_name and new_folder_name.strip():
            target_fid = gdrive.get_or_create_folder(new_folder_name.strip())
        elif folder_id and folder_id.strip():
            target_fid = folder_id.strip()
        else:
            target_fid = gdrive.get_or_create_folder("DomoweArchiwumFoto")

        uploaded_results = []
        if mode == "db_backup":
            mgr = BackupManager()
            b_path = mgr.create_backup("gdrive_cloud")
            res = gdrive.upload_and_verify(b_path, target_fid, purge_local_after_verify=False)
            uploaded_results.append(res)
            summary = f"Kopia zapasowa bazy SQLite ({b_path.name}) została zweryfikowana i zapisana w chmurze."
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
        folders, cloud_files, db_backups = _load_gdrive_fallback(folder_id)
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
                "db_backups": db_backups,
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

    temp_file: Optional[Path] = None
    try:
        gdrive = GoogleDriveManager()
        safe_fid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in file_id)
        temp_file = db_module.BACKUP_DIR / f"cloud_restore_{safe_fid}.db"
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

        resp = render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": active_fid,
                "cloud_files": cloud_files,
                "db_backups": _filter_db_backups(cloud_files),
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
    force_offline_todo: bool = Form(False),
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
    Moves selected or filtered files to a target device and folder.
    Supports both direct physical moves (when source and target drives are online)
    and Offline Mode logical moves queued in the TODO list (Requirement 4b).
    """
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
            res = svc.schedule_or_execute_moves(
                items, target_volume_id=tv_id, target_subfolder=sub_folder, force_offline_todo=force_offline_todo
            )
            msg = f"🚚 Przeniesiono fizycznie: {res['moved_physically']} plików"
            if res["queued_offline"] > 0:
                msg += f" | Zapisano na liście TODO (przeniesienie logiczne offline): {res['queued_offline']} plików (cel: {res['target_volume_label']}/{res['target_subfolder']})"
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

        if Path(drive_root).exists() and not force_offline_todo:
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

            res = svc.schedule_or_execute_moves(
                items, target_volume_id=target_vol.id, target_subfolder=rel_sub, force_offline_todo=False
            )
            msg = f"🚚 Przeniesiono fizycznie {res['moved_physically']} plików do {dest_dir_resolved}"
            if res["queued_offline"] > 0:
                msg += f" (w kolejce TODO dla odłączonych dysków źródłowych: {res['queued_offline']})"
            return HTMLResponse(
                f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
                headers={"HX-Trigger": "refresh-all"},
            )
        else:
            # Target drive letter is offline -> match existing WindowsVolume by drive letter or create placeholder
            target_vol = db.scalar(
                select(WindowsVolume).where(WindowsVolume.last_known_drive_letter.ilike(f"{drive_part}%"))
            )
            if not target_vol:
                return HTMLResponse(
                    f"<span class='text-rose-400 text-xs'>Dysk {drive_root} nie jest podłączony ani zarejestrowany w bazie. Wybierz zarejestrowany dysk z listy.</span>"
                )
            rel_parts = [p for p in dest_dir.parts if not p.endswith(":\\") and not p.endswith(":")]
            rel_sub = "/".join(rel_parts)
            res = svc.schedule_or_execute_moves(
                items, target_volume_id=target_vol.id, target_subfolder=rel_sub, force_offline_todo=True
            )
            return HTMLResponse(
                f"<span class='text-amber-300 text-xs font-semibold'>"
                f"📋 Dysk jest Offline — logicznie przeniesiono i dodano do listy TODO: {res['queued_offline']} plików."
                f"</span>",
                headers={"HX-Trigger": "refresh-all"},
            )
    except Exception as e:
        return HTMLResponse(f"<span class='text-rose-400 text-xs font-semibold'>Błąd przenoszenia: {e}</span>")


def _execute_move_to_trash_items(db: Session, item_ids: List[str], job_id: Optional[int] = None) -> Tuple[int, int]:
    trash_dir = DATA_DIR / "kosz"
    trash_dir.mkdir(parents=True, exist_ok=True)
    items = list(db.scalars(select(MediaItem).where(MediaItem.id.in_(item_ids))).all())
    moved_files = 0
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
            return processed, moved_files

        vol = db.get(WindowsVolume, it.volume_id)
        if vol:
            src_p = Path(vol.last_known_drive_letter) / it.relative_path
            if src_p.exists():
                dest_p = trash_dir / f"{it.id}_{it.file_name}"
                try:
                    shutil.move(str(src_p), str(dest_p))
                    moved_files += 1
                except Exception as exc:
                    logger.debug("Move to trash physical warning: %s", exc)
        it.is_trashed = True
        processed += 1

    if job_id:
        ProcessManager.finish_job(
            job_id,
            status="COMPLETED",
            message=f"Przeniesiono do Kosza: {processed} pozycji (fizycznie: {moved_files}).",
            db=db,
        )
    db.commit()
    return processed, moved_files


@router.post("/move-to-trash")
def move_items_to_local_trash(
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    trash_group_id: Optional[str] = Form(None),
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
    BackupManager().create_backup("pre_trash_move")

    t_gid = to_int(trash_group_id)
    if t_gid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(t_gid)
        items = list(db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == False)).all())
        for gid in subtree_ids:
            g_obj = db.get(MediaGroup, gid)
            if g_obj:
                g_obj.is_trashed = True
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
        )

    if not items and not t_gid:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do przeniesienia do kosza.</span>")

    item_ids = [it.id for it in items]
    job_id = ProcessManager.create_job(
        job_type="move_to_trash",
        title=f"Przenoszenie do Kosza ({len(item_ids)} poz.)",
        item_ids=item_ids,
        db=db,
    )
    processed, moved_files = _execute_move_to_trash_items(db, item_ids, job_id=job_id)

    return HTMLResponse(
        f"<span class='text-amber-300 text-xs font-semibold'>"
        f"🗑️ Przeniesiono do Kosza: {processed} pozycji (fizycznie przeniesiono do data/kosz: {moved_files}). Możesz je przywrócić z widoku Kosza."
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
                db.execute(delete(PendingFileMove).where(PendingFileMove.media_id.in_(batch_ids)))
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
def unindex_items_from_db(
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    unindex_volume_id: Optional[str] = Form(None),
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
    u_vid = to_int(unindex_volume_id)
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
    items = list(db.scalars(select(MediaItem).where(MediaItem.id.in_(item_ids))).all())
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
def delete_files_permanently_from_disk(
    scope: str = Form("selected"),
    selected_ids: List[str] = Form(default=[]),
    delete_group_id: Optional[str] = Form(None),
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
    BackupManager().create_backup("pre_permanent_disk_delete")

    d_gid = to_int(delete_group_id)
    grp_to_delete = None
    if d_gid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(d_gid)
        items = list(db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids))).all())
        grp_to_delete = db.get(MediaGroup, d_gid)
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only,
            untagged_only=untagged_only, unvectored_only=unvectored_only,
            unanalyzed_quality_only=unanalyzed_quality_only,
        )

    if not items and not grp_to_delete:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do trwałego usunięcia.</span>")

    item_ids = [it.id for it in items]
    job_id = ProcessManager.create_job(
        job_type="delete_permanently",
        title=f"Trwałe usuwanie plików z dysku ({len(item_ids)} poz.)",
        item_ids=item_ids,
        db=db,
    )
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
