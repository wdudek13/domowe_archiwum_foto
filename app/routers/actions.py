from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import traceback
from typing import List, Optional
from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.backup import BackupManager
from app.database import get_db, SessionLocal, THUMBNAILS_DIR, DATA_DIR
from app.models import WindowsVolume, MediaItem, MediaGroup, MediaTag, MediaEmbedding
from app.scanner import WindowsVolumeInspector, FileScannerService, ThumbnailService, IMAGE_EXTS, VIDEO_EXTS
from app.storage_ops import StorageAutomationService
from app.ai.quality import MediaQualityAnalyzer
from app.ai.tagger import OnDemandObjectTagger
from app.ai.embeddings import OnDemandClipEmbedder
from app.routers.media import apply_media_filters, to_int

logger = logging.getLogger("ActionsRouter")
router = APIRouter(prefix="/actions", tags=["Actions"])


def get_best_readable_path(item: MediaItem, vol: Optional[WindowsVolume]) -> Optional[Path]:
    """Returns physical file path on USB if online, otherwise falls back to cached WebP thumbnail."""
    if vol:
        full_p = Path(vol.last_known_drive_letter) / item.relative_path
        if full_p.exists() and full_p.stat().st_size > 0:
            return full_p
    thumb_p = THUMBNAILS_DIR / item.thumbnail_relative_path
    if thumb_p.exists() and thumb_p.stat().st_size > 0:
        return thumb_p
    return None


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
    limit: Optional[int] = None,
) -> List[MediaItem]:
    if scope == "selected" and selected_ids:
        stmt = select(MediaItem).where(MediaItem.id.in_(selected_ids))
    else:
        stmt = apply_media_filters(
            select(MediaItem), db, q, media_type, to_int(volume_id), folder_prefix,
            to_int(group_id), tag, to_int(year),
            anomaly_only, blurry_only, duplicates_only, untagged_only, ungrouped_only
        ).distinct()
    if limit:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).unique().all())


def execute_scan(path_str: str, force_rescan: bool = False) -> dict:
    clean = path_str.strip(' \t\n\r"\'')
    if len(clean) == 2 and clean[1] == ":":
        clean += "\\"
    target = Path(clean).resolve()

    if not target.exists():
        return {"status": "error", "message": f"Ścieżka nie istnieje: {clean}"}

    try:
        vsn, label, root = WindowsVolumeInspector.get_volume_info(target)
        root_path = Path(root).resolve()

        with SessionLocal() as db:
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
            else:
                vol.last_known_drive_letter = root
                vol.is_online = True
                db.commit()

            existing_rows = db.execute(
                select(MediaItem.relative_path, MediaItem.id, MediaItem.thumbnail_relative_path)
                .where(MediaItem.volume_id == vol.id)
            ).all()
            existing_map = {r[0]: (r[1], r[2]) for r in existing_rows}

            candidates_to_add = []
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
                    try:
                        rel = full.resolve().relative_to(root_path).as_posix()
                    except ValueError:
                        rel = full.name

                    if rel in existing_map and not force_rescan:
                        already_in_db += 1
                        item_id, thumb_name = existing_map[rel]
                        thumb_file = THUMBNAILS_DIR / thumb_name
                        if not thumb_file.exists() or thumb_file.stat().st_size == 0:
                            try:
                                ThumbnailService.generate(full, item_id, ext in IMAGE_EXTS)
                                repaired_thumbs += 1
                            except Exception:
                                pass
                    else:
                        if rel in existing_map and force_rescan:
                            db.execute(delete(MediaItem).where(MediaItem.id == existing_map[rel][0]))
                        candidates_to_add.append((full, rel))

            def _worker(args):
                f_path, r_path = args
                return FileScannerService.process_file(f_path, vol, rel_override=r_path)

            indexed = 0
            if candidates_to_add:
                with ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 2))) as pool:
                    for item in pool.map(_worker, candidates_to_add):
                        if item:
                            db.add(item)
                            indexed += 1
                            if indexed % 50 == 0:
                                db.commit()
                db.commit()

            svc = StorageAutomationService(db)
            dup_total = svc.flag_duplicate_clusters()

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


import threading

AI_BATCH_STATE = {
    "running": False,
    "stop_requested": False,
    "processed": 0,
    "total": 0,
    "tagged": 0,
    "vectors": 0,
    "blurry": 0,
}


def _run_background_ai_batch(item_ids: List[str]):
    AI_BATCH_STATE["running"] = True
    AI_BATCH_STATE["stop_requested"] = False
    AI_BATCH_STATE["processed"] = 0
    AI_BATCH_STATE["total"] = len(item_ids)
    AI_BATCH_STATE["tagged"] = 0
    AI_BATCH_STATE["vectors"] = 0
    AI_BATCH_STATE["blurry"] = 0

    try:
        with OnDemandObjectTagger() as tagger:
            chunk_size = 50
            for start in range(0, len(item_ids), chunk_size):
                if AI_BATCH_STATE["stop_requested"]:
                    break
                batch_ids = item_ids[start : start + chunk_size]

                # KROK A: Szybki odczyt ścieżek z bazy (transakcja zamknięta natychmiast)
                with SessionLocal() as db:
                    rows = db.execute(
                        select(
                            MediaItem.id,
                            MediaItem.media_type,
                            MediaItem.thumbnail_relative_path,
                            MediaItem.relative_path,
                            WindowsVolume.last_known_drive_letter,
                        )
                        .join(WindowsVolume, WindowsVolume.id == MediaItem.volume_id)
                        .where(MediaItem.id.in_(batch_ids))
                    ).all()

                # KROK B: Ciężkie obliczenia CPU (ONNX + OpenCV) w RAM BEZ blokowania bazy SQLite!
                computed_batch = []
                for mid, mtype, thumb_rel, rel_p, drive_let in rows:
                    if AI_BATCH_STATE["stop_requested"]:
                        break
                    thumb_p = THUMBNAILS_DIR / thumb_rel
                    src_p = Path(drive_let) / rel_p
                    readable_path = thumb_p if (thumb_p.exists() and thumb_p.stat().st_size > 100) else (src_p if src_p.exists() else None)

                    if not readable_path:
                        AI_BATCH_STATE["processed"] += 1
                        continue

                    tags = tagger.tag_image(readable_path)
                    if mtype == "video":
                        tags.append(("wideo", 1.0))

                    vec_blob = None
                    try:
                        vec_blob = OnDemandClipEmbedder.generate_image_embedding(readable_path)
                    except Exception:
                        pass

                    corrupted, blurry, score = MediaQualityAnalyzer.analyze_image(readable_path)
                    computed_batch.append((mid, tags, vec_blob, corrupted, blurry, score))
                    AI_BATCH_STATE["processed"] += 1

                # KROK C: Błyskawiczny zapis wyników do SQLite (<10 ms blokady)
                if computed_batch:
                    with SessionLocal() as db:
                        for mid, tags, vec_blob, corrupted, blurry, score in computed_batch:
                            db.execute(delete(MediaTag).where(MediaTag.media_id == mid))
                            for t_name, conf in tags:
                                db.add(MediaTag(media_id=mid, tag=t_name, confidence=conf, source_model="opencv+onnx"))
                            if tags:
                                AI_BATCH_STATE["tagged"] += 1

                            if vec_blob is not None:
                                db.merge(
                                    MediaEmbedding(
                                        media_id=mid,
                                        model_name=OnDemandClipEmbedder.MODEL_NAME,
                                        dimensions=512,
                                        vector=vec_blob,
                                    )
                                )
                                AI_BATCH_STATE["vectors"] += 1

                            it_obj = db.get(MediaItem, mid)
                            if it_obj:
                                it_obj.is_corrupted = corrupted
                                it_obj.is_blurry = blurry
                                it_obj.blur_score = score
                                if blurry:
                                    AI_BATCH_STATE["blurry"] += 1
                        db.commit()
    finally:
        AI_BATCH_STATE["running"] = False

@router.get("/ai-status")
def get_ai_batch_status():
    st = AI_BATCH_STATE
    if st["running"]:
        pct = int((st["processed"] / st["total"]) * 100) if st["total"] else 0
        return HTMLResponse(
            f"<div hx-get='/actions/ai-status' hx-trigger='every 1.5s' hx-swap='outerHTML' class='flex items-center space-x-3 text-xs'>"
            f"<span class='text-purple-300 font-semibold animate-pulse'>⚡ Przetwarzanie AI w tle: {st['processed']} / {st['total']} ({pct}%)</span>"
            f"<span class='text-slate-400 font-mono'>[Tagi: {st['tagged']} | Wektory: {st['vectors']} | Rozmyte: {st['blurry']}]</span>"
            f"<button hx-post='/actions/ai-stop' hx-target='#global-notification' class='bg-rose-800 hover:bg-rose-700 text-white px-2 py-0.5 rounded text-[10px]'>⏹️ Zatrzymaj</button>"
            f"</div>"
        )
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>"
        f"✓ Masowe AI zakończone! Przetworzono: {st['processed']} / {st['total']} | Otagowano: {st['tagged']} | Nowe wektory 512-D: {st['vectors']}"
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/ai-stop")
def stop_ai_batch():
    AI_BATCH_STATE["stop_requested"] = True
    return HTMLResponse("<span class='text-amber-400 text-xs font-semibold'>⏹️ Zatrzymywanie procesu AI po bieżącej paczce...</span>")


@router.post("/analyze-all")
def trigger_full_ai_analysis(
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
    db: Session = Depends(get_db),
):
    if AI_BATCH_STATE["running"]:
        return get_ai_batch_status()

    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only, limit=None
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików w wybranym zakresie.</span>")

    item_ids = [it.id for it in items]
    t = threading.Thread(target=_run_background_ai_batch, args=(item_ids,), daemon=True)
    t.start()
    return get_ai_batch_status()

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
    db: Session = Depends(get_db),
):
    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików do przypisania.</span>")

    ex_id = to_int(existing_group_id)
    par_id = to_int(parent_id)

    if ex_id:
        grp = db.get(MediaGroup, ex_id)
    elif group_name and group_name.strip():
        clean_name = group_name.strip()
        grp = db.scalar(select(MediaGroup).where(MediaGroup.name == clean_name, MediaGroup.parent_id == par_id))
        if not grp:
            grp = MediaGroup(name=clean_name, parent_id=par_id)
            db.add(grp)
            db.commit()
            db.refresh(grp)
    else:
        return HTMLResponse("<span class='text-rose-400 text-xs'>Podaj nazwę grupy.</span>")

    for it in items:
        it.group_id = grp.id
    db.commit()

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>Przypisano {len(items)} plików do grupy '{grp.name}'.</span>",
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
    db: Session = Depends(get_db),
):
    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików w zbiorze.</span>")

    svc = StorageAutomationService(db)
    if mode == "date":
        cnt = svc.auto_group_items_by_date(items)
        msg = f"Zorganizowano {cnt} plików w drzewo grup Rok -> Miesiąc."
    else:
        cnt = svc.auto_group_items_by_folder(items)
        msg = f"Zorganizowano {cnt} plików w drzewo na podstawie katalogów z dysku."

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
    db: Session = Depends(get_db),
):
    grp = db.get(MediaGroup, target_group_id)
    if not grp:
        return HTMLResponse("<span class='text-rose-400 text-xs'>Grupa nie istnieje.</span>")

    par_id = to_int(new_parent_id)
    if action == "create_sub" and new_name:
        sub = MediaGroup(name=new_name.strip(), parent_id=grp.id)
        db.add(sub)
        db.commit()
        msg = f"Utworzono podgrupę '{sub.name}' wewnątrz '{grp.name}'."
    elif action == "rename" and new_name:
        grp.name = new_name.strip()
        db.commit()
        msg = f"Zmieniono nazwę grupy na '{grp.name}'."
    elif action == "reparent":
        if par_id == grp.id:
            return HTMLResponse("<span class='text-rose-400 text-xs'>Grupa nie może być swoim własnym rodzicem.</span>")
        grp.parent_id = par_id if par_id and par_id > 0 else None
        db.commit()
        msg = f"Przeniesiono grupę '{grp.name}' w drzewie."
    elif action == "delete":
        name = grp.name
        for it in db.scalars(select(MediaItem).where(MediaItem.group_id == grp.id)).all():
            it.group_id = None
        db.delete(grp)
        db.commit()
        msg = f"Usunięto grupę '{name}'."
    else:
        msg = "Nieznana operacja."

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
        cnt = svc.consolidate_group(group_id, target_volume_id)
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>Skonsolidowano i przeniesiono fizycznie {cnt} plików na dysk docelowy.</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    except Exception as e:
        return HTMLResponse(f"<span class='text-rose-400 text-xs font-semibold'>Błąd konsolidacji: {e}</span>")


@router.get("/gdrive-panel")
def get_gdrive_panel(
    request: Request,
    folder_id: Optional[str] = None,
):
    from app.routers.media import render_template
    from app.cloud.google_drive import GoogleDriveManager

    try:
        gdrive = GoogleDriveManager()
        folders = gdrive.list_folders()
        if not folders:
            default_id = gdrive.get_or_create_folder("HydraArchive")
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
    db: Session = Depends(get_db),
):
    import datetime
    from app.routers.media import render_template
    from app.cloud.google_drive import GoogleDriveManager

    try:
        gdrive = GoogleDriveManager()
        if new_folder_name and new_folder_name.strip():
            target_fid = gdrive.get_or_create_folder(new_folder_name.strip())
        elif folder_id and folder_id.strip():
            target_fid = folder_id.strip()
        else:
            target_fid = gdrive.get_or_create_folder("HydraArchive")

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
                group_id, tag, year, anomaly_only, blurry_only, duplicates_only, limit=100
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

        folders = gdrive.list_folders()
        cloud_files = gdrive.list_files_in_folder(target_fid)

        resp = render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": target_fid,
                "cloud_files": cloud_files,
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
        folders = []
        cloud_files = []
        try:
            gdrive = GoogleDriveManager()
            folders = gdrive.list_folders()
            if folder_id:
                cloud_files = gdrive.list_files_in_folder(folder_id)
        except Exception:
            pass
        return render_template(
            request,
            "partials/gdrive_panel.html",
            {
                "folders": folders,
                "active_folder_id": folder_id or "",
                "cloud_files": cloud_files,
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
    db: Session = Depends(get_db),
):
    import datetime
    import json

    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików do eksportu.</span>")

    export_dir = DATA_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = export_dir / f"hydra_export_{ts}.json"

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
                "group": grp.name if grp else None,
                "effective_date": it.effective_date.isoformat(),
                "exif_date": it.exif_date.isoformat() if it.exif_date else None,
                "file_size_bytes": it.file_size_bytes,
                "dimensions": {"width": it.width, "height": it.height},
                "file_hash": it.file_hash,
                "is_duplicate": not it.is_primary_duplicate,
                "is_blurry": it.is_blurry,
                "blur_score": it.blur_score,
            }
        )

    dest_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>"
        f"✓ Zapisano {len(payload)} rekordów na dysku: <code class='bg-slate-900 px-1.5 py-0.5 rounded text-white'>{dest_file}</code>"
        f"</span>"
    )


@router.post("/move-files")
def move_physical_files_action(
    destination_path: str = Form(...),
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
    db: Session = Depends(get_db),
):
    """
    Physically moves selected or filtered files to any destination folder on disk/USB,
    updating their WindowsVolume and relative_path in SQLite transactionally.
    """
    import shutil

    clean_dest = destination_path.strip(' \t\n\r"\'')
    if len(clean_dest) == 2 and clean_dest[1] == ":":
        clean_dest += "\\"
    dest_dir = Path(clean_dest).resolve()

    items = resolve_target_items(
        db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
        group_id, tag, year, anomaly_only, blurry_only, duplicates_only
    )
    if not items:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak plików do przeniesienia.</span>")

    try:
        BackupManager().create_backup("pre_physical_move")
        dest_dir.mkdir(parents=True, exist_ok=True)

        vsn, label, root = WindowsVolumeInspector.get_volume_info(dest_dir)
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

        moved_cnt = 0
        for it in items:
            src_vol = db.get(WindowsVolume, it.volume_id)
            if not src_vol:
                continue
            src_file = Path(src_vol.last_known_drive_letter) / it.relative_path
            if not src_file.exists():
                continue

            dest_file = dest_dir / it.file_name
            if src_file.resolve() == dest_file.resolve():
                continue

            c = 1
            while dest_file.exists():
                dest_file = dest_dir / f"{src_file.stem}_{c}{src_file.suffix}"
                c += 1

            shutil.move(str(src_file), str(dest_file))
            it.file_name = dest_file.name
            it.volume_id = target_vol.id
            try:
                it.relative_path = dest_file.resolve().relative_to(root_path).as_posix()
            except ValueError:
                it.relative_path = dest_file.name
            moved_cnt += 1

        db.commit()
        return HTMLResponse(
            f"<span class='text-emerald-400 text-xs font-semibold'>"
            f"🚚 Przeniesiono fizycznie {moved_cnt} plików do: <code class='bg-slate-900 px-1.5 py-0.5 rounded text-white'>{dest_dir}</code>"
            f"</span>",
            headers={"HX-Trigger": "refresh-all"},
        )
    except Exception as e:
        return HTMLResponse(f"<span class='text-rose-400 text-xs font-semibold'>Błąd przenoszenia: {e}</span>")


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
    db: Session = Depends(get_db),
):
    """
    Reversible Trash: moves physical file to data/kosz/{item.id}_{item.file_name},
    retaining volume_id, relative_path, and group_id in SQLite so it can be restored to its exact origin!
    """
    import shutil

    BackupManager().create_backup("pre_trash_move")
    trash_dir = DATA_DIR / "kosz"
    trash_dir.mkdir(parents=True, exist_ok=True)

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
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only
        )

    if not items and not t_gid:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do przeniesienia do kosza.</span>")

    moved_files = 0
    for it in items:
        vol = db.get(WindowsVolume, it.volume_id)
        if vol:
            src_p = Path(vol.last_known_drive_letter) / it.relative_path
            if src_p.exists():
                dest_p = trash_dir / f"{it.id}_{it.file_name}"
                try:
                    shutil.move(str(src_p), str(dest_p))
                    moved_files += 1
                except Exception:
                    pass
        it.is_trashed = True

    db.commit()
    return HTMLResponse(
        f"<span class='text-amber-300 text-xs font-semibold'>"
        f"🗑️ Przeniesiono do Kosza: {len(items)} pozycji (fizycznie przeniesiono do data/kosz: {moved_files}). Możesz je przywrócić z widoku Kosza."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/restore-from-trash")
def restore_from_trash(
    restore_mode: str = Form("selected"),  # 'selected', 'all', or 'group'
    restore_group_id: Optional[str] = Form(None),
    selected_ids: List[str] = Form(default=[]),
    db: Session = Depends(get_db),
):
    """
    Restores trashed files and/or groups back to their original directory on the original USB volume.
    Recreates the original subfolder if needed.
    """
    import shutil

    trash_dir = DATA_DIR / "kosz"
    r_gid = to_int(restore_group_id)

    if restore_mode == "group" and r_gid:
        svc = StorageAutomationService(db)
        subtree_ids = svc.get_group_subtree_ids(r_gid)
        for gid in subtree_ids:
            g_obj = db.get(MediaGroup, gid)
            if g_obj:
                g_obj.is_trashed = False
                # Also restore parent groups if they were trashed
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
    offline_skipped = 0

    for it in items:
        vol = db.get(WindowsVolume, it.volume_id)
        trash_file = trash_dir / f"{it.id}_{it.file_name}"

        if vol and Path(vol.last_known_drive_letter).exists():
            orig_full_path = Path(vol.last_known_drive_letter) / it.relative_path
            orig_parent_dir = orig_full_path.parent
            orig_parent_dir.mkdir(parents=True, exist_ok=True)

            if trash_file.exists():
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
            if it.group_id:
                g_obj = db.get(MediaGroup, it.group_id)
                if g_obj:
                    g_obj.is_trashed = False
        else:
            offline_skipped += 1

    db.commit()
    msg = f"♻️ Przywrócono z Kosza na pierwotne miejsce: {restored_files} plików."
    if offline_skipped > 0:
        msg += f" (Pominięto {offline_skipped} plików – ich dysk źródłowy jest obecnie odłączony)."

    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>{msg}</span>",
        headers={"HX-Trigger": "refresh-all"},
    )


@router.post("/empty-trash")
def empty_trash_permanently(
    selected_ids: List[str] = Form(default=[]),
    db: Session = Depends(get_db),
):
    """Permanently deletes files from data/kosz and removes their records from SQLite."""
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
            except OSError:
                pass
        thumb_p = THUMBNAILS_DIR / it.thumbnail_relative_path
        if thumb_p.exists():
            try:
                thumb_p.unlink()
            except OSError:
                pass
        db.delete(it)
        deleted_cnt += 1

    if not selected_ids:
        for tg in db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == True)).all():
            db.delete(tg)

    db.commit()
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

    vol.volume_label = new_label.strip()[:64]
    db.commit()
    return HTMLResponse(
        f"<span class='text-emerald-400 text-xs font-semibold'>✓ Zmieniono nazwę dysku na: {vol.volume_label}</span>",
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
    db: Session = Depends(get_db),
):
    """
    Removes media records and cached thumbnails from SQLite without touching
    the original physical files on disk/USB.
    """
    u_vid = to_int(unindex_volume_id)
    if u_vid:
        items = list(db.scalars(select(MediaItem).where(MediaItem.volume_id == u_vid)).all())
    else:
        items = resolve_target_items(
            db, scope, selected_ids, q, media_type, volume_id, folder_prefix,
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only
        )

    if not items and not u_vid:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do wyindeksowania z bazy.</span>")

    removed_cnt = 0
    for it in items:
        thumb_p = THUMBNAILS_DIR / it.thumbnail_relative_path
        if thumb_p.exists():
            try:
                thumb_p.unlink()
            except OSError:
                pass
        db.delete(it)
        removed_cnt += 1

    if u_vid:
        vol = db.get(WindowsVolume, u_vid)
        if vol:
            db.delete(vol)

    db.commit()
    return HTMLResponse(
        f"<span class='text-sky-300 text-xs font-semibold'>"
        f"✖️ Usunięto z indeksu bazy: {removed_cnt} pozycji (oryginalne pliki na dysku pozostały nienaruszone)."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )

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
    db: Session = Depends(get_db),
):
    """
    Permanently deletes the physical files from the source USB/local disk,
    removes their thumbnails, and deletes their records from SQLite.
    """
    BackupManager().create_backup("pre_permanent_disk_delete")
    trash_dir = DATA_DIR / "kosz"

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
            group_id, tag, year, anomaly_only, blurry_only, duplicates_only
        )

    if not items and not grp_to_delete:
        return HTMLResponse("<span class='text-amber-400 text-xs'>Brak wybranych elementów do trwałego usunięcia.</span>")

    deleted_disk = 0
    removed_db = 0
    for it in items:
        vol = db.get(WindowsVolume, it.volume_id)
        if vol:
            src_p = Path(vol.last_known_drive_letter) / it.relative_path
            if src_p.exists():
                try:
                    src_p.unlink()
                    deleted_disk += 1
                except OSError:
                    pass

        trash_p = trash_dir / f"{it.id}_{it.file_name}"
        if trash_p.exists():
            try:
                trash_p.unlink()
                deleted_disk += 1
            except OSError:
                pass

        thumb_p = THUMBNAILS_DIR / it.thumbnail_relative_path
        if thumb_p.exists():
            try:
                thumb_p.unlink()
            except OSError:
                pass

        db.delete(it)
        removed_db += 1

    if grp_to_delete:
        db.delete(grp_to_delete)

    db.commit()
    return HTMLResponse(
        f"<span class='text-rose-400 text-xs font-semibold'>"
        f"🔥 Trwale usunięto fizycznie z dysku: {deleted_disk} plików (usunięto z bazy: {removed_db})."
        f"</span>",
        headers={"HX-Trigger": "refresh-all"},
    )