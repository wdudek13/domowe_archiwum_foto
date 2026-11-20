import datetime
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, or_, and_
from sqlalchemy.orm import Session, joinedload
import numpy as np

from app.database import get_db, THUMBNAILS_DIR
from app.models import MediaEmbedding, MediaFace, MediaItem, MediaTag, MediaGroup, WindowsVolume
from app.ai.embeddings import OnDemandClipEmbedder
from app.ai.faces import OnDemandFaceEmbedder
from app.routers.media import (
    render_template,
    build_flat_album_choices,
    build_search_context,
    build_offline_drives_queues,
    _build_cross_vault_backup_map,
)

logger = logging.getLogger("SemanticRouter")
router = APIRouter(prefix="/semantic", tags=["Semantic Search"])

SIMILARITY_MODES = {
    "hybrid": "🧠 Hybrydowe (Wszystkie cechy)",
    "visual_duplicate": "🎯 Prawie identyczne / Seria (Układ 8x8 LAB)",
    "semantic_objects": "🏷️ Te same obiekty i scena (Tagi AI)",
    "color_mood": "🎨 Paleta barw i klimat (Kolory HSV)",
    "same_event": "📅 Ta sama sesja / Wydarzenie (Czas + Obraz)",
    "composition": "👤 Kompozycja, Kadr i Ludzie",
    "same_person": "👤 Ta sama osoba (Analiza wyciętej twarzy z miniatury)",
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-7 else v


def _compute_vector_in_ram(item: MediaItem) -> np.ndarray:
    """
    Returns the 512-D vector from DB if valid, or computes it in RAM from the local thumbnail
    without writing to SQLite (100% lock-free during GET requests).
    """
    if (
        item.embedding
        and item.embedding.vector
        and len(item.embedding.vector) == 512 * 4
        and item.embedding.model_name == OnDemandClipEmbedder.MODEL_NAME
    ):
        return np.frombuffer(item.embedding.vector, dtype=np.float32)

    from app.scanner import ThumbnailService
    thumb_p = THUMBNAILS_DIR / item.thumbnail_relative_path
    if not ThumbnailService.is_valid_thumbnail(thumb_p):
        if item.volume:
            from app.scanner import WindowsVolumeInspector
            cmap = WindowsVolumeInspector.get_connected_vsn_map()
            vsn = (item.volume.volume_serial_number or "").strip()
            v_root = Path(cmap[vsn][0]) if vsn in cmap else (
                Path(item.volume.last_known_drive_letter)
                if (item.volume.is_online and item.volume.last_known_drive_letter and not cmap)
                else None
            )
            if v_root is not None:
                from app.file_safety import safe_resolve_volume_path
                try:
                    src_p = safe_resolve_volume_path(v_root, item.relative_path)
                except ValueError:
                    src_p = None
                if src_p is not None and src_p.exists():
                    thumb_p = src_p

    if thumb_p.exists() and thumb_p.stat().st_size > 0:
        try:
            blob = OnDemandClipEmbedder.generate_image_embedding(thumb_p)
            return np.frombuffer(blob, dtype=np.float32)
        except Exception as exc:
            logger.debug("In-RAM vector computation skipped: %s", exc)

    return np.zeros(512, dtype=np.float32)


def _get_query_face_vectors(
    db: Session,
    target_item: MediaItem,
    face_index: Optional[int] = None,
    fe: Optional[OnDemandFaceEmbedder] = None,
) -> List[np.ndarray]:
    """
    Returns L2-normalized 256-D face vector(s) for `target_item`.
    When `face_index` is specified (e.g. clicking Face #2 on a group photo),
    strictly filters to that single face instead of matching all faces in the photo.
    """
    dim_bytes = OnDemandFaceEmbedder.VECTOR_DIM * 4
    face_stmt = select(MediaFace).where(MediaFace.media_id == target_item.id)
    if face_index is not None:
        face_stmt = face_stmt.where(MediaFace.face_index == face_index)
    face_stmt = face_stmt.order_by(MediaFace.face_index.asc())
    target_face_rows = list(db.scalars(face_stmt).all())
    q_vecs = [
        np.frombuffer(f.vector, dtype=np.float32)
        for f in target_face_rows
        if f.vector and len(f.vector) == dim_bytes
    ]
    if q_vecs:
        return q_vecs

    # If the photo already has indexed faces in SQLite, do not silently fall back for an invalid face_index
    if face_index is not None:
        any_stored = db.scalar(
            select(MediaFace.id).where(MediaFace.media_id == target_item.id).limit(1)
        )
        if any_stored is not None:
            return []

    if not target_item.thumbnail_relative_path:
        return []
    t_thumb = THUMBNAILS_DIR / target_item.thumbnail_relative_path
    if not t_thumb.exists() or t_thumb.stat().st_size == 0:
        return []

    if fe is not None:
        extracted = fe.extract_faces_from_thumbnail(t_thumb, allow_center_portrait_fallback=False)
    else:
        with OnDemandFaceEmbedder() as local_fe:
            extracted = local_fe.extract_faces_from_thumbnail(t_thumb, allow_center_portrait_fallback=False)

    # Persist newly detected faces for target_item so modal_detail immediately shows Face #1, #2 pills
    if extracted:
        try:
            any_existing = db.scalar(
                select(MediaFace.id).where(MediaFace.media_id == target_item.id).limit(1)
            )
            if any_existing is None:
                for ef in extracted:
                    db.add(
                        MediaFace(
                            media_id=target_item.id,
                            face_index=int(ef["face_index"]),
                            bbox_json=json.dumps(ef["bbox_norm"]),
                            confidence=float(ef["confidence"]),
                            vector=ef["vector_bytes"],
                        )
                    )
                db.commit()
        except Exception as exc:
            logger.debug("On-the-fly face cache save skipped: %s", exc)
            db.rollback()

    if face_index is not None:
        matched = [f["vector"] for f in extracted if int(f.get("face_index", -1)) == face_index]
        if matched:
            return matched
        if 0 <= face_index < len(extracted):
            return [extracted[face_index]["vector"]]
        return []
    return [f["vector"] for f in extracted]


@router.get("/similar/{media_id}")
def find_visually_similar(
    request: Request,
    media_id: str,
    mode: str = Query("hybrid"),
    face_index: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    if mode not in SIMILARITY_MODES:
        mode = "hybrid"

    target_item = db.scalars(
        select(MediaItem)
        .options(
            joinedload(MediaItem.tags),
            joinedload(MediaItem.embedding),
            joinedload(MediaItem.volume),
            joinedload(MediaItem.group),
        )
        .where(MediaItem.id == media_id)
    ).unique().first()

    if not target_item:
        return render_template(
            request,
            "partials/media_grid.html",
            {
                "timeline": {},
                "items_flat": [],
                "total_matching": 0,
                "total_size_mb": 0,
                "page": 1,
                "total_pages": 1,
                "view_mode": "grid",
                "similarity_scores": {},
            },
        )

    # Task 4: Dedicated 100% offline, thumbnail-only Same-Person Face Recognition mode
    if mode == "same_person":
        dim_bytes = OnDemandFaceEmbedder.VECTOR_DIM * 4

        # Collect pre-indexed candidate faces from SQLite (active, non-backup items)
        cand_face_rows = [
            (str(r[0]), r[1])
            for r in db.execute(
                select(MediaFace.media_id, MediaFace.vector)
                .join(MediaItem, MediaItem.id == MediaFace.media_id)
                .where(
                    MediaItem.id != target_item.id,
                    MediaItem.is_trashed == False,
                    MediaItem.is_backup_copy == False,
                    MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
                )
            ).all()
            if r[1] and len(r[1]) == dim_bytes
        ]

        # Check if query face vector(s) are already in SQLite; open ONNX/OpenCV session strictly on demand
        stored_q_stmt = select(MediaFace).where(MediaFace.media_id == target_item.id)
        if face_index is not None:
            stored_q_stmt = stored_q_stmt.where(MediaFace.face_index == face_index)
        stored_q_rows = list(db.scalars(stored_q_stmt.order_by(MediaFace.face_index.asc())).all())
        query_face_vecs = [
            np.frombuffer(f.vector, dtype=np.float32)
            for f in stored_q_rows
            if f.vector and len(f.vector) == dim_bytes
        ]

        need_on_the_fly = (not query_face_vecs) or (not cand_face_rows)
        if need_on_the_fly:
            with OnDemandFaceEmbedder() as fe:
                if not query_face_vecs:
                    query_face_vecs = _get_query_face_vectors(
                        db, target_item, face_index=face_index, fe=fe
                    )

                if query_face_vecs and not cand_face_rows:
                    # Fallback on-the-fly thumbnail scan: prioritize face-tagged items, then recent active items (100% offline!)
                    face_tag_mids = list(
                        db.scalars(
                            select(MediaTag.media_id)
                            .where(MediaTag.tag.in_(["osoba / twarz", "portret", "grupa osob", "ludzie"]))
                            .distinct()
                            .limit(250)
                        ).all()
                    )
                    cand_rows_meta = []
                    if face_tag_mids:
                        cand_rows_meta = list(
                            db.execute(
                                select(MediaItem.id, MediaItem.thumbnail_relative_path).where(
                                    MediaItem.id.in_(face_tag_mids),
                                    MediaItem.id != target_item.id,
                                    MediaItem.is_trashed == False,
                                    MediaItem.is_backup_copy == False,
                                    MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
                                )
                            ).all()
                        )
                    if len(cand_rows_meta) < 20:
                        seen_mids = {str(r[0]) for r in cand_rows_meta} | {target_item.id}
                        extra_meta = list(
                            db.execute(
                                select(MediaItem.id, MediaItem.thumbnail_relative_path)
                                .where(
                                    MediaItem.is_trashed == False,
                                    MediaItem.is_backup_copy == False,
                                    MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
                                )
                                .order_by(MediaItem.effective_date.desc())
                                .limit(150)
                            ).all()
                        )
                        for r in extra_meta:
                            if str(r[0]) not in seen_mids:
                                cand_rows_meta.append(r)
                                seen_mids.add(str(r[0]))

                    for cmid, cthumb_rel in cand_rows_meta:
                        if not cthumb_rel:
                            continue
                        cthumb = THUMBNAILS_DIR / cthumb_rel
                        if not cthumb.exists() or cthumb.stat().st_size == 0:
                            continue
                        for cf in fe.extract_faces_from_thumbnail(cthumb):
                            cand_face_rows.append((str(cmid), cf["vector_bytes"]))

        ranked_faces = OnDemandFaceEmbedder.rank_similar_faces(
            query_face_vecs, cand_face_rows, min_score=0.45, top_n=47
        )
        matched_ids = [mid for mid, _ in ranked_faces]
        cand_by_id = {}
        if matched_ids:
            for item_obj in db.scalars(
                select(MediaItem)
                .options(
                    joinedload(MediaItem.volume),
                    joinedload(MediaItem.group),
                    joinedload(MediaItem.tags),
                )
                .where(MediaItem.id.in_(matched_ids))
            ).unique().all():
                cand_by_id[item_obj.id] = item_obj

        sorted_items = [target_item] + [cand_by_id[mid] for mid in matched_ids if mid in cand_by_id]
        scores_map = {target_item.id: 100.0}
        for mid, sim_val in ranked_faces:
            if mid in cand_by_id:
                scores_map[mid] = round(min(99.9, max(0.0, sim_val) * 100.0), 1)

        total_bytes = sum(it.file_size_bytes for it in sorted_items)
        all_groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False)).all())
        all_volumes = list(db.scalars(select(WindowsVolume)).all())
        sim_vol_counts = defaultdict(int)
        for it in sorted_items:
            sim_vol_counts[it.volume_id] += 1

        face_idx_note = f" (Twarz #{face_index + 1})" if face_index is not None else ""
        face_status_note = "" if query_face_vecs else " (nie wykryto wyraźnej twarzy na miniaturze wzorca)"
        label = f"{SIMILARITY_MODES[mode]} — Wzorzec (100%): {target_item.file_name}{face_idx_note}{face_status_note}"
        return render_template(
            request,
            "partials/media_grid.html",
            {
                "timeline": {label: sorted_items},
                "items_flat": sorted_items,
                "total_matching": len(sorted_items),
                "total_size_mb": total_bytes / (1024 * 1024),
                "page": 1,
                "page_size": 48,
                "total_pages": 1,
                "view_mode": "grid",
                "sort_by": "similarity",
                "similarity_target_id": target_item.id,
                "similarity_target_name": target_item.file_name,
                "similarity_mode": mode,
                "similarity_modes": SIMILARITY_MODES,
                "active_face_index": face_index,
                "face_not_found": not bool(query_face_vecs),
                "matched_person_ids": [it.id for it in sorted_items],
                "all_groups": all_groups,
                "flat_album_choices": build_flat_album_choices(all_groups),
                "all_volumes": all_volumes,
                "offline_drives_info": build_offline_drives_queues(sorted_items, sim_vol_counts, all_volumes),
                "cross_vault_backup_by_item_id": _build_cross_vault_backup_map(db, sorted_items),
                "similarity_scores": scores_map,
            },
        )

    # 1. Compute 512-D pattern vector in RAM without locking SQLite
    q_full = _compute_vector_in_ram(target_item)
    q_lab = _unit(q_full[0:192])
    q_hsv = _unit(q_full[192:448])
    q_tex = _unit(q_full[448:512])

    target_tags = {t.tag: t.confidence for t in target_item.tags}
    target_aspect = (target_item.width / target_item.height) if (target_item.width and target_item.height) else 1.0
    target_parent_dir = str(target_item.relative_path).rsplit("/", 1)[0] if "/" in target_item.relative_path else ""

    # 2. Gather candidate pool from SQLite
    time_low = target_item.effective_date - datetime.timedelta(hours=72)
    time_high = target_item.effective_date + datetime.timedelta(hours=72)

    contextual_ids = set(
        db.scalars(
            select(MediaItem.id)
            .where(
                MediaItem.is_trashed == False,
                or_(
                    MediaItem.id == target_item.id,
                    and_(
                        MediaItem.is_backup_copy == False,
                        MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
                    ),
                ),
                or_(
                    and_(MediaItem.effective_date >= time_low, MediaItem.effective_date <= time_high),
                    MediaItem.relative_path.ilike(f"{target_parent_dir}/%") if target_parent_dir else False,
                ),
            )
            .limit(120)
        ).all()
    )

    if target_tags:
        tag_ids = set(
            db.scalars(
                select(MediaTag.media_id)
                .where(MediaTag.tag.in_(list(target_tags.keys())))
                .limit(200)
            ).all()
        )
        contextual_ids.update(tag_ids)

    all_emb_rows = db.execute(select(MediaEmbedding.media_id, MediaEmbedding.vector)).all()
    if all_emb_rows and np.linalg.norm(q_full) > 0:
        ranked_vis = OnDemandClipEmbedder.rank_similar(q_full, [(r[0], r[1]) for r in all_emb_rows], top_n=200)
        for mid, _ in ranked_vis:
            contextual_ids.add(mid)

    contextual_ids.add(target_item.id)

    candidates = list(
        db.scalars(
            select(MediaItem)
            .options(
                joinedload(MediaItem.volume),
                joinedload(MediaItem.group),
                joinedload(MediaItem.tags),
                joinedload(MediaItem.embedding),
            )
            .where(
                MediaItem.id.in_(list(contextual_ids)),
                MediaItem.is_trashed == False,
                or_(
                    MediaItem.id == target_item.id,
                    and_(
                        MediaItem.is_backup_copy == False,
                        MediaItem.duplicate_role.notin_(["backup_green", "backup_orange"]),
                    ),
                ),
            )
        ).unique().all()
    )

    # 3. Multi-criteria similarity scoring in RAM
    scored_results = [(target_item, 100.0)]

    for cand in candidates:
        if cand.id == target_item.id:
            continue

        c_full = _compute_vector_in_ram(cand)
        if np.linalg.norm(c_full) > 0 and np.linalg.norm(q_full) > 0:
            c_lab = _unit(c_full[0:192])
            c_hsv = _unit(c_full[192:448])
            c_tex = _unit(c_full[448:512])

            sim_lab = max(0.0, float(np.dot(q_lab, c_lab))) ** 1.3
            sim_hsv = max(0.0, float(np.dot(q_hsv, c_hsv))) ** 1.4
            sim_tex = max(0.0, float(np.dot(q_tex, c_tex))) ** 1.2
            sim_vis_combined = (0.50 * sim_lab) + (0.35 * sim_hsv) + (0.15 * sim_tex)
        else:
            sim_lab = sim_hsv = sim_tex = sim_vis_combined = 0.0

        cand_tags = {t.tag: t.confidence for t in cand.tags}
        if target_tags and cand_tags:
            shared = set(target_tags.keys()) & set(cand_tags.keys())
            union = set(target_tags.keys()) | set(cand_tags.keys())
            sim_tags = sum(min(target_tags[k], cand_tags[k]) for k in shared) / sum(
                max(target_tags.get(k, 0.5), cand_tags.get(k, 0.5)) for k in union
            )
        else:
            sim_tags = 0.0

        diff_hours = abs((cand.effective_date - target_item.effective_date).total_seconds()) / 3600.0
        if diff_hours <= 0.5:
            sim_time = 1.0
        elif diff_hours <= 3.0:
            sim_time = 0.85
        elif diff_hours <= 24.0:
            sim_time = 0.65
        elif diff_hours <= 72.0:
            sim_time = 0.35
        else:
            sim_time = 0.0

        cand_aspect = (cand.width / cand.height) if (cand.width and cand.height) else 1.0
        same_orient = (
            1.0
            if (
                (target_aspect >= 1.05 and cand_aspect >= 1.05)
                or (target_aspect <= 0.95 and cand_aspect <= 0.95)
                or (0.95 < target_aspect < 1.05 and 0.95 < cand_aspect < 1.05)
            )
            else 0.0
        )
        face_tags = {"osoba / twarz", "portret", "grupa osob"}
        both_have_faces = 1.0 if ((set(target_tags.keys()) & face_tags) and (set(cand_tags.keys()) & face_tags)) else 0.0
        cand_parent_dir = str(cand.relative_path).rsplit("/", 1)[0] if "/" in cand.relative_path else ""
        same_folder = 1.0 if (cand_parent_dir == target_parent_dir and cand.volume_id == target_item.volume_id) else 0.0
        sim_comp = (0.45 * same_orient) + (0.35 * both_have_faces) + (0.20 * same_folder)

        if mode == "visual_duplicate":
            score = (0.75 * sim_lab) + (0.15 * sim_tex) + (0.10 * same_orient)
            min_threshold = 35.0
        elif mode == "semantic_objects":
            score = (0.75 * sim_tags) + (0.25 * sim_vis_combined)
            min_threshold = 15.0
        elif mode == "color_mood":
            score = (0.80 * sim_hsv) + (0.20 * sim_lab)
            min_threshold = 35.0
        elif mode == "same_event":
            score = (0.65 * sim_time) + (0.25 * sim_vis_combined) + (0.10 * same_folder)
            min_threshold = 20.0
        elif mode == "composition":
            score = (0.45 * sim_lab) + (0.40 * sim_comp) + (0.15 * sim_tex)
            min_threshold = 35.0
        else:
            eff_tags = sim_tags if (target_tags and cand_tags) else sim_vis_combined
            score = (0.50 * sim_vis_combined) + (0.25 * eff_tags) + (0.15 * sim_time) + (0.10 * sim_comp)
            min_threshold = 25.0

        pct = round(min(99.9, score * 100.0), 1)
        if pct >= min_threshold:
            scored_results.append((cand, pct))

    others = sorted(scored_results[1:], key=lambda x: x[1], reverse=True)[:47]
    top_matches = [scored_results[0]] + others

    sorted_items = [x[0] for x in top_matches]
    scores_map = {x[0].id: x[1] for x in top_matches}
    total_bytes = sum(it.file_size_bytes for it in sorted_items)
    all_groups = list(db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False)).all())
    all_volumes = list(db.scalars(select(WindowsVolume)).all())
    sim_vol_counts = defaultdict(int)
    for it in sorted_items:
        sim_vol_counts[it.volume_id] += 1

    label = f"{SIMILARITY_MODES[mode]} — Wzorzec (100%): {target_item.file_name}"
    return render_template(
        request,
        "partials/media_grid.html",
        {
            "timeline": {label: sorted_items},
            "items_flat": sorted_items,
            "total_matching": len(sorted_items),
            "total_size_mb": total_bytes / (1024 * 1024),
            "page": 1,
            "page_size": 48,
            "total_pages": 1,
            "view_mode": "grid",
            "sort_by": "similarity",
            "similarity_target_id": target_item.id,
            "similarity_target_name": target_item.file_name,
            "similarity_mode": mode,
            "similarity_modes": SIMILARITY_MODES,
            "all_groups": all_groups,
            "flat_album_choices": build_flat_album_choices(all_groups),
            "all_volumes": all_volumes,
            "offline_drives_info": build_offline_drives_queues(sorted_items, sim_vol_counts, all_volumes),
            "cross_vault_backup_by_item_id": _build_cross_vault_backup_map(db, sorted_items),
            "similarity_scores": scores_map,
        },
    )


@router.get("/search")
def semantic_search(request: Request, q: str = Query(""), db: Session = Depends(get_db)):
    """Live search endpoint sharing all gallery & backup-visibility filters with /media/search."""
    return render_template(request, "partials/media_grid.html", build_search_context(request, db))
