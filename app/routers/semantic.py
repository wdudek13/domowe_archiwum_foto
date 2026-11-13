import datetime
from collections import defaultdict
from pathlib import Path
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, or_, and_
from sqlalchemy.orm import Session, joinedload
import numpy as np

from app.database import get_db, THUMBNAILS_DIR
from app.models import MediaEmbedding, MediaItem, MediaTag, MediaGroup, WindowsVolume
from app.ai.embeddings import OnDemandClipEmbedder
from app.ai.tagger import PL_CATEGORY_MAPPING
from app.routers.media import render_template

router = APIRouter(prefix="/semantic", tags=["Semantic Search"])

SIMILARITY_MODES = {
    "hybrid": "🧠 Hybrydowe (Wszystkie cechy)",
    "visual_duplicate": "🎯 Prawie identyczne / Seria (Układ 8x8 LAB)",
    "semantic_objects": "🏷️ Te same obiekty i scena (Tagi AI)",
    "color_mood": "🎨 Paleta barw i klimat (Kolory HSV)",
    "same_event": "📅 Ta sama sesja / Wydarzenie (Czas + Obraz)",
    "composition": "👤 Kompozycja, Kadr i Ludzie",
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

    thumb_p = THUMBNAILS_DIR / item.thumbnail_relative_path
    if not thumb_p.exists() or thumb_p.stat().st_size < 100:
        if item.volume:
            src_p = Path(item.volume.last_known_drive_letter) / item.relative_path
            if src_p.exists():
                thumb_p = src_p

    if thumb_p.exists() and thumb_p.stat().st_size > 0:
        try:
            blob = OnDemandClipEmbedder.generate_image_embedding(thumb_p)
            return np.frombuffer(blob, dtype=np.float32)
        except Exception:
            pass

    return np.zeros(512, dtype=np.float32)


@router.get("/similar/{media_id}")
def find_visually_similar(
    request: Request,
    media_id: str,
    mode: str = Query("hybrid"),
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
        return render_template(request, "partials/media_grid.html", {"timeline": {}, "items_flat": [], "total_matching": 0, "total_size_mb": 0, "page": 1, "total_pages": 1, "view_mode": "grid", "similarity_scores": {}})

    # 1. Wylicz wektor 512-D wzorca w pamięci RAM (zero zapisów do bazy = zero blokad!)
    q_full = _compute_vector_in_ram(target_item)
    q_lab = _unit(q_full[0:192])
    q_hsv = _unit(q_full[192:448])
    q_tex = _unit(q_full[448:512])

    target_tags = {t.tag: t.confidence for t in target_item.tags}
    target_aspect = (target_item.width / target_item.height) if (target_item.width and target_item.height) else 1.0
    target_parent_dir = str(target_item.relative_path).rsplit("/", 1)[0] if "/" in target_item.relative_path else ""

    # 2. Zbierz pulę kandydatów z całej bazy:
    time_low = target_item.effective_date - datetime.timedelta(hours=72)
    time_high = target_item.effective_date + datetime.timedelta(hours=72)

    contextual_ids = set(
        db.scalars(
            select(MediaItem.id)
            .where(
                MediaItem.is_trashed == False,
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
            .where(MediaItem.id.in_(list(contextual_ids)), MediaItem.is_trashed == False)
        ).unique().all()
    )

    # 3. Wielokryterialna ocena podobieństwa w RAM
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
            "all_groups": db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False)).all(),
            "all_volumes": db.scalars(select(WindowsVolume)).all(),
            "similarity_scores": scores_map,
        },
    )


@router.get("/search")
def semantic_search(request: Request, q: str = Query(...), db: Session = Depends(get_db)):
    clean_q = q.strip().lower()
    if not clean_q:
        return render_template(request, "partials/media_grid.html", {"timeline": {}, "items_flat": [], "total_matching": 0, "total_size_mb": 0, "page": 1, "total_pages": 1, "view_mode": "grid", "similarity_scores": {}})

    search_terms = [clean_q]
    for pl_cat, en_words in PL_CATEGORY_MAPPING.items():
        if clean_q in pl_cat or any(clean_q in w or w in clean_q for w in en_words):
            search_terms.append(pl_cat)
            search_terms.extend(en_words)

    conds = [MediaItem.file_name.ilike(f"%{clean_q}%"), MediaItem.relative_path.ilike(f"%{clean_q}%")]
    for term in set(search_terms):
        conds.append(MediaTag.tag.ilike(f"%{term}%"))

    matched_items = db.scalars(
        select(MediaItem)
        .options(joinedload(MediaItem.volume), joinedload(MediaItem.group), joinedload(MediaItem.tags), joinedload(MediaItem.embedding))
        .outerjoin(MediaItem.tags)
        .where(MediaItem.is_trashed == False, or_(*conds))
        .distinct()
        .limit(150)
    ).unique().all()

    grouped = defaultdict(list)
    total_bytes = 0
    for it in matched_items:
        grouped[it.effective_date.strftime("%Y-%m-%d")].append(it)
        total_bytes += it.file_size_bytes

    return render_template(
        request,
        "partials/media_grid.html",
        {
            "timeline": dict(grouped),
            "items_flat": matched_items,
            "total_matching": len(matched_items),
            "total_size_mb": total_bytes / (1024 * 1024),
            "page": 1,
            "page_size": 150,
            "total_pages": 1,
            "view_mode": "grid",
            "sort_by": "semantic",
            "all_groups": db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False)).all(),
            "all_volumes": db.scalars(select(WindowsVolume)).all(),
            "similarity_scores": {},
        },
    )