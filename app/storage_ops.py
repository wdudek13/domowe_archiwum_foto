from collections import defaultdict
import datetime
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import select, text, update, delete, func
from sqlalchemy.orm import Session, joinedload

from app.backup import BackupManager
from app.models import MediaItem, WindowsVolume, MediaGroup, PendingFileMove
from app.operation_log import OperationLogger
from app.scanner import FastHasher, WindowsVolumeInspector

logger = logging.getLogger("StorageOps")


class StorageAutomationService:
    """Handles duplicate clustering, 100%-verified backup & deduplication, album management, and offline move queues."""

    MIN_FREE_SPACE_BUFFER_BYTES = 20 * 1024 * 1024  # 20 MB safety reserve on target drives

    def __init__(self, db: Session):
        self.db = db
        self.backup_mgr = BackupManager()

    def refresh_volumes_online_status(self) -> List[WindowsVolume]:
        """Updates online status and drive letters for all tracked volumes using hardware VSN detection."""
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        volumes = list(self.db.scalars(select(WindowsVolume)).all())
        now = datetime.datetime.utcnow()
        for vol in volumes:
            if vol.volume_serial_number in connected_map:
                root_letter, _ = connected_map[vol.volume_serial_number]
                vol.last_known_drive_letter = root_letter
                vol.is_online = True
                vol.last_seen_at = now
            else:
                vol.is_online = Path(vol.last_known_drive_letter).exists()
                if vol.is_online:
                    vol.last_seen_at = now
        self.db.commit()
        return volumes

    def flag_duplicate_clusters(self) -> int:
        """
        Marks primary vs duplicate records across all items in SQLite.
        Rule 1d & 2d:
        - Copies in 'backup/' folder are marked is_backup_copy = 1.
        - Primary original (is_primary_duplicate = 1) prefers non-backup copies on a volume that already
          has multiple non-backup copies (so the original on that volume is eligible for the 'Ważne' album),
          breaking ties by oldest effective_date and id.
        """
        mark_backup_sql = text("""
            UPDATE media_items
            SET is_backup_copy = CASE
                WHEN LOWER(relative_path) LIKE 'backup/%' OR LOWER(relative_path) LIKE 'backup\\%' THEN 1
                ELSE 0
            END;
        """)
        self.db.execute(mark_backup_sql)

        rank_sql = text("""
            WITH vol_counts AS (
                SELECT
                    id,
                    file_hash,
                    is_backup_copy,
                    effective_date,
                    relative_path,
                    SUM(CASE WHEN is_backup_copy = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY file_hash, volume_id) AS same_vol_cnt
                FROM media_items
                WHERE is_trashed = 0
            ),
            ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY file_hash
                        ORDER BY is_backup_copy ASC, same_vol_cnt DESC, effective_date ASC, relative_path ASC, id ASC
                    ) AS rn
                FROM vol_counts
            )
            UPDATE media_items
            SET is_primary_duplicate = CASE
                WHEN id IN (SELECT id FROM ranked WHERE rn = 1) THEN 1
                ELSE 0
            END
            WHERE is_trashed = 0;
        """)
        self.db.execute(rank_sql)
        self.db.commit()
        dup_count = (
            self.db.scalar(
                select(func.count(MediaItem.id)).where(
                    MediaItem.is_trashed == False,
                    MediaItem.is_primary_duplicate == False,
                )
            )
            or 0
        )
        return int(dup_count)

    def get_group_subtree_ids(self, root_group_id: int) -> List[int]:
        all_groups = self.db.execute(select(MediaGroup.id, MediaGroup.parent_id)).all()
        children_map: Dict[Optional[int], List[int]] = {}
        for gid, pid in all_groups:
            children_map.setdefault(pid, []).append(gid)

        result: List[int] = []
        stack = [root_group_id]
        visited: Set[int] = set()
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                result.append(curr)
                stack.extend(children_map.get(curr, []))
        return result

    def delete_album_fast(self, album_id: int) -> str:
        """
        Quickly deletes an album (and its subalbums) without deleting photos;
        all photos are immediately returned to the free pool (group_id = NULL).
        """
        grp = self.db.get(MediaGroup, album_id)
        if not grp:
            raise ValueError("Album nie istnieje.")
        album_name = grp.name
        subtree_ids = self.get_group_subtree_ids(album_id)
        self.db.execute(
            update(MediaItem).where(MediaItem.group_id.in_(subtree_ids)).values(group_id=None)
        )
        for gid in reversed(subtree_ids):
            self.db.execute(delete(MediaGroup).where(MediaGroup.id == gid))
        self.db.commit()
        OperationLogger.log(
            "ALBUM_DELETE",
            f"Usunięto album '{album_name}' (ID: {album_id}); zdjęcia trafiły do swobodnej puli.",
            status="OK",
            db=self.db,
        )
        return album_name

    def transfer_album_and_delete_source(self, source_album_id: int, target_album_id: int) -> Tuple[int, str, str]:
        """
        Moves all photos from source_album_id (and its subalbums) into target_album_id
        and deletes the source album.
        """
        if source_album_id == target_album_id:
            raise ValueError("Album źródłowy i docelowy muszą być różne.")
        src_grp = self.db.get(MediaGroup, source_album_id)
        dst_grp = self.db.get(MediaGroup, target_album_id)
        if not src_grp or not dst_grp:
            raise ValueError("Wybrany album źródłowy lub docelowy nie istnieje.")

        subtree_ids = self.get_group_subtree_ids(source_album_id)
        if target_album_id in subtree_ids:
            # Detach target album to root before removing source hierarchy
            dst_grp.parent_id = src_grp.parent_id
            self.db.flush()
            subtree_ids = [gid for gid in subtree_ids if gid != target_album_id]

        moved_count = (
            self.db.scalar(select(func.count(MediaItem.id)).where(MediaItem.group_id.in_(subtree_ids))) or 0
        )
        self.db.execute(
            update(MediaItem).where(MediaItem.group_id.in_(subtree_ids)).values(group_id=target_album_id)
        )
        src_name = src_grp.name
        dst_name = dst_grp.name
        for gid in reversed(subtree_ids):
            self.db.execute(delete(MediaGroup).where(MediaGroup.id == gid))
        self.db.commit()

        OperationLogger.log(
            "ALBUM_TRANSFER",
            f"Przerzucono {moved_count} zdjęć z albumu '{src_name}' do '{dst_name}' i usunięto album '{src_name}'.",
            status="OK",
            db=self.db,
        )
        return int(moved_count), src_name, dst_name

    def ensure_important_album_hierarchy(self, item: MediaItem, vol_label: str) -> MediaGroup:
        """
        Ensures album 'Ważne - <vol_label>' -> '<YYYY>' -> '<YYYY-MM>' exists
        and assigns item.group_id to the month subalbum (Requirement 2b & 2d).
        """
        root_name = f"Ważne - {vol_label}"
        yr_name = str(item.effective_date.year)
        mo_name = item.effective_date.strftime("%Y-%m")

        root_grp = self.db.scalar(
            select(MediaGroup).where(MediaGroup.name == root_name, MediaGroup.parent_id.is_(None), MediaGroup.is_trashed == False)
        )
        if not root_grp:
            root_grp = MediaGroup(name=root_name, parent_id=None, description=f"Ważne zdjęcia z urządzenia {vol_label}")
            self.db.add(root_grp)
            self.db.flush()

        yr_grp = self.db.scalar(
            select(MediaGroup).where(MediaGroup.name == yr_name, MediaGroup.parent_id == root_grp.id, MediaGroup.is_trashed == False)
        )
        if not yr_grp:
            yr_grp = MediaGroup(name=yr_name, parent_id=root_grp.id)
            self.db.add(yr_grp)
            self.db.flush()

        mo_grp = self.db.scalar(
            select(MediaGroup).where(MediaGroup.name == mo_name, MediaGroup.parent_id == yr_grp.id, MediaGroup.is_trashed == False)
        )
        if not mo_grp:
            mo_grp = MediaGroup(name=mo_name, parent_id=yr_grp.id)
            self.db.add(mo_grp)
            self.db.flush()

        item.group_id = mo_grp.id
        return mo_grp

    def detect_files_without_backup(self) -> List[MediaItem]:
        """
        Returns all non-trashed primary MediaItems that do NOT have a backup copy
        on a different physical device (volume_id != primary.volume_id).
        """
        self.flag_duplicate_clusters()
        all_items = list(
            self.db.scalars(
                select(MediaItem)
                .options(joinedload(MediaItem.volume), joinedload(MediaItem.group))
                .where(MediaItem.is_trashed == False)
                .order_by(MediaItem.effective_date.desc())
            ).unique().all()
        )
        by_hash: Dict[str, List[MediaItem]] = defaultdict(list)
        for it in all_items:
            by_hash[it.file_hash].append(it)

        unbacked: List[MediaItem] = []
        for fhash, cluster in by_hash.items():
            primary = next((x for x in cluster if x.is_primary_duplicate), cluster[0])
            has_other_device_backup = any(
                (x.volume_id != primary.volume_id and x.is_backup_copy) for x in cluster
            )
            if not has_other_device_backup:
                unbacked.append(primary)
        return unbacked

    @classmethod
    def check_volume_free_space(cls, drive_letter: str, required_bytes: int) -> Tuple[bool, int]:
        """Checks if the mounted volume has enough free space (including safety buffer)."""
        try:
            usage = shutil.disk_usage(drive_letter)
            has_space = usage.free >= (required_bytes + cls.MIN_FREE_SPACE_BUFFER_BYTES)
            return has_space, int(usage.free)
        except Exception:
            return False, 0

    def build_backup_and_dedup_plan(self, preferred_target_volume_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyzes the entire database to build a pairwise device execution plan for:
        1) Assigning Originals with same-device duplicates to 'Ważne - <device>' -> YYYY -> YYYY-MM (Rule 2b, 2d).
        2) Creating missing backups in 'backup/' folder on a DIFFERENT device than Original (Rule 1a, 1d, 1g).
        3) Removing all extra duplicate copies after 100% multi-layer verification so only [1 Original + 1 Backup] remain (Rule 1e, 1f).
        4) Grouping all file operations by unordered volume pairs {V_a, V_b} so each connected pair is processed bidirectionally once (Rule 1h).
        """
        self.refresh_volumes_online_status()
        self.flag_duplicate_clusters()

        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        all_items = list(
            self.db.scalars(
                select(MediaItem)
                .where(MediaItem.is_trashed == False)
                .order_by(MediaItem.effective_date.asc(), MediaItem.relative_path.asc(), MediaItem.id.asc())
            ).all()
        )

        by_hash: Dict[str, List[MediaItem]] = defaultdict(list)
        for it in all_items:
            by_hash[it.file_hash].append(it)

        # Choose default backup target volume for each source volume
        sorted_vols = sorted(volumes.values(), key=lambda v: (not v.is_online, v.id))

        def pick_backup_volume(src_vol_id: int) -> Optional[WindowsVolume]:
            if preferred_target_volume_id and preferred_target_volume_id != src_vol_id and preferred_target_volume_id in volumes:
                return volumes[preferred_target_volume_id]
            for cand in sorted_vols:
                if cand.id != src_vol_id:
                    return cand
            return None

        pair_tasks: Dict[Tuple[int, int], Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: {"backups_to_copy": [], "backups_to_relocate": [], "duplicates_to_delete": []}
        )
        important_candidates: List[Tuple[MediaItem, WindowsVolume]] = []
        unbacked_count = 0
        redundant_duplicates_count = 0
        single_device_warning = len(volumes) < 2

        for fhash, cluster in by_hash.items():
            # Group non-backup copies by volume to apply Rule 2d & 2b
            non_backup_by_vol: Dict[int, List[MediaItem]] = defaultdict(list)
            for it in cluster:
                if not it.is_backup_copy:
                    non_backup_by_vol[it.volume_id].append(it)

            # Select Original: prefer volume with >=2 non-backup copies (Rule 2d), else existing primary
            best_orig: Optional[MediaItem] = None
            best_vol_dup_cnt = -1
            for vid, v_items in non_backup_by_vol.items():
                if len(v_items) > best_vol_dup_cnt:
                    best_vol_dup_cnt = len(v_items)
                    best_orig = v_items[0]
            if best_orig is None:
                best_orig = cluster[0]

            orig_vol = volumes.get(best_orig.volume_id)
            if best_vol_dup_cnt >= 2 and orig_vol:
                important_candidates.append((best_orig, orig_vol))

            # Find existing backup copy on a DIFFERENT device
            other_vol_copies = [x for x in cluster if x.volume_id != best_orig.volume_id and x.id != best_orig.id]
            chosen_backup: Optional[MediaItem] = None
            if other_vol_copies:
                # Prefer copy already in 'backup/' folder, then online volume
                other_vol_copies.sort(
                    key=lambda x: (not x.is_backup_copy, not volumes[x.volume_id].is_online if x.volume_id in volumes else True, x.id)
                )
                chosen_backup = other_vol_copies[0]

            if chosen_backup is None:
                unbacked_count += 1
                target_vol = pick_backup_volume(best_orig.volume_id)
                if target_vol:
                    pair_key = (min(best_orig.volume_id, target_vol.id), max(best_orig.volume_id, target_vol.id))
                    pair_tasks[pair_key]["backups_to_copy"].append(
                        {
                            "orig_item_id": best_orig.id,
                            "src_vol_id": best_orig.volume_id,
                            "dst_vol_id": target_vol.id,
                            "file_size": best_orig.file_size_bytes,
                            "file_name": best_orig.file_name,
                        }
                    )
            else:
                if not chosen_backup.is_backup_copy:
                    pair_key = (min(best_orig.volume_id, chosen_backup.volume_id), max(best_orig.volume_id, chosen_backup.volume_id))
                    pair_tasks[pair_key]["backups_to_relocate"].append(
                        {
                            "backup_item_id": chosen_backup.id,
                            "vol_id": chosen_backup.volume_id,
                        }
                    )

            # All other copies (neither best_orig nor chosen_backup) are redundant duplicates to be removed
            for item in cluster:
                if item.id == best_orig.id:
                    continue
                if chosen_backup and item.id == chosen_backup.id:
                    continue
                redundant_duplicates_count += 1
                ref_vol_id = best_orig.volume_id
                pair_key = (min(ref_vol_id, item.volume_id), max(ref_vol_id, item.volume_id))
                pair_tasks[pair_key]["duplicates_to_delete"].append(
                    {
                        "dup_item_id": item.id,
                        "dup_vol_id": item.volume_id,
                        "ref_item_id": best_orig.id,
                        "ref_vol_id": best_orig.volume_id,
                        "backup_item_id": chosen_backup.id if chosen_backup else None,
                    }
                )

        # Order pairs: pairs where both volumes are online first, then pairs with 1 online volume, then offline
        ordered_pairs = []
        for (va_id, vb_id), tasks in pair_tasks.items():
            va = volumes.get(va_id)
            vb = volumes.get(vb_id)
            both_online = bool(va and vb and va.is_online and vb.is_online)
            one_online = bool((va and va.is_online) or (vb and vb.is_online))
            total_ops = (
                len(tasks["backups_to_copy"])
                + len(tasks["backups_to_relocate"])
                + len(tasks["duplicates_to_delete"])
            )
            if total_ops > 0:
                ordered_pairs.append(
                    {
                        "pair_key": (va_id, vb_id),
                        "vol_a": va,
                        "vol_b": vb,
                        "both_online": both_online,
                        "one_online": one_online,
                        "tasks": tasks,
                        "total_ops": total_ops,
                    }
                )

        ordered_pairs.sort(key=lambda p: (not p["both_online"], not p["one_online"], p["pair_key"]))

        return {
            "volumes": volumes,
            "important_candidates": important_candidates,
            "ordered_pairs": ordered_pairs,
            "unbacked_count": unbacked_count,
            "redundant_duplicates_count": redundant_duplicates_count,
            "single_device_warning": single_device_warning,
        }

    def execute_backup_and_deduplication(
        self,
        preferred_target_volume_id: Optional[int] = None,
        job_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the automated backup and 100%-verified deduplication workflow:
        - Assigns same-device duplicate originals to 'Ważne - <device>' -> YYYY -> YYYY-MM.
        - Processes all currently ONLINE volume pairs {V_a, V_b} bidirectionally (A->B and B->A).
        - Checks available free space on target volumes automatically.
        - Copies backups into the 'backup' directory on the other device.
        - Verifies 100% identity (size, SHA-256, BLAKE2b, MD5, byte-by-byte stream, image decode) before deleting any duplicate.
        - Returns the next required offline volume pair if any devices need to be connected by the user.
        """
        from app.operation_log import ProcessManager

        self.backup_mgr.create_backup("pre_auto_backup_dedup")
        plan = self.build_backup_and_dedup_plan(preferred_target_volume_id=preferred_target_volume_id)
        volumes: Dict[int, WindowsVolume] = plan["volumes"]

        # Step 1: Assign Originals with same-device duplicates to 'Ważne - <volume_label>' -> YYYY -> YYYY-MM (Rule 2b, 2d)
        important_assigned = 0
        for orig_item, orig_vol in plan["important_candidates"]:
            self.ensure_important_album_hierarchy(orig_item, orig_vol.volume_label)
            important_assigned += 1
        if important_assigned > 0:
            self.db.commit()

        backups_created = 0
        backups_relocated = 0
        duplicates_deleted = 0
        verification_blocks = 0
        space_warnings: List[str] = []
        completed_pairs: List[str] = []
        pending_offline_pairs: List[Dict[str, Any]] = []

        for pair_entry in plan["ordered_pairs"]:
            if job_id and ProcessManager.is_stop_requested(job_id):
                break

            va: Optional[WindowsVolume] = pair_entry["vol_a"]
            vb: Optional[WindowsVolume] = pair_entry["vol_b"]
            tasks = pair_entry["tasks"]

            if not (va and vb and va.is_online and vb.is_online):
                pending_offline_pairs.append(pair_entry)
                continue

            # Map newly created backup items by original item ID so same-pass deduplication knows backup is ready
            created_backup_by_orig: Dict[str, MediaItem] = {}

            # 2A: Bidirectional backup copying (Va -> Vb\backup and Vb -> Va\backup)
            for copy_task in tasks["backups_to_copy"]:
                if job_id and ProcessManager.is_stop_requested(job_id):
                    break
                orig_item = self.db.get(MediaItem, copy_task["orig_item_id"])
                src_vol = volumes.get(copy_task["src_vol_id"])
                dst_vol = volumes.get(copy_task["dst_vol_id"])
                if not orig_item or not src_vol or not dst_vol:
                    continue

                src_path = Path(src_vol.last_known_drive_letter) / orig_item.relative_path
                if not src_path.exists():
                    continue

                has_space, free_bytes = self.check_volume_free_space(
                    dst_vol.last_known_drive_letter, orig_item.file_size_bytes
                )
                if not has_space:
                    space_warnings.append(
                        f"Brak miejsca na dysku {dst_vol.volume_label} ({dst_vol.last_known_drive_letter}): "
                        f"wymagane {orig_item.file_size_bytes / (1024*1024):.1f} MB, wolne {free_bytes / (1024*1024):.1f} MB."
                    )
                    continue

                dst_root = Path(dst_vol.last_known_drive_letter)
                sub_month = orig_item.effective_date.strftime("%Y/%Y-%m")
                backup_dir = dst_root / "backup" / sub_month
                backup_dir.mkdir(parents=True, exist_ok=True)
                dest_path = backup_dir / orig_item.file_name
                c = 1
                while dest_path.exists():
                    # If an identical file is already at dest_path, check if it's already indexed
                    dest_path = backup_dir / f"{src_path.stem}_{c}{src_path.suffix}"
                    c += 1

                shutil.copy2(str(src_path), str(dest_path))

                # 100% Verification of newly created backup copy against original
                verified, reason, full_sha = FastHasher.verify_files_identical_100_percent(src_path, dest_path)
                if not verified:
                    try:
                        dest_path.unlink()
                    except OSError as exc:
                        logger.debug("Cleanup failed backup copy %s: %s", dest_path, exc)
                    verification_blocks += 1
                    OperationLogger.log(
                        "BACKUP_VERIFY_FAIL",
                        f"Weryfikacja 100% nie powiodła się dla backupu {orig_item.file_name}: {reason}",
                        status="ERROR",
                        db=self.db,
                        commit=False,
                    )
                    continue

                rel_backup = dest_path.resolve().relative_to(dst_root.resolve()).as_posix()
                orig_item.verified_sha256 = full_sha
                import uuid
                new_bkp = MediaItem(
                    id=str(uuid.uuid4()),
                    volume_id=dst_vol.id,
                    relative_path=rel_backup,
                    file_name=dest_path.name,
                    file_size_bytes=orig_item.file_size_bytes,
                    media_type=orig_item.media_type,
                    mime_type=orig_item.mime_type,
                    file_hash=orig_item.file_hash,
                    hash_algorithm=orig_item.hash_algorithm,
                    is_primary_duplicate=False,
                    is_backup_copy=True,
                    verified_sha256=full_sha,
                    width=orig_item.width,
                    height=orig_item.height,
                    duration_seconds=orig_item.duration_seconds,
                    rotation_angle=orig_item.rotation_angle,
                    exif_date=orig_item.exif_date,
                    filename_date=orig_item.filename_date,
                    fs_ctime=orig_item.fs_ctime,
                    fs_mtime=orig_item.fs_mtime,
                    effective_date=orig_item.effective_date,
                    has_date_anomaly=orig_item.has_date_anomaly,
                    date_anomaly_reason=orig_item.date_anomaly_reason,
                    is_corrupted=False,
                    is_blurry=orig_item.is_blurry,
                    blur_score=orig_item.blur_score,
                    thumbnail_relative_path=orig_item.thumbnail_relative_path,
                    group_id=None,
                    is_trashed=False,
                )
                self.db.add(new_bkp)
                self.db.flush()
                created_backup_by_orig[orig_item.id] = new_bkp
                backups_created += 1
                if job_id:
                    ProcessManager.update_progress(
                        job_id, processed_delta=1, message=f"Utworzono backup: {orig_item.file_name}", db=self.db
                    )

            # 2B: Relocate existing cross-device copy into 'backup/' folder so backup is always in 'backup' (Rule 1d)
            for reloc_task in tasks["backups_to_relocate"]:
                bkp_item = self.db.get(MediaItem, reloc_task["backup_item_id"])
                bkp_vol = volumes.get(reloc_task["vol_id"])
                if not bkp_item or not bkp_vol or not bkp_vol.is_online:
                    continue
                vol_root = Path(bkp_vol.last_known_drive_letter)
                curr_path = vol_root / bkp_item.relative_path
                if not curr_path.exists():
                    continue
                sub_month = bkp_item.effective_date.strftime("%Y/%Y-%m")
                target_bkp_dir = vol_root / "backup" / sub_month
                target_bkp_dir.mkdir(parents=True, exist_ok=True)
                target_bkp_path = target_bkp_dir / bkp_item.file_name
                if curr_path.resolve() != target_bkp_path.resolve():
                    c = 1
                    while target_bkp_path.exists():
                        target_bkp_path = target_bkp_dir / f"{curr_path.stem}_{c}{curr_path.suffix}"
                        c += 1
                    shutil.move(str(curr_path), str(target_bkp_path))
                    bkp_item.file_name = target_bkp_path.name
                    bkp_item.relative_path = target_bkp_path.resolve().relative_to(vol_root.resolve()).as_posix()
                bkp_item.is_backup_copy = True
                bkp_item.is_primary_duplicate = False
                backups_relocated += 1

            # 2C: 100%-Verified Removal of Redundant Duplicates (leaving only 1 Original + 1 Backup)
            for dup_task in tasks["duplicates_to_delete"]:
                if job_id and ProcessManager.is_stop_requested(job_id):
                    break
                dup_item = self.db.get(MediaItem, dup_task["dup_item_id"])
                ref_item = self.db.get(MediaItem, dup_task["ref_item_id"])
                if not dup_item or not ref_item:
                    continue

                dup_vol = volumes.get(dup_item.volume_id)
                ref_vol = volumes.get(ref_item.volume_id)
                if not dup_vol or not ref_vol or not dup_vol.is_online or not ref_vol.is_online:
                    continue

                dup_path = Path(dup_vol.last_known_drive_letter) / dup_item.relative_path
                ref_path = Path(ref_vol.last_known_drive_letter) / ref_item.relative_path

                # Safety guard: ensure that if there are multiple volumes in the system, a backup copy exists
                bkp_id = dup_task.get("backup_item_id")
                bkp_obj = self.db.get(MediaItem, bkp_id) if bkp_id else created_backup_by_orig.get(ref_item.id)
                if len(volumes) >= 2 and bkp_obj is None:
                    # Do not delete a duplicate if backup on another device has not been secured yet
                    continue

                if not dup_path.exists():
                    # Physical duplicate file already absent, but verify original exists before cleaning DB record
                    if ref_path.exists() and ref_path.stat().st_size == ref_item.file_size_bytes:
                        self.db.delete(dup_item)
                        duplicates_deleted += 1
                    continue

                # Perform 100% multi-layer verification between Original (ref_path) and Duplicate (dup_path)
                is_identical, verify_msg, full_sha = FastHasher.verify_files_identical_100_percent(ref_path, dup_path)
                if not is_identical:
                    verification_blocks += 1
                    OperationLogger.log(
                        "DEDUP_SAFETY_BLOCK",
                        f"Zablokowano usunięcie {dup_path} (brak 100% zgodności z oryginałem {ref_path}): {verify_msg}",
                        status="WARN",
                        db=self.db,
                        commit=False,
                    )
                    continue

                # If backup copy is also on an online drive right now, verify backup exists on disk too
                if bkp_obj:
                    bkp_vol = volumes.get(bkp_obj.volume_id)
                    if bkp_vol and bkp_vol.is_online:
                        bkp_disk_p = Path(bkp_vol.last_known_drive_letter) / bkp_obj.relative_path
                        if not bkp_disk_p.exists() or bkp_disk_p.resolve() == dup_path.resolve():
                            verification_blocks += 1
                            continue

                # Safe to remove redundant duplicate file
                try:
                    dup_path.unlink()
                    ref_item.verified_sha256 = full_sha
                    self.db.delete(dup_item)
                    duplicates_deleted += 1
                    if job_id:
                        ProcessManager.update_progress(
                            job_id, processed_delta=1, message=f"Usunięto zweryfikowany duplikat: {dup_item.file_name}", db=self.db
                        )
                except OSError as exc:
                    logger.warning("Could not delete verified duplicate %s: %s", dup_path, exc)

            self.db.commit()
            pair_label = (
                f"{va.volume_label} ({va.last_known_drive_letter})"
                if va.id == vb.id
                else f"{va.volume_label} ({va.last_known_drive_letter}) ↔ {vb.volume_label} ({vb.last_known_drive_letter})"
            )
            completed_pairs.append(pair_label)

        self.flag_duplicate_clusters()
        remaining_unbacked = len(self.detect_files_without_backup())

        next_pair_prompt = None
        if pending_offline_pairs:
            nxt = pending_offline_pairs[0]
            va = nxt["vol_a"]
            vb = nxt["vol_b"]
            next_pair_prompt = {
                "vol_a_label": f"{va.volume_label} ({va.last_known_drive_letter}, VSN: {va.volume_serial_number})" if va else "Nieznany",
                "vol_b_label": f"{vb.volume_label} ({vb.last_known_drive_letter}, VSN: {vb.volume_serial_number})" if vb else "Nieznany",
                "vol_a_online": bool(va and va.is_online),
                "vol_b_online": bool(vb and vb.is_online),
                "backups_to_copy": len(nxt["tasks"]["backups_to_copy"]),
                "duplicates_to_delete": len(nxt["tasks"]["duplicates_to_delete"]),
                "remaining_pairs_count": len(pending_offline_pairs),
            }

        summary_msg = (
            f"Backup utworzony: {backups_created} | Przeniesiono do 'backup/': {backups_relocated} | "
            f"Usunięto duplikatów (100% weryfikacji): {duplicates_deleted} | "
            f"Dodano do albumu 'Ważne': {important_assigned}"
        )
        OperationLogger.log(
            "BACKUP_DEDUP",
            summary_msg,
            status="OK" if not pending_offline_pairs else "PARTIAL",
            details={
                "backups_created": backups_created,
                "backups_relocated": backups_relocated,
                "duplicates_deleted": duplicates_deleted,
                "important_assigned": important_assigned,
                "verification_blocks": verification_blocks,
                "completed_pairs": completed_pairs,
                "pending_pairs": len(pending_offline_pairs),
            },
            db=self.db,
        )

        return {
            "backups_created": backups_created,
            "backups_relocated": backups_relocated,
            "duplicates_deleted": duplicates_deleted,
            "important_assigned": important_assigned,
            "verification_blocks": verification_blocks,
            "space_warnings": space_warnings,
            "completed_pairs": completed_pairs,
            "pending_offline_pairs_count": len(pending_offline_pairs),
            "next_pair_prompt": next_pair_prompt,
            "remaining_unbacked": remaining_unbacked,
            "single_device_warning": plan["single_device_warning"],
            "summary": summary_msg,
        }

    def schedule_or_execute_moves(
        self,
        items: List[MediaItem],
        target_volume_id: int,
        target_subfolder: str = "",
        force_offline_todo: bool = False,
    ) -> Dict[str, Any]:
        """
        Moves files between devices or directories.
        Requirement 4b: If source or target device is offline (or force_offline_todo=True),
        records the move in pending_file_moves (TODO list) and immediately updates the logical
        location (volume_id, relative_path) in media_items so the user sees the file in its new logical place.
        """
        self.refresh_volumes_online_status()
        target_vol = self.db.get(WindowsVolume, target_volume_id)
        if not target_vol:
            raise ValueError("Dysk docelowy nie istnieje w bazie.")

        clean_sub = target_subfolder.strip().strip("/\\")
        moved_physically = 0
        queued_offline = 0

        if target_vol.is_online and not force_offline_todo:
            self.backup_mgr.create_backup("pre_move_or_todo")

        for it in items:
            src_vol = self.db.get(WindowsVolume, it.volume_id)
            existing_todo = self.db.scalar(
                select(PendingFileMove).where(
                    PendingFileMove.media_id == it.id,
                    PendingFileMove.status == "PENDING",
                )
            )

            true_src_vol_id = existing_todo.source_volume_id if existing_todo else it.volume_id
            true_src_rel = existing_todo.source_relative_path if existing_todo else it.relative_path
            true_src_name = existing_todo.source_file_name if existing_todo else it.file_name
            true_src_vol = self.db.get(WindowsVolume, true_src_vol_id)

            new_rel = f"{clean_sub}/{it.file_name}" if clean_sub else it.file_name
            new_rel = Path(new_rel).as_posix()

            can_move_physically = (
                not force_offline_todo
                and true_src_vol is not None
                and true_src_vol.is_online
                and target_vol.is_online
                and (Path(true_src_vol.last_known_drive_letter) / true_src_rel).exists()
            )

            if can_move_physically:
                src_file = Path(true_src_vol.last_known_drive_letter) / true_src_rel
                dst_root = Path(target_vol.last_known_drive_letter)
                dst_dir = (dst_root / clean_sub) if clean_sub else dst_root
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / true_src_name

                if src_file.resolve() != dst_file.resolve():
                    c = 1
                    while dst_file.exists():
                        dst_file = dst_dir / f"{src_file.stem}_{c}{src_file.suffix}"
                        c += 1
                    shutil.move(str(src_file), str(dst_file))

                it.volume_id = target_vol.id
                it.file_name = dst_file.name
                it.relative_path = dst_file.resolve().relative_to(dst_root.resolve()).as_posix()
                if existing_todo:
                    self.db.delete(existing_todo)
                moved_physically += 1
            else:
                # Logical move in DB + schedule physical move in pending_file_moves (Lista TODO)
                if true_src_vol_id == target_vol.id and true_src_rel == new_rel:
                    # Returned to original physical position -> remove pending todo
                    if existing_todo:
                        self.db.delete(existing_todo)
                    it.volume_id = true_src_vol_id
                    it.relative_path = true_src_rel
                    it.file_name = true_src_name
                else:
                    if existing_todo:
                        existing_todo.target_volume_id = target_vol.id
                        existing_todo.target_relative_path = new_rel
                        existing_todo.target_file_name = it.file_name
                    else:
                        todo = PendingFileMove(
                            media_id=it.id,
                            source_volume_id=true_src_vol_id,
                            source_relative_path=true_src_rel,
                            source_file_name=true_src_name,
                            target_volume_id=target_vol.id,
                            target_relative_path=new_rel,
                            target_file_name=it.file_name,
                            status="PENDING",
                        )
                        self.db.add(todo)
                    it.volume_id = target_vol.id
                    it.relative_path = new_rel
                    queued_offline += 1

        self.db.commit()
        OperationLogger.log(
            "FILE_MOVE",
            f"Przeniesiono fizycznie: {moved_physically}, dodano do listy TODO (offline): {queued_offline} -> {target_vol.volume_label}/{clean_sub}",
            status="OK",
            db=self.db,
        )
        return {
            "moved_physically": moved_physically,
            "queued_offline": queued_offline,
            "target_volume_label": target_vol.volume_label,
            "target_subfolder": clean_sub,
        }

    def get_pending_moves_summary(self) -> Dict[str, Any]:
        """Returns the full list of pending offline file moves and groups them by volume pairs (Rule 4c, 4e)."""
        self.refresh_volumes_online_status()
        todos = list(
            self.db.scalars(
                select(PendingFileMove)
                .options(
                    joinedload(PendingFileMove.media_item),
                    joinedload(PendingFileMove.source_volume),
                    joinedload(PendingFileMove.target_volume),
                )
                .where(PendingFileMove.status == "PENDING")
                .order_by(PendingFileMove.id.asc())
            ).all()
        )

        pairs_map: Dict[Tuple[int, int], List[PendingFileMove]] = defaultdict(list)
        for td in todos:
            pk = (min(td.source_volume_id, td.target_volume_id), max(td.source_volume_id, td.target_volume_id))
            pairs_map[pk].append(td)

        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        ordered_pairs = []
        for (va_id, vb_id), p_todos in pairs_map.items():
            va = volumes.get(va_id)
            vb = volumes.get(vb_id)
            both_online = bool(va and vb and va.is_online and vb.is_online)
            one_online = bool((va and va.is_online) or (vb and vb.is_online))
            a_to_b = sum(1 for x in p_todos if x.source_volume_id == va_id and x.target_volume_id == vb_id)
            b_to_a = sum(1 for x in p_todos if x.source_volume_id == vb_id and x.target_volume_id == va_id)
            ordered_pairs.append(
                {
                    "pair_key": (va_id, vb_id),
                    "vol_a": va,
                    "vol_b": vb,
                    "both_online": both_online,
                    "one_online": one_online,
                    "a_to_b": a_to_b,
                    "b_to_a": b_to_a,
                    "todos": p_todos,
                }
            )
        ordered_pairs.sort(key=lambda p: (not p["both_online"], not p["one_online"], p["pair_key"]))

        next_offline_pair = next((p for p in ordered_pairs if not p["both_online"]), None)
        return {
            "todos": todos,
            "total_pending": len(todos),
            "ordered_pairs": ordered_pairs,
            "next_offline_pair": next_offline_pair,
        }

    def rollback_pending_moves(self, todo_ids: Optional[List[int]] = None) -> int:
        """
        Requirement 4d: Restores previous file locations in the database and clears the TODO list
        when the user cancels pending moves or when a physical move operation fails.
        """
        stmt = select(PendingFileMove).where(PendingFileMove.status == "PENDING")
        if todo_ids:
            stmt = stmt.where(PendingFileMove.id.in_(todo_ids))
        todos = list(self.db.scalars(stmt).all())

        reverted = 0
        for td in todos:
            it = self.db.get(MediaItem, td.media_id)
            if it:
                it.volume_id = td.source_volume_id
                it.relative_path = td.source_relative_path
                it.file_name = td.source_file_name
            self.db.delete(td)
            reverted += 1

        self.db.commit()
        if reverted > 0:
            OperationLogger.log(
                "TODO_ROLLBACK",
                f"Przywrócono pierwotne lokalizacje w bazie i wyczyszczono listę TODO dla {reverted} plików.",
                status="WARN",
                db=self.db,
            )
        return reverted

    def execute_pending_moves_pairwise(self) -> Dict[str, Any]:
        """
        Requirement 4d & 4e: Executes pending physical moves pair-by-pair bidirectionally (Va <-> Vb).
        If any physical operation fails on a connected pair, rolls back the database state to original
        locations and clears the TODO list as required by 4d.
        """
        summary = self.get_pending_moves_summary()
        ordered_pairs = summary["ordered_pairs"]
        if not ordered_pairs:
            return {
                "executed_count": 0,
                "rolled_back_count": 0,
                "remaining_count": 0,
                "next_offline_pair": None,
                "status": "empty",
                "message": "Lista TODO jest pusta — brak oczekujących przeniesień.",
            }

        self.backup_mgr.create_backup("pre_todo_pairwise_exec")
        executed_count = 0
        completed_physical_moves: List[Tuple[Path, Path, MediaItem, PendingFileMove]] = []

        try:
            for pair in ordered_pairs:
                if not pair["both_online"]:
                    continue
                va: WindowsVolume = pair["vol_a"]
                vb: WindowsVolume = pair["vol_b"]

                for td in pair["todos"]:
                    it = self.db.get(MediaItem, td.media_id)
                    src_vol = va if td.source_volume_id == va.id else vb
                    dst_vol = va if td.target_volume_id == va.id else vb

                    src_path = Path(src_vol.last_known_drive_letter) / td.source_relative_path
                    if not src_path.exists():
                        raise FileNotFoundError(
                            f"Brak pliku źródłowego na podłączonym dysku: {src_path}"
                        )

                    has_space, _ = self.check_volume_free_space(
                        dst_vol.last_known_drive_letter, src_path.stat().st_size
                    )
                    if not has_space:
                        raise RuntimeError(
                            f"Brak wystarczającego miejsca na dysku docelowym {dst_vol.volume_label} ({dst_vol.last_known_drive_letter})."
                        )

                    dst_root = Path(dst_vol.last_known_drive_letter)
                    dst_path = dst_root / td.target_relative_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)

                    if src_path.resolve() != dst_path.resolve():
                        c = 1
                        while dst_path.exists():
                            dst_path = dst_path.parent / f"{src_path.stem}_{c}{src_path.suffix}"
                            c += 1
                        shutil.move(str(src_path), str(dst_path))
                        completed_physical_moves.append((src_path, dst_path, it, td))

                    if it:
                        it.volume_id = dst_vol.id
                        it.file_name = dst_path.name
                        it.relative_path = dst_path.resolve().relative_to(dst_root.resolve()).as_posix()
                    self.db.delete(td)
                    executed_count += 1

            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            # Revert any physical files moved during the failed transaction
            for orig_p, moved_p, _, _ in reversed(completed_physical_moves):
                try:
                    if moved_p.exists() and not orig_p.exists():
                        orig_p.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(moved_p), str(orig_p))
                except OSError as move_exc:
                    logger.warning("Rollback physical move warning: %s", move_exc)

            # Requirement 4d: Restore original locations in DB and clear TODO list on failure
            rolled_back = self.rollback_pending_moves()
            OperationLogger.log(
                "TODO_EXEC_FAIL",
                f"Błąd fizycznego przenoszenia ({exc}). Przywrócono oryginalne lokalizacje dla {rolled_back} plików.",
                status="ERROR",
                db=self.db,
            )
            return {
                "executed_count": 0,
                "rolled_back_count": rolled_back,
                "remaining_count": 0,
                "next_offline_pair": None,
                "status": "rolled_back",
                "message": f"Operacja fizyczna nie powiodła się ({exc}). Przywrócono poprzednie lokalizacje w bazie i wyczyszczono listę TODO ({rolled_back} poz.).",
            }

        updated_summary = self.get_pending_moves_summary()
        OperationLogger.log(
            "TODO_EXEC_SUCCESS",
            f"Wykonano fizyczne przeniesienia dla {executed_count} plików. Pozostało w kolejce: {updated_summary['total_pending']}.",
            status="OK",
            db=self.db,
        )
        return {
            "executed_count": executed_count,
            "rolled_back_count": 0,
            "remaining_count": updated_summary["total_pending"],
            "next_offline_pair": updated_summary["next_offline_pair"],
            "status": "ok",
            "message": f"Wykonano fizyczne przeniesienie {executed_count} plików (w obie strony dla podłączonych par dysków).",
        }

    def consolidate_group(self, group_id: int, target_volume_id: int) -> Dict[str, int]:
        """
        Moves all physical files of an album AND its child subalbums onto target USB drive,
        preserving the subalbum folder hierarchy on disk.
        If source or target volume is offline, schedules the moves in the offline TODO list (Rule 4b).
        """
        self.refresh_volumes_online_status()
        target_vol = self.db.get(WindowsVolume, target_volume_id)
        if not target_vol:
            raise RuntimeError("Dysk docelowy nie istnieje.")

        subtree_ids = self.get_group_subtree_ids(group_id)
        groups_by_id = {g.id: g for g in self.db.scalars(select(MediaGroup)).all()}

        def get_group_rel_folder(gid: Optional[int]) -> Path:
            parts = []
            curr = groups_by_id.get(gid) if gid else None
            depth = 0
            while curr and depth < 10:
                parts.append(curr.name)
                curr = groups_by_id.get(curr.parent_id) if curr.parent_id else None
                depth += 1
            return Path(*reversed(parts)) if parts else Path("Albumy")

        items = list(self.db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids), MediaItem.is_trashed == False)).all())
        by_subfolder: Dict[str, List[MediaItem]] = defaultdict(list)
        for it in items:
            rel_folder = (Path("DomoweArchiwumAlbumy") / get_group_rel_folder(it.group_id)).as_posix()
            by_subfolder[rel_folder].append(it)

        moved_phys = 0
        queued_todo = 0
        for rel_folder, sub_items in by_subfolder.items():
            res = self.schedule_or_execute_moves(sub_items, target_volume_id, target_subfolder=rel_folder)
            moved_phys += res["moved_physically"]
            queued_todo += res["queued_offline"]

        return {"moved_physically": moved_phys, "queued_offline": queued_todo}

    def auto_group_items_by_date(self, items: List[MediaItem]) -> int:
        """Automatically builds Year -> Year-Month album hierarchy for a list of items."""
        existing = {(g.name, g.parent_id): g for g in self.db.scalars(select(MediaGroup)).all()}
        updated = 0

        for it in items:
            yr_str = str(it.effective_date.year)
            mo_str = it.effective_date.strftime("%Y-%m")

            yr_grp = existing.get((yr_str, None))
            if not yr_grp:
                yr_grp = MediaGroup(name=yr_str, parent_id=None)
                self.db.add(yr_grp)
                self.db.flush()
                existing[(yr_str, None)] = yr_grp

            mo_grp = existing.get((mo_str, yr_grp.id))
            if not mo_grp:
                mo_grp = MediaGroup(name=mo_str, parent_id=yr_grp.id)
                self.db.add(mo_grp)
                self.db.flush()
                existing[(mo_str, yr_grp.id)] = mo_grp

            it.group_id = mo_grp.id
            updated += 1

        self.db.commit()
        return updated

    def auto_group_items_by_folder(self, items: List[MediaItem]) -> int:
        """Mirrors physical USB directory structure into hierarchical albums."""
        existing = {(g.name, g.parent_id): g for g in self.db.scalars(select(MediaGroup)).all()}
        updated = 0

        for it in items:
            parts = [p for p in Path(it.relative_path).parent.parts if p not in ("", ".", "/")]
            if not parts:
                parts = ["Root_Folder"]

            parent_id = None
            target_grp = None
            for part in parts[:4]:
                grp = existing.get((part, parent_id))
                if not grp:
                    grp = MediaGroup(name=part, parent_id=parent_id)
                    self.db.add(grp)
                    self.db.flush()
                    existing[(part, parent_id)] = grp
                parent_id = grp.id
                target_grp = grp

            if target_grp:
                it.group_id = target_grp.id
                updated += 1

        self.db.commit()
        return updated
