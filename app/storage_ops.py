from collections import defaultdict
import datetime
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import select, text, update, delete, func
from sqlalchemy.orm import Session, joinedload

from app.backup import BackupManager
import app.database as db_module
from app.models import MediaItem, WindowsVolume, MediaGroup, PendingFileMove
from app.operation_log import OperationLogger
from app.scanner import FastHasher, MetadataDateResolver, WindowsVolumeInspector

logger = logging.getLogger("StorageOps")


class StorageAutomationService:
    """Handles duplicate clustering, 100%-verified backup & deduplication, album management, and offline move queues."""

    MIN_FREE_SPACE_BUFFER_BYTES = 20 * 1024 * 1024  # 20 MB safety reserve on target drives
    BACKUP_DIR_NAME = "daf_backup"

    def __init__(self, db: Session):
        self.db = db
        self.backup_mgr = BackupManager()

    def refresh_volumes_online_status(self) -> List[WindowsVolume]:
        """Updates online status, drive letters, and persisted free/total disk capacity for all tracked volumes."""
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
                dl = (vol.last_known_drive_letter or "").strip()
                is_drive_root = bool(re.fullmatch(r"[A-Za-z]:\\?", dl))
                is_hw_vsn = bool(re.fullmatch(r"[0-9A-F]{4}-[0-9A-F]{4}", vol.volume_serial_number or ""))
                if connected_map and (is_drive_root or is_hw_vsn):
                    # Real drive enumeration found no match for this VSN, so the drive is genuinely disconnected
                    # even if another USB device reused the same Windows drive letter.
                    vol.is_online = False
                else:
                    vol.is_online = bool(dl and Path(dl).exists())
                    if vol.is_online:
                        vol.last_seen_at = now

            if vol.is_online:
                try:
                    usage = shutil.disk_usage(vol.last_known_drive_letter)
                    vol.total_space_bytes = int(usage.total)
                    vol.free_space_bytes = int(usage.free)
                except Exception as exc:
                    logger.debug("disk_usage update skipped for %s: %s", vol.last_known_drive_letter, exc)
        self.db.commit()
        return volumes

    def get_volume_space_map(self, volumes: List[WindowsVolume], commit_updates: bool = True) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Returns available capacity per volume, refreshing live usage for online drives and falling
        back to persisted SQLite values when drives are offline.
        """
        vol_space: Dict[int, Optional[Dict[str, Any]]] = {}
        dirty = False
        for v in volumes:
            if v.is_online:
                try:
                    u = shutil.disk_usage(v.last_known_drive_letter)
                    v.total_space_bytes = int(u.total)
                    v.free_space_bytes = int(u.free)
                    dirty = True
                except Exception as exc:
                    logger.debug("disk_usage warning for %s: %s", v.last_known_drive_letter, exc)

            total_b = int(v.total_space_bytes or 0)
            free_b = int(v.free_space_bytes or 0)
            if total_b > 0 or free_b > 0:
                used_b = max(0, total_b - free_b) if total_b else 0
                used_pct = max(0, min(100, int((used_b / total_b) * 100))) if total_b else 0
                vol_space[v.id] = {
                    "free_bytes": free_b,
                    "total_bytes": total_b,
                    "free_gb": free_b / (1024 ** 3),
                    "total_gb": total_b / (1024 ** 3),
                    "used_pct": used_pct,
                    "is_offline_cached": not v.is_online,
                }
            else:
                vol_space[v.id] = None

        if dirty and commit_updates:
            self.db.commit()
        return vol_space

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
                WHEN LOWER(relative_path) LIKE 'daf_backup/%' OR LOWER(relative_path) LIKE 'daf_backup\\%'
                  OR LOWER(relative_path) LIKE 'backup/%' OR LOWER(relative_path) LIKE 'backup\\%' THEN 1
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

    @staticmethod
    def parse_pair_keys(selected_pair_keys: Optional[List[str]]) -> Optional[Set[Tuple[int, int]]]:
        """Parses strings like '1:2' into canonical sorted (min_id, max_id) volume pair tuples."""
        if selected_pair_keys is None:
            return None
        parsed: Set[Tuple[int, int]] = set()
        for raw in selected_pair_keys:
            parts = str(raw).strip().split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                a, b = int(parts[0]), int(parts[1])
                parsed.add((min(a, b), max(a, b)))
        return parsed

    def run_exact_duplicate_analysis(
        self,
        target_ids: Optional[List[str]] = None,
        progress_cb=None,
    ) -> Dict[str, int]:
        """
        Performs 100%-verified duplicate detection across candidate clusters using full SHA-256,
        BLAKE2b, MD5, byte-by-byte filecmp, and image header verification on online drives.
        Separates any partial-hash collisions into unique primary records.
        """
        self.refresh_volumes_online_status()
        self.flag_duplicate_clusters()

        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        stmt = (
            select(MediaItem)
            .where(MediaItem.is_trashed == False)
            .order_by(MediaItem.effective_date.asc(), MediaItem.id.asc())
        )
        all_items = list(self.db.scalars(stmt).all())

        by_hash: Dict[str, List[MediaItem]] = defaultdict(list)
        for it in all_items:
            by_hash[it.file_hash].append(it)

        target_set = set(target_ids) if target_ids else None
        clusters_checked = 0
        verified_identical = 0
        mismatches_separated = 0
        offline_skipped = 0

        for fhash, cluster in by_hash.items():
            if len(cluster) < 2:
                continue
            if target_set and not any(x.id in target_set for x in cluster):
                continue

            clusters_checked += 1
            primary = next((x for x in cluster if x.is_primary_duplicate), cluster[0])
            p_vol = volumes.get(primary.volume_id)
            p_path = (Path(p_vol.last_known_drive_letter) / primary.relative_path) if (p_vol and p_vol.is_online) else None

            if not p_path or not p_path.exists():
                offline_skipped += len(cluster) - 1
                continue

            for candidate in cluster:
                if candidate.id == primary.id:
                    continue
                c_vol = volumes.get(candidate.volume_id)
                c_path = (Path(c_vol.last_known_drive_letter) / candidate.relative_path) if (c_vol and c_vol.is_online) else None
                if not c_path or not c_path.exists():
                    offline_skipped += 1
                    continue

                is_same, _, full_sha = FastHasher.verify_files_identical_100_percent(p_path, c_path)
                if is_same and full_sha:
                    primary.verified_sha256 = full_sha
                    candidate.verified_sha256 = full_sha
                    verified_identical += 1
                else:
                    # Resolve genuine non-identical file (e.g. partial hash collision on videos)
                    try:
                        true_sha, _, _ = FastHasher.compute_multi_hashes(c_path)
                        candidate.file_hash = true_sha
                        candidate.hash_algorithm = "sha256_verified"
                        candidate.is_primary_duplicate = True
                        mismatches_separated += 1
                    except Exception as exc:
                        logger.debug("Exact hash recompute skipped for %s: %s", c_path, exc)

                if progress_cb:
                    progress_cb(candidate.file_name)

        self.db.commit()
        dup_total = self.flag_duplicate_clusters()
        OperationLogger.log(
            "EXACT_DUP_SCAN",
            f"Dokładna analiza duplikatów 100%: klastry={clusters_checked}, potwierdzone={verified_identical}, rozdzielone kolizje={mismatches_separated}, pominięte offline={offline_skipped}",
            status="OK",
            db=self.db,
        )
        return {
            "clusters": clusters_checked,
            "verified_identical": verified_identical,
            "mismatches_separated": mismatches_separated,
            "offline_skipped": offline_skipped,
            "dup_total": dup_total,
        }

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

    def detect_files_without_backup(self, recluster: bool = True) -> List[MediaItem]:
        """
        Returns all non-trashed primary MediaItems that do NOT have a backup copy
        on a different physical device (volume_id != primary.volume_id).
        """
        if recluster:
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
    def check_volume_free_space(
        cls,
        drive_letter: str,
        required_bytes: int,
        fallback_free_bytes: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """Checks if the mounted volume (or cached offline space) has enough free space including buffer."""
        try:
            usage = shutil.disk_usage(drive_letter)
            has_space = usage.free >= (required_bytes + cls.MIN_FREE_SPACE_BUFFER_BYTES)
            return has_space, int(usage.free)
        except Exception:
            if fallback_free_bytes is not None and fallback_free_bytes > 0:
                has_space = fallback_free_bytes >= (required_bytes + cls.MIN_FREE_SPACE_BUFFER_BYTES)
                return has_space, int(fallback_free_bytes)
            return False, 0

    def build_backup_and_dedup_plan(
        self,
        preferred_target_volume_id: Optional[int] = None,
        selected_pair_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes the entire database to build a pairwise device execution plan and offline capacity balance for:
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

        vol_ids = sorted(volumes.keys())
        all_possible_pairs: List[Tuple[int, int]] = []
        for i in range(len(vol_ids)):
            for j in range(i + 1, len(vol_ids)):
                all_possible_pairs.append((vol_ids[i], vol_ids[j]))

        selected_pairs_set: Optional[Set[Tuple[int, int]]] = self.parse_pair_keys(selected_pair_keys)

        reclaimed_bytes_by_vol: Dict[int, int] = defaultdict(int)
        required_bytes_by_vol: Dict[int, int] = defaultdict(int)

        def pick_backup_volume(src_vol_id: int) -> Optional[WindowsVolume]:
            candidates: List[WindowsVolume] = []
            for cand in volumes.values():
                if cand.id == src_vol_id:
                    continue
                if selected_pairs_set is not None:
                    pk = (min(src_vol_id, cand.id), max(src_vol_id, cand.id))
                    if pk not in selected_pairs_set:
                        continue
                candidates.append(cand)

            if not candidates:
                return None

            if preferred_target_volume_id and any(c.id == preferred_target_volume_id for c in candidates):
                return volumes[preferred_target_volume_id]

            candidates.sort(
                key=lambda c: (
                    not c.is_online,
                    -(int(c.free_space_bytes or 0) + reclaimed_bytes_by_vol.get(c.id, 0) - required_bytes_by_vol.get(c.id, 0)),
                    c.id,
                )
            )
            return candidates[0]

        pair_tasks: Dict[Tuple[int, int], Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: {"backups_to_copy": [], "backups_to_relocate": [], "duplicates_to_delete": []}
        )
        pair_metrics: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
            lambda: {
                "backups_to_copy_count": 0,
                "backups_to_relocate_count": 0,
                "duplicates_to_delete_count": 0,
                "important_to_assign_count": 0,
            }
        )
        important_candidates: List[Tuple[MediaItem, WindowsVolume]] = []
        unbacked_items: List[MediaItem] = []
        unbacked_pair_map: Dict[str, str] = {}
        unbacked_pair_keys_str: Dict[str, str] = {}
        unbacked_count = 0
        unbacked_count_by_vol: Dict[int, int] = defaultdict(int)
        cross_dup_by_pair: Dict[str, int] = defaultdict(int)
        same_vol_dup_by_vol: Dict[int, int] = defaultdict(int)
        redundant_duplicates_count = 0
        single_device_warning = len(volumes) < 2

        for fhash, cluster in by_hash.items():
            non_backup_by_vol: Dict[int, List[MediaItem]] = defaultdict(list)
            for it in cluster:
                if not it.is_backup_copy:
                    non_backup_by_vol[it.volume_id].append(it)

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

            other_vol_copies = [x for x in cluster if x.volume_id != best_orig.volume_id and x.id != best_orig.id]
            chosen_backup: Optional[MediaItem] = None
            if other_vol_copies:
                other_vol_copies.sort(
                    key=lambda x: (not x.is_backup_copy, not volumes[x.volume_id].is_online if x.volume_id in volumes else True, x.id)
                )
                chosen_backup = other_vol_copies[0]

            # Compute pair_metrics for all possible pairs
            for (va, vb) in all_possible_pairs:
                if chosen_backup is None:
                    if best_orig.volume_id == va or best_orig.volume_id == vb:
                        pair_metrics[(va, vb)]["backups_to_copy_count"] += 1
                else:
                    pair_k = (min(best_orig.volume_id, chosen_backup.volume_id), max(best_orig.volume_id, chosen_backup.volume_id))
                    if (va, vb) == pair_k and not chosen_backup.is_backup_copy:
                        pair_metrics[(va, vb)]["backups_to_relocate_count"] += 1

                for item in cluster:
                    if item.id == best_orig.id:
                        continue
                    if chosen_backup and item.id == chosen_backup.id:
                        continue
                    if item.volume_id != best_orig.volume_id:
                        p_k = (min(best_orig.volume_id, item.volume_id), max(best_orig.volume_id, item.volume_id))
                        if (va, vb) == p_k:
                            pair_metrics[(va, vb)]["duplicates_to_delete_count"] += 1
                    else:
                        if best_orig.volume_id in (va, vb):
                            pair_metrics[(va, vb)]["duplicates_to_delete_count"] += 1

            target_vol: Optional[WindowsVolume] = None
            if chosen_backup is None:
                unbacked_count += 1
                unbacked_items.append(best_orig)
                unbacked_count_by_vol[best_orig.volume_id] += 1
                eligible_pks = [
                    f"{min(best_orig.volume_id, vo)}:{max(best_orig.volume_id, vo)}"
                    for vo in volumes if vo != best_orig.volume_id
                ]
                pks_str = ",".join(eligible_pks)
                unbacked_pair_map[best_orig.id] = pks_str
                unbacked_pair_keys_str[best_orig.id] = pks_str

                target_vol = pick_backup_volume(best_orig.volume_id)
                if target_vol:
                    pair_key = (min(best_orig.volume_id, target_vol.id), max(best_orig.volume_id, target_vol.id))
                    dst_rel = f"{self.BACKUP_DIR_NAME}/{best_orig.effective_date.strftime('%Y/%Y-%m')}/{best_orig.file_name}"
                    required_bytes_by_vol[target_vol.id] += int(best_orig.file_size_bytes or 0)
                    pair_tasks[pair_key]["backups_to_copy"].append(
                        {
                            "orig_item_id": best_orig.id,
                            "src_vol_id": best_orig.volume_id,
                            "dst_vol_id": target_vol.id,
                            "file_size": best_orig.file_size_bytes,
                            "file_name": best_orig.file_name,
                            "src_relative_path": best_orig.relative_path,
                            "dst_relative_path": dst_rel,
                            "file_hash": best_orig.file_hash,
                        }
                    )
            else:
                pair_key = (min(best_orig.volume_id, chosen_backup.volume_id), max(best_orig.volume_id, chosen_backup.volume_id))
                if (selected_pairs_set is None or pair_key in selected_pairs_set) and not chosen_backup.is_backup_copy:
                    dst_rel = f"{self.BACKUP_DIR_NAME}/{chosen_backup.effective_date.strftime('%Y/%Y-%m')}/{chosen_backup.file_name}"
                    pair_tasks[pair_key]["backups_to_relocate"].append(
                        {
                            "backup_item_id": chosen_backup.id,
                            "vol_id": chosen_backup.volume_id,
                            "file_name": chosen_backup.file_name,
                            "file_size": chosen_backup.file_size_bytes,
                            "src_relative_path": chosen_backup.relative_path,
                            "dst_relative_path": dst_rel,
                        }
                    )

            for item in cluster:
                if item.id == best_orig.id:
                    continue
                if chosen_backup and item.id == chosen_backup.id:
                    continue
                redundant_duplicates_count += 1
                ref_vol_id = best_orig.volume_id
                if item.volume_id != ref_vol_id:
                    pair_key = (min(ref_vol_id, item.volume_id), max(ref_vol_id, item.volume_id))
                    cross_dup_by_pair[f"{pair_key[0]}:{pair_key[1]}"] += 1
                else:
                    same_vol_dup_by_vol[item.volume_id] += 1
                    partner_vid = None
                    if target_vol and target_vol.id != ref_vol_id:
                        partner_vid = target_vol.id
                    elif chosen_backup and chosen_backup.volume_id != ref_vol_id:
                        partner_vid = chosen_backup.volume_id
                    else:
                        cand_list = selected_pairs_set if selected_pairs_set is not None else all_possible_pairs
                        for cand_p in cand_list:
                            if ref_vol_id in cand_p:
                                partner_vid = cand_p[1] if cand_p[0] == ref_vol_id else cand_p[0]
                                break
                    if partner_vid is not None:
                        pair_key = (min(ref_vol_id, partner_vid), max(ref_vol_id, partner_vid))
                    else:
                        pair_key = (ref_vol_id, item.volume_id)

                if selected_pairs_set is None or pair_key in selected_pairs_set:
                    reclaimed_bytes_by_vol[item.volume_id] += int(item.file_size_bytes or 0)
                    pair_tasks[pair_key]["duplicates_to_delete"].append(
                        {
                            "dup_item_id": item.id,
                            "dup_vol_id": item.volume_id,
                            "ref_item_id": best_orig.id,
                            "ref_vol_id": best_orig.volume_id,
                            "backup_item_id": chosen_backup.id if chosen_backup else None,
                            "file_name": item.file_name,
                            "file_size": item.file_size_bytes,
                            "dup_relative_path": item.relative_path,
                            "ref_relative_path": best_orig.relative_path,
                        }
                    )

        important_count_by_vol: Dict[int, int] = defaultdict(int)
        for orig_item, orig_vol in important_candidates:
            important_count_by_vol[orig_vol.id] += 1

        for (va, vb) in all_possible_pairs:
            pair_metrics[(va, vb)]["important_to_assign_count"] = (
                important_count_by_vol[va] + important_count_by_vol[vb]
            )

        important_rows: List[Dict[str, Any]] = []
        for orig_item, orig_vol in important_candidates:
            eligible_pks = [
                f"{min(orig_vol.id, vo)}:{max(orig_vol.id, vo)}"
                for vo in volumes if vo != orig_vol.id
            ]
            important_rows.append(
                {
                    "item_id": orig_item.id,
                    "file_name": orig_item.file_name,
                    "file_size": int(orig_item.file_size_bytes or 0),
                    "vol_id": orig_vol.id,
                    "vol_label": orig_vol.volume_label,
                    "vol_letter": orig_vol.last_known_drive_letter,
                    "relative_path": orig_item.relative_path,
                    "target_album": f"Ważne - {orig_vol.volume_label} / {orig_item.effective_date.strftime('%Y / %Y-%m')}",
                    "pair_keys": eligible_pks,
                    "pair_keys_str": ",".join(eligible_pks),
                }
            )

        volume_space_summary: Dict[int, Dict[str, Any]] = {}
        for vid, v in volumes.items():
            known_free = int(v.free_space_bytes or 0)
            known_total = int(v.total_space_bytes or 0)
            has_known = known_total > 0 or known_free > 0
            reclaimed = reclaimed_bytes_by_vol.get(vid, 0)
            required = required_bytes_by_vol.get(vid, 0)
            projected = known_free + reclaimed - required
            volume_space_summary[vid] = {
                "volume": v,
                "known_free_bytes": known_free,
                "known_total_bytes": known_total,
                "has_known_capacity": has_known,
                "reclaimed_bytes": reclaimed,
                "required_bytes": required,
                "projected_free_bytes": projected,
                "space_ok": (not has_known) or (projected >= self.MIN_FREE_SPACE_BUFFER_BYTES),
            }

        ordered_pairs = []
        for (va_id, vb_id) in all_possible_pairs:
            va = volumes.get(va_id)
            vb = volumes.get(vb_id)
            both_online = bool(va and vb and va.is_online and vb.is_online)
            one_online = bool((va and va.is_online) or (vb and vb.is_online))
            tasks = pair_tasks[(va_id, vb_id)]
            metrics = pair_metrics[(va_id, vb_id)]
            total_ops = (
                len(tasks["backups_to_copy"])
                + len(tasks["backups_to_relocate"])
                + len(tasks["duplicates_to_delete"])
            )
            ordered_pairs.append(
                {
                    "pair_key": (va_id, vb_id),
                    "pair_key_str": f"{va_id}:{vb_id}",
                    "vol_a": va,
                    "vol_b": vb,
                    "both_online": both_online,
                    "one_online": one_online,
                    "tasks": tasks,
                    "total_ops": total_ops,
                    "metrics": metrics,
                }
            )

        ordered_pairs.sort(key=lambda p: (not p["both_online"], not p["one_online"], p["pair_key"]))

        duplicate_rows: List[Dict[str, Any]] = []
        for fhash, cluster in by_hash.items():
            if len(cluster) <= 1:
                continue
            primary = next((x for x in cluster if x.is_primary_duplicate), cluster[0])
            other_vol_copies = [x for x in cluster if x.volume_id != primary.volume_id and x.id != primary.id]
            chosen_backup = other_vol_copies[0] if other_vol_copies else None

            for item in cluster:
                if item.id == primary.id:
                    continue
                if chosen_backup and item.id == chosen_backup.id:
                    continue
                dup_v = volumes.get(item.volume_id)
                ref_v = volumes.get(primary.volume_id)
                if item.volume_id != primary.volume_id:
                    p_keys = [f"{min(primary.volume_id, item.volume_id)}:{max(primary.volume_id, item.volume_id)}"]
                    p_key_str = p_keys[0]
                else:
                    p_keys = [f"{min(item.volume_id, vo)}:{max(item.volume_id, vo)}" for vo in volumes if vo != item.volume_id]
                    p_key_str = p_keys[0] if p_keys else f"{item.volume_id}:0"

                duplicate_rows.append(
                    {
                        "pair_key_str": p_key_str,
                        "pair_keys": p_keys,
                        "pair_keys_str": ",".join(p_keys),
                        "dup_item_id": item.id,
                        "file_name": item.file_name,
                        "file_size": int(item.file_size_bytes or 0),
                        "dup_vol_id": item.volume_id,
                        "dup_vol_label": dup_v.volume_label if dup_v else "?",
                        "dup_vol_letter": dup_v.last_known_drive_letter if dup_v else "?",
                        "dup_relative_path": item.relative_path,
                        "ref_vol_id": primary.volume_id,
                        "ref_vol_label": ref_v.volume_label if ref_v else "?",
                        "ref_vol_letter": ref_v.last_known_drive_letter if ref_v else "?",
                        "ref_relative_path": primary.relative_path,
                    }
                )

        return {
            "volumes": volumes,
            "volume_space_summary": volume_space_summary,
            "important_candidates": important_candidates,
            "important_rows": important_rows,
            "important_count_by_vol": dict(important_count_by_vol),
            "ordered_pairs": ordered_pairs,
            "duplicate_rows": duplicate_rows,
            "unbacked_items": unbacked_items,
            "unbacked_pair_map": unbacked_pair_map,
            "unbacked_pair_keys_str": unbacked_pair_keys_str,
            "unbacked_count": unbacked_count,
            "redundant_duplicates_count": redundant_duplicates_count,
            "single_device_warning": single_device_warning,
            "unbacked_count_by_vol": dict(unbacked_count_by_vol),
            "cross_dup_by_pair": dict(cross_dup_by_pair),
            "same_vol_dup_by_vol": dict(same_vol_dup_by_vol),
        }

    def execute_backup_and_deduplication(
        self,
        preferred_target_volume_id: Optional[int] = None,
        job_id: Optional[int] = None,
        operation_mode: str = "both",
        selected_pair_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the automated backup and 100%-verified deduplication workflow:
        - Supports operation_mode: 'both', 'backup_only', or 'dedup_only'.
        - Supports filtering execution strictly to user-selected volume pairs (selected_pair_keys).
        - Processes currently ONLINE selected volume pairs {V_a, V_b} bidirectionally (A->B and B->A).
        - Verifies 100% identity (size, SHA-256, BLAKE2b, MD5, byte-by-byte stream, image decode) before deleting any duplicate.
        """
        from app.operation_log import ProcessManager

        norm_mode = operation_mode if operation_mode in ("both", "backup_only", "dedup_only") else "both"
        do_backup = norm_mode in ("both", "backup_only")
        do_dedup = norm_mode in ("both", "dedup_only")
        selected_pairs_set = self.parse_pair_keys(selected_pair_keys)

        self.backup_mgr.create_backup("pre_auto_backup_dedup")
        plan = self.build_backup_and_dedup_plan(
            preferred_target_volume_id=preferred_target_volume_id,
            selected_pair_keys=selected_pair_keys,
        )
        volumes: Dict[int, WindowsVolume] = plan["volumes"]

        # Step 1: Assign Originals with same-device duplicates to 'Ważne - <volume_label>' -> YYYY -> YYYY-MM (Rule 2b, 2d)
        important_assigned = 0
        active_vols = {v for p in selected_pairs_set for v in p} if selected_pairs_set is not None else None
        for orig_item, orig_vol in plan["important_candidates"]:
            if active_vols is not None and orig_vol.id not in active_vols:
                continue
            self.ensure_important_album_hierarchy(orig_item, orig_vol.volume_label)
            important_assigned += 1
        if important_assigned > 0:
            self.db.commit()

        backups_created = 0
        backups_relocated = 0
        duplicates_deleted = 0
        verification_blocks = 0
        skipped_unselected_pairs = 0
        space_warnings: List[str] = []
        completed_pairs: List[str] = []
        pending_offline_pairs: List[Dict[str, Any]] = []

        for pair_entry in plan["ordered_pairs"]:
            if job_id and ProcessManager.is_stop_requested(job_id):
                break

            if selected_pairs_set is not None and pair_entry["pair_key"] not in selected_pairs_set:
                skipped_unselected_pairs += 1
                continue

            va: Optional[WindowsVolume] = pair_entry["vol_a"]
            vb: Optional[WindowsVolume] = pair_entry["vol_b"]
            tasks = pair_entry["tasks"]

            if not (va and vb and va.is_online and vb.is_online):
                pending_offline_pairs.append(pair_entry)
                continue

            # Map newly created backup items by original item ID so same-pass deduplication knows backup is ready
            created_backup_by_orig: Dict[str, MediaItem] = {}

            # 2A: Bidirectional backup copying (Va -> Vb\backup and Vb -> Va\backup)
            if do_backup:
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
                    backup_dir = dst_root / self.BACKUP_DIR_NAME / sub_month
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = backup_dir / orig_item.file_name
                    c = 1
                    while dest_path.exists():
                        dest_path = backup_dir / f"{src_path.stem}_{c}{src_path.suffix}"
                        c += 1

                    shutil.copy2(str(src_path), str(dest_path))

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
                    target_bkp_dir = vol_root / self.BACKUP_DIR_NAME / sub_month
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
            if do_dedup:
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

                    bkp_id = dup_task.get("backup_item_id")
                    bkp_obj = self.db.get(MediaItem, bkp_id) if bkp_id else created_backup_by_orig.get(ref_item.id)
                    if len(volumes) >= 2 and bkp_obj is None:
                        continue

                    if not dup_path.exists():
                        if ref_path.exists() and ref_path.stat().st_size == ref_item.file_size_bytes:
                            self.db.delete(dup_item)
                            duplicates_deleted += 1
                        continue

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

                    if bkp_obj:
                        bkp_vol = volumes.get(bkp_obj.volume_id)
                        if bkp_vol and bkp_vol.is_online:
                            bkp_disk_p = Path(bkp_vol.last_known_drive_letter) / bkp_obj.relative_path
                            if not bkp_disk_p.exists() or bkp_disk_p.resolve() == dup_path.resolve():
                                verification_blocks += 1
                                continue

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
        remaining_unbacked = len(self.detect_files_without_backup(recluster=False))

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

        mode_labels = {
            "both": "Backup + Deduplikacja (oba)",
            "backup_only": "Tylko Backup",
            "dedup_only": "Tylko Deduplikacja",
        }
        summary_msg = (
            f"Tryb: {mode_labels[norm_mode]} | "
            f"Backup utworzony: {backups_created} | Przeniesiono do '{self.BACKUP_DIR_NAME}/': {backups_relocated} | "
            f"Usunięto duplikatów (100% weryfikacji): {duplicates_deleted} | "
            f"Dodano do albumu 'Ważne': {important_assigned}"
        )
        if skipped_unselected_pairs > 0:
            summary_msg += f" | Pominięto niewybrane pary dysków: {skipped_unselected_pairs}"

        OperationLogger.log(
            "BACKUP_DEDUP",
            summary_msg,
            status="OK" if not pending_offline_pairs else "PARTIAL",
            details={
                "operation_mode": norm_mode,
                "backups_created": backups_created,
                "backups_relocated": backups_relocated,
                "duplicates_deleted": duplicates_deleted,
                "important_assigned": important_assigned,
                "verification_blocks": verification_blocks,
                "completed_pairs": completed_pairs,
                "pending_pairs": len(pending_offline_pairs),
                "skipped_unselected_pairs": skipped_unselected_pairs,
            },
            db=self.db,
        )

        return {
            "operation_mode": norm_mode,
            "backups_created": backups_created,
            "backups_relocated": backups_relocated,
            "duplicates_deleted": duplicates_deleted,
            "important_assigned": important_assigned,
            "verification_blocks": verification_blocks,
            "space_warnings": space_warnings,
            "completed_pairs": completed_pairs,
            "pending_offline_pairs_count": len(pending_offline_pairs),
            "skipped_unselected_pairs": skipped_unselected_pairs,
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

    def generate_offline_procedure_report(
        self,
        preferred_target_volume_id: Optional[int] = None,
        operation_mode: str = "both",
        selected_pair_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Builds the offline backup, deduplication, and pending move plan using persisted disk capacities
        and saves a human-readable Polish text report on the local disk before any physical changes occur.
        """
        norm_mode = operation_mode if operation_mode in ("both", "backup_only", "dedup_only") else "both"
        do_backup = norm_mode in ("both", "backup_only")
        do_dedup = norm_mode in ("both", "dedup_only")
        selected_pairs_set = self.parse_pair_keys(selected_pair_keys)

        plan = self.build_backup_and_dedup_plan(
            preferred_target_volume_id=preferred_target_volume_id,
            selected_pair_keys=selected_pair_keys,
        )
        pending_summary = self.get_pending_moves_summary()
        volumes: Dict[int, WindowsVolume] = plan["volumes"]
        vol_space_summary: Dict[int, Dict[str, Any]] = plan["volume_space_summary"]

        filtered_pairs = [
            p for p in plan["ordered_pairs"]
            if selected_pairs_set is None or p["pair_key"] in selected_pairs_set
        ]

        mode_names = {
            "both": "Backup + Deduplikacja (oba)",
            "backup_only": "Tylko Backup",
            "dedup_only": "Tylko Deduplikacja",
        }
        now_dt = datetime.datetime.now()
        ts_str = now_dt.strftime("%Y%m%d_%H%M%S")
        lines: List[str] = [
            "=" * 84,
            "DOMOWE ARCHIWUM FOTO — RAPORT PROCEDURY BACKUPU, DEDUPLIKACJI I ZMIAN (TRYB OFFLINE)",
            f"Wygenerowano: {now_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Wybrany tryb operacji: {mode_names[norm_mode]} | Wybrane pary dysków: {len(filtered_pairs)} z {len(plan['ordered_pairs'])}",
            "Przeznaczenie: Weryfikacja poprawności planowanych operacji przed podjęciem fizycznych zmian na dyskach.",
            "=" * 84,
            "",
            "1. STAN DYSKÓW I BILANS MIEJSCA (ZAPISANE W BAZIE DLA TRYBU OFFLINE):",
        ]

        if not volumes:
            lines.append("   Brak zarejestrowanych urządzeń w bazie.")
        else:
            for vid in sorted(volumes.keys()):
                v = volumes[vid]
                sp = vol_space_summary[vid]
                status_lbl = "ONLINE" if v.is_online else "OFFLINE (Dane z bazy SQLite)"
                if sp["has_known_capacity"]:
                    free_mb = sp["known_free_bytes"] / (1024 * 1024)
                    total_mb = sp["known_total_bytes"] / (1024 * 1024)
                    recl_mb = sp["reclaimed_bytes"] / (1024 * 1024)
                    req_mb = sp["required_bytes"] / (1024 * 1024)
                    proj_mb = sp["projected_free_bytes"] / (1024 * 1024)
                    cap_note = "OK" if sp["space_ok"] else "UWAGA: NIEWYSTARCZAJĄCE MIEJSCE!"
                    lines.append(
                        f"   - [{status_lbl}] {v.volume_label} ({v.last_known_drive_letter}, VSN: {v.volume_serial_number})"
                    )
                    lines.append(
                        f"     Pojemność całkowita: {total_mb:.2f} MB | Wolne w bazie: {free_mb:.2f} MB | "
                        f"Do odzyskania z duplikatów: +{recl_mb:.2f} MB | Wymagane na backup: -{req_mb:.2f} MB | "
                        f"Prognoza po operacji: {proj_mb:.2f} MB [{cap_note}]"
                    )
                else:
                    lines.append(
                        f"   - [{status_lbl}] {v.volume_label} ({v.last_known_drive_letter}, VSN: {v.volume_serial_number}) — brak zapisanej pojemności"
                    )

        lines.extend(["", "2. PLANOWANE OPERACJE BACKUPU I DEDUPLIKACJI (WG WYBRANYCH PAR URZĄDZEŃ A <-> B):"])
        if not filtered_pairs:
            lines.append("   Brak operacji dla wybranych par urządzeń i trybu.")
        else:
            for idx, pair in enumerate(filtered_pairs, start=1):
                va = pair["vol_a"]
                vb = pair["vol_b"]
                va_lbl = f"{va.volume_label} ({va.last_known_drive_letter})" if va else "?"
                vb_lbl = f"{vb.volume_label} ({vb.last_known_drive_letter})" if vb else "?"
                pair_state = "GOTOWA (OBA ONLINE)" if pair["both_online"] else "OFFLINE / WYMAGA PODŁĄCZENIA"
                lines.append(f"   --- Para #{idx}: {va_lbl} <-> {vb_lbl} [{pair_state}] ---")
                tasks = pair["tasks"]

                if do_backup:
                    for ct in tasks["backups_to_copy"]:
                        sv = volumes.get(ct["src_vol_id"])
                        dv = volumes.get(ct["dst_vol_id"])
                        sz_mb = (ct["file_size"] or 0) / (1024 * 1024)
                        lines.append(
                            f"     [KOPIOWANIE BACKUPU] {(sv.volume_label if sv else '?')}:{ct.get('src_relative_path', ct['file_name'])} "
                            f"-> {(dv.volume_label if dv else '?')}:{ct.get('dst_relative_path', f'{self.BACKUP_DIR_NAME}/')} ({sz_mb:.2f} MB)"
                        )

                    for rt in tasks["backups_to_relocate"]:
                        rv = volumes.get(rt["vol_id"])
                        lines.append(
                            f"     [PRZENIESIENIE DO {self.BACKUP_DIR_NAME}/] {(rv.volume_label if rv else '?')}:{rt.get('src_relative_path', '')} "
                            f"-> {(rv.volume_label if rv else '?')}:{rt.get('dst_relative_path', f'{self.BACKUP_DIR_NAME}/')}"
                        )

                if do_dedup:
                    for dt in tasks["duplicates_to_delete"]:
                        dv = volumes.get(dt["dup_vol_id"])
                        rv = volumes.get(dt["ref_vol_id"])
                        sz_mb = (dt.get("file_size") or 0) / (1024 * 1024)
                        lines.append(
                            f"     [USUNIĘCIE DUPLIKATU] {(dv.volume_label if dv else '?')}:{dt.get('dup_relative_path', dt.get('file_name', ''))} "
                            f"({sz_mb:.2f} MB) | Zachowany oryginał: {(rv.volume_label if rv else '?')}:{dt.get('ref_relative_path', '')}"
                        )

        lines.extend(["", "3. OCZEKUJĄCE PRZENIESIENIA PLIKÓW Z LISTY TODO (TRYB OFFLINE):"])
        if not pending_summary["todos"]:
            lines.append("   Lista TODO jest pusta — brak oczekujących przeniesień plików.")
        else:
            for td in pending_summary["todos"]:
                sv_lbl = td.source_volume.volume_label if td.source_volume else "?"
                tv_lbl = td.target_volume.volume_label if td.target_volume else "?"
                lines.append(
                    f"   - [PRZENIESIENIE TODO #{td.id}] {sv_lbl}:{td.source_relative_path} -> {tv_lbl}:{td.target_relative_path}"
                )

        lines.extend(["", "4. ORYGINAŁY Z DUPLIKATAMI NA TYM SAMYM DYSKU (DO ALBUMU 'WAŻNE'):"])
        active_vols = {v for p in selected_pairs_set for v in p} if selected_pairs_set is not None else None
        imp_candidates = [
            (it, v) for it, v in plan["important_candidates"]
            if active_vols is None or v.id in active_vols
        ]
        if not imp_candidates:
            lines.append("   Brak nowych kandydatów.")
        else:
            for orig_it, orig_v in imp_candidates:
                lines.append(
                    f"   - {orig_v.volume_label}:{orig_it.relative_path} -> Album: Ważne - {orig_v.volume_label} / {orig_it.effective_date.strftime('%Y / %Y-%m')}"
                )

        lines.extend(["", "=" * 84])
        report_text = "\n".join(lines)

        reports_dir = db_module.REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"raport_procedura_offline_{ts_str}.txt"
        latest_file = reports_dir / "raport_procedura_offline_ostatni.txt"
        report_file.write_text(report_text, encoding="utf-8")
        latest_file.write_text(report_text, encoding="utf-8")

        OperationLogger.log(
            "OFFLINE_REPORT",
            f"Wygenerowano raport procedury offline: {report_file.name}",
            status="OK",
            details={
                "report_path": str(report_file),
                "unbacked_count": plan["unbacked_count"],
                "redundant_duplicates_count": plan["redundant_duplicates_count"],
                "pending_todos_count": pending_summary["total_pending"],
            },
            db=self.db,
        )

        return {
            "report_path": str(report_file),
            "latest_path": str(latest_file),
            "report_text": report_text,
            "plan": plan,
            "pending_summary": pending_summary,
        }

    def recalculate_date_anomalies(self) -> Dict[str, int]:
        """
        Re-evaluates date discrepancies across folder name, filename, EXIF, and filesystem timestamps
        for all active records, assigning the most probable date and flagging copy-modified timestamps.
        """
        items = list(self.db.scalars(select(MediaItem).where(MediaItem.is_trashed == False)).all())
        evaluated = 0
        anomalies_found = 0
        dates_updated = 0

        for it in items:
            evaluated += 1
            fn_dt = it.filename_date or MetadataDateResolver.parse_filename(it.file_name)
            it.filename_date = fn_dt
            folder_info = MetadataDateResolver.parse_folder_date(Path(it.relative_path), rel_path=it.relative_path)
            effective, anomaly, reason = MetadataDateResolver.evaluate_date_sources(
                it.exif_date, fn_dt, it.fs_ctime, it.fs_mtime, folder_info=folder_info
            )
            if it.effective_date != effective:
                it.effective_date = effective
                dates_updated += 1
            it.has_date_anomaly = anomaly
            it.date_anomaly_reason = reason
            if anomaly:
                anomalies_found += 1

        self.db.commit()
        OperationLogger.log(
            "DATE_ANOMALY_RECALC",
            f"Przeliczono daty i anomalie dla {evaluated} plików (anomalii: {anomalies_found}, skorygowanych dat: {dates_updated}).",
            status="OK",
            db=self.db,
        )
        return {
            "evaluated": evaluated,
            "anomalies_found": anomalies_found,
            "dates_updated": dates_updated,
        }
