from collections import defaultdict
import datetime
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from sqlalchemy import select, text, update, delete, func, or_
from sqlalchemy.orm import Session, joinedload

from app.backup import BackupManager
import app.database as db_module
from app.models import MediaItem, WindowsVolume, MediaGroup, DeletedDuplicateHistory
from app.operation_log import OperationLogger
from app.scanner import FastHasher, MetadataDateResolver, WindowsVolumeInspector

logger = logging.getLogger("StorageOps")


def _safe_unlink(path: Path) -> None:
    """Deletes a file on disk, clearing Windows Read-Only attribute if needed."""
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


def _safe_move(src: Path, dst: Path) -> None:
    """Moves a file on disk, clearing Windows Read-Only attribute if needed."""
    try:
        shutil.move(str(src), str(dst))
    except PermissionError:
        os.chmod(src, stat.S_IWRITE)
        shutil.move(str(src), str(dst))


class StorageAutomationService:
    """Handles duplicate clustering, 100%-verified backup & deduplication, album management, and offline move queues."""

    MIN_FREE_SPACE_BUFFER_BYTES = 20 * 1024 * 1024  # 20 MB safety reserve on target drives
    BACKUP_DIR_NAME = "daf_backup"
    ORIGINAL_DIR_NAME = "daf_oryginal"
    ALBUMS_EXPORT_DIR_NAME = "daf_albums"

    def __init__(self, db: Session):
        self.db = db
        self.backup_mgr = BackupManager()

    def verify_and_resolve_online_volume_root(
        self,
        vol: Optional[WindowsVolume],
        connected_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> Optional[Path]:
        """
        Verifies via live OS VSN enumeration that `vol` is truly mounted right now and resolves
        its current drive root. Prevents accidental disk operations if Windows reassigned drive letters.
        """
        if vol is None:
            return None
        if connected_map is None:
            connected_map = WindowsVolumeInspector.get_connected_vsn_map()

        vsn = (vol.volume_serial_number or "").strip()
        if vsn in connected_map:
            mount_root, _ = connected_map[vsn]
            if vol.last_known_drive_letter != mount_root or not vol.is_online:
                vol.last_known_drive_letter = mount_root
                vol.is_online = True
            return Path(mount_root)

        dl = (vol.last_known_drive_letter or "").strip()
        norm_dl = dl.rstrip("\\/").upper()
        for other_vsn, (other_root, _) in connected_map.items():
            if other_root.rstrip("\\/").upper() == norm_dl and other_vsn != vsn:
                vol.is_online = False
                logger.warning(
                    "VSN mismatch on %s: expected %s (%s), found %s. Disk operation blocked.",
                    dl, vsn, vol.volume_label, other_vsn,
                )
                return None

        is_drive_root = bool(re.fullmatch(r"[A-Za-z]:\\?", dl))
        is_hw_vsn = bool(re.fullmatch(r"[0-9A-F]{4}-[0-9A-F]{4}", vsn))
        if connected_map and (is_drive_root or is_hw_vsn):
            vol.is_online = False
            return None

        if not vol.is_online or not dl or not Path(dl).exists():
            return None
        return Path(dl)

    def refresh_volumes_online_status(self) -> List[WindowsVolume]:
        """Updates online status, drive letters, and persisted free/total disk capacity for all tracked volumes."""
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        volumes = list(self.db.scalars(select(WindowsVolume)).all())
        now = datetime.datetime.utcnow()
        dirty = False
        for vol in volumes:
            prev_online = vol.is_online
            prev_letter = vol.last_known_drive_letter
            prev_total = int(vol.total_space_bytes or 0)
            prev_free = int(vol.free_space_bytes or 0)

            if vol.volume_serial_number in connected_map:
                root_letter, _ = connected_map[vol.volume_serial_number]
                new_letter = root_letter
                new_online = True
            else:
                dl = (vol.last_known_drive_letter or "").strip()
                is_drive_root = bool(re.fullmatch(r"[A-Za-z]:\\?", dl))
                is_hw_vsn = bool(re.fullmatch(r"[0-9A-F]{4}-[0-9A-F]{4}", vol.volume_serial_number or ""))
                if connected_map and (is_drive_root or is_hw_vsn):
                    # Real drive enumeration found no match for this VSN, so the drive is genuinely disconnected
                    # even if another USB device reused the same Windows drive letter.
                    new_online = False
                else:
                    new_online = bool(dl and Path(dl).exists())
                new_letter = vol.last_known_drive_letter

            new_total, new_free = prev_total, prev_free
            if new_online:
                try:
                    usage = shutil.disk_usage(new_letter)
                    new_total = int(usage.total)
                    new_free = int(usage.free)
                except Exception as exc:
                    logger.debug("disk_usage update skipped for %s: %s", new_letter, exc)

            stale_seen = (
                new_online
                and (
                    vol.last_seen_at is None
                    or abs((now - vol.last_seen_at.replace(tzinfo=None)).total_seconds()) > 60
                )
            )
            changed = (
                (new_online != prev_online)
                or (new_letter != prev_letter)
                or (new_total != prev_total)
                or (abs(new_free - prev_free) > 1024 * 1024)
                or stale_seen
            )
            if changed:
                vol.is_online = new_online
                vol.last_known_drive_letter = new_letter
                vol.total_space_bytes = new_total
                vol.free_space_bytes = new_free
                if new_online:
                    vol.last_seen_at = now
                dirty = True

        if dirty:
            try:
                self.db.commit()
            except Exception as exc:
                logger.debug("refresh_volumes_online_status commit deferred (DB busy): %s", exc)
                self.db.rollback()
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
                    new_total = int(u.total)
                    new_free = int(u.free)
                    prev_total = int(v.total_space_bytes or 0)
                    prev_free = int(v.free_space_bytes or 0)
                    if new_total != prev_total or abs(new_free - prev_free) > 1024 * 1024:
                        v.total_space_bytes = new_total
                        v.free_space_bytes = new_free
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
            try:
                self.db.commit()
            except Exception as exc:
                logger.debug("get_volume_space_map commit deferred (DB busy): %s", exc)
                self.db.rollback()
        return vol_space

    def flag_duplicate_clusters(self) -> int:
        """
        Marks primary vs backup vs duplicate roles across all non-trashed items in SQLite.
        Cluster rules:
        - Copies in 'daf_backup/' (or legacy 'backup/') are flagged is_backup_copy = 1.
        - 1st item in cluster: duplicate_role = 'original', is_primary_duplicate = 1.
          Prefers non-backup copies on a volume that already has multiple non-backup copies
          (eligible for album 'Ważne'), breaking ties by oldest effective_date, relative_path, id.
        - 2nd item in cluster: duplicate_role = 'backup_green' (on different device than Original,
          preferring copies already in daf_backup/) OR 'backup_orange' (on same device as Original).
        - 3rd+ item in cluster: duplicate_role = 'duplicate', is_primary_duplicate = 0.
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
                    volume_id,
                    is_backup_copy,
                    effective_date,
                    relative_path,
                    SUM(CASE WHEN is_backup_copy = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY file_hash, volume_id) AS same_vol_cnt
                FROM media_items
                WHERE is_trashed = 0
            ),
            orig_ranked AS (
                SELECT
                    id,
                    file_hash,
                    volume_id,
                    is_backup_copy,
                    effective_date,
                    relative_path,
                    ROW_NUMBER() OVER (
                        PARTITION BY file_hash
                        ORDER BY is_backup_copy ASC, same_vol_cnt DESC, effective_date ASC, relative_path ASC, id ASC
                    ) AS orig_rn,
                    FIRST_VALUE(volume_id) OVER (
                        PARTITION BY file_hash
                        ORDER BY is_backup_copy ASC, same_vol_cnt DESC, effective_date ASC, relative_path ASC, id ASC
                    ) AS orig_vol_id
                FROM vol_counts
            ),
            role_ranked AS (
                SELECT
                    id,
                    volume_id,
                    orig_vol_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY file_hash
                        ORDER BY
                            (CASE WHEN orig_rn = 1 THEN 0 WHEN volume_id != orig_vol_id THEN 1 ELSE 2 END) ASC,
                            is_backup_copy DESC,
                            effective_date ASC,
                            relative_path ASC,
                            id ASC
                    ) AS final_rn
                FROM orig_ranked
            ),
            computed AS (
                SELECT
                    id,
                    CASE WHEN final_rn = 1 THEN 1 ELSE 0 END AS new_primary,
                    CASE
                        WHEN final_rn = 1 THEN 'original'
                        WHEN final_rn = 2 AND volume_id != orig_vol_id THEN 'backup_green'
                        WHEN final_rn = 2 THEN 'backup_orange'
                        ELSE 'duplicate'
                    END AS new_role
                FROM role_ranked
            )
            UPDATE media_items
            SET
                is_primary_duplicate = computed.new_primary,
                duplicate_role = computed.new_role
            FROM computed
            WHERE media_items.id = computed.id;
        """)
        self.db.execute(rank_sql)
        self.db.commit()
        self.db.expire_all()
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
        Ensures album 'Ważne - <vol_label>' -> '<YYYY>' -> '<MM>' -> '<DD>' exists
        and assigns item.group_id to the day subalbum (Requirement 2b).
        """
        root_name = f"Ważne - {vol_label}"
        yr_name = str(item.effective_date.year)
        mo_name = item.effective_date.strftime("%m")
        day_name = item.effective_date.strftime("%d")

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

        day_grp = self.db.scalar(
            select(MediaGroup).where(MediaGroup.name == day_name, MediaGroup.parent_id == mo_grp.id, MediaGroup.is_trashed == False)
        )
        if not day_grp:
            day_grp = MediaGroup(name=day_name, parent_id=mo_grp.id)
            self.db.add(day_grp)
            self.db.flush()

        item.group_id = day_grp.id
        return day_grp

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
        target_volume_ids: Optional[List[int]] = None,
        minimize_transfers: bool = True,
        include_dedup: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyzes the database to build the Backup plan and capacity projection:
        1) Computes exact cluster stats: missing + green_backups + orange_backups == total_unique.
        2) Selects default target backup devices with minimal originals and sufficient capacity.
        3) Plans:
           - Kopie lokalne (green BACKUP moved into daf_backup/YYYY/MM/DD on the same device),
           - Kopie zdalne (when minimize_transfers=False, consolidating backups onto minimal backup drives),
           - Nowe kopie (missing & orange backups copied from Original to target daf_backup/YYYY/MM/DD),
           - Do albumu 'Ważne - <device>' -> YYYY -> MM -> DD (Originals with same-device copies).
        """
        self.refresh_volumes_online_status()
        self.flag_duplicate_clusters()

        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        groups_by_id = {g.id: g for g in self.db.scalars(select(MediaGroup).where(MediaGroup.is_trashed == False)).all()}
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

        def is_in_important_album(item: MediaItem, vol_label: str) -> bool:
            if not item.group_id or item.group_id not in groups_by_id:
                return False
            parts: List[str] = []
            curr = groups_by_id.get(item.group_id)
            depth = 0
            while curr and depth < 6:
                parts.append(curr.name)
                curr = groups_by_id.get(curr.parent_id) if curr.parent_id else None
                depth += 1
            parts.reverse()
            expected = [
                f"Ważne - {vol_label}",
                str(item.effective_date.year),
                item.effective_date.strftime("%m"),
                item.effective_date.strftime("%d"),
            ]
            return parts == expected

        # Pre-pass: inspect every cluster for stats, originals per volume, and backup state
        total_unique_count = len(by_hash)
        missing_backup_count = 0
        green_backup_count = 0
        green_in_vault_count = 0
        green_outside_vault_count = 0
        orange_backup_count = 0
        orig_count_by_vol: Dict[int, int] = defaultdict(int)
        backup_count_by_vol: Dict[int, int] = defaultdict(int)
        backup_bytes_by_vol: Dict[int, int] = defaultdict(int)
        bkp_valid_in_vault_by_vol: Dict[int, int] = defaultdict(int)
        bkp_reloc_green_by_vol: Dict[int, int] = defaultdict(int)
        bkp_reloc_orange_by_vol: Dict[int, int] = defaultdict(int)
        needed_bytes_from_vol: Dict[int, int] = defaultdict(int)

        cluster_infos: List[Dict[str, Any]] = []
        for fhash, cluster in by_hash.items():
            best_orig = next((x for x in cluster if x.duplicate_role == "original"), None)
            if best_orig is None:
                best_orig = next((x for x in cluster if x.is_primary_duplicate), cluster[0])
            orig_count_by_vol[best_orig.volume_id] += 1

            same_vol_non_bkp_cnt = sum(
                1 for x in cluster if x.volume_id == best_orig.volume_id and not x.is_backup_copy
            )

            other_vol_copies = [x for x in cluster if x.volume_id != best_orig.volume_id and x.id != best_orig.id]
            same_vol_other_copies = [x for x in cluster if x.volume_id == best_orig.volume_id and x.id != best_orig.id]

            for x in cluster:
                if x.id == best_orig.id:
                    continue
                if x.is_backup_copy or x.duplicate_role in ("backup_green", "backup_orange"):
                    backup_count_by_vol[x.volume_id] += 1
                    backup_bytes_by_vol[x.volume_id] += int(x.file_size_bytes or 0)
                    if x.is_backup_copy and x.volume_id != best_orig.volume_id:
                        bkp_valid_in_vault_by_vol[x.volume_id] += 1
                    elif x.volume_id != best_orig.volume_id:
                        bkp_reloc_green_by_vol[x.volume_id] += 1
                    else:
                        bkp_reloc_orange_by_vol[x.volume_id] += 1

            if other_vol_copies:
                green_backup_count += 1
                if any(x.is_backup_copy for x in other_vol_copies):
                    green_in_vault_count += 1
                else:
                    green_outside_vault_count += 1
            elif same_vol_other_copies:
                orange_backup_count += 1
                needed_bytes_from_vol[best_orig.volume_id] += int(best_orig.file_size_bytes or 0)
            else:
                missing_backup_count += 1
                needed_bytes_from_vol[best_orig.volume_id] += int(best_orig.file_size_bytes or 0)

            cluster_infos.append(
                {
                    "hash": fhash,
                    "cluster": cluster,
                    "orig": best_orig,
                    "same_vol_non_bkp_cnt": same_vol_non_bkp_cnt,
                    "other_vol_copies": other_vol_copies,
                    "has_green": bool(other_vol_copies),
                }
            )

        # Default target backup drive selection: minimal originals & sufficient capacity
        ranked_target_vols = sorted(
            volumes.values(),
            key=lambda v: (
                orig_count_by_vol.get(v.id, 0),
                -int(v.free_space_bytes or 0),
                not v.is_online,
                v.id,
            ),
        )
        default_target_ids: List[int] = []
        if ranked_target_vols:
            if len(ranked_target_vols) == 1:
                default_target_ids = [ranked_target_vols[0].id]
            else:
                chosen_set: Set[int] = set()
                for cand_v in ranked_target_vols:
                    chosen_set.add(cand_v.id)
                    default_target_ids.append(cand_v.id)
                    # Required bytes from drives outside chosen_set
                    req_outside = sum(
                        needed_bytes_from_vol.get(vid, 0)
                        for vid in volumes if vid not in chosen_set
                    )
                    if not minimize_transfers:
                        req_outside += sum(
                            int(cinfo["orig"].file_size_bytes or 0)
                            for cinfo in cluster_infos
                            if cinfo["has_green"] and all(x.volume_id not in chosen_set for x in cinfo["other_vol_copies"])
                        )
                    known_chosen_cap = any(
                        (int(volumes[cid].total_space_bytes or 0) > 0 or int(volumes[cid].free_space_bytes or 0) > 0)
                        for cid in chosen_set
                    )
                    avail_chosen = sum(
                        max(0, int(volumes[cid].free_space_bytes or 0) - self.MIN_FREE_SPACE_BUFFER_BYTES)
                        for cid in chosen_set
                    )
                    # Stop adding more drives once chosen capacity suffices and (dedicated drive or covers external originals)
                    if (not known_chosen_cap or avail_chosen >= req_outside):
                        break

        if target_volume_ids is not None:
            effective_target_ids = [vid for vid in target_volume_ids if vid in volumes]
        elif preferred_target_volume_id and preferred_target_volume_id in volumes:
            effective_target_ids = [preferred_target_volume_id]
        else:
            effective_target_ids = list(default_target_ids)
        effective_target_set: Set[int] = set(effective_target_ids)

        freed_bytes_by_vol: Dict[int, int] = defaultdict(int)
        required_bytes_by_vol: Dict[int, int] = defaultdict(int)

        def vol_has_known_cap(vid: int) -> bool:
            v = volumes.get(vid)
            return bool(v and (int(v.total_space_bytes or 0) > 0 or int(v.free_space_bytes or 0) > 0))

        def vol_avail_free(vid: int) -> int:
            v = volumes.get(vid)
            if not v:
                return 0
            return int(v.free_space_bytes or 0) + freed_bytes_by_vol.get(vid, 0) - required_bytes_by_vol.get(vid, 0)

        def pick_backup_volume(src_vol_id: int, file_size: int, respect_pair_selection: bool = False) -> Optional[WindowsVolume]:
            candidates: List[WindowsVolume] = []
            for vid in effective_target_ids:
                if vid == src_vol_id:
                    continue
                if respect_pair_selection and selected_pairs_set is not None:
                    pk = (min(src_vol_id, vid), max(src_vol_id, vid))
                    if pk not in selected_pairs_set:
                        continue
                candidates.append(volumes[vid])

            if not candidates:
                return None

            if preferred_target_volume_id and any(c.id == preferred_target_volume_id for c in candidates):
                return volumes[preferred_target_volume_id]

            def fits(v: WindowsVolume) -> bool:
                if not vol_has_known_cap(v.id):
                    return True
                return (vol_avail_free(v.id) - file_size) >= self.MIN_FREE_SPACE_BUFFER_BYTES

            candidates.sort(
                key=lambda c: (
                    not fits(c),
                    orig_count_by_vol.get(c.id, 0),
                    not c.is_online,
                    -vol_avail_free(c.id),
                    c.id,
                )
            )
            return candidates[0]

        pair_tasks: Dict[Tuple[int, int], Dict[str, List[Any]]] = defaultdict(
            lambda: {
                "backups_to_copy": [],
                "backups_to_relocate": [],
                "backups_remote_move": [],
                "duplicates_to_delete": [],
                "important_items": [],
            }
        )
        pair_metrics: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
            lambda: {
                "local_copies_count": 0,
                "remote_copies_count": 0,
                "new_copies_count": 0,
                "backups_to_copy_count": 0,
                "backups_to_relocate_count": 0,
                "duplicates_to_delete_count": 0,
                "important_to_assign_count": 0,
            }
        )
        orig_cluster_pair: Dict[str, Tuple[int, int]] = {}
        important_candidates: List[Tuple[MediaItem, WindowsVolume]] = []
        unbacked_items: List[MediaItem] = []
        unbacked_pair_map: Dict[str, str] = {}
        unbacked_pair_keys_str: Dict[str, str] = {}
        local_reloc_rows: List[Dict[str, Any]] = []
        remote_move_rows: List[Dict[str, Any]] = []
        unassigned_unbacked_by_vol: Dict[int, int] = defaultdict(int)
        unbacked_count = 0
        unbacked_count_by_vol: Dict[int, int] = defaultdict(int)
        unbacked_count_by_pair: Dict[str, int] = defaultdict(int)
        local_reloc_by_pair: Dict[str, int] = defaultdict(int)
        remote_move_by_pair: Dict[str, int] = defaultdict(int)
        cross_dup_by_pair: Dict[str, int] = defaultdict(int)
        same_vol_dup_by_vol: Dict[int, int] = defaultdict(int)
        redundant_duplicates_count = sum(1 for it in all_items if it.duplicate_role == "duplicate")
        single_device_warning = len(volumes) < 2

        for cinfo in cluster_infos:
            cluster = cinfo["cluster"]
            best_orig: MediaItem = cinfo["orig"]
            orig_vol = volumes.get(best_orig.volume_id)
            fsize = int(best_orig.file_size_bytes or 0)

            if cinfo["same_vol_non_bkp_cnt"] >= 2 and orig_vol and not is_in_important_album(best_orig, orig_vol.volume_label or ""):
                important_candidates.append((best_orig, orig_vol))

            other_vol_copies = sorted(
                cinfo["other_vol_copies"],
                key=lambda x: (
                    not x.is_backup_copy,
                    x.volume_id not in effective_target_set,
                    not volumes[x.volume_id].is_online if x.volume_id in volumes else True,
                    x.id,
                ),
            )
            chosen_backup: Optional[MediaItem] = other_vol_copies[0] if other_vol_copies else None

            if chosen_backup is None:
                unbacked_count += 1
                unbacked_items.append(best_orig)
                unbacked_count_by_vol[best_orig.volume_id] += 1

                # Pick target backup drive (first honoring selected_pairs_set if set, else overall)
                target_vol = pick_backup_volume(best_orig.volume_id, fsize, respect_pair_selection=False)
                if selected_pairs_set is not None and target_vol is not None:
                    pk_check = (min(best_orig.volume_id, target_vol.id), max(best_orig.volume_id, target_vol.id))
                    if pk_check not in selected_pairs_set:
                        alt_vol = pick_backup_volume(best_orig.volume_id, fsize, respect_pair_selection=True)
                        if alt_vol is not None:
                            target_vol = alt_vol

                if target_vol:
                    pair_key = (min(best_orig.volume_id, target_vol.id), max(best_orig.volume_id, target_vol.id))
                    orig_cluster_pair[best_orig.id] = pair_key
                    pk_str = f"{pair_key[0]}:{pair_key[1]}"
                    unbacked_pair_map[best_orig.id] = pk_str
                    unbacked_pair_keys_str[best_orig.id] = pk_str
                    unbacked_count_by_pair[pk_str] += 1
                    pair_metrics[pair_key]["new_copies_count"] += 1
                    pair_metrics[pair_key]["backups_to_copy_count"] += 1

                    dst_rel = f"{self.BACKUP_DIR_NAME}/{best_orig.effective_date.strftime('%Y/%m/%d')}/{best_orig.file_name}"
                    if selected_pairs_set is None or pair_key in selected_pairs_set:
                        required_bytes_by_vol[target_vol.id] += fsize
                    pair_tasks[pair_key]["backups_to_copy"].append(
                        {
                            "orig_item_id": best_orig.id,
                            "src_vol_id": best_orig.volume_id,
                            "dst_vol_id": target_vol.id,
                            "file_size": fsize,
                            "file_name": best_orig.file_name,
                            "src_relative_path": best_orig.relative_path,
                            "dst_relative_path": dst_rel,
                            "file_hash": best_orig.file_hash,
                        }
                    )
                else:
                    unassigned_unbacked_by_vol[best_orig.volume_id] += 1
            else:
                # Cluster already has a Green BACKUP on another device (chosen_backup)
                should_consolidate_remotely = (not minimize_transfers) and (chosen_backup.volume_id not in effective_target_set)
                remote_target_vol = None
                if should_consolidate_remotely:
                    # Pick a target volume != best_orig.volume_id (also != chosen_backup.volume_id)
                    cands_rem = [
                        volumes[vid] for vid in effective_target_ids
                        if vid != best_orig.volume_id and vid != chosen_backup.volume_id
                    ]
                    if cands_rem:
                        cands_rem.sort(
                            key=lambda c: (
                                vol_has_known_cap(c.id) and ((vol_avail_free(c.id) - fsize) < self.MIN_FREE_SPACE_BUFFER_BYTES),
                                orig_count_by_vol.get(c.id, 0),
                                not c.is_online,
                                -vol_avail_free(c.id),
                                c.id,
                            )
                        )
                        remote_target_vol = cands_rem[0]

                if remote_target_vol is not None:
                    pair_key = (
                        min(chosen_backup.volume_id, remote_target_vol.id),
                        max(chosen_backup.volume_id, remote_target_vol.id),
                    )
                    orig_cluster_pair[best_orig.id] = pair_key
                    pk_str = f"{pair_key[0]}:{pair_key[1]}"
                    src_v = volumes.get(chosen_backup.volume_id)
                    dst_rel = f"{self.BACKUP_DIR_NAME}/{chosen_backup.effective_date.strftime('%Y/%m/%d')}/{chosen_backup.file_name}"
                    pair_metrics[pair_key]["remote_copies_count"] += 1
                    remote_move_by_pair[pk_str] += 1
                    if selected_pairs_set is None or pair_key in selected_pairs_set:
                        freed_bytes_by_vol[chosen_backup.volume_id] += int(chosen_backup.file_size_bytes or 0)
                        required_bytes_by_vol[remote_target_vol.id] += int(chosen_backup.file_size_bytes or 0)
                    task_obj = {
                        "backup_item_id": chosen_backup.id,
                        "src_vol_id": chosen_backup.volume_id,
                        "dst_vol_id": remote_target_vol.id,
                        "src_vol_label": src_v.volume_label if src_v else "?",
                        "src_vol_letter": src_v.last_known_drive_letter if src_v else "?",
                        "dst_vol_label": remote_target_vol.volume_label,
                        "dst_vol_letter": remote_target_vol.last_known_drive_letter,
                        "file_name": chosen_backup.file_name,
                        "file_size": int(chosen_backup.file_size_bytes or 0),
                        "src_relative_path": chosen_backup.relative_path,
                        "dst_relative_path": dst_rel,
                        "pair_key_str": pk_str,
                    }
                    remote_move_rows.append(task_obj)
                    pair_tasks[pair_key]["backups_remote_move"].append(task_obj)
                elif not chosen_backup.is_backup_copy:
                    # Green BACKUP outside daf_backup/ stays on its device -> Kopie lokalne
                    pair_key = (min(best_orig.volume_id, chosen_backup.volume_id), max(best_orig.volume_id, chosen_backup.volume_id))
                    orig_cluster_pair[best_orig.id] = pair_key
                    pk_str = f"{pair_key[0]}:{pair_key[1]}"
                    b_vol = volumes.get(chosen_backup.volume_id)
                    dst_rel = f"{self.BACKUP_DIR_NAME}/{chosen_backup.effective_date.strftime('%Y/%m/%d')}/{chosen_backup.file_name}"
                    pair_metrics[pair_key]["local_copies_count"] += 1
                    pair_metrics[pair_key]["backups_to_relocate_count"] += 1
                    local_reloc_by_pair[pk_str] += 1
                    reloc_obj = {
                        "backup_item_id": chosen_backup.id,
                        "vol_id": chosen_backup.volume_id,
                        "vol_label": b_vol.volume_label if b_vol else "?",
                        "vol_letter": b_vol.last_known_drive_letter if b_vol else "?",
                        "file_name": chosen_backup.file_name,
                        "file_size": int(chosen_backup.file_size_bytes or 0),
                        "src_relative_path": chosen_backup.relative_path,
                        "dst_relative_path": dst_rel,
                        "pair_key_str": pk_str,
                    }
                    local_reloc_rows.append(reloc_obj)
                    pair_tasks[pair_key]["backups_to_relocate"].append(reloc_obj)

        important_count_by_vol: Dict[int, int] = defaultdict(int)
        important_by_pair: Dict[str, int] = defaultdict(int)
        important_rows: List[Dict[str, Any]] = []

        def _pick_pair_for_important(orig_id: str, orig_vid: int) -> Optional[Tuple[int, int]]:
            # Reuse the pair already scheduled for this item's copy/reloc/remote task
            existing_pk = orig_cluster_pair.get(orig_id)
            if existing_pk and (selected_pairs_set is None or existing_pk in selected_pairs_set):
                return existing_pk
            cand_pks = [p for p in all_possible_pairs if orig_vid in p]
            if not cand_pks:
                return None

            def _rank(pk: Tuple[int, int]):
                va_o = bool(volumes.get(pk[0]) and volumes[pk[0]].is_online)
                vb_o = bool(volumes.get(pk[1]) and volumes[pk[1]].is_online)
                in_sel = (selected_pairs_set is None) or (pk in selected_pairs_set)
                ops_here = (
                    len(pair_tasks[pk]["backups_to_copy"])
                    + len(pair_tasks[pk]["backups_to_relocate"])
                    + len(pair_tasks[pk]["backups_remote_move"])
                )
                partner_id = pk[1] if pk[0] == orig_vid else pk[0]
                return (
                    not in_sel,
                    not (va_o and vb_o),
                    ops_here == 0,
                    partner_id not in effective_target_set,
                    -ops_here,
                    pk,
                )

            cand_pks.sort(key=_rank)
            return cand_pks[0]

        for orig_item, orig_vol in important_candidates:
            important_count_by_vol[orig_vol.id] += 1
            chosen_pk = _pick_pair_for_important(orig_item.id, orig_vol.id)
            if chosen_pk is not None:
                pk_str = f"{chosen_pk[0]}:{chosen_pk[1]}"
                pair_tasks[chosen_pk]["important_items"].append((orig_item, orig_vol))
                pair_metrics[chosen_pk]["important_to_assign_count"] += 1
                important_by_pair[pk_str] += 1
                pks_for_row = [pk_str]
            else:
                pks_for_row = []
            important_rows.append(
                {
                    "item_id": orig_item.id,
                    "file_name": orig_item.file_name,
                    "file_size": int(orig_item.file_size_bytes or 0),
                    "vol_id": orig_vol.id,
                    "vol_label": orig_vol.volume_label,
                    "vol_letter": orig_vol.last_known_drive_letter,
                    "relative_path": orig_item.relative_path,
                    "target_album": f"Ważne - {orig_vol.volume_label} / {orig_item.effective_date.strftime('%Y / %m / %d')}",
                    "pair_keys": pks_for_row,
                    "pair_keys_str": ",".join(pks_for_row),
                }
            )

        volume_space_summary: Dict[int, Dict[str, Any]] = {}
        space_problems: List[str] = []
        for vid, v in volumes.items():
            known_free = int(v.free_space_bytes or 0)
            known_total = int(v.total_space_bytes or 0)
            has_known = known_total > 0 or known_free > 0
            reclaimed = freed_bytes_by_vol.get(vid, 0)
            required = required_bytes_by_vol.get(vid, 0)
            projected = known_free + reclaimed - required
            space_ok = (not has_known) or (projected >= self.MIN_FREE_SPACE_BUFFER_BYTES) or (required == 0)
            volume_space_summary[vid] = {
                "volume": v,
                "known_free_bytes": known_free,
                "known_total_bytes": known_total,
                "has_known_capacity": has_known,
                "reclaimed_bytes": reclaimed,
                "required_bytes": required,
                "projected_free_bytes": projected,
                "space_ok": space_ok,
                "orig_count": orig_count_by_vol.get(vid, 0),
                "backup_count": backup_count_by_vol.get(vid, 0),
                "backup_bytes": backup_bytes_by_vol.get(vid, 0),
                "bkp_valid_in_vault_count": bkp_valid_in_vault_by_vol.get(vid, 0),
                "bkp_reloc_green_count": bkp_reloc_green_by_vol.get(vid, 0),
                "bkp_reloc_orange_count": bkp_reloc_orange_by_vol.get(vid, 0),
            }
            if not space_ok and required > 0:
                deficit_mb = max(0.0, (self.MIN_FREE_SPACE_BUFFER_BYTES - projected) / (1024 * 1024))
                space_problems.append(
                    f"Brak miejsca na dysku docelowym {v.volume_label} ({v.last_known_drive_letter}): "
                    f"wymagane {required / (1024*1024):.1f} MB, zabraknie ok. {deficit_mb:.1f} MB."
                )

        if not effective_target_ids and len(volumes) >= 2:
            space_problems.append("Nie wybrano żadnego urządzenia docelowego do backupu — zaznacz co najmniej jeden dysk powyżej.")
        for src_vid, cnt in unassigned_unbacked_by_vol.items():
            sv = volumes.get(src_vid)
            if sv and cnt > 0 and effective_target_ids:
                space_problems.append(
                    f"{cnt} plików bez backupu z dysku {sv.volume_label} ({sv.last_known_drive_letter}) "
                    f"nie ma innego wybranego dysku docelowego (zaznacz dodatkowy dysk jako docelowy, aby wykonać ich kopię)."
                )

        def _device_pair_metrics(dev_id: int, tasks_dict: Dict[str, List[Any]]) -> Dict[str, int]:
            loc_c = sum(1 for t in tasks_dict["backups_to_relocate"] if t["vol_id"] == dev_id)
            rem_c = sum(1 for t in tasks_dict["backups_remote_move"] if t["dst_vol_id"] == dev_id)
            new_c = sum(1 for t in tasks_dict["backups_to_copy"] if t["dst_vol_id"] == dev_id)
            imp_c = sum(1 for _, ov in tasks_dict["important_items"] if ov.id == dev_id)
            return {
                "local_copies_count": loc_c,
                "remote_copies_count": rem_c,
                "new_copies_count": new_c,
                "important_to_assign_count": imp_c,
                "total_touched": loc_c + rem_c + new_c + imp_c,
            }

        ordered_pairs = []
        for (va_id, vb_id) in all_possible_pairs:
            va = volumes.get(va_id)
            vb = volumes.get(vb_id)
            both_online = bool(va and vb and va.is_online and vb.is_online)
            one_online = bool((va and va.is_online) or (vb and vb.is_online))
            tasks = pair_tasks[(va_id, vb_id)]
            metrics = pair_metrics[(va_id, vb_id)]
            a_metrics = _device_pair_metrics(va_id, tasks)
            b_metrics = _device_pair_metrics(vb_id, tasks)
            total_ops = (
                len(tasks["backups_to_copy"])
                + len(tasks["backups_to_relocate"])
                + len(tasks["backups_remote_move"])
            )
            has_work = (total_ops + metrics["important_to_assign_count"]) > 0
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
                    "has_work": has_work,
                    "metrics": metrics,
                    "a_metrics": a_metrics,
                    "b_metrics": b_metrics,
                }
            )

        ordered_pairs.sort(key=lambda p: (not p["has_work"], not p["both_online"], not p["one_online"], p["pair_key"]))
        scheduled_pairs = [p for p in ordered_pairs if p["has_work"]]
        reconciliation = {
            "local_copies_pairs_sum": sum(p["metrics"]["local_copies_count"] for p in scheduled_pairs),
            "local_copies_plan_total": len(local_reloc_rows),
            "remote_copies_pairs_sum": sum(p["metrics"]["remote_copies_count"] for p in scheduled_pairs),
            "remote_copies_plan_total": len(remote_move_rows),
            "new_copies_pairs_sum": sum(p["metrics"]["new_copies_count"] for p in scheduled_pairs),
            "new_copies_plan_total": unbacked_count,
            "unassigned_new_copies": sum(unassigned_unbacked_by_vol.values()),
            "important_pairs_sum": sum(p["metrics"]["important_to_assign_count"] for p in scheduled_pairs),
            "important_plan_total": len(important_candidates),
        }

        return {
            "volumes": volumes,
            "volume_space_summary": volume_space_summary,
            "space_problems": space_problems,
            "default_target_volume_ids": default_target_ids,
            "effective_target_volume_ids": effective_target_ids,
            "minimize_transfers": bool(minimize_transfers),
            "stats": {
                "total_unique": total_unique_count,
                "missing_backups": missing_backup_count,
                "green_backups": green_backup_count,
                "green_in_vault": green_in_vault_count,
                "green_outside_vault": green_outside_vault_count,
                "orange_backups": orange_backup_count,
                "total_files": len(all_items),
                "redundant_duplicates": redundant_duplicates_count,
            },
            "important_candidates": important_candidates,
            "important_rows": important_rows,
            "important_count_by_vol": dict(important_count_by_vol),
            "important_by_pair": dict(important_by_pair),
            "ordered_pairs": ordered_pairs,
            "scheduled_pairs": scheduled_pairs,
            "reconciliation": reconciliation,
            "duplicate_rows": [],
            "unbacked_items": unbacked_items,
            "unbacked_pair_map": unbacked_pair_map,
            "unbacked_pair_keys_str": unbacked_pair_keys_str,
            "local_reloc_rows": local_reloc_rows,
            "remote_move_rows": remote_move_rows,
            "unbacked_count": unbacked_count,
            "redundant_duplicates_count": redundant_duplicates_count,
            "single_device_warning": single_device_warning,
            "unbacked_count_by_vol": dict(unbacked_count_by_vol),
            "unbacked_count_by_pair": dict(unbacked_count_by_pair),
            "local_reloc_by_pair": dict(local_reloc_by_pair),
            "remote_move_by_pair": dict(remote_move_by_pair),
            "cross_dup_by_pair": dict(cross_dup_by_pair),
            "same_vol_dup_by_vol": dict(same_vol_dup_by_vol),
        }

    def execute_backup_and_deduplication(
        self,
        preferred_target_volume_id: Optional[int] = None,
        job_id: Optional[int] = None,
        operation_mode: str = "backup_only",
        selected_pair_keys: Optional[List[str]] = None,
        target_volume_ids: Optional[List[int]] = None,
        minimize_transfers: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes the automated Backup workflow for selected volume pairs and target drives:
        1) Assigns Originals with same-device duplicates to 'Ważne - <device>' -> YYYY -> MM -> DD.
        2) Local relocations (Kopie lokalne): moves green BACKUPs into daf_backup/YYYY/MM/DD on the same drive.
        3) Remote moves (Kopie zdalne): moves green BACKUPs onto chosen minimal backup drives (100% verified).
        4) New copies (Nowe kopie): copies unbacked / orange Originals to target daf_backup/YYYY/MM/DD (100% verified).
        Prompts user to connect offline drives for any remaining pending pairs.
        """
        from app.operation_log import ProcessManager
        import uuid

        selected_pairs_set = self.parse_pair_keys(selected_pair_keys)

        self.backup_mgr.create_backup("pre_auto_backup")
        plan = self.build_backup_and_dedup_plan(
            preferred_target_volume_id=preferred_target_volume_id,
            selected_pair_keys=selected_pair_keys,
            target_volume_ids=target_volume_ids,
            minimize_transfers=minimize_transfers,
        )
        volumes: Dict[int, WindowsVolume] = plan["volumes"]

        # Step 1: Assign Originals with same-device duplicates to 'Ważne - <volume_label>' -> YYYY -> MM -> DD
        # Each candidate is assigned 1-to-1 to a single scheduled pair (or directly when <2 volumes exist).
        important_assigned = 0
        if len(volumes) < 2:
            for orig_item, orig_vol in plan["important_candidates"]:
                self.ensure_important_album_hierarchy(orig_item, orig_vol.volume_label)
                important_assigned += 1
        else:
            for pair_entry in plan["ordered_pairs"]:
                if selected_pairs_set is not None and pair_entry["pair_key"] not in selected_pairs_set:
                    continue
                for orig_item, orig_vol in pair_entry["tasks"]["important_items"]:
                    self.ensure_important_album_hierarchy(orig_item, orig_vol.volume_label)
                    important_assigned += 1
        if important_assigned > 0:
            self.db.commit()

        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        backups_created = 0
        backups_relocated = 0
        backups_remote_moved = 0
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

            # Local relocations on an online volume can succeed even if the partner volume is offline
            local_done_in_pair = 0
            for reloc_task in tasks["backups_to_relocate"]:
                bkp_item = self.db.get(MediaItem, reloc_task["backup_item_id"])
                bkp_vol = volumes.get(reloc_task["vol_id"])
                vol_root = self.verify_and_resolve_online_volume_root(bkp_vol, connected_map=connected_map)
                if not bkp_item or not bkp_vol or vol_root is None:
                    continue
                curr_path = vol_root / bkp_item.relative_path
                if not curr_path.exists():
                    continue
                sub_day = bkp_item.effective_date.strftime("%Y/%m/%d")
                target_bkp_dir = vol_root / self.BACKUP_DIR_NAME / sub_day
                target_bkp_dir.mkdir(parents=True, exist_ok=True)
                target_bkp_path = target_bkp_dir / bkp_item.file_name
                if curr_path.resolve() != target_bkp_path.resolve():
                    c = 1
                    while target_bkp_path.exists():
                        target_bkp_path = target_bkp_dir / f"{curr_path.stem}_{c}{curr_path.suffix}"
                        c += 1
                    _safe_move(curr_path, target_bkp_path)
                    bkp_item.file_name = target_bkp_path.name
                    bkp_item.relative_path = target_bkp_path.resolve().relative_to(vol_root.resolve()).as_posix()
                bkp_item.is_backup_copy = True
                bkp_item.is_primary_duplicate = False
                bkp_item.duplicate_role = "backup_green"
                backups_relocated += 1
                local_done_in_pair += 1

            va_root = self.verify_and_resolve_online_volume_root(va, connected_map=connected_map)
            vb_root = self.verify_and_resolve_online_volume_root(vb, connected_map=connected_map)
            has_cross_work = bool(tasks["backups_to_copy"] or tasks["backups_remote_move"])
            if va_root is None or vb_root is None:
                if has_cross_work or (len(tasks["backups_to_relocate"]) > local_done_in_pair):
                    pending_offline_pairs.append(pair_entry)
                if local_done_in_pair > 0:
                    self.db.commit()
                continue

            # Cross-device Remote Moves (Kopie zdalne) with 100% verification
            for rmove in tasks["backups_remote_move"]:
                if job_id and ProcessManager.is_stop_requested(job_id):
                    break
                bkp_item = self.db.get(MediaItem, rmove["backup_item_id"])
                src_vol = volumes.get(rmove["src_vol_id"])
                dst_vol = volumes.get(rmove["dst_vol_id"])
                src_root_v = self.verify_and_resolve_online_volume_root(src_vol, connected_map=connected_map)
                dst_root = self.verify_and_resolve_online_volume_root(dst_vol, connected_map=connected_map)
                if not bkp_item or not src_vol or not dst_vol or src_root_v is None or dst_root is None:
                    continue
                src_path = src_root_v / bkp_item.relative_path
                if not src_path.exists():
                    continue
                has_space, free_bytes = self.check_volume_free_space(
                    str(dst_root), bkp_item.file_size_bytes
                )
                if not has_space:
                    space_warnings.append(
                        f"Brak miejsca na dysku {dst_vol.volume_label} ({dst_root}): "
                        f"wymagane {bkp_item.file_size_bytes / (1024*1024):.1f} MB, wolne {free_bytes / (1024*1024):.1f} MB."
                    )
                    continue
                sub_day = bkp_item.effective_date.strftime("%Y/%m/%d")
                backup_dir = dst_root / self.BACKUP_DIR_NAME / sub_day
                backup_dir.mkdir(parents=True, exist_ok=True)
                dest_path = backup_dir / bkp_item.file_name
                c = 1
                while dest_path.exists():
                    dest_path = backup_dir / f"{src_path.stem}_{c}{src_path.suffix}"
                    c += 1
                shutil.copy2(str(src_path), str(dest_path))
                verified, reason, full_sha = FastHasher.verify_files_identical_100_percent(src_path, dest_path)
                if not verified:
                    try:
                        _safe_unlink(dest_path)
                    except OSError as exc:
                        logger.debug("Cleanup failed remote backup move %s: %s", dest_path, exc)
                    verification_blocks += 1
                    continue
                try:
                    _safe_unlink(src_path)
                except OSError as exc:
                    logger.debug("Source unlink warning after remote backup move %s: %s", src_path, exc)
                bkp_item.volume_id = dst_vol.id
                bkp_item.file_name = dest_path.name
                bkp_item.relative_path = dest_path.resolve().relative_to(dst_root.resolve()).as_posix()
                bkp_item.is_backup_copy = True
                bkp_item.is_primary_duplicate = False
                bkp_item.duplicate_role = "backup_green"
                bkp_item.verified_sha256 = full_sha
                backups_remote_moved += 1
                if job_id:
                    ProcessManager.update_progress(
                        job_id, processed_delta=1, message=f"Przeniesiono backup: {bkp_item.file_name}", db=self.db
                    )

            # New Backup Copying (Nowe kopie) with 100% verification
            for copy_task in tasks["backups_to_copy"]:
                if job_id and ProcessManager.is_stop_requested(job_id):
                    break
                orig_item = self.db.get(MediaItem, copy_task["orig_item_id"])
                src_vol = volumes.get(copy_task["src_vol_id"])
                dst_vol = volumes.get(copy_task["dst_vol_id"])
                src_root_v = self.verify_and_resolve_online_volume_root(src_vol, connected_map=connected_map)
                dst_root = self.verify_and_resolve_online_volume_root(dst_vol, connected_map=connected_map)
                if not orig_item or not src_vol or not dst_vol or src_root_v is None or dst_root is None:
                    continue

                src_path = src_root_v / orig_item.relative_path
                if not src_path.exists():
                    continue

                has_space, free_bytes = self.check_volume_free_space(
                    str(dst_root), orig_item.file_size_bytes
                )
                if not has_space:
                    space_warnings.append(
                        f"Brak miejsca na dysku {dst_vol.volume_label} ({dst_root}): "
                        f"wymagane {orig_item.file_size_bytes / (1024*1024):.1f} MB, wolne {free_bytes / (1024*1024):.1f} MB."
                    )
                    continue
                sub_day = orig_item.effective_date.strftime("%Y/%m/%d")
                backup_dir = dst_root / self.BACKUP_DIR_NAME / sub_day
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
                    duplicate_role="backup_green",
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
                backups_created += 1
                if job_id:
                    ProcessManager.update_progress(
                        job_id, processed_delta=1, message=f"Utworzono backup: {orig_item.file_name}", db=self.db
                    )

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
                "remote_moves": len(nxt["tasks"]["backups_remote_move"]),
                "local_relocs": len(nxt["tasks"]["backups_to_relocate"]),
                "remaining_pairs_count": len(pending_offline_pairs),
            }

        summary_msg = (
            f"Nowe kopie (backupy): {backups_created} | Kopie lokalne do '{self.BACKUP_DIR_NAME}/': {backups_relocated} | "
            f"Kopie zdalne: {backups_remote_moved} | Dodano do albumu 'Ważne': {important_assigned}"
        )
        if skipped_unselected_pairs > 0:
            summary_msg += f" | Pominięto niewybrane pary dysków: {skipped_unselected_pairs}"

        OperationLogger.log(
            "BACKUP_EXEC",
            summary_msg,
            status="OK" if not pending_offline_pairs else "PARTIAL",
            details={
                "backups_created": backups_created,
                "backups_relocated": backups_relocated,
                "backups_remote_moved": backups_remote_moved,
                "important_assigned": important_assigned,
                "verification_blocks": verification_blocks,
                "completed_pairs": completed_pairs,
                "pending_pairs": len(pending_offline_pairs),
                "skipped_unselected_pairs": skipped_unselected_pairs,
            },
            db=self.db,
        )

        return {
            "operation_mode": "backup_only",
            "backups_created": backups_created,
            "backups_relocated": backups_relocated,
            "backups_remote_moved": backups_remote_moved,
            "duplicates_deleted": 0,
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

    def _record_deleted_duplicate_history(
        self,
        cand: MediaItem,
        vol: Optional[WindowsVolume],
        orig_id: Optional[str] = None,
    ) -> None:
        """Records the former path and volume of a permanently deleted duplicate linked to its cluster file_hash."""
        self.db.execute(
            text(
                """
                INSERT OR IGNORE INTO deleted_duplicate_history (
                    deleted_item_id, original_item_id, file_hash, volume_id,
                    volume_label, drive_letter, relative_path, file_name, deleted_at
                ) VALUES (
                    :deleted_item_id, :original_item_id, :file_hash, :volume_id,
                    :volume_label, :drive_letter, :relative_path, :file_name, :deleted_at
                )
                """
            ),
            {
                "deleted_item_id": cand.id,
                "original_item_id": orig_id,
                "file_hash": cand.file_hash,
                "volume_id": cand.volume_id,
                "volume_label": vol.volume_label if vol else "",
                "drive_letter": vol.last_known_drive_letter if vol else "",
                "relative_path": cand.relative_path,
                "file_name": cand.file_name,
                "deleted_at": datetime.datetime.utcnow(),
            },
        )

    def recover_deleted_duplicate_history_from_backup(self, backup_db_path: Path) -> int:
        """
        Imports deleted duplicate paths from a pre-deletion SQLite backup database into deleted_duplicate_history
        for all items that were removed from media_items but still have a surviving copy in the current DB.
        """
        if not backup_db_path or not backup_db_path.exists():
            return 0
        self.db.commit()
        target_db = db_module.DB_PATH
        conn = sqlite3.connect(target_db.as_posix(), timeout=30.0)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA busy_timeout = 30000;")
            cur.execute("ATTACH DATABASE ? AS old_db", (backup_db_path.as_posix(),))
            before_cnt = cur.execute("SELECT count(*) FROM main.deleted_duplicate_history").fetchone()[0]
            cur.execute(
                """
                INSERT OR IGNORE INTO main.deleted_duplicate_history (
                    deleted_item_id, original_item_id, file_hash, volume_id,
                    volume_label, drive_letter, relative_path, file_name, deleted_at
                )
                SELECT
                    o.id,
                    (
                        SELECT m.id FROM main.media_items m
                        WHERE m.file_hash = o.file_hash AND m.is_trashed = 0
                        ORDER BY m.is_primary_duplicate DESC, m.effective_date ASC, m.id ASC
                        LIMIT 1
                    ),
                    o.file_hash,
                    o.volume_id,
                    COALESCE(v.volume_label, ''),
                    COALESCE(v.last_known_drive_letter, ''),
                    o.relative_path,
                    o.file_name,
                    CURRENT_TIMESTAMP
                FROM old_db.media_items o
                LEFT JOIN old_db.windows_volumes v ON v.id = o.volume_id
                WHERE NOT EXISTS (SELECT 1 FROM main.media_items m WHERE m.id = o.id)
                  AND EXISTS (SELECT 1 FROM main.media_items m2 WHERE m2.file_hash = o.file_hash AND m2.is_trashed = 0);
                """
            )
            conn.commit()
            after_cnt = cur.execute("SELECT count(*) FROM main.deleted_duplicate_history").fetchone()[0]
            cur.execute("DETACH DATABASE old_db")
            return max(0, int(after_cnt - before_cnt))
        finally:
            conn.close()

    def execute_safe_duplicate_removal(
        self,
        item_ids: List[str],
        delete_mode: str = "trash",
        force_duplicate_verification: bool = False,
        job_id: Optional[int] = None,
        trash_dir: Optional[Path] = None,
        thumbnails_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Safely moves to trash or permanently deletes items disk-by-disk (ONLINE drives first).
        For any duplicate (or when force_duplicate_verification=True):
        - Performs 100% exact multi-layer binary verification against a surviving copy on an online drive.
        - Ensures BOTH Original AND Backup survive in the cluster (>= 2 copies remain) before deleting.
        - Queues offline-drive items so the user is prompted to connect the next drive one by one.
        """
        from app.operation_log import ProcessManager

        eff_trash_dir = trash_dir or (db_module.DATA_DIR / "kosz")
        eff_thumbs_dir = thumbnails_dir or db_module.THUMBNAILS_DIR

        if progress_cb:
            progress_cb(0, len(item_ids), "Weryfikacja dysków i aktualizacja klastrów duplikatów...", "")
        if job_id:
            ProcessManager.update_progress(
                job_id,
                processed_count=0,
                total_count=len(item_ids),
                message="Weryfikacja dysków i aktualizacja klastrów duplikatów...",
                db=self.db,
            )

        self.refresh_volumes_online_status()
        needs_initial_flag = (
            self.db.scalar(
                select(MediaItem.id)
                .where(MediaItem.is_trashed == False, MediaItem.duplicate_role.is_(None))
                .limit(1)
            )
            is not None
        )
        if needs_initial_flag:
            self.flag_duplicate_clusters()

        target_items: List[MediaItem] = []
        for idx in range(0, len(item_ids), 5000):
            chunk_ids = item_ids[idx : idx + 5000]
            target_items.extend(
                self.db.scalars(
                    select(MediaItem).where(MediaItem.id.in_(chunk_ids), MediaItem.is_trashed == False)
                ).all()
            )
        if not target_items:
            return {
                "processed_count": 0,
                "physical_count": 0,
                "blocked_safety": 0,
                "blocked_verify": 0,
                "missing_on_disk": 0,
                "offline_items": [],
                "offline_drives": [],
                "next_offline_prompt": None,
                "processed_drives": [],
                "stopped_by_user": False,
            }

        total_candidates = len(target_items)
        step_idx = 0
        stopped_by_user = False

        def _report_step(drive_text: str, msg_text: str) -> None:
            if progress_cb:
                progress_cb(step_idx, total_candidates, msg_text, drive_text)
            if job_id:
                ProcessManager.update_cache_only(
                    job_id,
                    message=msg_text,
                    processed_count=step_idx,
                    total_count=total_candidates,
                )

        if progress_cb:
            progress_cb(0, total_candidates, f"Przygotowanie usuwania ({total_candidates} plików)...", "")
        if job_id:
            ProcessManager.update_progress(
                job_id,
                processed_count=0,
                total_count=total_candidates,
                message=f"Przygotowanie usuwania ({total_candidates} plików)...",
                db=self.db,
            )

        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        touched_hashes = list({it.file_hash for it in target_items})
        cluster_members: List[MediaItem] = []
        for idx in range(0, len(touched_hashes), 5000):
            h_chunk = touched_hashes[idx : idx + 5000]
            cluster_members.extend(
                self.db.scalars(
                    select(MediaItem)
                    .where(MediaItem.file_hash.in_(h_chunk), MediaItem.is_trashed == False)
                ).all()
            )
        cluster_members.sort(key=lambda m: (m.effective_date, m.id))
        by_hash: Dict[str, List[MediaItem]] = defaultdict(list)
        for m in cluster_members:
            by_hash[m.file_hash].append(m)

        # Resolve VSN-verified live roots before sorting ONLINE drives first
        verified_roots: Dict[int, Optional[Path]] = {
            vid: self.verify_and_resolve_online_volume_root(v, connected_map=connected_map)
            for vid, v in volumes.items()
        }

        # Group items by volume_id and process ONLINE volumes first (disk-by-disk)
        items_by_vol: Dict[int, List[MediaItem]] = defaultdict(list)
        for it in target_items:
            items_by_vol[it.volume_id].append(it)

        sorted_vids = sorted(
            items_by_vol.keys(),
            key=lambda vid: (verified_roots.get(vid) is None, vid),
        )

        removed_ids: Set[str] = set()
        processed_count = 0
        physical_count = 0
        blocked_safety = 0
        blocked_verify = 0
        missing_on_disk = 0
        offline_items: List[MediaItem] = []
        offline_vol_order: List[int] = []
        offline_ref_counts: Dict[int, int] = defaultdict(int)
        processed_drives: List[str] = []

        def _mark_offline(it_obj: MediaItem, needed_vid: int, is_ref_only: bool = False):
            offline_items.append(it_obj)
            if needed_vid not in offline_vol_order:
                offline_vol_order.append(needed_vid)
            if is_ref_only:
                offline_ref_counts[needed_vid] += 1

        for vid in sorted_vids:
            if stopped_by_user:
                break
            vol = volumes.get(vid)
            vol_root = verified_roots.get(vid)
            vol_items = items_by_vol[vid]
            drive_label = f"{vol.volume_label} ({vol_root or vol.last_known_drive_letter})" if vol else f"#{vid}"

            if not vol or vol_root is None:
                for it in vol_items:
                    if job_id and ProcessManager.is_stop_requested(job_id):
                        stopped_by_user = True
                        break
                    step_idx += 1
                    needs_dup_check = force_duplicate_verification or (it.duplicate_role in ("duplicate", "backup_green", "backup_orange"))
                    if needs_dup_check:
                        cluster = by_hash.get(it.file_hash, [])
                        surviving = [
                            x for x in cluster
                            if x.id != it.id and x.id not in removed_ids and not x.is_trashed
                        ]
                        if it.duplicate_role == "original" or it.is_backup_copy or len(surviving) < 2:
                            blocked_safety += 1
                            continue
                        orig = next((x for x in surviving if x.duplicate_role == "original"), surviving[0])
                        bkp = next(
                            (x for x in surviving if x.id != orig.id and x.volume_id != orig.volume_id),
                            next((x for x in surviving if x.id != orig.id), None),
                        )
                        if orig is None or bkp is None:
                            blocked_safety += 1
                            continue
                        _mark_offline(it, vid)
                    else:
                        if delete_mode == "trash":
                            it.is_trashed = True
                        else:
                            self.db.delete(it)
                        self.db.commit()
                        removed_ids.add(it.id)
                        processed_count += 1
                    _report_step(drive_label, f"[{step_idx}/{total_candidates}] {drive_label} (offline): {it.file_name}")
                continue

            drive_processed_cnt = 0
            for cand in vol_items:
                if job_id and ProcessManager.is_stop_requested(job_id):
                    stopped_by_user = True
                    break

                step_idx += 1
                _report_step(drive_label, f"[{step_idx}/{total_candidates}] Weryfikacja 100%: {cand.file_name} ({drive_label})")

                cluster = by_hash.get(cand.file_hash, [])
                needs_dup_check = force_duplicate_verification or (cand.duplicate_role == "duplicate")

                if needs_dup_check:
                    surviving = [
                        x for x in cluster
                        if x.id != cand.id and x.id not in removed_ids and not x.is_trashed
                    ]
                    # Safety gate 1: Must keep BOTH Original AND Backup (>= 2 copies survive) & never delete Original or vault Backup
                    if cand.duplicate_role == "original" or cand.is_backup_copy or len(surviving) < 2:
                        blocked_safety += 1
                        continue

                    orig = next((x for x in surviving if x.duplicate_role == "original"), surviving[0])
                    bkp = next(
                        (x for x in surviving if x.id != orig.id and x.volume_id != orig.volume_id),
                        next((x for x in surviving if x.id != orig.id), None),
                    )
                    if orig is None or bkp is None:
                        blocked_safety += 1
                        continue

                    cand_path = vol_root / cand.relative_path
                    if not cand_path.exists():
                        if vol_root.exists():
                            # Duplicate file was already physically deleted before previous session -> reconcile DB record
                            if delete_mode == "trash":
                                cand.is_trashed = True
                            else:
                                self._record_deleted_duplicate_history(cand, vol, orig_id=orig.id if orig else None)
                                self.db.delete(cand)
                            self.db.commit()
                            removed_ids.add(cand.id)
                            processed_count += 1
                            drive_processed_cnt += 1
                        missing_on_disk += 1
                        continue

                    # Safety gate 2: Need an ONLINE VSN-verified reference file to perform 100% binary verification
                    ref_candidates = [orig, bkp] + [x for x in surviving if x.id not in (orig.id, bkp.id)]
                    ref_item: Optional[MediaItem] = None
                    ref_path: Optional[Path] = None
                    for rc in ref_candidates:
                        rv_root = verified_roots.get(rc.volume_id)
                        if rv_root is not None:
                            rp = rv_root / rc.relative_path
                            if rp.exists() and rp.resolve() != cand_path.resolve():
                                ref_item = rc
                                ref_path = rp
                                break

                    # If backup is currently online, verify its file actually exists on disk before trusting it
                    bkp_root = verified_roots.get(bkp.volume_id)
                    if bkp_root is not None:
                        bkp_p = bkp_root / bkp.relative_path
                        if not bkp_p.exists() and len(surviving) == 2:
                            blocked_safety += 1
                            continue

                    if ref_item is not None and ref_path is not None:
                        is_identical, verify_msg, full_sha = FastHasher.verify_files_identical_100_percent(ref_path, cand_path)
                    else:
                        can_verify_from_meta = (
                            orig.file_size_bytes == bkp.file_size_bytes == cand.file_size_bytes
                            and (
                                cand.media_type == "image"
                                or orig.hash_algorithm == "partial_3x64k"
                                or bool(orig.verified_sha256 or bkp.verified_sha256)
                            )
                        )
                        if not can_verify_from_meta:
                            # Do NOT push an ONLINE drive's file back into offline_items (prevents disk ping-pong).
                            blocked_verify += 1
                            continue
                        expected_vsha = orig.verified_sha256 or bkp.verified_sha256
                        is_identical, verify_msg, full_sha = FastHasher.verify_candidate_against_cluster_metadata(
                            cand_path,
                            orig.file_size_bytes,
                            orig.file_hash,
                            orig.hash_algorithm,
                            expected_verified_sha256=expected_vsha,
                        )

                    if not is_identical:
                        blocked_verify += 1
                        OperationLogger.log(
                            "DUP_VERIFY_BLOCK",
                            f"Zablokowano usunięcie {cand_path} ({verify_msg})",
                            status="WARN",
                            db=self.db,
                            commit=False,
                        )
                        try:
                            true_sha, _, _ = FastHasher.compute_multi_hashes(cand_path)
                            cand.file_hash = true_sha
                            cand.hash_algorithm = "sha256_verified"
                        except Exception as exc:
                            logger.debug("Rehash collision separate skipped for %s: %s", cand_path, exc)
                        self.db.commit()
                        continue

                    if ref_item is not None and full_sha:
                        ref_item.verified_sha256 = full_sha
                    if full_sha:
                        orig.verified_sha256 = full_sha

                    if delete_mode == "trash":
                        # Logical-only KOSZ attribute: physical file stays completely untouched on disk
                        cand.is_trashed = True
                    else:
                        try:
                            _safe_unlink(cand_path)
                        except OSError as exc:
                            blocked_verify += 1
                            OperationLogger.log(
                                "DUP_DELETE_LOCKED",
                                f"Pominięto zablokowany plik {cand_path}: {exc}",
                                status="WARN",
                                db=self.db,
                                commit=True,
                            )
                            continue
                        physical_count += 1
                        thumb_shared = any(
                            x.thumbnail_relative_path == cand.thumbnail_relative_path
                            for x in cluster if x.id != cand.id and x.id not in removed_ids
                        )
                        if not thumb_shared and cand.thumbnail_relative_path:
                            tp = eff_thumbs_dir / cand.thumbnail_relative_path
                            if tp.exists():
                                try:
                                    _safe_unlink(tp)
                                except OSError as exc:
                                    logger.debug("Thumb unlink warning %s: %s", tp, exc)
                        self._record_deleted_duplicate_history(cand, vol, orig_id=orig.id if orig else None)
                        self.db.delete(cand)

                    removed_ids.add(cand.id)
                    processed_count += 1
                    drive_processed_cnt += 1
                else:
                    # Standard non-duplicate removal
                    src_p = vol_root / cand.relative_path
                    if delete_mode == "trash":
                        # Logical-only KOSZ attribute: physical file stays untouched on disk
                        cand.is_trashed = True
                    else:
                        if src_p.exists():
                            try:
                                _safe_unlink(src_p)
                                physical_count += 1
                            except OSError as exc:
                                logger.debug("Permanent delete unlink warning: %s", exc)
                        trash_p = eff_trash_dir / f"{cand.id}_{cand.file_name}"
                        if trash_p.exists():
                            try:
                                _safe_unlink(trash_p)
                                physical_count += 1
                            except OSError as exc:
                                logger.debug("Permanent delete trash unlink warning: %s", exc)
                        if cand.thumbnail_relative_path:
                            thumb_p = eff_thumbs_dir / cand.thumbnail_relative_path
                            if thumb_p.exists():
                                try:
                                    _safe_unlink(thumb_p)
                                except OSError as exc:
                                    logger.debug("Permanent delete thumb unlink warning: %s", exc)
                        self.db.delete(cand)

                    removed_ids.add(cand.id)
                    processed_count += 1
                    drive_processed_cnt += 1

                self.db.commit()

            if drive_processed_cnt > 0:
                processed_drives.append(f"{vol.volume_label} ({vol_root}): {drive_processed_cnt} szt.")

        self.db.commit()
        if (removed_ids or blocked_verify > 0) and not offline_items and not stopped_by_user:
            if progress_cb:
                progress_cb(step_idx, total_candidates, "Aktualizacja tagów klastrów duplikatów...", "")
            if job_id:
                ProcessManager.update_progress(
                    job_id,
                    processed_count=step_idx,
                    total_count=total_candidates,
                    message="Aktualizacja tagów klastrów duplikatów...",
                    db=self.db,
                )
            self.flag_duplicate_clusters()

        offline_delete_counts: Dict[int, int] = defaultdict(int)
        for x in offline_items:
            offline_delete_counts[x.volume_id] += 1

        offline_drives_info: List[Dict[str, Any]] = []
        seen_ovids = set(offline_vol_order) | set(offline_delete_counts.keys())
        for ovid in seen_ovids:
            ov = volumes.get(ovid)
            if ov and ov.is_online:
                continue
            del_cnt = offline_delete_counts.get(ovid, 0)
            if del_cnt <= 0:
                continue
            offline_drives_info.append(
                {
                    "volume_id": ovid,
                    "vol_label": ov.volume_label if ov else f"#{ovid}",
                    "drive_letter": ov.last_known_drive_letter if ov else "?",
                    "vsn": ov.volume_serial_number if ov else "?",
                    "count": del_cnt,
                    "delete_count": del_cnt,
                    "verify_only_count": offline_ref_counts.get(ovid, 0),
                }
            )
        # Sort offline drives with duplicates to delete from most duplicates to fewest
        offline_drives_info.sort(key=lambda d: (-d["delete_count"], -d["verify_only_count"], d["volume_id"]))

        if not offline_drives_info and offline_items:
            blocked_verify += len(offline_items)
            offline_items.clear()

        next_offline_prompt = offline_drives_info[0] if offline_drives_info else None
        return {
            "processed_count": processed_count,
            "physical_count": physical_count,
            "blocked_safety": blocked_safety,
            "blocked_verify": blocked_verify,
            "missing_on_disk": missing_on_disk,
            "offline_items": offline_items,
            "offline_ids": [x.id for x in offline_items],
            "offline_drives": offline_drives_info,
            "next_offline_prompt": next_offline_prompt,
            "processed_drives": processed_drives,
            "stopped_by_user": stopped_by_user,
        }

    def analyze_move_backups(
        self,
        source_volume_id: Optional[int] = None,
        target_volume_id: Optional[int] = None,
        include_in_vault: bool = True,
        include_green_outside: bool = True,
        include_orange_same_disk: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyzes backups across all volumes (in daf_backup/, green outside vault, orange same-disk)
        and evaluates how many selected backup files can be safely moved from source_volume_id ->
        target_volume_id/daf_backup/YYYY/MM/DD while keeping Original on a DIFFERENT physical device.
        """
        self.refresh_volumes_online_status()
        self.flag_duplicate_clusters()

        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume).order_by(WindowsVolume.id.asc())).all()}
        vol_space = self.get_volume_space_map(list(volumes.values()))

        all_items = list(
            self.db.scalars(
                select(MediaItem).where(MediaItem.is_trashed == False).order_by(MediaItem.effective_date.asc(), MediaItem.id.asc())
            ).all()
        )
        by_hash: Dict[str, List[MediaItem]] = defaultdict(list)
        bkp_stats_by_vol: Dict[int, Dict[str, int]] = {
            vid: {
                "count": 0,
                "bytes": 0,
                "in_vault_count": 0,
                "in_vault_bytes": 0,
                "green_outside_count": 0,
                "green_outside_bytes": 0,
                "orange_count": 0,
                "orange_bytes": 0,
            }
            for vid in volumes
        }

        def _classify_bkp(item_obj: MediaItem) -> Optional[str]:
            if item_obj.is_backup_copy:
                return "in_vault"
            if item_obj.duplicate_role == "backup_green":
                return "green_outside"
            if item_obj.duplicate_role == "backup_orange":
                return "orange"
            return None

        for it in all_items:
            by_hash[it.file_hash].append(it)
            bkind = _classify_bkp(it)
            if bkind and it.volume_id in bkp_stats_by_vol:
                sz = int(it.file_size_bytes or 0)
                st = bkp_stats_by_vol[it.volume_id]
                st["count"] += 1
                st["bytes"] += sz
                st[f"{bkind}_count"] += 1
                st[f"{bkind}_bytes"] += sz

        movable_rows: List[Dict[str, Any]] = []
        movable_bytes = 0
        blocked_same_device_count = 0
        blocked_same_device_bytes = 0
        already_on_target_count = 0
        filtered_out_by_type_count = 0
        movable_counts_by_kind = {"in_vault": 0, "green_outside": 0, "orange": 0}
        src_vol = volumes.get(source_volume_id) if source_volume_id else None
        dst_vol = volumes.get(target_volume_id) if target_volume_id else None

        if src_vol and dst_vol and src_vol.id != dst_vol.id:
            src_backups = [
                it for it in all_items
                if it.volume_id == src_vol.id and _classify_bkp(it) is not None
            ]
            for bkp in src_backups:
                bkind = _classify_bkp(bkp)
                if (
                    (bkind == "in_vault" and not include_in_vault)
                    or (bkind == "green_outside" and not include_green_outside)
                    or (bkind == "orange" and not include_orange_same_disk)
                ):
                    filtered_out_by_type_count += 1
                    continue

                cluster = by_hash.get(bkp.file_hash, [])
                fsize = int(bkp.file_size_bytes or 0)
                # Ensure Original / non-backup copy is NOT on dst_vol (otherwise Original & Backup would share one disk!)
                has_orig_on_target = any(
                    x.id != bkp.id and x.volume_id == dst_vol.id and (x.duplicate_role == "original" or not x.is_backup_copy)
                    for x in cluster
                )
                has_other_device_orig = any(
                    x.id != bkp.id and x.volume_id != dst_vol.id and not x.is_backup_copy
                    for x in cluster
                )
                if has_orig_on_target or not has_other_device_orig:
                    blocked_same_device_count += 1
                    blocked_same_device_bytes += fsize
                    continue

                if any(x.id != bkp.id and x.volume_id == dst_vol.id and x.is_backup_copy for x in cluster):
                    already_on_target_count += 1
                    continue

                orig = next((x for x in cluster if x.id != bkp.id and x.volume_id != dst_vol.id), cluster[0])
                orig_vol = volumes.get(orig.volume_id)
                dst_rel = f"{self.BACKUP_DIR_NAME}/{bkp.effective_date.strftime('%Y/%m/%d')}/{bkp.file_name}"
                movable_bytes += fsize
                if bkind in movable_counts_by_kind:
                    movable_counts_by_kind[bkind] += 1
                movable_rows.append(
                    {
                        "item_id": bkp.id,
                        "file_name": bkp.file_name,
                        "file_size": fsize,
                        "backup_kind": bkind,
                        "src_relative_path": bkp.relative_path,
                        "dst_relative_path": dst_rel,
                        "orig_vol_label": orig_vol.volume_label if orig_vol else "?",
                        "orig_vol_letter": orig_vol.last_known_drive_letter if orig_vol else "?",
                    }
                )

        dst_sp = vol_space.get(dst_vol.id) if dst_vol else None
        src_sp = vol_space.get(src_vol.id) if src_vol else None
        dst_free_before = int(dst_sp["free_bytes"]) if dst_sp else 0
        dst_free_after = dst_free_before - movable_bytes
        src_free_before = int(src_sp["free_bytes"]) if src_sp else 0
        src_free_after = src_free_before + movable_bytes
        space_ok = (dst_sp is None) or (dst_free_after >= self.MIN_FREE_SPACE_BUFFER_BYTES) or (movable_bytes == 0)

        return {
            "volumes": list(volumes.values()),
            "volumes_map": volumes,
            "vol_space": vol_space,
            "bkp_stats_by_vol": bkp_stats_by_vol,
            "source_volume_id": source_volume_id,
            "target_volume_id": target_volume_id,
            "include_in_vault": include_in_vault,
            "include_green_outside": include_green_outside,
            "include_orange_same_disk": include_orange_same_disk,
            "src_vol": src_vol,
            "dst_vol": dst_vol,
            "movable_rows": movable_rows,
            "movable_count": len(movable_rows),
            "movable_bytes": movable_bytes,
            "movable_counts_by_kind": movable_counts_by_kind,
            "filtered_out_by_type_count": filtered_out_by_type_count,
            "blocked_same_device_count": blocked_same_device_count,
            "blocked_same_device_bytes": blocked_same_device_bytes,
            "already_on_target_count": already_on_target_count,
            "src_free_before": src_free_before,
            "src_free_after": src_free_after,
            "dst_free_before": dst_free_before,
            "dst_free_after": dst_free_after,
            "space_ok": space_ok,
        }

    def execute_move_backups(
        self,
        source_volume_id: int,
        target_volume_id: int,
        include_in_vault: bool = True,
        include_green_outside: bool = True,
        include_orange_same_disk: bool = True,
    ) -> Dict[str, Any]:
        """Moves verified backup files physically from source_volume_id to target_volume_id/daf_backup/YYYY/MM/DD."""
        analysis = self.analyze_move_backups(
            source_volume_id,
            target_volume_id,
            include_in_vault=include_in_vault,
            include_green_outside=include_green_outside,
            include_orange_same_disk=include_orange_same_disk,
        )
        src_vol: Optional[WindowsVolume] = analysis["src_vol"]
        dst_vol: Optional[WindowsVolume] = analysis["dst_vol"]

        if not src_vol or not dst_vol or src_vol.id == dst_vol.id:
            return {"status": "error", "message": "Wybierz dwa różne urządzenia (źródłowe i docelowe).", "moved_count": 0}

        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        src_root = self.verify_and_resolve_online_volume_root(src_vol, connected_map=connected_map)
        dst_root = self.verify_and_resolve_online_volume_root(dst_vol, connected_map=connected_map)
        if src_root is None or dst_root is None:
            offline_names = [
                f"{v.volume_label} ({v.last_known_drive_letter})"
                for v, rt in ((src_vol, src_root), (dst_vol, dst_root)) if rt is None
            ]
            return {
                "status": "offline",
                "message": f"Podłącz dyski offline / zweryfikuj VSN przed przeniesieniem backupów: {', '.join(offline_names)}.",
                "moved_count": 0,
            }

        movable_rows = analysis["movable_rows"]
        if not movable_rows:
            return {"status": "empty", "message": "Brak backupów możliwych do przeniesienia między wybraną parą dysków.", "moved_count": 0}

        self.backup_mgr.create_backup("pre_move_backups")

        moved_count = 0
        moved_bytes = 0
        verification_blocks = 0
        space_stopped = False

        for row in movable_rows:
            item = self.db.get(MediaItem, row["item_id"])
            if not item:
                continue
            src_path = src_root / item.relative_path
            if not src_path.exists():
                continue

            has_space, _ = self.check_volume_free_space(str(dst_root), item.file_size_bytes)
            if not has_space:
                space_stopped = True
                break

            sub_day = item.effective_date.strftime("%Y/%m/%d")
            target_dir = dst_root / self.BACKUP_DIR_NAME / sub_day
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / item.file_name
            c = 1
            while dest_path.exists():
                dest_path = target_dir / f"{src_path.stem}_{c}{src_path.suffix}"
                c += 1

            shutil.copy2(str(src_path), str(dest_path))
            verified, _, full_sha = FastHasher.verify_files_identical_100_percent(src_path, dest_path)
            if not verified:
                try:
                    _safe_unlink(dest_path)
                except OSError as exc:
                    logger.debug("Cleanup failed backup transfer %s: %s", dest_path, exc)
                verification_blocks += 1
                continue

            try:
                _safe_unlink(src_path)
            except OSError as exc:
                logger.debug("Source unlink warning during move backup %s: %s", src_path, exc)

            item.volume_id = dst_vol.id
            item.file_name = dest_path.name
            item.relative_path = dest_path.resolve().relative_to(dst_root.resolve()).as_posix()
            item.is_backup_copy = True
            item.is_primary_duplicate = False
            item.duplicate_role = "backup_green"
            item.verified_sha256 = full_sha
            moved_count += 1
            moved_bytes += int(item.file_size_bytes or 0)

        self.db.commit()
        self.flag_duplicate_clusters()

        msg = (
            f"Przeniesiono backupów z {src_vol.volume_label} -> {dst_vol.volume_label} (do {self.BACKUP_DIR_NAME}/RRRR/MM/DD/): "
            f"{moved_count} plików ({moved_bytes / (1024*1024):.2f} MB)."
        )
        if space_stopped:
            msg += " Zatrzymano z powodu zapełnienia dysku docelowego."
        if verification_blocks > 0:
            msg += f" Zablokowanych przez weryfikację: {verification_blocks}."
        OperationLogger.log("MOVE_BACKUPS", msg, status="OK" if not space_stopped else "WARN", db=self.db)
        return {
            "status": "ok" if not space_stopped else "partial_space",
            "message": msg,
            "moved_count": moved_count,
            "moved_bytes": moved_bytes,
            "verification_blocks": verification_blocks,
        }

    def schedule_or_execute_moves(
        self,
        items: List[MediaItem],
        target_volume_id: int,
        target_subfolder: str = "",
    ) -> Dict[str, Any]:
        """
        Physically moves files between online devices or directories with live VSN verification.
        Skips items whose source or target device is currently offline or VSN-mismatched.
        """
        self.refresh_volumes_online_status()
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        target_vol = self.db.get(WindowsVolume, target_volume_id)
        if not target_vol:
            raise ValueError("Dysk docelowy nie istnieje w bazie.")

        dst_root = self.verify_and_resolve_online_volume_root(target_vol, connected_map=connected_map)
        clean_sub = target_subfolder.strip().strip("/\\")
        moved_physically = 0
        skipped_offline = 0

        if dst_root is not None:
            self.backup_mgr.create_backup("pre_move_files")

        for it in items:
            src_vol = self.db.get(WindowsVolume, it.volume_id)
            src_root = self.verify_and_resolve_online_volume_root(src_vol, connected_map=connected_map)
            can_move_physically = (
                src_root is not None
                and dst_root is not None
                and (src_root / it.relative_path).exists()
            )
            if not can_move_physically:
                skipped_offline += 1
                continue

            src_file = src_root / it.relative_path
            dst_dir = (dst_root / clean_sub) if clean_sub else dst_root
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / it.file_name

            if src_file.resolve() != dst_file.resolve():
                c = 1
                while dst_file.exists():
                    dst_file = dst_dir / f"{src_file.stem}_{c}{src_file.suffix}"
                    c += 1
                _safe_move(src_file, dst_file)

            it.volume_id = target_vol.id
            it.file_name = dst_file.name
            it.relative_path = dst_file.resolve().relative_to(dst_root.resolve()).as_posix()
            moved_physically += 1

        self.db.commit()
        OperationLogger.log(
            "FILE_MOVE",
            f"Przeniesiono fizycznie: {moved_physically}, pominięto offline: {skipped_offline} -> {target_vol.volume_label}/{clean_sub}",
            status="OK",
            db=self.db,
        )
        return {
            "moved_physically": moved_physically,
            "skipped_offline": skipped_offline,
            "target_volume_label": target_vol.volume_label,
            "target_subfolder": clean_sub,
        }

    def consolidate_group(self, group_id: int, target_volume_id: int) -> Dict[str, int]:
        """
        Moves all physical files of an album AND its child subalbums onto target online USB drive,
        preserving the subalbum folder hierarchy on disk.
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
        skipped_offline = 0
        for rel_folder, sub_items in by_subfolder.items():
            res = self.schedule_or_execute_moves(sub_items, target_volume_id, target_subfolder=rel_folder)
            moved_phys += res["moved_physically"]
            skipped_offline += res["skipped_offline"]

        return {"moved_physically": moved_phys, "skipped_offline": skipped_offline}

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

    def verify_db_disk_consistency(self, volume_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Checks all online, VSN-verified drives against non-trashed SQLite MediaItem records
        without altering any existing AI tags, embeddings, quality scores, or album assignments.
        Reports missing physical files so the user can inspect and decide whether to clean up DB rows.
        """
        self.refresh_volumes_online_status()
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        stmt = select(WindowsVolume).order_by(WindowsVolume.id.asc())
        if volume_id is not None:
            stmt = stmt.where(WindowsVolume.id == volume_id)
        volumes = list(self.db.scalars(stmt).all())

        hash_counts = dict(
            self.db.execute(
                select(MediaItem.file_hash, func.count(MediaItem.id))
                .where(MediaItem.is_trashed == False)
                .group_by(MediaItem.file_hash)
            ).all()
        )

        checked_online_volumes: List[Dict[str, Any]] = []
        offline_volumes: List[Dict[str, Any]] = []
        missing_items: List[Dict[str, Any]] = []
        valid_online_count = 0
        size_mismatch_count = 0
        missing_with_backup_count = 0
        missing_without_backup_count = 0

        for vol in volumes:
            root = self.verify_and_resolve_online_volume_root(vol, connected_map=connected_map)
            if root is None:
                db_cnt = int(
                    self.db.scalar(
                        select(func.count(MediaItem.id)).where(
                            MediaItem.volume_id == vol.id, MediaItem.is_trashed == False
                        )
                    )
                    or 0
                )
                offline_volumes.append({
                    "id": vol.id,
                    "label": vol.volume_label,
                    "drive_letter": vol.last_known_drive_letter,
                    "vsn": vol.volume_serial_number,
                    "db_count": db_cnt,
                })
                continue

            items = list(
                self.db.scalars(
                    select(MediaItem).where(
                        MediaItem.volume_id == vol.id, MediaItem.is_trashed == False
                    )
                ).all()
            )
            vol_valid = 0
            vol_missing = 0
            for it in items:
                fp = root / it.relative_path
                if not fp.exists() or not fp.is_file():
                    vol_missing += 1
                    other_copies = max(0, int(hash_counts.get(it.file_hash, 1)) - 1)
                    if other_copies > 0:
                        missing_with_backup_count += 1
                    else:
                        missing_without_backup_count += 1
                    missing_items.append({
                        "id": it.id,
                        "file_name": it.file_name,
                        "relative_path": it.relative_path,
                        "volume_id": vol.id,
                        "volume_label": vol.volume_label,
                        "drive_letter": str(root),
                        "duplicate_role": it.duplicate_role,
                        "is_backup_copy": it.is_backup_copy,
                        "file_size_bytes": int(it.file_size_bytes or 0),
                        "other_copies_count": other_copies,
                    })
                else:
                    vol_valid += 1
                    valid_online_count += 1
                    try:
                        if fp.stat().st_size != it.file_size_bytes:
                            size_mismatch_count += 1
                    except OSError:
                        size_mismatch_count += 1

            checked_online_volumes.append({
                "id": vol.id,
                "label": vol.volume_label,
                "drive_letter": str(root),
                "vsn": vol.volume_serial_number,
                "db_count": len(items),
                "valid_count": vol_valid,
                "missing_count": vol_missing,
            })

        OperationLogger.log(
            "CONSISTENCY_CHECK",
            f"Sprawdzenie spójności baza↔dysk: OK na dysku={valid_online_count}, "
            f"brakujące na dysku={len(missing_items)}, inny rozmiar={size_mismatch_count}, dyski offline={len(offline_volumes)}.",
            status="OK" if not missing_items else "WARN",
            db=self.db,
        )
        return {
            "checked_online_volumes": checked_online_volumes,
            "offline_volumes": offline_volumes,
            "valid_online_count": valid_online_count,
            "size_mismatch_count": size_mismatch_count,
            "missing_count": len(missing_items),
            "missing_with_backup_count": missing_with_backup_count,
            "missing_without_backup_count": missing_without_backup_count,
            "missing_items": missing_items,
            "missing_ids": [x["id"] for x in missing_items],
        }

    def cleanup_missing_db_items(
        self,
        item_ids: Optional[List[str]] = None,
        volume_id: Optional[int] = None,
        thumbnails_dir: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Removes DB records whose physical files are confirmed missing on VSN-verified ONLINE drives.
        Never touches records on offline or VSN-mismatched drives.
        """
        self.refresh_volumes_online_status()
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        verified_roots: Dict[int, Optional[Path]] = {
            vid: self.verify_and_resolve_online_volume_root(v, connected_map=connected_map)
            for vid, v in volumes.items()
        }

        if item_ids is None:
            check = self.verify_db_disk_consistency(volume_id=volume_id)
            item_ids = check["missing_ids"]

        if not item_ids:
            return {"removed_count": 0, "skipped_exists_or_offline": 0}

        eff_thumbs_dir = thumbnails_dir or db_module.THUMBNAILS_DIR
        self.backup_mgr.create_backup("pre_missing_cleanup")

        targets: List[MediaItem] = []
        for idx in range(0, len(item_ids), 2000):
            chunk = item_ids[idx : idx + 2000]
            stmt = select(MediaItem).where(MediaItem.id.in_(chunk), MediaItem.is_trashed == False)
            if volume_id is not None:
                stmt = stmt.where(MediaItem.volume_id == volume_id)
            targets.extend(self.db.scalars(stmt).all())

        hash_counts = dict(
            self.db.execute(
                select(MediaItem.file_hash, func.count(MediaItem.id))
                .where(MediaItem.is_trashed == False)
                .group_by(MediaItem.file_hash)
            ).all()
        )
        removed_count = 0
        skipped = 0
        for it in targets:
            root = verified_roots.get(it.volume_id)
            if root is None:
                skipped += 1
                continue
            fp = root / it.relative_path
            if fp.exists():
                skipped += 1
                continue
            vol = volumes.get(it.volume_id)
            if hash_counts.get(it.file_hash, 0) > 1:
                self._record_deleted_duplicate_history(it, vol)
                hash_counts[it.file_hash] = hash_counts[it.file_hash] - 1
            if it.thumbnail_relative_path:
                other_same_thumb = self.db.scalar(
                    select(MediaItem.id).where(
                        MediaItem.thumbnail_relative_path == it.thumbnail_relative_path,
                        MediaItem.id != it.id,
                    ).limit(1)
                )
                if not other_same_thumb:
                    tp = eff_thumbs_dir / it.thumbnail_relative_path
                    if tp.exists():
                        try:
                            _safe_unlink(tp)
                        except OSError as exc:
                            logger.debug("Cleanup missing item thumb warning: %s", exc)
            self.db.delete(it)
            removed_count += 1

        self.db.commit()
        if removed_count > 0:
            self.flag_duplicate_clusters()
        OperationLogger.log(
            "DB_DISK_CLEANUP",
            f"Usunięto z bazy {removed_count} osieroconych rekordów brakujących na dysku (pominięto: {skipped}).",
            status="OK",
            db=self.db,
        )
        return {"removed_count": removed_count, "skipped_exists_or_offline": skipped}

    def organize_ungrouped_originals(self, volume_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Groups ungrouped non-trashed Originals (group_id IS NULL) into daf_oryginal/YYYY/MM/DD/
        on their own VSN-verified device while preserving `original_relative_path` with drive label
        for search and folder filtering. Skips clusters already having a file in daf_oryginal/ on any disk.
        """
        self.refresh_volumes_online_status()
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        verified_roots: Dict[int, Optional[Path]] = {
            vid: self.verify_and_resolve_online_volume_root(v, connected_map=connected_map)
            for vid, v in volumes.items()
        }

        already_organized_hashes = set(
            self.db.scalars(
                select(MediaItem.file_hash).where(
                    MediaItem.is_trashed == False,
                    or_(
                        func.lower(MediaItem.relative_path).like(f"{self.ORIGINAL_DIR_NAME}/%"),
                        func.lower(MediaItem.relative_path).like(f"{self.ORIGINAL_DIR_NAME}\\%"),
                    ),
                )
            ).all()
        )

        stmt = (
            select(MediaItem)
            .where(
                MediaItem.is_trashed == False,
                MediaItem.group_id.is_(None),
                MediaItem.is_backup_copy == False,
                MediaItem.duplicate_role == "original",
            )
            .order_by(MediaItem.effective_date.asc(), MediaItem.id.asc())
        )
        if volume_id is not None:
            stmt = stmt.where(MediaItem.volume_id == volume_id)
        candidates = list(self.db.scalars(stmt).all())

        moved_count = 0
        skipped_already_organized = 0
        skipped_offline = 0
        missing_on_disk = 0
        errors = 0

        if candidates:
            self.backup_mgr.create_backup("pre_organize_originals")

        for it in candidates:
            rel_norm = (it.relative_path or "").replace("\\", "/").lstrip("/")
            if rel_norm.lower().startswith(f"{self.ORIGINAL_DIR_NAME}/") or it.file_hash in already_organized_hashes:
                skipped_already_organized += 1
                continue

            vol = volumes.get(it.volume_id)
            root = verified_roots.get(it.volume_id)
            if not vol or root is None:
                skipped_offline += 1
                continue

            src_path = root / it.relative_path
            if not src_path.exists() or not src_path.is_file():
                missing_on_disk += 1
                continue

            sub_day = it.effective_date.strftime("%Y/%m/%d")
            dst_dir = root / self.ORIGINAL_DIR_NAME / sub_day
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / it.file_name
            if src_path.resolve() != dst_path.resolve():
                c = 1
                while dst_path.exists():
                    dst_path = dst_dir / f"{src_path.stem}_{c}{src_path.suffix}"
                    c += 1
                try:
                    _safe_move(src_path, dst_path)
                except OSError as exc:
                    logger.debug("Organize original move error for %s: %s", src_path, exc)
                    errors += 1
                    continue

            if not it.original_relative_path:
                it.original_relative_path = f"{vol.volume_label}/{rel_norm}"
            it.file_name = dst_path.name
            it.relative_path = dst_path.resolve().relative_to(root.resolve()).as_posix()
            already_organized_hashes.add(it.file_hash)
            moved_count += 1
            if moved_count % 200 == 0:
                self.db.commit()

        self.db.commit()
        msg = (
            f"Uporządkowano oryginałów (bez albumu) do {self.ORIGINAL_DIR_NAME}/RRRR/MM/DD/: {moved_count} szt. "
            f"| Już w {self.ORIGINAL_DIR_NAME}/: {skipped_already_organized} | Offline: {skipped_offline} | Brak na dysku: {missing_on_disk}."
        )
        OperationLogger.log("ORGANIZE_ORIGINALS", msg, status="OK" if errors == 0 else "WARN", db=self.db)
        return {
            "moved_count": moved_count,
            "skipped_already_organized": skipped_already_organized,
            "skipped_offline": skipped_offline,
            "missing_on_disk": missing_on_disk,
            "errors": errors,
            "message": msg,
        }

    def export_physical_album(
        self,
        items: List[MediaItem],
        target_volume_id: int,
        album_name: str,
    ) -> Dict[str, Any]:
        """
        Copies selected/album photos physically into daf_albums/<AlbumName>/ on a chosen
        VSN-verified online device without registering copies in media_items.
        """
        self.refresh_volumes_online_status()
        connected_map = WindowsVolumeInspector.get_connected_vsn_map()
        target_vol = self.db.get(WindowsVolume, target_volume_id)
        if not target_vol:
            return {"status": "error", "message": "Wybrany dysk docelowy nie istnieje.", "copied_count": 0}

        dst_root = self.verify_and_resolve_online_volume_root(target_vol, connected_map=connected_map)
        if dst_root is None:
            return {
                "status": "offline",
                "message": f"Dysk docelowy {target_vol.volume_label} ({target_vol.last_known_drive_letter}) jest offline lub niezgodny VSN.",
                "copied_count": 0,
            }

        clean_name = re.sub(r'[<>:"/\\|?*]+', "_", (album_name or "").strip()).strip(". ")
        if not clean_name:
            clean_name = f"Album_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        export_dir = dst_root / self.ALBUMS_EXPORT_DIR_NAME / clean_name
        export_dir.mkdir(parents=True, exist_ok=True)

        volumes = {v.id: v for v in self.db.scalars(select(WindowsVolume)).all()}
        verified_roots: Dict[int, Optional[Path]] = {
            vid: self.verify_and_resolve_online_volume_root(v, connected_map=connected_map)
            for vid, v in volumes.items()
        }

        copied_count = 0
        copied_bytes = 0
        skipped_offline = 0
        space_stopped = False

        for it in items:
            src_root = verified_roots.get(it.volume_id)
            if src_root is None:
                skipped_offline += 1
                continue
            src_file = src_root / it.relative_path
            if not src_file.exists() or not src_file.is_file():
                skipped_offline += 1
                continue

            has_space, _ = self.check_volume_free_space(str(dst_root), int(it.file_size_bytes or 0))
            if not has_space:
                space_stopped = True
                break

            dst_file = export_dir / it.file_name
            c = 1
            while dst_file.exists():
                dst_file = export_dir / f"{src_file.stem}_{c}{src_file.suffix}"
                c += 1
            shutil.copy2(str(src_file), str(dst_file))
            copied_count += 1
            copied_bytes += int(it.file_size_bytes or 0)

        msg = (
            f"Wyeksportowano fizycznie {copied_count} plików albumu ({copied_bytes / (1024*1024):.2f} MB) "
            f"do {export_dir} (pomijane przez skaner jako katalog eksportu)."
        )
        if skipped_offline > 0:
            msg += f" Pominięto offline/brak pliku: {skipped_offline}."
        if space_stopped:
            msg += " Zatrzymano z powodu zapełnienia dysku docelowego."
        OperationLogger.log("EXPORT_PHYSICAL_ALBUM", msg, status="OK" if not space_stopped else "WARN", db=self.db)
        return {
            "status": "ok" if not space_stopped else "partial_space",
            "copied_count": copied_count,
            "copied_bytes": copied_bytes,
            "skipped_offline": skipped_offline,
            "export_path": str(export_dir),
            "message": msg,
        }

    def verify_deleted_duplicates_integrity(self) -> Dict[str, Any]:
        """
        Verifies across media_vault.db and deleted_duplicate_history / pre-delete snapshots
        that every permanently deleted duplicate retains an active Original with matching file_hash.
        """
        active_count = int(
            self.db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_trashed == False)) or 0
        )
        deleted_count = int(self.db.scalar(select(func.count(DeletedDuplicateHistory.id))) or 0)

        active_hashes_subq = select(MediaItem.file_hash).where(MediaItem.is_trashed == False)
        orig_hashes_subq = select(MediaItem.file_hash).where(
            MediaItem.is_trashed == False,
            or_(MediaItem.duplicate_role == "original", MediaItem.is_primary_duplicate == True),
        )
        with_active_copy_count = int(
            self.db.scalar(
                select(func.count(DeletedDuplicateHistory.id)).where(
                    DeletedDuplicateHistory.file_hash.in_(active_hashes_subq)
                )
            )
            or 0
        )
        with_active_original_count = int(
            self.db.scalar(
                select(func.count(DeletedDuplicateHistory.id)).where(
                    DeletedDuplicateHistory.file_hash.in_(orig_hashes_subq)
                )
            )
            or 0
        )

        # Inspect largest pre-deletion snapshot if present in data/backups
        snapshot_name: Optional[str] = None
        snapshot_total: Optional[int] = None
        if db_module.BACKUP_DIR.exists():
            db_candidates = sorted(
                db_module.BACKUP_DIR.rglob("*.db"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            for snap in db_candidates[:3]:
                try:
                    sc = sqlite3.connect(snap.as_posix(), timeout=5.0)
                    cnt_row = sc.execute("SELECT COUNT(*) FROM media_items WHERE is_trashed = 0").fetchone()
                    sc.close()
                    if cnt_row and int(cnt_row[0]) >= active_count + deleted_count:
                        snapshot_name = snap.name
                        snapshot_total = int(cnt_row[0])
                        break
                except Exception:
                    continue

        pre_delete_total = snapshot_total if snapshot_total is not None else (active_count + deleted_count)
        pct = (with_active_original_count / deleted_count * 100.0) if deleted_count > 0 else 100.0
        missing_orig_count = max(0, deleted_count - with_active_original_count)

        def _fmt(n: int) -> str:
            return f"{n:,}".replace(",", " ")

        report_text = (
            f"Wykonałem zapytanie łączące bieżącą bazę {db_module.DB_PATH.name} ({_fmt(active_count)} aktywnych plików) "
            f"z historią/kopią sprzed usuwania ({_fmt(pre_delete_total)} plików): "
            f"Liczba usuniętych duplikatów: {_fmt(deleted_count)} | "
            f"Liczba usuniętych duplikatów, które mają w bieżącej bazie zachowany aktywny Oryginał o tym samym file_hash: "
            f"{_fmt(with_active_original_count)} z {_fmt(deleted_count)} ({pct:.1f}%)."
        )
        return {
            "active_count": active_count,
            "pre_delete_total": pre_delete_total,
            "snapshot_name": snapshot_name,
            "deleted_count": deleted_count,
            "with_active_copy_count": with_active_copy_count,
            "with_active_original_count": with_active_original_count,
            "missing_orig_count": missing_orig_count,
            "pct": pct,
            "report_text": report_text,
        }
