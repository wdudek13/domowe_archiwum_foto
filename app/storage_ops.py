from pathlib import Path
import shutil
from typing import List, Set
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from app.backup import BackupManager
from app.models import MediaItem, WindowsVolume, MediaGroup


class StorageAutomationService:
    def __init__(self, db: Session):
        self.db = db
        self.backup_mgr = BackupManager()

    def flag_duplicate_clusters(self) -> int:
        """
        Marks primary vs duplicate records across 100k+ items in a single SQL window query (<50ms).
        """
        sql = text("""
            UPDATE media_items
            SET is_primary_duplicate = CASE
                WHEN id IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (PARTITION BY file_hash ORDER BY effective_date ASC, id ASC) AS rn
                        FROM media_items
                    ) WHERE rn = 1
                ) THEN 1 ELSE 0 END;
        """)
        self.db.execute(sql)
        self.db.commit()
        dup_count = self.db.scalar(select(text("COUNT(*)")).select_from(MediaItem).where(MediaItem.is_primary_duplicate == False)) or 0
        return int(dup_count)

    def get_group_subtree_ids(self, root_group_id: int) -> List[int]:
        all_groups = self.db.execute(select(MediaGroup.id, MediaGroup.parent_id)).all()
        children_map = {}
        for gid, pid in all_groups:
            children_map.setdefault(pid, []).append(gid)

        result = []
        stack = [root_group_id]
        visited: Set[int] = set()
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                result.append(curr)
                stack.extend(children_map.get(curr, []))
        return result

    def consolidate_group(self, group_id: int, target_volume_id: int) -> int:
        """
        Moves all physical files of a group AND its child subgroups onto target USB drive,
        preserving the subgroup folder hierarchy on disk.
        """
        self.backup_mgr.create_backup(f"pre_consolidate_grp_{group_id}")
        target_vol = self.db.get(WindowsVolume, target_volume_id)
        if not target_vol or not Path(target_vol.last_known_drive_letter).exists():
            raise RuntimeError("Dysk docelowy jest niedostepny (Offline).")

        subtree_ids = self.get_group_subtree_ids(group_id)
        groups_by_id = {g.id: g for g in self.db.scalars(select(MediaGroup)).all()}

        def get_group_rel_folder(gid: int) -> Path:
            parts = []
            curr = groups_by_id.get(gid)
            depth = 0
            while curr and depth < 10:
                parts.append(curr.name)
                curr = groups_by_id.get(curr.parent_id) if curr.parent_id else None
                depth += 1
            return Path(*reversed(parts)) if parts else Path("Grouped")

        target_root = Path(target_vol.last_known_drive_letter)
        items = self.db.scalars(select(MediaItem).where(MediaItem.group_id.in_(subtree_ids))).all()
        moved = 0

        for it in items:
            src_vol = self.db.get(WindowsVolume, it.volume_id)
            if not src_vol:
                continue
            src_path = Path(src_vol.last_known_drive_letter) / it.relative_path
            if not src_path.exists():
                continue

            sub_dir = target_root / "HydraGroups" / get_group_rel_folder(it.group_id)
            sub_dir.mkdir(parents=True, exist_ok=True)
            dest_path = sub_dir / it.file_name

            if src_path.resolve() == dest_path.resolve():
                continue

            c = 1
            while dest_path.exists():
                dest_path = sub_dir / f"{src_path.stem}_{c}{src_path.suffix}"
                c += 1

            shutil.move(str(src_path), str(dest_path))
            it.volume_id = target_vol.id
            it.relative_path = dest_path.relative_to(target_root).as_posix()
            moved += 1

        self.db.commit()
        return moved

    def auto_group_items_by_date(self, items: List[MediaItem]) -> int:
        """Automatically builds Year -> Year-Month group hierarchy for a list of items."""
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
        """Mirrors physical USB directory structure into hierarchical MediaGroups."""
        existing = {(g.name, g.parent_id): g for g in self.db.scalars(select(MediaGroup)).all()}
        updated = 0

        for it in items:
            parts = [p for p in Path(it.relative_path).parent.parts if p not in ("", ".", "/")]
            if not parts:
                parts = ["Root_Folder"]

            parent_id = None
            target_grp = None
            for part in parts[:4]:  # Max 4 levels deep
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
