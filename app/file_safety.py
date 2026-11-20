import datetime
import errno
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import threading
from typing import Any, List, Optional, Tuple

import app.database as db_module

logger = logging.getLogger("FileSafety")

QUARANTINE_DIR_NAME = "daf_kwarantanna"
_AUDIT_LOCK = threading.Lock()


def get_audit_log_path() -> Path:
    """
    Returns the path to the append-only JSONL file operations audit log.
    Uses `db_module.OPERATIONS_LOG_FILE.parent` so production writes to
    `data/logs/file_operations_audit.jsonl` while unit tests patching
    `OPERATIONS_LOG_FILE` automatically stay isolated inside their temp directory.
    """
    log_dir = Path(db_module.OPERATIONS_LOG_FILE).parent
    return log_dir / "file_operations_audit.jsonl"


def log_file_operation(
    op: str,
    src_path: Optional[Path],
    dst_path: Optional[Path] = None,
    *,
    size_bytes: Optional[int] = None,
    file_hash: str = "",
    vsn: str = "",
    job_id: Optional[int] = None,
    reason: str = "",
    status: str = "OK",
) -> None:
    """Appends a single JSONL audit record to `file_operations_audit.jsonl` (independent of SQLite)."""
    entry = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "op": op,
        "src_path": str(src_path) if src_path is not None else "",
        "dst_path": str(dst_path) if dst_path is not None else "",
        "size_bytes": size_bytes,
        "file_hash": file_hash or "",
        "vsn": vsn or "",
        "job_id": job_id,
        "reason": reason or "",
        "status": status,
    }
    try:
        audit_path = get_audit_log_path()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _AUDIT_LOCK:
            with open(audit_path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except OSError as exc:
        logger.debug("Could not append to file_operations_audit.jsonl: %s", exc)


def is_path_in_daf_backup(path: Path, volume_root: Optional[Path] = None) -> bool:
    """
    Returns True if `path` resides inside a protected media backup folder:
    - `daf_backup/` at any depth (unique DAF system folder), OR
    - top-level `backup/` directly under the volume/drive root (matching scanner.py & is_item_in_daf_backup,
      without falsely blocking unrelated user subfolders like `E:\\DCIM\\backup\\...`).
    Excludes SQLite `db_backups` rotation files and uncommitted `.daf_tmp` / `.daf_move_tmp` staging files.
    """
    p_obj = Path(path)
    lname = p_obj.name.lower()
    if lname.endswith(".daf_tmp") or lname.endswith(".daf_move_tmp"):
        return False

    rel_parts: List[str] = []
    if volume_root is not None:
        try:
            rel_parts = [p.lower() for p in p_obj.resolve().relative_to(Path(volume_root).resolve()).parts]
        except ValueError:
            rel_parts = []
    if not rel_parts:
        anchor = p_obj.anchor
        rel_parts = [p.lower() for p in p_obj.parts if p not in ("", anchor, "\\", "/")]

    if "db_backups" in rel_parts:
        return False
    if "daf_backup" in rel_parts:
        return True
    return bool(rel_parts and rel_parts[0] == "backup")


def is_item_in_daf_backup(item: Any) -> bool:
    """Returns True if a MediaItem represents a protected backup file inside `daf_backup/` or top-level `backup/`."""
    if getattr(item, "is_backup_copy", False):
        return True
    rel = (getattr(item, "relative_path", "") or "").replace("\\", "/").lower()
    parts = [p for p in rel.split("/") if p]
    return ("daf_backup" in parts) or (bool(parts) and parts[0] == "backup")


def is_path_in_quarantine(path: Path, volume_root: Optional[Path] = None) -> bool:
    """Returns True if `path` resides inside `daf_kwarantanna/`."""
    p_obj = Path(path)
    rel_parts: List[str] = []
    if volume_root is not None:
        try:
            rel_parts = [p.lower() for p in p_obj.resolve().relative_to(Path(volume_root).resolve()).parts]
        except ValueError:
            rel_parts = []
    if not rel_parts:
        anchor = p_obj.anchor
        rel_parts = [p.lower() for p in p_obj.parts if p not in ("", anchor, "\\", "/")]
    return QUARANTINE_DIR_NAME in rel_parts


def safe_resolve_volume_path(vol_root: Path, relative_path: str) -> Path:
    """
    H-06: Resolves `relative_path` strictly inside `vol_root`, blocking absolute paths,
    Windows drive prefixes, `..` traversal, and symlinks escaping `vol_root`.
    """
    rel_str = (relative_path or "").strip()
    if (
        not rel_str
        or os.path.isabs(rel_str)
        or ":" in Path(rel_str).drive
        or rel_str.startswith(("/", "\\"))
        or (len(rel_str) >= 2 and rel_str[1] == ":")
    ):
        raise ValueError(f"Niedozwolona ścieżka względna: {relative_path}")
    rel_parts = Path(rel_str).parts
    if not rel_parts or ".." in rel_parts:
        raise ValueError(f"Wykryto '..' w ścieżce względnej: {relative_path}")
    root_res = Path(vol_root).resolve()
    full_res = (root_res / rel_str).resolve()
    if full_res == root_res or not full_res.is_relative_to(root_res):
        raise ValueError(f"Ścieżka {full_res} wykracza poza korzeń wolumenu {root_res}")
    return full_res


def is_read_only(path: Path) -> bool:
    """Returns True if `path` exists and has the Read-Only attribute (lacks write bit)."""
    try:
        st = os.stat(path)
        return not bool(st.st_mode & stat.S_IWRITE)
    except OSError:
        return False


def assert_not_read_only(path: Path) -> None:
    """
    C-03: Hard-blocks mutating/deleting Read-Only user files.
    NEVER calls os.chmod(..., stat.S_IWRITE) to strip user-set Read-Only protection.
    """
    if path.exists() and is_read_only(path):
        raise PermissionError(f"Plik chroniony atrybutem Tylko do odczytu (Read-Only): {path}")


class FileSafetyGuard:
    """
    Single central choke point for all physical file mutations on user media volumes.
    Enforces Zero-Trust invariants:
    - daf_backup/ & top-level backup/ immutability
    - Read-Only attribute respect (never stripped)
    - No overwrite of existing destination files (H-03)
    - Path containment within volume root (H-06)
    - Same-volume atomic quarantine (`daf_kwarantanna/`) instead of irreversible unlink (C-03)
    - Last-copy hard invariant: hard unlink forbidden without live 100%-verified online copy (C-04)
    - Append-only JSONL audit trail (L-01)
    """

    @staticmethod
    def quarantine_file(
        src_path: Path,
        vol_root: Path,
        *,
        job_id: Optional[int] = None,
        reason: str = "",
        file_hash: str = "",
        vsn: str = "",
    ) -> Path:
        """
        C-03 & L-01: Moves `src_path` atomically via `os.rename` on the same volume into
        `<vol_root>/daf_kwarantanna/<YYYYMMDD>/<relative_path>` and records entries in
        `<vol_root>/daf_kwarantanna/manifest.jsonl` and `data/logs/file_operations_audit.jsonl`.
        """
        src_obj = Path(src_path)
        root_res = Path(vol_root).resolve()

        if is_path_in_daf_backup(src_obj, volume_root=root_res):
            log_file_operation(
                "quarantine",
                src_obj,
                file_hash=file_hash,
                vsn=vsn,
                job_id=job_id,
                reason=reason or "daf_backup_blocked",
                status="BLOCKED",
            )
            raise PermissionError(
                f"Ochrona daf_backup: zabronione usuwanie istniejącego pliku w katalogu backupu: {src_obj}"
            )

        if is_path_in_quarantine(src_obj, volume_root=root_res):
            log_file_operation(
                "quarantine",
                src_obj,
                file_hash=file_hash,
                vsn=vsn,
                job_id=job_id,
                reason=reason or "already_in_quarantine",
                status="BLOCKED",
            )
            raise PermissionError(f"Plik znajduje się już w katalogu kwarantanny: {src_obj}")

        assert_not_read_only(src_obj)

        src_res = src_obj.resolve()
        if src_res == root_res or not src_res.is_relative_to(root_res):
            log_file_operation(
                "quarantine",
                src_obj,
                file_hash=file_hash,
                vsn=vsn,
                job_id=job_id,
                reason="outside_volume_root",
                status="BLOCKED",
            )
            raise ValueError(f"Ścieżka {src_res} wykracza poza korzeń wolumenu {root_res}")

        rel = src_res.relative_to(root_res)
        size_bytes: Optional[int] = None
        try:
            size_bytes = src_res.stat().st_size
        except OSError:
            pass

        date_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
        quarantine_root = root_res / QUARANTINE_DIR_NAME
        dst_dir = quarantine_root / date_str / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / rel.name
        c = 1
        while dst_path.exists():
            dst_path = dst_dir / f"{rel.stem}_{c}{rel.suffix}"
            c += 1

        if dst_path.exists():
            raise FileExistsError(f"Ochrona plików: plik kwarantanny już istnieje: {dst_path}")

        os.rename(str(src_res), str(dst_path))

        manifest_entry = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "original_path": str(src_res),
            "relative_path": rel.as_posix(),
            "quarantine_path": str(dst_path),
            "size_bytes": size_bytes,
            "file_hash": file_hash or "",
            "vsn": vsn or "",
            "job_id": job_id,
            "reason": reason or "",
        }
        try:
            manifest_path = quarantine_root / "manifest.jsonl"
            with _AUDIT_LOCK:
                with open(manifest_path, "a", encoding="utf-8") as mfh:
                    mfh.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.debug("Could not write quarantine manifest entry: %s", exc)

        log_file_operation(
            "quarantine",
            src_res,
            dst_path,
            size_bytes=size_bytes,
            file_hash=file_hash,
            vsn=vsn,
            job_id=job_id,
            reason=reason,
            status="OK",
        )
        return dst_path

    @staticmethod
    def quarantine_or_unlink(
        path: Path,
        volume_root: Optional[Path] = None,
        *,
        verified_copy_path: Optional[Path] = None,
        allow_hard_unlink: bool = False,
        job_id: Optional[int] = None,
        reason: str = "",
        file_hash: str = "",
        vsn: str = "",
    ) -> Tuple[str, Optional[Path]]:
        """
        C-03 & C-04:
        - For user media files (`volume_root` provided): defaults to same-volume atomic quarantine
          (`daf_kwarantanna/`). Hard `unlink()` is strictly blocked unless `allow_hard_unlink=True`
          AND `verified_copy_path` passes live 100% SHA-256 verification right now.
        - For internal app artifacts (`volume_root is None`, e.g. thumbnails or `data/kosz` staging):
          unlinks only if not in `daf_backup/` and not Read-Only (never strips Read-Only).
        """
        p_obj = Path(path)
        if is_path_in_daf_backup(p_obj, volume_root=volume_root):
            log_file_operation(
                "unlink",
                p_obj,
                file_hash=file_hash,
                vsn=vsn,
                job_id=job_id,
                reason=reason or "daf_backup_blocked",
                status="BLOCKED",
            )
            raise PermissionError(
                f"Ochrona daf_backup: zabronione usuwanie istniejącego pliku w katalogu backupu: {p_obj}"
            )

        assert_not_read_only(p_obj)

        if volume_root is not None:
            if allow_hard_unlink:
                from app.scanner import FastHasher

                if verified_copy_path is None or not Path(verified_copy_path).exists():
                    log_file_operation(
                        "unlink",
                        p_obj,
                        file_hash=file_hash,
                        vsn=vsn,
                        job_id=job_id,
                        reason="missing_verified_online_copy",
                        status="BLOCKED",
                    )
                    raise PermissionError(
                        f"Ochrona ostatniej kopii: fizyczny unlink zabroniony bez zweryfikowanej online drugiej kopii: {p_obj}"
                    )
                ok, verify_reason, full_sha = FastHasher.verify_files_identical_100_percent(
                    p_obj, Path(verified_copy_path)
                )
                if not ok:
                    log_file_operation(
                        "unlink",
                        p_obj,
                        Path(verified_copy_path),
                        file_hash=file_hash,
                        vsn=vsn,
                        job_id=job_id,
                        reason=f"copy_verify_failed:{verify_reason}",
                        status="BLOCKED",
                    )
                    raise PermissionError(
                        f"Ochrona ostatniej kopii: weryfikacja 100% drugiej kopii nie powiodła się ({verify_reason}): {p_obj}"
                    )
                size_bytes = p_obj.stat().st_size if p_obj.exists() else None
                p_obj.unlink()
                log_file_operation(
                    "unlink",
                    p_obj,
                    Path(verified_copy_path),
                    size_bytes=size_bytes,
                    file_hash=full_sha or file_hash,
                    vsn=vsn,
                    job_id=job_id,
                    reason=reason,
                    status="OK",
                )
                return "unlinked", None

            q_dst = FileSafetyGuard.quarantine_file(
                p_obj,
                volume_root,
                job_id=job_id,
                reason=reason,
                file_hash=file_hash,
                vsn=vsn,
            )
            return "quarantined", q_dst

        size_bytes = p_obj.stat().st_size if p_obj.exists() else None
        p_obj.unlink()
        log_file_operation(
            "unlink",
            p_obj,
            size_bytes=size_bytes,
            file_hash=file_hash,
            vsn=vsn,
            job_id=job_id,
            reason=reason,
            status="OK",
        )
        return "unlinked", None

    @staticmethod
    def remove_verified_move_source(
        src_path: Path,
        verified_dst_path: Path,
        src_volume_root: Optional[Path] = None,
        *,
        verified_sha256: str,
        job_id: Optional[int] = None,
        reason: str = "verified_move",
        vsn: str = "",
    ) -> None:
        """
        Removes `src_path` ONLY after a 100%-SHA-256-verified copy has just been finalized at `verified_dst_path`.
        Enforces daf_backup immutability, Read-Only respect, distinct path, and matching size.
        """
        src_obj = Path(src_path)
        dst_obj = Path(verified_dst_path)

        if is_path_in_daf_backup(src_obj, volume_root=src_volume_root):
            raise PermissionError(
                f"Ochrona daf_backup: zabronione usuwanie istniejącego pliku w katalogu backupu: {src_obj}"
            )
        assert_not_read_only(src_obj)

        if not verified_sha256 or not dst_obj.exists() or not dst_obj.is_file():
            raise PermissionError(
                f"Ochrona przenoszenia: brak zweryfikowanego pliku docelowego dla {src_obj} -> {dst_obj}"
            )
        if src_obj.resolve() == dst_obj.resolve():
            raise PermissionError(f"Ochrona przenoszenia: źródło i cel wskazują ten sam plik: {src_obj}")
        src_sz = src_obj.stat().st_size
        dst_sz = dst_obj.stat().st_size
        if src_sz != dst_sz:
            raise OSError(
                f"Ochrona przenoszenia: niezgodny rozmiar źródła ({src_sz}) i celu ({dst_sz}) dla {src_obj}"
            )

        src_obj.unlink()
        log_file_operation(
            "unlink_move_source",
            src_obj,
            dst_obj,
            size_bytes=src_sz,
            file_hash=verified_sha256,
            vsn=vsn,
            job_id=job_id,
            reason=reason,
            status="OK",
        )

    @staticmethod
    def _cross_drive_verified_move(
        src: Path,
        dst: Path,
        src_volume_root: Optional[Path],
        *,
        job_id: Optional[int] = None,
        file_hash: str = "",
        vsn: str = "",
    ) -> None:
        from app.scanner import FastHasher

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_name(f"{dst.name}.daf_move_tmp")
        if tmp_dst.exists():
            try:
                tmp_dst.unlink()
            except OSError:
                pass
        shutil.copy2(str(src), str(tmp_dst))
        verified, reason, full_sha = FastHasher.verify_files_identical_100_percent(src, tmp_dst)
        if not verified:
            try:
                tmp_dst.unlink()
            except OSError:
                pass
            raise OSError(
                f"Weryfikacja 100% po przeniesieniu między dyskami nie powiodła się ({src} -> {dst}): {reason}"
            )
        if dst.exists():
            try:
                tmp_dst.unlink()
            except OSError:
                pass
            raise FileExistsError(
                f"Ochrona plików: plik docelowy już istnieje i nie może zostać nadpisany: {dst}"
            )
        # os.rename NEVER overwrites an existing destination file on Windows (unlike Path.replace)
        os.rename(str(tmp_dst), str(dst))
        FileSafetyGuard.remove_verified_move_source(
            src,
            dst,
            src_volume_root=src_volume_root,
            verified_sha256=full_sha or file_hash or "verified",
            job_id=job_id,
            reason="cross_drive_move",
            vsn=vsn,
        )
        log_file_operation(
            "move",
            src,
            dst,
            size_bytes=dst.stat().st_size if dst.exists() else None,
            file_hash=full_sha or file_hash,
            vsn=vsn,
            job_id=job_id,
            reason="cross_drive_move",
            status="OK",
        )

    @staticmethod
    def safe_move(
        src: Path,
        dst: Path,
        src_volume_root: Optional[Path] = None,
        dst_volume_root: Optional[Path] = None,
        *,
        job_id: Optional[int] = None,
        file_hash: str = "",
        vsn: str = "",
    ) -> None:
        """
        H-03, C-03, L-01: Moves a file safely without EVER overwriting an existing destination
        and without EVER stripping Read-Only protection:
        - Hard-blocks mutating/deleting existing files in `daf_backup/` or top-level `backup/`.
        - Hard-blocks overwriting ANY existing destination file (`FileExistsError`).
        - Hard-blocks moving Read-Only files (`PermissionError`).
        - Cross-drive moves use transactional copy -> 100% SHA-256 verification -> non-overwriting `os.rename` -> verified source unlink.
        - Same-drive moves use non-overwriting `os.rename` (falling back to verified copy on EXDEV).
        """
        src_obj = Path(src)
        dst_obj = Path(dst)

        if is_path_in_daf_backup(src_obj, volume_root=src_volume_root):
            log_file_operation("move", src_obj, dst_obj, file_hash=file_hash, vsn=vsn, job_id=job_id, reason="src_in_daf_backup", status="BLOCKED")
            raise PermissionError(
                f"Ochrona daf_backup: zabronione przenoszenie/usuwanie pliku z katalogu backupu: {src_obj}"
            )
        if is_path_in_daf_backup(dst_obj, volume_root=dst_volume_root) and dst_obj.exists():
            log_file_operation("move", src_obj, dst_obj, file_hash=file_hash, vsn=vsn, job_id=job_id, reason="dst_in_daf_backup", status="BLOCKED")
            raise PermissionError(
                f"Ochrona daf_backup: zabronione nadpisywanie istniejącego pliku w katalogu backupu: {dst_obj}"
            )
        if dst_obj.exists():
            log_file_operation("move", src_obj, dst_obj, file_hash=file_hash, vsn=vsn, job_id=job_id, reason="dst_exists", status="BLOCKED")
            raise FileExistsError(
                f"Ochrona plików: plik docelowy już istnieje i nie może zostać nadpisany: {dst_obj}"
            )

        assert_not_read_only(src_obj)

        src_drive = str(src_obj.resolve().drive).lower() if src_obj.exists() else str(src_obj.drive).lower()
        dst_drive = str(dst_obj.parent.resolve().drive).lower() if dst_obj.parent.exists() else str(dst_obj.drive).lower()

        if src_drive and dst_drive and src_drive != dst_drive and src_obj.exists() and src_obj.is_file():
            FileSafetyGuard._cross_drive_verified_move(
                src_obj, dst_obj, src_volume_root, job_id=job_id, file_hash=file_hash, vsn=vsn
            )
            return

        dst_obj.parent.mkdir(parents=True, exist_ok=True)
        size_bytes = src_obj.stat().st_size if src_obj.exists() else None
        try:
            os.rename(str(src_obj), str(dst_obj))
        except OSError as exc:
            if exc.errno == errno.EXDEV and src_obj.exists() and src_obj.is_file():
                FileSafetyGuard._cross_drive_verified_move(
                    src_obj, dst_obj, src_volume_root, job_id=job_id, file_hash=file_hash, vsn=vsn
                )
                return
            raise

        log_file_operation(
            "move",
            src_obj,
            dst_obj,
            size_bytes=size_bytes,
            file_hash=file_hash,
            vsn=vsn,
            job_id=job_id,
            reason="same_drive_move",
            status="OK",
        )

    @staticmethod
    def safe_copy_new_to_daf_backup(
        src_path: Path,
        dest_path: Path,
        *,
        job_id: Optional[int] = None,
        vsn: str = "",
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Append-only verified write into `daf_backup/`:
        1) Refuses if `dest_path` already exists (never overwrites an existing backup file).
        2) Copies to a `.daf_tmp` staging filename, runs 100% SHA-256 + media verification.
        3) Atomically renames `.daf_tmp` -> `dest_path` using non-overwriting `os.rename` on 100% pass.
        """
        from app.scanner import FastHasher

        src_obj = Path(src_path)
        dst_obj = Path(dest_path)
        if dst_obj.exists():
            raise PermissionError(
                f"Ochrona daf_backup: plik docelowy już istnieje i nie może zostać nadpisany: {dst_obj}"
            )
        tmp_path = dst_obj.with_name(f"{dst_obj.name}.daf_tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        shutil.copy2(str(src_obj), str(tmp_path))
        verified, reason, full_sha = FastHasher.verify_files_identical_100_percent(src_obj, tmp_path)
        if not verified:
            try:
                tmp_path.unlink()
            except OSError as exc:
                logger.debug("Cleanup failed staging backup copy %s: %s", tmp_path, exc)
            return False, reason, None
        if dst_obj.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise PermissionError(
                f"Ochrona daf_backup: wykryto istniejący plik podczas finalizacji: {dst_obj}"
            )
        os.rename(str(tmp_path), str(dst_obj))
        log_file_operation(
            "copy_backup",
            src_obj,
            dst_obj,
            size_bytes=dst_obj.stat().st_size if dst_obj.exists() else None,
            file_hash=full_sha or "",
            vsn=vsn,
            job_id=job_id,
            reason="daf_backup_append",
            status="OK",
        )
        return True, reason, full_sha

    @staticmethod
    def safe_copy_export(
        src_path: Path,
        dest_path: Path,
        *,
        job_id: Optional[int] = None,
        file_hash: str = "",
        vsn: str = "",
    ) -> None:
        """Non-overwriting copy for physical album exports (`daf_albums/`)."""
        src_obj = Path(src_path)
        dst_obj = Path(dest_path)
        if dst_obj.exists():
            raise FileExistsError(
                f"Ochrona plików: plik eksportu już istnieje i nie może zostać nadpisany: {dst_obj}"
            )
        dst_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_obj.with_name(f"{dst_obj.name}.daf_tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        shutil.copy2(str(src_obj), str(tmp_path))
        if dst_obj.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise FileExistsError(
                f"Ochrona plików: wykryto istniejący plik podczas eksportu: {dst_obj}"
            )
        os.rename(str(tmp_path), str(dst_obj))
        log_file_operation(
            "copy_export",
            src_obj,
            dst_obj,
            size_bytes=dst_obj.stat().st_size if dst_obj.exists() else None,
            file_hash=file_hash,
            vsn=vsn,
            job_id=job_id,
            reason="album_export",
            status="OK",
        )


def quarantine_file(
    src_path: Path,
    vol_root: Path,
    *,
    job_id: Optional[int] = None,
    reason: str = "",
    file_hash: str = "",
    vsn: str = "",
) -> Path:
    return FileSafetyGuard.quarantine_file(
        src_path,
        vol_root,
        job_id=job_id,
        reason=reason,
        file_hash=file_hash,
        vsn=vsn,
    )


def _safe_unlink(
    path: Path,
    volume_root: Optional[Path] = None,
    *,
    verified_copy_path: Optional[Path] = None,
    allow_hard_unlink: bool = False,
    job_id: Optional[int] = None,
    reason: str = "",
    file_hash: str = "",
    vsn: str = "",
) -> None:
    FileSafetyGuard.quarantine_or_unlink(
        path,
        volume_root=volume_root,
        verified_copy_path=verified_copy_path,
        allow_hard_unlink=allow_hard_unlink,
        job_id=job_id,
        reason=reason,
        file_hash=file_hash,
        vsn=vsn,
    )


def _safe_move(
    src: Path,
    dst: Path,
    src_volume_root: Optional[Path] = None,
    dst_volume_root: Optional[Path] = None,
    *,
    job_id: Optional[int] = None,
    file_hash: str = "",
    vsn: str = "",
) -> None:
    FileSafetyGuard.safe_move(
        src,
        dst,
        src_volume_root=src_volume_root,
        dst_volume_root=dst_volume_root,
        job_id=job_id,
        file_hash=file_hash,
        vsn=vsn,
    )


def _safe_copy_new_to_daf_backup(
    src_path: Path,
    dest_path: Path,
    *,
    job_id: Optional[int] = None,
    vsn: str = "",
) -> Tuple[bool, str, Optional[str]]:
    return FileSafetyGuard.safe_copy_new_to_daf_backup(
        src_path,
        dest_path,
        job_id=job_id,
        vsn=vsn,
    )
