import argparse
from pathlib import Path
import sys
from sqlalchemy import select, func

from app.database import engine, Base, SessionLocal
from app.backup import BackupManager
from app.models import MediaItem, WindowsVolume, MediaGroup
from app.scanner import WindowsVolumeInspector, FileScannerService
from app.storage_ops import StorageAutomationService


def cmd_integrity(_args) -> None:
    print("[*] Running SQLite physical integrity check...")
    mgr = BackupManager()
    mgr.check_integrity()
    print("[+] Integrity Check: PASSED (Zero data corruption detected).")


def cmd_backup(args) -> None:
    mgr = BackupManager()
    path = mgr.create_backup(tag=args.tag)
    print(f"[+] Native SQLite transactional backup written to: {path}")


def cmd_scan(args) -> None:
    scan_path = Path(args.path).resolve()
    if not scan_path.exists():
        print(f"[-] Path does not exist: {scan_path}")
        sys.exit(1)

    print(f"[*] Detecting Windows volume for path: {scan_path}")
    vsn, label, root = WindowsVolumeInspector.get_volume_info(scan_path)
    print(f"[+] Detected Volume: {label} (VSN: {vsn}, Root: {root})")

    db = SessionLocal()
    try:
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

        indexed_count = 0
        print("[*] Scanning directory recursively...")
        for cur_root, _, files in scan_path.walk():
            for f in files:
                file_full = cur_root / f
                try:
                    rel = file_full.resolve().relative_to(Path(root)).as_posix()
                except ValueError:
                    rel = f

                exists = db.scalar(
                    select(MediaItem.id).where(
                        MediaItem.volume_id == vol.id, MediaItem.relative_path == rel
                    )
                )
                if not exists:
                    item = FileScannerService.process_file(file_full, vol)
                    if item:
                        db.add(item)
                        indexed_count += 1
                        if indexed_count % 50 == 0:
                            db.commit()
                            print(f"    Indexed {indexed_count} items...", end="\r")

        db.commit()
        print(f"\n[+] Scan complete. Total new items indexed: {indexed_count}")

        # Update duplicates
        svc = StorageAutomationService(db)
        dups = svc.flag_duplicate_clusters()
        print(f"[+] Flagged duplicate clusters. Duplicate items marked: {dups}")
    finally:
        db.close()


def cmd_stats(_args) -> None:
    db = SessionLocal()
    try:
        total_items = db.scalar(select(func.count(MediaItem.id))) or 0
        total_size = db.scalar(select(func.sum(MediaItem.file_size_bytes))) or 0
        anomalies = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.has_date_anomaly == True)) or 0
        duplicates = db.scalar(select(func.count(MediaItem.id)).where(MediaItem.is_primary_duplicate == False)) or 0
        volumes = db.scalars(select(WindowsVolume)).all()

        print("=" * 50)
        print("          HYDRA MEDIA VAULT - STATISTICS          ")
        print("=" * 50)
        print(f"Total media indexed : {total_items}")
        print(f"Total storage tracked: {total_size / (1024**3):.2f} GB")
        print(f"Date anomalies      : {anomalies}")
        print(f"Duplicate files     : {duplicates}")
        print("-" * 50)
        print("Tracked Volumes:")
        for v in volumes:
            online_str = "ONLINE" if Path(v.last_known_drive_letter).exists() else "OFFLINE"
            print(f" - [{online_str}] {v.volume_label} (VSN: {v.volume_serial_number}, Drive: {v.last_known_drive_letter})")
        print("=" * 50)
    finally:
        db.close()


def main():
    Base.metadata.create_all(bind=engine)
    parser = argparse.ArgumentParser(description="Hydra Media CLI Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: integrity
    subparsers.add_parser("integrity", help="Check SQLite physical integrity")

    # Subcommand: backup
    backup_p = subparsers.add_parser("backup", help="Create atomic SQLite backup")
    backup_p.add_argument("--tag", default="cli_manual", help="Backup tag identifier")

    # Subcommand: scan
    scan_p = subparsers.add_parser("scan", help="Index a directory/USB drive")
    scan_p.add_argument("path", help="Path to drive or directory (e.g. E:\\Photos)")

    # Subcommand: stats
    subparsers.add_parser("stats", help="Display storage and anomaly metrics")

    args = parser.parse_args()
    if args.command == "integrity":
        cmd_integrity(args)
    elif args.command == "backup":
        cmd_backup(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()