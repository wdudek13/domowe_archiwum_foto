import datetime
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

import app.database as db_module
from app.models import BackgroundJob, OperationLogEntry

logger = logging.getLogger("OperationControl")


class OperationLogger:
    """Writes persistent audit log entries to disk (data/logs/operations.log) and SQLite."""

    _file_lock = threading.Lock()

    @classmethod
    def log(
        cls,
        operation_type: str,
        message: str,
        status: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
        log_file: Optional[Path] = None,
        commit: bool = True,
    ) -> None:
        target_log_file = log_file or db_module.OPERATIONS_LOG_FILE
        now = datetime.datetime.utcnow()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        details_str = json.dumps(details, ensure_ascii=False) if details else ""
        line = f"[{ts_str}] [{status}] [{operation_type}] {message}"
        if details_str:
            line += f" | {details_str}"

        with cls._file_lock:
            try:
                target_log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(target_log_file, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                logger.warning("Failed to write to operation log file %s: %s", target_log_file, exc)

        own_session = False
        session = db
        if session is None:
            session = db_module.SessionLocal()
            own_session = True
        try:
            entry = OperationLogEntry(
                operation_type=operation_type,
                status=status,
                message=message,
                details_json=details_str if details_str else None,
                created_at=now,
            )
            session.add(entry)
            if own_session or commit:
                session.commit()
            else:
                session.flush()
        except Exception as exc:
            logger.debug("Failed to persist log entry to DB: %s", exc)
            if own_session:
                session.rollback()
        finally:
            if own_session:
                session.close()

    @classmethod
    def get_recent_logs(cls, db: Session, limit: int = 100, log_file: Optional[Path] = None) -> List[Dict[str, Any]]:
        target_log_file = log_file or db_module.OPERATIONS_LOG_FILE
        rows = db.scalars(
            select(OperationLogEntry).order_by(desc(OperationLogEntry.id)).limit(limit)
        ).all()
        if rows:
            return [
                {
                    "id": r.id,
                    "timestamp": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "operation_type": r.operation_type,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details_json,
                }
                for r in rows
            ]
        if target_log_file.exists():
            lines = target_log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
            return [
                {
                    "id": idx,
                    "timestamp": "",
                    "operation_type": "LOG",
                    "status": "INFO",
                    "message": ln,
                    "details": None,
                }
                for idx, ln in enumerate(reversed(lines), start=1)
            ]
        return []


class ProcessManager:
    """
    Central registry and controller for all background and multi-step processes
    (AI analysis, scanning, backup/deduplication, file deletion, trash operations, physical moves).
    Supports status polling, stopping/pausing, and resuming unfinished tasks.
    """

    _lock = threading.Lock()
    _storage_mutation_lock = threading.Lock()
    _active_storage_op: Optional[str] = None
    _stop_flags: Dict[int, bool] = {}
    _active_threads: Dict[int, threading.Thread] = {}
    _job_cache: Dict[int, Dict[str, Any]] = {}
    _last_db_sync: Dict[int, float] = {}

    @classmethod
    def try_acquire_storage_lock(cls, op_name: str) -> Tuple[bool, Optional[str]]:
        """V-13: Non-blocking global storage mutation guard to prevent concurrent conflicting disk/DB writes."""
        if cls._storage_mutation_lock.acquire(blocking=False):
            cls._active_storage_op = op_name
            return True, None
        return False, cls._active_storage_op or "inna operacja dyskowa"

    @classmethod
    def release_storage_lock(cls) -> None:
        cls._active_storage_op = None
        try:
            cls._storage_mutation_lock.release()
        except RuntimeError:
            pass

    @classmethod
    def create_job(
        cls,
        job_type: str,
        title: str,
        item_ids: List[str],
        params: Optional[Dict[str, Any]] = None,
        initial_status: str = "RUNNING",
        db: Optional[Session] = None,
    ) -> int:
        own_session = False
        session = db
        if session is None:
            session = db_module.SessionLocal()
            own_session = True
        try:
            job = BackgroundJob(
                job_type=job_type,
                title=title,
                status=initial_status,
                processed_count=0,
                total_count=len(item_ids),
                params_json=json.dumps(params or {}, ensure_ascii=False),
                remaining_ids_json=json.dumps(item_ids, ensure_ascii=False),
                last_message=f"Zainicjowano zadanie: {title}",
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id
            with cls._lock:
                cls._stop_flags[job_id] = False
                cls._job_cache[job_id] = {
                    "id": job_id,
                    "job_type": job_type,
                    "title": title,
                    "status": initial_status,
                    "processed_count": 0,
                    "total_count": len(item_ids),
                    "last_message": f"Zainicjowano zadanie: {title}",
                }
            OperationLogger.log(
                operation_type=job_type.upper(),
                message=f"Start procesu #{job_id}: {title} (elementów: {len(item_ids)})",
                status="START",
                details=params,
                db=session,
                commit=True,
            )
            return job_id
        finally:
            if own_session:
                session.close()

    @classmethod
    def request_stop(cls, job_id: Optional[int] = None) -> List[int]:
        stopped_ids: List[int] = []
        with cls._lock:
            if job_id is not None:
                cls._stop_flags[job_id] = True
                stopped_ids.append(job_id)
            else:
                for jid in list(cls._stop_flags.keys()):
                    cls._stop_flags[jid] = True
                    stopped_ids.append(jid)

        with db_module.SessionLocal() as db:
            stmt = select(BackgroundJob).where(BackgroundJob.status.in_(["RUNNING", "WAITING_FOR_DEVICE"]))
            if job_id is not None:
                stmt = stmt.where(BackgroundJob.id == job_id)
            jobs = db.scalars(stmt).all()
            for j in jobs:
                if j.id not in stopped_ids:
                    stopped_ids.append(j.id)
                with cls._lock:
                    is_alive = j.id in cls._active_threads and cls._active_threads[j.id].is_alive()
                if not is_alive:
                    j.status = "STOPPED"
                    j.last_message = "Proces zatrzymany przez użytkownika (można wznowić)."
            db.commit()

        if stopped_ids:
            OperationLogger.log(
                operation_type="PROCESS_CONTROL",
                message=f"Zażądano zatrzymania procesów: {stopped_ids}",
                status="WARN",
            )
        return stopped_ids

    @classmethod
    def is_stop_requested(cls, job_id: int) -> bool:
        with cls._lock:
            return bool(cls._stop_flags.get(job_id, False))

    @classmethod
    def clear_stop(cls, job_id: int) -> None:
        with cls._lock:
            cls._stop_flags[job_id] = False

    @classmethod
    def update_progress(
        cls,
        job_id: int,
        processed_delta: int = 0,
        processed_count: Optional[int] = None,
        total_count: Optional[int] = None,
        remaining_ids: Optional[List[str]] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> None:
        now_ts = time.time()
        is_backup_job = False
        with cls._lock:
            cached = cls._job_cache.setdefault(job_id, {})
            cached["id"] = job_id
            if total_count is not None:
                cached["total_count"] = total_count
            if processed_count is not None:
                cached["processed_count"] = processed_count
            elif processed_delta:
                cached["processed_count"] = min(cached.get("total_count", 100), cached.get("processed_count", 0) + processed_delta)
            if message is not None:
                cached["last_message"] = message
            if status is not None:
                cached["status"] = status
            if cached.get("job_type") in ("DB_BACKUP_MEDIUM", "GDRIVE_DB_BACKUP"):
                is_backup_job = True

        # For backup jobs, NEVER write to SQLite during the live backup step!
        # Updating the source database forces sqlite3.Connection.backup() to rewind or restart page copying.
        if is_backup_job and status is None and db is None:
            return

        should_sync_db = (status is not None) or (now_ts - cls._last_db_sync.get(job_id, 0) >= 1.0)
        if not should_sync_db and db is None:
            return

        cls._last_db_sync[job_id] = now_ts
        own_session = False
        session = db
        if session is None:
            session = db_module.SessionLocal()
            own_session = True
        try:
            job = session.get(BackgroundJob, job_id)
            if not job:
                return
            with cls._lock:
                c = cls._job_cache.get(job_id, {})
                job.processed_count = c.get("processed_count", job.processed_count)
                job.total_count = c.get("total_count", job.total_count)
                job.last_message = c.get("last_message", job.last_message)
                if status is not None:
                    job.status = c.get("status", job.status)
            if remaining_ids is not None:
                job.remaining_ids_json = json.dumps(remaining_ids, ensure_ascii=False)
            job.updated_at = datetime.datetime.now(datetime.UTC)
            if own_session:
                session.commit()
            else:
                session.flush()
        except Exception as exc:
            logger.debug("update_progress DB sync deferred: %s", exc)
        finally:
            if own_session:
                try:
                    session.close()
                except Exception:
                    pass

    @classmethod
    def finish_job(
        cls,
        job_id: int,
        status: str = "COMPLETED",
        message: str = "Zakończono pomyślnie.",
        db: Optional[Session] = None,
    ) -> None:
        with cls._lock:
            cached = cls._job_cache.setdefault(job_id, {})
            cached["status"] = status
            cached["last_message"] = message
            if status == "COMPLETED" and "total_count" in cached:
                cached["processed_count"] = cached["total_count"]

        own_session = False
        session = db
        if session is None:
            session = db_module.SessionLocal()
            own_session = True
        try:
            job = session.get(BackgroundJob, job_id)
            if job:
                job.status = status
                job.last_message = message
                if status == "COMPLETED":
                    job.remaining_ids_json = "[]"
                    job.processed_count = job.total_count
                job.updated_at = datetime.datetime.now(datetime.UTC)
                j_type = job.job_type
                j_title = job.title
                if own_session:
                    session.commit()
                else:
                    session.flush()
                OperationLogger.log(
                    operation_type=j_type.upper(),
                    message=f"Proces #{job_id} ({j_title}) -> {status}: {message}",
                    status="OK" if status == "COMPLETED" else status,
                    db=session,
                )
        finally:
            if own_session:
                try:
                    session.close()
                except Exception:
                    pass
        with cls._lock:
            cls._stop_flags.pop(job_id, None)
            cls._active_threads.pop(job_id, None)
            cls._last_db_sync.pop(job_id, None)

    @classmethod
    def get_active_backup_job(cls, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        with cls._lock:
            for jid, c in reversed(list(cls._job_cache.items())):
                if c.get("status") == "RUNNING" and c.get("job_type") in ("DB_BACKUP_MEDIUM", "GDRIVE_DB_BACKUP"):
                    return dict(c)
        own_session = False
        session = db
        if session is None:
            session = db_module.SessionLocal()
            own_session = True
        try:
            job = session.scalar(
                select(BackgroundJob).where(
                    BackgroundJob.status == "RUNNING",
                    BackgroundJob.job_type.in_(["DB_BACKUP_MEDIUM", "GDRIVE_DB_BACKUP"]),
                ).order_by(BackgroundJob.id.desc())
            )
            if job:
                return {
                    "id": job.id,
                    "job_type": job.job_type,
                    "title": job.title,
                    "status": job.status,
                    "processed_count": job.processed_count,
                    "total_count": job.total_count,
                    "last_message": job.last_message,
                }
            return None
        except Exception:
            return None
        finally:
            if own_session:
                try:
                    session.close()
                except Exception:
                    pass

    @classmethod
    def start_background_thread(cls, job_id: int, target_fn: Callable, *args, **kwargs) -> None:
        with cls._lock:
            cls._stop_flags[job_id] = False
            t = threading.Thread(target=target_fn, args=(job_id, *args), kwargs=kwargs, daemon=True)
            cls._active_threads[job_id] = t
            t.start()

    @classmethod
    def update_cache_only(
        cls,
        job_id: int,
        message: Optional[str] = None,
        processed_count: Optional[int] = None,
        total_count: Optional[int] = None,
    ) -> None:
        """Updates RAM cache for live UI inspection without writing to SQLite (safe during SQLite backup)."""
        with cls._lock:
            cached = cls._job_cache.setdefault(job_id, {"id": job_id})
            if message is not None:
                cached["last_message"] = message
            if processed_count is not None:
                cached["processed_count"] = processed_count
            if total_count is not None:
                cached["total_count"] = total_count

    @classmethod
    def get_jobs_summary(cls, db: Session) -> Dict[str, Any]:
        jobs = list(
            db.scalars(
                select(BackgroundJob).order_by(desc(BackgroundJob.updated_at), desc(BackgroundJob.id)).limit(30)
            ).all()
        )
        with cls._lock:
            for j in jobs:
                c = cls._job_cache.get(j.id)
                if c and j.status == "RUNNING":
                    try:
                        db.expunge(j)
                    except Exception:
                        pass
                    if "processed_count" in c:
                        j.processed_count = int(c["processed_count"] or 0)
                    if "total_count" in c:
                        j.total_count = int(c["total_count"] or 0)
                    if c.get("last_message"):
                        j.last_message = str(c["last_message"])
        running = [j for j in jobs if j.status == "RUNNING"]
        unfinished = [j for j in jobs if j.status in ("STOPPED", "PAUSED", "WAITING_FOR_DEVICE", "FAILED")]
        completed = [j for j in jobs if j.status == "COMPLETED"]
        return {
            "all_jobs": jobs,
            "running_jobs": running,
            "unfinished_jobs": unfinished,
            "completed_jobs": completed,
            "has_running": len(running) > 0,
            "has_unfinished": len(unfinished) > 0,
        }
