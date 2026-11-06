"""Durable, idempotent submission for long-running subtitle removal jobs."""
from __future__ import annotations

import hashlib
import fcntl
import errno
import json
import math
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit


STATE_NAME = ".async-job.json"
ACTIVE_STATUSES = {"receiving", "queued", "running", "unknown"}
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
STORE_ID = re.compile(r"[a-f0-9]{32}")


def _log_exception(job_id, exc):
    """Record debugging frames without printing user input, URLs or credentials."""
    import traceback
    frames = traceback.extract_tb(exc.__traceback__)
    locations = [f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}" for frame in frames]
    print(f"ERROR async_job={job_id} error_type={type(exc).__name__} frames={locations}", flush=True)


class JobConflict(Exception):
    def __init__(self, code, status=None):
        super().__init__(code)
        self.code = code
        self.status = status


class ResultUncertain(Exception):
    """Processing returned, but constructing its result failed."""


class RemovableQueue(queue.Queue):
    """Allow an expired pending job to release its bounded queue slot."""

    def remove_waiting(self, job_id):
        with self.mutex:
            try:
                self.queue.remove(job_id)
            except ValueError:
                # A worker has already claimed it and will observe its terminal
                # state before executing the processor.
                return False
            self.unfinished_tasks -= 1
            if not self.unfinished_tasks:
                self.all_tasks_done.notify_all()
            self.not_full.notify()
            return True


def _sync_directory(directory):
    """Persist directory entries where the underlying filesystem supports it."""
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(fd)


def fingerprint(payload, source):
    """Bind an idempotency key to all request fields and the uploaded bytes."""
    public = {key: value for key, value in payload.items()
              if key not in {"_uploaded_video_path", "filename"}}
    request = json.dumps(public, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    video = hashlib.sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            video.update(block)
    return hashlib.sha256(request + b"\0" + video.digest()).hexdigest()


def valid_deadline(value):
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_state(state, job_id):
    return not (not isinstance(state, dict) or state.get("job_id") != job_id
                or not isinstance(state.get("status"), str)
                or state.get("status") not in ACTIVE_STATUSES | {"succeeded", "failed"}
                or (state.get("status") == "succeeded" and
                    (not isinstance(state.get("result"), dict)
                     or not isinstance(state["result"].get("download_url"), str)
                     or not isinstance(state["result"].get("fallback_triggered"), bool)))
                or (state.get("status") in {"queued", "running"} and
                    (not isinstance(state.get("fingerprint"), str)
                     or not isinstance(state.get("payload"), dict)
                     or not valid_deadline(state["payload"].get("queue_deadline_at")))))


class AsyncJobs:
    """A single-process queue with durable states and conservative recovery.

    ``processor`` receives (payload, job_id, job_dir) and returns a result dict.
    A process restart never assumes a previously running job has stopped safely.
    """

    def __init__(self, root, processor, progress, process_lock, *, max_pending=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # A second server sharing this directory must not mistake a live
        # receiver/worker for an interrupted job during its recovery scan.
        self.owner_file = (self.root / ".async-owner.lock").open("a+b")
        try:
            fcntl.flock(self.owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.owner_file.close()
            raise RuntimeError("async work directory is already in use") from exc
        self.store_id = self._load_store_id()
        self.jobs = self.root / "jobs"
        self.incoming = self.root / "incoming"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.processor = processor
        self.progress = progress
        self.process_lock = process_lock
        self.lock = threading.RLock()
        self.queue = RemovableQueue()
        self.scheduled = set()
        self.worker = None
        # Count uploads, waiting and running jobs, not merely queue entries.
        # Admission is protected by self.lock, so multiple request threads
        # cannot collectively exceed the configured capacity.
        self.max_pending = int(os.environ.get("VSR_ASYNC_MAX_PENDING", "16")) if max_pending is None else int(max_pending)
        if self.max_pending < 1:
            raise ValueError("VSR_ASYNC_MAX_PENDING must be positive")
        self.inflight = set()

    def _load_store_id(self):
        """Persist an identity for this work directory across process restarts."""
        path = self.root / ".async-store-id"
        try:
            value = path.read_text("ascii").strip()
        except FileNotFoundError:
            value = uuid.uuid4().hex
            # The owner lock excludes another server process. Publish only a
            # complete, fsynced identity; an interrupted first boot can leave
            # a temporary file but never a partial canonical identity.
            tmp = self.root / ".async-store-id.tmp"
            try:
                with tmp.open("w", encoding="ascii") as stream:
                    stream.write(value + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, path)
                _sync_directory(self.root)
            finally:
                tmp.unlink(missing_ok=True)
        if not STORE_ID.fullmatch(value):
            raise RuntimeError("invalid async work directory identity")
        return value

    def _dir(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError("invalid job_id")
        return self.jobs / job_id

    def _read(self, job_id):
        try:
            state = json.loads((self._dir(job_id) / STATE_NAME).read_text("utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeError):
            state = None
        if not valid_state(state, job_id):
            # Preserve the original file for diagnosis and never rerun an
            # unrecognized state. One damaged job must not block the server.
            return {"job_id": job_id, "status": "unknown",
                    "error": {"code": "JOB_STATE_UNREADABLE"}}
        return state

    def _write(self, job_id, state):
        self._write_to_directory(self._dir(job_id), state)

    def _write_to_directory(self, directory, state):
        path = directory / STATE_NAME
        tmp = directory / (STATE_NAME + ".tmp")
        state["updated_at"] = time.time()
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _sync_directory(directory)

    def reserve(self, job_id):
        """Make receiving visible before reading request bytes; never clobber a job."""
        with self.lock:
            directory = self._dir(job_id)
            # Existing IDs must remain queryable even when the queue is full.
            if not directory.exists() and len(self.inflight) >= self.max_pending:
                # DIS may have stopped polling an expired queued job. Reclaim
                # it before refusing admission, even while the GPU worker is
                # blocked on a different, long-running task.
                for pending_id in tuple(self.inflight):
                    pending = self._read(pending_id)
                    if pending and pending.get("status") == "queued":
                        self._expire_queued(pending_id, pending)
                if len(self.inflight) >= self.max_pending:
                    raise JobConflict("QUEUE_FULL")
            # The API holds manager.lock when creating synchronous jobs too.
            # Check before rename, which can replace an empty POSIX directory.
            if directory.exists():
                state = self._read(job_id)
                if state is None:
                    raise JobConflict("JOB_ID_IN_USE")
                if state["status"] == "receiving":
                    raise JobConflict("JOB_RECEIVING", "receiving")
                if not state.get("fingerprint"):
                    raise JobConflict("JOB_STATE_UNKNOWN", "unknown")
                return False
            # Construct a complete state before making the job ID visible.
            # Staging in jobs keeps rename on the same filesystem; a process
            # crash before publication leaves no ambiguous job directory.
            with tempfile.TemporaryDirectory(prefix=".async-reserve-", dir=self.jobs) as staged:
                self._write_to_directory(Path(staged), {"job_id": job_id, "status": "receiving",
                                                        "created_at": time.time()})
                os.rename(staged, directory)
                try:
                    _sync_directory(self.jobs)
                except Exception:
                    # The handler has not yet received ownership; no upload
                    # or GPU work can be running for this ID.
                    self.abandon_receiving(job_id)
                    raise
            self.inflight.add(job_id)
            return True

    def accept(self, job_id, fresh, payload, source):
        deadline = payload.get("queue_deadline_at")
        if not valid_deadline(deadline):
            raise JobConflict("QUEUE_DEADLINE_INVALID")
        digest = fingerprint(payload, source)
        with self.lock:
            state = self._read(job_id)
            if state is None:
                raise JobConflict("JOB_STATE_UNKNOWN", "unknown")
            if not fresh:
                if state.get("fingerprint") != digest:
                    raise JobConflict("JOB_INPUT_CONFLICT", state.get("status"))
                return self.public(job_id), False
            if state["status"] != "receiving":
                raise JobConflict("JOB_STATE_UNKNOWN", state["status"])
            source = Path(source)
            with source.open("rb") as stream:
                os.fsync(stream.fileno())
            destination = self._dir(job_id) / "source.mp4"
            os.replace(source, destination)
            _sync_directory(destination.parent)
            stored = dict(payload)
            stored["filename"] = destination.name
            stored["_uploaded_video_path"] = str(destination)
            state.update(status="queued", fingerprint=digest, payload=stored)
            self._write(job_id, state)
            if not self._expire_queued(job_id, state):
                self._schedule(job_id)
            return self.public(job_id), True

    def abandon_receiving(self, job_id):
        with self.lock:
            state = self._read(job_id)
            if state and state.get("status") == "receiving":
                # No GPU worker is scheduled until after queued is durable.
                # This handler has stopped receiving, so there is no producer
                # left that can later advance this receiving state.
                state.update(status="failed", error={"code": "RECEIVE_INTERRUPTED"})
                self._write(job_id, state)
                self.inflight.discard(job_id)

    def _expire_queued(self, job_id, state):
        """Expire only queued work, while holding the same lock as start."""
        deadline = state.get("payload", {}).get("queue_deadline_at")
        if state.get("status") != "queued" or deadline is None or time.time() < deadline:
            return False
        state.update(status="failed", error={"code": "QUEUE_EXPIRED"})
        self._write(job_id, state)
        self.inflight.discard(job_id)
        if self.queue.remove_waiting(job_id):
            self.scheduled.discard(job_id)
        return True

    def public(self, job_id):
        with self.lock:
            state = self._read(job_id)
            if state is None:
                if self._dir(job_id).exists():
                    return {"job_id": job_id, "status": "unknown", "phase": "unknown",
                            "percent": 0, "stage": "unknown",
                            "error": {"code": "JOB_STATE_UNKNOWN"}}
                return None
            status = state["status"]
            if status == "queued":
                if self._expire_queued(job_id, state):
                    status = "failed"
                else:
                    # A transient worker start failure must not strand work.
                    self.inflight.add(job_id)
                    self._schedule(job_id)
            current = self.progress(job_id) if status == "running" else None
            body = {"job_id": job_id, "status": status,
                    "phase": (current or {}).get("phase", status),
                    "percent": (current or {}).get("percent", 0),
                    "stage": (current or {}).get("stage", status)}
            if status == "succeeded":
                body["percent"] = 100.0
                body["download_url"] = state["result"]["download_url"]
                body["fallback_triggered"] = state["result"]["fallback_triggered"]
            if state.get("error"):
                body["error"] = state["error"]
            return body

    def _schedule(self, job_id):
        if not self.worker or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._run_queue, daemon=True,
                                           name="vsr-async-jobs")
            self.worker.start()
        if job_id in self.scheduled:
            return
        self.scheduled.add(job_id)
        self.queue.put(job_id)

    def _run_queue(self):
        while True:
            job_id = self.queue.get()
            try:
                # Recovery may race another server process sharing the same work dir.
                with (self._dir(job_id) / ".async-job.lock").open("a+b") as claim:
                    fcntl.flock(claim.fileno(), fcntl.LOCK_EX)
                    self._execute_claimed(job_id)
            except Exception as exc:
                _log_exception(job_id, exc)
                try:
                    with self.lock:
                        state = self._read(job_id)
                        if state and state.get("status") in {"queued", "running"}:
                            state.update(status="unknown", error={"code": "WORKER_INTERRUPTED"})
                            self._write(job_id, state)
                except Exception as recovery_error:
                    _log_exception(job_id, recovery_error)
            finally:
                with self.lock:
                    self.scheduled.discard(job_id)
                    self.inflight.discard(job_id)
                self.queue.task_done()

    def _execute_claimed(self, job_id):
        # The same lock protects synchronous API work and automatic detection.
        with self.process_lock:
            with self.lock:
                state = self._read(job_id)
                if not state or state.get("status") != "queued":
                    return
                if self._expire_queued(job_id, state):
                    return
                state["status"] = "running"
                self._write(job_id, state)
            try:
                result = self.processor(state["payload"], job_id, self._dir(job_id))
            except ResultUncertain as exc:
                _log_exception(job_id, exc)
                with self.lock:
                    state.update(status="unknown", error={"code": "RESULT_UNCERTAIN",
                                                           "type": type(exc).__name__})
                    self._write(job_id, state)
                return
            except Exception as exc:
                _log_exception(job_id, exc)
                with self.lock:
                    state.update(status="failed", error={"code": "REMOVAL_FAILED",
                                                          "type": type(exc).__name__})
                    self._write(job_id, state)
                return
            try:
                prefix = "/download/" + job_id + "/"
                url = result.get("download_url")
                parsed = urlsplit(url) if isinstance(url, str) else None
                filename = unquote(parsed.path[len(prefix):]) if parsed and parsed.path.startswith(prefix) else ""
                if (not parsed or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                        or not filename or Path(filename).name != filename
                        or not (self._dir(job_id) / filename).is_file()):
                    raise ValueError("invalid async output path")
                with self.lock:
                    state.update(status="succeeded", result={"download_url": url,
                                                           "fallback_triggered": bool(result.get("fallback_triggered"))})
                    self._write(job_id, state)
            except Exception as exc:
                _log_exception(job_id, exc)
                with self.lock:
                    # The processor already returned; an invalid result or a
                    # persistence error does not prove no video was produced.
                    state.pop("result", None)
                    state.update(status="unknown", error={"code": "RESULT_UNCERTAIN",
                                                           "type": type(exc).__name__})
                    self._write(job_id, state)

    def recover(self):
        """Recover durable queued work; ambiguous interrupted work stays unknown."""
        with self.lock:
            # The owner lock was acquired before recovery, so no receiver in
            # another server process can be using these abandoned temporary dirs.
            for directory in self.incoming.glob("receive-*"):
                if directory.is_dir() and not directory.is_symlink():
                    try:
                        shutil.rmtree(directory)
                    except OSError as exc:
                        _log_exception("incoming-cleanup", exc)
            for directory in sorted(self.jobs.iterdir()):
                if directory.name.startswith(".async-reserve-") and directory.is_dir() and not directory.is_symlink():
                    try:
                        shutil.rmtree(directory)
                    except OSError as exc:
                        _log_exception("reserve-cleanup", exc)
                    continue
                if not directory.is_dir() or not JOB_ID.fullmatch(directory.name):
                    continue
                state = self._read(directory.name)
                if not state:
                    continue
                if state.get("status") == "receiving":
                    # The old server owns the directory lock until its process
                    # has exited; its unfinished upload cannot subsequently be
                    # queued. GPU work is scheduled only after queued is saved.
                    state.update(status="failed", error={"code": "RECEIVE_INTERRUPTED"})
                    self._write(directory.name, state)
                elif state.get("status") == "running":
                    state.update(status="unknown", error={"code": "PROCESS_INTERRUPTED"})
                    self._write(directory.name, state)
                elif state.get("status") == "queued":
                    if not self._expire_queued(directory.name, state):
                        self.inflight.add(directory.name)
                        self._schedule(directory.name)
