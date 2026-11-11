"""Retention removes expired jobs without touching active work or downloads."""
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import api
from async_jobs import AsyncJobs, STATE_NAME


def _make_job(manager, job_id, status):
    directory = manager.jobs / job_id
    directory.mkdir()
    state = {"job_id": job_id, "status": status, "fingerprint": "stable-hash", "updated_at": 1}
    if status in {"queued", "running"}:
        state["payload"] = {}
    if status == "succeeded":
        state["result"] = {"download_url": f"/download/{job_id}/output.mp4", "fallback_triggered": False}
    if status == "unknown":
        state["error"] = {"code": "PROCESS_INTERRUPTED"}
    (directory / STATE_NAME).write_text(json.dumps(state))
    (directory / "source.mp4").write_bytes(b"original")
    (directory / "output.mp4").write_bytes(b"output")
    nested = directory / "intermediate"
    nested.mkdir()
    (nested / "frame.png").write_bytes(b"frame")
    os.utime(directory, (1, 1))
    return directory


def test_cleanup_removes_expired_jobs_and_keeps_active_jobs():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        for name, status in (("success", "succeeded"), ("failure", "failed"), ("unknown", "unknown"),
                             ("queued", "queued"), ("running", "running"), ("receiving", "receiving")):
            _make_job(manager, name, status)
        legacy = manager.jobs / "legacy"
        legacy.mkdir()
        (legacy / "output.mp4").write_bytes(b"old")
        os.utime(legacy, (1, 1))
        with patch.object(api, "WORK_DIR", root), patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
             patch.object(api, "_ACTIVE_JOBS", {}), patch.object(api, "_DOWNLOAD_COUNTS", {}), \
             patch.object(api, "_last_janitor_at", 0):
            api._sweep_stale_job_dirs()
        for name in ("success", "failure", "unknown"):
            directory = manager.jobs / name
            assert not directory.exists()
            assert manager.public(name) is None
        for name in ("queued", "running", "receiving"):
            assert (manager.jobs / name / "source.mp4").exists()
        assert not legacy.exists()
        # 作业过期后不再保留去重身份。
        assert manager.reserve("success") is True


def test_download_in_progress_is_not_cleaned():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        directory = _make_job(manager, "job", "succeeded")
        entered = threading.Event()
        release = threading.Event()
        class BlockingStream(io.BytesIO):
            def write(self, data):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test stream not released")
                return super().write(data)
        handler = SimpleNamespace(wfile=BlockingStream(), send_response=lambda *_: None,
                                  send_header=lambda *_: None, end_headers=lambda: None,
                                  _send_json=lambda *_: None)
        with patch.object(api, "WORK_DIR", root), patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
             patch.object(api, "_ACTIVE_JOBS", {}), patch.object(api, "_DOWNLOAD_COUNTS", {}):
            thread = threading.Thread(target=api.RemoverAPIHandler._handle_download,
                                      args=(handler, "/download/job/output.mp4"))
            thread.start()
            try:
                assert entered.wait(2)
                with patch.object(api, "_last_janitor_at", 0):
                    api._sweep_stale_job_dirs()
                assert (directory / "output.mp4").exists()
                assert manager.public("job")["status"] == "succeeded"
            finally:
                release.set()
                thread.join(3)
            assert not thread.is_alive()
            assert handler.wfile.getvalue() == b"output"
            with patch.object(api, "_last_janitor_at", 0):
                api._sweep_stale_job_dirs()
            assert not directory.exists()
            assert manager.public("job") is None


def test_cleanup_removes_previously_cleaned_tombstone(capsys):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        directory = _make_job(manager, "job", "failed")
        state = manager._read("job")
        state["artifacts_expired_at"] = 1
        state["artifacts_cleaned_at"] = 1
        manager._write("job", state)
        with patch.object(api, "WORK_DIR", root), patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
             patch.object(api, "_ACTIVE_JOBS", {}), patch.object(api, "_DOWNLOAD_COUNTS", {}):
            with patch.object(api, "_last_janitor_at", 0):
                api._sweep_stale_job_dirs()
            assert not directory.exists()
            assert "swept stale async job job" in capsys.readouterr().out
            with patch.object(api, "_last_janitor_at", 0):
                api._sweep_stale_job_dirs()
            assert "swept stale async job job" not in capsys.readouterr().out
        manager.owner_file.close()


def test_cleanup_retries_after_partial_artifact_removal(capsys):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        directory = _make_job(manager, "job", "succeeded")
        original = Path.unlink
        def fail_output(path, *args, **kwargs):
            if path.name == "output.mp4":
                raise PermissionError("injected failure")
            return original(path, *args, **kwargs)
        with patch.object(api, "WORK_DIR", root), patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
             patch.object(api, "_ACTIVE_JOBS", {}), patch.object(api, "_DOWNLOAD_COUNTS", {}):
            with patch.object(api, "_last_janitor_at", 0), patch.object(Path, "unlink", fail_output):
                api._sweep_stale_job_dirs()
            state = manager._read("job")
            assert state["artifacts_expired_at"]
            assert "artifacts_cleaned_at" not in state
            assert (directory / "output.mp4").exists()
            assert "job cleanup failed" in capsys.readouterr().out
            with patch.object(api, "_last_janitor_at", 0):
                api._sweep_stale_job_dirs()
            assert not directory.exists()
        manager.owner_file.close()


def test_terminal_job_is_not_cleaned_until_worker_releases_it():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        directory = _make_job(manager, "job", "succeeded")
        with patch.object(api, "WORK_DIR", root), patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
             patch.object(api, "_ACTIVE_JOBS", {}), patch.object(api, "_DOWNLOAD_COUNTS", {}):
            for active in (manager.scheduled, manager.inflight):
                active.add("job")
                with patch.object(api, "_last_janitor_at", 0):
                    api._sweep_stale_job_dirs()
                assert (directory / STATE_NAME).exists()
                active.remove("job")
            with patch.object(api, "_last_janitor_at", 0):
                api._sweep_stale_job_dirs()
            assert not directory.exists()
        manager.owner_file.close()


def test_janitor_runs_without_requests():
    stop = SimpleNamespace(wait=lambda _: next(waits))
    waits = iter([False, True])
    with patch.object(api, "_prune_active_jobs") as sweep:
        api._janitor_loop(stop)
    sweep.assert_called_once_with()


def test_janitor_logs_low_disk_space(capsys):
    waits = iter([False, True])
    stop = SimpleNamespace(wait=lambda _: next(waits))
    usage = SimpleNamespace(total=100, free=10)
    with patch.object(api, '_prune_active_jobs'), patch.object(api.shutil, 'disk_usage', return_value=usage):
        api._janitor_loop(stop)
    assert 'vsr_disk_low free_bytes=10 total_bytes=100' in capsys.readouterr().out
