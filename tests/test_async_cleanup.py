"""Retention must bound artifact storage without losing job identity."""
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


def test_cleanup_keeps_tombstones_and_active_jobs():
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
            assert (directory / STATE_NAME).exists()
            assert not (directory / "source.mp4").exists()
            assert not (directory / "output.mp4").exists()
            assert not (directory / "intermediate").exists()
            assert manager._read(name)["fingerprint"] == "stable-hash"
        assert manager.public("success")["error"]["code"] == "RESULT_EXPIRED"
        assert manager.public("success")["status"] == "failed"
        assert manager.public("unknown")["error"]["code"] == "PROCESS_INTERRUPTED"
        assert manager.public("failure")["status"] == "failed"
        for name in ("queued", "running", "receiving"):
            assert (manager.jobs / name / "source.mp4").exists()
        assert not legacy.exists()
        # A second submission must still encounter the saved identity.
        assert manager.reserve("success") is False


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
            assert not (directory / "output.mp4").exists()
            assert manager.public("job")["error"]["code"] == "RESULT_EXPIRED"


def test_cleaned_tombstone_does_not_rescan_artifacts_or_log_repeatedly(capsys):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        directory = _make_job(manager, "job", "succeeded")
        with patch.object(api, "WORK_DIR", root), patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
             patch.object(api, "_ACTIVE_JOBS", {}), patch.object(api, "_DOWNLOAD_COUNTS", {}):
            with patch.object(api, "_last_janitor_at", 0):
                api._sweep_stale_job_dirs()
            assert manager._read("job")["artifacts_cleaned_at"]
            assert "swept stale async artifacts job" in capsys.readouterr().out
            calls = []
            original = Path.iterdir
            def track(path):
                if path == directory:
                    calls.append(path)
                return original(path)
            with patch.object(api, "_last_janitor_at", 0), patch.object(Path, "iterdir", track):
                api._sweep_stale_job_dirs()
            assert not calls
            assert "swept stale async artifacts job" not in capsys.readouterr().out
            assert manager.reserve("job") is False
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
            assert not (directory / "output.mp4").exists()
            assert manager._read("job")["artifacts_cleaned_at"]
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
