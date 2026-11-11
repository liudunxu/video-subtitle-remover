import json
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import api
import async_jobs
from async_jobs import AsyncJobs, JobConflict, STATE_NAME


def finished_processor(payload, job_id, job_dir):
    (job_dir / "result.mp4").write_bytes(b"clean")
    return {"download_url": f"/download/{job_id}/result.mp4", "fallback_triggered": False}


def wait_status(manager, job_id, expected, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        state = manager.public(job_id)
        if state and state["status"] == expected:
            return state
        time.sleep(.01)
    raise AssertionError(f"job {job_id} did not reach {expected}: {manager.public(job_id)}")


class AsyncJobsTest(unittest.TestCase):
    def test_crash_during_reservation_never_publishes_a_partial_job(self):
        script = '''
import os, sys, threading
from async_jobs import AsyncJobs
manager = AsyncJobs(sys.argv[1], lambda *args: None, lambda _: None, threading.RLock())
os.rename = lambda *_: os._exit(73)
manager.reserve("job")
'''
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-c", script, temp], check=False)
            self.assertEqual(result.returncode, 73)
            jobs = Path(temp) / "jobs"
            self.assertFalse((jobs / "job").exists())
            self.assertEqual(len(list(jobs.glob(".async-reserve-*"))), 1)
            # A pre-existing synchronous directory is not a staging directory.
            legacy = jobs / "legacy"
            legacy.mkdir()
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            manager.recover()
            self.assertEqual(list(jobs.glob(".async-reserve-*")), [])
            self.assertTrue(legacy.is_dir())
            self.assertIsNone(manager.public("job"))
            self.assertTrue(manager.reserve("job"))
            with self.assertRaises(JobConflict) as caught:
                manager.reserve("legacy")
            self.assertEqual(caught.exception.code, "JOB_ID_IN_USE")
            self.assertTrue(legacy.is_dir())
            manager.owner_file.close()

    def test_crash_after_reservation_publication_recovers_as_failed(self):
        script = '''
import os, sys, threading
from pathlib import Path
import async_jobs
manager = async_jobs.AsyncJobs(sys.argv[1], lambda *args: None, lambda _: None, threading.RLock())
original = async_jobs._sync_directory
def sync(directory):
    if Path(directory) == manager.jobs and (manager.jobs / "job").exists():
        os._exit(74)
    return original(directory)
async_jobs._sync_directory = sync
manager.reserve("job")
'''
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-c", script, temp], check=False)
            self.assertEqual(result.returncode, 74)
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            manager.recover()
            self.assertEqual(manager.public("job")["status"], "failed")
            self.assertEqual(manager.public("job")["error"]["code"], "RECEIVE_INTERRUPTED")
            manager.owner_file.close()

    def test_publication_sync_failure_marks_unstarted_job_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            original = async_jobs._sync_directory
            def fail_parent_sync(directory):
                if directory == manager.jobs:
                    raise OSError("sync failed")
                return original(directory)
            with patch.object(async_jobs, "_sync_directory", side_effect=fail_parent_sync):
                with self.assertRaises(OSError):
                    manager.reserve("job")
            self.assertEqual(manager.public("job")["status"], "failed")
            self.assertEqual(manager.public("job")["error"]["code"], "RECEIVE_INTERRUPTED")
            manager.owner_file.close()

    def test_incomplete_multipart_is_not_accepted_after_video_part(self):
        body = (b'--test\r\nContent-Disposition: form-data; name="video"; '
                b'filename="input.mp4"\r\n\r\nvideo\r\n--test\r\n'
                b'Content-Disposition: form-data; name="payload"\r\n\r\n{"options":')
        handler = SimpleNamespace(headers={"Content-Length": str(len(body)),
                                          "Content-Type": "multipart/form-data; boundary=test"},
                                  rfile=io.BytesIO(body))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(api.RequestError) as caught:
                api._read_multipart(handler, Path(temp))
            self.assertEqual(caught.exception.status, 400)

    def test_receive_is_visible_and_duplicates_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = threading.RLock()
            manager = AsyncJobs(temp, finished_processor, lambda _: None, lock)
            with lock:
                self.assertTrue(manager.reserve("job1"))
                self.assertEqual(manager.public("job1")["status"], "receiving")
                with self.assertRaises(JobConflict) as caught:
                    manager.reserve("job1")
                self.assertEqual(caught.exception.code, "JOB_RECEIVING")
                upload = Path(temp) / "upload.mp4"
                upload.write_bytes(b"video-a")
                request = {"options": {"mode": "propainter"}, "sub_area": [1, 2, 3, 4]}
                state, created = manager.accept("job1", True, request, upload)
                self.assertTrue(created)
                self.assertEqual(state["status"], "queued")
                self.assertEqual((manager.jobs / "job1" / "source.mp4").read_bytes(), b"video-a")

                duplicate = Path(temp) / "same.mp4"
                duplicate.write_bytes(b"video-a")
                self.assertFalse(manager.reserve("job1"))
                _, created = manager.accept("job1", False, request, duplicate)
                self.assertFalse(created)
                duplicate.write_bytes(b"video-b")
                with self.assertRaises(JobConflict) as caught:
                    manager.accept("job1", False, request, duplicate)
                self.assertEqual(caught.exception.code, "JOB_INPUT_CONFLICT")
                self.assertEqual((manager.jobs / "job1" / "source.mp4").read_bytes(), b"video-a")
            completed = wait_status(manager, "job1", "succeeded")
            self.assertEqual(completed["download_url"], "/download/job1/result.mp4")
            self.assertFalse(completed["fallback_triggered"])

    def test_simultaneous_first_receipts_have_only_one_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            barrier = threading.Barrier(8)
            results = []

            def reserve():
                barrier.wait()
                try:
                    results.append(manager.reserve("same"))
                except JobConflict as exc:
                    results.append(exc.code)

            threads = [threading.Thread(target=reserve) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(results.count(True), 1)
            self.assertEqual(results.count("JOB_RECEIVING"), 7)

    def test_restart_recovers_queued_but_marks_ambiguous_work_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            for job_id, status in (("receive", "receiving"), ("run", "running"), ("queue", "queued")):
                directory = manager.jobs / job_id
                directory.mkdir()
                (directory / "source.mp4").write_bytes(b"video")
                manager._write(job_id, {"job_id": job_id, "status": status,
                    "fingerprint": "hash", "payload": {"_uploaded_video_path": str(directory / "source.mp4")}})
            manager.recover()
            self.assertEqual(manager.public("receive")["status"], "failed")
            self.assertEqual(manager.public("receive")["error"]["code"], "RECEIVE_INTERRUPTED")
            self.assertEqual(manager.public("run")["status"], "unknown")
            wait_status(manager, "queue", "succeeded")

    def test_interrupted_upload_is_failed_but_queued_or_running_is_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            self.assertTrue(manager.reserve("receive"))
            manager.abandon_receiving("receive")
            self.assertEqual(manager.public("receive")["status"], "failed")
            self.assertEqual(manager.public("receive")["error"]["code"], "RECEIVE_INTERRUPTED")
            for job_id, status in (("queued", "queued"), ("running", "running")):
                directory = manager.jobs / job_id
                directory.mkdir()
                manager._write(job_id, {"job_id": job_id, "status": status,
                                        "fingerprint": "hash", "payload": {}})
                manager.abandon_receiving(job_id)
                self.assertEqual(manager._read(job_id)["status"], status)
            manager.owner_file.close()

    def test_processor_failure_is_terminal_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temp:
            def fail(*_):
                raise ValueError("internal error")
            manager = AsyncJobs(temp, fail, lambda _: None, threading.RLock())
            self.assertTrue(manager.reserve("failed"))
            upload = Path(temp) / "upload.mp4"
            upload.write_bytes(b"video")
            manager.accept("failed", True, {}, upload)
            state = wait_status(manager, "failed", "failed")
            self.assertEqual(state["error"], {"code": "REMOVAL_FAILED", "type": "ValueError"})
            self.assertTrue((manager.jobs / "failed" / "source.mp4").is_file())

    def test_invalid_output_is_uncertain_after_processor_returns(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, lambda *_: {"download_url": "/download/invalid/missing.mp4"},
                                lambda _: None, threading.RLock())
            manager.reserve("invalid")
            upload = Path(temp) / "upload.mp4"
            upload.write_bytes(b"video")
            manager.accept("invalid", True, {}, upload)
            state = wait_status(manager, "invalid", "unknown")
            self.assertEqual(state["error"]["code"], "RESULT_UNCERTAIN")

    def test_failure_after_gpu_returns_is_uncertain(self):
        with tempfile.TemporaryDirectory() as temp:
            def processor(payload, job_id, job_dir):
                return api._process_removal(payload, job_id, job_dir, time.time(), job_id,
                                            lambda job, filename: f"/download/{job}/{filename}")

            def fake_gpu(source, *_args, **_kwargs):
                output = source.with_name("clean.mp4")
                output.write_bytes(b"processed")
                return output, False

            manager = AsyncJobs(temp, processor, lambda _: None, threading.RLock())
            manager.reserve("gpu")
            upload = Path(temp) / "upload.mp4"
            upload.write_bytes(b"video")
            # The glyph audit file is absent, simulating a finalization error
            # after the costly processing has already produced a video.
            with patch.object(api, "_normalize_area", return_value=(1, 2, 3, 4)), \
                 patch.object(api, "_normalize_options", return_value={"mask_strategy": "glyph", "sub_area_margin": 0}), \
                 patch.object(api, "_probe_video_stream_geometry", return_value=None), \
                 patch.object(api, "_run_subtitle_remover", side_effect=fake_gpu):
                manager.accept("gpu", True, {}, upload)
                state = wait_status(manager, "gpu", "unknown")
            self.assertEqual(state["error"]["code"], "RESULT_UNCERTAIN")
            self.assertTrue((manager.jobs / "gpu" / "clean.mp4").is_file())

    def test_corrupt_job_does_not_prevent_recovery_or_get_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            corrupt = manager.jobs / "corrupt"
            corrupt.mkdir()
            path = corrupt / STATE_NAME
            path.write_text('{"status":', encoding="utf-8")
            good = manager.jobs / "good"
            good.mkdir()
            (good / "source.mp4").write_bytes(b"video")
            manager._write("good", {"job_id": "good", "status": "queued", "fingerprint": "hash",
                                    "payload": {"_uploaded_video_path": str(good / "source.mp4")}})
            manager.recover()
            self.assertEqual(manager.public("corrupt")["error"]["code"], "JOB_STATE_UNREADABLE")
            with self.assertRaises(JobConflict) as conflict:
                manager.reserve("corrupt")
            self.assertEqual(conflict.exception.code, "JOB_STATE_UNKNOWN")
            self.assertEqual(path.read_text(encoding="utf-8"), '{"status":')
            wait_status(manager, "good", "succeeded")

    def test_success_state_write_failure_is_uncertain(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            write = manager._write

            def fail_success_once(job_id, state):
                if state.get("status") == "succeeded":
                    raise OSError("injected persistence failure")
                return write(job_id, state)

            manager.reserve("persist")
            upload = Path(temp) / "upload.mp4"
            upload.write_bytes(b"video")
            with patch.object(manager, "_write", side_effect=fail_success_once):
                manager.accept("persist", True, {}, upload)
                state = wait_status(manager, "persist", "unknown")
            self.assertEqual(state["error"]["code"], "RESULT_UNCERTAIN")
            self.assertTrue((manager.jobs / "persist" / "result.mp4").is_file())

    def test_restart_removes_abandoned_receiving_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            abandoned = manager.incoming / "receive-abandoned"
            abandoned.mkdir()
            (abandoned / "partial.mp4").write_bytes(b"partial video")
            unrelated = manager.incoming / "other"
            unrelated.mkdir()
            manager.recover()
            self.assertFalse(abandoned.exists())
            self.assertTrue(unrelated.exists())

    def test_queued_job_recovers_after_worker_start_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = AsyncJobs(temp, finished_processor, lambda _: None, threading.RLock())
            manager.reserve("retry")
            upload = Path(temp) / "upload.mp4"
            upload.write_bytes(b"video")
            with patch.object(threading.Thread, "start", side_effect=RuntimeError("no thread")):
                with self.assertRaises(RuntimeError):
                    manager.accept("retry", True, {}, upload)
            self.assertEqual(manager._read("retry")["status"], "queued")
            # Status polling restarts the worker for the already durable job.
            manager.public("retry")
            wait_status(manager, "retry", "succeeded")

    def test_janitor_retains_in_progress_and_unreadable_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = AsyncJobs(root, finished_processor, lambda _: None, threading.RLock())
            for job_id, status in (("queue", "queued"), ("receiving", "receiving"),
                                   ("running", "running"), ("unknown", "unknown"), ("done", "succeeded")):
                directory = root / "jobs" / job_id
                directory.mkdir(parents=True)
                state = {"job_id": job_id, "status": status}
                if status in {"queued", "running"}:
                    state.update(fingerprint="hash", payload={})
                if status == "succeeded":
                    state["result"] = {"download_url": "/download/done/result.mp4", "fallback_triggered": False}
                (directory / STATE_NAME).write_text(json.dumps(state))
                os.utime(directory, (1, 1))
            broken = root / "jobs" / "broken"
            broken.mkdir()
            (broken / STATE_NAME).write_text('{"job_id": "broken", "status": "invalid"}')
            os.utime(broken, (1, 1))
            with patch.object(api, "WORK_DIR", root), patch.object(api, "_last_janitor_at", 0), \
                 patch.object(api, "_ASYNC_MANAGERS", {root.resolve(): manager}), \
                 patch.object(api, "_ACTIVE_JOBS", {}):
                api._sweep_stale_job_dirs()
            for job_id in ("queue", "receiving", "running", "broken"):
                self.assertTrue((root / "jobs" / job_id).exists())
            self.assertIsNone(manager.public("unknown"))
            self.assertIsNone(manager.public("done"))


if __name__ == "__main__":
    unittest.main()
