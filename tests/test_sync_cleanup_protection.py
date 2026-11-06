"""Legacy synchronous work is protected for its entire handler lifetime."""
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import api
from async_jobs import AsyncJobs


def handler(path='/api/remove-subtitle', job='legacy'):
    obj = object.__new__(api.RemoverAPIHandler)
    obj.path = path
    obj.headers = {'X-Job-Id': job, 'Content-Type': 'application/json'}
    obj.responses = []
    obj.close_connection = False
    obj._send_json = lambda status, value: obj.responses.append((int(status), value))
    return obj


def sweep_old(directory):
    os.utime(directory, (1, 1))
    with patch.object(api, '_last_janitor_at', 0):
        api._sweep_stale_job_dirs()


def test_sync_removal_stays_protected_while_worker_is_waiting():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
        entered, release = threading.Event(), threading.Event()
        request = handler()
        def process(*args):
            entered.set()
            assert release.wait(3)
            return {'ok': True}
        with patch.object(api, 'WORK_DIR', root), patch.object(api, '_ASYNC_MANAGERS', {root.resolve(): manager}), \
             patch.object(api, '_async_manager', return_value=manager), patch.object(api, '_ACTIVE_JOBS', {}), \
             patch.object(api, '_SYNC_ACTIVE_COUNTS', {}), patch.object(api, '_DOWNLOAD_COUNTS', {}), \
             patch.object(api, '_read_payload', return_value={'video_url': 'input'}), \
             patch.object(api, '_process_removal', side_effect=process):
            thread = threading.Thread(target=request.do_POST)
            thread.start()
            try:
                assert entered.wait(2)
                directory = root / 'jobs' / 'legacy'
                assert api._SYNC_ACTIVE_COUNTS == {'legacy': 1}
                sweep_old(directory)
                assert directory.is_dir()
            finally:
                release.set()
                thread.join(3)
            assert not thread.is_alive()
            assert request.responses == [(200, {'ok': True})]
            assert api._SYNC_ACTIVE_COUNTS == {}
            sweep_old(directory)
            assert not directory.exists()


def test_overlapping_sync_requests_keep_reference_until_last_finishes():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        counter = iter(range(2))
        counter_lock = threading.Lock()
        def process(*args):
            with counter_lock:
                i = next(counter)
            entered[i].set()
            assert release[i].wait(3)
            return {'ok': True}
        requests = [handler(), handler()]
        with patch.object(api, 'WORK_DIR', root), patch.object(api, '_ASYNC_MANAGERS', {root.resolve(): manager}), \
             patch.object(api, '_async_manager', return_value=manager), patch.object(api, '_ACTIVE_JOBS', {}), \
             patch.object(api, '_SYNC_ACTIVE_COUNTS', {}), patch.object(api, '_DOWNLOAD_COUNTS', {}), \
             patch.object(api, '_read_payload', return_value={'video_url': 'input'}), \
             patch.object(api, '_process_removal', side_effect=process):
            threads = [threading.Thread(target=req.do_POST) for req in requests]
            for thread in threads:
                thread.start()
            try:
                assert all(event.wait(2) for event in entered)
                directory = root / 'jobs' / 'legacy'
                assert api._SYNC_ACTIVE_COUNTS == {'legacy': 2}
                release[0].set()
                deadline = time.monotonic() + 2
                while api._SYNC_ACTIVE_COUNTS.get('legacy') != 1 and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert api._SYNC_ACTIVE_COUNTS == {'legacy': 1}
                sweep_old(directory)
                assert directory.is_dir()
            finally:
                for event in release:
                    event.set()
                for thread in threads:
                    thread.join(3)
            assert api._SYNC_ACTIVE_COUNTS == {}


def test_sync_parse_error_releases_registration_and_keeps_response():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
        request = handler()
        with patch.object(api, 'WORK_DIR', root), patch.object(api, '_async_manager', return_value=manager), \
             patch.object(api, '_SYNC_ACTIVE_COUNTS', {}), \
             patch.object(api, '_read_payload', side_effect=api.RequestError(400, 'bad input')):
            request.do_POST()
            assert request.responses == [(400, {'error': 'bad input'})]
            assert api._SYNC_ACTIVE_COUNTS == {}
            assert not (root / 'jobs' / 'legacy').exists()


def test_detect_job_is_protected_while_processing():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
        entered, release = threading.Event(), threading.Event()
        request = handler('/api/detect-subtitle-area')
        def detect(*args):
            entered.set()
            assert release.wait(3)
            return {}
        with patch.object(api, 'WORK_DIR', root), patch.object(api, '_ASYNC_MANAGERS', {root.resolve(): manager}), \
             patch.object(api, '_ACTIVE_JOBS', {}), patch.object(api, '_SYNC_ACTIVE_COUNTS', {}), \
             patch.object(api, '_DOWNLOAD_COUNTS', {}), \
             patch.object(api, '_read_payload', return_value={'video_url': 'input'}), \
             patch.object(api, '_stage_video', return_value=Path('/tmp/test-input.mp4')), \
             patch.object(api, '_run_subtitle_area_detection', side_effect=detect):
            thread = threading.Thread(target=request.do_POST)
            thread.start()
            try:
                assert entered.wait(2)
                jobs = list((root / 'jobs').iterdir())
                assert len(jobs) == 1
                sweep_old(jobs[0])
                assert jobs[0].is_dir()
            finally:
                release.set()
                thread.join(3)
            assert api._SYNC_ACTIVE_COUNTS == {}
            assert request.responses[0][0] == 200


def test_detect_conflicting_id_cannot_touch_async_job():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
        manager.reserve('same')
        request = handler('/api/detect-subtitle-area')
        with patch.object(api, 'WORK_DIR', root), patch.object(api, '_async_manager', return_value=manager), \
             patch.object(api.uuid, 'uuid4', return_value=SimpleNamespace(hex='same')), \
             patch.object(api, '_SYNC_ACTIVE_COUNTS', {}), \
             patch.object(api, '_read_payload') as read_payload:
            request.do_POST()
            assert request.responses[0][0] == 409
            assert request.responses[0][1]['error_code'] == 'JOB_ID_IN_USE'
            read_payload.assert_not_called()
            assert manager.public('same')['status'] == 'receiving'
            assert api._SYNC_ACTIVE_COUNTS == {}
        manager.owner_file.close()
