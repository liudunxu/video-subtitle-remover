import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import api
from pathlib import Path

from async_jobs import AsyncJobs, JobConflict


class QueueTest(unittest.TestCase):
    def test_store_identity_survives_restart_and_changes_with_directory(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            first = AsyncJobs(root, None, lambda _: None, threading.RLock())
            identity = first.store_id
            self.assertRegex(identity, r'^[0-9a-f]{32}$')
            first.owner_file.close()
            restarted = AsyncJobs(root, None, lambda _: None, threading.RLock())
            elsewhere = AsyncJobs(other, None, lambda _: None, threading.RLock())
            self.assertEqual(identity, restarted.store_id)
            self.assertNotEqual(identity, elsewhere.store_id)
            restarted.owner_file.close()
            elsewhere.owner_file.close()

    def test_invalid_existing_identity_refuses_start(self):
        with tempfile.TemporaryDirectory() as root:
            identity = Path(root) / '.async-store-id'
            identity.write_text('corrupt')
            with self.assertRaisesRegex(RuntimeError, 'invalid async work directory identity'):
                AsyncJobs(root, None, lambda _: None, threading.RLock())
            self.assertEqual(identity.read_text(), 'corrupt')

    def test_interrupted_identity_creation_does_not_publish_partial_id(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            # Simulate an interrupted write before the identity is published.
            with patch('async_jobs.os.fsync', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    AsyncJobs._load_store_id(SimpleNamespace(root=path))
            self.assertFalse((path / '.async-store-id').exists())
            self.assertFalse((path / '.async-store-id.tmp').exists())
            # An orphaned temporary file from a killed process is harmless.
            (path / '.async-store-id.tmp').write_text('partial')
            manager = AsyncJobs(path, None, lambda _: None, threading.RLock())
            self.assertEqual((path / '.async-store-id').read_text().strip(), manager.store_id)
            self.assertFalse((path / '.async-store-id.tmp').exists())
            manager.owner_file.close()

    def test_atomic_capacity_rejection_leaves_no_job_record(self):
        with tempfile.TemporaryDirectory() as root:
            manager = AsyncJobs(root, None, lambda _: None, threading.RLock(), max_pending=2)
            barrier = threading.Barrier(10)
            results = {}
            def submit(index):
                barrier.wait()
                job_id = f'job{index}'
                try:
                    manager.reserve(job_id)
                    results[job_id] = 'accepted'
                except JobConflict as exc:
                    results[job_id] = exc.code
            threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(list(results.values()).count('accepted'), 2)
            for job_id, result in results.items():
                self.assertEqual((manager.jobs / job_id).exists(), result == 'accepted')
            accepted = next(job_id for job_id, result in results.items() if result == 'accepted')
            with self.assertRaises(JobConflict) as conflict:
                manager.reserve(accepted)
            self.assertEqual(conflict.exception.code, 'JOB_RECEIVING')
            manager.abandon_receiving(accepted)
            rejected = next(job_id for job_id, result in results.items() if result == 'QUEUE_FULL')
            self.assertTrue(manager.reserve(rejected))
            manager.owner_file.close()

    def test_expired_queued_job_cannot_start_even_after_gpu_becomes_free(self):
        with tempfile.TemporaryDirectory() as root:
            process_lock = threading.RLock()
            calls = []
            manager = AsyncJobs(root, lambda *args: calls.append(args), lambda _: None, process_lock)
            with process_lock:
                manager.reserve('job')
                source = Path(root) / 'video'
                source.write_bytes(b'video')
                manager.accept('job', True, {'queue_deadline_at': time.time() + .1}, source)
                time.sleep(.15)
                state = manager.public('job')
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['error']['code'], 'QUEUE_EXPIRED')
            manager.queue.join()
            self.assertEqual(calls, [])
            self.assertEqual(manager.public('job')['status'], 'failed')
            manager.owner_file.close()

    def test_job_expires_at_gpu_start_without_a_status_request(self):
        with tempfile.TemporaryDirectory() as root:
            process_lock = threading.RLock()
            calls = []
            manager = AsyncJobs(root, lambda *args: calls.append(args), lambda _: None, process_lock)
            with process_lock:
                manager.reserve('job')
                source = Path(root) / 'video'
                source.write_bytes(b'video')
                manager.accept('job', True, {'queue_deadline_at': time.time() + .1}, source)
                time.sleep(.15)
            manager.queue.join()
            self.assertEqual(calls, [])
            self.assertEqual(manager.public('job')['error']['code'], 'QUEUE_EXPIRED')
            manager.owner_file.close()

    def test_expired_waiting_jobs_release_capacity_and_are_removed_from_queue(self):
        with tempfile.TemporaryDirectory() as root:
            started = threading.Event()
            release = threading.Event()
            calls = []
            def process(_payload, job_id, directory):
                calls.append(job_id)
                if job_id == 'running':
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError('timed out')
                (directory / 'clean.mp4').write_bytes(b'clean')
                return {'download_url': f'/download/{job_id}/clean.mp4', 'fallback_triggered': False}
            manager = AsyncJobs(root, process, lambda _: None, threading.RLock(), max_pending=2)
            def add(job_id, payload):
                manager.reserve(job_id)
                source = Path(root) / f'{job_id}.mp4'
                source.write_bytes(b'video')
                manager.accept(job_id, True, payload, source)
            try:
                add('running', {})
                self.assertTrue(started.wait(2))
                for index in range(5):
                    job_id = f'queued{index}'
                    add(job_id, {'queue_deadline_at': time.time() + .05})
                    self.assertEqual(manager.queue.qsize(), 1)
                    time.sleep(.07)
                    self.assertEqual(manager.public(job_id)['error']['code'], 'QUEUE_EXPIRED')
                    self.assertEqual(manager.queue.qsize(), 0)
                add('next', {})
            finally:
                release.set()
            manager.queue.join()
            self.assertEqual(calls, ['running', 'next'])
            self.assertEqual(manager.public('next')['status'], 'succeeded')
            manager.owner_file.close()

    def test_expired_already_claimed_job_also_releases_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            lock = threading.RLock()
            calls = []
            def process(_payload, job_id, directory):
                calls.append(job_id)
                (directory / 'clean.mp4').write_bytes(b'clean')
                return {'download_url': f'/download/{job_id}/clean.mp4', 'fallback_triggered': False}
            manager = AsyncJobs(root, process, lambda _: None, lock, max_pending=1)
            with lock:
                manager.reserve('old')
                source = Path(root) / 'old.mp4'
                source.write_bytes(b'video')
                manager.accept('old', True, {'queue_deadline_at': time.time() + .1}, source)
                until = time.monotonic() + 2
                while manager.queue.qsize() and time.monotonic() < until:
                    time.sleep(.01)
                self.assertEqual(manager.queue.qsize(), 0)
                time.sleep(.12)
                self.assertEqual(manager.public('old')['error']['code'], 'QUEUE_EXPIRED')
                self.assertIn('old', manager.scheduled)  # worker already claimed it
                manager.reserve('next')
                source = Path(root) / 'next.mp4'
                source.write_bytes(b'video')
                manager.accept('next', True, {}, source)
            manager.queue.join()
            self.assertEqual(calls, ['next'])
            manager.owner_file.close()

    def test_admission_reclaims_expired_job_without_status_polling(self):
        with tempfile.TemporaryDirectory() as root:
            started = threading.Event()
            release = threading.Event()
            calls = []
            def process(_payload, job_id, directory):
                calls.append(job_id)
                if job_id == 'running':
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError('timed out')
                (directory / 'clean.mp4').write_bytes(b'clean')
                return {'download_url': f'/download/{job_id}/clean.mp4', 'fallback_triggered': False}
            manager = AsyncJobs(root, process, lambda _: None, threading.RLock(), max_pending=2)
            def add(job_id, payload):
                manager.reserve(job_id)
                source = Path(root) / f'{job_id}.mp4'
                source.write_bytes(b'video')
                manager.accept(job_id, True, payload, source)
            try:
                add('running', {})
                self.assertTrue(started.wait(2))
                add('expired', {'queue_deadline_at': time.time() + .1})
                time.sleep(.15)
                # Neither the expired job nor its status was queried; a new
                # admission must reclaim the slot before returning QUEUE_FULL.
                add('next', {})
                self.assertEqual(manager.public('expired')['error']['code'], 'QUEUE_EXPIRED')
                self.assertEqual(manager.queue.qsize(), 1)
            finally:
                release.set()
            manager.queue.join()
            self.assertEqual(calls, ['running', 'next'])
            manager.owner_file.close()

    def test_expired_before_accept_is_terminal_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            manager = AsyncJobs(root, lambda *args: calls.append(args), lambda _: None, threading.RLock())
            source = Path(root) / 'video'
            source.write_bytes(b'video')
            payload = {'queue_deadline_at': time.time() - 1}
            manager.reserve('job')
            state, created = manager.accept('job', True, payload, source)
            self.assertTrue(created)
            self.assertEqual(state['error']['code'], 'QUEUE_EXPIRED')
            duplicate = Path(root) / 'duplicate'
            duplicate.write_bytes(b'video')
            self.assertFalse(manager.reserve('job'))
            state, created = manager.accept('job', False, payload, duplicate)
            self.assertFalse(created)
            self.assertEqual(state['error']['code'], 'QUEUE_EXPIRED')
            self.assertEqual(calls, [])
            manager.owner_file.close()

    def test_expired_job_does_not_run_on_restart(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            manager = AsyncJobs(root, lambda *args: calls.append(args), lambda _: None, threading.RLock())
            directory = manager.jobs / 'job'
            directory.mkdir()
            manager._write('job', {'job_id': 'job', 'status': 'queued', 'fingerprint': 'abc',
                                   'payload': {'queue_deadline_at': time.time() - 1}})
            manager.recover()
            self.assertEqual(manager.public('job')['error']['code'], 'QUEUE_EXPIRED')
            self.assertEqual(calls, [])
            manager.owner_file.close()


class HandlerContractTest(unittest.TestCase):
    def handler(self, path, headers):
        handler = object.__new__(api.RemoverAPIHandler)
        handler.path = path
        handler.headers = headers
        handler.close_connection = False
        handler.responses = []
        handler._send_json = lambda status, body: handler.responses.append((int(status), body))
        return handler

    def test_capabilities_advertises_stable_store_identity(self):
        with tempfile.TemporaryDirectory() as root:
            manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
            handler = self.handler('/api/capabilities', {})
            with patch.object(api, '_async_manager', return_value=manager):
                handler.do_GET()
            self.assertEqual(handler.responses[0][0], 200)
            self.assertEqual(handler.responses[0][1]['job_store_id'], manager.store_id)
            manager.owner_file.close()

    def test_store_mismatch_neither_reads_nor_reserves_job(self):
        with tempfile.TemporaryDirectory() as root:
            manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
            headers = {'X-Job-Store-Id': 'wrong', 'X-Job-Id': 'job', 'Content-Type': 'multipart/form-data'}
            post = self.handler('/api/jobs/remove-subtitle', headers)
            get = self.handler('/api/jobs/job', headers)
            with patch.object(api, '_async_manager', return_value=manager), \
                 patch.object(manager, 'public', wraps=manager.public) as public, \
                 patch.object(manager, 'reserve', wraps=manager.reserve) as reserve:
                post.do_POST()
                get.do_GET()
            self.assertEqual(post.responses[0][0], 409)
            self.assertEqual(get.responses[0][0], 409)
            self.assertEqual(post.responses[0][1]['error_code'], 'JOB_STORE_MISMATCH')
            self.assertEqual(get.responses[0][1]['error_code'], 'JOB_STORE_MISMATCH')
            public.assert_not_called()
            reserve.assert_not_called()
            self.assertFalse((manager.jobs / 'job').exists())
            manager.owner_file.close()

    def test_queue_full_response_does_not_create_job(self):
        with tempfile.TemporaryDirectory() as root:
            manager = AsyncJobs(root, None, lambda _: None, threading.RLock(), max_pending=1)
            manager.reserve('existing')
            handler = self.handler('/api/jobs/remove-subtitle', {'X-Job-Store-Id': manager.store_id,
                                   'X-Job-Id': 'new', 'Content-Type': 'multipart/form-data'})
            with patch.object(api, '_async_manager', return_value=manager):
                handler.do_POST()
            self.assertEqual(handler.responses[0][0], 429)
            self.assertEqual(handler.responses[0][1]['error_code'], 'QUEUE_FULL')
            self.assertFalse((manager.jobs / 'new').exists())
            manager.owner_file.close()


if __name__ == '__main__':
    unittest.main()
