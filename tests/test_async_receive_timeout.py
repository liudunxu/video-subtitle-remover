"""Partial asynchronous uploads cannot hold a bounded admission slot forever."""
import io
import tempfile
import threading
from unittest.mock import patch

import api
from async_jobs import AsyncJobs


class Socket:
    def __init__(self):
        self.timeout = None
        self.values = []

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value
        self.values.append(value)


class SlowStream:
    def __init__(self, clock):
        self.clock = clock

    def read1(self, size):
        # Always making progress must not reset the total request deadline.
        self.clock[0] += 1
        return b'-'


def handler(manager, stream, length=100):
    obj = object.__new__(api.RemoverAPIHandler)
    obj.path = '/api/jobs/remove-subtitle'
    obj.headers = {'X-Job-Id': 'job', 'X-Job-Store-Id': manager.store_id,
                   'Content-Type': 'multipart/form-data; boundary=test', 'Content-Length': str(length)}
    obj.rfile = stream
    obj.connection = Socket()
    obj.close_connection = False
    obj.responses = []
    obj._send_json = lambda status, payload: obj.responses.append((int(status), payload))
    return obj


def test_slow_trickling_upload_times_out_and_releases_slot():
    with tempfile.TemporaryDirectory() as root:
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock(), max_pending=1)
        clock = [0.0]
        request = handler(manager, SlowStream(clock))
        with patch.object(api, '_async_manager', return_value=manager), \
             patch.object(api, 'ASYNC_RECEIVE_TIMEOUT_SECONDS', 2), \
             patch.object(api.time, 'monotonic', side_effect=lambda: clock[0]):
            request.do_POST()
        assert request.responses[0][0] == 408
        assert request.close_connection
        assert request.connection.timeout is None
        assert request.connection.values == [2.0, 1.0, None]
        assert manager.public('job')['status'] == 'failed'
        assert manager.public('job')['error']['code'] == 'RECEIVE_INTERRUPTED'
        assert 'job' not in manager.inflight
        assert list(manager.incoming.iterdir()) == []
        assert manager.reserve('another')


def test_complete_multipart_restores_original_socket_timeout():
    body = (b'--test\r\nContent-Disposition: form-data; name="payload"\r\n\r\n{}\r\n'
            b'--test\r\nContent-Disposition: form-data; name="video"; filename="in.mp4"\r\n'
            b'Content-Type: video/mp4\r\n\r\nvideo\r\n--test--\r\n')
    with tempfile.TemporaryDirectory() as root:
        manager = AsyncJobs(root, lambda *_: None, lambda _: None, threading.RLock())
        request = handler(manager, io.BytesIO(body), len(body))
        request.connection.timeout = 17
        # Stop immediately after the body is parsed; this test needs no worker.
        with patch.object(api, '_async_manager', return_value=manager), \
             patch.object(manager, 'accept', side_effect=RuntimeError('test boundary')):
            request.do_POST()
        assert request.responses[0][0] == 500
        assert request.connection.timeout == 17
        assert request.connection.values[-1] == 17
        assert 'job' not in manager.inflight
