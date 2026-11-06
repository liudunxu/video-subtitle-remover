"""An async result must come from the same work directory as its status."""
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import api
from async_jobs import AsyncJobs


def test_download_store_mismatch_never_reads_result():
    with tempfile.TemporaryDirectory() as root:
        manager = AsyncJobs(root, None, lambda _: None, threading.RLock())
        result = Path(root) / 'jobs' / 'job' / 'result.mp4'
        result.parent.mkdir()
        result.write_bytes(b'wrong-store-result')
        handler = object.__new__(api.RemoverAPIHandler)
        handler.headers = {'X-Job-Store-Id': 'other-store'}
        handler.close_connection = False
        handler.responses = []
        handler._send_json = lambda status, body: handler.responses.append((int(status), body))
        with patch.object(api, '_async_manager', return_value=manager), patch.object(api, 'WORK_DIR', Path(root)):
            handler._handle_download('/download/job/result.mp4')
        assert handler.responses[0][0] == 409
        assert handler.responses[0][1]['error_code'] == 'JOB_STORE_MISMATCH'
        assert api._DOWNLOAD_COUNTS.get('job') is None
        manager.owner_file.close()
