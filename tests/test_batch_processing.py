import time
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

_BATCH_PROCESSING_PATH = Path(__file__).resolve().parents[1] / 'realesrgan' / 'batch_processing'
_SPEC = importlib.util.spec_from_file_location(
    'batch_processing', _BATCH_PROCESSING_PATH / '__init__.py', submodule_search_locations=[str(_BATCH_PROCESSING_PATH)])
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules['batch_processing'] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BatchQueue = _MODULE.BatchQueue
GPUScheduler = _MODULE.GPUScheduler


def _wait_for_state(batch_queue, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = batch_queue.get_job_status(job_id)
        if status['state'] in {'completed', 'failed'}:
            return status
        time.sleep(0.05)
    raise TimeoutError(f'Job {job_id} did not finish in time')


def test_gpu_scheduler_fair_assignment_and_release():

    def fake_stats(_):
        return 16000.0, 12000.0, 1000.0

    scheduler = GPUScheduler(gpu_ids=[0, 1], min_free_memory_mb=512, stats_provider=fake_stats)

    first = scheduler.try_acquire_gpu()
    second = scheduler.try_acquire_gpu()
    third = scheduler.try_acquire_gpu()

    assert (first, second, third) == (0, 1, 0)

    scheduler.release_gpu(0)
    scheduler.release_gpu(1)
    status = scheduler.get_status()
    assert status[0]['active_jobs'] == 1
    assert status[1]['active_jobs'] == 0


def test_batch_queue_priority_processing(tmp_path):
    low_path = tmp_path / 'low.png'
    high_path = tmp_path / 'high.png'
    cv2.imwrite(str(low_path), np.full((8, 8, 3), 32, dtype=np.uint8))
    cv2.imwrite(str(high_path), np.full((8, 8, 3), 224, dtype=np.uint8))

    queue = BatchQueue(model_name='identity', max_batch_size=1, gpu_ids=[0], num_workers=1, mp_context='fork')
    low_job = queue.submit_job(str(low_path), priority='low', timeout=30)
    high_job = queue.submit_job(str(high_path), priority='high', timeout=30)

    queue.start()
    try:
        low_status = _wait_for_state(queue, low_job)
        high_status = _wait_for_state(queue, high_job)
        assert low_status['state'] == 'completed'
        assert high_status['state'] == 'completed'

        completion_order = queue.completed_job_ids()
        assert completion_order[0] == high_job
        assert set(completion_order[:2]) == {high_job, low_job}

        high_result = queue.get_result(high_job)
        assert high_result['output'].shape == (8, 8, 3)
    finally:
        queue.shutdown()


def test_batch_queue_failed_job_state(tmp_path):
    queue = BatchQueue(model_name='identity', max_batch_size=1, gpu_ids=[0], num_workers=1, mp_context='fork')
    queue.start()
    try:
        job_id = queue.submit_job(str(tmp_path / 'missing.png'), priority='normal', timeout=30)
        status = _wait_for_state(queue, job_id)
        assert status['state'] == 'failed'
        assert 'Unable to read image' in status['error']
    finally:
        queue.shutdown()
