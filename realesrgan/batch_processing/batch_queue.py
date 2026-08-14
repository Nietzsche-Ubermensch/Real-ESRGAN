import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections import deque

import torch

from .gpu_scheduler import GPUScheduler
from .worker_pool import WorkerPool

_PRIORITY_LEVEL = {'high': 0, 'normal': 1, 'low': 2}


class BatchQueue:
    """Asynchronous batch queue manager with priority and status tracking."""

    def __init__(self,
                 model_name='identity',
                 max_batch_size=1,
                 gpu_ids=None,
                 num_workers=None,
                 min_free_memory_mb=256,
                 batch_size_per_gpu=None,
                 worker_batch_wait_ms=25,
                 max_retries=1,
                 mp_context='spawn'):
        self._submission_queues = {
            'high': mp.Queue(),
            'normal': mp.Queue(),
            'low': mp.Queue(),
        }
        self._status_lock = threading.Lock()
        self._jobs = {}
        self._results = {}
        self._completed_order = deque(maxlen=4096)
        self._running = False
        self._dispatcher_thread = None
        self._result_thread = None
        self._stop_event = threading.Event()
        self._metrics = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'started_at': None,
            'total_latency_s': 0.0,
        }

        self.gpu_scheduler = GPUScheduler(gpu_ids=gpu_ids, min_free_memory_mb=min_free_memory_mb)
        if batch_size_per_gpu is None:
            batch_size_per_gpu = {'default': max_batch_size}
        self.worker_pool = WorkerPool(
            gpu_ids=list(self.gpu_scheduler.get_status().keys()),
            model_name=model_name,
            num_workers=num_workers,
            batch_size_per_gpu=batch_size_per_gpu,
            batch_wait_ms=worker_batch_wait_ms,
            max_retries=max_retries,
            use_cuda=torch.cuda.is_available(),
            mp_context=mp_context,
        )

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._metrics['started_at'] = time.monotonic()
        self.worker_pool.start()
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._result_thread = threading.Thread(target=self._result_loop, daemon=True)
        self._dispatcher_thread.start()
        self._result_thread.start()

    def shutdown(self):
        self._running = False
        self._stop_event.set()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=2)
        if self._result_thread is not None:
            self._result_thread.join(timeout=2)
        self.worker_pool.shutdown()

    def submit_job(self, image_path, priority='normal', timeout=300, metadata=None):
        if priority not in _PRIORITY_LEVEL:
            raise ValueError(f'Unsupported priority: {priority}')

        job_id = uuid.uuid4().hex
        now = time.monotonic()
        payload = {
            'job_id': job_id,
            'image_path': image_path,
            'priority': priority,
            'timeout': timeout,
            'metadata': metadata or {},
            'created_at': now,
        }
        with self._status_lock:
            self._jobs[job_id] = {
                'state': 'pending',
                'priority': priority,
                'created_at': now,
                'image_path': image_path,
                'timeout': timeout,
                'attempts': 0,
                'gpu_id': None,
                'error': None,
            }
            self._metrics['submitted'] += 1
        self._submission_queues[priority].put(payload)
        return job_id

    def get_job_status(self, job_id):
        with self._status_lock:
            status = self._jobs.get(job_id)
            if status is None:
                return {'state': 'missing'}
            return dict(status)

    def get_result(self, job_id, remove=True):
        with self._status_lock:
            if job_id not in self._results:
                return None
            result = self._results[job_id]
            if remove:
                del self._results[job_id]
            return result

    def get_metrics(self):
        with self._status_lock:
            elapsed = 0.0
            if self._metrics['started_at'] is not None:
                elapsed = max(time.monotonic() - self._metrics['started_at'], 1e-6)
            completed = self._metrics['completed']
            throughput = completed / elapsed if elapsed else 0.0
            avg_latency = self._metrics['total_latency_s'] / completed if completed else 0.0
            return {
                'submitted': self._metrics['submitted'],
                'completed': completed,
                'failed': self._metrics['failed'],
                'throughput_jobs_per_sec': throughput,
                'avg_latency_s': avg_latency,
                'gpu_status': self.gpu_scheduler.get_status(),
            }

    def completed_job_ids(self):
        with self._status_lock:
            return list(self._completed_order)

    def _dispatch_loop(self):
        while not self._stop_event.is_set():
            job = self._dequeue_next_job()
            if job is None:
                time.sleep(0.01)
                continue

            timeout = job.get('timeout')
            now = time.monotonic()
            if timeout is not None and job['created_at'] + timeout < now:
                self._mark_failed(job['job_id'], 'job timed out while waiting in queue')
                continue

            gpu_id = self.gpu_scheduler.try_acquire_gpu()
            if gpu_id is None:
                self._submission_queues[job['priority']].put(job)
                time.sleep(0.01)
                continue

            with self._status_lock:
                if job['job_id'] in self._jobs:
                    self._jobs[job['job_id']]['state'] = 'processing'
                    self._jobs[job['job_id']]['gpu_id'] = gpu_id
                    self._jobs[job['job_id']]['started_at'] = time.monotonic()

            self.worker_pool.submit(gpu_id, job)

    def _dequeue_next_job(self):
        for name in ('high', 'normal', 'low'):
            try:
                return self._submission_queues[name].get_nowait()
            except queue.Empty:
                continue
        return None

    def _result_loop(self):
        while not self._stop_event.is_set():
            item = self.worker_pool.get_result(timeout=0.1)
            if item is None:
                continue

            job_id = item['job_id']
            gpu_id = item.get('gpu_id')
            if gpu_id is not None:
                self.gpu_scheduler.release_gpu(gpu_id)

            with self._status_lock:
                if job_id not in self._jobs:
                    continue

                if item['state'] == 'completed':
                    self._jobs[job_id]['state'] = 'completed'
                    self._jobs[job_id]['completed_at'] = time.monotonic()
                    self._jobs[job_id]['attempts'] = item.get('attempts', 1)
                    self._results[job_id] = {
                        'output': item['output'],
                        'latency_s': item.get('latency_s'),
                        'memory_before_mb': item.get('memory_before_mb'),
                        'memory_after_mb': item.get('memory_after_mb'),
                    }
                    self._metrics['completed'] += 1
                    self._metrics['total_latency_s'] += item.get('latency_s', 0.0)
                    self._completed_order.append(job_id)
                else:
                    self._jobs[job_id]['state'] = 'failed'
                    self._jobs[job_id]['completed_at'] = time.monotonic()
                    self._jobs[job_id]['error'] = item.get('error', 'unknown error')
                    self._jobs[job_id]['attempts'] = item.get('attempts', 1)
                    self._metrics['failed'] += 1
                    self._completed_order.append(job_id)

    def _mark_failed(self, job_id, error):
        with self._status_lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]['state'] = 'failed'
            self._jobs[job_id]['error'] = error
            self._jobs[job_id]['completed_at'] = time.monotonic()
            self._metrics['failed'] += 1
            self._completed_order.append(job_id)
