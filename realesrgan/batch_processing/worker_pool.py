import multiprocessing as mp
import queue
import time

from .inference_batch import InferenceBatchEngine


def _worker_loop(gpu_id, input_queue, output_queue, stop_event, worker_config):
    device = f'cuda:{gpu_id}' if worker_config.get('use_cuda', False) else 'cpu'
    engine = InferenceBatchEngine(
        model_name=worker_config.get('model_name', 'identity'),
        device=device,
        max_retries=worker_config.get('max_retries', 1),
    )

    batch_size = max(1, int(worker_config.get('batch_size', 1)))
    batch_wait_ms = max(1, int(worker_config.get('batch_wait_ms', 25)))
    batch_wait_s = batch_wait_ms / 1000.0

    while not stop_event.is_set():
        try:
            first_job = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if first_job is None:
            break

        jobs = [first_job]
        deadline = time.monotonic() + batch_wait_s
        while len(jobs) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_job = input_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_job is None:
                stop_event.set()
                break
            jobs.append(next_job)

        active_jobs = []
        now = time.monotonic()
        for job in jobs:
            timeout = job.get('timeout')
            created_at = job.get('created_at', now)
            if timeout is not None and created_at + timeout < now:
                output_queue.put({
                    'job_id': job['job_id'],
                    'gpu_id': gpu_id,
                    'state': 'failed',
                    'error': 'job timed out before processing',
                })
            else:
                active_jobs.append(job)

        if not active_jobs:
            continue

        for item in engine.process_batch(active_jobs):
            item['gpu_id'] = gpu_id
            output_queue.put(item)


class WorkerPool:
    """Persistent worker processes pinned to specific GPUs."""

    def __init__(self,
                 gpu_ids,
                 model_name='identity',
                 num_workers=None,
                 batch_size_per_gpu=None,
                 batch_wait_ms=25,
                 max_retries=1,
                 use_cuda=True,
                 mp_context='spawn'):
        self.gpu_ids = list(gpu_ids)
        self.num_workers = num_workers or len(self.gpu_ids)
        self.batch_size_per_gpu = batch_size_per_gpu or {}
        self.batch_wait_ms = batch_wait_ms
        self.mp = mp.get_context(mp_context)
        self.stop_event = self.mp.Event()
        self.result_queue = self.mp.Queue()
        self.input_queues = {}
        self.processes = {}
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_cuda = use_cuda

    def start(self):
        for gpu_id in self.gpu_ids[:self.num_workers]:
            input_queue = self.mp.Queue()
            self.input_queues[gpu_id] = input_queue
            worker_config = {
                'model_name': self.model_name,
                'batch_size': self.batch_size_per_gpu.get(gpu_id, self.batch_size_per_gpu.get('default', 1)),
                'batch_wait_ms': self.batch_wait_ms,
                'max_retries': self.max_retries,
                'use_cuda': self.use_cuda,
            }
            process = self.mp.Process(
                target=_worker_loop,
                args=(gpu_id, input_queue, self.result_queue, self.stop_event, worker_config),
                daemon=True)
            process.start()
            self.processes[gpu_id] = process

    def submit(self, gpu_id, job):
        self.input_queues[gpu_id].put(job)

    def get_result(self, timeout=0.1):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self, timeout=5):
        self.stop_event.set()
        for gpu_id, input_queue in self.input_queues.items():
            try:
                input_queue.put_nowait(None)
            except Exception:
                input_queue.put(None)

        for process in self.processes.values():
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
