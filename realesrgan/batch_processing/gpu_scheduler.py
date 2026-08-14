import math
import threading
import time
from dataclasses import dataclass

import torch


@dataclass
class GPUState:
    gpu_id: int
    total_memory_mb: float = math.inf
    free_memory_mb: float = math.inf
    allocated_memory_mb: float = 0.0
    active_jobs: int = 0
    last_assigned_index: int = 0


class GPUScheduler:
    """Fair GPU scheduler with lightweight memory-aware assignment."""

    def __init__(self, gpu_ids=None, min_free_memory_mb=256, stats_provider=None):
        self._gpu_ids = list(gpu_ids) if gpu_ids is not None else self._detect_gpu_ids()
        if not self._gpu_ids:
            # keep a synthetic CPU worker id so the backend remains functional on CPU-only machines
            self._gpu_ids = [0]
        self._min_free_memory_mb = float(min_free_memory_mb)
        self._stats_provider = stats_provider
        self._states = {gpu_id: GPUState(gpu_id=gpu_id) for gpu_id in self._gpu_ids}
        self._assign_counter = 0
        self._lock = threading.Lock()
        self.refresh()

    @staticmethod
    def _detect_gpu_ids():
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    def refresh(self):
        with self._lock:
            for gpu_id in self._gpu_ids:
                state = self._states[gpu_id]
                total, free, allocated = self._read_gpu_stats(gpu_id)
                state.total_memory_mb = total
                state.free_memory_mb = free
                state.allocated_memory_mb = allocated

    def _read_gpu_stats(self, gpu_id):
        if self._stats_provider is not None:
            return self._stats_provider(gpu_id)

        if not torch.cuda.is_available():
            return math.inf, math.inf, 0.0

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            allocated_bytes = torch.cuda.memory_allocated(gpu_id)
            mb = 1024 * 1024
            return total_bytes / mb, free_bytes / mb, allocated_bytes / mb
        except Exception:
            return math.inf, math.inf, 0.0

    def acquire_gpu(self, timeout=None, poll_interval=0.05):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            gpu_id = self.try_acquire_gpu()
            if gpu_id is not None:
                return gpu_id
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('No GPU available before timeout')
            time.sleep(poll_interval)

    def try_acquire_gpu(self):
        self.refresh()
        with self._lock:
            candidates = [
                state for state in self._states.values() if state.free_memory_mb >= self._min_free_memory_mb
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda s: (s.active_jobs, s.last_assigned_index, s.gpu_id))
            selected = candidates[0]
            self._assign_counter += 1
            selected.last_assigned_index = self._assign_counter
            selected.active_jobs += 1
            return selected.gpu_id

    def release_gpu(self, gpu_id):
        with self._lock:
            if gpu_id in self._states and self._states[gpu_id].active_jobs > 0:
                self._states[gpu_id].active_jobs -= 1

    def get_status(self):
        self.refresh()
        with self._lock:
            return {
                gpu_id: {
                    'active_jobs': state.active_jobs,
                    'total_memory_mb': state.total_memory_mb,
                    'free_memory_mb': state.free_memory_mb,
                    'allocated_memory_mb': state.allocated_memory_mb,
                }
                for gpu_id, state in self._states.items()
            }
