import time

import cv2
import numpy as np
import torch


class InferenceBatchEngine:
    """Batched inference engine with retry and memory profiling hooks."""

    def __init__(self, model_name='identity', device='cpu', max_retries=1, model=None):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_retries = int(max_retries)
        self.model = model if model is not None else self._build_model(model_name)
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def _build_model(self, model_name):
        if model_name == 'identity':
            return None
        if model_name == 'chrome_restoration':
            from realesrgan.archs.chrome_restoration_arch import ChromeRestorationNetwork
            return ChromeRestorationNetwork()
        raise ValueError(f'Unsupported model_name: {model_name}')

    def _gpu_memory_snapshot_mb(self):
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return {'allocated_mb': 0.0, 'reserved_mb': 0.0}
        idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
        mb = 1024 * 1024
        return {
            'allocated_mb': torch.cuda.memory_allocated(idx) / mb,
            'reserved_mb': torch.cuda.memory_reserved(idx) / mb,
        }

    def _run_model(self, image):
        if self.model is None:
            return image

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.shape[2] == 1:
            tensor = tensor.repeat(1, 1, 3)
        tensor = tensor[:, :, :3]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.detach().float().clamp_(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        return output

    def process_batch(self, jobs):
        results = []
        for job in jobs:
            attempts = 0
            last_error = None
            while attempts <= self.max_retries:
                attempts += 1
                started = time.monotonic()
                mem_before = self._gpu_memory_snapshot_mb()
                try:
                    image = cv2.imread(job['image_path'], cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise FileNotFoundError(f"Unable to read image: {job['image_path']}")

                    output = self._run_model(image)
                    mem_after = self._gpu_memory_snapshot_mb()
                    results.append({
                        'job_id': job['job_id'],
                        'state': 'completed',
                        'output': output,
                        'latency_s': time.monotonic() - started,
                        'attempts': attempts,
                        'memory_before_mb': mem_before,
                        'memory_after_mb': mem_after,
                    })
                    last_error = None
                    break
                except RuntimeError as error:
                    last_error = error
                    message = str(error).lower()
                    if 'out of memory' in message and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    if attempts > self.max_retries:
                        break
                except Exception as error:
                    last_error = error
                    if attempts > self.max_retries:
                        break

            if last_error is not None:
                results.append({
                    'job_id': job['job_id'],
                    'state': 'failed',
                    'error': str(last_error),
                    'attempts': attempts,
                })

        return results
