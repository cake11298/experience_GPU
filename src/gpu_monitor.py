"""Background GPU utilization monitor using nvidia-smi.

Polls nvidia-smi every interval_s seconds in a daemon thread.
"""

import shutil
import subprocess
import threading
import time
import warnings
from typing import List, Tuple


class GPUMonitor:
    """Sample GPU utilization in a background thread."""

    def __init__(self, interval_s: float = 0.5, dry_run: bool = False):
        self.interval_s = interval_s
        self.dry_run = dry_run
        self._samples: List[Tuple[float, float, float]] = []  # (gpu%, mem%, mem_mb)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

        if not self._has_nvidia_smi and not dry_run:
            warnings.warn(
                "nvidia-smi not found — GPU monitor will return synthetic values.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 3:
                        gpu_util = float(parts[0].strip())
                        mem_util = float(parts[1].strip())
                        mem_used = float(parts[2].strip())
                        self._samples.append((gpu_util, mem_util, mem_used))
            except Exception:
                pass
            self._stop_event.wait(timeout=self.interval_s)

    def _synthetic_poll_loop(self) -> None:
        """Simulate realistic GPU samples for dry-run / no nvidia-smi."""
        import random
        rng = random.Random(99)
        while not self._stop_event.is_set():
            gpu_util = max(0.0, min(100.0, rng.gauss(68.0, 8.0)))
            mem_util = max(0.0, min(100.0, rng.gauss(45.0, 5.0)))
            mem_used = max(0.0, rng.gauss(3500.0, 200.0))
            self._samples.append((gpu_util, mem_util, mem_used))
            self._stop_event.wait(timeout=self.interval_s)

    def start(self) -> None:
        """Launch background sampling thread."""
        self._samples.clear()
        self._stop_event.clear()

        if self.dry_run or not self._has_nvidia_smi:
            target = self._synthetic_poll_loop
        else:
            target = self._poll_loop

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and return aggregated statistics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if not self._samples:
            return {
                "mean_gpu_util": 0.0,
                "max_gpu_util": 0.0,
                "std_gpu_util": 0.0,
                "mean_mem_util": 0.0,
                "max_mem_util": 0.0,
                "mean_mem_used_mb": 0.0,
                "n_samples": 0,
            }

        gpu_utils = [s[0] for s in self._samples]
        mem_utils = [s[1] for s in self._samples]
        mem_useds = [s[2] for s in self._samples]

        import statistics
        return {
            "mean_gpu_util": statistics.mean(gpu_utils),
            "max_gpu_util": max(gpu_utils),
            "std_gpu_util": statistics.stdev(gpu_utils) if len(gpu_utils) > 1 else 0.0,
            "mean_mem_util": statistics.mean(mem_utils),
            "max_mem_util": max(mem_utils),
            "mean_mem_used_mb": statistics.mean(mem_useds),
            "n_samples": len(self._samples),
        }
