"""Background disk read/write throughput monitor using psutil.

Samples per-device I/O counters every interval_s seconds.
"""

import threading
import time
import warnings
from typing import List, Tuple


class DiskMonitor:
    """Sample disk I/O throughput in a background thread.

    Args:
        device: Kernel device name, e.g. "sdb" for /mnt/ssd, "sdd" for /mnt/hdd,
                "nvme0n1" for NVMe root.
        interval_s: Polling interval in seconds.
        dry_run: If True, return synthetic values without accessing hardware.
    """

    def __init__(self, device: str, interval_s: float = 0.5, dry_run: bool = False):
        self.device = device
        self.interval_s = interval_s
        self.dry_run = dry_run

        self._samples: List[Tuple[float, float]] = []  # (read_mb_s, write_mb_s)
        self._total_read_bytes: int = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._device_available = False

        if not dry_run:
            try:
                import psutil
                counters = psutil.disk_io_counters(perdisk=True)
                self._device_available = device in counters
                if not self._device_available:
                    warnings.warn(
                        f"Disk device '{device}' not found in psutil counters "
                        f"(available: {list(counters.keys())[:8]}). "
                        "DiskMonitor will return synthetic values.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception as exc:
                warnings.warn(
                    f"psutil unavailable ({exc}). DiskMonitor will return synthetic values.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _poll_loop(self) -> None:
        import psutil

        prev_read = None
        prev_write = None
        prev_time = None
        start_read = None

        while not self._stop_event.is_set():
            try:
                counters = psutil.disk_io_counters(perdisk=True)
                if self.device not in counters:
                    self._stop_event.wait(timeout=self.interval_s)
                    continue

                c = counters[self.device]
                now = time.monotonic()

                if start_read is None:
                    start_read = c.read_bytes

                if prev_read is not None:
                    dt = now - prev_time
                    if dt > 0:
                        read_mb_s = (c.read_bytes - prev_read) / dt / 1024 / 1024
                        write_mb_s = (c.write_bytes - prev_write) / dt / 1024 / 1024
                        self._samples.append((max(0.0, read_mb_s), max(0.0, write_mb_s)))

                prev_read = c.read_bytes
                prev_write = c.write_bytes
                prev_time = now

                if start_read is not None:
                    self._total_read_bytes = c.read_bytes - start_read

            except Exception:
                pass

            self._stop_event.wait(timeout=self.interval_s)

    def _synthetic_poll_loop(self) -> None:
        """Simulate realistic disk I/O for dry-run / missing device."""
        import random

        # Synthetic speeds keyed by common device types
        _speed_map = {
            "nvme": (480.0, 30.0),
            "sdb":  (300.0, 25.0),
            "sdd":  (95.0,  20.0),
        }
        # Pick closest match
        mean_read, std_read = _speed_map.get(
            self.device, _speed_map.get(self.device[:3], (200.0, 25.0))
        )
        rng = random.Random(hash(self.device) & 0xFFFF)

        while not self._stop_event.is_set():
            read_mb_s = max(0.0, rng.gauss(mean_read, std_read))
            write_mb_s = max(0.0, rng.gauss(5.0, 2.0))
            self._samples.append((read_mb_s, write_mb_s))
            self._stop_event.wait(timeout=self.interval_s)

    def start(self) -> None:
        """Launch background sampling thread."""
        self._samples.clear()
        self._total_read_bytes = 0
        self._stop_event.clear()

        if self.dry_run or not self._device_available:
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
                "mean_read_mb_s": 0.0,
                "max_read_mb_s": 0.0,
                "mean_write_mb_s": 0.0,
                "total_read_mb": 0.0,
            }

        read_rates = [s[0] for s in self._samples]
        write_rates = [s[1] for s in self._samples]

        import statistics
        return {
            "mean_read_mb_s": statistics.mean(read_rates),
            "max_read_mb_s": max(read_rates),
            "mean_write_mb_s": statistics.mean(write_rates),
            "total_read_mb": self._total_read_bytes / 1024 / 1024,
        }
