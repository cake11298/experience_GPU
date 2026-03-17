"""Tests for GPUMonitor."""

import time
import pytest
from src.gpu_monitor import GPUMonitor


EXPECTED_KEYS = {
    "mean_gpu_util",
    "max_gpu_util",
    "std_gpu_util",
    "mean_mem_util",
    "max_mem_util",
    "mean_mem_used_mb",
    "n_samples",
}


def test_dry_run_returns_valid_stats_dict():
    """GPUMonitor dry_run stop() returns a dict with all required keys."""
    mon = GPUMonitor(interval_s=0.1, dry_run=True)
    mon.start()
    time.sleep(0.3)
    stats = mon.stop()
    assert isinstance(stats, dict)
    assert EXPECTED_KEYS == set(stats.keys())


def test_dry_run_values_are_numeric():
    """All stats values are finite floats/ints in dry_run mode."""
    mon = GPUMonitor(interval_s=0.1, dry_run=True)
    mon.start()
    time.sleep(0.3)
    stats = mon.stop()
    for key, val in stats.items():
        if key == "n_samples":
            assert isinstance(val, int) and val >= 0, f"{key} should be non-negative int"
        else:
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
            assert val >= 0.0, f"{key} should be >= 0"


def test_dry_run_gpu_util_in_range():
    """Synthetic GPU utilization should be in [0, 100]."""
    mon = GPUMonitor(interval_s=0.05, dry_run=True)
    mon.start()
    time.sleep(0.4)
    stats = mon.stop()
    assert 0.0 <= stats["mean_gpu_util"] <= 100.0
    assert 0.0 <= stats["max_gpu_util"] <= 100.0


def test_start_stop_lifecycle_no_crash():
    """start() then stop() must not raise exceptions."""
    mon = GPUMonitor(interval_s=0.1, dry_run=True)
    mon.start()
    time.sleep(0.2)
    stats = mon.stop()
    assert stats is not None


def test_stop_without_start_returns_zeros():
    """Calling stop() before start() returns a zeroed stats dict."""
    mon = GPUMonitor(interval_s=0.1, dry_run=True)
    stats = mon.stop()
    assert isinstance(stats, dict)
    assert stats["n_samples"] == 0
    assert stats["mean_gpu_util"] == 0.0


def test_n_samples_increases_with_time():
    """More elapsed time → more samples collected."""
    mon = GPUMonitor(interval_s=0.05, dry_run=True)
    mon.start()
    time.sleep(0.6)
    stats = mon.stop()
    assert stats["n_samples"] >= 3, f"Expected >=3 samples, got {stats['n_samples']}"
