"""Tests for benchmark, disk_monitor, dataset_gen, and plot_results."""

import io
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import BenchmarkConfig, RunResult, run_single
from src.disk_monitor import DiskMonitor
from src.dataset_gen import DatasetConfig, generate


# ── DiskMonitor ──────────────────────────────────────────────────────────────

DISK_STATS_KEYS = {"mean_read_mb_s", "max_read_mb_s", "mean_write_mb_s", "total_read_mb"}


def test_disk_monitor_dry_run_returns_valid_stats():
    """DiskMonitor dry_run returns dict with required keys and non-negative values."""
    mon = DiskMonitor(device="sdb", interval_s=0.1, dry_run=True)
    mon.start()
    time.sleep(0.3)
    stats = mon.stop()
    assert isinstance(stats, dict)
    assert DISK_STATS_KEYS == set(stats.keys())
    for k, v in stats.items():
        assert isinstance(v, float), f"{k} should be float"
        assert v >= 0.0, f"{k} should be >= 0"


def test_disk_monitor_synthetic_read_speed_plausible():
    """Synthetic HDD read speed should be in a plausible range (10–300 MB/s)."""
    mon = DiskMonitor(device="sdd", interval_s=0.05, dry_run=True)
    mon.start()
    time.sleep(0.4)
    stats = mon.stop()
    assert 10.0 <= stats["mean_read_mb_s"] <= 300.0, (
        f"HDD read speed {stats['mean_read_mb_s']:.1f} MB/s out of plausible range"
    )


# ── BenchmarkConfig ───────────────────────────────────────────────────────────

def test_benchmark_config_dataclass_creation():
    """BenchmarkConfig can be instantiated with required fields."""
    cfg = BenchmarkConfig(
        disk_name="ssd",
        data_root="/mnt/ssd/datasets",
        disk_device="sdb",
    )
    assert cfg.disk_name == "ssd"
    assert cfg.epochs == 3       # default
    assert cfg.n_repeats == 3    # default
    assert cfg.dry_run is False  # default


def test_run_single_dry_run_returns_correct_epoch_count():
    """run_single dry_run returns exactly cfg.epochs RunResult objects."""
    cfg = BenchmarkConfig(
        disk_name="nvme",
        data_root="/tmp/fake",
        disk_device="nvme0n1",
        epochs=4,
        dry_run=True,
    )
    results = run_single(cfg, repeat_idx=0)
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"


def test_run_single_dry_run_all_positive():
    """All numeric fields in RunResult are positive in dry_run mode."""
    cfg = BenchmarkConfig(
        disk_name="ssd",
        data_root="/tmp/fake",
        disk_device="sdb",
        epochs=2,
        dry_run=True,
    )
    results = run_single(cfg, repeat_idx=0)
    for r in results:
        assert isinstance(r, RunResult)
        assert r.elapsed_s > 0, "elapsed_s must be > 0"
        assert r.throughput_img_s > 0, "throughput_img_s must be > 0"
        assert r.gpu_stats["mean_gpu_util"] >= 0
        assert r.disk_stats["mean_read_mb_s"] >= 0


# ── dataset_gen ──────────────────────────────────────────────────────────────

def test_dataset_gen_dry_run_no_files_written():
    """dataset_gen dry_run returns expected path list without writing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = DatasetConfig(
            root=tmpdir,
            n_classes=3,
            images_per_class=5,
            dry_run=True,
        )
        paths = generate(cfg)
        assert len(paths) == 15, f"Expected 15 paths, got {len(paths)}"
        # No files should have been written
        written = list(Path(tmpdir).rglob("*.jpg"))
        assert len(written) == 0, f"dry_run wrote {len(written)} files"


# ── plot_results ─────────────────────────────────────────────────────────────

def test_plot_results_loads_minimal_csv(tmp_path):
    """plot_results._load_and_aggregate can handle a minimal CSV without crashing."""
    from experiments.plot_results import _load_and_aggregate

    csv_path = tmp_path / "disk_benchmark.csv"
    # Minimal synthetic data
    rows = []
    for disk in ["nvme", "ssd", "hdd"]:
        for repeat in range(2):
            for epoch in range(2):
                rows.append({
                    "disk": disk, "repeat": repeat, "epoch": epoch,
                    "elapsed_s": 10.0, "throughput_img_s": 300.0 if disk == "nvme" else 100.0,
                    "mean_gpu_util": 70.0, "max_gpu_util": 89.0, "std_gpu_util": 5.0,
                    "mean_mem_util": 45.0, "max_mem_util": 60.0, "mean_mem_used_mb": 3500.0,
                    "mean_read_mb_s": 400.0, "max_read_mb_s": 500.0, "total_read_mb": 600.0,
                })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df, agg = _load_and_aggregate(csv_path)
    assert len(agg) == 3
    assert set(agg["disk"].tolist()) == {"nvme", "ssd", "hdd"}
