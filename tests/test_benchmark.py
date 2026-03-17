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

from src.benchmark import BenchmarkConfig, RunResult, run_single, drop_page_cache
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
    assert cfg.epochs == 3        # default
    assert cfg.n_repeats == 3     # default
    assert cfg.dry_run is False   # default
    assert cfg.drop_caches is True  # default — cache drop enabled


def test_drop_page_cache_dry_run_returns_true():
    """drop_page_cache(dry_run=True) returns True without touching the system."""
    result = drop_page_cache(dry_run=True)
    assert result is True


def test_cold_warm_pattern_in_dry_run():
    """Epoch 0 (cold) has lower throughput than epoch 1 (warm) on HDD."""
    cfg = BenchmarkConfig(
        disk_name="hdd",
        data_root="/tmp/fake",
        disk_device="sdd",
        epochs=3,
        dry_run=True,
    )
    results = run_single(cfg, repeat_idx=0)
    assert results[0].cold is True,  "epoch 0 must be cold"
    assert results[1].cold is False, "epoch 1 must be warm"
    assert results[2].cold is False, "epoch 2 must be warm"
    # Cold HDD should be significantly slower than warm HDD
    assert results[0].throughput_img_s < results[1].throughput_img_s, (
        f"Cold ({results[0].throughput_img_s:.1f}) should be slower than "
        f"warm ({results[1].throughput_img_s:.1f}) on HDD"
    )


def test_warm_disks_converge_in_dry_run():
    """Warm epochs: NVMe, SSD, HDD throughput should converge (all within 20%)."""
    results = {}
    for disk, device in [("nvme", "nvme0n1"), ("ssd", "sdb"), ("hdd", "sdd")]:
        cfg = BenchmarkConfig(
            disk_name=disk, data_root="/tmp/fake",
            disk_device=device, epochs=2, dry_run=True,
        )
        r = run_single(cfg, repeat_idx=0)
        results[disk] = r[1].throughput_img_s  # epoch 1 = warm

    # All warm throughputs should be within 20% of each other
    thr_vals = list(results.values())
    spread = (max(thr_vals) - min(thr_vals)) / max(thr_vals)
    assert spread < 0.20, (
        f"Warm epoch throughputs should converge; spread={spread:.1%} "
        f"(nvme={results['nvme']:.0f}, ssd={results['ssd']:.0f}, hdd={results['hdd']:.0f})"
    )


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
                cold = (epoch == 0)
                thr = {"nvme": 305.0, "ssd": 265.0, "hdd": 35.0}[disk] if cold \
                      else {"nvme": 318.0, "ssd": 314.0, "hdd": 311.0}[disk]
                gpu = {"nvme": 68.0, "ssd": 61.0, "hdd": 17.0}[disk] if cold \
                      else 71.0
                rows.append({
                    "disk": disk, "repeat": repeat, "epoch": epoch, "cold": cold,
                    "elapsed_s": 10.0, "throughput_img_s": thr,
                    "mean_gpu_util": gpu, "max_gpu_util": gpu + 12.0, "std_gpu_util": 4.0,
                    "mean_mem_util": 45.0, "max_mem_util": 60.0, "mean_mem_used_mb": 3500.0,
                    "mean_read_mb_s": 400.0 if cold else 4.0,
                    "max_read_mb_s": 500.0 if cold else 8.0, "total_read_mb": 600.0,
                })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df, agg = _load_and_aggregate(csv_path)
    assert len(agg) == 3
    assert set(agg["disk"].tolist()) == {"nvme", "ssd", "hdd"}
