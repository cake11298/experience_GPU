"""Core benchmark runner.

One run = iterate full dataset N epochs with real GPU compute (ResNet-18 forward pass).

Cold vs Warm semantics
----------------------
Before every repeat, the OS page cache is dropped (via drop_caches()).
This makes epoch 0 of each repeat a **cold** read — data must come from physical
disk.  Subsequent epochs (1, 2 …) are **warm**: the OS page cache is populated,
so all disks converge to RAM speed and GPU starvation disappears.

The expected pattern:
  Cold epoch  — NVMe: high GPU util;  HDD: very low GPU util  (disk bottleneck)
  Warm epochs — all disks converge to near-identical GPU util  (RAM masks disk)
"""

import subprocess
import time
import warnings
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class BenchmarkConfig:
    disk_name: str              # "nvme" | "ssd" | "hdd"
    data_root: str              # path to dataset
    disk_device: str            # kernel device name e.g. "nvme0n1", "sdb", "sdd"
    epochs: int = 3
    batch_size: int = 32
    num_workers: int = 4
    dry_run: bool = False
    n_repeats: int = 3          # repeat full experiment N times, report mean ± std
    drop_caches: bool = True    # drop OS page cache before each repeat


@dataclass
class RunResult:
    disk_name: str
    repeat_idx: int
    epoch: int
    cold: bool                  # True = cache was dropped before this repeat
    elapsed_s: float
    throughput_img_s: float
    gpu_stats: dict
    disk_stats: dict


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def drop_page_cache(dry_run: bool = False) -> bool:
    """Drop OS page cache to ensure cold disk reads at epoch 0.

    Requires passwordless sudo for the drop_caches command.
    Run scripts/setup_sudoers.sh once to configure this.

    Returns True if successful (or dry_run), False on failure (warning only).
    """
    if dry_run:
        print("  [dry-run] Would drop page cache "
              "(sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches')")
        return True

    try:
        result = subprocess.run(
            ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if result.returncode == 0:
            print("  Page cache dropped (cold read guaranteed).")
            return True
        warnings.warn(
            f"drop_caches failed (rc={result.returncode}): {result.stderr.strip()}\n"
            "Run scripts/setup_sudoers.sh to configure passwordless sudo.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    except Exception as exc:
        warnings.warn(
            f"drop_caches error: {exc}\n"
            "Continuing without cache drop — epoch 0 may not be cold.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# Dry-run synthetic data (cold/warm physics modelled)
# ---------------------------------------------------------------------------

# Cold  = epoch 0 after cache drop: disk is the bottleneck
# Warm  = epoch 1+ : OS page cache serves the data, disk speed irrelevant
_COLD_PARAMS = {
    #          (thr_img_s, gpu_util%, disk_read_MB_s)
    "nvme": (305.0, 68.0, 478.0),   # NVMe cold: fast enough, mild starvation
    "ssd":  (265.0, 61.0, 298.0),   # SATA SSD cold: noticeable starvation
    "hdd":  ( 35.0, 17.0,  88.0),   # HDD cold: severe GPU starvation
}
_WARM_PARAMS = {
    # Warm: data served from RAM (~10 GB/s+), disk reads drop to near-zero
    # All disks converge because they all read from the same page cache
    "nvme": (318.0, 72.0,  4.0),
    "ssd":  (314.0, 71.0,  3.5),
    "hdd":  (311.0, 70.0,  3.0),
}


def _dry_run_epoch(epoch: int, disk_name: str, repeat_idx: int) -> RunResult:
    """Return a plausible synthetic RunResult modelling cold/warm cache physics."""
    import random
    rng = random.Random(hash(f"{disk_name}-{repeat_idx}-{epoch}") & 0xFFFFFFFF)

    is_cold = (epoch == 0)
    params = _COLD_PARAMS if is_cold else _WARM_PARAMS
    base_thr, base_gpu, base_disk = params.get(disk_name, (200.0, 55.0, 200.0))

    noise = lambda base, pct=0.04: max(0.1, rng.gauss(base, base * pct))  # noqa: E731

    throughput = noise(base_thr)
    n_images = 20 * 200  # 4000 images
    elapsed = n_images / throughput
    gpu_util = min(100.0, max(0.0, noise(base_gpu, 0.06)))
    disk_read = max(0.0, noise(base_disk, 0.08))

    return RunResult(
        disk_name=disk_name,
        repeat_idx=repeat_idx,
        epoch=epoch,
        cold=is_cold,
        elapsed_s=elapsed,
        throughput_img_s=throughput,
        gpu_stats={
            "mean_gpu_util": gpu_util,
            "max_gpu_util": min(100.0, gpu_util + rng.uniform(8.0, 18.0)),
            "std_gpu_util": rng.uniform(2.0, 7.0),
            "mean_mem_util": max(0.0, rng.gauss(44.0, 4.0)),
            "max_mem_util": min(100.0, 56.0 + rng.uniform(4.0, 14.0)),
            "mean_mem_used_mb": max(0.0, rng.gauss(3500.0, 150.0)),
            "n_samples": max(1, int(elapsed / 0.5)),
        },
        disk_stats={
            "mean_read_mb_s": disk_read,
            "max_read_mb_s": disk_read * rng.uniform(1.05, 1.20),
            "mean_write_mb_s": max(0.0, rng.gauss(3.0, 1.0)),
            "total_read_mb": n_images * 150 / 1024,
        },
    )


# ---------------------------------------------------------------------------
# Real benchmark
# ---------------------------------------------------------------------------

def run_single(cfg: BenchmarkConfig, repeat_idx: int) -> List[RunResult]:
    """Run one full benchmark (all epochs) for a given repeat index.

    Drops page cache first so epoch 0 is a cold read.

    Steps per epoch:
      1. Build torchvision ImageFolder DataLoader (real I/O, num_workers, pin_memory)
         - transforms: Resize(256), CenterCrop(224), ToTensor(), Normalize(imagenet)
      2. Load ResNet-18 onto GPU in eval() mode
         - real forward pass → genuine GPU utilization; no grad overhead
      3. For each epoch:
         a. Start GPUMonitor + DiskMonitor
         b. Iterate all batches → move to GPU → model(images) → synchronize
         c. Stop monitors, record RunResult
    """
    if cfg.dry_run:
        results = []
        for epoch in range(cfg.epochs):
            time.sleep(0.05)
            results.append(_dry_run_epoch(epoch, cfg.disk_name, repeat_idx))
        return results

    # ---- Real benchmark ----
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch / torchvision not available. Install them or use --dry-run."
        ) from exc

    from .gpu_monitor import GPUMonitor
    from .disk_monitor import DiskMonitor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warnings.warn("CUDA not available — running benchmark on CPU.", RuntimeWarning)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(cfg.data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
    )

    model = models.resnet18(weights=None).to(device)
    model.eval()

    results: List[RunResult] = []
    n_images = len(dataset)

    with torch.no_grad():
        for epoch in range(cfg.epochs):
            gpu_mon = GPUMonitor(interval_s=0.5, dry_run=False)
            disk_mon = DiskMonitor(cfg.disk_device, interval_s=0.5, dry_run=False)

            gpu_mon.start()
            disk_mon.start()
            t0 = time.perf_counter()

            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            gpu_stats = gpu_mon.stop()
            disk_stats = disk_mon.stop()

            throughput = n_images / elapsed
            cold_tag = "[COLD]" if epoch == 0 else "[warm]"
            print(
                f"  [{cfg.disk_name}] repeat={repeat_idx} epoch={epoch}{cold_tag} "
                f"elapsed={elapsed:.1f}s thr={throughput:.1f} img/s "
                f"gpu={gpu_stats['mean_gpu_util']:.1f}% "
                f"disk={disk_stats['mean_read_mb_s']:.1f} MB/s"
            )

            results.append(RunResult(
                disk_name=cfg.disk_name,
                repeat_idx=repeat_idx,
                epoch=epoch,
                cold=(epoch == 0),
                elapsed_s=elapsed,
                throughput_img_s=throughput,
                gpu_stats=gpu_stats,
                disk_stats=disk_stats,
            ))

    return results


def run_benchmark(cfg: BenchmarkConfig) -> pd.DataFrame:
    """Run cfg.n_repeats times with cache drop before each repeat.

    Cache drop before each repeat guarantees:
    - epoch 0 = cold (disk must serve all reads → bottleneck visible on HDD)
    - epoch 1+ = warm (OS page cache populated → all disks converge)
    """
    all_rows = []

    for repeat in range(cfg.n_repeats):
        print(f"\n--- [{cfg.disk_name}] Repeat {repeat + 1}/{cfg.n_repeats} ---")

        # Drop caches before every repeat so epoch 0 is always cold
        if cfg.drop_caches:
            drop_page_cache(dry_run=cfg.dry_run)

        results = run_single(cfg, repeat_idx=repeat)

        for r in results:
            row = {
                "disk": r.disk_name,
                "repeat": r.repeat_idx,
                "epoch": r.epoch,
                "cold": r.cold,
                "elapsed_s": r.elapsed_s,
                "throughput_img_s": r.throughput_img_s,
                "mean_gpu_util": r.gpu_stats.get("mean_gpu_util", 0.0),
                "max_gpu_util": r.gpu_stats.get("max_gpu_util", 0.0),
                "std_gpu_util": r.gpu_stats.get("std_gpu_util", 0.0),
                "mean_mem_util": r.gpu_stats.get("mean_mem_util", 0.0),
                "max_mem_util": r.gpu_stats.get("max_mem_util", 0.0),
                "mean_mem_used_mb": r.gpu_stats.get("mean_mem_used_mb", 0.0),
                "mean_read_mb_s": r.disk_stats.get("mean_read_mb_s", 0.0),
                "max_read_mb_s": r.disk_stats.get("max_read_mb_s", 0.0),
                "total_read_mb": r.disk_stats.get("total_read_mb", 0.0),
            }
            all_rows.append(row)

    return pd.DataFrame(all_rows)
