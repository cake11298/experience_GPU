"""Core benchmark runner.

One run = iterate full dataset N epochs with real GPU compute (ResNet-18 forward pass).
"""

import time
import warnings
from dataclasses import dataclass, field
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


@dataclass
class RunResult:
    disk_name: str
    repeat_idx: int
    epoch: int
    elapsed_s: float
    throughput_img_s: float
    gpu_stats: dict
    disk_stats: dict


def _dry_run_epoch(epoch: int, disk_name: str, repeat_idx: int) -> RunResult:
    """Return a plausible synthetic RunResult without real I/O or GPU."""
    import random
    rng = random.Random(hash(f"{disk_name}-{repeat_idx}-{epoch}") & 0xFFFFFFFF)

    speed_map = {"nvme": 310.0, "ssd": 280.0, "hdd": 90.0}
    base_thr = speed_map.get(disk_name, 200.0)
    throughput = max(1.0, rng.gauss(base_thr, base_thr * 0.03))
    n_images = 20 * 200  # 4000 images
    elapsed = n_images / throughput

    gpu_base = {"nvme": 70.0, "ssd": 67.0, "hdd": 30.0}.get(disk_name, 50.0)
    disk_base = {"nvme": 480.0, "ssd": 305.0, "hdd": 95.0}.get(disk_name, 200.0)

    return RunResult(
        disk_name=disk_name,
        repeat_idx=repeat_idx,
        epoch=epoch,
        elapsed_s=elapsed,
        throughput_img_s=throughput,
        gpu_stats={
            "mean_gpu_util": rng.gauss(gpu_base, 3.0),
            "max_gpu_util": min(100.0, gpu_base + rng.uniform(10.0, 20.0)),
            "std_gpu_util": rng.uniform(2.0, 6.0),
            "mean_mem_util": rng.gauss(45.0, 4.0),
            "max_mem_util": min(100.0, 55.0 + rng.uniform(5.0, 15.0)),
            "mean_mem_used_mb": rng.gauss(3500.0, 150.0),
            "n_samples": int(elapsed / 0.5),
        },
        disk_stats={
            "mean_read_mb_s": rng.gauss(disk_base, disk_base * 0.05),
            "max_read_mb_s": disk_base * 1.15,
            "mean_write_mb_s": rng.gauss(5.0, 1.0),
            "total_read_mb": n_images * 150 / 1024,
        },
    )


def run_single(cfg: BenchmarkConfig, repeat_idx: int) -> List[RunResult]:
    """Run one full benchmark (all epochs) for a given repeat index.

    Steps per epoch:
      1. Build torchvision ImageFolder DataLoader (real I/O)
      2. Load ResNet-18 onto GPU in eval() mode
      3. Forward-pass every batch, recording GPU + disk stats
    """
    if cfg.dry_run:
        results = []
        for epoch in range(cfg.epochs):
            time.sleep(0.05)   # simulate a tiny bit of work
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
            print(
                f"  [{cfg.disk_name}] repeat={repeat_idx} epoch={epoch} "
                f"elapsed={elapsed:.1f}s thr={throughput:.1f} img/s "
                f"gpu={gpu_stats['mean_gpu_util']:.1f}% "
                f"disk={disk_stats['mean_read_mb_s']:.1f} MB/s"
            )

            results.append(RunResult(
                disk_name=cfg.disk_name,
                repeat_idx=repeat_idx,
                epoch=epoch,
                elapsed_s=elapsed,
                throughput_img_s=throughput,
                gpu_stats=gpu_stats,
                disk_stats=disk_stats,
            ))

    return results


def run_benchmark(cfg: BenchmarkConfig) -> pd.DataFrame:
    """Run cfg.n_repeats times, aggregate all results into a DataFrame."""
    all_rows = []

    for repeat in range(cfg.n_repeats):
        print(f"\n--- [{cfg.disk_name}] Repeat {repeat + 1}/{cfg.n_repeats} ---")
        results = run_single(cfg, repeat_idx=repeat)
        for r in results:
            row = {
                "disk": r.disk_name,
                "repeat": r.repeat_idx,
                "epoch": r.epoch,
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
