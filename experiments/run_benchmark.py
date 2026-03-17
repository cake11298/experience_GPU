"""Main CLI entrypoint for the GPU storage benchmark.

Usage:
    python experiments/run_benchmark.py --dry-run
    python experiments/run_benchmark.py --epochs 3 --repeats 3 --batch-size 32
    python experiments/run_benchmark.py --only ssd hdd   # skip nvme
"""

import argparse
import sys
from pathlib import Path

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.benchmark import BenchmarkConfig, run_benchmark
from src.dataset_gen import DatasetConfig, generate

DISK_CONFIGS = [
    {
        "disk_name": "nvme",
        "data_root": "/home/cake11298/datasets",
        "disk_device": "nvme0n1",
    },
    {
        "disk_name": "ssd",
        "data_root": "/mnt/ssd/datasets",
        "disk_device": "sdb",
    },
    {
        "disk_name": "hdd",
        "data_root": "/mnt/hdd/datasets",
        "disk_device": "sdd",
    },
]


def _ensure_dataset(data_root: str, dry_run: bool) -> None:
    """Auto-generate dataset if the root doesn't exist or is empty."""
    root = Path(data_root)
    has_data = root.exists() and any(root.rglob("*.jpg"))

    if has_data and not dry_run:
        print(f"  Dataset already exists at {data_root}, skipping generation.")
        return

    print(f"  Auto-generating dataset at {data_root} ...")
    cfg = DatasetConfig(
        root=data_root,
        n_classes=20,
        images_per_class=200,
        target_kb=150,
        dry_run=dry_run,
    )
    generate(cfg)


def _print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    agg = (
        df.groupby("disk")
        .agg(
            throughput_mean=("throughput_img_s", "mean"),
            throughput_std=("throughput_img_s", "std"),
            gpu_util_mean=("mean_gpu_util", "mean"),
            gpu_util_std=("mean_gpu_util", "std"),
            read_mb_s_mean=("mean_read_mb_s", "mean"),
            read_mb_s_std=("mean_read_mb_s", "std"),
        )
        .reset_index()
    )

    # Compute vs NVMe %
    nvme_thr = agg.loc[agg["disk"] == "nvme", "throughput_mean"]
    baseline = nvme_thr.values[0] if len(nvme_thr) > 0 else None

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"{'Disk':<6} {'Throughput (img/s)':<22} {'GPU util (%)':<18} {'Disk read (MB/s)':<20} {'vs NVMe'}"
    print(header)
    print("-" * 80)

    disk_order = ["nvme", "ssd", "hdd"]
    for disk in disk_order:
        row = agg[agg["disk"] == disk]
        if row.empty:
            continue
        r = row.iloc[0]
        thr_str = f"{r.throughput_mean:.1f} ± {r.throughput_std:.1f}" if pd.notna(r.throughput_std) else f"{r.throughput_mean:.1f}"
        gpu_str = f"{r.gpu_util_mean:.1f} ± {r.gpu_util_std:.1f}" if pd.notna(r.gpu_util_std) else f"{r.gpu_util_mean:.1f}"
        read_str = f"{r.read_mb_s_mean:.1f} ± {r.read_mb_s_std:.1f}" if pd.notna(r.read_mb_s_std) else f"{r.read_mb_s_mean:.1f}"

        if baseline is not None and disk == "nvme":
            vs_str = "baseline"
        elif baseline is not None and baseline > 0:
            pct = (r.throughput_mean - baseline) / baseline * 100
            vs_str = f"{pct:+.1f}%"
        else:
            vs_str = "N/A"

        print(f"{disk:<6} {thr_str:<22} {gpu_str:<18} {read_str:<20} {vs_str}")

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Storage Benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without real I/O or GPU usage")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--only", nargs="+", choices=["nvme", "ssd", "hdd"],
                        help="Run only specified disks")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "disk_benchmark.csv"

    all_dfs = []

    for dcfg in DISK_CONFIGS:
        disk_name = dcfg["disk_name"]
        if args.only and disk_name not in args.only:
            print(f"Skipping {disk_name} (not in --only filter)")
            continue

        print(f"\n{'='*60}")
        print(f"Benchmarking: {disk_name.upper()} ({dcfg['data_root']})")
        print(f"{'='*60}")

        # Ensure dataset exists (or simulate in dry-run)
        _ensure_dataset(dcfg["data_root"], dry_run=args.dry_run)

        cfg = BenchmarkConfig(
            disk_name=disk_name,
            data_root=dcfg["data_root"],
            disk_device=dcfg["disk_device"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            n_repeats=args.repeats,
        )

        df = run_benchmark(cfg)
        all_dfs.append(df)

    if not all_dfs:
        print("No disks were benchmarked.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    _print_summary(master_df)

    print("Run `python experiments/plot_results.py` to generate figures")


if __name__ == "__main__":
    main()
