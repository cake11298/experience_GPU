"""Generate figures and markdown summary table from benchmark results.

Reads results/disk_benchmark.csv and saves figures to results/figures/.
Also prints a markdown table to stdout.

Usage:
    python experiments/plot_results.py
    python experiments/plot_results.py --csv results/disk_benchmark.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour palette
COLORS = {"nvme": "#2a9d8f", "ssd": "#457b9d", "hdd": "#e76f51"}
DISK_ORDER = ["nvme", "ssd", "hdd"]
DISK_LABELS = {"nvme": "NVMe", "ssd": "SATA SSD", "hdd": "HDD"}


def _load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """Load CSV and compute mean ± std per disk (across all repeats and epochs)."""
    df = pd.read_csv(csv_path)
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
        .reindex(DISK_ORDER)
        .dropna(subset=["throughput_mean"])
        .reset_index()
    )
    return df, agg


def _bar_chart(
    ax: plt.Axes,
    disks: list,
    values: list,
    errors: list,
    ylabel: str,
    title: str,
    hline: float | None = None,
) -> None:
    x = np.arange(len(disks))
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=6,
        color=[COLORS[d] for d in disks],
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DISK_LABELS[d] for d in disks], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if hline is not None:
        ax.axhline(hline, linestyle="--", color="#e63946", linewidth=1.2, label=f"{hline:.0f}%")
        ax.legend(fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) * 0.01),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_throughput(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disks = agg["disk"].tolist()
    _bar_chart(
        ax,
        disks,
        agg["throughput_mean"].tolist(),
        agg["throughput_std"].fillna(0).tolist(),
        ylabel="Throughput (images / sec)",
        title="Data Loading Throughput by Storage Device",
    )
    fig.tight_layout()
    path = out_dir / "throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gpu_utilization(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disks = agg["disk"].tolist()
    _bar_chart(
        ax,
        disks,
        agg["gpu_util_mean"].tolist(),
        agg["gpu_util_std"].fillna(0).tolist(),
        ylabel="Mean GPU Utilization (%)",
        title="GPU Utilization During Training Loop",
        hline=100.0,
    )
    ax.set_ylim(0, 115)
    fig.tight_layout()
    path = out_dir / "gpu_utilization.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_disk_read_speed(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disks = agg["disk"].tolist()
    _bar_chart(
        ax,
        disks,
        agg["read_mb_s_mean"].tolist(),
        agg["read_mb_s_std"].fillna(0).tolist(),
        ylabel="Mean Sequential Read (MB/s)",
        title="Disk Read Throughput During Benchmark",
    )
    fig.tight_layout()
    path = out_dir / "disk_read_speed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_throughput_over_epochs(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart: throughput_img_s vs epoch, one line per disk."""
    epoch_agg = (
        df.groupby(["disk", "epoch"])["throughput_img_s"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for disk in DISK_ORDER:
        sub = epoch_agg[epoch_agg["disk"] == disk]
        if sub.empty:
            continue
        epochs = sub["epoch"].tolist()
        means = sub["mean"].tolist()
        stds = sub["std"].fillna(0).tolist()
        ax.plot(epochs, means, marker="o", label=DISK_LABELS[disk],
                color=COLORS[disk], linewidth=2, markersize=7)
        ax.fill_between(
            epochs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=COLORS[disk],
            alpha=0.15,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Throughput (images / sec)", fontsize=11)
    ax.set_title("Throughput per Epoch (Warmup / Caching Effects)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(sorted(df["epoch"].unique()))

    fig.tight_layout()
    path = out_dir / "throughput_over_epochs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_markdown_table(agg: pd.DataFrame) -> None:
    """Print a markdown-formatted results table."""
    nvme_row = agg[agg["disk"] == "nvme"]
    baseline = nvme_row["throughput_mean"].values[0] if len(nvme_row) > 0 else None

    header = (
        "| Disk | Throughput (img/s) | GPU util (%) | Disk read (MB/s) | vs NVMe |\n"
        "|------|-------------------|--------------|------------------|---------|\n"
    )

    rows = ""
    for _, r in agg.iterrows():
        thr = f"{r.throughput_mean:.1f} ± {r.throughput_std:.1f}" if pd.notna(r.throughput_std) else f"{r.throughput_mean:.1f}"
        gpu = f"{r.gpu_util_mean:.1f} ± {r.gpu_util_std:.1f}" if pd.notna(r.gpu_util_std) else f"{r.gpu_util_mean:.1f}"
        read = f"{r.read_mb_s_mean:.1f} ± {r.read_mb_s_std:.1f}" if pd.notna(r.read_mb_s_std) else f"{r.read_mb_s_mean:.1f}"

        if baseline is not None and r["disk"] == "nvme":
            vs = "baseline"
        elif baseline is not None and baseline > 0:
            pct = (r.throughput_mean - baseline) / baseline * 100
            vs = f"{pct:+.1f}%"
        else:
            vs = "N/A"

        rows += f"| {DISK_LABELS.get(r['disk'], r['disk'])} | {thr} | {gpu} | {read} | {vs} |\n"

    print("\n## Results Summary\n")
    print(header + rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent.parent / "results" / "disk_benchmark.csv"),
        help="Path to results CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: Results file not found: {csv_path}", file=sys.stderr)
        print("Run `python experiments/run_benchmark.py` first.", file=sys.stderr)
        sys.exit(1)

    out_dir = csv_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, agg = _load_and_aggregate(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Disks: {agg['disk'].tolist()}")
    print(f"Saving figures to: {out_dir}\n")

    plot_throughput(agg, out_dir)
    plot_gpu_utilization(agg, out_dir)
    plot_disk_read_speed(agg, out_dir)
    plot_throughput_over_epochs(df, out_dir)

    print_markdown_table(agg)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
