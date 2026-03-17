"""Generate figures and markdown summary table from benchmark results.

Reads results/disk_benchmark.csv and saves figures to results/figures/.
Also prints a markdown table to stdout.

Figures produced
----------------
1. throughput.png              — grouped bar: overall mean img/s per disk
2. gpu_utilization.png         — grouped bar: overall mean GPU util per disk
3. disk_read_speed.png         — grouped bar: mean disk read MB/s per disk
4. throughput_over_epochs.png  — line chart: img/s vs epoch (cold→warm warmup)
5. cold_vs_warm_gpu.png        — grouped bar: cold (epoch 0) vs warm (epoch 1+)
                                  GPU util per disk — the key starvation figure

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


def _load_and_aggregate(csv_path: Path):
    """Load CSV and compute mean ± std per disk (across all repeats and epochs)."""
    df = pd.read_csv(csv_path)
    # Backwards compat: add 'cold' column if missing (old CSVs without cache drop)
    if "cold" not in df.columns:
        df["cold"] = df["epoch"] == 0
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
    ax,
    disks: list,
    values: list,
    errors: list,
    ylabel: str,
    title: str,
    hline: float | None = None,
    ylim: tuple | None = None,
) -> list:
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
        ax.axhline(hline, linestyle="--", color="#e63946", linewidth=1.2,
                   label=f"{hline:.0f}%")
        ax.legend(fontsize=9)

    if ylim is not None:
        ax.set_ylim(*ylim)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(v for v in values if v == v) * 0.01),
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    return bars


def plot_throughput(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disks = agg["disk"].tolist()
    _bar_chart(ax, disks,
               agg["throughput_mean"].tolist(),
               agg["throughput_std"].fillna(0).tolist(),
               ylabel="Throughput (images / sec)",
               title="Data Loading Throughput by Storage Device")
    fig.tight_layout()
    path = out_dir / "throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gpu_utilization(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disks = agg["disk"].tolist()
    _bar_chart(ax, disks,
               agg["gpu_util_mean"].tolist(),
               agg["gpu_util_std"].fillna(0).tolist(),
               ylabel="Mean GPU Utilization (%)",
               title="GPU Utilization During Training Loop",
               hline=100.0, ylim=(0, 115))
    fig.tight_layout()
    path = out_dir / "gpu_utilization.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_disk_read_speed(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disks = agg["disk"].tolist()
    _bar_chart(ax, disks,
               agg["read_mb_s_mean"].tolist(),
               agg["read_mb_s_std"].fillna(0).tolist(),
               ylabel="Mean Sequential Read (MB/s)",
               title="Disk Read Throughput During Benchmark")
    fig.tight_layout()
    path = out_dir / "disk_read_speed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_throughput_over_epochs(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart: img/s vs epoch, one line per disk.

    Epoch 0 (cold) is shaded to highlight the cache-drop transition.
    The key visual: NVMe and SSD stay high across all epochs; HDD jumps
    dramatically from cold (low) to warm (converges with others).
    """
    epoch_agg = (
        df.groupby(["disk", "epoch"])["throughput_img_s"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade the cold epoch region
    ax.axvspan(-0.4, 0.4, color="#ffd6cc", alpha=0.35, zorder=0, label="Cold (cache dropped)")

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
            color=COLORS[disk], alpha=0.15,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Throughput (images / sec)", fontsize=11)
    ax.set_title("Throughput per Epoch — Cold Start vs Warm Cache", fontsize=13,
                 fontweight="bold")
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


def plot_cold_vs_warm_gpu(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar: Cold (epoch 0) vs Warm (epoch 1+) GPU util per disk.

    This is the key figure for demonstrating storage-induced GPU starvation:
    - HDD cold  → very low GPU util  (disk is the bottleneck)
    - HDD warm  → GPU util recovers (page cache serves the data)
    - NVMe/SSD  → minimal difference between cold and warm
    """
    cold_agg = (
        df[df["cold"] == True]
        .groupby("disk")["mean_gpu_util"]
        .agg(["mean", "std"])
        .reindex(DISK_ORDER)
        .dropna(subset=["mean"])
        .reset_index()
    )
    warm_agg = (
        df[df["cold"] == False]
        .groupby("disk")["mean_gpu_util"]
        .agg(["mean", "std"])
        .reindex(DISK_ORDER)
        .dropna(subset=["mean"])
        .reset_index()
    )

    if cold_agg.empty or warm_agg.empty:
        print("  Skipping cold_vs_warm_gpu.png (not enough cold/warm data).")
        return

    # Align disks present in both
    disks = [d for d in DISK_ORDER
             if d in cold_agg["disk"].values and d in warm_agg["disk"].values]

    cold_vals = [cold_agg.loc[cold_agg["disk"] == d, "mean"].values[0] for d in disks]
    cold_errs = [cold_agg.loc[cold_agg["disk"] == d, "std"].fillna(0).values[0] for d in disks]
    warm_vals = [warm_agg.loc[warm_agg["disk"] == d, "mean"].values[0] for d in disks]
    warm_errs = [warm_agg.loc[warm_agg["disk"] == d, "std"].fillna(0).values[0] for d in disks]

    x = np.arange(len(disks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    cold_bars = ax.bar(
        x - width / 2, cold_vals, width,
        yerr=cold_errs, capsize=5,
        color=[COLORS[d] for d in disks],
        alpha=0.55,
        edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
        label="Cold (epoch 0 — cache dropped)",
        hatch="//",
    )
    warm_bars = ax.bar(
        x + width / 2, warm_vals, width,
        yerr=warm_errs, capsize=5,
        color=[COLORS[d] for d in disks],
        alpha=0.95,
        edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
        label="Warm (epoch 1+ — page cache)",
    )

    # Annotate each pair with the recovery delta on HDD
    for i, (disk, cv, wv) in enumerate(zip(disks, cold_vals, warm_vals)):
        delta = wv - cv
        if abs(delta) > 2.0:
            ax.annotate(
                f"+{delta:.0f}%",
                xy=(x[i] + width / 2, wv),
                xytext=(x[i], wv + max(warm_vals) * 0.06),
                ha="center", fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DISK_LABELS[d] for d in disks], fontsize=12)
    ax.set_ylabel("Mean GPU Utilization (%)", fontsize=11)
    ax.set_title(
        "GPU Starvation: Cold Read vs Warm Cache\n"
        "(HDD cold epoch starves GPU; page cache masks disk speed)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 115)
    ax.axhline(100, linestyle="--", color="#e63946", linewidth=1.0, alpha=0.6)
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "cold_vs_warm_gpu.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_markdown_table(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    """Print a markdown-formatted results table including cold/warm breakdown."""
    nvme_row = agg[agg["disk"] == "nvme"]
    baseline = nvme_row["throughput_mean"].values[0] if len(nvme_row) > 0 else None

    # Overall table
    print("\n## Results Summary (all epochs)\n")
    header = (
        "| Disk | Throughput (img/s) | GPU util (%) | Disk read (MB/s) | vs NVMe |\n"
        "|------|-------------------|--------------|------------------|---------|\n"
    )
    rows = ""
    for _, r in agg.iterrows():
        thr = f"{r.throughput_mean:.1f} ± {r.throughput_std:.1f}" if pd.notna(r.throughput_std) else f"{r.throughput_mean:.1f}"
        gpu = f"{r.gpu_util_mean:.1f} ± {r.gpu_util_std:.1f}" if pd.notna(r.gpu_util_std) else f"{r.gpu_util_mean:.1f}"
        rd  = f"{r.read_mb_s_mean:.1f} ± {r.read_mb_s_std:.1f}" if pd.notna(r.read_mb_s_std) else f"{r.read_mb_s_mean:.1f}"
        if baseline is not None and r["disk"] == "nvme":
            vs = "baseline"
        elif baseline is not None and baseline > 0:
            pct = (r.throughput_mean - baseline) / baseline * 100
            vs = f"{pct:+.1f}%"
        else:
            vs = "N/A"
        rows += f"| {DISK_LABELS.get(r['disk'], r['disk'])} | {thr} | {gpu} | {rd} | {vs} |\n"
    print(header + rows)

    # Cold vs Warm GPU util table
    if "cold" in df.columns:
        print("\n## Cold vs Warm GPU Utilization (%)\n")
        print("| Disk | Cold (epoch 0) | Warm (epoch 1+) | Recovery |\n"
              "|------|---------------|-----------------|----------|\n", end="")
        for disk in DISK_ORDER:
            cold_sub = df[(df["disk"] == disk) & (df["cold"] == True)]["mean_gpu_util"]
            warm_sub = df[(df["disk"] == disk) & (df["cold"] == False)]["mean_gpu_util"]
            if cold_sub.empty or warm_sub.empty:
                continue
            cm, cs = cold_sub.mean(), cold_sub.std()
            wm, ws = warm_sub.mean(), warm_sub.std()
            rec = wm - cm
            print(f"| {DISK_LABELS[disk]} | {cm:.1f} ± {cs:.1f} | {wm:.1f} ± {ws:.1f} | +{rec:.1f}% |")
        print()


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
    plot_cold_vs_warm_gpu(df, out_dir)

    print_markdown_table(df, agg)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
