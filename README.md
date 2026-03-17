# GPU Storage Benchmark

Measures the impact of storage device speed (NVMe, SATA SSD, HDD) on GPU utilization and
data-loading throughput during a PyTorch training loop.

---

## Hardware Setup

| Component | Spec |
|-----------|------|
| CPU       | Intel i7-8700K |
| GPU       | NVIDIA GTX 1080 8GB |
| Driver    | 535 / CUDA 12.2 |
| RAM       | 32 GB |
| OS        | Ubuntu 20.04 (kernel 5.15) |
| Conda env | `imrfit` — Python 3.10, PyTorch 2.5.1+cu121, torchvision |

## Storage Devices

| Label  | Mount point              | Device      | Type      | Notes          |
|--------|--------------------------|-------------|-----------|----------------|
| NVMe   | `/home/cake11298/datasets` | `nvme0n1` | NVMe SSD  | Root partition, **baseline** |
| SSD    | `/mnt/ssd/datasets`      | `sdb5`      | SATA SSD  | ~49 GB         |
| HDD    | `/mnt/hdd/datasets`      | `sdd3`      | HDD       | 98 GB          |

---

## Quick Start

```bash
# 1. Clone and setup env
git clone <repo>
cd experience_GPU
conda activate imrfit
pip install -r requirements.txt

# 2. Dry-run to verify everything works (no real I/O)
bash scripts/run_all.sh --dry-run

# 3. Generate datasets (~600 MB per disk location)
bash scripts/setup_datasets.sh

# 4. Run benchmark (3 epochs × 3 repeats per disk ≈ 15–30 min)
python experiments/run_benchmark.py --epochs 3 --repeats 3

# 5. Plot results
python experiments/plot_results.py
# Figures saved to results/figures/
# Markdown table printed to stdout
```

### Run a single disk

```bash
python experiments/run_benchmark.py --only ssd --epochs 5
```

---

## What It Measures

Three core metrics are captured **per epoch** for each storage device:

| Metric | Source | Description |
|--------|--------|-------------|
| **Throughput (img/s)** | Wall-clock time | Images processed per second through DataLoader + GPU |
| **GPU utilization (%)** | `nvidia-smi` (0.5 s poll) | Mean GPU core busy % during the epoch |
| **Disk read (MB/s)** | `psutil.disk_io_counters` (0.5 s poll) | Mean sequential read throughput |

**Why NVMe is the baseline:** NVMe PCIe bandwidth (>3 GB/s) is fast enough that the DataLoader
never stalls the GPU. SATA SSD (~500 MB/s sequential) introduces mild starvation; HDD (~100 MB/s)
creates severe I/O bottlenecks that leave the GPU idle most of the time.

---

## Results Interpretation

```
NVMe → GPU util ~70%   → GPU is the bottleneck (good — storage is fast enough)
SSD  → GPU util ~65%   → Mild I/O starvation; DataLoader occasionally starves GPU
HDD  → GPU util ~30%   → Severe I/O bottleneck — GPU waits for disk most of the time
```

The key insight: **low GPU utilization with slow storage ≠ model efficiency**. It means your
training loop is I/O-bound; upgrading storage or increasing `num_workers` / prefetch can
recover that GPU utilization without changing the model at all.

---

## Experiment Design

- **Dataset:** 20 classes × 200 JPEG images × ~150 KB = ~600 MB synthetic ImageNet-style data
- **Model:** ResNet-18 (`torchvision.models.resnet18`) in `eval()` mode
  - Real forward pass ensures genuine GPU work (not just tensor operations)
  - No gradient overhead isolates the I/O bottleneck
- **DataLoader:** `num_workers=4`, `pin_memory=True`, `batch_size=32`
- **Repeats:** 3 independent runs per disk; mean ± std reported

---

## Output Files

```
results/
├── disk_benchmark.csv          ← raw per-epoch data (all repeats)
└── figures/
    ├── throughput.png           ← grouped bar: images/sec per device
    ├── gpu_utilization.png      ← grouped bar: mean GPU util per device
    ├── disk_read_speed.png      ← grouped bar: mean read MB/s per device
    └── throughput_over_epochs.png ← line chart showing warmup/caching
```

---

## Running Tests

```bash
pytest tests/ -v
```
