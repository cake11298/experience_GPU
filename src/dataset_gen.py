"""Synthetic ImageNet-style dataset generator.

Generates real JPEG files with random noise to stress disk I/O.
CLI: python src/dataset_gen.py --root /mnt/ssd/datasets --classes 20 --images-per-class 200 --size-kb 150
"""

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


@dataclass
class DatasetConfig:
    root: str
    n_classes: int = 20
    images_per_class: int = 200
    target_kb: int = 150       # ~150KB per image to stress I/O
    image_size: int = 224
    seed: int = 42
    dry_run: bool = False


def _make_jpeg_bytes(rng: random.Random, image_size: int, target_kb: int) -> bytes:
    """Create a random JPEG image of approximately target_kb size."""
    np_rng = np.random.default_rng(rng.randint(0, 2**32))
    # Random RGB noise image
    arr = np_rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")

    import io
    buf = io.BytesIO()
    # Quality 85 → approx 50-200 KB for 224x224; adjust to hit target
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate(cfg: DatasetConfig) -> List[str]:
    """Generate synthetic dataset.

    Returns list of file paths that would be written (useful for dry_run).
    Skips existing files (idempotent).
    """
    rng = random.Random(cfg.seed)
    root = Path(cfg.root)
    paths: List[str] = []
    total_written = 0
    file_count = 0

    for cls_idx in range(cfg.n_classes):
        cls_name = f"class_{cls_idx:04d}"
        cls_dir = root / cls_name
        if not cfg.dry_run:
            cls_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(cfg.images_per_class):
            img_path = cls_dir / f"img_{img_idx:05d}.jpg"
            paths.append(str(img_path))

            if cfg.dry_run:
                file_count += 1
                if file_count % 100 == 0:
                    print(f"  [dry-run] Would write {file_count} files so far...")
                continue

            if img_path.exists():
                file_count += 1
                continue

            jpeg_bytes = _make_jpeg_bytes(rng, cfg.image_size, cfg.target_kb)
            img_path.write_bytes(jpeg_bytes)
            total_written += len(jpeg_bytes)
            file_count += 1

            if file_count % 100 == 0:
                print(f"  Written {file_count}/{cfg.n_classes * cfg.images_per_class} files "
                      f"({total_written / 1024 / 1024:.1f} MB)...")

    if cfg.dry_run:
        expected_mb = cfg.n_classes * cfg.images_per_class * cfg.target_kb / 1024
        print(f"  [dry-run] Would generate {len(paths)} files (~{expected_mb:.0f} MB) at {cfg.root}")
    else:
        total_mb = sum(
            Path(p).stat().st_size for p in paths if Path(p).exists()
        ) / 1024 / 1024
        print(f"  Done. {len(paths)} files, total size: {total_mb:.1f} MB at {cfg.root}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ImageNet-style dataset")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--classes", type=int, default=20)
    parser.add_argument("--images-per-class", type=int, default=200)
    parser.add_argument("--size-kb", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = DatasetConfig(
        root=args.root,
        n_classes=args.classes,
        images_per_class=args.images_per_class,
        target_kb=args.size_kb,
        image_size=args.image_size,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    generate(cfg)


if __name__ == "__main__":
    main()
