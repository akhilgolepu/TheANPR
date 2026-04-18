#!/usr/bin/env python3
"""Split a flat YOLO dataset into train/val folders and generate YAML config."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/plate_dataset"),
        help="Dataset root containing images/ and labels/ directories.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio. Remaining samples go to val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="license_plate",
        help="Single class name to write in YAML.",
    )
    return parser.parse_args()


def ensure_flat_dataset(images_dir: Path, labels_dir: Path) -> None:
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Expected images/ and labels/ directories.")

    nested_images = [p for p in images_dir.iterdir() if p.is_dir()]
    nested_labels = [p for p in labels_dir.iterdir() if p.is_dir()]
    if nested_images or nested_labels:
        raise RuntimeError(
            "Expected a flat dataset in images/ and labels/. "
            "Found existing split folders."
        )


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
    pairs: list[tuple[Path, Path]] = []

    missing_labels: list[str] = []
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        pairs.append((image_path, label_path))

    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise RuntimeError(
            f"Missing labels for {len(missing_labels)} images. Sample: {preview}"
        )

    image_stems = {img.stem for img, _ in pairs}
    extra_labels = sorted(
        [p.name for p in labels_dir.iterdir() if p.is_file() and p.stem not in image_stems]
    )
    if extra_labels:
        preview = ", ".join(extra_labels[:10])
        raise RuntimeError(
            f"Found {len(extra_labels)} labels without images. Sample: {preview}"
        )

    return pairs


def make_split_dirs(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    for path in (train_images, val_images, train_labels, val_labels):
        path.mkdir(parents=True, exist_ok=True)

    return train_images, val_images, train_labels, val_labels


def split_pairs(
    pairs: list[tuple[Path, Path]], train_ratio: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train-ratio must be between 0 and 1.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    train_count = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    if not train_pairs or not val_pairs:
        raise RuntimeError("Split produced an empty train or val set.")

    return train_pairs, val_pairs


def move_pairs(
    pairs: list[tuple[Path, Path]], images_dst: Path, labels_dst: Path
) -> None:
    for image_path, label_path in pairs:
        shutil.move(str(image_path), str(images_dst / image_path.name))
        shutil.move(str(label_path), str(labels_dst / label_path.name))


def write_yaml(dataset_root: Path, class_name: str) -> Path:
    yaml_path = "configs/plate_detection.yaml"
    yaml_text = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: [{class_name}]\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    ensure_flat_dataset(images_dir, labels_dir)
    pairs = collect_pairs(images_dir, labels_dir)
    train_pairs, val_pairs = split_pairs(pairs, args.train_ratio, args.seed)

    train_images, val_images, train_labels, val_labels = make_split_dirs(dataset_root)
    move_pairs(train_pairs, train_images, train_labels)
    move_pairs(val_pairs, val_images, val_labels)

    yaml_path = write_yaml(dataset_root, args.class_name)

    print(f"Dataset root: {dataset_root}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
