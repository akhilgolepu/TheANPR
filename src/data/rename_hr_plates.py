#!/usr/bin/env python3
"""Rename cropped number plate images in a folder to img_01, img_02, ..."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename cropped number plate images to a sequential img_XX pattern."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="/home/akhil/TheANPR/data/raw/hr_plates",
        help="Folder containing the cropped plate images.",
    )
    parser.add_argument(
        "--prefix",
        default="img_",
        help="Filename prefix to use for the renamed images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned renames without changing files.",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> list[Path]:
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(images, key=natural_key)


def build_targets(images: list[Path], prefix: str) -> list[tuple[Path, Path]]:
    width = max(2, len(str(len(images))))
    targets: list[tuple[Path, Path]] = []

    for index, source_path in enumerate(images, start=1):
        target_name = f"{prefix}{index:0{width}d}{source_path.suffix.lower()}"
        targets.append((source_path, source_path.with_name(target_name)))

    return targets


def rename_files(rename_plan: list[tuple[Path, Path]], dry_run: bool) -> None:
    temp_paths: list[tuple[Path, Path]] = []

    for index, (source_path, target_path) in enumerate(rename_plan, start=1):
        temp_path = source_path.with_name(f".__rename_tmp_{index:04d}{source_path.suffix.lower()}")
        temp_paths.append((temp_path, target_path))
        if dry_run:
            print(f"{source_path.name} -> {target_path.name}")
        else:
            source_path.rename(temp_path)

    if dry_run:
        return

    for temp_path, target_path in temp_paths:
        temp_path.rename(target_path)


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory: {folder}")

    images = collect_images(folder)
    if not images:
        print(f"No images found in {folder}")
        return

    rename_plan = build_targets(images, args.prefix)
    existing_targets = {
        target_path.name
        for source_path, target_path in rename_plan
        if target_path.exists() and target_path != source_path
    }
    if existing_targets:
        raise RuntimeError(
            "Target names already exist in the folder: "
            + ", ".join(sorted(existing_targets)[:10])
        )

    rename_files(rename_plan, args.dry_run)
    print(f"Processed {len(rename_plan)} images in {folder}")


if __name__ == "__main__":
    main()