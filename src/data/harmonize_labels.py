import argparse
import csv
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
DEST_ROOT = PROC_DIR / "license_dataset"
DEST_IMAGES = DEST_ROOT / "images"
DEST_LABELS = DEST_ROOT / "labels"
REPORT_PATH = DEST_ROOT / "harmonize_report.json"
OCR_CSV_PATH = DEST_ROOT / "ocr_registry.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class Stats:
    images_written: int = 0
    labels_written: int = 0
    skipped_unlabeled: int = 0
    skipped_invalid_label: int = 0
    skipped_missing_image: int = 0
    skipped_artifacts: int = 0
    renamed_collisions: int = 0


def is_artifact(path: Path) -> bool:
    return ":Zone.Identifier" in path.name


def find_image_by_stem(search_root: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        candidate = search_root / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def unique_dest_stem(base_stem: str, seen: Dict[str, int], stats: Stats) -> str:
    key = sanitize_name(base_stem)
    if key not in seen:
        seen[key] = 0
        return key
    seen[key] += 1
    stats.renamed_collisions += 1
    return f"{key}__dup{seen[key]}"


def validate_yolo_line(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        class_id = int(parts[0])
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return False
    if class_id != 0:
        return False
    x, y, w, h = vals
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def xml_to_yolo(xml_path: Path) -> Tuple[List[str], Optional[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return [], None
    width_text = size.findtext("width")
    height_text = size.findtext("height")
    if not width_text or not height_text:
        return [], None

    img_w = float(width_text)
    img_h = float(height_text)
    if img_w <= 0 or img_h <= 0:
        return [], None

    yolo_boxes: List[str] = []
    plate_text: Optional[str] = None

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
        except ValueError:
            continue

        xmin, xmax = sorted((xmin, xmax))
        ymin, ymax = sorted((ymin, ymax))
        xmin = clamp(xmin, 0.0, img_w)
        xmax = clamp(xmax, 0.0, img_w)
        ymin = clamp(ymin, 0.0, img_h)
        ymax = clamp(ymax, 0.0, img_h)
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1.0 or bh <= 1.0:
            continue

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        w = bw / img_w
        h = bh / img_h
        yolo_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        attrs = obj.find("attributes")
        if attrs is not None:
            for attr in attrs.findall("attribute"):
                if attr.findtext("name") == "number_plate_text":
                    value = attr.findtext("value")
                    if value:
                        plate_text = value.strip()

    return yolo_boxes, plate_text


def write_sample(
    image_path: Path,
    yolo_lines: List[str],
    dataset_tag: str,
    base_stem: str,
    seen: Dict[str, int],
    stats: Stats,
) -> Optional[str]:
    if not yolo_lines:
        stats.skipped_unlabeled += 1
        return None

    valid = [line for line in yolo_lines if validate_yolo_line(line)]
    if not valid:
        stats.skipped_invalid_label += 1
        return None

    stem = unique_dest_stem(f"{dataset_tag}__{base_stem}", seen, stats)
    ext = image_path.suffix.lower() if image_path.suffix else ".jpg"
    dest_img = DEST_IMAGES / f"{stem}{ext}"
    dest_lbl = DEST_LABELS / f"{stem}.txt"

    shutil.copy2(image_path, dest_img)
    with dest_lbl.open("w", encoding="utf-8") as f:
        f.write("\n".join(valid))

    stats.images_written += 1
    stats.labels_written += 1
    return dest_img.name


def process_indian_plates(stats: Stats, seen: Dict[str, int], ocr_rows: List[Dict[str, str]]) -> None:
    dataset_tag = "indian_plates"
    xml_roots = [
        RAW_DIR / "indian_plates" / "Annotations" / "Annotations",
        RAW_DIR / "indian_plates" / "number_plate_annos_ocr" / "number_plate_annos_ocr",
    ]
    image_roots = [
        RAW_DIR / "indian_plates" / "Indian_Number_Plates" / "Sample_Images",
        RAW_DIR / "indian_plates" / "number_plate_images_ocr" / "number_plate_images_ocr",
    ]

    for xml_root in xml_roots:
        if not xml_root.exists():
            continue
        for xml_path in tqdm(sorted(xml_root.glob("*.xml")), desc=f"{dataset_tag}:{xml_root.name}"):
            if is_artifact(xml_path):
                stats.skipped_artifacts += 1
                continue

            boxes, plate_text = xml_to_yolo(xml_path)
            stem = xml_path.stem
            image_path = None
            for img_root in image_roots:
                image_path = find_image_by_stem(img_root, stem)
                if image_path is not None:
                    break
            if image_path is None:
                stats.skipped_missing_image += 1
                continue

            final_name = write_sample(image_path, boxes, dataset_tag, stem, seen, stats)
            if final_name and plate_text:
                ocr_rows.append({"filename": final_name, "text": plate_text})


def process_voc_tree(dataset_tag: str, root: Path, stats: Stats, seen: Dict[str, int]) -> None:
    if not root.exists():
        return

    for xml_path in tqdm(sorted(root.rglob("*.xml")), desc=f"{dataset_tag}:xml"):
        if is_artifact(xml_path):
            stats.skipped_artifacts += 1
            continue
        boxes, _ = xml_to_yolo(xml_path)

        image_path = None
        filename_from_xml = None
        try:
            filename_from_xml = ET.parse(xml_path).getroot().findtext("filename")
        except ET.ParseError:
            stats.skipped_invalid_label += 1
            continue

        if filename_from_xml:
            candidate = xml_path.parent / filename_from_xml
            if candidate.exists() and candidate.suffix.lower() in IMG_EXTS:
                image_path = candidate

        if image_path is None:
            image_path = find_image_by_stem(xml_path.parent, xml_path.stem)
        if image_path is None:
            for img in xml_path.parent.glob("*"):
                if img.is_file() and img.suffix.lower() in IMG_EXTS and img.stem == xml_path.stem:
                    image_path = img
                    break

        if image_path is None:
            stats.skipped_missing_image += 1
            continue

        write_sample(image_path, boxes, dataset_tag, xml_path.stem, seen, stats)


def process_yolo_folder(dataset_tag: str, image_dir: Path, label_dir: Path, stats: Stats, seen: Dict[str, int]) -> None:
    if not image_dir.exists() or not label_dir.exists():
        return

    for label_path in tqdm(sorted(label_dir.glob("*.txt")), desc=f"{dataset_tag}:txt"):
        if is_artifact(label_path):
            stats.skipped_artifacts += 1
            continue
        stem = label_path.stem
        image_path = find_image_by_stem(image_dir, stem)
        if image_path is None:
            stats.skipped_missing_image += 1
            continue
        with label_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        write_sample(image_path, lines, dataset_tag, stem, seen, stats)


def process_dataset_archive_2(stats: Stats, seen: Dict[str, int]) -> None:
    root = RAW_DIR / "dataset_archive_2"
    if not root.exists():
        return

    # Prefer YOLO txt if present, otherwise parse XML for the same stem.
    for split in ["train", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue

        labels_dir = split_dir / "labels"
        if labels_dir.exists() and any(labels_dir.glob("*.txt")):
            process_yolo_folder("dataset_archive_2", split_dir, labels_dir, stats, seen)
            continue

        txt_map: Dict[str, Path] = {}
        for txt in split_dir.glob("*.txt"):
            if is_artifact(txt):
                stats.skipped_artifacts += 1
                continue
            txt_map[txt.stem] = txt

        xml_map: Dict[str, Path] = {}
        for xml in split_dir.glob("*.xml"):
            if is_artifact(xml):
                stats.skipped_artifacts += 1
                continue
            xml_map[xml.stem] = xml

        all_stems = sorted(set(txt_map.keys()) | set(xml_map.keys()))
        for stem in tqdm(all_stems, desc=f"dataset_archive_2:{split}"):
            image_path = find_image_by_stem(split_dir, stem)
            if image_path is None:
                stats.skipped_missing_image += 1
                continue

            if stem in txt_map:
                with txt_map[stem].open("r", encoding="utf-8", errors="ignore") as f:
                    lines = [line.strip() for line in f if line.strip()]
                write_sample(image_path, lines, "dataset_archive_2", stem, seen, stats)
            elif stem in xml_map:
                lines, _ = xml_to_yolo(xml_map[stem])
                write_sample(image_path, lines, "dataset_archive_2", stem, seen, stats)


def process_indian_licence_plates_with_labels(stats: Stats, seen: Dict[str, int]) -> None:
    process_yolo_folder(
        "indian_licence_plates_with_labels",
        RAW_DIR / "indian_licence_plates_with_labels" / "images",
        RAW_DIR / "indian_licence_plates_with_labels" / "labels",
        stats,
        seen,
    )


def process_indian_vehicle_license_plate_dataset(stats: Stats, seen: Dict[str, int]) -> None:
    process_voc_tree(
        "indian_vehicle_license_plate_dataset",
        RAW_DIR / "Indian_vehicle_license_plate_dataset",
        stats,
        seen,
    )


def process_indian_license_plates(stats: Stats) -> None:
    # This source is mostly image-only crops and not detection-ready labels.
    root = RAW_DIR / "indian_license_plates"
    if not root.exists():
        return
    image_count = len(
        [
            p for p in root.rglob("*")
            if p.is_file() and not is_artifact(p) and p.suffix.lower() in IMG_EXTS
        ]
    )
    stats.skipped_unlabeled += image_count


def save_reports(stats: Stats, ocr_rows: List[Dict[str, str]]) -> None:
    report = {
        "output_root": str(DEST_ROOT),
        "images_written": stats.images_written,
        "labels_written": stats.labels_written,
        "skipped_unlabeled": stats.skipped_unlabeled,
        "skipped_invalid_label": stats.skipped_invalid_label,
        "skipped_missing_image": stats.skipped_missing_image,
        "skipped_artifacts": stats.skipped_artifacts,
        "renamed_collisions": stats.renamed_collisions,
        "ocr_pairs_written": len(ocr_rows),
    }

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if ocr_rows:
        with OCR_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "text"])
            writer.writeheader()
            writer.writerows(ocr_rows)


def reset_output() -> None:
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)


def harmonize_license_detection_datasets() -> None:
    reset_output()
    stats = Stats()
    seen: Dict[str, int] = {}
    ocr_rows: List[Dict[str, str]] = []

    process_dataset_archive_2(stats, seen)
    process_indian_licence_plates_with_labels(stats, seen)
    process_indian_plates(stats, seen, ocr_rows)
    process_indian_vehicle_license_plate_dataset(stats, seen)
    process_indian_license_plates(stats)

    save_reports(stats, ocr_rows)
    print("License dataset harmonization complete.")
    print(f"Wrote {stats.images_written} images and {stats.labels_written} labels to {DEST_ROOT}")
    print(f"Skipped unlabeled: {stats.skipped_unlabeled}")
    print(f"Skipped invalid labels: {stats.skipped_invalid_label}")
    print(f"Skipped missing image pairs: {stats.skipped_missing_image}")
    print(f"Skipped artifact files: {stats.skipped_artifacts}")
    if ocr_rows:
        print(f"Wrote OCR registry: {OCR_CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize raw plate datasets into YOLO-ready license_dataset")
    parser.parse_args()
    harmonize_license_detection_datasets()