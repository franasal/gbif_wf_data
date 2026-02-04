#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install with `pip install pillow`."
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_identifier(identifier: str) -> str:
    return hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:16]


def load_index_map(index_csv: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not index_csv.exists():
        print(f"Warning: missing index CSV: {index_csv}")
        return {}
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            occ = str(row.get("occurrenceKey") or "")
            identifier = row.get("media_identifier") or ""
            if not occ or not identifier:
                continue
            key = (occ, hash_identifier(identifier))
            if key not in mapping:
                mapping[key] = row
    return mapping


def parse_filename(path: Path) -> Optional[Tuple[str, str]]:
    stem = path.stem
    parts = stem.split("__")
    if len(parts) < 4:
        return None
    occ_key = parts[-3]
    id_hash = parts[-2]
    return (occ_key, id_hash)


def ensure_webp(
    src: Path,
    dst: Path,
    max_width: int,
    quality: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if w > max_width:
            new_h = int(h * (max_width / w))
            img = img.resize((max_width, new_h), Image.LANCZOS)
        img.save(dst, format="WEBP", quality=quality, method=6, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build light release ZIP from selected_light folders."
    )
    parser.add_argument(
        "--source-root",
        default="gbif_wf_data/assets/plant_images/gbif_samples",
        help="Root folder containing per-plant folders.",
    )
    parser.add_argument(
        "--selected-folder",
        default="selected_light",
        help="Curated folder name inside each plant folder.",
    )
    parser.add_argument(
        "--out-root",
        default="gbif_wf_data/assets/plant_images/light_build",
        help="Output root for webp images.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        help="Max width for webp conversion.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=82,
        help="WebP quality (0-100).",
    )
    parser.add_argument(
        "--index-csv",
        default="gbif_wf_data/assets/plant_images/gbif_samples/index.csv",
        help="Index CSV from downloader (for metadata lookup).",
    )
    parser.add_argument(
        "--manifest-json",
        default="gbif_wf_data/assets/plant_images/light_build/images_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="gbif_wf_data/assets/plant_images/light_build/images_manifest.csv",
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--zip-path",
        default="gbif_wf_data/assets/plant_images/light_build/gbif_light_1024.zip",
        help="Output ZIP path.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    manifest_json = Path(args.manifest_json)
    manifest_csv = Path(args.manifest_csv)
    zip_path = Path(args.zip_path)
    index_map = load_index_map(Path(args.index_csv))

    images: List[Dict[str, Any]] = []

    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    plant_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    for plant_dir in sorted(plant_dirs):
        selected_dir = plant_dir / args.selected_folder
        if not selected_dir.exists():
            continue
        for src in sorted(selected_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                continue
            parsed = parse_filename(src)
            occ_key = ""
            id_hash = ""
            meta_row: Dict[str, Any] = {}
            if parsed:
                occ_key, id_hash = parsed
                meta_row = index_map.get((occ_key, id_hash), {})

            rel_dir = plant_dir.name
            out_name = f"{src.stem}.webp"
            dst = out_root / rel_dir / out_name
            ensure_webp(src, dst, args.max_width, args.quality)

            image_entry = {
                "plant_slug": rel_dir,
                "source_path": str(src),
                "output_path": str(dst),
                "output_rel": str(dst.relative_to(out_root)),
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "occurrenceKey": occ_key,
                "media_identifier": meta_row.get("media_identifier", ""),
                "media_license_key": meta_row.get("media_license_key", ""),
                "media_license_label": meta_row.get("media_license_label", ""),
                "media_license_url": meta_row.get("media_license_url", ""),
                "media_creator": meta_row.get("media_creator", ""),
                "media_rightsHolder": meta_row.get("media_rightsHolder", ""),
                "media_references": meta_row.get("media_references", ""),
                "attribution_suggestion": meta_row.get("attribution_suggestion", ""),
            }
            images.append(image_entry)

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_width": args.max_width,
        "quality": args.quality,
        "images_count": len(images),
        "images": images,
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    if images:
        fieldnames = sorted({k for r in images for k in r.keys()})
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in images:
                writer.writerow(row)
    else:
        manifest_csv.write_text("")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(manifest_json, arcname="images_manifest.json")
        zf.write(manifest_csv, arcname="images_manifest.csv")
        for file_path in out_root.rglob("*"):
            if file_path.is_file() and file_path != manifest_json and file_path != manifest_csv:
                rel = file_path.relative_to(out_root)
                zf.write(file_path, arcname=str(rel))

    print("Done.")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
