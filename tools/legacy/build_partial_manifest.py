#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a partial manifest from already-downloaded images."
    )
    parser.add_argument(
        "--source-root",
        default="gbif_wf_data/assets/plant_images/gbif_samples",
        help="Root folder containing per-plant folders.",
    )
    parser.add_argument(
        "--manifest-json",
        default="gbif_wf_data/assets/plant_images/gbif_samples/partial_manifest.json",
        help="Output partial manifest JSON path.",
    )
    parser.add_argument(
        "--curated-json",
        default="gbif_wf_data/assets/plant_images/gbif_samples/curated_plants.json",
        help="Output curated plants JSON path.",
    )
    parser.add_argument(
        "--selected-folder",
        default="selected_light",
        help="Folder name used for curated light selections.",
    )
    parser.add_argument(
        "--curated-min",
        type=int,
        default=5,
        help="Min number of images in selected_light to consider a plant curated.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    manifest_json = Path(args.manifest_json)
    curated_json = Path(args.curated_json)

    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    images: List[Dict[str, str]] = []
    curated_plants: List[str] = []

    plant_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    for plant_dir in sorted(plant_dirs):
        plant_slug = plant_dir.name
        selected_dir = plant_dir / args.selected_folder
        if selected_dir.exists():
            selected_images = [
                p
                for p in selected_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
            if len(selected_images) >= args.curated_min:
                curated_plants.append(plant_slug)

        for img in plant_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = img.relative_to(source_root)
            images.append(
                {
                    "plant_slug": plant_slug,
                    "path": str(img),
                    "rel": str(rel),
                    "bytes": str(img.stat().st_size),
                    "sha256": sha256_file(img),
                }
            )

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images_count": len(images),
        "images": images,
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    curated_json.parent.mkdir(parents=True, exist_ok=True)
    curated_json.write_text(
        json.dumps(
            {
                "generated_at": manifest["generated_at"],
                "curated_min": args.curated_min,
                "curated_count": len(curated_plants),
                "plants": curated_plants,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("Done.")
    print(f"Partial manifest: {manifest_json}")
    print(f"Curated plants: {curated_json}")


if __name__ == "__main__":
    main()
