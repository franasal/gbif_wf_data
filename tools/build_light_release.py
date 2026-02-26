#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install with `pip install pillow`."
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}

KNOWN_PREFIXES = ["reviewed", "has_lookalike", "lookallike", "missing"]
REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT.parent
SYNONYMS_JSON_DEFAULT = APP_ROOT / "assets" / "data" / "synonyms.json"


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


def load_existing_manifest_map(manifest_json: Path) -> Dict[str, Dict[str, Any]]:
    if not manifest_json.exists():
        return {}
    try:
        data = json.loads(manifest_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    images = data.get("images")
    if not isinstance(images, list):
        return {}
    by_rel: Dict[str, Dict[str, Any]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        output_rel = str(item.get("output_rel") or "").strip()
        output_path = str(item.get("output_path") or "").strip()
        if output_rel:
            by_rel[output_rel] = item
        if output_path and output_path not in by_rel:
            by_rel[output_path] = item
    return by_rel


def parse_filename(path: Path) -> Optional[Tuple[str, str]]:
    stem = path.stem
    parts = stem.split("__")
    if len(parts) < 4:
        return None
    occ_key = parts[-3]
    id_hash = parts[-2]
    return (occ_key, id_hash)


def strip_prefixes(folder_name: str) -> Tuple[str, Dict[str, bool]]:
    flags = {p: False for p in KNOWN_PREFIXES}
    base = folder_name
    changed = True
    while changed:
        changed = False
        for prefix in KNOWN_PREFIXES:
            token = f"{prefix}_"
            if base.startswith(token):
                base = base[len(token) :]
                flags[prefix] = True
                changed = True
    return base, flags


def slugify_scientific(name: str) -> str:
    import re

    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def load_synonym_slug_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    root = data.get("scientificNameToAccepted") if isinstance(data, dict) else None
    if not isinstance(root, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in root.items():
        from_slug = slugify_scientific(str(k))
        to_slug = slugify_scientific(str(v))
        if from_slug and to_slug:
            out[from_slug] = to_slug
    return out


def canonical_slug(slug: str, synonym_slug_map: Dict[str, str]) -> str:
    return synonym_slug_map.get(slug, slug)


def ensure_webp(
    src: Path,
    dst: Path,
    max_width: int,
    max_long_edge: int,
    quality: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if max_long_edge > 0:
            long_edge = max(w, h)
            if long_edge > max_long_edge:
                scale = max_long_edge / long_edge
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
        elif w > max_width:
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
        "--keep-prefixes",
        action="store_true",
        default=True,
        help="Keep curation prefixes (reviewed/has_lookalike/lookallike/missing) in output paths.",
    )
    parser.add_argument(
        "--strip-prefixes",
        dest="keep_prefixes",
        action="store_false",
        help="Strip curation prefixes from output paths (legacy behavior).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        help="Max width for webp conversion.",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=0,
        help="Optional max long edge. If >0, applies to both portrait and landscape.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=82,
        help="WebP quality (0-100).",
    )
    parser.add_argument(
        "--max-images-per-plant",
        type=int,
        default=0,
        help="Optional cap per plant folder; 0 means no cap.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Progress log interval (images). Set 0 to disable.",
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
    parser.add_argument(
        "--names-edible",
        default="gbif_wf_data/data/names_edible.json",
        help="Path to edible names JSON (required when filtering or splitting).",
    )
    parser.add_argument(
        "--names-poisonous",
        default="gbif_wf_data/data/names_poisonous.json",
        help="Path to poisonous names JSON (required when filtering or splitting).",
    )
    parser.add_argument(
        "--synonyms-json",
        default=str(SYNONYMS_JSON_DEFAULT),
        help="Path to centralized synonyms.json (main app assets).",
    )
    parser.add_argument(
        "--class-filter",
        choices=["all", "edible", "poisonous"],
        default="all",
        help="Filter plants by class for the combined output.",
    )
    parser.add_argument(
        "--split-by-class",
        action="store_true",
        default=True,
        help="Produce separate outputs for edible and poisonous classes (default: on). Disable with --no-split-by-class.",
    )
    parser.add_argument(
        "--no-split-by-class",
        dest="split_by_class",
        action="store_false",
        help="Disable class split; produce a single combined pack.",
    )
    parser.add_argument(
        "--out-root-edible",
        default="gbif_wf_data/assets/plant_images/light_build/edible",
        help="Output root for edible images when splitting.",
    )
    parser.add_argument(
        "--out-root-poisonous",
        default="gbif_wf_data/assets/plant_images/light_build/poisonous",
        help="Output root for poisonous images when splitting.",
    )
    parser.add_argument(
        "--manifest-json-edible",
        default="gbif_wf_data/assets/plant_images/light_build/edible/images_manifest.json",
        help="Manifest JSON for edible output when splitting.",
    )
    parser.add_argument(
        "--manifest-json-poisonous",
        default="gbif_wf_data/assets/plant_images/light_build/poisonous/images_manifest.json",
        help="Manifest JSON for poisonous output when splitting.",
    )
    parser.add_argument(
        "--manifest-csv-edible",
        default="gbif_wf_data/assets/plant_images/light_build/edible/images_manifest.csv",
        help="Manifest CSV for edible output when splitting.",
    )
    parser.add_argument(
        "--manifest-csv-poisonous",
        default="gbif_wf_data/assets/plant_images/light_build/poisonous/images_manifest.csv",
        help="Manifest CSV for poisonous output when splitting.",
    )
    parser.add_argument(
        "--zip-path-edible",
        default="gbif_wf_data/assets/plant_images/light_build/edible/gbif_light_edible.zip",
        help="ZIP path for edible output when splitting.",
    )
    parser.add_argument(
        "--zip-path-poisonous",
        default="gbif_wf_data/assets/plant_images/light_build/poisonous/gbif_light_poisonous.zip",
        help="ZIP path for poisonous output when splitting.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    # Load index metadata + start timer
    start_time = time.time()
    index_map = load_index_map(Path(args.index_csv))
    synonym_slug_map = load_synonym_slug_map(Path(args.synonyms_json))

    # Prepare class filters
    edible_slugs: Optional[set[str]] = None
    poisonous_slugs: Optional[set[str]] = None
    if args.split_by_class or args.class_filter != "all":
        edible_path = Path(args.names_edible)
        poisonous_path = Path(args.names_poisonous)
        if not edible_path.exists() or not poisonous_path.exists():
            raise SystemExit(
                "names_edible.json and names_poisonous.json are required when filtering or splitting."
            )
        edible_data = json.loads(edible_path.read_text())
        poisonous_data = json.loads(poisonous_path.read_text())
        if not isinstance(edible_data, dict) or not isinstance(poisonous_data, dict):
            raise SystemExit("names_edible.json and names_poisonous.json must be JSON dicts.")
        edible_slugs = {canonical_slug(slugify_scientific(k), synonym_slug_map) for k in edible_data.keys()}
        poisonous_slugs = {canonical_slug(slugify_scientific(k), synonym_slug_map) for k in poisonous_data.keys()}

    def classify(base_slug: str) -> str:
        if edible_slugs is None or poisonous_slugs is None:
            return "all"
        in_edible = base_slug in edible_slugs
        in_pois = base_slug in poisonous_slugs
        if in_edible and in_pois:
            raise SystemExit(f"Slug appears in both edible and poisonous lists: {base_slug}")
        if in_edible:
            return "edible"
        if in_pois:
            return "poisonous"
        return "unknown"

    # Configure output buckets
    bucket_configs = {}
    if args.split_by_class:
        bucket_configs = {
            "edible": {
                "out_root": Path(args.out_root_edible),
                "manifest_json": Path(args.manifest_json_edible),
                "manifest_csv": Path(args.manifest_csv_edible),
                "zip_path": Path(args.zip_path_edible),
                "images": [],
                "existing": {},
            },
            "poisonous": {
                "out_root": Path(args.out_root_poisonous),
                "manifest_json": Path(args.manifest_json_poisonous),
                "manifest_csv": Path(args.manifest_csv_poisonous),
                "zip_path": Path(args.zip_path_poisonous),
                "images": [],
                "existing": {},
            },
        }
    else:
        bucket_configs = {
            "combined": {
                "out_root": Path(args.out_root),
                "manifest_json": Path(args.manifest_json),
                "manifest_csv": Path(args.manifest_csv),
                "zip_path": Path(args.zip_path),
                "images": [],
                "existing": {},
            }
        }

    # Remove existing zip files up-front
    for cfg in bucket_configs.values():
        zip_path = cfg["zip_path"]
        if zip_path.exists():
            zip_path.unlink()
        cfg["existing"] = load_existing_manifest_map(cfg["manifest_json"])

    plant_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    print(f"Found plant dirs: {len(plant_dirs)}", flush=True)
    if args.split_by_class:
        print("Split-by-class: on", flush=True)
    else:
        print("Split-by-class: off (combined pack)", flush=True)

    processed_images = 0
    skipped_unknown = 0
    skipped_no_selected = 0
    skipped_cap = 0
    skipped_ext = 0
    for plant_dir in sorted(plant_dirs):
        selected_dir = plant_dir / args.selected_folder
        if not selected_dir.exists():
            skipped_no_selected += 1
            continue
        base_slug, prefix_flags = strip_prefixes(plant_dir.name)
        base_slug = canonical_slug(base_slug, synonym_slug_map)
        class_label = classify(base_slug)
        if args.class_filter != "all" and not args.split_by_class:
            if class_label != args.class_filter:
                continue
            if class_label == "unknown":
                continue
        buckets_for_plant: Iterable[str]
        if args.split_by_class:
            if class_label in ("edible", "poisonous"):
                buckets_for_plant = (class_label,)
            else:
                skipped_unknown += 1
                # Skip unknowns when splitting to avoid cross-contamination.
                continue
        else:
            buckets_for_plant = ("combined",)

        rel_dir_base = canonical_slug(base_slug, synonym_slug_map)
        if args.keep_prefixes:
            prefix, _ = strip_prefixes(plant_dir.name)
            # keep existing curation prefixes but normalize the underlying slug
            rel_dir = plant_dir.name.replace(prefix, rel_dir_base, 1)
        else:
            rel_dir = rel_dir_base

        picked_for_plant = 0
        for src in sorted(selected_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                skipped_ext += 1
                continue
            if args.max_images_per_plant > 0 and picked_for_plant >= args.max_images_per_plant:
                skipped_cap += 1
                continue
            parsed = parse_filename(src)
            occ_key = ""
            id_hash = ""
            meta_row: Dict[str, Any] = {}
            if parsed:
                occ_key, id_hash = parsed
                meta_row = index_map.get((occ_key, id_hash), {})

            out_name = f"{src.stem}.webp"

            for bucket_name in buckets_for_plant:
                cfg = bucket_configs[bucket_name]
                dst = cfg["out_root"] / rel_dir / out_name
                ensure_webp(src, dst, args.max_width, args.max_long_edge, args.quality)
                image_entry = {
                    # Keep slug normalized for the app, but keep prefixes in paths for curation traceability.
                    "plant_slug": base_slug,
                    "plant_slug_raw": plant_dir.name,
                    "curation_prefixes": [k for k, v in prefix_flags.items() if v],
                    "class": class_label,
                    "source_path": str(src),
                    "output_path": str(dst),
                    "output_rel": str(dst.relative_to(cfg["out_root"])),
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
                # Preserve existing metadata if index CSV did not provide it.
                if cfg["existing"]:
                    existing = cfg["existing"].get(image_entry["output_rel"]) or cfg["existing"].get(
                        image_entry["output_path"]
                    )
                    if isinstance(existing, dict):
                        for key in (
                            "media_identifier",
                            "media_license_key",
                            "media_license_label",
                            "media_license_url",
                            "media_creator",
                            "media_rightsHolder",
                            "media_references",
                            "attribution_suggestion",
                        ):
                            if not image_entry.get(key) and existing.get(key):
                                image_entry[key] = existing.get(key)
                cfg["images"].append(image_entry)

                processed_images += 1
                if args.log_every > 0 and processed_images % args.log_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Processed images: {processed_images} | "
                        f"elapsed={elapsed:.1f}s | "
                        f"last_plant={plant_dir.name}",
                        flush=True,
                    )

            picked_for_plant += 1

    def write_outputs(cfg: Dict[str, Any]) -> None:
        images = cfg["images"]
        manifest_json = cfg["manifest_json"]
        manifest_csv = cfg["manifest_csv"]
        out_root = cfg["out_root"]
        zip_path = cfg["zip_path"]

        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_width": args.max_width,
            "max_long_edge": args.max_long_edge,
            "quality": args.quality,
            "max_images_per_plant": args.max_images_per_plant,
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
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, allowZip64=True) as zf:
            zf.write(manifest_json, arcname="images_manifest.json")
            zf.write(manifest_csv, arcname="images_manifest.csv")
            for file_path in out_root.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path != manifest_json
                    and file_path != manifest_csv
                    and file_path != zip_path
                    and file_path.suffix.lower() != ".zip"
                ):
                    rel = file_path.relative_to(out_root)
                    zf.write(file_path, arcname=str(rel))

        total_bytes = sum(int(r.get("bytes", 0) or 0) for r in images)
        print(f"[{out_root}] Images: {len(images)} | {total_bytes / 1024 / 1024:.2f} MB")
        print(f"Manifest JSON: {manifest_json}")
        print(f"Manifest CSV: {manifest_csv}")
        print(f"ZIP: {zip_path}")

    for cfg in bucket_configs.values():
        write_outputs(cfg)

    elapsed = time.time() - start_time
    print("Done.", flush=True)
    print(
        "Summary: "
        f"processed={processed_images}, "
        f"skipped_no_selected={skipped_no_selected}, "
        f"skipped_unknown={skipped_unknown}, "
        f"skipped_ext={skipped_ext}, "
        f"skipped_cap={skipped_cap}, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
