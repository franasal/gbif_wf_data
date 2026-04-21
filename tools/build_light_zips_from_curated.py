#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Missing dependency: Pillow. Install with `pip install pillow`.") from exc

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`.") from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}
KNOWN_PREFIXES = ["reviewed", "has_lookalike", "lookallike", "missing"]
GBIF_BASE = "https://api.gbif.org/v1"
GBIF_UA = "wild-forager-light-build/1.0"

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


def normalize_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        cleaned = raw
    if cleaned.endswith("/") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned


def candidate_hashes(value: str) -> List[str]:
    if not value:
        return []
    candidates = {value}
    norm = normalize_url(value)
    if norm:
        candidates.add(norm)
    if value.startswith("http://"):
        candidates.add("https://" + value[len("http://") :])
    if value.startswith("https://"):
        candidates.add("http://" + value[len("https://") :])
    if norm.startswith("http://"):
        candidates.add("https://" + norm[len("http://") :])
    if norm.startswith("https://"):
        candidates.add("http://" + norm[len("https://") :])
    return [hash_identifier(candidate) for candidate in candidates if candidate]


def normalize_license_to_key(lic: Any) -> str:
    if not lic:
        return ""
    s = str(lic).strip().lower()
    if "publicdomain/zero/1.0" in s or "cc0" in s:
        return "cc0-1.0"
    if "creativecommons.org/licenses/by/" in s:
        if "/4.0" in s:
            return "cc-by-4.0"
        if "/3.0" in s:
            return "cc-by-3.0"
    if "creativecommons.org/licenses/by-sa/" in s:
        if "/4.0" in s:
            return "cc-by-sa-4.0"
        if "/3.0" in s:
            return "cc-by-sa-3.0"
    return ""


COMMERCIAL_LICENSES = {
    "cc0-1.0": "CC0 1.0",
    "cc-by-4.0": "CC BY 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-3.0": "CC BY 3.0",
    "cc-by-sa-3.0": "CC BY-SA 3.0",
}


def load_index_map(index_csvs: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for index_csv in index_csvs:
        if not index_csv.exists():
            continue
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


def fetch_occurrence_media(
    session: requests.Session,
    occ_key: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    url = f"{GBIF_BASE}/occurrence/{occ_key}"
    payload: Dict[str, Any] | None = None
    try:
        response = session.get(url, headers={"User-Agent": GBIF_UA}, timeout=timeout)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
    except Exception:
        try:
            result = subprocess.run(
                ["curl", "-L", "-s", "--fail-with-body", "-A", GBIF_UA, url],
                check=True,
                capture_output=True,
                text=True,
            )
            stdout = result.stdout.strip()
            if not stdout:
                return []
            payload = json.loads(stdout)
        except Exception:
            return []
    if not isinstance(payload, dict):
        return []
    media = payload.get("media") or []
    return media if isinstance(media, list) else []


def media_row_from_occurrence(
    session: requests.Session,
    occ_key: str,
    id_hash: str,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = f"{occ_key}:{id_hash}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        media_list = fetch_occurrence_media(session, occ_key)
    except Exception:
        cache[cache_key] = {}
        return {}
    for media in media_list:
        if not isinstance(media, dict):
            continue
        identifier = str(media.get("identifier") or "").strip()
        references = str(media.get("references") or "").strip()
        hashes = set(candidate_hashes(identifier) + candidate_hashes(references))
        if id_hash not in hashes:
            continue
        license_url = str(media.get("license") or "").strip()
        license_key = normalize_license_to_key(license_url)
        row = {
            "media_identifier": identifier,
            "media_license_key": license_key,
            "media_license_label": COMMERCIAL_LICENSES.get(license_key, ""),
            "media_license_url": license_url,
            "media_creator": str(media.get("creator") or ""),
            "media_rightsHolder": str(media.get("rightsHolder") or ""),
            "media_references": references or identifier,
        }
        cache[cache_key] = row
        return row
    cache[cache_key] = {}
    return {}


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
        img.save(dst, format="WEBP", quality=quality, method=6, optimize=True)


def safe_clean_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise SystemExit(f"Refusing to clean non-directory: {path}")
    if "light_build" not in str(path):
        raise SystemExit(f"Refusing to clean path without 'light_build': {path}")
    shutil.rmtree(path)


def build_class(
    class_label: str,
    source_root: Path,
    out_root: Path,
    index_map: Dict[Tuple[str, str], Dict[str, Any]],
    synonym_slug_map: Dict[str, str],
    max_long_edge: int,
    quality: int,
    max_images_per_plant: int,
    clean: bool,
) -> None:
    if not source_root.exists():
        raise SystemExit(f"Missing curated source root: {source_root}")

    if clean:
        safe_clean_dir(out_root)

    images: List[Dict[str, Any]] = []
    gbif_session = requests.Session()
    gbif_media_cache: Dict[str, Dict[str, Any]] = {}

    plant_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    for plant_dir in sorted(plant_dirs):
        base_slug, prefix_flags = strip_prefixes(plant_dir.name)
        base_slug = canonical_slug(base_slug, synonym_slug_map)

        picked_for_plant = 0
        for src in sorted(plant_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                continue
            if max_images_per_plant > 0 and picked_for_plant >= max_images_per_plant:
                continue

            parsed = parse_filename(src)
            occ_key = ""
            id_hash = ""
            meta_row: Dict[str, Any] = {}
            if parsed:
                occ_key, id_hash = parsed
                meta_row = index_map.get((occ_key, id_hash), {})
                if not meta_row:
                    meta_row = media_row_from_occurrence(
                        gbif_session,
                        occ_key,
                        id_hash,
                        gbif_media_cache,
                    )

            out_name = f"{src.stem}.webp"
            dst = out_root / plant_dir.name / out_name
            ensure_webp(src, dst, max_long_edge, quality)

            image_entry = {
                "plant_slug": base_slug,
                "plant_slug_raw": plant_dir.name,
                "curation_prefixes": [k for k, v in prefix_flags.items() if v],
                "class": class_label,
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
            picked_for_plant += 1

    manifest_json = out_root / "images_manifest.json"
    manifest_csv = out_root / "images_manifest.csv"
    zip_path = out_root / ("gbif_light_edible.zip" if class_label == "edible" else "gbif_light_poisonous.zip")

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_long_edge": max_long_edge,
        "quality": quality,
        "max_images_per_plant": max_images_per_plant,
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
    print(f"[{class_label}] Images: {len(images)} | {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"ZIP: {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build optimized light zips from curated edible/poisonous folders."
    )
    parser.add_argument(
        "--curated-root",
        default="gbif_wf_data/assets/plant_images/light_build",
        help="Root containing curated edible/poisonous folders.",
    )
    parser.add_argument(
        "--out-root",
        default="gbif_wf_data/assets/plant_images/light_build_output",
        help="Output root for webp images + zips.",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=1280,
        help="Max long edge for webp conversion.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=75,
        help="WebP quality (0-100).",
    )
    parser.add_argument(
        "--max-images-per-plant",
        type=int,
        default=0,
        help="Optional cap per plant folder; 0 means no cap.",
    )
    parser.add_argument(
        "--index-csv",
        action="append",
        default=[],
        help="Index CSV for metadata lookup (can be repeated).",
    )
    parser.add_argument(
        "--synonyms-json",
        default=str(SYNONYMS_JSON_DEFAULT),
        help="Path to centralized synonyms.json (main app assets).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Clean output folders before building (default: on).",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Do not clean output folders before building.",
    )
    args = parser.parse_args()

    curated_root = Path(args.curated_root)
    out_root = Path(args.out_root)

    default_index = Path("gbif_wf_data/assets/plant_images/gbif_samples/index.csv")
    index_csvs = [default_index]
    for path in args.index_csv:
        index_csvs.append(Path(path))

    index_map = load_index_map(index_csvs)
    synonym_slug_map = load_synonym_slug_map(Path(args.synonyms_json))

    build_class(
        "edible",
        curated_root / "edible",
        out_root / "edible",
        index_map,
        synonym_slug_map,
        args.max_long_edge,
        args.quality,
        args.max_images_per_plant,
        args.clean,
    )
    build_class(
        "poisonous",
        curated_root / "poisonous",
        out_root / "poisonous",
        index_map,
        synonym_slug_map,
        args.max_long_edge,
        args.quality,
        args.max_images_per_plant,
        args.clean,
    )

    print("Done.")


if __name__ == "__main__":
    main()
