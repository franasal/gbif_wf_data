#!/usr/bin/env python3
"""
Download GBIF image candidates for plants missing curated light images.

This script:
1) Reads missing lists (edible + poisonous)
2) Filters resolved plant lists to those missing slugs
3) Runs the legacy downloader with generous parameters
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LEGACY_DOWNLOADER = REPO_ROOT / "tools" / "legacy" / "download_gbif_image_samples.py"
SYNONYMS_JSON_DEFAULT = REPO_ROOT.parent / "assets" / "data" / "synonyms.json"

DEFAULT_MISSING_EDIBLE = DATA_DIR / "missing_light_edible.json"
DEFAULT_MISSING_POISONOUS = DATA_DIR / "missing_light_poisonous.json"
DEFAULT_RESOLVED_EDIBLE = DATA_DIR / "plants_resolved_edible.json"
DEFAULT_RESOLVED_POISONOUS = DATA_DIR / "plants_resolved_poisonous.json"
DEFAULT_CONFIG = DATA_DIR / "gbif_download_config.json"

DEFAULT_OUT_ROOT = REPO_ROOT / "assets" / "plant_images" / "gbif_samples" / "missing_candidates"


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


def load_missing_slugs(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Missing list not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = data.get("missing") if isinstance(data, dict) else None
    if not isinstance(missing, list):
        raise SystemExit(f"Invalid missing list format: {path}")
    slugs = []
    for item in missing:
        if isinstance(item, dict):
            slug = item.get("slug")
            if slug:
                slugs.append(str(slug))
    return sorted(set(slugs))


def filter_resolved(resolved_path: Path, missing_slugs: List[str], synonym_slug_map: Dict[str, str]) -> List[dict]:
    if not resolved_path.exists():
        raise SystemExit(f"Missing resolved file: {resolved_path}")
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Invalid resolved file: {resolved_path}")

    missing_set = set(missing_slugs)
    out = []
    for item in data:
        sci = str(item.get("scientificName") or "")
        if not sci:
            continue
        slug = canonical_slug(slugify_scientific(sci), synonym_slug_map)
        if slug in missing_set:
            out.append(item)
    return out


def write_filtered(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def run_downloader(
    resolved_path: Path,
    out_dir: Path,
    meta_prefix: str,
    config_path: Path,
    limit_per_plant: int,
    min_candidates: int,
    limit: int,
    years_back: int,
    preferred_country: str,
    restrict_country: str,
    dry_run: bool,
) -> int:
    if not LEGACY_DOWNLOADER.exists():
        raise SystemExit(f"Legacy downloader not found: {LEGACY_DOWNLOADER}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_json = out_dir / f"index_{meta_prefix}.json"
    meta_csv = out_dir / f"index_{meta_prefix}.csv"

    cmd = [
        "python3",
        str(LEGACY_DOWNLOADER),
        "--resolved",
        str(resolved_path),
        "--config",
        str(config_path),
        "--limit-per-plant",
        str(limit_per_plant),
        "--min-candidates",
        str(min_candidates),
        "--limit",
        str(limit),
        "--years-back",
        str(years_back),
        "--preferred-country",
        preferred_country,
        "--out-dir",
        str(out_dir),
        "--meta-json",
        str(meta_json),
        "--meta-csv",
        str(meta_csv),
    ]

    if restrict_country:
        cmd.extend(["--restrict-country", restrict_country])

    print("\nCommand:")
    print(" ".join(cmd))

    if dry_run:
        print("(Dry run: not executing)")
        return 0

    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GBIF image candidates for plants missing curated light images."
    )
    parser.add_argument(
        "--missing-edible",
        default=str(DEFAULT_MISSING_EDIBLE),
        help="Missing edible list JSON.",
    )
    parser.add_argument(
        "--missing-poisonous",
        default=str(DEFAULT_MISSING_POISONOUS),
        help="Missing poisonous list JSON.",
    )
    parser.add_argument(
        "--resolved-edible",
        default=str(DEFAULT_RESOLVED_EDIBLE),
        help="Resolved edible plants JSON.",
    )
    parser.add_argument(
        "--resolved-poisonous",
        default=str(DEFAULT_RESOLVED_POISONOUS),
        help="Resolved poisonous plants JSON.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="GBIF download config JSON.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Root output folder for missing candidates.",
    )
    parser.add_argument(
        "--limit-per-plant",
        type=int,
        default=50,
        help="Target images per plant (default: 50).",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=200,
        help="Min candidates to consider (default: 200).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Total GBIF limit per search (default: 2000).",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=6,
        help="Years back for GBIF search (default: 6).",
    )
    parser.add_argument(
        "--preferred-country",
        default="DE",
        help="Prefer country code (default: DE).",
    )
    parser.add_argument(
        "--restrict-country",
        default="",
        help="Restrict to country code (optional).",
    )
    parser.add_argument(
        "--synonyms-json",
        default=str(SYNONYMS_JSON_DEFAULT),
        help="Path to centralized synonyms.json (main app assets).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done.",
    )
    args = parser.parse_args()

    synonym_slug_map = load_synonym_slug_map(Path(args.synonyms_json))

    missing_edible = load_missing_slugs(Path(args.missing_edible))
    missing_poisonous = load_missing_slugs(Path(args.missing_poisonous))

    edible_rows = filter_resolved(Path(args.resolved_edible), missing_edible, synonym_slug_map)
    poisonous_rows = filter_resolved(Path(args.resolved_poisonous), missing_poisonous, synonym_slug_map)

    edible_out = DATA_DIR / "plants_missing_edible.json"
    poisonous_out = DATA_DIR / "plants_missing_poisonous.json"

    write_filtered(edible_out, edible_rows)
    write_filtered(poisonous_out, poisonous_rows)

    out_root = Path(args.out_root)
    edible_dir = out_root / "edible"
    poisonous_dir = out_root / "poisonous"

    print("=" * 70)
    print("  GBIF Missing Image Downloader")
    print("=" * 70)
    print(f"Missing edible: {len(edible_rows)} -> {edible_out}")
    print(f"Missing poisonous: {len(poisonous_rows)} -> {poisonous_out}")

    if len(edible_rows) == 0 and len(poisonous_rows) == 0:
        print("\nNothing to download.")
        return

    if len(edible_rows) > 0:
        print("\n--- Edible ---")
        code = run_downloader(
            edible_out,
            edible_dir,
            "missing_edible",
            Path(args.config),
            args.limit_per_plant,
            args.min_candidates,
            args.limit,
            args.years_back,
            args.preferred_country,
            args.restrict_country,
            args.dry_run,
        )
        if code != 0:
            sys.exit(code)

    if len(poisonous_rows) > 0:
        print("\n--- Poisonous ---")
        code = run_downloader(
            poisonous_out,
            poisonous_dir,
            "missing_poisonous",
            Path(args.config),
            args.limit_per_plant,
            args.min_candidates,
            args.limit,
            args.years_back,
            args.preferred_country,
            args.restrict_country,
            args.dry_run,
        )
        if code != 0:
            sys.exit(code)

    print("\nDone.")


if __name__ == "__main__":
    main()
