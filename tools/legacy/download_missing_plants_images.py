#!/usr/bin/env python3
"""
Download GBIF images for plants marked as "missing_" in assets.

This script:
1. Identifies all "missing_" plant folders in assets/plant_images/gbif_samples
2. Creates a filtered plants JSON with only those species
3. Runs download_gbif_image_samples.py to fetch images for them
4. Organizes images into the existing missing_* folders

Usage:
    python3 tools/download_missing_plants_images.py [options]

Options:
    --limit-per-plant INT     Max images per plant (default: 20)
    --min-candidates INT      Min candidates to consider (default: 100)
    --limit INT               Total GBIF limit per search (default: 1000)
    --restrict-country CODE   Restrict to country code (default: empty)
    --preferred-country CODE  Prefer country code (default: DE)
    --dry-run                 Only show what would be done
    --help                    Show this help
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
GBIF_SAMPLES_DIR = REPO_ROOT / "assets" / "plant_images" / "gbif_samples"
DATA_DIR = REPO_ROOT / "data"
PLANTS_RESOLVED = DATA_DIR / "plants_resolved_edible.json"
CONFIG_FILE = DATA_DIR / "gbif_download_config.json"


def slugify(name: str) -> str:
    """Convert plant name to folder slug format."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def get_missing_plants() -> List[str]:
    """
    Get list of plants marked as "missing_" in assets.
    Returns list of scientific names with underscores.
    """
    if not GBIF_SAMPLES_DIR.exists():
        print(f"Error: {GBIF_SAMPLES_DIR} not found")
        return []

    missing = []
    for item in GBIF_SAMPLES_DIR.iterdir():
        if item.is_dir() and item.name.startswith("missing_"):
            # Extract scientific name from folder: missing_genus_species -> genus species
            plant_slug = item.name.replace("missing_", "")
            missing.append(plant_slug)

    return sorted(missing)


def load_all_plants() -> Dict[str, dict]:
    """Load all plants from resolved file and index by slug."""
    if not PLANTS_RESOLVED.exists():
        print(f"Error: {PLANTS_RESOLVED} not found")
        return {}

    plants_by_slug = {}
    with open(PLANTS_RESOLVED, "r", encoding="utf-8") as f:
        plants = json.load(f)

    for plant in plants:
        sci_name = plant.get("scientificName", "")
        if sci_name:
            slug = slugify(sci_name)
            plants_by_slug[slug] = plant

    return plants_by_slug


def create_missing_plants_json(
    all_plants: Dict[str, dict], missing_slugs: List[str], output_path: Path
) -> int:
    """
    Create a JSON file containing only the missing plants.
    Returns count of plants found.
    """
    found = []
    not_found = []

    for slug in missing_slugs:
        if slug in all_plants:
            found.append(all_plants[slug])
        else:
            not_found.append(slug)

    if not_found:
        print(f"\n⚠️  {len(not_found)} missing plants not in resolved file:")
        for slug in not_found:
            print(f"   - {slug}")

    if not found:
        print("Error: No missing plants found in resolved file")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Created {output_path} with {len(found)} plants")
    return len(found)


def run_download_script(
    plants_json: Path,
    limit_per_plant: int = 20,
    min_candidates: int = 100,
    limit: int = 1000,
    restrict_country: str = "",
    preferred_country: str = "DE",
    dry_run: bool = False,
) -> int:
    """
    Run the download_gbif_image_samples.py script.
    Returns exit code.
    """
    script = REPO_ROOT / "tools" / "download_gbif_image_samples.py"
    if not script.exists():
        print(f"Error: {script} not found")
        return 1

    cmd = [
        "python3",
        str(script),
        "--resolved",
        str(plants_json),
        "--config",
        str(CONFIG_FILE),
        "--limit-per-plant",
        str(limit_per_plant),
        "--min-candidates",
        str(min_candidates),
        "--limit",
        str(limit),
        "--preferred-country",
        preferred_country,
    ]

    if restrict_country:
        cmd.extend(["--restrict-country", restrict_country])

    cmd.extend([
        "--out-dir",
        str(GBIF_SAMPLES_DIR),
        "--meta-json",
        str(GBIF_SAMPLES_DIR / "index_missing.json"),
        "--meta-csv",
        str(GBIF_SAMPLES_DIR / "index_missing.csv"),
    ])

    print(f"\n📥 Running download script for {plants_json.name}...")
    print(f"\n Command: {' '.join(cmd)}\n")

    if dry_run:
        print("(Dry run: not executing)")
        return 0

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Download GBIF images for plants marked as missing_",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit-per-plant",
        type=int,
        default=20,
        help="Max images per plant (default: 20)",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=100,
        help="Min candidates to consider (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Total GBIF limit per search (default: 1000)",
    )
    parser.add_argument(
        "--restrict-country",
        default="",
        help="Restrict to country code (e.g., DE)",
    )
    parser.add_argument(
        "--preferred-country",
        default="DE",
        help="Prefer country code (default: DE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  GBIF Image Downloader for Missing Plants")
    print("=" * 70)

    # Get missing plants
    missing_slugs = get_missing_plants()
    if not missing_slugs:
        print("\n❌ No missing_* folders found in assets")
        return 1

    print(f"\n📋 Found {len(missing_slugs)} missing plants:")
    for i, slug in enumerate(missing_slugs, 1):
        print(f"   {i:2d}. {slug}")

    # Load all plants and create filtered JSON
    all_plants = load_all_plants()
    if not all_plants:
        print("\n❌ Could not load plants from resolved file")
        return 1

    filtered_json = DATA_DIR / "plants_missing_temp.json"
    count = create_missing_plants_json(all_plants, missing_slugs, filtered_json)
    if count == 0:
        return 1

    # Run download script
    exit_code = run_download_script(
        filtered_json,
        limit_per_plant=args.limit_per_plant,
        min_candidates=args.min_candidates,
        limit=args.limit,
        restrict_country=args.restrict_country,
        preferred_country=args.preferred_country,
        dry_run=args.dry_run,
    )

    if exit_code == 0 and not args.dry_run:
        print("\n" + "=" * 70)
        print("✅ Download complete!")
        print(f"   Images saved to: {GBIF_SAMPLES_DIR}")
        print(f"   Metadata: {GBIF_SAMPLES_DIR / 'index_missing.json'}")
        print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
