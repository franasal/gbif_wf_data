#!/usr/bin/env python3
"""
Build light release ZIP with external reviewed_ folders and skip missing_ folders.

This script handles the case where:
- reviewed_* folders have been moved to external storage
- missing_* folders are kept locally
- You want to build light assets excluding the missing folders

Usage:
    python3 tools/build_light_external.py --external-root /path/to/external/gbif_samples

Example:
    python3 tools/build_light_external.py \
        --external-root /mnt/external/gbif_samples \
        --skip-missing \
        --split-by-class \
        --keep-prefixes
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
GBIF_SAMPLES_DIR = REPO_ROOT / "assets" / "plant_images" / "gbif_samples"
DEFAULT_INDEX_CSV = GBIF_SAMPLES_DIR / "index.csv"
DEFAULT_NAMES_EDIBLE = REPO_ROOT / "data" / "names_edible.json"
DEFAULT_NAMES_POISONOUS = REPO_ROOT / "data" / "names_poisonous.json"

def get_plant_dirs(source_root: Path, skip_prefix: Optional[str] = None) -> List[Path]:
    """Get list of plant directories, optionally skipping those with a prefix."""
    if not source_root.exists():
        return []

    dirs = [p for p in source_root.iterdir() if p.is_dir()]

    if skip_prefix:
        dirs = [d for d in dirs if not d.name.startswith(f"{skip_prefix}_")]

    return sorted(dirs)


def create_symlink_tree(
    local_root: Path,
    external_root: Path,
    temp_root: Path,
    skip_missing: bool = True,
) -> int:
    """
    Create a temporary directory tree with symlinks to combine local and external folders.
    Returns count of linked folders.
    """
    linked = 0

    # Link local folders (excluding missing if requested)
    for local_dir in get_plant_dirs(local_root, skip_prefix="missing" if skip_missing else None):
        link_path = temp_root / local_dir.name
        link_path.symlink_to(local_dir)
        linked += 1

    # Link external folders (reviewed_ folders)
    for ext_dir in get_plant_dirs(external_root):
        link_path = temp_root / ext_dir.name
        if not link_path.exists():  # Don't overwrite if already linked
            link_path.symlink_to(ext_dir)
            linked += 1

    return linked


def main():
    parser = argparse.ArgumentParser(
        description="Build light release with external reviewed_ folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--external-root",
        required=True,
        help="Path to external storage with reviewed_* folders (e.g., /mnt/external/gbif_samples)",
    )
    parser.add_argument(
        "--local-root",
        default=str(GBIF_SAMPLES_DIR),
        help=f"Local gbif_samples root (default: {GBIF_SAMPLES_DIR})",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip missing_* folders (default: True)",
    )
    parser.add_argument(
        "--include-missing",
        dest="skip_missing",
        action="store_false",
        help="Include missing_* folders in build",
    )
    parser.add_argument(
        "--split-by-class",
        action="store_true",
        help="Split output into edible/poisonous (requires names JSONs)",
    )
    parser.add_argument(
        "--keep-prefixes",
        action="store_true",
        default=True,
        help="Keep curation prefixes in output paths (default: True)",
    )
    parser.add_argument(
        "--strip-prefixes",
        dest="keep_prefixes",
        action="store_false",
        help="Remove prefixes from output paths",
    )
    parser.add_argument(
        "--max-images-per-plant",
        type=int,
        default=0,
        help="Max images per plant (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        help="Max width for webp (default: 1024)",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=2048,
        help="Max long edge for webp (default: 2048)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="WebP quality 0-100 (default: 85)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running",
    )
    parser.add_argument(
    "--index-csv",
    default=str(DEFAULT_INDEX_CSV),
    help=f"Path to index CSV (default: {DEFAULT_INDEX_CSV})",
)
    parser.add_argument(
        "--names-edible",
        default=str(DEFAULT_NAMES_EDIBLE),
        help=f"Path to names_edible.json (default: {DEFAULT_NAMES_EDIBLE})",
    )
    parser.add_argument(
        "--names-poisonous",
        default=str(DEFAULT_NAMES_POISONOUS),
        help=f"Path to names_poisonous.json (default: {DEFAULT_NAMES_POISONOUS})",
    )

    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    if not index_csv.is_absolute():
        index_csv = (REPO_ROOT / index_csv)
    index_csv = index_csv.resolve()

    names_edible = Path(args.names_edible)
    if not names_edible.is_absolute():
        names_edible = (REPO_ROOT / names_edible)
    names_edible = names_edible.resolve()

    names_poisonous = Path(args.names_poisonous)
    if not names_poisonous.is_absolute():
        names_poisonous = (REPO_ROOT / names_poisonous)
    names_poisonous = names_poisonous.resolve()

    external_root = Path(args.external_root)
    local_root = Path(args.local_root)

    print("=" * 70)
    print("  Light Release Builder - External Storage Support")
    print("=" * 70)

    # Validate inputs
    if not external_root.exists():
        print(f"\n❌ External root not found: {external_root}")
        return 1

    if not local_root.exists():
        print(f"\n❌ Local root not found: {local_root}")
        return 1

    # Count what we'll be using
    external_dirs = get_plant_dirs(external_root)
    local_dirs = get_plant_dirs(local_root, skip_prefix="missing" if args.skip_missing else None)

    print(f"\n📂 Local folders:    {len(local_dirs)}")
    if args.skip_missing:
        print(f"   (excluding missing_*)")

    print(f"📂 External folders: {len(external_dirs)}")
    print(f"📂 Total to build:   {len(local_dirs) + len(external_dirs)}")

    # Create temp symlink tree
    print(f"\n🔗 Creating temporary symlink tree...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir) / "combined"
        temp_root.mkdir()

        linked = create_symlink_tree(local_root, external_root, temp_root, skip_missing=args.skip_missing)
        print(f"✅ Linked {linked} folders")

        # Build the command
        script = REPO_ROOT / "tools" / "build_light_release.py"
        cmd = [
            "python3",
            str(script),
            "--source-root",
            str(temp_root),
            "--index-csv",
            str(index_csv),
            "--names-edible", str(names_edible),
"--names-poisonous", str(names_poisonous),
        ]

        if args.split_by_class:
            cmd.append("--split-by-class")

        if args.keep_prefixes:
            cmd.append("--keep-prefixes")
        else:
            cmd.append("--strip-prefixes")

        if args.max_images_per_plant > 0:
            cmd.extend(["--max-images-per-plant", str(args.max_images_per_plant)])

        cmd.extend([
            "--max-width", str(args.max_width),
            "--max-long-edge", str(args.max_long_edge),
            "--quality", str(args.quality),
        ])

        print(f"\n📥 Running build_light_release.py...")
        print(f"\n Command: {' '.join(cmd)}\n")

        if args.dry_run:
            print("(Dry run: not executing)")
            return 0

        exit_code = subprocess.call(cmd, cwd=str(index_csv.parent))

        if exit_code == 0:
            print("\n" + "=" * 70)
            print("✅ Build completed successfully!")
            print(f"   Output: {REPO_ROOT / 'assets' / 'plant_images' / 'light_build'}")
            print("=" * 70)

        return exit_code


if __name__ == "__main__":
    sys.exit(main())
