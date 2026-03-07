#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

KNOWN_PREFIXES = ["reviewed", "has_lookalike", "lookallike", "missing"]

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT.parent
SYNONYMS_JSON_DEFAULT = APP_ROOT / "assets" / "data" / "synonyms.json"


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


def strip_prefixes(folder_name: str) -> str:
    base = folder_name
    changed = True
    while changed:
        changed = False
        for prefix in KNOWN_PREFIXES:
            token = f"{prefix}_"
            if base.startswith(token):
                base = base[len(token) :]
                changed = True
    return base


def load_names(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Missing names file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid names file (expected JSON object): {path}")
    return data


def collect_curated_slugs(curated_root: Path, synonym_slug_map: Dict[str, str]) -> List[str]:
    if not curated_root.exists():
        raise SystemExit(f"Missing curated root: {curated_root}")
    slugs: List[str] = []
    for entry in curated_root.iterdir():
        if not entry.is_dir():
            continue
        base = strip_prefixes(entry.name)
        base = canonical_slug(base, synonym_slug_map)
        if base:
            slugs.append(base)
    return sorted(set(slugs))


def compute_missing(
    names: Dict[str, str],
    curated_slugs: List[str],
    synonym_slug_map: Dict[str, str],
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, List[str]]]:
    expected_map: Dict[str, List[str]] = {}
    for sci in names.keys():
        slug = canonical_slug(slugify_scientific(sci), synonym_slug_map)
        expected_map.setdefault(slug, []).append(sci)

    expected_slugs = set(expected_map.keys())
    present_slugs = set(curated_slugs)

    missing_slugs = sorted(expected_slugs - present_slugs)
    unknown_present = sorted(present_slugs - expected_slugs)

    missing_entries: List[Dict[str, object]] = []
    for slug in missing_slugs:
        names_list = sorted(expected_map.get(slug, []))
        missing_entries.append(
            {
                "slug": slug,
                "scientificName": names_list[0] if names_list else "",
                "allNames": names_list,
            }
        )

    return missing_entries, unknown_present, expected_map


def write_output(
    out_path: Path,
    class_label: str,
    missing_entries: List[Dict[str, object]],
    unknown_present: List[str],
    expected_map: Dict[str, List[str]],
    curated_slugs: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "class": class_label,
        "expected_count": len(expected_map),
        "present_count": len(curated_slugs),
        "missing_count": len(missing_entries),
        "unknown_present_count": len(unknown_present),
        "missing": missing_entries,
        "unknown_present": unknown_present,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List plants missing curated light images by comparing names_* to curated folders."
    )
    parser.add_argument(
        "--names-edible",
        default="gbif_wf_data/data/names_edible.json",
        help="Path to names_edible.json",
    )
    parser.add_argument(
        "--names-poisonous",
        default="gbif_wf_data/data/names_poisonous.json",
        help="Path to names_poisonous.json",
    )
    parser.add_argument(
        "--curated-root",
        default="gbif_wf_data/assets/plant_images/light_build",
        help="Root containing curated edible/poisonous folders.",
    )
    parser.add_argument(
        "--synonyms-json",
        default=str(SYNONYMS_JSON_DEFAULT),
        help="Path to centralized synonyms.json (main app assets).",
    )
    parser.add_argument(
        "--out-edible",
        default="gbif_wf_data/data/missing_light_edible.json",
        help="Output JSON for missing edible plants.",
    )
    parser.add_argument(
        "--out-poisonous",
        default="gbif_wf_data/data/missing_light_poisonous.json",
        help="Output JSON for missing poisonous plants.",
    )
    args = parser.parse_args()

    synonym_slug_map = load_synonym_slug_map(Path(args.synonyms_json))
    curated_root = Path(args.curated_root)

    edible_root = curated_root / "edible"
    poisonous_root = curated_root / "poisonous"

    edible_names = load_names(Path(args.names_edible))
    poisonous_names = load_names(Path(args.names_poisonous))

    edible_slugs = collect_curated_slugs(edible_root, synonym_slug_map)
    poisonous_slugs = collect_curated_slugs(poisonous_root, synonym_slug_map)

    edible_missing, edible_unknown, edible_expected = compute_missing(
        edible_names, edible_slugs, synonym_slug_map
    )
    poisonous_missing, poisonous_unknown, poisonous_expected = compute_missing(
        poisonous_names, poisonous_slugs, synonym_slug_map
    )

    write_output(
        Path(args.out_edible),
        "edible",
        edible_missing,
        edible_unknown,
        edible_expected,
        edible_slugs,
    )
    write_output(
        Path(args.out_poisonous),
        "poisonous",
        poisonous_missing,
        poisonous_unknown,
        poisonous_expected,
        poisonous_slugs,
    )

    print("Done.")
    print(f"Edible missing: {len(edible_missing)} -> {args.out_edible}")
    print(f"Poisonous missing: {len(poisonous_missing)} -> {args.out_poisonous}")


if __name__ == "__main__":
    main()
