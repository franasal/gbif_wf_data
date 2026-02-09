#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


ROOT = Path("assets/plant_images/gbif_samples")
INDEX_JSON = ROOT / "index.json"
INDEX_CSV = ROOT / "index.csv"
PARTIAL_MANIFEST = ROOT / "partial_manifest.json"
CURATED_PLANTS = ROOT / "curated_plants.json"

POISONOUS_FILES = [
    Path("assets/data/poisonous/en.json"),
    Path("assets/data/poisonous/de.json"),
]

REVIEWED_MIN = 5

PREFIX_REVIEWED = "reviewed"
PREFIX_HAS_LOOKALIKE = "has_lookalike"
PREFIX_LOOKALLIKE = "lookallike"
PREFIX_MISSING = "missing"

KNOWN_PREFIXES = [
    PREFIX_REVIEWED,
    PREFIX_HAS_LOOKALIKE,
    PREFIX_LOOKALLIKE,
    PREFIX_MISSING,
]


def slugify_scientific(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def strip_prefixes(folder_name: str) -> tuple[str, dict[str, bool]]:
    base = folder_name
    flags = {p: False for p in KNOWN_PREFIXES}
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


def count_selected_light(folder: Path) -> int:
    selected = folder / "selected_light"
    if not selected.exists():
        return 0
    return sum(1 for p in selected.iterdir() if p.is_file())


def load_poisonous_sets() -> tuple[set[str], set[str]]:
    poisonous = set()
    lookalikes = set()
    for path in POISONOUS_FILES:
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        for entry in data:
            sci = entry.get("scientificName")
            if sci:
                poisonous.add(slugify_scientific(sci))
            for lk in entry.get("lookalikes", []):
                sci_lk = lk.get("scientificName")
                if sci_lk:
                    lookalikes.add(slugify_scientific(sci_lk))
    return poisonous, lookalikes


def build_prefix(
    is_reviewed: bool,
    is_poisonous: bool,
    has_lookalike: bool,
    keep_missing: bool,
) -> str:
    parts = []
    if is_reviewed:
        parts.append(PREFIX_REVIEWED)
    if keep_missing:
        parts.append(PREFIX_MISSING)
    if is_poisonous:
        parts.append(PREFIX_LOOKALLIKE)
    elif has_lookalike:
        parts.append(PREFIX_HAS_LOOKALIKE)
    if not parts:
        return ""
    return "_".join(parts) + "_"


def rename_folders() -> dict[str, str]:
    poisonous, lookalikes = load_poisonous_sets()
    mapping: dict[str, str] = {}
    for entry in sorted(ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"selected", "selected_light"}:
            continue
        base_slug, prefix_flags = strip_prefixes(entry.name)
        is_reviewed = count_selected_light(entry) >= REVIEWED_MIN
        is_poisonous = base_slug in poisonous
        has_lookalike = base_slug in lookalikes
        prefix = build_prefix(
            is_reviewed,
            is_poisonous,
            has_lookalike,
            prefix_flags.get(PREFIX_MISSING, False),
        )
        new_name = f"{prefix}{base_slug}"
        if new_name == entry.name:
            mapping[entry.name] = new_name
            continue
        target = entry.parent / new_name
        if target.exists():
            raise RuntimeError(f"Target folder already exists: {target}")
        os.rename(entry, target)
        mapping[entry.name] = new_name
    return mapping


def update_index_json(mapping: dict[str, str]) -> None:
    if not INDEX_JSON.exists():
        return
    with INDEX_JSON.open() as f:
        data = json.load(f)
    downloads = data.get("downloads", [])
    for item in downloads:
        local_path = item.get("local_path")
        if not local_path:
            continue
        for old, new in mapping.items():
            token_old = f"/gbif_samples/{old}/"
            token_new = f"/gbif_samples/{new}/"
            if token_old in local_path:
                item["local_path"] = local_path.replace(token_old, token_new)
                break
    with INDEX_JSON.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def update_index_csv(mapping: dict[str, str]) -> None:
    if not INDEX_CSV.exists():
        return
    with INDEX_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "local_path" not in fieldnames:
        return
    for row in rows:
        local_path = row.get("local_path")
        if not local_path:
            continue
        for old, new in mapping.items():
            token_old = f"/gbif_samples/{old}/"
            token_new = f"/gbif_samples/{new}/"
            if token_old in local_path:
                row["local_path"] = local_path.replace(token_old, token_new)
                break
    with INDEX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_partial_manifest(mapping: dict[str, str]) -> None:
    if not PARTIAL_MANIFEST.exists():
        return
    with PARTIAL_MANIFEST.open() as f:
        data = json.load(f)
    images = data.get("images", [])
    for item in images:
        path = item.get("path")
        rel = item.get("rel")
        if path:
            for old, new in mapping.items():
                token_old = f"/gbif_samples/{old}/"
                token_new = f"/gbif_samples/{new}/"
                if token_old in path:
                    item["path"] = path.replace(token_old, token_new)
                    break
        if rel:
            for old, new in mapping.items():
                token_old = f"{old}/"
                token_new = f"{new}/"
                if rel.startswith(token_old):
                    item["rel"] = rel.replace(token_old, token_new, 1)
                    break
    with PARTIAL_MANIFEST.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def update_curated_plants(mapping: dict[str, str]) -> None:
    if not CURATED_PLANTS.exists():
        return
    with CURATED_PLANTS.open() as f:
        data = json.load(f)
    plants = data.get("plants", [])
    updated = []
    for slug in plants:
        if slug in mapping:
            updated.append(mapping[slug])
        else:
            updated.append(slug)
    data["plants"] = updated
    data["curated_count"] = len(updated)
    with CURATED_PLANTS.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Missing folder: {ROOT}")
    mapping = rename_folders()
    update_index_json(mapping)
    update_index_csv(mapping)
    update_partial_manifest(mapping)
    update_curated_plants(mapping)
    print(f"Renamed {sum(1 for k, v in mapping.items() if k != v)} folders.")


if __name__ == "__main__":
    main()
