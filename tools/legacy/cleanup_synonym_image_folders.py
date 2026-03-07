#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT.parent
ROOT = REPO_ROOT / "assets" / "plant_images" / "gbif_samples"
INDEX_JSON = ROOT / "index.json"
INDEX_CSV = ROOT / "index.csv"
PARTIAL_MANIFEST = ROOT / "partial_manifest.json"
CURATED_PLANTS = ROOT / "curated_plants.json"
SYNONYMS_JSON = APP_ROOT / "assets" / "data" / "synonyms.json"

KNOWN_PREFIXES = ["reviewed", "has_lookalike", "lookallike", "missing"]


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


def build_name(base_slug: str, flags: dict[str, bool]) -> str:
    prefixes = [p for p in KNOWN_PREFIXES if flags.get(p)]
    return ("_".join(prefixes) + "_" if prefixes else "") + base_slug


def load_synonym_slug_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = data.get("scientificNameToAccepted") if isinstance(data, dict) else None
    if not isinstance(root, dict):
        raise SystemExit(f"Invalid synonyms file: {path}")
    out: dict[str, str] = {}
    for k, v in root.items():
        from_slug = slugify_scientific(str(k))
        to_slug = slugify_scientific(str(v))
        if from_slug and to_slug and from_slug != to_slug:
            out[from_slug] = to_slug
    return out


def merge_dirs(src: Path, dst: Path, apply: bool) -> list[str]:
    actions: list[str] = []
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        target = dst / child.name
        if child.is_dir():
            if target.exists() and not target.is_dir():
                raise RuntimeError(f"Conflict: dir/file mismatch {child} -> {target}")
            actions.extend(merge_dirs(child, target, apply))
            if apply and child.exists():
                try:
                    child.rmdir()
                except OSError:
                    pass
            continue

        if target.exists():
            # Keep existing target file and drop duplicate source file.
            actions.append(f"DELETE_DUP {child} (target exists: {target})")
            if apply:
                child.unlink()
        else:
            actions.append(f"MOVE {child} -> {target}")
            if apply:
                shutil.move(str(child), str(target))

    if apply and src.exists():
        try:
            src.rmdir()
        except OSError:
            pass
    return actions


def update_index_json(mapping: dict[str, str], apply: bool) -> None:
    if not INDEX_JSON.exists() or not mapping:
        return
    data = json.loads(INDEX_JSON.read_text())
    changed = False
    for item in data.get("downloads", []):
        local_path = item.get("local_path")
        if not local_path:
            continue
        for old, new in mapping.items():
            old_tok = f"/gbif_samples/{old}/"
            new_tok = f"/gbif_samples/{new}/"
            if old_tok in local_path:
                item["local_path"] = local_path.replace(old_tok, new_tok)
                changed = True
                break
    if changed and apply:
        INDEX_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def update_index_csv(mapping: dict[str, str], apply: bool) -> None:
    if not INDEX_CSV.exists() or not mapping:
        return
    with INDEX_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "local_path" not in fieldnames:
        return
    changed = False
    for row in rows:
        local_path = row.get("local_path")
        if not local_path:
            continue
        for old, new in mapping.items():
            old_tok = f"/gbif_samples/{old}/"
            new_tok = f"/gbif_samples/{new}/"
            if old_tok in local_path:
                row["local_path"] = local_path.replace(old_tok, new_tok)
                changed = True
                break
    if changed and apply:
        with INDEX_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def update_partial_manifest(mapping: dict[str, str], apply: bool) -> None:
    if not PARTIAL_MANIFEST.exists() or not mapping:
        return
    data = json.loads(PARTIAL_MANIFEST.read_text())
    changed = False
    for item in data.get("images", []):
        for key in ("path",):
            path = item.get(key)
            if not path:
                continue
            for old, new in mapping.items():
                old_tok = f"/gbif_samples/{old}/"
                new_tok = f"/gbif_samples/{new}/"
                if old_tok in path:
                    item[key] = path.replace(old_tok, new_tok)
                    changed = True
                    break
        rel = item.get("rel")
        if rel:
            for old, new in mapping.items():
                if rel.startswith(f"{old}/"):
                    item["rel"] = rel.replace(f"{old}/", f"{new}/", 1)
                    changed = True
                    break
    if changed and apply:
        PARTIAL_MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def update_curated_plants(mapping: dict[str, str], apply: bool) -> None:
    if not CURATED_PLANTS.exists() or not mapping:
        return
    data = json.loads(CURATED_PLANTS.read_text())
    plants = data.get("plants")
    if not isinstance(plants, list):
        return
    updated = [mapping.get(str(s), str(s)) for s in plants]
    changed = updated != plants
    if changed and apply:
        data["plants"] = updated
        data["curated_count"] = len(updated)
        CURATED_PLANTS.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-off cleanup: rename/merge GBIF sample image folders from synonym slugs to canonical slugs."
    )
    ap.add_argument("--root", default=str(ROOT), help="gbif_samples root")
    ap.add_argument("--synonyms-json", default=str(SYNONYMS_JSON), help="Centralized synonyms JSON")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    args = ap.parse_args()

    root = Path(args.root)
    synonyms_map = load_synonym_slug_map(Path(args.synonyms_json))
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    folder_mapping: dict[str, str] = {}
    file_actions: list[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"selected", "selected_light"}:
            continue
        base_slug, flags = strip_prefixes(entry.name)
        canonical_base = synonyms_map.get(base_slug)
        if not canonical_base:
            continue
        target_name = build_name(canonical_base, flags)
        if target_name == entry.name:
            continue
        target = entry.parent / target_name
        folder_mapping[entry.name] = target_name
        if target.exists():
            file_actions.extend(merge_dirs(entry, target, args.apply))
        else:
            file_actions.append(f"RENAME {entry} -> {target}")
            if args.apply:
                os.rename(entry, target)

    update_index_json(folder_mapping, args.apply)
    update_index_csv(folder_mapping, args.apply)
    update_partial_manifest(folder_mapping, args.apply)
    update_curated_plants(folder_mapping, args.apply)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] synonym folders matched: {len(folder_mapping)}")
    for old, new in sorted(folder_mapping.items()):
        print(f"  {old} -> {new}")
    if file_actions:
        print(f"[{mode}] file actions: {len(file_actions)}")
        for line in file_actions[:200]:
            print(" ", line)
        if len(file_actions) > 200:
            print(f"  ... ({len(file_actions) - 200} more)")


if __name__ == "__main__":
    main()
