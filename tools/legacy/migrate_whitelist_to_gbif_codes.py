#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert legacy name-based whitelist to GBIF taxonKey-keyed whitelist using plants_meta.json."
    )
    ap.add_argument("--names", required=True, help="Legacy JSON dict: {scientificName: commonName}")
    ap.add_argument("--plants-meta", default="assets/data/plants_meta.json", help="Path to plants_meta.json")
    ap.add_argument("--out", required=True, help="Output JSON keyed by GBIF taxonKey")
    ap.add_argument("--include-synonyms", action="store_true", help="Keep entries marked isSynonym in plants_meta")
    args = ap.parse_args()

    names_path = Path(args.names)
    meta_path = Path(args.plants_meta)
    out_path = Path(args.out)

    names = load_json(names_path, {})
    if not isinstance(names, dict) or not all(isinstance(v, str) for v in names.values()):
        raise SystemExit("--names must be a legacy dict {scientificName: commonName}")

    meta = load_json(meta_path, {})
    plants = meta.get("plants")
    if not isinstance(plants, list):
        raise SystemExit("plants_meta.json must contain a top-level 'plants' list")

    by_sci = {}
    by_id = {}
    for p in plants:
        if not isinstance(p, dict):
            continue
        sci = p.get("scientificName")
        pid = p.get("id")
        if isinstance(sci, str) and sci.strip():
            by_sci[sci.strip()] = p
        if isinstance(pid, str) and pid.strip():
            by_id[pid.strip()] = p

    out = {}
    report = {
        "total_input": len(names),
        "written": 0,
        "skipped_missing_meta": [],
        "skipped_missing_taxon_key": [],
        "skipped_synonyms": [],
        "deduped_by_taxon_key": [],
    }

    for sci, common in sorted(names.items(), key=lambda kv: kv[0].lower()):
        sci = str(sci).strip()
        common = str(common).strip()
        p = by_sci.get(sci)
        if not p:
            report["skipped_missing_meta"].append(sci)
            continue

        if p.get("isSynonym") and not args.include_synonyms:
            synonym_of = p.get("synonymOf")
            accepted = by_id.get(synonym_of) if isinstance(synonym_of, str) else None
            report["skipped_synonyms"].append(
                {
                    "scientificName": sci,
                    "synonymOf": synonym_of,
                    "acceptedScientificName": (accepted.get("scientificName") if isinstance(accepted, dict) else None),
                }
            )
            continue

        tk = p.get("taxonKey")
        try:
            tk = int(tk) if tk is not None else None
        except Exception:
            tk = None
        if tk is None:
            report["skipped_missing_taxon_key"].append(sci)
            continue

        key = str(tk)
        item = {
            "taxonKey": tk,
            "scientificName": str(p.get("scientificName") or sci),
            "de": common,
            "id": p.get("id"),
        }

        if key in out:
            report["deduped_by_taxon_key"].append(
                {"taxonKey": tk, "kept": out[key]["scientificName"], "dropped": item["scientificName"]}
            )
            continue

        out[key] = item

    report["written"] = len(out)
    save_json(out_path, out)
    print(f"Wrote {out_path} ({len(out)} entries)")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
