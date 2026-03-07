#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests

MATCH_URL = "https://api.gbif.org/v1/species/match"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_names_input(data: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
        for latin, common in data.items():
            entries.append({"scientificName": str(latin), "de": str(common), "taxonKey": None})
        return entries

    if isinstance(data, dict):
        iterable = data.items()
        for key, value in iterable:
            if not isinstance(value, dict):
                continue
            sci = value.get("scientificName") or value.get("scientific_name")
            common = value.get("de") or value.get("commonName") or value.get("common_name") or ""
            tk = value.get("taxonKey", key)
            try:
                tk = int(tk) if tk is not None else None
            except Exception:
                tk = None
            entries.append({"scientificName": sci, "de": common, "taxonKey": tk})
        return entries

    if isinstance(data, list):
        for value in data:
            if not isinstance(value, dict):
                continue
            sci = value.get("scientificName") or value.get("scientific_name")
            common = value.get("de") or value.get("commonName") or value.get("common_name") or ""
            tk = value.get("taxonKey")
            try:
                tk = int(tk) if tk is not None else None
            except Exception:
                tk = None
            entries.append({"scientificName": sci, "de": common, "taxonKey": tk})
        return entries

    return entries


def gbif_match(name: str, kingdom: str = "Plantae", rank: str = "SPECIES", strict: bool = True) -> Dict[str, Any]:
    params = {
        "name": name,
        "kingdom": kingdom,
        "rank": rank,
        "strict": "true" if strict else "false",
        "verbose": "true",
    }
    r = requests.get(MATCH_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def score_ok(j: Dict[str, Any]) -> bool:
    mt = (j.get("matchType") or "").upper()
    conf = j.get("confidence")
    try:
        conf = int(conf) if conf is not None else 0
    except Exception:
        conf = 0
    return (mt in {"EXACT", "HIGHERRANK"} and conf >= 90) or (mt == "EXACT" and conf >= 80)


def main():
    ap = argparse.ArgumentParser(description="Resolve scientific names to GBIF backbone taxonKey (usageKey).")
    ap.add_argument("--names", required=True, help="Path to JSON dict {latin: common}")
    ap.add_argument("--out", required=True, help="Output JSON list with taxonKey + match metadata")
    ap.add_argument("--cache", default="data/taxon_cache.json", help="Cache file to avoid re-querying unchanged names")
    ap.add_argument("--sleep", type=float, default=0.2, help="Delay between GBIF calls")
    ap.add_argument("--allow-fuzzy", action="store_true", help="Accept fuzzy matches (not recommended)")
    args = ap.parse_args()

    names_path = Path(args.names)
    out_path = Path(args.out)
    cache_path = Path(args.cache)

    names_raw = load_json(names_path, {})
    entries = parse_names_input(names_raw)
    if not entries:
        raise SystemExit(
            "--names must be legacy {latin: common} or objects containing scientificName (+ optional taxonKey/de)."
        )

    cache: Dict[str, Any] = load_json(cache_path, {})
    out: List[Dict[str, Any]] = []

    unresolved: List[Tuple[str, str]] = []

    for entry in sorted(entries, key=lambda x: str(x.get("scientificName") or "").lower()):
        latin = str(entry.get("scientificName") or "").strip()
        common = str(entry.get("de") or "").strip()
        preset_taxon = entry.get("taxonKey")
        if not latin:
            continue

        if preset_taxon is not None:
            j = {
                "usageKey": int(preset_taxon),
                "matchType": "EXACT",
                "confidence": 100,
                "canonicalName": latin,
                "scientificName": latin,
                "rank": "SPECIES",
                "status": "ACCEPTED",
            }
        elif latin in cache:
            j = cache[latin]
        else:
            j = gbif_match(latin, strict=not args.allow_fuzzy)
            cache[latin] = j
            time.sleep(args.sleep)

        usage = j.get("usageKey")
        ok = True if preset_taxon is not None else (score_ok(j) if not args.allow_fuzzy else bool(usage))

        item = {
            "scientificName": latin,
            "de": common,
            "taxonKey": int(usage) if usage is not None else None,
            "match": {
                "matchType": j.get("matchType"),
                "confidence": j.get("confidence"),
                "canonicalName": j.get("canonicalName"),
                "scientificName": j.get("scientificName"),
                "rank": j.get("rank"),
                "status": j.get("status"),
            },
        }

        if not ok or item["taxonKey"] is None:
            unresolved.append((latin, j.get("matchType") or "NONE"))
        out.append(item)

    save_json(cache_path, cache)
    save_json(out_path, out)

    if unresolved:
        print("Unresolved / suspicious matches:")
        for latin, mt in unresolved:
            print(f"  - {latin} (matchType={mt})")
        raise SystemExit("Refusing to continue with unresolved taxa. Fix names or use --allow-fuzzy.")

    print(f"Wrote: {out_path} ({len(out)} taxa)")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    main()
