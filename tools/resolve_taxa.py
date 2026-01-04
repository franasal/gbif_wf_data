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

    name_map: Dict[str, str] = load_json(names_path, {})
    if not isinstance(name_map, dict):
        raise SystemExit("--names must be a JSON dict { 'Latin name': 'Common name', ... }")

    cache: Dict[str, Any] = load_json(cache_path, {})
    out: List[Dict[str, Any]] = []

    unresolved: List[Tuple[str, str]] = []

    for latin, common in sorted(name_map.items(), key=lambda kv: kv[0].lower()):
        latin = (latin or "").strip()
        common = (common or "").strip()
        if not latin:
            continue

        if latin in cache:
            j = cache[latin]
        else:
            j = gbif_match(latin, strict=not args.allow_fuzzy)
            cache[latin] = j
            time.sleep(args.sleep)

        usage = j.get("usageKey")
        ok = score_ok(j) if not args.allow_fuzzy else bool(usage)

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
