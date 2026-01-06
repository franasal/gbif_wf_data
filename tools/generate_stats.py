#!/usr/bin/env python3
import argparse
import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def read_json_maybe_gz(path: Path) -> Dict[str, Any]:
    """
    Reads JSON from:
      - foo.json
      - foo.json.gz

    If the given path doesn't exist, also tries path + ".gz".
    """
    p = path
    if not p.exists():
        alt = Path(str(p) + ".gz")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Missing occ json: {path} (also tried {alt})")

    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)

    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate summary stats from occurrences_compact (json or json.gz).")
    ap.add_argument("--db", required=True, help="Path to dwca.sqlite")
    ap.add_argument("--occ-json", required=True, help="Path to occurrences_compact.json or .json.gz")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--country", default="DE")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    args = ap.parse_args()

    occ_path = Path(args.occ_json)
    data = read_json_maybe_gz(occ_path)

    plants = data.get("plants", {}) or {}

    totals = []
    sampled_points_total = 0
    for sci, obj in plants.items():
        total = int(obj.get("total") or 0)
        pts = obj.get("points") or []
        sampled_points_total += len(pts)
        totals.append((sci, total))

    totals.sort(key=lambda x: x[1], reverse=True)

    out_obj = {
        "meta": {
            "source_occ_json": str(occ_path),
            "country": args.country,
            "year_from": args.year_from,
            "year_to": args.year_to,
        },
        "counts": {
            "plants": len(plants),
            "sampled_points_total": sampled_points_total,
        },
        "top_by_total": [{"sci": s, "total": t} for s, t in totals[:50]],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
