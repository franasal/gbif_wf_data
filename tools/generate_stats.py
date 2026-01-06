#!/usr/bin/env python3
import argparse
import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def read_json_maybe_gz(path: Path) -> Dict[str, Any]:
    """
    Accepts:
      - occurrences_compact.json
      - occurrences_compact.json.gz
    If the given path doesn't exist, it will also try path + ".gz".
    """
    p = path
    if not p.exists():
        gz = Path(str(p) + ".gz")
        if gz.exists():
            p = gz
        else:
            raise FileNotFoundError(f"occ json not found: {path} (also tried {gz})")

    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)

    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate stats summary for the dataset (reads JSON or JSON.GZ).")
    ap.add_argument("--db", required=True, help="Path to dwca.sqlite")
    ap.add_argument("--occ-json", required=True, help="Path to occurrences_compact.json or .json.gz")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. data/stats_summary.json)")
    ap.add_argument("--country", default="DE")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    args = ap.parse_args()

    occ_path = Path(args.occ_json)
    data = read_json_maybe_gz(occ_path)

    plants: Dict[str, Any] = data.get("plants", {})
    meta: Dict[str, Any] = data.get("meta", {})

    # Minimal useful summary (you can expand)
    # - total plants
    # - total sampled points
    # - top 25 by total
    total_plants = len(plants)
    total_points = 0

    totals = []
    for sci, obj in plants.items():
        pts = obj.get("points") or []
        total_points += len(pts)
        totals.append((sci, int(obj.get("total") or 0)))

    totals.sort(key=lambda x: x[1], reverse=True)
    top_25 = [{"sci": s, "total": t} for s, t in totals[:25]]

    out_obj = {
        "meta": {
            "source_occ_json": str(occ_path),
            "generated_from": meta,
            "country": args.country,
            "year_from": args.year_from,
            "year_to": args.year_to,
        },
        "counts": {
            "plants": total_plants,
            "sampled_points_total": total_points,
        },
        "top_by_total": top_25,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
