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


def _resolve_json_path(p: Path) -> Path:
    """
    Accepts:
      - occurrences_compact.json
      - occurrences_compact.json.gz

    If p doesn't exist, tries p + ".gz".
    """
    if p.exists():
        return p
    alt = Path(str(p) + ".gz")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Missing occ json: {p} (also tried {alt})")


def read_json_maybe_gz(path: Path) -> Dict[str, Any]:
    p = _resolve_json_path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text(encoding="utf-8"))


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate summary stats from occurrences_compact (json or json.gz).")
    ap.add_argument("--db", required=True, help="Path to dwca.sqlite")
    ap.add_argument("--occ-json", required=True, help="Path to occurrences_compact.json or occurrences_compact.json.gz")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. data/stats_summary.json)")
    ap.add_argument("--country", default="DE")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    args = ap.parse_args()

    occ_path_in = Path(args.occ_json)
    occ_path = _resolve_json_path(occ_path_in)
    data = read_json_maybe_gz(occ_path)

    plants = data.get("plants", {}) or {}
    sampled_points_total = 0

    totals = []
    for sci, obj in plants.items():
        total = safe_int(obj.get("total"), 0)
        pts = obj.get("points") or obj.get("pointsSample") or []
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
