#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build a stats summary from occurrences_compact.json(.gz)")
    ap.add_argument("--in", dest="inp", required=True, help="data/occurrences_compact.json or .gz")
    ap.add_argument("--out-json", default="data/stats_summary.json")
    ap.add_argument("--out-md", default="data/stats_summary.md")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    data = read_json(Path(args.inp))
    plants = data.get("plants", {})
    meta = data.get("meta", {})
    region = data.get("region", {})

    # global totals (sample totals, since your dataset is sparse by design)
    total_points = 0
    total_raw = 0

    rows = []
    year_global = defaultdict(int)

    for sci, p in plants.items():
        total = int(p.get("total", 0) or 0)
        raw = int(p.get("total_raw", 0) or 0)
        total_points += total
        total_raw += raw

        yc = p.get("year_counts", {}) or {}
        for y, n in yc.items():
            try:
                year_global[str(y)] += int(n)
            except Exception:
                pass

        coverage_cells = int(p.get("coverage_cells", 0) or 0)
        coverage_total = int(p.get("coverage_cells_total", 0) or 0)
        coverage_pct = (coverage_cells / coverage_total) if coverage_total else 0.
