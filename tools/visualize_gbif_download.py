#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def list_tables(con: sqlite3.Connection) -> List[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r["name"] for r in rows]


def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]


def pick_occurrence_table(con: sqlite3.Connection) -> str:
    preferred = ["occ", "occurrence", "occurrences", "gbif_occurrence", "gbif_occurrences"]
    tables = list_tables(con)
    for table in preferred:
        if table in tables:
            return table
    raise RuntimeError(f"Could not find occurrence table. Available tables: {tables}")


def resolve_col(cols: List[str], options: List[str], required: bool = True) -> Optional[str]:
    s = set(cols)
    for opt in options:
        if opt in s:
            return opt
    if required:
        raise RuntimeError(f"Missing required column. Tried {options}. Available: {cols}")
    return None


def load_names(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return sorted([k.strip() for k in data.keys() if isinstance(k, str) and k.strip()])


def _cell_index(lat: float, lon: float, cell_size_deg: float) -> Tuple[int, int]:
    lat_idx = int(math.floor((lat + 90.0) / cell_size_deg))
    lon_idx = int(math.floor((lon + 180.0) / cell_size_deg))
    return lat_idx, lon_idx


def _cell_bounds(lat_idx: int, lon_idx: int, cell_size_deg: float) -> Tuple[float, float, float, float]:
    lat_min = -90.0 + lat_idx * cell_size_deg
    lat_max = lat_min + cell_size_deg
    lon_min = -180.0 + lon_idx * cell_size_deg
    lon_max = lon_min + cell_size_deg
    return lat_min, lat_max, lon_min, lon_max


def _polygon_feature(lat_min: float, lat_max: float, lon_min: float, lon_max: float, count: int) -> dict:
    return {
        "type": "Feature",
        "properties": {"count": count},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a whole-download visualization summary from GBIF SQLite before downsampling.",
    )
    ap.add_argument("--db", default="data/dwca.sqlite", help="Path to SQLite DB.")
    ap.add_argument(
        "--names-json",
        default=None,
        help="Optional whitelist of scientific names (JSON object keys).",
    )
    ap.add_argument("--country", default=None, help="Optional country filter (e.g. DE).")
    ap.add_argument("--year-from", type=int, default=None, help="Optional lower year bound.")
    ap.add_argument("--year-to", type=int, default=None, help="Optional upper year bound.")
    ap.add_argument(
        "--cell-size-deg",
        type=float,
        default=0.25,
        help="Grid cell size in degrees for whole-download map density.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/visualizations",
        help="Directory for generated summary files.",
    )
    ap.add_argument(
        "--top-species",
        type=int,
        default=50,
        help="How many species to keep in the top-species CSV/JSON summary.",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    names_path = Path(args.names_json) if args.names_json else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect(db_path)
    table = pick_occurrence_table(con)
    cols = table_columns(con, table)

    col_sci = resolve_col(cols, ["species", "scientificName", "speciesName", "taxon_name"])
    col_lat = resolve_col(cols, ["lat", "decimalLatitude", "latitude"])
    col_lon = resolve_col(cols, ["lon", "decimalLongitude", "longitude"])
    col_year = resolve_col(cols, ["year"], required=False)
    col_country = resolve_col(cols, ["countryCode", "country_code", "country"], required=False)

    where = [f"{col_lat} IS NOT NULL", f"{col_lon} IS NOT NULL"]
    params: List[object] = []

    names = load_names(names_path)
    if names:
        con.execute("DROP TABLE IF EXISTS _wanted_species_vis;")
        con.execute("CREATE TEMP TABLE _wanted_species_vis (sci TEXT PRIMARY KEY);")
        con.executemany("INSERT OR IGNORE INTO _wanted_species_vis(sci) VALUES (?)", [(n,) for n in names])
        where.append(f"{col_sci} IN (SELECT sci FROM _wanted_species_vis)")

    if args.country and col_country:
        where.append(f"{col_country} = ?")
        params.append(args.country)
    if args.year_from is not None and col_year:
        where.append(f"{col_year} >= ?")
        params.append(args.year_from)
    if args.year_to is not None and col_year:
        where.append(f"{col_year} <= ?")
        params.append(args.year_to)

    where_sql = " AND ".join(where)
    query = f"SELECT {col_sci} AS sci, {col_lat} AS lat, {col_lon} AS lon FROM {table} WHERE {where_sql}"

    species_counts: Counter = Counter()
    cell_counts: Dict[Tuple[int, int], int] = {}
    total_rows = 0
    bad_coords = 0
    min_lat = None
    max_lat = None
    min_lon = None
    max_lon = None

    for row in con.execute(query, params):
        total_rows += 1
        sci = (row["sci"] or "").strip()
        if sci:
            species_counts[sci] += 1
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except Exception:
            bad_coords += 1
            continue
        if lat < -90 or lat > 90 or lon < -180 or lon > 180:
            bad_coords += 1
            continue

        cell = _cell_index(lat, lon, args.cell_size_deg)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

        min_lat = lat if min_lat is None else min(min_lat, lat)
        max_lat = lat if max_lat is None else max(max_lat, lat)
        min_lon = lon if min_lon is None else min(min_lon, lon)
        max_lon = lon if max_lon is None else max(max_lon, lon)

    top_species = species_counts.most_common(max(1, args.top_species))
    total_species = len(species_counts)

    geojson = {
        "type": "FeatureCollection",
        "features": [],
    }
    for (lat_idx, lon_idx), count in sorted(cell_counts.items(), key=lambda kv: kv[1], reverse=True):
        lat_min, lat_max, lon_min, lon_max = _cell_bounds(lat_idx, lon_idx, args.cell_size_deg)
        geojson["features"].append(
            _polygon_feature(lat_min, lat_max, lon_min, lon_max, count),
        )

    summary = {
        "generated_at": utc_now_iso(),
        "source_db": str(db_path),
        "filters": {
            "names_json": str(names_path) if names_path else None,
            "country": args.country,
            "year_from": args.year_from,
            "year_to": args.year_to,
        },
        "pre_sampling": {
            "row_count": total_rows,
            "species_count": total_species,
            "bad_or_out_of_range_coords": bad_coords,
            "grid_cell_size_deg": args.cell_size_deg,
            "grid_cell_count": len(cell_counts),
            "bbox": [min_lat, max_lat, min_lon, max_lon],
            "top_species": [{"scientificName": n, "count": c} for n, c in top_species],
        },
    }

    summary_path = out_dir / "gbif_pre_sampling_summary.json"
    grid_geojson_path = out_dir / "gbif_pre_sampling_grid.geojson"
    top_species_csv_path = out_dir / "gbif_pre_sampling_top_species.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    grid_geojson_path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    with top_species_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scientificName", "count"])
        for name, count in top_species:
            writer.writerow([name, count])

    print(f"Wrote {summary_path}")
    print(f"Wrote {grid_geojson_path}")
    print(f"Wrote {top_species_csv_path}")
    print("Tip: open the GeoJSON in geojson.io or QGIS for a whole-download density view.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
