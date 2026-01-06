#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import heapq

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")      # ~200MB
        con.execute("PRAGMA mmap_size=268435456;")     # 256MB
    except Exception:
        pass
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
    for t in preferred:
        if t in tables:
            return t
    raise RuntimeError(f"Could not identify occurrence table. Tables: {tables}")


def resolve_col(cols: List[str], options: List[str], required: bool = True) -> Optional[str]:
    s = set(cols)
    for o in options:
        if o in s:
            return o
    if required:
        raise RuntimeError(f"Missing required column. Tried {options}. Available: {cols}")
    return None


def load_name_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
        return data
    raise RuntimeError("names JSON must be a dict: { 'Scientific name': 'German name', ... }")


def geohash_encode(lat: float, lon: float, precision: int = 5) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon >= mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat >= mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def ym_to_key(y: Optional[int], m: Optional[int], d: Optional[int]) -> int:
    # Comparable recency key, higher = newer
    yy = int(y) if y is not None else 0
    mm = int(m) if m is not None else 0
    dd = int(d) if d is not None else 0
    if mm < 0 or mm > 12:
        mm = 0
    if dd < 0 or dd > 31:
        dd = 0
    return yy * 10000 + mm * 100 + dd


def write_output(path: str, obj: dict, gzip_enabled: bool) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if gzip_enabled or path.endswith(".gz"):
        if not path.endswith(".gz"):
            path = path + ".gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"\nWrote (gzip): {path}", flush=True)
        return path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nWrote: {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Export coverage-first occurrences_compact.json (cell-centric, newest per cell)."
    )
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--country", default="DE", help="Country code filter (default: DE). Empty disables.")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)

    ap.add_argument("--top-n", type=int, default=250, help="Pick top N plants by raw total before sparsifying.")
    ap.add_argument("--grid-geohash", type=int, default=4, help="Geohash precision (4 ~20km, 5 ~5km).")
    ap.add_argument("--cell-top-plants", type=int, default=220, help="Keep top plants per cell by count.")
    ap.add_argument("--newest-per-plant-per-cell", type=int, default=5, help="Keep newest N points per plant per cell.")

    ap.add_argument("--max-points-per-plant", type=int, default=1200, help="Final cap after merging cells (0 disables).")
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--progress-every", type=int, default=250000)

    ap.add_argument("--region-name", default="Germany")
    ap.add_argument("--region-lat", type=float, default=51.0)
    ap.add_argument("--region-lon", type=float, default=10.0)

    ap.add_argument("--names-json", default=None, help="Scientific->German name mapping and implicit plant list.")
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    try:
        t_all = time.time()

        name_map = load_name_map(args.names_json)
        if not name_map:
            raise RuntimeError("names-json is required here (it defines the plant list).")

        con = connect(args.db)
        table = pick_occurrence_table(con)
        cols = table_columns(con, table)

        # prefer species over scientificName (less author noise)
        col_sci = resolve_col(cols, ["species", "scientificName", "speciesName", "taxon_name"])
        col_taxon = resolve_col(cols, ["taxonKey", "taxon_key"], required=False)

        col_lat = resolve_col(cols, ["lat", "decimalLatitude", "latitude"])
        col_lon = resolve_col(cols, ["lon", "decimalLongitude", "longitude"])
        col_year = resolve_col(cols, ["year"], required=False)
        col_month = resolve_col(cols, ["month"], required=False)
        col_day = resolve_col(cols, ["day"], required=False)
        col_country = resolve_col(cols, ["countryCode", "country_code", "country"], required=False)

        # 1) Base WHERE
        where = [f"{col_lat} IS NOT NULL", f"{col_lon} IS NOT NULL"]
        params: List[object] = []

        if args.country and col_country:
            where.append(f"{col_country} = ?")
            params.append(args.country)

        if args.year_from is not None and col_year:
            where.append(f"{col_year} >= ?")
            params.append(args.year_from)
        if args.year_to is not None and col_year:
            where.append(f"{col_year} <= ?")
            params.append(args.year_to)

        where_sql = "WHERE " + " AND ".join(where)

        # 2) pick top_n plants by raw totals, but only from your maintained list
        #    (so the repo controls what exists)
        plant_list = sorted(name_map.keys())
        # SQLite has a param limit; chunk plant_list into ORs if huge.
        # For your size (~400) this is fine:
        placeholders = ",".join(["?"] * len(plant_list))
        where_top = where_sql + f" AND {col_sci} IN ({placeholders})"

        q_top = f"""
            SELECT {col_sci} AS sci, COUNT(1) AS n
            FROM {table}
            {where_top}
            GROUP BY {col_sci}
            ORDER BY n DESC
            LIMIT ?
        """
        top_rows = con.execute(q_top, params + plant_list + [args.top_n]).fetchall()
        top_species = [r["sci"] for r in top_rows if r["sci"]]
        raw_totals = {r["sci"]: int(r["n"]) for r in top_rows if r["sci"]}

        if not top_species:
            raise RuntimeError("No plants matched filters + names-json list. Check DB content / filters.")

        top_set = set(top_species)
        print(f"[info] plants: {len(top_species)} grid_geohash={args.grid_geohash}", flush=True)

        # 3) One streaming pass across ALL rows (filtered) and bucket into grid cells
        # Structures:
        # cell_counts[cell][plant] = count
        # cell_heaps[cell][plant] = min-heap of (timekey, point)
        cell_counts: Dict[str, Dict[str, int]] = {}
        cell_heaps: Dict[str, Dict[str, List[Tuple[int, list]]]] = {}

        # also track plant->taxonKey first seen
        taxon_map: Dict[str, Optional[int]] = {}

        select_cols = [f"{col_sci} AS sci", f"{col_lat} AS lat", f"{col_lon} AS lon"]
        if col_taxon:
            select_cols.append(f"{col_taxon} AS taxonKey")
        else:
            select_cols.append("NULL AS taxonKey")
        select_cols.append((f"{col_year} AS y") if col_year else "NULL AS y")
        select_cols.append((f"{col_month} AS m") if col_month else "NULL AS m")
        select_cols.append((f"{col_day} AS d") if col_day else "NULL AS d")

        q_stream = f"SELECT {', '.join(select_cols)} FROM {table} {where_sql}"
        cur = con.execute(q_stream, params)

        processed = 0
        newest_n = max(1, int(args.newest_per_plant_per_cell))

        while True:
            chunk = cur.fetchmany(max(1000, int(args.chunk_size)))
            if not chunk:
                break

            for r in chunk:
                sci = r["sci"]
                if sci not in top_set:
                    continue

                try:
                    lat = float(r["lat"])
                    lon = float(r["lon"])
                except Exception:
                    continue

                y = r["y"]
                m = r["m"]
                d = r["d"]
                try:
                    yi = int(y) if y is not None else None
                except Exception:
                    yi = None
                try:
                    mi = int(m) if m is not None else None
                except Exception:
                    mi = None
                try:
                    di = int(d) if d is not None else None
                except Exception:
                    di = None

                tkey = ym_to_key(yi, mi, di)
                cell = geohash_encode(lat, lon, precision=int(args.grid_geohash))
                point = [lat, lon, yi, mi]

                # count
                byplant = cell_counts.get(cell)
                if byplant is None:
                    byplant = {}
                    cell_counts[cell] = byplant
                byplant[sci] = byplant.get(sci, 0) + 1

                # heap newest per plant per cell
                heaps_byplant = cell_heaps.get(cell)
                if heaps_byplant is None:
                    heaps_byplant = {}
                    cell_heaps[cell] = heaps_byplant
                h = heaps_byplant.get(sci)
                if h is None:
                    h = []
                    heaps_byplant[sci] = h

                if len(h) < newest_n:
                    heapq.heappush(h, (tkey, point))
                else:
                    # min-heap keeps the N newest (largest keys)
                    if tkey > h[0][0]:
                        heapq.heapreplace(h, (tkey, point))

                # taxonKey
                if sci not in taxon_map:
                    tk = r["taxonKey"]
                    if tk is None:
                        taxon_map[sci] = None
                    else:
                        try:
                            taxon_map[sci] = int(tk)
                        except Exception:
                            taxon_map[sci] = None

                processed += 1
                if processed % max(50000, int(args.progress_every)) == 0:
                    print(
                        f"[info] streamed={processed:,} cells={len(cell_counts):,}",
                        flush=True,
                    )

        print(f"[info] finished stream. cells={len(cell_counts):,}", flush=True)

        # 4) For each cell, keep only the top plants, then collect their newest points.
        cell_top = max(1, int(args.cell_top_plants))

        plants_points: Dict[str, List[Tuple[int, list]]] = {}
        plants_cell_presence: Dict[str, int] = {}
        total_cells_with_anything = len(cell_counts)

        for cell, counts in cell_counts.items():
            # pick top plants by raw count in this cell
            top_plants = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:cell_top]
            if not top_plants:
                continue

            heaps_byplant = cell_heaps.get(cell, {})
            for sci, _cnt in top_plants:
                h = heaps_byplant.get(sci)
                if not h:
                    continue

                plants_cell_presence[sci] = plants_cell_presence.get(sci, 0) + 1

                # heap has newest N but unsorted; keep timekey for later sorting
                out = plants_points.get(sci)
                if out is None:
                    out = []
                    plants_points[sci] = out
                out.extend(h)

        # 5) Build final per-plant objects, sort newest-first, optionally cap max points.
        max_points = int(args.max_points_per_plant)
        plants_out: Dict[str, dict] = {}

        total_points_kept = 0

        for sci in top_species:
            items = plants_points.get(sci, [])
            if not items:
                continue

            # sort newest-first
            items.sort(key=lambda x: x[0], reverse=True)
            if max_points and len(items) > max_points:
                items = items[:max_points]

            points = [pt for (_tk, pt) in items]
            total_points_kept += len(points)

            # year_counts + month_counts derived from kept points
            year_counts: Dict[str, int] = {}
            month_counts_all = [0] * 12
            last_obs = None

            for lat, lon, y, m in points:
                if y is not None:
                    ys = str(int(y))
                    year_counts[ys] = year_counts.get(ys, 0) + 1
                if m is not None:
                    try:
                        mi = int(m)
                    except Exception:
                        mi = None
                    if mi is not None and 1 <= mi <= 12:
                        month_counts_all[mi - 1] += 1

                # compute last_obs from first (newest) point by ordering
            y0 = points[0][2]
            m0 = points[0][3]
            if y0 is not None:
                last_obs = {"year": int(y0), "month": (int(m0) if m0 is not None else None)}

            plants_out[sci] = {
                "de": name_map.get(sci, ""),
                "taxonKey": taxon_map.get(sci),
                # raw total from DB for context + sampling total for UI ranking/local counts
                "total_raw": int(raw_totals.get(sci, 0)),
                "total": int(len(points)),
                "year_counts": year_counts,
                "month_counts_all": month_counts_all,
                "last_obs": last_obs,
                "coverage_cells": int(plants_cell_presence.get(sci, 0)),
                "coverage_cells_total": int(total_cells_with_anything),
                "points": points,
            }

        out = {
            "region": {"name": args.region_name, "center": {"lat": args.region_lat, "lon": args.region_lon}},
            "plants": plants_out,
            "meta": {
                "generated_at": utc_now_iso(),
                "source": os.path.basename(args.db),
                "table": table,
                "country": args.country or None,
                "year_from": args.year_from,
                "year_to": args.year_to,
                "top_n": args.top_n,
                "grid_geohash": args.grid_geohash,
                "cell_top_plants": args.cell_top_plants,
                "newest_per_plant_per_cell": args.newest_per_plant_per_cell,
                "max_points_per_plant": args.max_points_per_plant,
                "plants_out": len(plants_out),
                "points_out": total_points_kept,
            },
        }

        actual = write_output(args.out, out, gzip_enabled=bool(args.gzip))
        dt = time.time() - t_all
        print(f"[info] export done -> {actual} in {dt/60.0:.1f} min", flush=True)
        return 0

    except Exception as e:
        print(f"\n[error] export failed: {e}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
