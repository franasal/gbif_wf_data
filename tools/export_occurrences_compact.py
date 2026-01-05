#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Read-performance pragmas (safe for CI)
    try:
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")  # ~200MB cache
        con.execute("PRAGMA mmap_size=268435456;")  # 256MB
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

    best = None
    best_score = -1
    for t in tables:
        cols = set(table_columns(con, t))
        score = 0
        if any(c in cols for c in ("decimalLatitude", "lat", "latitude")):
            score += 2
        if any(c in cols for c in ("decimalLongitude", "lon", "longitude")):
            score += 2
        if any(c in cols for c in ("scientificName", "species", "speciesName", "taxon_name")):
            score += 2
        if any(c in cols for c in ("year", "eventDate", "event_date")):
            score += 1
        if score > best_score:
            best_score = score
            best = t

    if not best or best_score < 4:
        raise RuntimeError(f"Could not identify occurrence table. Tables: {tables}")
    return best


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


def load_images_index(path: Optional[str]) -> Dict[str, dict]:
    """
    Accepts:
      A) { "plants": { "Urtica dioica": {...}, ... } }
      B) { "Urtica dioica": {...}, ... }

    If file missing or invalid, returns {} (warns).
    """
    if not path:
        return {}

    if not os.path.exists(path):
        print(f"[warn] images-index not found, skipping: {path}", flush=True)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            print(f"[warn] images-index is empty, skipping: {path}", flush=True)
            return {}
        data = json.loads(raw)
    except Exception as e:
        print(f"[warn] images-index could not be read, skipping: {path} ({e})", flush=True)
        return {}

    if isinstance(data, dict) and "plants" in data and isinstance(data["plants"], dict):
        return data["plants"]
    if isinstance(data, dict):
        return data

    print(f"[warn] images-index has unexpected JSON type ({type(data)}), skipping: {path}", flush=True)
    return {}


# --- Geohash for stratification (internal only) ---

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


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


def ym_best(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if a is None:
        return b
    if b is None:
        return a
    return b if b > a else a


def stable_u32(*parts: object) -> int:
    """
    Deterministic 32-bit hash for reservoir sampling.
    Python's built-in hash() changes between runs. This does not.
    """
    s = "\x1f".join("" if p is None else str(p) for p in parts).encode("utf-8", errors="ignore")
    h = hashlib.blake2b(s, digest_size=4).digest()
    return int.from_bytes(h, "big", signed=False)


def ensure_indexes(
    con: sqlite3.Connection,
    table: str,
    col_sci: str,
    col_country: Optional[str],
    col_year: Optional[str],
) -> None:
    """
    Create indexes that match the exporter filters so SQLite stops doing table scans.
    """
    cur = con.cursor()

    # Exact filter column
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col_sci} ON {table}({col_sci})")

    # Composite for common filters
    if col_country and col_year:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_{col_country}_{col_year}_{col_sci} "
            f"ON {table}({col_country},{col_year},{col_sci})"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_{col_country}_{col_year} "
            f"ON {table}({col_country},{col_year})"
        )
    elif col_country:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_{col_country}_{col_sci} "
            f"ON {table}({col_country},{col_sci})"
        )
    elif col_year:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_{col_year}_{col_sci} "
            f"ON {table}({col_year},{col_sci})"
        )

    if col_year:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col_year} ON {table}({col_year})")

    con.commit()
    try:
        cur.execute("ANALYZE")
        con.commit()
    except Exception:
        pass


def write_output(path: str, obj: dict, gzip_enabled: bool) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if gzip_enabled or path.endswith(".gz"):
        if not path.endswith(".gz"):
            path = path + ".gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"\nWrote (gzip): {path}", flush=True)
        return path
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"\nWrote: {path}", flush=True)
        return path


def main():
    ap = argparse.ArgumentParser(description="Export compact occurrences JSON (no bins; stratified points).")
    ap.add_argument("--db", required=True, help="Path to dwca.sqlite")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. data/occurrences_compact.json)")
    ap.add_argument("--country", default="DE", help="Country code filter (default: DE). Empty disables.")
    ap.add_argument("--year-from", type=int, default=None, help="Start year (inclusive)")
    ap.add_argument("--year-to", type=int, default=None, help="End year (inclusive)")
    ap.add_argument("--top-n", type=int, default=50, help="Top N species by TOTAL count (default: 50)")

    ap.add_argument("--points", type=int, default=900, help="Stratified points per plant (default: 900)")
    ap.add_argument("--strata-geohash", type=int, default=5, help="Geohash precision for stratification (default: 5)")
    ap.add_argument("--min-per-cell", type=int, default=1, help="Minimum samples from a cell once selected (default: 1)")
    ap.add_argument("--max-per-cell", type=int, default=30, help="Cap samples per cell (default: 30)")
    ap.add_argument("--chunk-size", type=int, default=20000, help="DB fetch chunk size (default: 20000)")

    ap.add_argument("--progress-every", type=int, default=200000, help="Log progress every N rows per pass (default: 200000)")
    ap.add_argument("--ensure-indexes", action="store_true", help="Create/refresh SQLite indexes for faster export")
    ap.add_argument("--analyze", action="store_true", help="Run ANALYZE (stats) even if not creating indexes")

    ap.add_argument("--region-name", default="Germany (offline)")
    ap.add_argument("--region-lat", type=float, default=51.0)
    ap.add_argument("--region-lon", type=float, default=10.0)

    ap.add_argument("--names-json", default=None, help="Optional: JSON dict scientific->German name")
    ap.add_argument("--images-index", default=None, help="Optional: JSON index with image metadata to attach per plant")
    ap.add_argument("--gzip", action="store_true", help="Write gzipped JSON (.gz)")
    ap.add_argument("--debug", action="store_true", help="Print full traceback on error")

    args = ap.parse_args()

    try:
        t_all = time.time()

        con = connect(args.db)
        table = pick_occurrence_table(con)
        cols = table_columns(con, table)

        # Prefer 'species' first (less author-noise, usually better grouping)
        col_sci = resolve_col(cols, ["species", "scientificName", "speciesName", "taxon_name"])
        col_taxon = resolve_col(cols, ["taxonKey", "taxon_key"], required=False)
        col_lat = resolve_col(cols, ["decimalLatitude", "lat", "latitude"])
        col_lon = resolve_col(cols, ["decimalLongitude", "lon", "longitude"])

        col_year = resolve_col(cols, ["year"], required=False)
        col_month = resolve_col(cols, ["month"], required=False)
        col_country = resolve_col(cols, ["countryCode", "country_code", "country"], required=False)
        col_event = resolve_col(cols, ["eventDate", "event_date"], required=False)

        if args.ensure_indexes:
            print(f"[info] ensuring indexes on table={table} sci_col={col_sci}", flush=True)
            ensure_indexes(con, table=table, col_sci=col_sci, col_country=col_country, col_year=col_year)
        elif args.analyze:
            try:
                print("[info] running ANALYZE", flush=True)
                con.execute("ANALYZE")
            except Exception:
                pass

        name_map = load_name_map(args.names_json)
        images_map = load_images_index(args.images_index)

        # Filters (all queries)
        where = [f"{col_lat} IS NOT NULL", f"{col_lon} IS NOT NULL"]
        params: List[object] = []

        if args.country and col_country:
            where.append(f"{col_country} = ?")
            params.append(args.country)

        if args.year_from is not None or args.year_to is not None:
            if not col_year and not col_event:
                raise RuntimeError("Year filtering requested but neither 'year' nor 'eventDate' exists.")
            if col_year:
                if args.year_from is not None:
                    where.append(f"{col_year} >= ?")
                    params.append(args.year_from)
                if args.year_to is not None:
                    where.append(f"{col_year} <= ?")
                    params.append(args.year_to)
            else:
                if args.year_from is not None:
                    where.append(f"substr({col_event}, 1, 4) >= ?")
                    params.append(str(args.year_from))
                if args.year_to is not None:
                    where.append(f"substr({col_event}, 1, 4) <= ?")
                    params.append(str(args.year_to))

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # Year/month expressions (fallback from eventDate)
        if col_year:
            year_expr = col_year
        elif col_event:
            year_expr = f"CAST(substr({col_event}, 1, 4) AS INTEGER)"
        else:
            year_expr = "NULL"

        if col_month:
            month_expr = col_month
        elif col_event:
            month_expr = f"CAST(substr({col_event}, 6, 2) AS INTEGER)"
        else:
            month_expr = "NULL"

        # Top N species by count
        q_top = f"""
            SELECT {col_sci} AS sci, COUNT(1) AS n
            FROM {table}
            {where_sql}
            GROUP BY {col_sci}
            ORDER BY n DESC
            LIMIT ?
        """
        top_rows = con.execute(q_top, params + [args.top_n]).fetchall()
        top_species = [r["sci"] for r in top_rows if r["sci"]]

        if not top_species:
            raise RuntimeError("No species matched your filters. Check country/year and DB content.")

        print(
            f"[info] table={table} sci_col={col_sci} plants={len(top_species)} points_target={args.points}",
            flush=True,
        )

        plants_out: Dict[str, dict] = {}
        strata_prec = int(args.strata_geohash)
        points_target = int(args.points)
        min_per_cell = max(0, int(args.min_per_cell))
        max_per_cell = max(1, int(args.max_per_cell))
        chunk_size = max(1000, int(args.chunk_size))
        progress_every = max(50000, int(args.progress_every))

        for i, sci in enumerate(top_species, start=1):
            t0 = time.time()

            where2 = list(where)
            params2 = list(params)
            where2.append(f"{col_sci} = ?")
            params2.append(sci)
            where2_sql = "WHERE " + " AND ".join(where2)

            # total
            total = int(con.execute(f"SELECT COUNT(1) AS n FROM {table} {where2_sql}", params2).fetchone()["n"])

            # year_counts (list pairs)
            year_counts: List[List[int]] = []
            if col_year:
                q_years = f"""
                    SELECT {col_year} AS y, COUNT(1) AS n
                    FROM {table}
                    {where2_sql}
                    GROUP BY {col_year}
                    ORDER BY {col_year} ASC
                """
                for r in con.execute(q_years, params2).fetchall():
                    if r["y"] is None:
                        continue
                    year_counts.append([int(r["y"]), int(r["n"])])
            elif col_event:
                q_years = f"""
                    SELECT CAST(substr({col_event}, 1, 4) AS INTEGER) AS y, COUNT(1) AS n
                    FROM {table}
                    {where2_sql}
                    GROUP BY CAST(substr({col_event}, 1, 4) AS INTEGER)
                    ORDER BY y ASC
                """
                for r in con.execute(q_years, params2).fetchall():
                    if r["y"] is None:
                        continue
                    year_counts.append([int(r["y"]), int(r["n"])])

            # month_counts_all (12 ints)
            month_counts_all = [0] * 12
            q_months = f"""
                SELECT {month_expr} AS m, COUNT(1) AS n
                FROM {table}
                {where2_sql}
                GROUP BY {month_expr}
            """
            for r in con.execute(q_months, params2).fetchall():
                m = r["m"]
                if m is None:
                    continue
                try:
                    mi = int(m)
                except Exception:
                    continue
                if 1 <= mi <= 12:
                    month_counts_all[mi - 1] = int(r["n"])

            # last_obs + bbox + stratified points (two-pass)
            bbox = [None, None, None, None]  # minLat, maxLat, minLon, maxLon
            last_obs: Optional[Tuple[int, int]] = None

            q_stream = f"""
                SELECT
                  {col_lat} AS lat,
                  {col_lon} AS lon,
                  {year_expr} AS y,
                  {month_expr} AS m
                FROM {table}
                {where2_sql}
            """

            # Pass 1: cell counts + bbox + last_obs
            cell_counts: Dict[str, int] = {}
            processed1 = 0
            cur = con.execute(q_stream, params2)
            while True:
                chunk = cur.fetchmany(chunk_size)
                if not chunk:
                    break
                for r in chunk:
                    lat = r["lat"]
                    lon = r["lon"]
                    if lat is None or lon is None:
                        continue
                    try:
                        latf = float(lat)
                        lonf = float(lon)
                    except Exception:
                        continue

                    # bbox
                    if bbox[0] is None:
                        bbox = [latf, latf, lonf, lonf]
                    else:
                        bbox[0] = min(bbox[0], latf)
                        bbox[1] = max(bbox[1], latf)
                        bbox[2] = min(bbox[2], lonf)
                        bbox[3] = max(bbox[3], lonf)

                    # last_obs
                    y = r["y"]
                    m = r["m"]
                    try:
                        yi = int(y) if y is not None else None
                    except Exception:
                        yi = None
                    try:
                        mi = int(m) if m is not None else None
                    except Exception:
                        mi = None
                    if mi is not None and not (1 <= mi <= 12):
                        mi = None
                    if yi is not None:
                        last_obs = ym_best(last_obs, (yi, mi or 0))

                    cell = geohash_encode(latf, lonf, precision=strata_prec)
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1

                    processed1 += 1
                    if processed1 % progress_every == 0:
                        elapsed = time.time() - t0
                        print(
                            f"[{i}/{len(top_species)}] {sci}: pass1 processed={processed1:,} cells={len(cell_counts):,} elapsed={elapsed:.1f}s",
                            flush=True,
                        )

            # Determine per-cell quotas (sqrt weighting)
            weights: Dict[str, float] = {c: (n ** 0.5) for c, n in cell_counts.items() if n > 0}

            if not weights:
                points: List[list] = []
            else:
                total_w = sum(weights.values())
                quotas: Dict[str, int] = {}
                for c, w in weights.items():
                    q = int(round(points_target * (w / total_w)))
                    if q > 0:
                        q = max(q, min_per_cell)
                    q = min(q, max_per_cell)
                    quotas[c] = q

                # Fix total to exactly points_target
                current = sum(quotas.values())
                cells_sorted = sorted(quotas.keys(), key=lambda k: weights.get(k, 0.0), reverse=True)

                if current > points_target:
                    over = current - points_target
                    j = 0
                    while over > 0 and cells_sorted:
                        c = cells_sorted[j % len(cells_sorted)]
                        if quotas[c] > min_per_cell:
                            quotas[c] -= 1
                            over -= 1
                        j += 1
                elif current < points_target:
                    under = points_target - current
                    j = 0
                    while under > 0 and cells_sorted:
                        c = cells_sorted[j % len(cells_sorted)]
                        if quotas[c] < max_per_cell:
                            quotas[c] += 1
                            under -= 1
                        j += 1

                # Pass 2: reservoir sampling per cell (deterministic)
                reservoirs: Dict[str, Tuple[int, List[list]]] = {c: (0, []) for c, q in quotas.items() if q > 0}

                processed2 = 0
                cur2 = con.execute(q_stream, params2)
                while True:
                    chunk = cur2.fetchmany(chunk_size)
                    if not chunk:
                        break
                    for r in chunk:
                        lat = r["lat"]
                        lon = r["lon"]
                        if lat is None or lon is None:
                            continue
                        try:
                            latf = float(lat)
                            lonf = float(lon)
                        except Exception:
                            continue

                        cell = geohash_encode(latf, lonf, precision=strata_prec)
                        if cell not in reservoirs:
                            continue

                        seen, res = reservoirs[cell]
                        seen += 1

                        y = r["y"]
                        m = r["m"]
                        try:
                            yi = int(y) if y is not None else None
                        except Exception:
                            yi = None
                        try:
                            mi = int(m) if m is not None else None
                        except Exception:
                            mi = None
                        if mi is not None and not (1 <= mi <= 12):
                            mi = None

                        point = [latf, lonf, yi, mi]
                        k = quotas[cell]

                        if len(res) < k:
                            res.append(point)
                        else:
                            j = stable_u32(sci, cell, seen, latf, lonf, yi, mi) % seen
                            if j < k:
                                res[j] = point

                        reservoirs[cell] = (seen, res)

                        processed2 += 1
                        if processed2 % progress_every == 0:
                            elapsed = time.time() - t0
                            print(
                                f"[{i}/{len(top_species)}] {sci}: pass2 kept={processed2:,} sampled_cells={len(reservoirs):,} elapsed={elapsed:.1f}s",
                                flush=True,
                            )

                points = []
                for _, (_, res) in reservoirs.items():
                    points.extend(res)

                if len(points) > points_target:
                    points = points[:points_target]

            last_obs_obj = None
            if last_obs is not None:
                yy, mm = last_obs
                last_obs_obj = {"year": int(yy), "month": (int(mm) if mm != 0 else None)}

            # taxonKey: take first non-null
            taxon_key_val = None
            if col_taxon:
                q_tk = f"""
                    SELECT {col_taxon} AS tk
                    FROM {table}
                    {where2_sql}
                    AND {col_taxon} IS NOT NULL
                    LIMIT 1
                """
                row = con.execute(q_tk, params2).fetchone()
                if row and row["tk"] is not None:
                    try:
                        taxon_key_val = int(row["tk"])
                    except Exception:
                        taxon_key_val = None

            plant_obj = {
                "de": name_map.get(sci, ""),
                "taxonKey": taxon_key_val,
                "total": total,
                "year_counts": year_counts,
                "month_counts_all": month_counts_all,
                "last_obs": last_obs_obj,
                "bbox": bbox if bbox[0] is not None else None,
                "points": points,
            }

            if sci in images_map:
                plant_obj["image"] = images_map[sci]

            plants_out[sci] = plant_obj

            dt = time.time() - t0
            print(f"[{i}/{len(top_species)}] {sci}: total={total:,} points={len(points):,} took={dt:.1f}s", flush=True)

        out = {
            "meta": {
                "generated_at": utc_now_iso(),
                "source": os.path.basename(args.db),
                "table": table,
                "country": args.country or None,
                "year_from": args.year_from,
                "year_to": args.year_to,
                "top_n": args.top_n,
                "points": args.points,
                "strata_geohash": args.strata_geohash,
                "gzip": bool(args.gzip) or str(args.out).endswith(".gz"),
            },
            "region": {
                "name": args.region_name,
                "center": [args.region_lat, args.region_lon],
            },
            "plants": plants_out,
        }

        actual_path = write_output(args.out, out, gzip_enabled=bool(args.gzip))
        dt_all = time.time() - t_all
        print(f"[info] finished export -> {actual_path} in {dt_all/60.0:.1f} min", flush=True)
        return 0

    except Exception as e:
        print(f"\n[error] export failed: {e}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
