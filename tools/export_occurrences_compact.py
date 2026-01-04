#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
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


_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int = 6) -> str:
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


def ym_better(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if a is None:
        return b
    if b is None:
        return a
    return b if b > a else a


def main():
    ap = argparse.ArgumentParser(description="Export occurrences_compact.json with bins + pointsSample.")
    ap.add_argument("--db", required=True, help="Path to dwca.sqlite")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. data/occurrences_compact.json)")
    ap.add_argument("--country", default="DE", help="Country code filter (default: DE). Empty disables.")
    ap.add_argument("--year-from", type=int, default=None, help="Start year (inclusive)")
    ap.add_argument("--year-to", type=int, default=None, help="End year (inclusive)")
    ap.add_argument("--top-n", type=int, default=50, help="Top N species by TOTAL count (default: 50)")
    ap.add_argument("--points-sample", type=int, default=300, help="Most recent points kept for map rendering (default: 300)")
    ap.add_argument("--geohash-precision", type=int, default=6, help="Geohash precision for bins (default: 6)")
    ap.add_argument("--no-bin-month-counts", action="store_true", help="Disable per-bin month_counts (saves space)")
    ap.add_argument("--names-json", default=None, help="Optional: JSON mapping scientific->German name dict")

    ap.add_argument("--region-name", default="Germany (offline)")
    ap.add_argument("--region-lat", type=float, default=51.0)
    ap.add_argument("--region-lon", type=float, default=10.0)

    args = ap.parse_args()

    con = connect(args.db)
    table = pick_occurrence_table(con)
    cols = table_columns(con, table)

    # DB-schema friendly column resolution (your 'occ' table)
    col_sci = resolve_col(cols, ["scientificName", "species", "speciesName", "taxon_name"])
    col_taxon = resolve_col(cols, ["taxonKey", "taxon_key"], required=False)
    col_lat = resolve_col(cols, ["decimalLatitude", "lat", "latitude"])
    col_lon = resolve_col(cols, ["decimalLongitude", "lon", "longitude"])

    col_year = resolve_col(cols, ["year"], required=False)
    col_month = resolve_col(cols, ["month"], required=False)
    col_country = resolve_col(cols, ["countryCode", "country_code", "country"], required=False)
    col_event = resolve_col(cols, ["eventDate", "event_date"], required=False)

    name_map = load_name_map(args.names_json)

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

    where_sql = "WHERE " + " AND ".join(where)

    # year/month expressions (eventDate fallback)
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

    # ordering for "most recent"
    if col_event:
        order_by = f"{col_event} DESC"
    elif col_year and col_month:
        order_by = f"{col_year} DESC, {col_month} DESC"
    elif col_year:
        order_by = f"{col_year} DESC"
    else:
        order_by = None

    # Top N by total
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

    out_plants: Dict[str, dict] = {}
    include_bin_months = not args.no_bin_month_counts
    prec = int(args.geohash_precision)

    for sci in top_species:
        where2 = list(where)
        params2 = list(params)
        where2.append(f"{col_sci} = ?")
        params2.append(sci)
        where2_sql = "WHERE " + " AND ".join(where2)

        # true total
        q_total = f"SELECT COUNT(1) AS n FROM {table} {where2_sql}"
        total = int(con.execute(q_total, params2).fetchone()["n"])

        # year_counts (true)
        year_counts: Dict[str, int] = {}
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
                year_counts[str(int(r["y"]))] = int(r["n"])
        elif col_event:
            q_years = f"""
                SELECT substr({col_event}, 1, 4) AS y, COUNT(1) AS n
                FROM {table}
                {where2_sql}
                GROUP BY substr({col_event}, 1, 4)
                ORDER BY y ASC
            """
            for r in con.execute(q_years, params2).fetchall():
                y = r["y"]
                if not y or y == "None":
                    continue
                year_counts[str(y)] = int(r["n"])

        # pointsSample (map dots)
        select_sample_cols = [
            f"{col_lat} AS lat",
            f"{col_lon} AS lon",
            f"{year_expr} AS year",
            f"{month_expr} AS month",
        ]
        if col_taxon:
            select_sample_cols.append(f"{col_taxon} AS taxonKey")

        q_sample = f"""
            SELECT {", ".join(select_sample_cols)}
            FROM {table}
            {where2_sql}
            {("ORDER BY " + order_by) if order_by else ""}
            LIMIT ?
        """
        sample_rows = con.execute(q_sample, params2 + [args.points_sample]).fetchall()

        points_sample: List[list] = []
        taxon_key_val: Optional[int] = None
        for r in sample_rows:
            lat = r["lat"]
            lon = r["lon"]
            if lat is None or lon is None:
                continue
            try:
                latf = float(lat)
                lonf = float(lon)
            except Exception:
                continue

            y = r["year"]
            m = r["month"]
            try:
                y = int(y) if y is not None else None
            except Exception:
                y = None
            try:
                m = int(m) if m is not None else None
            except Exception:
                m = None

            points_sample.append([latf, lonf, y, m])

            if col_taxon and taxon_key_val is None:
                tk = r["taxonKey"]
                if tk is not None:
                    try:
                        taxon_key_val = int(tk)
                    except Exception:
                        taxon_key_val = None

        # bins + bbox + month_counts_all + last_obs
        bins: Dict[str, dict] = {}
        month_counts_all = [0] * 12
        bbox_min_lat = bbox_max_lat = None
        bbox_min_lon = bbox_max_lon = None
        plant_last_obs: Optional[Tuple[int, int]] = None

        q_stream = f"""
            SELECT
              {col_lat} AS lat,
              {col_lon} AS lon,
              {year_expr} AS year,
              {month_expr} AS month
            FROM {table}
            {where2_sql}
        """
        cur = con.execute(q_stream, params2)
        while True:
            chunk = cur.fetchmany(10000)
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

                y = r["year"]
                m = r["month"]
                try:
                    y = int(y) if y is not None else None
                except Exception:
                    y = None
                try:
                    m = int(m) if m is not None else None
                except Exception:
                    m = None

                # bbox
                if bbox_min_lat is None:
                    bbox_min_lat = bbox_max_lat = latf
                    bbox_min_lon = bbox_max_lon = lonf
                else:
                    bbox_min_lat = min(bbox_min_lat, latf)
                    bbox_max_lat = max(bbox_max_lat, latf)
                    bbox_min_lon = min(bbox_min_lon, lonf)
                    bbox_max_lon = max(bbox_max_lon, lonf)

                # plant month histogram
                if m is not None and 1 <= m <= 12:
                    month_counts_all[m - 1] += 1

                # plant last_obs
                if y is not None:
                    mm = m if (m is not None and 1 <= m <= 12) else 0
                    plant_last_obs = ym_better(plant_last_obs, (y, mm))

                # binning
                cell = geohash_encode(latf, lonf, precision=prec)
                b = bins.get(cell)
                if b is None:
                    b = {
                        "cell": cell,
                        "count": 0,
                        "lat_sum": 0.0,
                        "lon_sum": 0.0,
                        "last_obs": None,
                        "month_counts": [0] * 12 if include_bin_months else None,
                    }
                    bins[cell] = b

                b["count"] += 1
                b["lat_sum"] += latf
                b["lon_sum"] += lonf

                if include_bin_months and m is not None and 1 <= m <= 12:
                    b["month_counts"][m - 1] += 1

                if y is not None:
                    mm = m if (m is not None and 1 <= m <= 12) else 0
                    b["last_obs"] = ym_better(b["last_obs"], (y, mm))

        bins_list = []
        bins_last_obs: Optional[Tuple[int, int]] = None
        for cell, b in bins.items():
            cnt = int(b["count"])
            latc = b["lat_sum"] / cnt
            lonc = b["lon_sum"] / cnt

            lo = b["last_obs"]
            lo_obj = None
            if lo is not None:
                yy, mm = lo
                lo_obj = {"year": int(yy), "month": int(mm) if mm != 0 else None}
                bins_last_obs = ym_better(bins_last_obs, (yy, mm))

            obj = {"cell": cell, "lat": latc, "lon": lonc, "count": cnt}
            if include_bin_months:
                obj["month_counts"] = b["month_counts"]
            if lo_obj is not None:
                obj["last_obs"] = lo_obj

            bins_list.append(obj)

        bins_list.sort(key=lambda x: x["count"], reverse=True)
        if bins_last_obs is not None:
            plant_last_obs = bins_last_obs

        last_obs_obj = None
        if plant_last_obs is not None:
            yy, mm = plant_last_obs
            last_obs_obj = {"year": int(yy), "month": int(mm) if mm != 0 else None}

        bbox = None
        if bbox_min_lat is not None:
            bbox = [bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon]

        out_plants[sci] = {
            "de": name_map.get(sci, ""),
            "taxonKey": taxon_key_val,
            "total": total,
            "last_obs": last_obs_obj,
            "month_counts_all": month_counts_all,
            "year_counts": year_counts,
            "bbox": bbox,
            "bins": bins_list,
            "pointsSample": points_sample,
        }

        print(f"{sci}: total={total:,} bins={len(bins_list):,} pointsSample={len(points_sample):,}")

    out = {
        "meta": {
            "generated_at": utc_now_iso(),
            "source": os.path.basename(args.db),
            "table": table,
            "country": args.country or None,
            "year_from": args.year_from,
            "year_to": args.year_to,
            "top_n": args.top_n,
            "points_sample": args.points_sample,
            "geohash_precision": args.geohash_precision,
            "bin_month_counts": not args.no_bin_month_counts,
            "ordering": "most_recent_first(pointsSample)",
        },
        "region": {
            "name": args.region_name,
            "center": {"lat": args.region_lat, "lon": args.region_lon},
        },
        "plants": out_plants,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
