#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import sqlite3
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
LATEST_POINTS_RESERVED = 10


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
    if isinstance(data, dict):
        out: Dict[str, str] = {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            sci = v.get("scientificName") or v.get("scientific_name")
            de = v.get("de") or v.get("commonName") or v.get("common_name") or ""
            if isinstance(sci, str) and sci.strip():
                out[sci.strip()] = str(de or "").strip()
        if out:
            return out
    if isinstance(data, list):
        out: Dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            sci = item.get("scientificName") or item.get("scientific_name")
            de = item.get("de") or item.get("commonName") or item.get("common_name") or ""
            if isinstance(sci, str) and sci.strip():
                out[sci.strip()] = str(de or "").strip()
        if out:
            return out
    raise RuntimeError(
        "names JSON must be legacy {scientificName: common} or objects containing scientificName (+ optional de/commonName)."
    )


def load_taxon_cache(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[warn] taxon-cache not found, skipping: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[warn] taxon-cache could not be read, skipping: {path} ({e})")
        return {}
    if isinstance(data, dict):
        return data
    print(f"[warn] taxon-cache invalid format, expected object: {path}")
    return {}


def load_images_index(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[warn] images-index not found, skipping: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[warn] images-index could not be read, skipping: {path} ({e})")
        return {}
    if isinstance(data, dict) and "plants" in data and isinstance(data["plants"], dict):
        return data["plants"]
    if isinstance(data, dict):
        return data
    print(f"[warn] images-index unexpected JSON type, skipping: {path}")
    return {}


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


def write_output(path: str, obj: dict, gzip_enabled: bool) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if gzip_enabled or path.endswith(".gz"):
        if not path.endswith(".gz"):
            path = path + ".gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"\nWrote (gzip): {path}")
        return path
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"\nWrote: {path}")
        return path


def _source_label(dataset_name: Optional[str], publisher: Optional[str], references_url: Optional[str]) -> Optional[str]:
    hay = " ".join(
        [
            (dataset_name or ""),
            (publisher or ""),
            (references_url or ""),
        ]
    ).lower()
    if not hay:
        return None
    if "inaturalist" in hay:
        return "iNaturalist"
    if "naturgucker" in hay and "nabu" in hay:
        return "NABU naturgucker"
    if "naturgucker" in hay:
        return "naturgucker"
    if "observation.org" in hay:
        return "Observation.org"
    if "artportalen" in hay:
        return "Artportalen"
    if "gbif.org" in hay:
        return "GBIF"
    return (dataset_name or publisher or None)


def _trim_trailing_nulls(values: List[object]) -> List[object]:
    out = list(values)
    while out and out[-1] is None:
        out.pop()
    return out


def _adaptive_caps(
    true_total: int,
    base_keep_per_cell: int,
    base_max_points: int,
) -> Tuple[int, int, str]:
    # Tuned for map UX: keep more for sparse taxa, cap dense taxa without losing coverage.
    if true_total <= 250:
        return (max(base_keep_per_cell, 8), max(base_max_points, 1200), "rare")
    if true_total <= 1500:
        return (max(base_keep_per_cell, 6), max(base_max_points, 900), "medium")
    if true_total <= 6000:
        return (max(base_keep_per_cell, 4), max(base_max_points, 700), "common")
    return (max(2, min(base_keep_per_cell, 3)), max(400, min(base_max_points, 600)), "very_common")


def _point_dedup_key(
    gbif_id_num: Optional[int],
    latf: float,
    lonf: float,
    yi: Optional[int],
    mi: Optional[int],
    di: Optional[int],
) -> str:
    if gbif_id_num is not None:
        return f"gbif:{gbif_id_num}"
    return f"coord:{latf:.6f}|{lonf:.6f}|{yi or 0}|{mi or 0}|{di or 0}"


def _point_to_obj(
    point: List[object],
    *,
    scientific_name: str,
    common_name: str,
    taxon_key: Optional[int],
    cell: Optional[str] = None,
    drop_reason: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "scientificName": scientific_name,
        "commonName": common_name,
        "taxonKey": taxon_key,
        "lat": point[0] if len(point) > 0 else None,
        "lon": point[1] if len(point) > 1 else None,
        "year": point[2] if len(point) > 2 else None,
        "month": point[3] if len(point) > 3 else None,
        "gbifId": point[4] if len(point) > 4 else None,
        "sourceLabel": point[5] if len(point) > 5 else None,
        "datasetName": point[6] if len(point) > 6 else None,
        "mediaImageUrl": point[7] if len(point) > 7 else None,
        "mediaSourcePageUrl": point[8] if len(point) > 8 else None,
        "occurrenceReferenceUrl": point[9] if len(point) > 9 else None,
        "cell": cell,
        "dropReason": drop_reason,
    }


def main():
    ap = argparse.ArgumentParser(description="Export sparse occurrences JSON: newest N per cell, hotspot-capped.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="DE")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=250)

    ap.add_argument("--cell-precision", type=int, default=5, help="Geohash precision (default 5)")
    ap.add_argument("--keep-per-cell", type=int, default=6, help="Newest points per plant per cell (default 6)")
    ap.add_argument("--max-points-per-plant", type=int, default=700, help="Global cap per plant (default 700)")
    ap.add_argument(
        "--adaptive-sampling",
        action="store_true",
        help="Use adaptive per-plant map sampling caps based on true_total (recommended).",
    )
    ap.add_argument(
        "--recent-out",
        default=None,
        help="Optional second export path with latest raw observations per plant (no geohash sampling).",
    )
    ap.add_argument(
        "--recent-max-points-per-plant",
        type=int,
        default=0,
        help="Max raw latest observations per plant for --recent-out (0 disables recent export).",
    )

    ap.add_argument("--region-name", default="Germany")
    ap.add_argument("--region-lat", type=float, default=51.0)
    ap.add_argument("--region-lon", type=float, default=10.0)

    ap.add_argument("--names-json", default=None)
    ap.add_argument("--taxon-cache", default=None, help="Optional taxon cache JSON for synonym -> acceptedUsageKey.")
    ap.add_argument("--images-index", default=None)
    ap.add_argument(
        "--loss-report-out",
        default=None,
        help="Optional JSON path to write per-run sampling loss diagnostics.",
    )
    ap.add_argument(
        "--loss-report-sample-per-plant",
        type=int,
        default=30,
        help="Max dropped-point samples to retain per plant in --loss-report-out.",
    )
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N rows scanned")

    args = ap.parse_args()

    try:
        con = connect(args.db)
        table = pick_occurrence_table(con)
        cols = table_columns(con, table)

        col_sci = resolve_col(cols, ["species", "scientificName", "speciesName", "taxon_name"])
        col_taxon = resolve_col(cols, ["taxonKey", "taxon_key"], required=False)
        col_lat = resolve_col(cols, ["lat", "decimalLatitude", "latitude"])
        col_lon = resolve_col(cols, ["lon", "decimalLongitude", "longitude"])
        col_gbif_id = resolve_col(cols, ["gbifID", "gbifId", "id"], required=False)
        col_event_date = resolve_col(cols, ["eventDate", "event_date"], required=False)
        col_year = resolve_col(cols, ["year"], required=False)
        col_month = resolve_col(cols, ["month"], required=False)
        col_day = resolve_col(cols, ["day"], required=False)
        col_country = resolve_col(cols, ["countryCode", "country_code", "country"], required=False)
        col_dataset_name = resolve_col(cols, ["datasetName"], required=False)
        col_publisher = resolve_col(cols, ["publisher"], required=False)
        col_references = resolve_col(cols, ["referencesUrl", "references"], required=False)

        name_map = load_name_map(args.names_json)
        images_map = load_images_index(args.images_index)
        taxon_cache = load_taxon_cache(args.taxon_cache)

        preferred_taxon_by_species: Dict[str, Optional[int]] = {}
        if name_map and col_taxon and taxon_cache:
            for sci in name_map.keys():
                entry = taxon_cache.get(sci)
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status") or "").upper()
                usage_key = entry.get("usageKey") or entry.get("acceptedUsageKey")
                accepted_key = entry.get("acceptedUsageKey")
                preferred = accepted_key if status == "SYNONYM" and accepted_key is not None else usage_key
                try:
                    if preferred is not None:
                        preferred_taxon_by_species[sci] = int(preferred)
                except Exception:
                    continue

        # Base WHERE (true totals + scan)
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

        # 1) Choose plants:
        # If names-json is provided, treat it as the authoritative whitelist.
        # Otherwise fall back to “top N by counts”.
        true_totals: Dict[str, int] = {}

        use_taxon_whitelist = False
        if name_map:
            # Whitelist is authoritative; include all configured plants.
            top_species = list(name_map.keys())

            use_taxon_whitelist = bool(col_taxon) and bool(preferred_taxon_by_species)
            if use_taxon_whitelist:
                wanted_taxa: List[Tuple[int, str]] = []
                seen_taxa: set[int] = set()
                for sci in top_species:
                    tk = preferred_taxon_by_species.get(sci)
                    if tk is None:
                        continue
                    if tk in seen_taxa:
                        print(f"[warn] duplicate preferred taxonKey {tk} for {sci}, skipping", file=sys.stderr)
                        continue
                    seen_taxa.add(tk)
                    wanted_taxa.append((tk, sci))
                if wanted_taxa:
                    con.execute("DROP TABLE IF EXISTS _wanted_taxa;")
                    con.execute("CREATE TEMP TABLE _wanted_taxa (tk INTEGER PRIMARY KEY, sci TEXT);")
                    con.executemany("INSERT OR IGNORE INTO _wanted_taxa(tk, sci) VALUES (?, ?)", wanted_taxa)
                    con.commit()
                else:
                    use_taxon_whitelist = False

            if not use_taxon_whitelist:
                # Create temp table once (used by stream query too)
                con.execute("DROP TABLE IF EXISTS _wanted_species;")
                con.execute("CREATE TEMP TABLE _wanted_species (sci TEXT PRIMARY KEY);")
                con.executemany("INSERT OR IGNORE INTO _wanted_species(sci) VALUES (?)", [(s,) for s in top_species])
                con.commit()

            if use_taxon_whitelist:
                q_totals = f"""
                  SELECT w.sci AS sci, COUNT(1) AS n
                  FROM {table} o
                  JOIN _wanted_taxa w ON w.tk = o.{col_taxon}
                  {where_sql}
                  GROUP BY w.sci
                """
            else:
                q_totals = f"""
                  SELECT o.{col_sci} AS sci, COUNT(1) AS n
                  FROM {table} o
                  JOIN _wanted_species w ON w.sci = o.{col_sci}
                  {where_sql}
                  GROUP BY o.{col_sci}
                """
            rows = con.execute(q_totals, params).fetchall()
            true_totals = {r["sci"]: int(r["n"]) for r in rows if r["sci"]}

        else:
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
            true_totals = {r["sci"]: int(r["n"]) for r in top_rows if r["sci"]}

            # Create temp table for stream query
            con.execute("DROP TABLE IF EXISTS _wanted_species;")
            con.execute("CREATE TEMP TABLE _wanted_species (sci TEXT PRIMARY KEY);")
            con.executemany("INSERT OR IGNORE INTO _wanted_species(sci) VALUES (?)", [(s,) for s in top_species])
            con.commit()

        if not top_species:
            raise RuntimeError("No species matched filters.")

        # TaxonKey dict (best effort, always defined)
        taxon_by_species: Dict[str, Optional[int]] = {s: None for s in top_species}
        if use_taxon_whitelist and preferred_taxon_by_species:
            for sci in top_species:
                taxon_by_species[sci] = preferred_taxon_by_species.get(sci)
        elif col_taxon:
            q_taxon = f"""
              SELECT o.{col_sci} AS sci, MAX(o.{col_taxon}) AS tk
              FROM {table} o
              JOIN _wanted_species w ON w.sci = o.{col_sci}
              {where_sql} AND o.{col_taxon} IS NOT NULL
              GROUP BY o.{col_sci}
            """
            for row in con.execute(q_taxon, params):
                if row["sci"] is None or row["tk"] is None:
                    continue
                try:
                    taxon_by_species[str(row["sci"])] = int(row["tk"])
                except Exception:
                    taxon_by_species[str(row["sci"])] = None

        # 2) Precompute true month counts (all filtered rows, not sampled points)
        true_month_counts: Dict[str, List[int]] = {s: [0] * 12 for s in top_species}
        if col_month:
            if use_taxon_whitelist:
                q_months = f"""
                  SELECT w.sci AS sci, o.{col_month} AS m, COUNT(1) AS n
                  FROM {table} o
                  JOIN _wanted_taxa w ON w.tk = o.{col_taxon}
                  {where_sql} AND o.{col_month} BETWEEN 1 AND 12
                  GROUP BY w.sci, o.{col_month}
                """
            else:
                q_months = f"""
                  SELECT o.{col_sci} AS sci, o.{col_month} AS m, COUNT(1) AS n
                  FROM {table} o
                  JOIN _wanted_species w ON w.sci = o.{col_sci}
                  {where_sql} AND o.{col_month} BETWEEN 1 AND 12
                  GROUP BY o.{col_sci}, o.{col_month}
                """
            for row in con.execute(q_months, params):
                sci = row["sci"]
                if not sci:
                    continue
                try:
                    mi = int(row["m"])
                    n = int(row["n"])
                except Exception:
                    continue
                if 1 <= mi <= 12 and sci in true_month_counts:
                    true_month_counts[sci][mi - 1] = n

        # 3) Prepare output containers
        keep_per_cell = max(1, int(args.keep_per_cell))
        max_points = max(1, int(args.max_points_per_plant))
        prec = max(1, int(args.cell_precision))
        recent_max_points = max(0, int(args.recent_max_points_per_plant or 0))
        write_recent = bool(args.recent_out and recent_max_points > 0)

        plants_out: Dict[str, dict] = {}
        per_cell_counts: Dict[Tuple[str, str], int] = {}  # (sci, cell) -> kept
        per_plant_keep_per_cell: Dict[str, int] = {}
        per_plant_max_points: Dict[str, int] = {}
        sampling_mode_by_species: Dict[str, str] = {}
        for sci in top_species:
            if args.adaptive_sampling:
                kp, mp, bucket = _adaptive_caps(int(true_totals.get(sci, 0)), keep_per_cell, max_points)
            else:
                kp, mp, bucket = keep_per_cell, max_points, "fixed"
            per_plant_keep_per_cell[sci] = kp
            per_plant_max_points[sci] = mp
            sampling_mode_by_species[sci] = bucket

        latest_reserved = int(LATEST_POINTS_RESERVED)
        if max_points < latest_reserved:
            raise RuntimeError(
                f"max_points_per_plant ({max_points}) must be >= latest reserved points ({latest_reserved})"
            )

        sampled_points: Dict[str, List[list]] = {s: [] for s in top_species}
        latest_points: Dict[str, List[list]] = {s: [] for s in top_species}
        latest_seen_keys: Dict[str, set[str]] = {s: set() for s in top_species}
        sampled_seen_keys: Dict[str, set[str]] = {s: set() for s in top_species}
        raw_cells_by_species: Dict[str, set[str]] = {s: set() for s in top_species}
        source_counts: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in top_species}
        loss_sample_limit = max(0, int(args.loss_report_sample_per_plant or 0))
        loss_counts: Dict[str, Dict[str, int]] = {
            s: {
                "kept_latest_reserved": 0,
                "kept_geohash_sample": 0,
                "dropped_duplicate": 0,
                "dropped_cell_quota": 0,
                "dropped_global_cap": 0,
            }
            for s in top_species
        }
        dropped_point_samples: Dict[str, List[dict[str, object]]] = {s: [] for s in top_species}
        recent_points: Dict[str, List[list]] = {s: [] for s in top_species} if write_recent else {}
        done_species = set()
        recent_done_species = set()

        # 4) Stream newest -> oldest once, keep newest per cell per plant
        year_expr = col_year if col_year else "0"
        month_expr = col_month if col_month else "0"
        day_expr = col_day if col_day else "0"
        available_tables = set(list_tables(con))
        can_join_media = "occ_media" in available_tables and bool(col_gbif_id)

        sci_select = f"o.{col_sci} AS sci"
        if use_taxon_whitelist:
            sci_select = "w.sci AS sci"
        select_cols = [
            sci_select,
            f"o.{col_lat} AS lat",
            f"o.{col_lon} AS lon",
            (f"o.{col_event_date} AS event_date" if col_event_date else "NULL AS event_date"),
            f"o.{year_expr} AS y",
            f"o.{month_expr} AS m",
            f"o.{day_expr} AS d",
        ]
        if col_gbif_id:
            select_cols.append(f"o.{col_gbif_id} AS gbif_id")
        if col_dataset_name:
            select_cols.append(f"o.{col_dataset_name} AS dataset_name")
        if col_publisher:
            select_cols.append(f"o.{col_publisher} AS publisher")
        if col_references:
            select_cols.append(f"o.{col_references} AS refs_url")
        if can_join_media:
            select_cols.append("m.imageUrl AS media_image_url")
            select_cols.append("m.sourcePageUrl AS media_source_page")
        else:
            select_cols.append("NULL AS media_image_url")
            select_cols.append("NULL AS media_source_page")

        media_join_sql = f"LEFT JOIN occ_media m ON m.gbifID = o.{col_gbif_id}" if can_join_media else ""

        order_parts = []
        if col_event_date:
            order_parts.extend([f"(o.{col_event_date} IS NULL) ASC", f"o.{col_event_date} DESC"])
        if col_year:
            order_parts.append(f"o.{col_year} DESC")
        if col_month:
            order_parts.append(f"o.{col_month} DESC")
        if col_day:
            order_parts.append(f"o.{col_day} DESC")
        if not order_parts:
            order_parts = [f"o.{col_sci} ASC"]
        join_sql = f"JOIN _wanted_species w ON w.sci = o.{col_sci}"
        if use_taxon_whitelist:
            join_sql = f"JOIN _wanted_taxa w ON w.tk = o.{col_taxon}"
        q_stream = f"""
          SELECT
            {", ".join(select_cols)}
          FROM {table} o
          {join_sql}
          {media_join_sql}
          {where_sql}
          ORDER BY {", ".join(order_parts)}
        """

        cur = con.execute(q_stream, params)
        scanned = 0

        for r in cur:
            scanned += 1
            if args.progress_every and scanned % int(args.progress_every) == 0:
                print(f"[scan] rows={scanned:,} done_plants={len(done_species)}/{len(top_species)}")

            sci = r["sci"]
            if not sci or sci not in loss_counts:
                continue

            lat = r["lat"]
            lon = r["lon"]
            if lat is None or lon is None:
                continue
            try:
                latf = float(lat)
                lonf = float(lon)
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
            if mi is not None and not (1 <= mi <= 12):
                mi = None
            try:
                di = int(d) if d is not None else None
            except Exception:
                di = None
            if di is not None and not (1 <= di <= 31):
                di = None

            cell = geohash_encode(latf, lonf, precision=prec)
            key = (sci, cell)
            raw_cells_by_species[sci].add(cell)

            gbif_id_val = r["gbif_id"] if col_gbif_id else None
            try:
                gbif_id_num = int(gbif_id_val) if gbif_id_val is not None else None
            except Exception:
                gbif_id_num = None

            dataset_name = r["dataset_name"] if col_dataset_name else None
            publisher = r["publisher"] if col_publisher else None
            refs_url = r["refs_url"] if col_references else None
            media_image_url = r["media_image_url"]
            media_source_page = r["media_source_page"]
            source_label = _source_label(
                str(dataset_name) if dataset_name is not None else None,
                str(publisher) if publisher is not None else None,
                str(refs_url) if refs_url is not None else None,
            )
            if source_label:
                source_counts[sci][source_label] += 1

            point = _trim_trailing_nulls(
                [
                    latf,
                    lonf,
                    yi,
                    mi,
                    gbif_id_num,
                    source_label,
                    (str(dataset_name) if dataset_name else None),
                    (str(media_image_url) if media_image_url else None),
                    (str(media_source_page) if media_source_page else None),
                    (str(refs_url) if refs_url else None),
                ]
            )
            row_key = _point_dedup_key(gbif_id_num, latf, lonf, yi, mi, di)
            common_name = name_map.get(sci, "")
            taxon_key = taxon_by_species.get(sci)

            def mark_drop(reason: str) -> None:
                loss_counts[sci][reason] += 1
                if loss_sample_limit <= 0:
                    return
                samples = dropped_point_samples[sci]
                if len(samples) >= loss_sample_limit:
                    return
                samples.append(
                    _point_to_obj(
                        point,
                        scientific_name=sci,
                        common_name=common_name,
                        taxon_key=taxon_key,
                        cell=cell,
                        drop_reason=reason,
                    )
                )

            # Always reserve true newest points first; these must survive map sampling caps.
            latest_for_sci = latest_points[sci]
            latest_keys_for_sci = latest_seen_keys[sci]
            sampled_keys_for_sci = sampled_seen_keys[sci]
            already_seen = row_key in latest_keys_for_sci or row_key in sampled_keys_for_sci
            was_reserved = False
            if len(latest_for_sci) < latest_reserved and not already_seen:
                latest_for_sci.append(point)
                latest_keys_for_sci.add(row_key)
                loss_counts[sci]["kept_latest_reserved"] += 1
                was_reserved = True

            # Optional "recent raw" export: newest observations per plant, no geohash sampling.
            if write_recent and sci not in recent_done_species:
                rp = recent_points[sci]
                if len(rp) < recent_max_points:
                    rp.append(point)
                if len(rp) >= recent_max_points:
                    recent_done_species.add(sci)

            # Core compact export: adaptive/fixed geohash sampling.
            if sci not in done_species:
                sci_keep_per_cell = per_plant_keep_per_cell[sci]
                sci_max_points = per_plant_max_points[sci]
                sci_sampling_budget = max(0, sci_max_points - len(latest_points[sci]))
                if was_reserved:
                    if len(sampled_points[sci]) >= sci_sampling_budget and len(latest_points[sci]) >= latest_reserved:
                        done_species.add(sci)
                elif already_seen:
                    mark_drop("dropped_duplicate")
                elif sci_sampling_budget <= 0 and len(latest_points[sci]) >= latest_reserved:
                    mark_drop("dropped_global_cap")
                    done_species.add(sci)
                elif per_cell_counts.get(key, 0) < sci_keep_per_cell:
                    pts = sampled_points[sci]
                    if len(pts) < sci_sampling_budget:
                        pts.append(point)
                        sampled_keys_for_sci.add(row_key)
                        per_cell_counts[key] = per_cell_counts.get(key, 0) + 1
                        loss_counts[sci]["kept_geohash_sample"] += 1
                    else:
                        mark_drop("dropped_global_cap")

                    if len(sampled_points[sci]) >= sci_sampling_budget and len(latest_points[sci]) >= latest_reserved:
                        done_species.add(sci)
                else:
                    mark_drop("dropped_cell_quota")
            elif not was_reserved and already_seen:
                mark_drop("dropped_duplicate")
            elif not was_reserved:
                mark_drop("dropped_global_cap")

        # 5) Build plant objects
        for sci in top_species:
            reserved = latest_points[sci]
            sampled = sampled_points[sci]
            pts = reserved + sampled
            sci_max_points = per_plant_max_points[sci]
            if len(pts) > sci_max_points:
                pts = pts[:sci_max_points]

            yc: Dict[str, int] = {}
            bb: List[Optional[float]] = [None, None, None, None]
            lo: Optional[Tuple[int, int]] = None
            for p in pts:
                yi = p[2] if len(p) > 2 else None
                mi = p[3] if len(p) > 3 else None
                latv = p[0] if len(p) > 0 else None
                lonv = p[1] if len(p) > 1 else None

                try:
                    if yi is not None:
                        ys = str(int(yi))
                        yc[ys] = yc.get(ys, 0) + 1
                except Exception:
                    pass

                try:
                    latf = float(latv)
                    lonf = float(lonv)
                    if bb[0] is None:
                        bb = [latf, latf, lonf, lonf]
                    else:
                        bb[0] = min(bb[0], latf)
                        bb[1] = max(bb[1], latf)
                        bb[2] = min(bb[2], lonf)
                        bb[3] = max(bb[3], lonf)
                except Exception:
                    pass

                if lo is None:
                    try:
                        if yi is not None:
                            lo = (int(yi), int(mi) if mi is not None else None)
                    except Exception:
                        pass

            last_obs_obj = None
            if lo is not None:
                yy, mm = lo
                last_obs_obj = {"year": int(yy), "month": (int(mm) if mm else None)}

            obj_total = int(true_totals.get(sci, 0))
            month_total = int(sum(true_month_counts[sci]))
            shipped_cells = {
                geohash_encode(float(p[0]), float(p[1]), precision=prec)
                for p in pts
                if len(p) > 1
            }
            raw_occupied_cells = len(raw_cells_by_species.get(sci, set()))
            shipped_occupied_cells = len(shipped_cells)

            obj = {
                "de": name_map.get(sci, ""),
                "taxonKey": taxon_by_species.get(sci),
                "total": obj_total,
                "year_counts": yc,
                # Seasonality must reflect all filtered GBIF rows, not sampled map points.
                "month_counts_all": true_month_counts[sci],
                "month_counts_all_total": month_total,
                "month_counts_all_coverage": (
                    round(month_total / obj_total, 6) if obj_total > 0 else None
                ),
                "last_obs": last_obs_obj,
                "bbox": bb if bb[0] is not None else None,
                "points": pts,
                "sampled_total": len(pts),
                "sampling": {
                    "mode": ("adaptive" if args.adaptive_sampling else "fixed"),
                    "bucket": sampling_mode_by_species.get(sci, "fixed"),
                    "keep_per_cell": per_plant_keep_per_cell.get(sci, keep_per_cell),
                    "max_points_per_plant": per_plant_max_points.get(sci, max_points),
                    "latest_reserved": latest_reserved,
                    "latest_injected_count": min(latest_reserved, len(reserved)),
                    "ratio": (round(len(pts) / obj_total, 6) if obj_total > 0 else None),
                    "grid_precision": prec,
                    "raw_occupied_cells": raw_occupied_cells,
                    "shipped_occupied_cells": shipped_occupied_cells,
                    "grid_coverage": (
                        round(shipped_occupied_cells / raw_occupied_cells, 6)
                        if raw_occupied_cells > 0
                        else None
                    ),
                },
                "observation_sources": dict(sorted(source_counts[sci].items())),
                "loss_counts": dict(loss_counts[sci]),
            }

            if sci in images_map:
                obj["image"] = images_map[sci]

            plants_out[sci] = obj
            print(f"{sci}: true_total={obj['total']:,} sampled_points={len(pts):,}")

        out = {
            "region": {"name": args.region_name, "center": {"lat": args.region_lat, "lon": args.region_lon}},
            "plants": plants_out,
            "meta": {
                "generated_at": utc_now_iso(),
                "source": os.path.basename(args.db),
                "country": args.country or None,
                "year_from": args.year_from,
                "year_to": args.year_to,
                "top_n": args.top_n,
                "cell_precision": prec,
                "keep_per_cell": keep_per_cell,
                "max_points_per_plant": max_points,
                "adaptive_sampling": bool(args.adaptive_sampling),
                "latest_points_reserved": latest_reserved,
                "scanned_rows": scanned,
                "points_schema": [
                    "lat",
                    "lon",
                    "year",
                    "month",
                    "gbifID",
                    "source_label",
                    "dataset_name",
                    "media_image_url",
                    "media_source_page_url",
                    "occurrence_reference_url",
                ],
            },
        }

        write_output(args.out, out, gzip_enabled=bool(args.gzip))
        if args.loss_report_out:
            plants_loss: List[Dict[str, Any]] = []
            totals = {
                "plants": len(top_species),
                "raw_points_visible_window": 0,
                "kept_latest_reserved": 0,
                "kept_geohash_sample": 0,
                "shipped_points": 0,
                "dropped_duplicate": 0,
                "dropped_cell_quota": 0,
                "dropped_global_cap": 0,
                "dropped_total": 0,
            }
            for sci in top_species:
                counts = dict(loss_counts[sci])
                raw_total = int(true_totals.get(sci, 0))
                shipped_total = counts["kept_latest_reserved"] + counts["kept_geohash_sample"]
                dropped_total = max(0, raw_total - shipped_total)
                raw_occupied_cells = len(raw_cells_by_species.get(sci, set()))
                shipped_occupied_cells = len(
                    {
                        geohash_encode(float(p[0]), float(p[1]), precision=prec)
                        for p in (latest_points[sci] + sampled_points[sci])
                        if len(p) > 1
                    }
                )
                plant_loss = {
                    "scientificName": sci,
                    "commonName": name_map.get(sci, ""),
                    "taxonKey": taxon_by_species.get(sci),
                    "rawTotal": raw_total,
                    "shippedTotal": shipped_total,
                    "droppedTotal": dropped_total,
                    "rawOccupiedCells": raw_occupied_cells,
                    "shippedOccupiedCells": shipped_occupied_cells,
                    "droppedOccupiedCells": max(0, raw_occupied_cells - shipped_occupied_cells),
                    "gridCoverage": (
                        round(shipped_occupied_cells / raw_occupied_cells, 6)
                        if raw_occupied_cells > 0
                        else None
                    ),
                    "dropReasonCounts": counts,
                    "sampling": {
                        "mode": ("adaptive" if args.adaptive_sampling else "fixed"),
                        "bucket": sampling_mode_by_species.get(sci, "fixed"),
                        "keep_per_cell": per_plant_keep_per_cell.get(sci, keep_per_cell),
                        "max_points_per_plant": per_plant_max_points.get(sci, max_points),
                        "latest_reserved": latest_reserved,
                        "grid_precision": prec,
                    },
                    "droppedPointsSample": dropped_point_samples[sci],
                }
                plants_loss.append(plant_loss)
                totals["raw_points_visible_window"] += raw_total
                totals["kept_latest_reserved"] += counts["kept_latest_reserved"]
                totals["kept_geohash_sample"] += counts["kept_geohash_sample"]
                totals["shipped_points"] += shipped_total
                totals["dropped_duplicate"] += counts["dropped_duplicate"]
                totals["dropped_cell_quota"] += counts["dropped_cell_quota"]
                totals["dropped_global_cap"] += counts["dropped_global_cap"]
                totals["dropped_total"] += dropped_total
            loss_out = {
                "generatedAt": utc_now_iso(),
                "source": os.path.basename(args.db),
                "window": {
                    "country": args.country or None,
                    "year_from": args.year_from,
                    "year_to": args.year_to,
                },
                "totals": totals,
                "plants": plants_loss,
            }
            write_output(args.loss_report_out, loss_out, gzip_enabled=False)
        if write_recent:
            recent_plants_out: Dict[str, dict] = {}
            for sci in top_species:
                recent_obj = {
                    "de": name_map.get(sci, ""),
                    "taxonKey": taxon_by_species.get(sci),
                    "total": int(true_totals.get(sci, 0)),
                    "points": recent_points.get(sci, []),
                    "sampled_total": len(recent_points.get(sci, [])),
                }
                if sci in images_map:
                    recent_obj["image"] = images_map[sci]
                recent_plants_out[sci] = recent_obj
            recent_out = {
                "region": {"name": args.region_name, "center": {"lat": args.region_lat, "lon": args.region_lon}},
                "plants": recent_plants_out,
                "meta": {
                    "generated_at": utc_now_iso(),
                    "source": os.path.basename(args.db),
                    "country": args.country or None,
                    "year_from": args.year_from,
                    "year_to": args.year_to,
                    "top_n": args.top_n,
                    "export_kind": "recent_raw_points",
                    "recent_max_points_per_plant": recent_max_points,
                    "ordering": "eventDate desc, year desc, month desc, day desc",
                    "points_schema": out["meta"]["points_schema"],
                },
            }
            write_output(args.recent_out, recent_out, gzip_enabled=bool(args.gzip))
        return 0

    except Exception as e:
        print(f"\n[error] export failed: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
