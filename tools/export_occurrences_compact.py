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
from typing import Dict, List, Optional, Tuple

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


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

    ap.add_argument("--region-name", default="Germany")
    ap.add_argument("--region-lat", type=float, default=51.0)
    ap.add_argument("--region-lon", type=float, default=10.0)

    ap.add_argument("--names-json", default=None)
    ap.add_argument("--images-index", default=None)
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
        col_year = resolve_col(cols, ["year"], required=False)
        col_month = resolve_col(cols, ["month"], required=False)
        col_country = resolve_col(cols, ["countryCode", "country_code", "country"], required=False)
        col_dataset_name = resolve_col(cols, ["datasetName"], required=False)
        col_publisher = resolve_col(cols, ["publisher"], required=False)
        col_references = resolve_col(cols, ["referencesUrl", "references"], required=False)

        name_map = load_name_map(args.names_json)
        images_map = load_images_index(args.images_index)

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

        if name_map:
            # Whitelist is authoritative; include all configured plants.
            top_species = list(name_map.keys())

            # Create temp table once (used by stream query too)
            con.execute("DROP TABLE IF EXISTS _wanted_species;")
            con.execute("CREATE TEMP TABLE _wanted_species (sci TEXT PRIMARY KEY);")
            con.executemany("INSERT OR IGNORE INTO _wanted_species(sci) VALUES (?)", [(s,) for s in top_species])
            con.commit()

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
        if col_taxon:
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

        # 2) Prepare output containers
        keep_per_cell = max(1, int(args.keep_per_cell))
        max_points = max(1, int(args.max_points_per_plant))
        prec = max(1, int(args.cell_precision))

        plants_out: Dict[str, dict] = {}
        per_cell_counts: Dict[Tuple[str, str], int] = {}  # (sci, cell) -> kept

        sampled_year_counts: Dict[str, Dict[str, int]] = {s: {} for s in top_species}
        sampled_month_counts: Dict[str, List[int]] = {s: [0] * 12 for s in top_species}
        bbox_map: Dict[str, List[Optional[float]]] = {s: [None, None, None, None] for s in top_species}
        last_obs_map: Dict[str, Optional[Tuple[int, int]]] = {s: None for s in top_species}

        kept_points: Dict[str, List[list]] = {s: [] for s in top_species}
        source_counts: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in top_species}
        done_species = set()

        # 3) Stream newest -> oldest once, keep newest per cell per plant
        year_expr = col_year if col_year else "0"
        month_expr = col_month if col_month else "0"
        available_tables = set(list_tables(con))
        can_join_media = "occ_media" in available_tables and bool(col_gbif_id)

        select_cols = [
            f"o.{col_sci} AS sci",
            f"o.{col_lat} AS lat",
            f"o.{col_lon} AS lon",
            f"o.{year_expr} AS y",
            f"o.{month_expr} AS m",
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

        q_stream = f"""
          SELECT
            {", ".join(select_cols)}
          FROM {table} o
          JOIN _wanted_species w ON w.sci = o.{col_sci}
          {media_join_sql}
          {where_sql}
          ORDER BY o.{year_expr} DESC, o.{month_expr} DESC
        """

        cur = con.execute(q_stream, params)
        scanned = 0

        for r in cur:
            scanned += 1
            if args.progress_every and scanned % int(args.progress_every) == 0:
                print(f"[scan] rows={scanned:,} done_plants={len(done_species)}/{len(top_species)}")

            sci = r["sci"]
            if sci in done_species:
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

            cell = geohash_encode(latf, lonf, precision=prec)
            key = (sci, cell)

            if per_cell_counts.get(key, 0) >= keep_per_cell:
                continue

            pts = kept_points[sci]
            if len(pts) >= max_points:
                done_species.add(sci)
                continue

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
            pts.append(point)
            per_cell_counts[key] = per_cell_counts.get(key, 0) + 1

            if yi is not None:
                ys = str(yi)
                sampled_year_counts[sci][ys] = sampled_year_counts[sci].get(ys, 0) + 1

            if mi is not None:
                sampled_month_counts[sci][mi - 1] += 1

            bb = bbox_map[sci]
            if bb[0] is None:
                bbox_map[sci] = [latf, latf, lonf, lonf]
            else:
                bb[0] = min(bb[0], latf)
                bb[1] = max(bb[1], latf)
                bb[2] = min(bb[2], lonf)
                bb[3] = max(bb[3], lonf)

            if last_obs_map[sci] is None and yi is not None:
                last_obs_map[sci] = (yi, mi or None)

            if len(pts) >= max_points:
                done_species.add(sci)

            if len(done_species) == len(top_species):
                break

        # 4) Build plant objects
        for sci in top_species:
            pts = kept_points[sci]
            yc = sampled_year_counts[sci]

            lo = last_obs_map[sci]
            last_obs_obj = None
            if lo is not None:
                yy, mm = lo
                last_obs_obj = {"year": int(yy), "month": (int(mm) if mm else None)}

            obj = {
                "de": name_map.get(sci, ""),
                "taxonKey": taxon_by_species.get(sci),
                "total": int(true_totals.get(sci, 0)),
                "year_counts": yc,
                "month_counts_all": sampled_month_counts[sci],
                "last_obs": last_obs_obj,
                "bbox": bbox_map[sci] if bbox_map[sci][0] is not None else None,
                "points": pts,
                "sampled_total": len(pts),
                "observation_sources": dict(sorted(source_counts[sci].items())),
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
        return 0

    except Exception as e:
        print(f"\n[error] export failed: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
