#!/usr/bin/env python3
import argparse
import os
import csv
import sqlite3

PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"
]

def find_occurrence(dwca_dir: str) -> str:
    p = os.path.join(dwca_dir, "occurrence.txt")
    if os.path.exists(p):
        return p
    txts = [os.path.join(dwca_dir, f) for f in os.listdir(dwca_dir) if f.lower().endswith(".txt")]
    if not txts:
        raise FileNotFoundError(f"No .txt found in {dwca_dir}")
    txts.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return txts[0]

def init_db(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS occ (
      gbifID TEXT PRIMARY KEY,
      scientificName TEXT,
      species TEXT,
      eventDate TEXT,
      year INTEGER,
      month INTEGER,
      day INTEGER,
      countryCode TEXT,
      basisOfRecord TEXT,
      datasetKey TEXT,
      license TEXT,
      lat REAL,
      lon REAL,
      raw_tsv TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_latlon ON occ(lat, lon);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_date ON occ(year, month);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_species ON occ(species, scientificName);")
    con.commit()
    return con

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def set_meta(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))
    con.commit()

def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default

def load(dwca_dir: str, db_path: str, limit: int = 0, commit_every: int = 2000, keep_raw: bool = True):
    occ_path = find_occurrence(dwca_dir)
    print("Using:", occ_path)

    con = init_db(db_path)
    cur = con.cursor()

    with open(occ_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        print("Columns in file:", len(cols))

        set_meta(con, "occurrence_file", occ_path)
        set_meta(con, "columns", ",".join(cols))

        n = 0
        batch = []

        for row in reader:
            gbifID = row.get("gbifID") or row.get("id") or row.get("occurrenceID")
            if not gbifID:
                continue

            scientificName = row.get("scientificName")
            species = row.get("species") or row.get("acceptedScientificName") or row.get("canonicalName")
            eventDate = row.get("eventDate")
            year = to_int(row.get("year"))
            month = to_int(row.get("month"))
            day = to_int(row.get("day"))
            countryCode = row.get("countryCode")
            basisOfRecord = row.get("basisOfRecord")
            datasetKey = row.get("datasetKey")
            license_ = row.get("license")
            lat = to_float(row.get("decimalLatitude"))
            lon = to_float(row.get("decimalLongitude"))

            raw_tsv = ""
            if keep_raw:
                raw_tsv = "\t".join([row.get(c, "") or "" for c in cols])

            batch.append((
                gbifID, scientificName, species, eventDate, year, month, day,
                countryCode, basisOfRecord, datasetKey, license_, lat, lon, raw_tsv
            ))

            n += 1
            if limit and n >= limit:
                break

            if len(batch) >= commit_every:
                cur.executemany("""
                INSERT OR REPLACE INTO occ
                (gbifID, scientificName, species, eventDate, year, month, day, countryCode, basisOfRecord, datasetKey, license, lat, lon, raw_tsv)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                con.commit()
                batch = []
                print("Loaded:", n)

        if batch:
            cur.executemany("""
            INSERT OR REPLACE INTO occ
            (gbifID, scientificName, species, eventDate, year, month, day, countryCode, basisOfRecord, datasetKey, license, lat, lon, raw_tsv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            con.commit()

    con.close()
    print("Done. Rows:", n, "DB:", db_path)

def count_rows(db_path: str):
    con = sqlite3.connect(db_path)
    n = con.execute("SELECT COUNT(*) FROM occ").fetchone()[0]
    con.close()
    print("rows:", n)

def columns(db_path: str):
    con = sqlite3.connect(db_path)
    cols = get_meta(con, "columns", "")
    con.close()
    if cols:
        for c in cols.split(","):
            print(c)
    else:
        print("No column metadata stored (did you run load?).")

def top_values(db_path: str, col: str, n: int = 20, where: str = ""):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    where_sql = f"WHERE {where}" if where else ""
    q = f"""
    SELECT {col} AS value, COUNT(*) AS count
    FROM occ
    {where_sql}
    GROUP BY 1
    ORDER BY count DESC
    LIMIT ?
    """
    rows = cur.execute(q, (n,)).fetchall()
    con.close()
    for v, c in rows:
        print(f"{v}\t{c}")

def sql(db_path: str, q: str, limit: int = 50):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    q2 = q.strip().rstrip(";")
    if q2.lower().startswith("select") and " limit " not in q2.lower():
        q2 += f" LIMIT {limit}"
    rows = cur.execute(q2).fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    if cols:
        print("\t".join(cols))
    for r in rows:
        print("\t".join("" if x is None else str(x) for x in r))
    con.close()

def _sp_expr():
    return "COALESCE(species, scientificName)"

def _stable_color_map(species_list):
    uniq = sorted(dict.fromkeys([s for s in species_list if s]))
    cmap = {}
    for i, s in enumerate(uniq):
        cmap[s] = PALETTE[i % len(PALETTE)]
    return cmap

def _build_filters(country=None, year_from=None, year_to=None, bbox=None):
    where = ["lat IS NOT NULL", "lon IS NOT NULL"]
    params = []

    if country:
        where.append("countryCode = ?")
        params.append(country)

    if year_from is not None and year_to is not None:
        where.append("year BETWEEN ? AND ?")
        params.extend([year_from, year_to])

    # bbox format: "minLon,minLat,maxLon,maxLat"
    if bbox:
        minLon, minLat, maxLon, maxLat = bbox
        where.append("lon BETWEEN ? AND ?")
        params.extend([minLon, maxLon])
        where.append("lat BETWEEN ? AND ?")
        params.extend([minLat, maxLat])

    return " AND ".join(where), params

def export_map_top_html(db_path: str, out_html: str, top_n: int, limit_points: int,
                        lat0: float, lon0: float, zoom: int,
                        year_from=None, year_to=None, country=None, bbox=None,
                        radius=2):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    sp_expr = _sp_expr()
    where_sql, base_params = _build_filters(country=country, year_from=year_from, year_to=year_to, bbox=bbox)

    # Select top N species by count under the current filters
    q_top = f"""
    SELECT {sp_expr} AS sp, COUNT(*) AS c
    FROM occ
    WHERE {where_sql}
    GROUP BY 1
    ORDER BY c DESC
    LIMIT ?
    """
    top_rows = cur.execute(q_top, base_params + [top_n]).fetchall()
    species_list = [r[0] for r in top_rows if r[0]]

    if not species_list:
        con.close()
        raise SystemExit("No species found under the selected filters. Try loosening filters.")

    cmap = _stable_color_map(species_list)

    # Split points budget across species so one dominant species doesn't hog it all
    per_species_cap = max(50, limit_points // max(1, len(species_list)))

    all_rows = []
    for sp in species_list:
        q_pts = f"""
        SELECT lat, lon, {sp_expr} AS sp, gbifID, year, month, eventDate
        FROM occ
        WHERE {where_sql}
          AND {sp_expr} = ?
        LIMIT ?
        """
        rows = cur.execute(q_pts, base_params + [sp, per_species_cap]).fetchall()
        all_rows.extend(rows)

    con.close()

    legend_items = "\n".join([
        f"<div><span style='display:inline-block;width:12px;height:12px;background:{cmap[s]};border-radius:50%;margin-right:6px;'></span>{s}</div>"
        for s in sorted(cmap.keys())
    ])

    pts_js = []
    for lat, lon, sp, gid, y, m, ev in all_rows:
        color = cmap.get(sp, "#000000")
        sp_s = (sp or "").replace("\\", "\\\\").replace("'", "\\'")
        gid_s = "" if gid is None else str(gid).replace("\\", "\\\\").replace("'", "\\'")
        ev_s = "" if ev is None else str(ev).replace("\\", "\\\\").replace("'", "\\'")
        pts_js.append(f"[{lat}, {lon}, '{sp_s}', '{color}', '{gid_s}', '{ev_s}']")

    subtitle_bits = []
    if country:
        subtitle_bits.append(f"country={country}")
    if year_from is not None and year_to is not None:
        subtitle_bits.append(f"years={year_from}-{year_to}")
    if bbox:
        subtitle_bits.append("bbox=" + ",".join(str(x) for x in bbox))
    subtitle = " | ".join(subtitle_bits) if subtitle_bits else "no filters"

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DWCA top species map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html, body, #map {{ height: 100%; margin: 0; }}
  .legend {{
  position: fixed;
  bottom: 12px;
  left: 12px;
  background: white;
  padding: 10px;
  border-radius: 8px;
  font-family: sans-serif;
  font-size: 13px;
  max-height: 45vh;
  overflow: auto;
  box-shadow: 0 2px 10px rgba(0,0,0,0.15);

  z-index: 9999;          /* actually stay on top */
  pointer-events: auto;    /* clickable/scrollable */
}}
  .title {{ font-weight: 700; margin-bottom: 6px; }}
  .sub {{ font-size: 12px; opacity: 0.8; margin-bottom: 8px; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">
  <div class="title">Top {len(species_list)} species</div>
  <div class="sub">{subtitle}</div>
  {legend_items}
  <div style="margin-top:8px;"><b>Points:</b> {len(all_rows)} (cap ~{per_species_cap}/species)</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([{lat0}, {lon0}], {zoom});
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  const pts = [{",".join(pts_js)}];
  for (const p of pts) {{
    const lat = p[0], lon = p[1], sp = p[2], color = p[3], gid = p[4], ev = p[5];
    L.circleMarker([lat, lon], {{
      radius: {radius},
      weight: 0,
      fillOpacity: 0.65,
      color: color,
      fillColor: color
    }}).bindPopup(`<b>${{sp}}</b><br/>gbifID=${{gid}}<br/>eventDate=${{ev}}<br/>lat=${{lat}} lon=${{lon}}`).addTo(map);
  }}
</script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote:", out_html)
    print("Species plotted:", ", ".join(species_list))

def _parse_bbox(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'minLon,minLat,maxLon,maxLat'")
    try:
        minLon, minLat, maxLon, maxLat = map(float, parts)
    except Exception:
        raise argparse.ArgumentTypeError("bbox values must be numbers")
    return (minLon, minLat, maxLon, maxLat)

def main():
    ap = argparse.ArgumentParser(prog="dwca_sqlite.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load")
    p_load.add_argument("--dwca", required=True)
    p_load.add_argument("--db", default="dwca.sqlite")
    p_load.add_argument("--limit", type=int, default=0)
    p_load.add_argument("--commit-every", type=int, default=2000)
    p_load.add_argument("--no-raw", action="store_true")

    p_count = sub.add_parser("count")
    p_count.add_argument("--db", default="dwca.sqlite")

    p_cols = sub.add_parser("cols")
    p_cols.add_argument("--db", default="dwca.sqlite")

    p_top = sub.add_parser("top")
    p_top.add_argument("--db", default="dwca.sqlite")
    p_top.add_argument("--col", required=True)
    p_top.add_argument("-n", type=int, default=20)
    p_top.add_argument("--where", default="")

    p_sql = sub.add_parser("sql")
    p_sql.add_argument("--db", default="dwca.sqlite")
    p_sql.add_argument("--q", required=True)
    p_sql.add_argument("--limit", type=int, default=50)

    p_map_top = sub.add_parser("map_top", help="Export Leaflet map for top N species (different colors)")
    p_map_top.add_argument("--db", default="dwca.sqlite")
    p_map_top.add_argument("--out", default="map_top.html")
    p_map_top.add_argument("--top-n", type=int, default=10, help="How many top species to show")
    p_map_top.add_argument("--limit", type=int, default=6000, help="Total points cap (split across species)")
    p_map_top.add_argument("--lat0", type=float, default=51.3397)
    p_map_top.add_argument("--lon0", type=float, default=12.3731)
    p_map_top.add_argument("--zoom", type=int, default=6)
    p_map_top.add_argument("--year-from", type=int, default=None)
    p_map_top.add_argument("--year-to", type=int, default=None)
    p_map_top.add_argument("--country", default=None, help="Country code filter (e.g., DE)")
    p_map_top.add_argument("--bbox", type=_parse_bbox, default=None, help="minLon,minLat,maxLon,maxLat")
    p_map_top.add_argument("--radius", type=int, default=2)

    args = ap.parse_args()

    if args.cmd == "load":
        load(args.dwca, args.db, limit=args.limit, commit_every=args.commit_every, keep_raw=not args.no_raw)
    elif args.cmd == "count":
        count_rows(args.db)
    elif args.cmd == "cols":
        columns(args.db)
    elif args.cmd == "top":
        top_values(args.db, args.col, args.n, args.where)
    elif args.cmd == "sql":
        sql(args.db, args.q, args.limit)
    elif args.cmd == "map_top":
        export_map_top_html(
            db_path=args.db,
            out_html=args.out,
            top_n=args.top_n,
            limit_points=args.limit,
            lat0=args.lat0,
            lon0=args.lon0,
            zoom=args.zoom,
            year_from=args.year_from,
            year_to=args.year_to,
            country=args.country,
            bbox=args.bbox,
            radius=args.radius
        )

if __name__ == "__main__":
    main()
