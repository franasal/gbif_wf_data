#!/usr/bin/env python3
import argparse
import os
import csv
import json
import sqlite3
from datetime import datetime, timezone

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


def find_optional_multimedia(dwca_dir: str) -> str | None:
    candidates = [
        os.path.join(dwca_dir, "multimedia.txt"),
        os.path.join(dwca_dir, "Multimedia.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for name in os.listdir(dwca_dir):
        if name.lower() == "multimedia.txt":
            p = os.path.join(dwca_dir, name)
            if os.path.isfile(p):
                return p
    return None


def _table_columns(con, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    names = set()
    for r in rows:
        try:
            names.add(r["name"])
        except Exception:
            names.add(r[1])
    return names


def _ensure_column(con, table: str, name: str, ddl_type: str) -> None:
    cols = _table_columns(con, table)
    if name not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}")

def init_db(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS occ (
      gbifID TEXT PRIMARY KEY,
      scientificName TEXT,
      species TEXT,
      taxonKey INTEGER,
      eventDate TEXT,
      year INTEGER,
      month INTEGER,
      day INTEGER,
      countryCode TEXT,
      stateProvince TEXT,
      basisOfRecord TEXT,
      datasetKey TEXT,
      datasetName TEXT,
      publisher TEXT,
      referencesUrl TEXT,
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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS occ_media (
      gbifID TEXT PRIMARY KEY,
      imageUrl TEXT,
      mediaType TEXT,
      sourcePageUrl TEXT,
      license TEXT,
      creator TEXT,
      title TEXT
    );
    """)

    # Migrate older DBs in place.
    _ensure_column(con, "occ", "datasetName", "TEXT")
    _ensure_column(con, "occ", "publisher", "TEXT")
    _ensure_column(con, "occ", "referencesUrl", "TEXT")

    # Helpful indexes for export + stats
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_latlon ON occ(lat, lon);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_yearmonth ON occ(year DESC, month DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_species ON occ(species, scientificName);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_scientificName ON occ(scientificName);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_species_only ON occ(species);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_species_yearmonth ON occ(species, year DESC, month DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_state ON occ(stateProvince);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_occ_dataset_name ON occ(datasetName);")
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

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _norm_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _norm_float(x):
    try:
        return round(float(x), 6)
    except Exception:
        return None

def _norm_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def _compute_field_changes(incoming) -> dict:
    changed = {}
    for key, (old, new) in incoming.items():
        if old != new:
            changed[key] = {"from": old, "to": new}
    return changed


def _pick_col(row: dict, keys: list[str]):
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


def _looks_like_image(media_type, fmt, identifier) -> bool:
    s = " ".join(
        [
            str(media_type or "").lower(),
            str(fmt or "").lower(),
            str(identifier or "").lower(),
        ]
    )
    return any(t in s for t in ["image", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff"])


def load_multimedia(dwca_dir: str, con) -> int:
    path = find_optional_multimedia(dwca_dir)
    if not path:
        print("No multimedia.txt found; skipping media load.")
        return 0

    print("Using multimedia:", path)
    cur = con.cursor()
    loaded = 0
    batch = []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gbif_id = _pick_col(row, ["gbifID", "coreid", "coreId", "id"])
            if gbif_id is None:
                continue
            gbif_id = str(gbif_id).strip()
            if not gbif_id:
                continue

            identifier = _pick_col(row, ["identifier", "accessURI", "accessUri"])
            refs = _pick_col(row, ["references", "source", "sourcePage", "uri"])
            media_type = _pick_col(row, ["type", "dc:type", "mediaType"])
            fmt = _pick_col(row, ["format", "dc:format"])
            if not _looks_like_image(media_type, fmt, identifier):
                continue

            batch.append(
                (
                    gbif_id,
                    (str(identifier).strip() if identifier is not None else None),
                    (str(media_type).strip() if media_type is not None else None),
                    (str(refs).strip() if refs is not None else None),
                    (_pick_col(row, ["license", "dcterms:license"])),
                    (_pick_col(row, ["creator", "dcterms:creator"])),
                    (_pick_col(row, ["title", "dcterms:title"])),
                )
            )
            loaded += 1

            if len(batch) >= 2000:
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO occ_media
                    (gbifID, imageUrl, mediaType, sourcePageUrl, license, creator, title)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                con.commit()
                batch = []

    if batch:
        cur.executemany(
            """
            INSERT OR IGNORE INTO occ_media
            (gbifID, imageUrl, mediaType, sourcePageUrl, license, creator, title)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        con.commit()

    print("Loaded media rows (image candidates):", loaded)
    return loaded

def load(
    dwca_dir: str,
    db_path: str,
    limit: int = 0,
    commit_every: int = 2000,
    keep_raw: bool = True,
    changes_out: str | None = None,
):
    occ_path = find_occurrence(dwca_dir)
    print("Using:", occ_path)

    con = init_db(db_path)
    cur = con.cursor()

    track_changes = bool(changes_out)
    existing_count = con.execute("SELECT COUNT(1) FROM occ").fetchone()[0]
    can_compare = track_changes and existing_count > 0

    changes_summary = {
        "generated_at": _utc_now_iso(),
        "rows_scanned": 0,
        "rows_new": 0,
        "rows_updated": 0,
        "fields_changed": {},
    }

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
            taxonKey = to_int(row.get("taxonKey") or row.get("taxonID"))

            eventDate = row.get("eventDate")
            year = to_int(row.get("year"))
            month = to_int(row.get("month"))
            day = to_int(row.get("day"))

            countryCode = row.get("countryCode")
            stateProvince = row.get("stateProvince")

            basisOfRecord = row.get("basisOfRecord")
            datasetKey = row.get("datasetKey")
            datasetName = row.get("datasetName")
            publisher = row.get("publisher") or row.get("publishingOrg")
            referencesUrl = row.get("references")
            license_ = row.get("license")

            lat = to_float(row.get("decimalLatitude"))
            lon = to_float(row.get("decimalLongitude"))

            raw_tsv = ""
            if keep_raw:
                raw_tsv = "\t".join([row.get(c, "") or "" for c in cols])

            if track_changes:
                changes_summary["rows_scanned"] += 1
                if can_compare:
                    existing = cur.execute(
                        """
                        SELECT scientificName, species, taxonKey, eventDate, year, month, day,
                               countryCode, stateProvince, basisOfRecord, datasetKey, datasetName, publisher, referencesUrl,
                               license, lat, lon
                        FROM occ
                        WHERE gbifID = ?
                        """,
                        (gbifID,),
                    ).fetchone()
                else:
                    existing = None

                incoming = {
                    "scientificName": (_norm_str(existing["scientificName"]) if existing else None, _norm_str(scientificName)),
                    "species": (_norm_str(existing["species"]) if existing else None, _norm_str(species)),
                    "taxonKey": (_norm_int(existing["taxonKey"]) if existing else None, _norm_int(taxonKey)),
                    "eventDate": (_norm_str(existing["eventDate"]) if existing else None, _norm_str(eventDate)),
                    "year": (_norm_int(existing["year"]) if existing else None, _norm_int(year)),
                    "month": (_norm_int(existing["month"]) if existing else None, _norm_int(month)),
                    "day": (_norm_int(existing["day"]) if existing else None, _norm_int(day)),
                    "countryCode": (_norm_str(existing["countryCode"]) if existing else None, _norm_str(countryCode)),
                    "stateProvince": (_norm_str(existing["stateProvince"]) if existing else None, _norm_str(stateProvince)),
                    "basisOfRecord": (_norm_str(existing["basisOfRecord"]) if existing else None, _norm_str(basisOfRecord)),
                    "datasetKey": (_norm_str(existing["datasetKey"]) if existing else None, _norm_str(datasetKey)),
                    "datasetName": (_norm_str(existing["datasetName"]) if existing else None, _norm_str(datasetName)),
                    "publisher": (_norm_str(existing["publisher"]) if existing else None, _norm_str(publisher)),
                    "referencesUrl": (_norm_str(existing["referencesUrl"]) if existing else None, _norm_str(referencesUrl)),
                    "license": (_norm_str(existing["license"]) if existing else None, _norm_str(license_)),
                    "lat": (_norm_float(existing["lat"]) if existing else None, _norm_float(lat)),
                    "lon": (_norm_float(existing["lon"]) if existing else None, _norm_float(lon)),
                }

                if existing is None:
                    changes_summary["rows_new"] += 1
                else:
                    field_changes = _compute_field_changes(incoming)
                    if field_changes:
                        changes_summary["rows_updated"] += 1
                        for field in field_changes.keys():
                            changes_summary["fields_changed"][field] = (
                                changes_summary["fields_changed"].get(field, 0) + 1
                            )

            batch.append((
                gbifID, scientificName, species, taxonKey,
                eventDate, year, month, day,
                countryCode, stateProvince,
                basisOfRecord, datasetKey, datasetName, publisher, referencesUrl, license_,
                lat, lon, raw_tsv
            ))

            n += 1
            if limit and n >= limit:
                break

            if len(batch) >= commit_every:
                cur.executemany("""
                INSERT OR REPLACE INTO occ
                (gbifID, scientificName, species, taxonKey, eventDate, year, month, day, countryCode, stateProvince,
                 basisOfRecord, datasetKey, datasetName, publisher, referencesUrl, license, lat, lon, raw_tsv)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                con.commit()
                batch = []
                print("Loaded:", n)

        if batch:
            cur.executemany("""
            INSERT OR REPLACE INTO occ
            (gbifID, scientificName, species, taxonKey, eventDate, year, month, day, countryCode, stateProvince,
             basisOfRecord, datasetKey, datasetName, publisher, referencesUrl, license, lat, lon, raw_tsv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            con.commit()

    # Populate best-effort media table after occurrences are loaded.
    load_multimedia(dwca_dir, con)

    con.close()
    print("Done. Rows:", n, "DB:", db_path)
    if track_changes:
        out_path = changes_out
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(changes_summary, f, ensure_ascii=False, indent=2)
        print("Wrote changes summary:", out_path)

def main():
    ap = argparse.ArgumentParser(prog="dwca_sqlite.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load")
    p_load.add_argument("--dwca", required=True)
    p_load.add_argument("--db", default="dwca.sqlite")
    p_load.add_argument("--limit", type=int, default=0)
    p_load.add_argument("--commit-every", type=int, default=2000)
    p_load.add_argument("--no-raw", action="store_true")
    p_load.add_argument("--changes-out", default=None, help="Write change summary JSON (diffs vs existing DB)")

    args = ap.parse_args()

    if args.cmd == "load":
        load(
            args.dwca,
            args.db,
            limit=args.limit,
            commit_every=args.commit_every,
            keep_raw=not args.no_raw,
            changes_out=args.changes_out,
        )

if __name__ == "__main__":
    main()
