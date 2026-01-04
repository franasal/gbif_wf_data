#!/usr/bin/env python3
import json
import os
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


GBIF_REQUEST_URL = "https://api.gbif.org/v1/occurrence/download/request"
GBIF_STATUS_URL = "https://api.gbif.org/v1/occurrence/download/{key}"
GBIF_ZIP_URL = "https://api.gbif.org/occurrence/download/request/{key}.zip"


def utc_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def iso_date(d: datetime) -> str:
    return d.date().isoformat()


@dataclass
class Config:
    country: Optional[str]
    year_from: Optional[int]
    year_to: Optional[int]
    taxon_keys: List[int]
    require_coordinate: bool
    overlap_days: int


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> Config:
    data = load_json(path, {})
    return Config(
        country=data.get("country") or None,
        year_from=data.get("year_from"),
        year_to=data.get("year_to"),
        taxon_keys=[int(x) for x in data.get("taxon_keys", [])],
        require_coordinate=bool(data.get("require_coordinate", True)),
        overlap_days=int(data.get("overlap_days", 2)),
    )


def build_predicates(cfg: Config, interpreted_since: str) -> dict:
    preds = []

    if cfg.require_coordinate:
        preds.append({"type": "equals", "key": "HAS_COORDINATE", "value": "true"})

    if cfg.country:
        preds.append({"type": "equals", "key": "COUNTRY", "value": cfg.country})

    if cfg.taxon_keys:
        preds.append({"type": "in", "key": "TAXON_KEY", "values": [str(x) for x in cfg.taxon_keys]})

    if cfg.year_from is not None:
        preds.append({"type": "greaterThanOrEquals", "key": "YEAR", "value": str(cfg.year_from)})
    if cfg.year_to is not None:
        preds.append({"type": "lessThanOrEquals", "key": "YEAR", "value": str(cfg.year_to)})

    # Incremental part: only records interpreted since last run (or overlap)
    preds.append({"type": "greaterThanOrEquals", "key": "LAST_INTERPRETED", "value": interpreted_since})

    return {"type": "and", "predicates": preds}


def gbif_request_download(user: str, pwd: str, email: str, predicate: dict) -> str:
    body = {
        "creator": user,
        "notificationAddresses": [email],
        "sendNotification": False,  # actions doesn’t need email spam
        "format": "DWCA",
        "predicate": predicate,
    }
    r = requests.post(GBIF_REQUEST_URL, auth=(user, pwd), json=body, timeout=60)
    r.raise_for_status()
    key = r.text.strip().strip('"')
    if not key or "-" not in key:
        raise RuntimeError(f"Unexpected download key response: {r.text[:200]}")
    return key


def gbif_poll_until_ready(key: str, timeout_s: int = 3600, poll_s: int = 30) -> dict:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        r = requests.get(GBIF_STATUS_URL.format(key=key), timeout=60)
        r.raise_for_status()
        info = r.json()
        status = info.get("status")
        if status != last_status:
            print(f"GBIF download {key} status: {status}")
            last_status = status

        if status == "SUCCEEDED":
            return info
        if status in ("KILLED", "CANCELLED", "FAILED"):
            raise RuntimeError(f"GBIF download ended with status={status}: {info}")

        time.sleep(poll_s)

    raise TimeoutError(f"GBIF download {key} not ready after {timeout_s}s (last_status={last_status})")


def download_zip(key: str, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(GBIF_ZIP_URL.format(key=key), stream=True, timeout=300) as r:
        r.raise_for_status()
        with out_zip.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def open_sqlite(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS occurrence (
          gbifID INTEGER PRIMARY KEY,
          scientificName TEXT,
          taxonKey INTEGER,
          decimalLatitude REAL,
          decimalLongitude REAL,
          year INTEGER,
          month INTEGER,
          countryCode TEXT,
          eventDate TEXT,
          lastInterpreted TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_occ_sci ON occurrence(scientificName);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_occ_taxon ON occurrence(taxonKey);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_occ_year ON occurrence(year);")


def parse_int(x: str) -> Optional[int]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def parse_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def read_tsv_rows(path: Path, wanted: List[str]) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}

        missing = [c for c in wanted if c not in idx]
        if missing:
            raise RuntimeError(f"occurrence.txt missing required columns: {missing}")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = {}
            for c in wanted:
                i = idx[c]
                row[c] = parts[i] if i < len(parts) else ""
            yield row


def upsert_occurrences(con: sqlite3.Connection, occ_path: Path) -> int:
    # These columns exist in GBIF DWCA occurrence.txt (interpreted)
    wanted = [
        "gbifID",
        "scientificName",
        "taxonKey",
        "decimalLatitude",
        "decimalLongitude",
        "year",
        "month",
        "countryCode",
        "eventDate",
        "lastInterpreted",
    ]

    sql = """
    INSERT INTO occurrence (
      gbifID, scientificName, taxonKey, decimalLatitude, decimalLongitude,
      year, month, countryCode, eventDate, lastInterpreted
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(gbifID) DO UPDATE SET
      scientificName=excluded.scientificName,
      taxonKey=excluded.taxonKey,
      decimalLatitude=excluded.decimalLatitude,
      decimalLongitude=excluded.decimalLongitude,
      year=excluded.year,
      month=excluded.month,
      countryCode=excluded.countryCode,
      eventDate=excluded.eventDate,
      lastInterpreted=excluded.lastInterpreted
    """

    buf: List[Tuple] = []
    n = 0
    for row in read_tsv_rows(occ_path, wanted=wanted):
        gbif_id = parse_int(row["gbifID"])
        if gbif_id is None:
            continue

        lat = parse_float(row["decimalLatitude"])
        lon = parse_float(row["decimalLongitude"])
        if lat is None or lon is None:
            continue

        buf.append(
            (
                gbif_id,
                row["scientificName"] or None,
                parse_int(row["taxonKey"] or ""),
                lat,
                lon,
                parse_int(row["year"] or ""),
                parse_int(row["month"] or ""),
                row["countryCode"] or None,
                row["eventDate"] or None,
                row["lastInterpreted"] or None,
            )
        )

        if len(buf) >= 5000:
            con.executemany(sql, buf)
            n += len(buf)
            buf.clear()

    if buf:
        con.executemany(sql, buf)
        n += len(buf)

    con.commit()
    return n


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "data" / "gbif_download_config.json"
    state_path = repo_root / "data" / "gbif_state.json"
    db_path = repo_root / "data" / "dwca.sqlite"

    user = os.environ["GBIF_USER"]
    pwd = os.environ["GBIF_PWD"]
    email = os.environ["GBIF_EMAIL"]

    cfg = load_config(cfg_path)
    if not cfg.taxon_keys:
        raise RuntimeError("No taxon_keys configured in data/gbif_download_config.json")

    state = load_json(state_path, {"last_interpreted_since": iso_date(utc_today() - timedelta(days=1))})
    last_since = state.get("last_interpreted_since") or iso_date(utc_today() - timedelta(days=1))

    # Overlap to avoid missing records due to processing delays
    try:
        last_dt = datetime.fromisoformat(last_since).replace(tzinfo=timezone.utc)
    except Exception:
        last_dt = utc_today() - timedelta(days=1)

    since_dt = last_dt - timedelta(days=max(0, cfg.overlap_days))
    since_str = iso_date(since_dt)
    print(f"Incremental filter: LAST_INTERPRETED >= {since_str}")

    predicate = build_predicates(cfg, interpreted_since=since_str)

    key = gbif_request_download(user, pwd, email, predicate=predicate)
    print(f"Requested GBIF download: {key}")

    gbif_poll_until_ready(key, timeout_s=3600, poll_s=30)

    tmp_dir = repo_root / ".tmp_gbif"
    tmp_zip = tmp_dir / f"{key}.zip"
    download_zip(key, tmp_zip)

    with zipfile.ZipFile(tmp_zip, "r") as z:
        members = set(z.namelist())
        if "occurrence.txt" not in members:
            raise RuntimeError(f"DWCA zip missing occurrence.txt. Contains: {list(members)[:20]}")
        z.extract("occurrence.txt", path=tmp_dir)

    occ_path = tmp_dir / "occurrence.txt"

    con = open_sqlite(db_path)
    ensure_schema(con)
    inserted = upsert_occurrences(con, occ_path)
    con.close()

    print(f"Upserted rows (processed): {inserted:,}")

    # Advance state to today (UTC). Next run will subtract overlap_days anyway.
    new_state = {"last_interpreted_since": iso_date(utc_today())}
    save_json(state_path, new_state)
    print(f"Updated state: {new_state}")


if __name__ == "__main__":
    main()
