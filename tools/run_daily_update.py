#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests

GBIF_REQUEST_URL = "https://api.gbif.org/v1/occurrence/download/request"
GBIF_STATUS_URL = "https://api.gbif.org/v1/occurrence/download/{key}"
GBIF_ZIP_URL = "https://api.gbif.org/v1/occurrence/download/request/{key}.zip"


def utc_today_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd: list[str], env: dict | None = None) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_occurrence_file(dwca_dir: Path) -> Path:
    occ = dwca_dir / "occurrence.txt"
    if occ.exists():
        return occ
    txts = sorted(dwca_dir.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
    if not txts:
        raise FileNotFoundError(f"No .txt files found in {dwca_dir}")
    return txts[0]


def parse_iso_date(value: str) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        if len(value) >= 10:
            return value[:10]
        return datetime.fromisoformat(value).date().isoformat()
    except Exception:
        return None


def resolve_first(row: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        val = row.get(key)
        if val:
            return str(val).strip()
    return None


def summarize_updates(
    occ_path: Path,
    allowed_scientific_names: set[str],
    out_path: Path,
    window_start: str,
    window_days: int,
    window_label: str,
    download_key: str,
    interpreted_since: str,
) -> None:
    import csv

    if not occ_path.exists():
        raise SystemExit(f"Missing occurrence file for summary: {occ_path}")

    summary = {
        "generated_at": utc_now_iso(),
        "download_key": download_key,
        "interpreted_since": interpreted_since,
        "window_start": window_start,
        "window_days": window_days,
        "window_label": window_label,
        "total_new_points": 0,
        "per_species": {},
    }

    with occ_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sci = resolve_first(
                row,
                [
                    "scientificName",
                    "species",
                    "acceptedScientificName",
                    "canonicalName",
                ],
            )
            if not sci or sci not in allowed_scientific_names:
                continue

            interpreted = resolve_first(row, ["lastInterpreted", "last_interpreted", "modified", "lastModified"])
            interpreted_date = parse_iso_date(interpreted or "")
            if interpreted_date is None or interpreted_date < window_start:
                continue

            summary["total_new_points"] += 1
            summary["per_species"][sci] = summary["per_species"].get(sci, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote update summary: {out_path}", flush=True)


def build_predicate(resolved_plants: list[dict], cfg: dict, interpreted_since: str) -> dict:
    preds = []

    if cfg.get("require_coordinate", True):
        preds.append({"type": "equals", "key": "HAS_COORDINATE", "value": "true"})

    country = cfg.get("country") or None
    if country:
        preds.append({"type": "equals", "key": "COUNTRY", "value": country})

    taxon_keys = []
    for p in resolved_plants:
        tk = p.get("taxonKey")
        if tk is None:
            continue
        taxon_keys.append(str(int(tk)))

    if not taxon_keys:
        raise RuntimeError("No taxonKeys found in plants_resolved.json (resolver failed or empty names_de.json).")

    preds.append({"type": "in", "key": "TAXON_KEY", "values": taxon_keys})

    y_from = cfg.get("year_from")
    y_to = cfg.get("year_to")
    if y_from is not None:
        preds.append({"type": "greaterThanOrEquals", "key": "YEAR", "value": str(int(y_from))})
    if y_to is not None:
        preds.append({"type": "lessThanOrEquals", "key": "YEAR", "value": str(int(y_to))})

    preds.append({"type": "greaterThanOrEquals", "key": "LAST_INTERPRETED", "value": interpreted_since})

    return {"type": "and", "predicates": preds}


def request_download(user: str, pwd: str, email: str, predicate: dict) -> str:
    body = {
        "creator": user,
        "notificationAddresses": [email],
        "sendNotification": False,
        "format": "DWCA",
        "predicate": predicate,
    }
    r = requests.post(GBIF_REQUEST_URL, auth=(user, pwd), json=body, timeout=60)
    r.raise_for_status()
    key = r.text.strip().strip('"')
    if not key or "-" not in key:
        raise RuntimeError(f"Unexpected GBIF response: {r.text[:200]}")
    return key


def poll_until_succeeded(key: str, timeout_s: int = 3 * 3600, poll_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = requests.get(GBIF_STATUS_URL.format(key=key), timeout=60)
        r.raise_for_status()
        status = r.json().get("status")
        if status != last:
            print(f"GBIF {key}: {status}", flush=True)
            last = status
        if status == "SUCCEEDED":
            return
        if status in ("KILLED", "CANCELLED", "FAILED"):
            raise RuntimeError(f"GBIF download failed: {status}")
        time.sleep(poll_s)
    raise TimeoutError(f"GBIF {key} not ready after {timeout_s}s")


def download_zip(key: str, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(GBIF_ZIP_URL.format(key=key), stream=True, timeout=300) as r:
        r.raise_for_status()
        with out_zip.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def compute_download_window(state: dict, cfg: dict) -> tuple[str, str, int, bool]:
    today = datetime.now(timezone.utc).date()
    daily_window_days = int(cfg.get("daily_window_days", 1))
    weekly_window_days = int(cfg.get("weekly_window_days", 7))
    weekly_refresh_days = int(cfg.get("weekly_refresh_days", 7))

    last_weekly = state.get("last_weekly_refresh")
    last_weekly_date = None
    if last_weekly:
        try:
            last_weekly_date = datetime.fromisoformat(last_weekly).date()
        except Exception:
            last_weekly_date = None

    weekly_due = (
        last_weekly_date is None
        or (today - last_weekly_date).days >= max(1, weekly_refresh_days)
    )

    if weekly_due:
        window_days = max(1, weekly_window_days)
        since_dt = today - timedelta(days=window_days)
    else:
        window_days = max(1, daily_window_days)
        since_dt = today - timedelta(days=window_days)

    since = since_dt.isoformat()
    return since, today.isoformat(), window_days, weekly_due


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily GBIF update pipeline (supports db-only/export-only).")
    ap.add_argument("--db-only", action="store_true", help="Run steps up to SQLite load only.")
    ap.add_argument("--export-only", action="store_true", help="Run export+stats only (requires existing data/dwca.sqlite).")
    args = ap.parse_args()

    if args.db_only and args.export_only:
        raise SystemExit("Use at most one of --db-only or --export-only.")

    mode = "all"
    if args.db_only:
        mode = "db-only"
    elif args.export_only:
        mode = "export-only"
    print(f"Mode: {mode}", flush=True)

    repo = Path(__file__).resolve().parents[1]

    # Inputs you maintain
    names_path = repo / "data" / "names_de.json"

    # Generated by resolver
    resolved_path = repo / "data" / "plants_resolved.json"
    cache_path = repo / "data" / "taxon_cache.json"

    # Config + state
    cfg_path = repo / "data" / "gbif_download_config.json"
    state_path = repo / "data" / "gbif_state.json"

    # DB + output
    db_path = repo / "data" / "dwca.sqlite"
    out_json_plain = repo / "data" / "occurrences_compact.json"
    out_json_gz = repo / "data" / "occurrences_compact.json.gz"
    updates_summary_path = repo / "data" / "updates_summary.json"

    # Scripts
    resolver = repo / "tools" / "resolve_taxa.py"
    loader = repo / "tools" / "dwca_sqlite.py"
    exporter = repo / "tools" / "export_occurrences_compact.py"
    if not names_path.exists():
        raise SystemExit(f"Missing: {names_path}")
    for p in (resolver, loader, exporter):
        if not p.exists():
            raise SystemExit(f"Missing script: {p}")

    cfg = load_json(cfg_path, {})
    state = load_json(state_path, {"last_interpreted_since": utc_today_date()})

    country = cfg.get("country", "DE")
    y_from = cfg.get("year_from")
    y_to = cfg.get("year_to")
    gzip_json = bool(cfg.get("gzip_json", False))

    export_out = out_json_gz if gzip_json else out_json_plain

    # ---------- DB STEP ----------
    if mode in ("all", "db-only"):
        names_hash = file_sha256(names_path)
        has_cached = resolved_path.exists() and cache_path.exists()
        if state.get("names_hash") == names_hash and has_cached:
            print("Names unchanged; using cached taxa resolution.", flush=True)
        else:
            run([
                "python", "-u", str(resolver),
                "--names", str(names_path),
                "--out", str(resolved_path),
                "--cache", str(cache_path),
            ])
            state["names_hash"] = names_hash
            save_json(state_path, state)

        resolved_plants = load_json(resolved_path, [])
        if not isinstance(resolved_plants, list) or not resolved_plants:
            raise SystemExit(f"{resolved_path} is empty or invalid.")

        since, window_end, window_days, weekly_due = compute_download_window(state, cfg)
        window_label = "weekly" if weekly_due else "daily"
        print(
            f"Delta filter: LAST_INTERPRETED >= {since} ({window_label} window_days={window_days})",
            flush=True,
        )

        user = os.environ["GBIF_USER"]
        pwd = os.environ["GBIF_PWD"]
        email = os.environ.get("GBIF_EMAIL", "noreply@example.org")

        predicate = build_predicate(resolved_plants, cfg, interpreted_since=since)
        key = request_download(user, pwd, email, predicate)
        print(f"Requested download: {key}", flush=True)

        state["pending"] = {
            "download_key": key,
            "since": since,
            "requested_at": utc_now_iso(),
            "window_days": window_days,
            "window_label": window_label,
        }
        save_json(state_path, state)

        poll_until_succeeded(key)

        tmp = repo / ".tmp_gbif" / key
        tmp.mkdir(parents=True, exist_ok=True)
        zip_path = tmp / f"{key}.zip"

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            download_zip(key, zip_path)
        else:
            print(f"ZIP already present: {zip_path}", flush=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        occ_path = find_occurrence_file(tmp)
        window_start = since
        allowed_names = {p.get("scientificName") for p in resolved_plants if p.get("scientificName")}
        summarize_updates(
            occ_path=occ_path,
            allowed_scientific_names=allowed_names,
            out_path=updates_summary_path,
            window_start=window_start,
            window_days=window_days,
            window_label=window_label,
            download_key=key,
            interpreted_since=since,
        )

        run([
            "python", "-u", str(loader), "load",
            "--dwca", str(tmp),
            "--db", str(db_path),
            "--no-raw",
        ])

        print(f"DB ready: {db_path}", flush=True)

        if weekly_due:
            state["last_weekly_refresh"] = window_end
            save_json(state_path, state)

        if mode == "db-only":
            print("DB-only run finished. Export is handled by the next job.", flush=True)
            return

    # ---------- EXPORT STEP ----------
    if mode in ("all", "export-only"):
        if not db_path.exists():
            raise SystemExit(f"Missing DB: {db_path} (did you run db-only job / restore cache/artifact?)")

        export_args = [
            "python", "-u", str(exporter),
            "--db", str(db_path),
            "--out", str(export_out),
            "--names-json", str(names_path),
            "--top-n", str(int(cfg.get("top_n", 250))),
            "--cell-precision", str(int(cfg.get("cell_precision", 5))),
            "--keep-per-cell", str(int(cfg.get("keep_per_cell", 6))),
            "--max-points-per-plant", str(int(cfg.get("max_points_per_plant", 700))),
        ]

        if country:
            export_args += ["--country", str(country)]
        if y_from is not None:
            export_args += ["--year-from", str(int(y_from))]
        if y_to is not None:
            export_args += ["--year-to", str(int(y_to))]

        if cfg.get("images_index"):
            export_args += ["--images-index", str(repo / cfg["images_index"])]

        if gzip_json:
            export_args += ["--gzip"]

        run(export_args)

        # Advance state ONLY after export success
        new_state = dict(state)
        new_state["last_interpreted_since"] = utc_today_date()
        if "pending" in new_state:
            new_state["pending"]["completed_at"] = utc_now_iso()
            new_state["pending"]["status"] = "exported"
        save_json(state_path, new_state)

        print(f"Updated state: last_interpreted_since={new_state['last_interpreted_since']}", flush=True)
        print(f"Wrote: {export_out}", flush=True)


if __name__ == "__main__":
    main()
