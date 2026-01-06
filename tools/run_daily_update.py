#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def compute_since(state: dict, cfg: dict) -> tuple[str, str]:
    last = state.get("last_interpreted_since") or utc_today_date()
    overlap_days = int(cfg.get("overlap_days", 2))

    try:
        last_dt = datetime.fromisoformat(last).replace(tzinfo=timezone.utc)
    except Exception:
        last_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    since_dt = last_dt - timedelta(days=max(0, overlap_days))
    since = since_dt.date().isoformat()
    return since, last


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

    # Scripts
    resolver = repo / "tools" / "resolve_taxa.py"
    loader = repo / "tools" / "dwca_sqlite.py"
    exporter = repo / "tools" / "export_occurrences_compact.py"
    stats_script = repo / "tools" / "generate_stats.py"

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
        run([
            "python", "-u", str(resolver),
            "--names", str(names_path),
            "--out", str(resolved_path),
            "--cache", str(cache_path),
        ])

        resolved_plants = load_json(resolved_path, [])
        if not isinstance(resolved_plants, list) or not resolved_plants:
            raise SystemExit(f"{resolved_path} is empty or invalid.")

        since, last_used = compute_since(state, cfg)
        overlap_days = int(cfg.get("overlap_days", 2))
        print(f"Delta filter: LAST_INTERPRETED >= {since} (overlap_days={overlap_days}, last_state={last_used})", flush=True)

        user = os.environ["GBIF_USER"]
        pwd = os.environ["GBIF_PWD"]
        email = os.environ.get("GBIF_EMAIL", "noreply@example.org")

        predicate = build_predicate(resolved_plants, cfg, interpreted_since=since)
        key = request_download(user, pwd, email, predicate)
        print(f"Requested download: {key}", flush=True)

        state["pending"] = {"download_key": key, "since": since, "requested_at": utc_now_iso()}
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

        run([
            "python", "-u", str(loader), "load",
            "--dwca", str(tmp),
            "--db", str(db_path),
            "--no-raw",
        ])

        print(f"DB ready: {db_path}", flush=True)

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

        # Stats
        if bool(cfg.get("stats_enabled", False)):
            if not stats_script.exists():
                raise SystemExit(f"stats_enabled=true but missing: {stats_script}")

            # Use *exactly the file we just wrote*
            occ_for_stats = export_out
            if not occ_for_stats.exists():
                # ultra defensive: if exporter appends .gz despite path
                gz_alt = Path(str(occ_for_stats) + ".gz")
                if gz_alt.exists():
                    occ_for_stats = gz_alt
                else:
                    raise FileNotFoundError(f"Expected export output missing: {export_out}")

            stats_out = repo / "data" / "stats_summary.json"
            stats_args = [
                "python", "-u", str(stats_script),
                "--db", str(db_path),
                "--occ-json", str(occ_for_stats),
                "--out", str(stats_out),
                "--country", str(country) if country else "DE",
            ]
            if y_from is not None:
                stats_args += ["--year-from", str(int(y_from))]
            if y_to is not None:
                stats_args += ["--year-to", str(int(y_to))]

            run(stats_args)

        print(f"Updated state: last_interpreted_since={new_state['last_interpreted_since']}", flush=True)
        print(f"Wrote: {export_out}", flush=True)


if __name__ == "__main__":
    main()
