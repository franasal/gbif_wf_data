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


def whitelist_to_name_map(data) -> dict[str, str]:
    if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
        return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip()}

    out: dict[str, str] = {}
    items = data.values() if isinstance(data, dict) else data
    if isinstance(items, list) or hasattr(items, "__iter__"):
        for item in items:
            if not isinstance(item, dict):
                continue
            sci = item.get("scientificName") or item.get("scientific_name")
            de = item.get("de") or item.get("commonName") or item.get("common_name") or ""
            if isinstance(sci, str) and sci.strip():
                out[sci.strip()] = str(de or "").strip()
    return out


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
    window_field: str = "LAST_INTERPRETED",
) -> None:
    import csv

    if not occ_path.exists():
        raise SystemExit(f"Missing occurrence file for summary: {occ_path}")

    summary = {
        "generated_at": utc_now_iso(),
        "download_key": download_key,
        "interpreted_since": interpreted_since,
        "window_since": window_start,
        "window_field": window_field,
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

            if (window_field or "").upper() == "EVENT_DATE":
                raw_date = resolve_first(row, ["eventDate"])
            else:
                raw_date = resolve_first(row, ["lastInterpreted", "last_interpreted", "modified", "lastModified"])

            row_date = parse_iso_date(raw_date or "")
            if row_date is None or row_date < window_start:
                continue

            summary["total_new_points"] += 1
            summary["per_species"][sci] = summary["per_species"].get(sci, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote update summary: {out_path}", flush=True)


def build_predicate(
    resolved_plants: list[dict],
    cfg: dict,
    interpreted_since: str,
    taxon_cache: dict | None = None,
) -> dict:
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
        if taxon_cache:
            entry = taxon_cache.get(p.get("scientificName"))
            if isinstance(entry, dict):
                status = str(entry.get("status") or "").upper()
                accepted = entry.get("acceptedUsageKey")
                usage = entry.get("usageKey") or tk
                # GBIF occurrence downloads don't reliably expand synonyms for TAXON_KEY predicates.
                # To avoid silently dropping taxa, include both the synonym usageKey and its
                # acceptedUsageKey (when available).
                if status == "SYNONYM" and accepted is not None:
                    taxon_keys.append(str(int(usage)))
                    taxon_keys.append(str(int(accepted)))
                    continue
                tk = usage
        taxon_keys.append(str(int(tk)))

    if not taxon_keys:
        raise RuntimeError("No taxonKeys found in resolved plant lists (resolver failed or empty names files).")

    # De-duplicate but keep stable order (helps reproducible predicates + debugging).
    deduped = []
    seen = set()
    for v in taxon_keys:
        if v in seen:
            continue
        seen.add(v)
        deduped.append(v)
    preds.append({"type": "in", "key": "TAXON_KEY", "values": deduped})

    y_from = cfg.get("year_from")
    y_to = cfg.get("year_to")
    if y_from is not None:
        preds.append({"type": "greaterThanOrEquals", "key": "YEAR", "value": str(int(y_from))})
    if y_to is not None:
        preds.append({"type": "lessThanOrEquals", "key": "YEAR", "value": str(int(y_to))})

    window_key = str(cfg.get("window_filter_key") or "EVENT_DATE").upper()
    preds.append({"type": "greaterThanOrEquals", "key": window_key, "value": interpreted_since})

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


def compute_download_window(state: dict, cfg: dict, *, force_weekly: bool = False) -> tuple[str, str, int, bool]:
    today = datetime.now(timezone.utc).date()
    daily_window_days = int(cfg.get("daily_window_days", 1))
    weekly_refresh_weekday = int(cfg.get("weekly_refresh_weekday", 2))
    rolling_window_days = int(cfg.get("rolling_window_days", 0) or 0)
    year_from = cfg.get("year_from")

    last_weekly = state.get("last_weekly_refresh")
    last_weekly_date = None
    if last_weekly:
        try:
            last_weekly_date = datetime.fromisoformat(last_weekly).date()
        except Exception:
            last_weekly_date = None

    weekly_due = force_weekly or (today.weekday() == weekly_refresh_weekday and last_weekly_date != today)

    if weekly_due and rolling_window_days > 0:
        since_dt = today - timedelta(days=rolling_window_days)
    elif weekly_due and year_from is not None:
        since_dt = datetime(int(year_from), 1, 1, tzinfo=timezone.utc).date()
    else:
        window_days = max(1, daily_window_days)
        since_dt = today - timedelta(days=window_days)

    window_days = max(1, (today - since_dt).days)
    since = since_dt.isoformat()
    return since, today.isoformat(), window_days, weekly_due


def prune_db_by_date(db_path: Path, cutoff_date: str) -> int:
    import sqlite3

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")

    delete_sql = """
      DELETE FROM occ
      WHERE
        (
          eventDate IS NOT NULL
          AND length(eventDate) >= 10
          AND date(substr(eventDate, 1, 10)) < date(?)
        )
        OR
        (
          year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
          AND date(printf('%04d-%02d-%02d', year, month, day)) < date(?)
        )
    """
    cur.execute(delete_sql, (cutoff_date, cutoff_date))
    deleted = cur.rowcount if cur.rowcount is not None else 0
    con.commit()
    con.close()
    return deleted


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily GBIF update pipeline (supports db-only/export-only).")
    ap.add_argument("--db-only", action="store_true", help="Run steps up to SQLite load only.")
    ap.add_argument("--export-only", action="store_true", help="Run export+stats only (requires existing data/dwca.sqlite).")
    ap.add_argument("--download-key", default=None, help="Reuse an existing GBIF download key instead of requesting a new one.")
    ap.add_argument(
        "--force-weekly",
        action="store_true",
        help="Force a weekly-style refresh window (rolling_window_days or year_from), regardless of weekday.",
    )
    ap.add_argument("--no-prune", action="store_true", help="Skip rolling-window DB pruning after load.")
    ap.add_argument("--no-change-log", action="store_true", help="Skip writing change summary JSON.")
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
    names_edible_path = repo / "data" / "names_edible.json"
    names_poisonous_path = repo / "data" / "names_poisonous.json"
    names_prod_path = repo / "data" / "names_prod.json"

    # Generated by resolver
    resolved_edible_path = repo / "data" / "plants_resolved_edible.json"
    resolved_poisonous_path = repo / "data" / "plants_resolved_poisonous.json"
    resolved_prod_path = repo / "data" / "plants_resolved.json"
    cache_path = repo / "data" / "taxon_cache.json"

    # Config + state
    cfg_path = repo / "data" / "gbif_download_config.json"
    state_path = repo / "data" / "gbif_state.json"

    # DB + output
    db_path = repo / "data" / "dwca.sqlite"
    out_edible_plain = repo / "data" / "occurrences_compact_edible.json"
    out_edible_gz = repo / "data" / "occurrences_compact_edible.json.gz"
    out_poisonous_plain = repo / "data" / "occurrences_compact_poisonous.json"
    out_poisonous_gz = repo / "data" / "occurrences_compact_poisonous.json.gz"
    out_prod_plain = repo / "data" / "occurrences_compact.json"
    out_prod_gz = repo / "data" / "occurrences_compact.json.gz"
    updates_edible_path = repo / "data" / "updates_summary_edible.json"
    updates_poisonous_path = repo / "data" / "updates_summary_poisonous.json"
    updates_summary_path = repo / "data" / "updates_summary.json"
    changes_summary_path = repo / "data" / "changes_summary.json"

    # Scripts
    resolver = repo / "tools" / "resolve_taxa.py"
    loader = repo / "tools" / "dwca_sqlite.py"
    exporter = repo / "tools" / "export_occurrences_compact.py"
    if not names_edible_path.exists():
        raise SystemExit(f"Missing: {names_edible_path}")
    if not names_poisonous_path.exists():
        raise SystemExit(f"Missing: {names_poisonous_path}")
    for p in (resolver, loader, exporter):
        if not p.exists():
            raise SystemExit(f"Missing script: {p}")

    cfg = load_json(cfg_path, {})
    state = load_json(state_path, {"last_interpreted_since": utc_today_date()})

    edible_raw = load_json(names_edible_path, {})
    poisonous_raw = load_json(names_poisonous_path, {})
    edible_map = whitelist_to_name_map(edible_raw)
    poisonous_map = whitelist_to_name_map(poisonous_raw)
    edible_names = set(edible_map.keys())
    poisonous_names = set(poisonous_map.keys())
    overlap = sorted(edible_names & poisonous_names)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise SystemExit(f"Edible/poisonous lists overlap ({len(overlap)}): {preview}")

    # Build legacy/prod combined list for compatibility.
    if not isinstance(edible_map, dict) or not isinstance(poisonous_map, dict):
        raise SystemExit("names_edible.json and names_poisonous.json must be JSON dicts.")
    prod_map = dict(edible_map)
    prod_map.update(poisonous_map)
    save_json(names_prod_path, prod_map)

    country = cfg.get("country", "DE")
    y_from = cfg.get("year_from")
    y_to = cfg.get("year_to")
    gzip_json = bool(cfg.get("gzip_json", False))
    rolling_window_days = int(cfg.get("rolling_window_days", 0) or 0)
    if rolling_window_days > 0:
        today = datetime.now(timezone.utc).date()
        cutoff = today - timedelta(days=rolling_window_days)
        y_from = cutoff.year
        y_to = today.year

    export_edible_out = out_edible_gz if gzip_json else out_edible_plain
    export_poisonous_out = out_poisonous_gz if gzip_json else out_poisonous_plain
    export_prod_out = out_prod_gz if gzip_json else out_prod_plain

    # ---------- DB STEP ----------
    if mode in ("all", "db-only"):
        names_edible_hash = file_sha256(names_edible_path)
        names_poisonous_hash = file_sha256(names_poisonous_path)
        has_cached = (
            resolved_edible_path.exists()
            and resolved_poisonous_path.exists()
            and cache_path.exists()
        )
        if (
            state.get("names_edible_hash") == names_edible_hash
            and state.get("names_poisonous_hash") == names_poisonous_hash
            and has_cached
        ):
            print("Names unchanged; using cached taxa resolution.", flush=True)
        else:
            run([
                "python", "-u", str(resolver),
                "--names", str(names_edible_path),
                "--out", str(resolved_edible_path),
                "--cache", str(cache_path),
            ])
            run([
                "python", "-u", str(resolver),
                "--names", str(names_poisonous_path),
                "--out", str(resolved_poisonous_path),
                "--cache", str(cache_path),
            ])
            state["names_edible_hash"] = names_edible_hash
            state["names_poisonous_hash"] = names_poisonous_hash
            save_json(state_path, state)

        resolved_edible = load_json(resolved_edible_path, [])
        resolved_poisonous = load_json(resolved_poisonous_path, [])
        if not isinstance(resolved_edible, list) or not resolved_edible:
            raise SystemExit(f"{resolved_edible_path} is empty or invalid.")
        if not isinstance(resolved_poisonous, list) or not resolved_poisonous:
            raise SystemExit(f"{resolved_poisonous_path} is empty or invalid.")
        resolved_plants = resolved_edible + resolved_poisonous

        # Write legacy combined resolved file for compatibility.
        seen = set()
        resolved_prod = []
        for item in resolved_plants:
            if not isinstance(item, dict):
                continue
            sci = item.get("scientificName")
            if not sci or sci in seen:
                continue
            seen.add(sci)
            resolved_prod.append(item)
        save_json(resolved_prod_path, resolved_prod)

        since, window_end, window_days, weekly_due = compute_download_window(state, cfg, force_weekly=bool(args.force_weekly))
        window_label = "weekly" if weekly_due else "daily"
        window_filter_key = str(cfg.get("window_filter_key") or "EVENT_DATE").upper()
        print(
            f"Window filter: {window_filter_key} >= {since} ({window_label} window_days={window_days})",
            flush=True,
        )

        user = os.environ["GBIF_USER"]
        pwd = os.environ["GBIF_PWD"]
        email = os.environ.get("GBIF_EMAIL", "noreply@example.org")

        taxon_cache = load_json(repo / "data" / "taxon_cache.json", {})
        predicate = build_predicate(resolved_plants, cfg, interpreted_since=since, taxon_cache=taxon_cache)
        # A GBIF download key is a snapshot of a past query. Reusing one can make it look like
        # "nothing changes" even when predicates or refresh windows change. We only reuse a key
        # when explicitly provided via CLI.
        key = args.download_key
        if key:
            print(f"Using existing GBIF download: {key}", flush=True)
        else:
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

        tmp = repo / ".tmp_gbif" / key
        tmp.mkdir(parents=True, exist_ok=True)
        zip_path = tmp / f"{key}.zip"

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            poll_until_succeeded(key)
            download_zip(key, zip_path)
        else:
            print(f"ZIP already present: {zip_path}", flush=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        occ_path = find_occurrence_file(tmp)
        window_start = since
        allowed_edible = {p.get("scientificName") for p in resolved_edible if p.get("scientificName")}
        allowed_poisonous = {p.get("scientificName") for p in resolved_poisonous if p.get("scientificName")}
        allowed_prod = allowed_edible | allowed_poisonous
        summarize_updates(
            occ_path=occ_path,
            allowed_scientific_names=allowed_edible,
            out_path=updates_edible_path,
            window_start=window_start,
            window_days=window_days,
            window_label=window_label,
            download_key=key,
            interpreted_since=since,
            window_field=window_filter_key,
        )
        summarize_updates(
            occ_path=occ_path,
            allowed_scientific_names=allowed_prod,
            out_path=updates_summary_path,
            window_start=window_start,
            window_days=window_days,
            window_label=window_label,
            download_key=key,
            interpreted_since=since,
            window_field=window_filter_key,
        )
        summarize_updates(
            occ_path=occ_path,
            allowed_scientific_names=allowed_poisonous,
            out_path=updates_poisonous_path,
            window_start=window_start,
            window_days=window_days,
            window_label=window_label,
            download_key=key,
            interpreted_since=since,
            window_field=window_filter_key,
        )

        load_cmd = [
            "python", "-u", str(loader), "load",
            "--dwca", str(tmp),
            "--db", str(db_path),
            "--no-raw",
        ]
        if not args.no_change_log:
            load_cmd += ["--changes-out", str(changes_summary_path)]
        run(load_cmd)

        print(f"DB ready: {db_path}", flush=True)

        if rolling_window_days > 0 and not args.no_prune:
            cutoff = (datetime.now(timezone.utc).date() - timedelta(days=rolling_window_days)).isoformat()
            deleted = prune_db_by_date(db_path, cutoff)
            print(f"Pruned rows older than {cutoff}: {deleted}", flush=True)
        elif rolling_window_days > 0 and args.no_prune:
            print("Prune skipped (--no-prune).", flush=True)

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

        def build_export_args(out_path: Path, names_path: Path) -> list[str]:
            args = [
                "python", "-u", str(exporter),
                "--db", str(db_path),
                "--out", str(out_path),
                "--names-json", str(names_path),
                "--taxon-cache", str(repo / "data" / "taxon_cache.json"),
                "--top-n", str(int(cfg.get("top_n", 250))),
                "--cell-precision", str(int(cfg.get("cell_precision", 5))),
                "--keep-per-cell", str(int(cfg.get("keep_per_cell", 6))),
                "--max-points-per-plant", str(int(cfg.get("max_points_per_plant", 700))),
            ]
            if bool(cfg.get("adaptive_sampling", False)):
                args += ["--adaptive-sampling"]
            recent_max = int(cfg.get("recent_max_points_per_plant", 0) or 0)
            if recent_max > 0:
                recent_name = out_path.name.replace("occurrences_compact", "occurrences_recent")
                args += [
                    "--recent-out",
                    str(out_path.with_name(recent_name)),
                    "--recent-max-points-per-plant",
                    str(recent_max),
                ]

            if country:
                args += ["--country", str(country)]
            if y_from is not None:
                args += ["--year-from", str(int(y_from))]
            if y_to is not None:
                args += ["--year-to", str(int(y_to))]

            if cfg.get("images_index"):
                args += ["--images-index", str(repo / cfg["images_index"])]

            if gzip_json:
                args += ["--gzip"]

            return args

        run(build_export_args(export_edible_out, names_edible_path))
        run(build_export_args(export_poisonous_out, names_poisonous_path))
        run(build_export_args(export_prod_out, names_prod_path))

        # Advance state ONLY after export success
        new_state = dict(state)
        new_state["last_interpreted_since"] = utc_today_date()
        if "pending" in new_state:
            new_state["pending"]["completed_at"] = utc_now_iso()
            new_state["pending"]["status"] = "exported"
        save_json(state_path, new_state)

        print(f"Updated state: last_interpreted_since={new_state['last_interpreted_since']}", flush=True)
        print(f"Wrote: {export_edible_out}", flush=True)
        print(f"Wrote: {export_poisonous_out}", flush=True)
        print(f"Wrote: {export_prod_out}", flush=True)


if __name__ == "__main__":
    main()
