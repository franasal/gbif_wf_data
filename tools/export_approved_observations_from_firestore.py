#!/usr/bin/env python3
import argparse
import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _non_empty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = value.strip()
        return parsed or None
    if hasattr(value, "isoformat"):
        try:
            out = value.isoformat()
            if out.endswith("+00:00"):
                return out.replace("+00:00", "Z")
            return out
        except Exception:
            return None
    return None


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if hasattr(value, "astimezone"):
        try:
            return value.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _prune_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, val in obj.items():
            cleaned = _prune_none(val)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            out[key] = cleaned
        return out
    if isinstance(obj, list):
        return [_prune_none(v) for v in obj if _prune_none(v) is not None]
    return obj


def _normalize_name(value: Any) -> str | None:
    text = _non_empty(value)
    if text is None:
        return None
    return " ".join(text.lower().split())


def _load_bundle_identity_index(bundle_path: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load plant identity index from plants_bundle.json (Flutter app asset format)."""
    by_plant_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    try:
        with open(bundle_path, "r", encoding="utf-8") as handle:
            decoded = json.load(handle)
    except Exception:
        return by_plant_id, by_name

    plants = decoded.get("plants")
    if not isinstance(plants, list):
        return by_plant_id, by_name

    for plant in plants:
        if not isinstance(plant, dict):
            continue
        identity = {
            "taxonKey": plant.get("taxonKey"),
            "scientificName": _non_empty(plant.get("scientificName")),
        }
        plant_id = _non_empty(plant.get("id"))
        if plant_id is not None:
            by_plant_id[plant_id] = identity
        for candidate in (
            plant.get("scientificName"),
            plant.get("commonName"),
            plant.get("de"),
        ):
            normalized = _normalize_name(candidate)
            if normalized is not None and normalized not in by_name:
                by_name[normalized] = identity
    return by_plant_id, by_name


def _load_resolved_plants_index(
    resolved_paths: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load plant identity index from plants_resolved_*.json (flat list format used by the pipeline).

    Each entry: {"scientificName": "...", "taxonKey": 123456, ...}
    The plant id is derived as "<genus>_<species>" matching the Dart _fallbackId logic.
    """
    by_plant_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for path in resolved_paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                taxon_key = entry.get("taxonKey")
                if not isinstance(taxon_key, int):
                    continue
                sci = _non_empty(entry.get("scientificName"))
                if sci is None:
                    continue
                identity = {"taxonKey": taxon_key, "scientificName": sci}
                # Normalized scientific name as name-lookup key.
                key = _normalize_name(sci)
                if key and key not in by_name:
                    by_name[key] = identity
                # Derive plant id the same way as Dart _fallbackId.
                parts = sci.strip().split()
                if len(parts) >= 2:
                    plant_id = f"{parts[0].lower()}_{parts[1].lower()}"
                    by_plant_id.setdefault(plant_id, identity)
        except Exception as exc:
            print(f"[warn] could not load resolved plants from {path}: {exc}", flush=True)
    return by_plant_id, by_name


def _load_trigger(
    db: firestore.Client,
    collection_name: str,
    doc_id: str,
) -> dict[str, Any] | None:
    snap = db.collection(collection_name).document(doc_id).get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    return data


def _write_trigger_result(
    db: firestore.Client,
    collection_name: str,
    doc_id: str,
    *,
    count: int,
) -> None:
    db.collection(collection_name).document(doc_id).set(
        {
            "requested": False,
            "status": "published",
            "lastPublishedAt": firestore.SERVER_TIMESTAMP,
            "lastPublishedCount": count,
            "lastPublishedRunId": os.environ.get("GITHUB_RUN_ID"),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )


def _status_values(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for v in raw:
        value = v.strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export approved observations from Firestore to gzip JSON overlay.",
    )
    ap.add_argument("--service-account", required=True)
    ap.add_argument("--database-id", default="(default)")
    ap.add_argument("--out", default="data/approved_observations.json.gz")
    ap.add_argument(
        "--statuses",
        nargs="+",
        default=["approved", "published", "accepted"],
    )
    ap.add_argument("--trigger-collection", default="admin_jobs")
    ap.add_argument("--trigger-doc", default="publish_approved_observations")
    ap.add_argument("--require-trigger", action="store_true")
    ap.add_argument("--mark-trigger-processed", action="store_true")
    ap.add_argument("--bundle-path", default="../assets/data/plants_bundle.json")
    ap.add_argument("--rolling-window-days", type=int, default=1095)
    ap.add_argument(
        "--resolved-plants",
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "One or more plants_resolved_*.json files (flat list, pipeline format) "
            "for taxonKey enrichment. Merged with --bundle-path if both are present."
        ),
    )
    args = ap.parse_args()

    cred = credentials.Certificate(args.service_account)
    firebase_admin.initialize_app(cred)
    db = firestore.client(database_id=args.database_id)

    if args.require_trigger:
        trigger = _load_trigger(db, args.trigger_collection, args.trigger_doc)
        requested = bool(trigger and trigger.get("requested") is True)
        if not requested:
            print(
                "[skip] publish trigger not queued: "
                f"{args.trigger_collection}/{args.trigger_doc}",
                flush=True,
            )
            return 0

    statuses = _status_values(args.statuses)
    by_plant_id, by_name = _load_bundle_identity_index(args.bundle_path)
    if args.resolved_plants:
        res_by_id, res_by_name = _load_resolved_plants_index(args.resolved_plants)
        # Resolved files take precedence — they're the authoritative pipeline data.
        by_plant_id = {**by_plant_id, **res_by_id}
        by_name = {**by_name, **res_by_name}
    if by_plant_id or by_name:
        print(
            f"[info] plant identity index: {len(by_plant_id)} by id, "
            f"{len(by_name)} by name",
            flush=True,
        )
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.rolling_window_days)
    by_id: dict[str, dict[str, Any]] = {}
    for status in statuses:
        query = db.collection("observations").where("status", "==", status)
        for snap in query.stream():
            data = snap.to_dict() or {}
            data["_id"] = snap.id
            by_id[snap.id] = data

    observations: list[dict[str, Any]] = []
    for data in by_id.values():
        lat = _to_float(data.get("lat"))
        lon = _to_float(data.get("lon"))
        if lat is None or lon is None:
            continue
        plant_id = _non_empty(data.get("plantId"))
        plant_name = _non_empty(data.get("plantName"))
        if not plant_id and not plant_name:
            continue
        scientific_name = _non_empty(data.get("scientificName"))
        taxon_key = data.get("taxonKey")
        resolved = None
        if plant_id and plant_id in by_plant_id:
            resolved = by_plant_id[plant_id]
        elif scientific_name:
            resolved = by_name.get(_normalize_name(scientific_name) or "")
        elif plant_name:
            resolved = by_name.get(_normalize_name(plant_name) or "")
        if resolved is not None:
            if taxon_key is None:
                taxon_key = resolved.get("taxonKey")
            if scientific_name is None:
                scientific_name = _non_empty(resolved.get("scientificName"))

        observed_at = _to_iso(data.get("observedAt"))
        event_date = _to_iso(data.get("eventDate"))
        comparable_date = _to_datetime(observed_at) or _to_datetime(event_date)
        if comparable_date is not None and comparable_date < cutoff:
            continue

        created_by = data.get("createdBy") if isinstance(data.get("createdBy"), dict) else {}
        obs = {
            "sourceObservationId": _non_empty(data.get("_id")),
            "source": "wild_forager_approved",
            "plantId": plant_id,
            "plantName": plant_name,
            "scientificName": scientific_name,
            "taxonKey": taxon_key,
            "lat": lat,
            "lon": lon,
            "observedAt": observed_at,
            "eventDate": event_date,
            "status": _non_empty(data.get("status")) or "approved",
            "createdBy": {
                "uid": _non_empty(created_by.get("uid")),
                "displayName": _non_empty(created_by.get("displayName")),
                "email": _non_empty(created_by.get("email")),
            },
            "inlinePhotoBase64": _non_empty(data.get("inlinePhotoBase64")),
            "inlinePhotoContentType": _non_empty(data.get("inlinePhotoContentType")),
            "mediaImageUrl": _non_empty(data.get("mediaImageUrl")),
            "mediaSourcePageUrl": _non_empty(data.get("mediaSourcePageUrl")),
            "occurrenceReferenceUrl": _non_empty(data.get("occurrenceReferenceUrl")),
        }
        observations.append(_prune_none(obs))

    observations.sort(
        key=lambda row: (
            row.get("observedAt")
            or row.get("eventDate")
            or ""
        ),
        reverse=True,
    )

    payload = {
        "observations": observations,
        "meta": {
            "generated_at": _now_iso(),
            "source": "firestore-approved-export",
            "schema_version": 2,
            "count": len(observations),
            "statuses": statuses,
            "rolling_window_days": args.rolling_window_days,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"[ok] wrote {args.out} (observations={len(observations)})", flush=True)

    if args.require_trigger and args.mark_trigger_processed:
        _write_trigger_result(
            db,
            args.trigger_collection,
            args.trigger_doc,
            count=len(observations),
        )
        print(
            "[ok] marked trigger processed: "
            f"{args.trigger_collection}/{args.trigger_doc}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
