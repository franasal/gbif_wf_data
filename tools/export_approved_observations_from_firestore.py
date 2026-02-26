#!/usr/bin/env python3
import argparse
import gzip
import json
import os
from datetime import datetime, timezone
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

        created_by = data.get("createdBy") if isinstance(data.get("createdBy"), dict) else {}
        obs = {
            "id": _non_empty(data.get("_id")),
            "plantId": plant_id,
            "plantName": plant_name,
            "lat": lat,
            "lon": lon,
            "observedAt": _to_iso(data.get("observedAt")),
            "eventDate": _to_iso(data.get("eventDate")),
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
            "schema_version": 1,
            "count": len(observations),
            "statuses": statuses,
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
