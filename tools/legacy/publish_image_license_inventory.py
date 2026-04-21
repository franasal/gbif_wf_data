#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


CSV_FIELDS = [
    "manifest",
    "source_type",
    "plant_name",
    "asset_path",
    "image_url",
    "source_page_url",
    "license_label",
    "license_url",
    "creator",
    "rights_holder",
    "attribution_text",
    "publisher",
    "occurrence_key",
    "dataset_key",
]


TODO_FIELDS = [
    "source_type",
    "plant_name",
    "asset_path",
    "image_url",
    "source_page_url",
    "license_label",
    "license_url",
    "creator",
    "rights_holder",
    "attribution_text",
    "occurrence_key",
    "dataset_key",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_gbif_samples(path: Path) -> List[Dict[str, str]]:
    data = _load_json(path)
    downloads = data.get("downloads")
    if not isinstance(downloads, list):
        raise ValueError(f"{path}: expected 'downloads' list")

    rows: List[Dict[str, str]] = []
    for item in downloads:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "manifest": str(path),
                "source_type": "gbif_samples",
                "plant_name": _clean(item.get("plant_scientific") or item.get("scientificName")),
                "asset_path": _clean(item.get("local_path")),
                "image_url": _clean(item.get("media_identifier")),
                "source_page_url": _clean(item.get("media_references")),
                "license_label": _clean(item.get("media_license_label")),
                "license_url": _clean(item.get("media_license_url")),
                "creator": _clean(item.get("media_creator")),
                "rights_holder": _clean(item.get("media_rightsHolder")),
                "attribution_text": _clean(item.get("attribution_suggestion")),
                "publisher": _clean(item.get("publisher")),
                "occurrence_key": _clean(item.get("occurrenceKey")),
                "dataset_key": _clean(item.get("datasetKey")),
            }
        )
    return rows


def _rows_from_light_manifest(path: Path) -> List[Dict[str, str]]:
    data = _load_json(path)
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{path}: expected 'images' list")

    rows: List[Dict[str, str]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "manifest": str(path),
                "source_type": "light_manifest",
                "plant_name": _clean(item.get("plant_slug_raw") or item.get("plant_slug")),
                "asset_path": _clean(item.get("output_path") or item.get("source_path")),
                "image_url": _clean(item.get("media_identifier")),
                "source_page_url": _clean(item.get("media_references")),
                "license_label": _clean(item.get("media_license_label")),
                "license_url": _clean(item.get("media_license_url")),
                "creator": _clean(item.get("media_creator")),
                "rights_holder": _clean(item.get("media_rightsHolder")),
                "attribution_text": _clean(item.get("attribution_suggestion")),
                "publisher": "",
                "occurrence_key": _clean(item.get("occurrenceKey")),
                "dataset_key": "",
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stats(rows: List[Dict[str, str]], required_types: set[str]) -> Dict[str, Any]:
    by_source_type: Dict[str, int] = {}
    missing_license_count = 0
    missing_license_required_count = 0
    missing_source_page_count = 0
    missing_creator_count = 0

    for row in rows:
        source_type = row.get("source_type", "unknown")
        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
        missing_license = not row.get("license_label") and not row.get("license_url")
        if missing_license:
            missing_license_count += 1
            if source_type in required_types:
                missing_license_required_count += 1
        if not row.get("source_page_url"):
            missing_source_page_count += 1
        if not row.get("creator"):
            missing_creator_count += 1

    return {
        "bySourceType": by_source_type,
        "missingLicenseCount": missing_license_count,
        "missingLicenseRequiredCount": missing_license_required_count,
        "missingSourcePageCount": missing_source_page_count,
        "missingCreatorCount": missing_creator_count,
    }


def _todo_entry(row: Dict[str, str]) -> Dict[str, str]:
    return {k: row.get(k, "") for k in TODO_FIELDS}


def _counts_by_source_type(items: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in items:
        source_type = row.get("source_type", "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _build_todo_report(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    missing_license = []
    missing_source_page = []
    missing_creator = []
    for row in rows:
        if not row.get("license_label") and not row.get("license_url"):
            missing_license.append(_todo_entry(row))
        if not row.get("source_page_url"):
            missing_source_page.append(_todo_entry(row))
        if not row.get("creator"):
            missing_creator.append(_todo_entry(row))

    return {
        "summary": {
            "missingLicenseCount": len(missing_license),
            "missingSourcePageCount": len(missing_source_page),
            "missingCreatorCount": len(missing_creator),
            "missingLicenseBySourceType": _counts_by_source_type(missing_license),
            "missingSourcePageBySourceType": _counts_by_source_type(missing_source_page),
            "missingCreatorBySourceType": _counts_by_source_type(missing_creator),
        },
        "missingLicense": missing_license,
        "missingSourcePage": missing_source_page,
        "missingCreator": missing_creator,
    }


def _normalize_collection_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if not normalized:
        raise SystemExit("Missing collection path: set --collection or LEGAL_INVENTORY_COLLECTION")
    parts = [p for p in normalized.split("/") if p]
    # Firestore collection paths must have an odd number of path elements.
    if len(parts) % 2 == 0:
        raise SystemExit(
            f"Invalid collection path '{path}'. It must point to a collection "
            "(odd path segments), e.g. 'legal_inventory_versions' or "
            "'apps/wild_forager/legal_inventory_versions'."
        )
    return "/".join(parts)


def _normalize_doc_id(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise SystemExit(f"Missing {label}.")
    if "/" in normalized:
        raise SystemExit(f"Invalid {label} '{value}': document IDs must not contain '/'.")
    return normalized


def _normalize_project_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise SystemExit("Missing --project-id or FIREBASE_PROJECT_ID")
    if any(ch.isspace() for ch in normalized):
        raise SystemExit(
            "Invalid project id: contains whitespace. Check FIREBASE_PROJECT_ID secret for newlines/spaces."
        )
    if "/" in normalized:
        raise SystemExit(
            "Invalid project id: expected plain project id (e.g. 'wild-forager-8159c'), not a path."
        )
    return normalized


def _normalize_firestore_database_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    if any(ch.isspace() for ch in normalized):
        raise SystemExit(
            "Invalid FIRESTORE_DATABASE_ID: contains whitespace. Check secret for newlines/spaces."
        )
    if normalized.startswith("projects/") and "/databases/" in normalized:
        normalized = normalized.split("/databases/", 1)[1]
    if "/" in normalized:
        raise SystemExit(
            "Invalid FIRESTORE_DATABASE_ID: expected database id only "
            "(e.g. 'wild--forager-db' or '(default)')."
        )
    return normalized


def _publish_to_firebase(
    payload: Dict[str, Any],
    project_id: str,
    firestore_database_id: str,
    collection: str,
    latest_doc_id: str,
) -> None:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'firebase-admin'. Install with: pip install firebase-admin"
        ) from exc

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if service_account_json:
        cred = credentials.Certificate(json.loads(service_account_json))
    else:
        cred = credentials.ApplicationDefault()

    app = firebase_admin.initialize_app(cred, {"projectId": project_id})

    version = payload["version"]
    db = (
        firestore.client(app=app)
        if not firestore_database_id
        else firestore.client(app=app, database_id=firestore_database_id)
    )
    meta = {
        "version": version,
        "generatedAt": payload["generatedAt"],
        "rowCount": payload["rowCount"],
        "stats": payload["stats"],
        "sourceFiles": payload["sourceFiles"],
        "artifacts": {
            "csv": {
                "path": payload["artifacts"]["csvPath"],
                "sha256": payload["artifacts"]["csvSha256"],
            },
            "json": {
                "path": payload["artifacts"]["jsonPath"],
                "sha256": payload["artifacts"]["jsonSha256"],
            },
            "todo": {
                "path": payload["artifacts"]["todoPath"],
                "sha256": payload["artifacts"]["todoSha256"],
            },
        },
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    db.collection(collection).document(version).set(meta)
    db.collection(collection).document(latest_doc_id).set(meta)


def _default_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build image license inventory and optionally publish to Firebase."
    )
    parser.add_argument(
        "--gbif-index-json",
        default="assets/plant_images/gbif_samples/index.json",
    )
    parser.add_argument(
        "--light-manifest-json",
        action="append",
        default=[],
        help=(
            "Optional light manifest JSON path. Repeat for multiple manifests. "
            "If omitted, the script uses the split curated manifests when present."
        ),
    )
    parser.add_argument(
        "--out-csv",
        default="data/legal/image_license_inventory.csv",
    )
    parser.add_argument(
        "--out-json",
        default="data/legal/image_license_inventory.json",
    )
    parser.add_argument(
        "--out-todo-json",
        default="data/legal/image_license_todo.json",
    )
    parser.add_argument("--version", default=_default_version())
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--fail-on-missing-license", action="store_true")
    parser.add_argument(
        "--required-source-type",
        action="append",
        default=["gbif_samples"],
        help="Source type(s) to enforce license gate on (repeatable).",
    )
    parser.add_argument("--project-id", default=os.getenv("FIREBASE_PROJECT_ID", ""))
    parser.add_argument(
        "--firestore-database-id",
        default=os.getenv("FIRESTORE_DATABASE_ID", ""),
        help="Firestore database ID. Leave empty to use default database.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("LEGAL_INVENTORY_COLLECTION", "legal_inventory_versions"),
    )
    parser.add_argument(
        "--latest-doc-id",
        default=os.getenv("LEGAL_INVENTORY_LATEST_DOC_ID", "latest"),
    )
    args = parser.parse_args()

    gbif_index = Path(args.gbif_index_json)
    light_manifest_args = [Path(p) for p in (args.light_manifest_json or []) if str(p).strip()]
    default_light_manifests = [
        Path("assets/plant_images/light_build/edible/images_manifest.json"),
        Path("assets/plant_images/light_build/poisonous/images_manifest.json"),
    ]
    light_manifests = light_manifest_args or [p for p in default_light_manifests if p.is_file()]
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_todo_json = Path(args.out_todo_json)
    project_id = _normalize_project_id(args.project_id)
    firestore_database_id = _normalize_firestore_database_id(args.firestore_database_id)
    collection = _normalize_collection_path(args.collection)
    latest_doc_id = _normalize_doc_id(args.latest_doc_id, "latest doc id")
    version = _normalize_doc_id(args.version, "version")

    if not gbif_index.is_file():
        raise SystemExit(f"Missing file: {gbif_index}")

    missing_light = [str(path) for path in light_manifest_args if not path.is_file()]
    if missing_light:
        raise SystemExit(f"Missing file(s): {', '.join(missing_light)}")

    rows = _rows_from_gbif_samples(gbif_index)
    for light_manifest in light_manifests:
        rows += _rows_from_light_manifest(light_manifest)
    _write_csv(out_csv, rows)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    required_types = {s.strip() for s in args.required_source_type if s.strip()}
    payload: Dict[str, Any] = {
        "version": version,
        "generatedAt": generated_at,
        "rowCount": len(rows),
        "stats": _stats(rows, required_types),
        "sourceFiles": {
            "gbifIndexJson": str(gbif_index),
            "lightManifestJsons": [str(path) for path in light_manifests],
        },
        "artifacts": {
            "csvPath": str(out_csv),
            "csvSha256": _sha256(out_csv),
        },
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["artifacts"]["jsonPath"] = str(out_json)
    payload["artifacts"]["jsonSha256"] = _sha256(out_json)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    todo = _build_todo_report(rows)
    out_todo_json.parent.mkdir(parents=True, exist_ok=True)
    out_todo_json.write_text(json.dumps(todo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["artifacts"]["todoPath"] = str(out_todo_json)
    payload["artifacts"]["todoSha256"] = _sha256(out_todo_json)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "Built inventory:",
        f"rows={payload['rowCount']}",
        f"csv={out_csv}",
        f"json={out_json}",
        f"todo={out_todo_json}",
    )
    print(
        "Quality:",
        f"missingLicense={payload['stats']['missingLicenseCount']}",
        f"missingLicenseRequired={payload['stats']['missingLicenseRequiredCount']}",
        f"missingSourcePage={payload['stats']['missingSourcePageCount']}",
        f"missingCreator={payload['stats']['missingCreatorCount']}",
    )

    if args.fail_on_missing_license and payload["stats"]["missingLicenseRequiredCount"] > 0:
        print(
            "Failing due to required source types missing license:",
            payload["stats"]["missingLicenseRequiredCount"],
        )
        return 2

    if not args.publish:
        return 0

    _publish_to_firebase(
        payload=payload,
        project_id=project_id,
        firestore_database_id=firestore_database_id,
        collection=collection,
        latest_doc_id=latest_doc_id,
    )
    print(
        "Published to Firebase:",
        f"project={project_id}",
        f"collection={collection}",
        f"version={version}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
