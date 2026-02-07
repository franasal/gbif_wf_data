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


def _publish_to_firebase(
    payload: Dict[str, Any],
    csv_path: Path,
    json_path: Path,
    project_id: str,
    bucket_name: str,
    collection: str,
    latest_doc_id: str,
    storage_prefix: str,
) -> None:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'firebase-admin'. Install with: pip install firebase-admin"
        ) from exc

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if service_account_json:
        cred = credentials.Certificate(json.loads(service_account_json))
    else:
        cred = credentials.ApplicationDefault()

    app = firebase_admin.initialize_app(
        cred,
        {"projectId": project_id, "storageBucket": bucket_name},
    )

    version = payload["version"]
    prefix = storage_prefix.strip("/")
    csv_obj = f"{prefix}/{version}/image_license_inventory.csv"
    json_obj = f"{prefix}/{version}/image_license_inventory.json"

    bucket = storage.bucket(app=app)
    bucket.blob(csv_obj).upload_from_filename(str(csv_path), content_type="text/csv")
    bucket.blob(json_obj).upload_from_filename(str(json_path), content_type="application/json")

    db = firestore.client(app=app)
    meta = {
        "version": version,
        "generatedAt": payload["generatedAt"],
        "rowCount": payload["rowCount"],
        "stats": payload["stats"],
        "sourceFiles": payload["sourceFiles"],
        "artifacts": {
            "csv": {"storagePath": csv_obj, "sha256": payload["artifacts"]["csvSha256"]},
            "json": {"storagePath": json_obj, "sha256": payload["artifacts"]["jsonSha256"]},
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
        default="assets/plant_images/light_build/images_manifest.json",
    )
    parser.add_argument(
        "--out-csv",
        default="data/legal/image_license_inventory.csv",
    )
    parser.add_argument(
        "--out-json",
        default="data/legal/image_license_inventory.json",
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
    parser.add_argument("--bucket", default=os.getenv("FIREBASE_STORAGE_BUCKET", ""))
    parser.add_argument(
        "--collection",
        default=os.getenv("LEGAL_INVENTORY_COLLECTION", "legal_inventory_versions"),
    )
    parser.add_argument(
        "--latest-doc-id",
        default=os.getenv("LEGAL_INVENTORY_LATEST_DOC_ID", "latest"),
    )
    parser.add_argument(
        "--storage-prefix",
        default=os.getenv("LEGAL_INVENTORY_STORAGE_PREFIX", "legal/image-license-inventory"),
    )
    args = parser.parse_args()

    gbif_index = Path(args.gbif_index_json)
    light_manifest = Path(args.light_manifest_json)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    if not gbif_index.is_file():
        raise SystemExit(f"Missing file: {gbif_index}")
    if not light_manifest.is_file():
        raise SystemExit(f"Missing file: {light_manifest}")

    rows = _rows_from_gbif_samples(gbif_index) + _rows_from_light_manifest(light_manifest)
    _write_csv(out_csv, rows)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    required_types = {s.strip() for s in args.required_source_type if s.strip()}
    payload: Dict[str, Any] = {
        "version": args.version,
        "generatedAt": generated_at,
        "rowCount": len(rows),
        "stats": _stats(rows, required_types),
        "sourceFiles": {
            "gbifIndexJson": str(gbif_index),
            "lightManifestJson": str(light_manifest),
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

    print(
        "Built inventory:",
        f"rows={payload['rowCount']}",
        f"csv={out_csv}",
        f"json={out_json}",
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

    if not args.project_id:
        raise SystemExit("Missing --project-id or FIREBASE_PROJECT_ID")
    if not args.bucket:
        raise SystemExit("Missing --bucket or FIREBASE_STORAGE_BUCKET")

    _publish_to_firebase(
        payload=payload,
        csv_path=out_csv,
        json_path=out_json,
        project_id=args.project_id,
        bucket_name=args.bucket,
        collection=args.collection,
        latest_doc_id=args.latest_doc_id,
        storage_prefix=args.storage_prefix,
    )
    print(
        "Published to Firebase:",
        f"project={args.project_id}",
        f"bucket={args.bucket}",
        f"collection={args.collection}",
        f"version={args.version}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
