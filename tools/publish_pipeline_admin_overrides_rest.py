#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import google.auth
from google.auth.transport.requests import AuthorizedSession, Request


def _firestore_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_firestore_value(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "mapValue": {
                "fields": {str(k): _firestore_value(v) for k, v in value.items()}
            }
        }
    return {"stringValue": str(value)}


def _document_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _firestore_value(v) for k, v in data.items()}


def _patch_doc(
    session: AuthorizedSession,
    *,
    project_id: str,
    database_id: str,
    collection: str,
    document_id: str,
    data: dict[str, Any],
) -> None:
    url = (
        f"https://firestore.googleapis.com/v1/projects/{project_id}"
        f"/databases/{database_id}/documents/{collection}/{document_id}"
    )
    resp = session.patch(url, json={"fields": _document_fields(data)}, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Firestore write failed for {collection}/{document_id}: {resp.status_code} {resp.text[:500]}")


def _run_doc_id(generated_at: Any) -> str:
    return str(generated_at or "unknown").replace(":", "-").replace(".", "-")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish pipeline diagnostics and point explorer payloads to Firestore via REST/ADC.",
    )
    ap.add_argument("--project-id", default="wild-forager-8159c")
    ap.add_argument("--database-id", default="wild--forager-db")
    ap.add_argument("--diagnostics", required=True)
    ap.add_argument("--point-explorer", required=True)
    args = ap.parse_args()

    diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    point_explorer = json.loads(Path(args.point_explorer).read_text(encoding="utf-8"))

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    session = AuthorizedSession(creds)

    _patch_doc(
        session,
        project_id=args.project_id,
        database_id=args.database_id,
        collection="admin_overrides",
        document_id="pipeline_diagnostics_summary",
        data={
            "payload": diagnostics,
            "source": "gbif_daily_pipeline_rest",
        },
    )
    _patch_doc(
        session,
        project_id=args.project_id,
        database_id=args.database_id,
        collection="admin_pipeline_runs",
        document_id=_run_doc_id(diagnostics.get("generatedAt")),
        data={
            "generatedAt": diagnostics.get("generatedAt"),
            "window": diagnostics.get("window"),
            "totals": diagnostics.get("totals"),
            "changes": diagnostics.get("changes"),
            "source": "gbif_daily_pipeline_rest",
        },
    )

    details = point_explorer.get("details")
    if not isinstance(details, dict):
        details = {}
    index = point_explorer.get("index")
    if not isinstance(index, list):
        index = []
    _patch_doc(
        session,
        project_id=args.project_id,
        database_id=args.database_id,
        collection="admin_overrides",
        document_id="pipeline_point_explorer_index",
        data={
            "payload": {
                "generatedAt": point_explorer.get("generatedAt"),
                "plantCount": point_explorer.get("plantCount"),
                "index": index,
            },
            "source": "gbif_daily_pipeline_rest",
        },
    )

    written = 0
    for document_id, payload in details.items():
        if not isinstance(payload, dict):
            continue
        _patch_doc(
            session,
            project_id=args.project_id,
            database_id=args.database_id,
            collection="admin_overrides",
            document_id=document_id,
            data={
                "payload": payload,
                "source": "gbif_daily_pipeline_rest",
            },
        )
        written += 1

    print(
        f"[ok] published diagnostics, point explorer index, and {written} detail docs",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
