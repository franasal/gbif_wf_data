#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish pipeline diagnostics summary to Firestore admin_overrides.",
    )
    ap.add_argument("--service-account", required=True)
    ap.add_argument("--database-id", default="(default)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--document-id", default="pipeline_diagnostics_summary")
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cred = credentials.Certificate(args.service_account)
    firebase_admin.initialize_app(cred)
    db = firestore.client(database_id=args.database_id)
    db.collection("admin_overrides").document(args.document_id).set(
        {
            "payload": payload,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "source": "gbif_daily_pipeline",
        },
        merge=True,
    )
    print(
        f"[ok] published pipeline diagnostics to admin_overrides/{args.document_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
