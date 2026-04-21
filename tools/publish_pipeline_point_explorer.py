#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish per-plant point explorer payloads to Firestore admin_overrides.",
    )
    ap.add_argument("--service-account", required=True)
    ap.add_argument("--database-id", default="(default)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--index-document-id", default="pipeline_point_explorer_index")
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cred = credentials.Certificate(args.service_account)
    firebase_admin.initialize_app(cred)
    db = firestore.client(database_id=args.database_id)

    index = payload.get("index", [])
    details = payload.get("details", {})
    db.collection("admin_overrides").document(args.index_document_id).set(
        {
            "payload": {
                "generatedAt": payload.get("generatedAt"),
                "plantCount": payload.get("plantCount"),
                "index": index,
            },
            "source": "gbif_daily_pipeline",
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    written = 0
    for doc_id, detail_payload in details.items():
      db.collection("admin_overrides").document(doc_id).set(
          {
              "payload": detail_payload,
              "source": "gbif_daily_pipeline",
              "updatedAt": firestore.SERVER_TIMESTAMP,
          },
          merge=True,
      )
      written += 1

    print(
        f"[ok] published point explorer index and {written} plant detail docs",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
