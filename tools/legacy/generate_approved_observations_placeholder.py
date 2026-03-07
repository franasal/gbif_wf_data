#!/usr/bin/env python3
import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a placeholder approved_observations overlay asset when backend export is not ready."
    )
    ap.add_argument(
        "--out",
        default="data/approved_observations.json.gz",
        help="Output gzip JSON path",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file (default: skip if file exists)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[skip] exists: {out}")
        return 0

    payload = {
        "observations": [],
        "meta": {
            "generated_at": utc_now_iso(),
            "source": "placeholder",
            "status": "backend_export_not_configured",
            "schema_version": 1,
            "observation_schema": [
                "id",
                "plantId",
                "plantName",
                "lat",
                "lon",
                "observedAt",
                "status",
                "createdBy.displayName",
                "inlinePhotoBase64",
                "inlinePhotoContentType",
                "mediaImageUrl",
                "mediaSourcePageUrl",
                "occurrenceReferenceUrl",
            ],
            "todo": "Replace placeholder by approved observations export from moderation backend.",
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote placeholder: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
