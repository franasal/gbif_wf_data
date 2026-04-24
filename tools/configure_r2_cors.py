#!/usr/bin/env python3
import argparse
import os

import boto3
from botocore.config import Config


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _normalize_account_id(raw: str) -> str:
    value = raw.strip()
    value = value.removeprefix("https://").removeprefix("http://")
    value = value.removesuffix(".r2.cloudflarestorage.com")
    value = value.strip("/")
    if not value:
        raise SystemExit("R2 account id resolved to an empty value.")
    return value


def _r2_client():
    account_id = _normalize_account_id(
        os.environ.get("R2_ACCOUNT_ID")
        or os.environ.get("WF_R2_ACCOUNT_ID")
        or "",
    )
    access_key = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get(
        "WF_R2_ACCESS_KEY_ID"
    )
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get(
        "WF_R2_SECRET_ACCESS_KEY"
    )
    if not access_key:
        raise SystemExit("Missing required env var: R2_ACCESS_KEY_ID or WF_R2_ACCESS_KEY_ID")
    if not secret_key:
        raise SystemExit(
            "Missing required env var: R2_SECRET_ACCESS_KEY or WF_R2_SECRET_ACCESS_KEY"
        )
    endpoint = os.environ.get("R2_ENDPOINT", "").strip()
    if not endpoint:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint.rstrip("/"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Configure CORS for the public browser-facing R2 asset bucket.",
    )
    ap.add_argument(
        "--bucket",
        default=os.environ.get("R2_BUCKET") or os.environ.get("WF_R2_BUCKET") or "",
        help="R2 bucket name. Defaults to R2_BUCKET or WF_R2_BUCKET.",
    )
    ap.add_argument(
        "--origin",
        action="append",
        default=[],
        help="Allowed browser origin. Repeat for multiple origins. Defaults to '*'.",
    )
    args = ap.parse_args()

    bucket = args.bucket.strip()
    if not bucket:
        bucket = _env("R2_BUCKET")
    origins = args.origin or ["*"]

    cors = {
        "CORSRules": [
            {
                "AllowedOrigins": origins,
                "AllowedMethods": ["GET", "HEAD"],
                "AllowedHeaders": ["*"],
                "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
                "MaxAgeSeconds": 86400,
            }
        ]
    }
    _r2_client().put_bucket_cors(Bucket=bucket, CORSConfiguration=cors)
    print(
        f"[ok] configured CORS for bucket {bucket}: origins={', '.join(origins)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
