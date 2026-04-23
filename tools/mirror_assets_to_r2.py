#!/usr/bin/env python3
import argparse
import mimetypes
import os
from pathlib import Path

import boto3


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _client():
    account_id = _env("R2_ACCOUNT_ID")
    access_key = _env("R2_ACCESS_KEY_ID")
    secret_key = _env("R2_SECRET_ACCESS_KEY")
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def _content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _upload_file(client, bucket: str, src: Path, dest_key: str, cache_control: str) -> None:
    extra_args = {
        "ContentType": _content_type_for(src),
        "CacheControl": cache_control,
    }
    client.upload_file(str(src), bucket, dest_key, ExtraArgs=extra_args)
    print(f"[ok] mirrored {src} -> s3://{bucket}/{dest_key}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mirror selected pipeline assets to Cloudflare R2.",
    )
    ap.add_argument(
        "--asset",
        nargs=2,
        metavar=("SRC", "DEST_KEY"),
        action="append",
        default=[],
        help="Local source file and destination object key inside the R2 bucket.",
    )
    ap.add_argument(
        "--cache-control",
        default="public, max-age=300",
        help="Cache-Control header for uploaded objects.",
    )
    args = ap.parse_args()

    if not args.asset:
        print("[skip] no assets requested for mirroring", flush=True)
        return 0

    bucket = _env("R2_BUCKET")
    client = _client()

    for src_raw, dest_key in args.asset:
        src = Path(src_raw)
        if not src.exists():
            print(f"[skip] missing asset: {src}", flush=True)
            continue
        _upload_file(client, bucket, src, dest_key, args.cache_control)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
