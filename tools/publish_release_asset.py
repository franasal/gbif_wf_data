#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_gh() -> None:
    try:
        subprocess.check_output(["gh", "--version"])
    except Exception:
        raise SystemExit("GitHub CLI 'gh' not found. In Actions use: apt-get install gh OR use actions/setup-gh (or just rely on ubuntu-latest which usually has gh).")


def require_token() -> None:
    # gh uses GH_TOKEN; it can also use GITHUB_TOKEN in some setups,
    # but GH_TOKEN is the official env for gh auth in Actions.
    if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        raise SystemExit("Missing auth token. Set env GH_TOKEN: ${{ secrets.GITHUB_TOKEN }} (or a PAT).")


def release_exists(tag: str) -> bool:
    try:
        subprocess.check_output(["gh", "release", "view", tag], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False


def create_release(tag: str, title: str, notes: str, target: str | None) -> None:
    cmd = ["gh", "release", "create", tag, "--title", title, "--notes", notes]
    if target:
        cmd += ["--target", target]
    run(cmd)


def upload_asset(tag: str, path: Path, clobber: bool = True) -> None:
    if not path.exists():
        print(f"[skip] missing asset: {path}", flush=True)
        return
    cmd = ["gh", "release", "upload", tag, str(path)]
    if clobber:
        cmd += ["--clobber"]
    run(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create/update a GitHub Release and upload pipeline assets.")
    ap.add_argument("--tag", default="latest", help="Release tag to create/update (default: latest)")
    ap.add_argument("--title", default="Latest GBIF export", help="Release title")
    ap.add_argument("--notes", default="Auto-generated data export assets.", help="Release notes")
    ap.add_argument("--target", default=None, help="Target commit/branch (optional)")

    ap.add_argument("--dataset", default="data/occurrences_compact.json.gz", help="Dataset asset path")
    ap.add_argument("--stats", default="data/stats_summary.json", help="Stats asset path (optional)")
    ap.add_argument("--thumbs-pack", default=None, help="Optional thumbs pack zip (e.g. data/thumbs_pack_latest.zip)")

    ap.add_argument("--no-clobber", action="store_true", help="Do not replace existing assets")
    args = ap.parse_args()

    ensure_gh()
    require_token()

    tag = args.tag
    clobber = not args.no_clobber

    if not release_exists(tag):
        create_release(tag, args.title, args.notes, args.target)
    else:
        print(f"Release '{tag}' exists, updating assets.", flush=True)

    dataset = Path(args.dataset)
    stats = Path(args.stats) if args.stats else None
    thumbs = Path(args.thumbs_pack) if args.thumbs_pack else None

    upload_asset(tag, dataset, clobber=clobber)
    if stats:
        upload_asset(tag, stats, clobber=clobber)
    if thumbs:
        upload_asset(tag, thumbs, clobber=clobber)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
