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

    ap.add_argument("--dataset", default=None, help="Legacy dataset asset path (compat)")
    ap.add_argument("--dataset-legacy", default="data/occurrences_compact.json.gz", help="Legacy dataset asset path")
    ap.add_argument("--dataset-edible", default="data/occurrences_compact_edible.json.gz", help="Edible dataset asset path")
    ap.add_argument("--dataset-poisonous", default="data/occurrences_compact_poisonous.json.gz", help="Poisonous dataset asset path")
    ap.add_argument("--recent-dataset", default=None, help="Optional recent raw legacy dataset asset path")
    ap.add_argument("--recent-dataset-edible", default=None, help="Optional recent raw edible dataset asset path")
    ap.add_argument("--recent-dataset-poisonous", default=None, help="Optional recent raw poisonous dataset asset path")
    ap.add_argument("--approved-observations", default=None, help="Optional approved community observations overlay asset path")
    ap.add_argument("--protected-areas", default=None, help="Optional protected areas asset path")
    ap.add_argument("--stats", default="data/stats_summary.json", help="Stats asset path (optional)")
    ap.add_argument("--pipeline-diagnostics", default=None, help="Optional pipeline diagnostics JSON asset path")
    ap.add_argument("--pipeline-point-explorer", default=None, help="Optional pipeline point explorer JSON asset path")
    ap.add_argument("--thumbs-pack", default=None, help="Optional thumbs pack zip (e.g. data/thumbs_pack_latest.zip)")
    ap.add_argument("--light-pack-edible", default="assets/plant_images/light_build/edible/gbif_light_edible.zip", help="Light pack (edible) zip path")
    ap.add_argument("--light-pack-poisonous", default="assets/plant_images/light_build/poisonous/gbif_light_poisonous.zip", help="Light pack (poisonous) zip path")

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

    legacy_path = Path(args.dataset) if args.dataset else Path(args.dataset_legacy)
    dataset_edible = Path(args.dataset_edible)
    dataset_poisonous = Path(args.dataset_poisonous)
    recent_legacy = Path(args.recent_dataset) if args.recent_dataset else None
    recent_edible = Path(args.recent_dataset_edible) if args.recent_dataset_edible else None
    recent_poisonous = Path(args.recent_dataset_poisonous) if args.recent_dataset_poisonous else None
    approved_observations = Path(args.approved_observations) if args.approved_observations else None
    protected_areas = Path(args.protected_areas) if args.protected_areas else None
    light_edible = Path(args.light_pack_edible) if args.light_pack_edible else None
    light_poisonous = Path(args.light_pack_poisonous) if args.light_pack_poisonous else None
    stats = Path(args.stats) if args.stats else None
    pipeline_diagnostics = Path(args.pipeline_diagnostics) if args.pipeline_diagnostics else None
    pipeline_point_explorer = Path(args.pipeline_point_explorer) if args.pipeline_point_explorer else None
    thumbs = Path(args.thumbs_pack) if args.thumbs_pack else None

    upload_asset(tag, legacy_path, clobber=clobber)
    upload_asset(tag, dataset_edible, clobber=clobber)
    upload_asset(tag, dataset_poisonous, clobber=clobber)
    if recent_legacy:
        upload_asset(tag, recent_legacy, clobber=clobber)
    if recent_edible:
        upload_asset(tag, recent_edible, clobber=clobber)
    if recent_poisonous:
        upload_asset(tag, recent_poisonous, clobber=clobber)
    if approved_observations:
        upload_asset(tag, approved_observations, clobber=clobber)
    if protected_areas:
        upload_asset(tag, protected_areas, clobber=clobber)
    if pipeline_diagnostics:
        upload_asset(tag, pipeline_diagnostics, clobber=clobber)
    if pipeline_point_explorer:
        upload_asset(tag, pipeline_point_explorer, clobber=clobber)
    if light_edible:
        upload_asset(tag, light_edible, clobber=clobber)
    if light_poisonous:
        upload_asset(tag, light_poisonous, clobber=clobber)
    if stats:
        upload_asset(tag, stats, clobber=clobber)
    if thumbs:
        upload_asset(tag, thumbs, clobber=clobber)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
