#!/usr/bin/env python3
"""
Backfill missing copyright/license metadata in light manifests by querying GBIF.

This replaces the old gbif_samples index backfill. It operates directly on the
edible/poisonous light manifests and fills media_* fields for entries missing
license data.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import requests

BASE = "https://api.gbif.org/v1"
UA = "wild-forager-light-backfill/1.0"

COMMERCIAL_LICENSES = {
    "cc0-1.0": "CC0 1.0",
    "cc-by-4.0": "CC BY 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-3.0": "CC BY 3.0",
    "cc-by-sa-3.0": "CC BY-SA 3.0",
}


def hash_identifier(identifier: str) -> str:
    return hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:16]


def normalize_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        # Drop query/fragment to stabilize identifiers that add tracking params.
        cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        cleaned = raw
    if cleaned.endswith("/") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned


def candidate_hashes(value: str) -> List[str]:
    if not value:
        return []
    candidates = {value}
    norm = normalize_url(value)
    if norm:
        candidates.add(norm)
    # Swap http/https variants to handle scheme changes in GBIF.
    if value.startswith("http://"):
        candidates.add("https://" + value[len("http://") :])
    if value.startswith("https://"):
        candidates.add("http://" + value[len("https://") :])
    if norm.startswith("http://"):
        candidates.add("https://" + norm[len("http://") :])
    if norm.startswith("https://"):
        candidates.add("http://" + norm[len("https://") :])
    return [hash_identifier(c) for c in candidates if c]


def parse_filename(path: str) -> Optional[Tuple[str, str]]:
    """Extract (occurrenceKey, identifier_hash) from a filename stem."""
    stem = Path(path).stem
    parts = stem.split("__")
    if len(parts) < 4:
        return None
    occ_key = parts[-3]
    id_hash = parts[-2]
    return (occ_key, id_hash)


def normalize_license_to_key(lic: Any) -> str:
    if not lic:
        return ""
    s = str(lic).strip().lower()

    if "publicdomain/zero/1.0" in s or "cc0" in s:
        return "cc0-1.0"

    if "creativecommons.org/licenses/by/" in s:
        if "/4.0" in s:
            return "cc-by-4.0"
        if "/3.0" in s:
            return "cc-by-3.0"

    if "creativecommons.org/licenses/by-sa/" in s:
        if "/4.0" in s:
            return "cc-by-sa-4.0"
        if "/3.0" in s:
            return "cc-by-sa-3.0"

    if s in COMMERCIAL_LICENSES:
        return s

    return ""


def guess_attribution(occ: Dict[str, Any], media_obj: Dict[str, Any], license_label: str) -> str:
    creator = media_obj.get("creator") or ""
    rights = media_obj.get("rightsHolder") or ""
    source = media_obj.get("references") or media_obj.get("identifier") or ""
    dataset_key = occ.get("datasetKey") or ""
    occ_key = occ.get("key") or ""

    parts = []
    if creator:
        parts.append(f"Photo: {creator}")
    elif rights:
        parts.append(f"Photo: {rights}")
    else:
        parts.append("Photo: (creator not provided)")

    if license_label:
        parts.append(f"License: {license_label}")
    if source:
        parts.append(f"Source: {source}")
    if dataset_key:
        parts.append(f"GBIF dataset: {dataset_key}")
    if occ_key:
        parts.append(f"Occurrence: {occ_key}")
    return " | ".join(parts)


def http_get_occurrence(session: requests.Session, occ_key: str, timeout: int) -> Dict[str, Any]:
    url = f"{BASE}/occurrence/{occ_key}"
    headers = {"User-Agent": UA}
    try:
        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code == 404:
            return {}
        if r.status_code == 429 or r.status_code >= 500:
            raise RuntimeError(f"GBIF error {r.status_code}")
        r.raise_for_status()
        return r.json()
    except Exception:
        try:
            result = subprocess.run(
                ["curl", "-L", "-s", "--fail-with-body", "-A", UA, url],
                check=True,
                capture_output=True,
                text=True,
            )
            stdout = result.stdout.strip()
            if not stdout:
                return {}
            payload = json.loads(stdout)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise SystemExit(f"Invalid manifest: {path}")
    return data


def needs_backfill(item: Dict[str, Any], include_missing_creator: bool) -> bool:
    missing_license = not (item.get("media_license_label") or item.get("media_license_url"))
    missing_creator = include_missing_creator and not item.get("media_creator")
    return missing_license or missing_creator


def iter_targets(manifest: Dict[str, Any], include_missing_creator: bool) -> Iterable[Tuple[Dict[str, Any], str, str]]:
    for item in manifest["images"]:
        if not isinstance(item, dict):
            continue
        if not needs_backfill(item, include_missing_creator):
            continue
        # Prefer output_path/output_rel/source_path to parse the hash.
        src = (
            item.get("output_path")
            or item.get("output_rel")
            or item.get("source_path")
            or ""
        )
        parsed = parse_filename(str(src))
        if not parsed:
            continue
        occ_key, id_hash = parsed
        if not occ_key or not id_hash:
            continue
        # Prefer explicit occurrenceKey if present.
        occ_key = str(item.get("occurrenceKey") or occ_key).strip()
        if not occ_key:
            continue
        yield (item, occ_key, id_hash)


def apply_media(item: Dict[str, Any], occ: Dict[str, Any], media_obj: Dict[str, Any]) -> bool:
    lic_key = normalize_license_to_key(media_obj.get("license"))
    lic_label = COMMERCIAL_LICENSES.get(lic_key, "") if lic_key else ""

    item["media_identifier"] = media_obj.get("identifier") or item.get("media_identifier", "")
    item["media_license_key"] = lic_key
    item["media_license_label"] = lic_label
    item["media_license_url"] = media_obj.get("license") or ""
    item["media_creator"] = media_obj.get("creator") or ""
    item["media_rightsHolder"] = media_obj.get("rightsHolder") or ""
    item["media_references"] = media_obj.get("references") or media_obj.get("identifier") or ""
    item["attribution_suggestion"] = guess_attribution(occ, media_obj, lic_label)
    return True


def backfill_manifest(
    manifest: Dict[str, Any],
    session: requests.Session,
    max_requests: int,
    sleep_s: float,
    timeout_s: int,
    log_every: int,
    include_missing_creator: bool,
) -> Dict[str, int]:
    targets: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
    total_targets = 0
    for item, occ_key, id_hash in iter_targets(manifest, include_missing_creator):
        targets.setdefault(occ_key, []).append((item, id_hash))
        total_targets += 1

    stats = {
        "targets": total_targets,
        "occurrences": len(targets),
        "requests": 0,
        "matched": 0,
        "skipped": 0,
        "updated": 0,
    }

    for idx, (occ_key, items) in enumerate(sorted(targets.items()), start=1):
        if stats["requests"] >= max_requests:
            break
        try:
            stats["requests"] += 1
            occ = http_get_occurrence(session, occ_key, timeout_s)
        except Exception:
            stats["skipped"] += len(items)
            if log_every and stats["requests"] % log_every == 0:
                print(f"[{idx}/{len(targets)}] {occ_key} -> error", flush=True)
            time.sleep(sleep_s)
            continue

        media_list = occ.get("media") or []
        if not isinstance(media_list, list) or not media_list:
            stats["skipped"] += len(items)
            time.sleep(sleep_s)
            continue

        media_by_hash: Dict[str, Dict[str, Any]] = {}
        for m in media_list:
            if not isinstance(m, dict):
                continue
            identifier = (m.get("identifier") or "").strip()
            references = (m.get("references") or "").strip()
            for h in candidate_hashes(identifier):
                media_by_hash.setdefault(h, m)
            for h in candidate_hashes(references):
                media_by_hash.setdefault(h, m)

        for item, id_hash in items:
            m = media_by_hash.get(id_hash)
            if not m:
                stats["skipped"] += 1
                continue
            if apply_media(item, occ, m):
                stats["matched"] += 1
                stats["updated"] += 1

        if log_every and stats["requests"] % log_every == 0:
            print(
                f"Progress: requests={stats['requests']} matched={stats['matched']} skipped={stats['skipped']}",
                flush=True,
            )

        time.sleep(sleep_s)

    return stats


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest_csv(path: Path, manifest: Dict[str, Any]) -> None:
    images = manifest.get("images") or []
    if not isinstance(images, list) or not images:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for image in images if isinstance(image, dict) for key in image.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image in images:
            if isinstance(image, dict):
                writer.writerow(image)


def refresh_zip(zip_path: Path, manifest_json: Path, manifest_csv: Path) -> None:
    if not zip_path.exists():
        return
    tmp_zip = zip_path.with_suffix(f"{zip_path.suffix}.tmp")
    with ZipFile(zip_path, "r") as src_zip, ZipFile(
        tmp_zip,
        "w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as dst_zip:
        for info in src_zip.infolist():
            if info.filename in {"images_manifest.json", "images_manifest.csv"}:
                continue
            dst_zip.writestr(info, src_zip.read(info.filename))
        dst_zip.write(manifest_json, arcname="images_manifest.json")
        dst_zip.write(manifest_csv, arcname="images_manifest.csv")
    tmp_zip.replace(zip_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing license/creator metadata in light manifests by querying GBIF."
    )
    parser.add_argument(
        "--edible-manifest",
        default="gbif_wf_data/assets/plant_images/light_build/edible/images_manifest.json",
        help="Path to edible light manifest JSON.",
    )
    parser.add_argument(
        "--poisonous-manifest",
        default="gbif_wf_data/assets/plant_images/light_build/poisonous/images_manifest.json",
        help="Path to poisonous light manifest JSON.",
    )
    parser.add_argument(
        "--out-edible",
        default="",
        help="Optional output path for edible manifest (default: in-place).",
    )
    parser.add_argument(
        "--out-poisonous",
        default="",
        help="Optional output path for poisonous manifest (default: in-place).",
    )
    parser.add_argument(
        "--include-missing-creator",
        action="store_true",
        help="Also backfill entries missing creator (even if license is present).",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between GBIF requests.")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds.")
    parser.add_argument("--max-requests", type=int, default=300, help="Cap on GBIF requests.")
    parser.add_argument("--log-every", type=int, default=25, help="Log progress every N requests.")
    parser.add_argument(
        "--update-sibling-csv",
        action="store_true",
        help="Also rewrite sibling images_manifest.csv files next to the JSON manifests.",
    )
    parser.add_argument(
        "--update-sibling-zips",
        action="store_true",
        help="Also refresh sibling gbif_light_*.zip archives with the updated manifest JSON/CSV.",
    )
    args = parser.parse_args()

    edible_path = Path(args.edible_manifest)
    poisonous_path = Path(args.poisonous_manifest)
    if not edible_path.is_file():
        raise SystemExit(f"Missing file: {edible_path}")
    if not poisonous_path.is_file():
        raise SystemExit(f"Missing file: {poisonous_path}")

    edible = load_manifest(edible_path)
    poisonous = load_manifest(poisonous_path)

    session = requests.Session()
    print("Backfilling edible manifest...", flush=True)
    edible_stats = backfill_manifest(
        edible,
        session,
        max_requests=args.max_requests,
        sleep_s=args.sleep,
        timeout_s=args.timeout,
        log_every=args.log_every,
        include_missing_creator=args.include_missing_creator,
    )
    print("Backfilling poisonous manifest...", flush=True)
    poisonous_stats = backfill_manifest(
        poisonous,
        session,
        max_requests=args.max_requests,
        sleep_s=args.sleep,
        timeout_s=args.timeout,
        log_every=args.log_every,
        include_missing_creator=args.include_missing_creator,
    )

    out_edible = Path(args.out_edible) if args.out_edible else edible_path
    out_poisonous = Path(args.out_poisonous) if args.out_poisonous else poisonous_path

    write_manifest(out_edible, edible)
    write_manifest(out_poisonous, poisonous)

    edible_csv = out_edible.with_name("images_manifest.csv")
    poisonous_csv = out_poisonous.with_name("images_manifest.csv")
    if args.update_sibling_csv or args.update_sibling_zips:
        write_manifest_csv(edible_csv, edible)
        write_manifest_csv(poisonous_csv, poisonous)

    edible_zip = out_edible.with_name("gbif_light_edible.zip")
    poisonous_zip = out_poisonous.with_name("gbif_light_poisonous.zip")
    if args.update_sibling_zips:
        refresh_zip(edible_zip, out_edible, edible_csv)
        refresh_zip(poisonous_zip, out_poisonous, poisonous_csv)

    summary = {
        "edible": edible_stats,
        "poisonous": poisonous_stats,
        "out_edible": str(out_edible),
        "out_poisonous": str(out_poisonous),
        "edible_csv": str(edible_csv) if (args.update_sibling_csv or args.update_sibling_zips) else "",
        "poisonous_csv": str(poisonous_csv) if (args.update_sibling_csv or args.update_sibling_zips) else "",
        "edible_zip": str(edible_zip) if args.update_sibling_zips else "",
        "poisonous_zip": str(poisonous_zip) if args.update_sibling_zips else "",
    }
    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
